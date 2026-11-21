"""Budgeted live form extraction using local gating and a remote state delta."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Protocol

from live_form_eval import Extractor, Partial, Session
from strategies.delta import DeltaError, make_messages, parse_delta
from strategies.gate import Gate, has_content
from strategies.openrouter import Completion, OpenRouterClient
from strategies.services import requested_services
from strategies.dictation import DigitDictation
from strategies.spelling import SpelledName

_TRANSCRIPT_CHARS = 2600
_RECENT_CHARS = 650
_CALLER_ONLY = os.environ.get("GATED_CALLER_ONLY", "0") == "1"


class Backend(Protocol):
    async def complete(
        self, messages: list[dict[str, str]], *, max_tokens: int
    ) -> Completion | None: ...


@dataclass(frozen=True)
class Batch:
    """A fixed snapshot, so a reply can acknowledge only the evidence it saw."""

    end: int
    transcript: str
    recent: str
    partial: Partial | None
    evidence: str
    first_at: float
    finished: bool
    more: bool
    caller_spoke: bool = True  # the new evidence holds caller speech (or a speaker not yet known)


def _unseen(message: Partial, reviewed: Partial | None) -> str:
    """Find new words in a growing partial, retaining any rewritten utterance."""
    if reviewed and message.turn_id == reviewed.turn_id:
        if message.labelled_text() == reviewed.labelled_text():
            return ""
        if message.text.startswith(reviewed.text + " ") and message.speaker == reviewed.speaker:
            return message.text[len(reviewed.text):].strip()
    return message.text


def _recent(turns: list[Partial], end: int) -> str:
    lines = []
    size = 0
    for index in range(end - 1, max(-1, end - 3), -1):
        line = turns[index].labelled_text()
        lines.append(line)
        size += len(line) + 1
        if size >= _RECENT_CHARS:
            break
    return "\n".join(reversed(lines))[-_RECENT_CHARS:]


def _batch(
    turns: list[Partial], cursor: int, partial: Partial | None, reviewed: Partial | None,
    partial_since: float | None, reconsider_finished: bool,
) -> Batch:
    lines: list[str] = []
    evidence: list[tuple[float, str]] = []
    end, size = cursor, 0
    caller_spoke = False
    while end < len(turns):
        turn = turns[end]
        line = turn.labelled_text()
        new_text = _unseen(turn, reviewed)
        if reconsider_finished and reviewed and turn.turn_id == reviewed.turn_id and not has_content(new_text):
            new_text = turn.text
        if new_text and lines and size + len(line) + 1 > _TRANSCRIPT_CHARS:
            break
        if new_text:
            lines.append(line)
            evidence.append((turn.t, new_text))
            size += len(line) + 1
            caller_spoke = caller_spoke or any(speaker != "B" and has_content(text) for speaker, text in turn.pieces())
        end += 1
    has_finished_evidence = any(has_content(text) for _, text in evidence)
    active = None
    more = end < len(turns)
    if end == len(turns) and partial is not None and (new_text := _unseen(partial, reviewed)):
        if lines and size + len(partial.labelled_text()) > _TRANSCRIPT_CHARS:
            more = True
        else:
            active = partial
            first_partial = partial.t if partial_since is None else partial_since
            if reviewed and partial.turn_id == reviewed.turn_id:
                first_partial = max(first_partial, reviewed.t)
            evidence.append((first_partial, new_text))
            caller_spoke = caller_spoke or (partial.speaker != "B" and has_content(new_text))
    first_at = next((t for t, text in evidence if has_content(text)), evidence[0][0] if evidence else float("inf"))
    return Batch(end, "\n".join(lines), _recent(turns, cursor), active, "\n".join(text for _, text in evidence), first_at,
                 has_finished_evidence, more, caller_spoke)


async def _wait_for_evidence(changed: asyncio.Event, session: Session, deadline: float | None) -> None:
    if deadline is None:
        await changed.wait()
        return
    tasks = [asyncio.create_task(changed.wait()), asyncio.create_task(session.sleep_until(deadline))]
    try:
        finished, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class MyExtractor(Extractor):
    """One backend budget per run and one independent form state per conversation."""

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = OpenRouterClient.from_env() if backend is None else backend

    async def run(self, session: Session) -> None:
        gate = Gate(session.form)
        changed = asyncio.Event()
        turns: list[Partial] = []
        active: Partial | None = None
        partial_since: float | None = None
        ended = False
        current: dict[str, Any] = {}
        services_seen: dict[str, set[str]] = {}
        # Fields set from a spelled name or dictated digits. A model reply cannot overwrite them.
        locked: set[str] = set()
        spelling = (SpelledName(caller="A") if any(spec.field_id == "profile_name" for spec in session.form.fields)
                    else None)

        number_fields = {spec.field_id for spec in session.form.fields
                         if spec.field_id in ("profile_phone", "profile_id_number")}
        dictation = DigitDictation(number_fields, caller="A") if number_fields else None

        def publish_digits(found: tuple[str, str] | None) -> None:
            if found is not None and found[1] != current.get(found[0]):
                session.update(found[0], found[1])
                current[found[0]] = found[1]
                locked.add(found[0])

        def publish_spelling(name: str | None) -> None:
            if name and name != current.get("profile_name"):
                session.update("profile_name", name)
                current["profile_name"] = name
                locked.add("profile_name")

        async def listen() -> None:
            nonlocal active, partial_since, ended
            async for message in session.partials():
                if message.kind == "turn":
                    turns.append(message)
                    active = None
                    partial_since = None
                else:
                    if active is None or active.turn_id != message.turn_id:
                        partial_since = None
                    if partial_since is None and has_content(message.text):
                        partial_since = message.t
                    active = message
                # The booking form identifies the first speaker (A) as the caller.
                if explicitly_requested := requested_services(session.form, message, caller="A"):
                    seen = services_seen.setdefault(message.turn_id, set())
                    additions = explicitly_requested - seen
                    seen.update(explicitly_requested)
                    if additions - set(current.get("services", ())):
                        selected = additions | set(current.get("services", ()))
                        value = [option for option in session.form.field("services").options if option in selected]
                        session.update("services", value)
                        current["services"] = value
                if dictation is not None:
                    if message.kind == "turn":
                        for speaker, text in message.pieces():
                            publish_digits(dictation.observe(speaker, text))
                    else:
                        publish_digits(dictation.observe(message.speaker, message.text, partial=True))
                if spelling is not None and message.kind == "turn":
                    for speaker, text in message.pieces():
                        publish_spelling(spelling.observe(speaker, text, current.get("profile_name")))
                    # "my name is mary edwards" fills an empty name at once; the model may still correct it.
                    if (stated := spelling.take_stated()) and "profile_name" not in current:
                        session.update("profile_name", stated)
                        current["profile_name"] = stated
                changed.set()
            if spelling is not None:
                publish_spelling(spelling.finish(current.get("profile_name")))
            ended = True
            changed.set()

        async def extract() -> None:
            cursor = 0
            reviewed: Partial | None = None
            reconsider_finished = False
            last_call = float("-inf")
            failed: Batch | None = None
            while True:
                changed.clear()
                batch = _batch(turns, cursor, active, reviewed, partial_since, reconsider_finished)
                schedule = gate.schedule(
                    batch.evidence, recent=batch.recent, first_at=batch.first_at,
                    last_call=last_call, finished=batch.finished, final=ended,
                )
                if schedule is None and batch.more:
                    cursor = batch.end
                    continue
                if schedule is None or batch == failed:
                    if ended:
                        return
                    await _wait_for_evidence(changed, session, None)
                    continue
                if _CALLER_ONLY and not batch.caller_spoke and not ended and not gate.meeting:
                    # Agent speech alone waits for the caller's next words, which carry it as context.
                    await _wait_for_evidence(changed, session, None)
                    continue
                if schedule.due > session.now():
                    await _wait_for_evidence(changed, session, schedule.due)
                    continue

                last_call = session.now()
                final = ended and not batch.more
                allow_clear = batch.finished or final
                partial = batch.partial.labelled_text() if batch.partial else ""
                state_at_request = current.copy()
                messages = make_messages(
                    session.form, state_at_request, batch.transcript,
                    recent=batch.recent, partial=partial,
                    final=final, allow_clear=allow_clear,
                )
                result = await self.backend.complete(messages, max_tokens=920 if gate.meeting else 420)
                if result is None:
                    raise RuntimeError("OpenRouter stopped extraction at the run budget or an unresolved charge.")

                error = result.error
                delta: dict[str, Any] = {}
                diagnostics: dict[str, Any] = {}
                if error is None:
                    try:
                        delta = parse_delta(
                            session.form, state_at_request, result.text, allow_clear=allow_clear,
                            evidence_sources=(batch.transcript, partial, batch.recent),
                            diagnostics=diagnostics,
                        )
                        # Local service detection can publish a newer value while the model is answering.
                        delta = {field: value for field, value in delta.items()
                                 if current.get(field) == state_at_request.get(field) and field not in locked
                                 and not (dictation is not None and dictation.claims(field, value))}
                        diagnostics["accepted"] = len(delta)
                    except DeltaError as exc:
                        error = exc.code
                session.report_usage(
                    model=result.model, extraction=True, usage=result.usage, prices=result.prices,
                    latency_s=result.latency_s, error=error,
                    trigger="drain" if schedule.reason == "final" and not final else schedule.reason,
                    cost_usd=result.cost_usd, cost_source=result.cost_source,
                    delta_diagnostics=diagnostics,
                )
                if error is not None:
                    failed = batch
                    continue
                if delta:
                    session.update_snapshot(delta)
                    current.update(delta)
                    for field_id, value in delta.items():
                        if value is None:
                            del current[field_id]
                cursor = batch.end
                if batch.partial is not None:
                    reviewed = batch.partial
                    # A no-op may mean the model was waiting for this utterance to finish.
                    reconsider_finished = not delta
                failed = None
                if final:
                    return

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(listen())
            tasks.create_task(extract())
