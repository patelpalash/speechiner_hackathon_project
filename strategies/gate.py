"""Free, deterministic scheduling for typed forms and meeting notes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from live_form_eval import FieldType, Form
from strategies.delta import _SPELLING

_WORD = re.compile(r"[\w@]+", re.UNICODE)
_EVIDENCE = re.compile(
    r"\d|@|\b(?:actually|instead|change|correction|correct|confirm|"
    r"tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"one|two|three|four|five|six|seven|eight|nine|zero|half|noon|midnight)\b",
    re.IGNORECASE,
)
_MEETING = re.compile(
    r"\b(?:agree|decid\w*|propos\w*|suggest\w*|assign\w*|action|deadline|"
    r"should|must|need to|will|instead|prefer\w*|reject\w*)\b",
    re.IGNORECASE,
)
_CONFIRM = re.compile(r"^(?:yes|yeah|yep|right|correct|sure|okay|ok|that works|sounds good)(?:\b|$)")
_SMALL_TALK = frozenset({
    "ah", "bye", "great", "hello", "hey", "hi", "hmm", "mm", "okay", "ok", "oh", "please", "thank",
    "thanks", "uh", "um", "welcome", "well", "yeah", "yep", "yes", "you",
})


# Minimum seconds between booking-form requests. Values count as fast only within 8 s of being said.
_SLOT_SPACING_S = 1.0
# Meeting notes: answers count as fast within 80 s, so batch, but not so long that facts arrive late.
_MEETING_SPACING_S = 14.0


@dataclass(frozen=True)
class Schedule:
    reason: str
    due: float


def has_content(text: str) -> bool:
    return any(token not in _SMALL_TALK for token in _WORD.findall(text.casefold()))


class Gate:
    """Coalesce unreviewed evidence and prioritize likely value changes."""

    def __init__(self, form: Form) -> None:
        self.meeting = any(field.type == FieldType.textarea for field in form.fields)
        options = {
            option.casefold()
            for field in form.fields
            for option in field.options
            if option.casefold() not in _SMALL_TALK and len(option) >= 3
        }
        self.option_words = frozenset(option for option in options if " " not in option)
        self.option_phrases = tuple(sorted(option for option in options if " " in option))

    def _has_option(self, text: str, tokens: list[str]) -> bool:
        return bool(self.option_words.intersection(tokens)) or any(phrase in text for phrase in self.option_phrases)

    def schedule(
        self, evidence: str, *, recent: str, first_at: float, last_call: float, finished: bool, final: bool
    ) -> Schedule | None:
        lowered = evidence.casefold()
        tokens = [_SPELLING.get(token, token) for token in _WORD.findall(lowered)]
        if not tokens:
            return None
        confirmation = False
        if not self.meeting and any(_CONFIRM.match(line.strip()) for line in lowered.splitlines()):
            preceding = recent.casefold()
            confirmation = bool("?" in preceding or _EVIDENCE.search(preceding)
                                or self._has_option(preceding, _WORD.findall(preceding)))
        informative = sum(token not in _SMALL_TALK for token in tokens)
        if not informative and not confirmation:
            return None
        if final:
            return Schedule("final", first_at)
        if self.meeting:
            due = last_call + _MEETING_SPACING_S
            if informative >= 12 and _MEETING.search(evidence):
                return Schedule("meeting_signal", due)
            if informative >= 26:
                return Schedule("meeting_batch", due)
            return Schedule("meeting_deadline", max(due, first_at + 22))

        due = last_call + _SLOT_SPACING_S
        has_option = self._has_option(lowered, tokens)
        if confirmation or ((finished or informative >= 3) and (_EVIDENCE.search(evidence) or has_option)):
            return Schedule("slot_signal", due)
        if finished and informative >= 6:
            return Schedule("turn_batch", max(due, first_at + 1.5))
        return Schedule("slot_deadline", max(due, first_at + 6))
