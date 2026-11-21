"""Build more SpokenWOZ test calls in the bench format, for testing beyond the 6 shipped calls.

The bench ships 6 of SpokenWOZ's 1,000 test calls, and the code that builds its gold is not in this repository.
This script rebuilds other test calls from the public SpokenWOZ test text (Hugging Face
``ssz1111/SpokenWOZ-Test-Text-Fixed``, ``data.json``) with a simpler gold:

- words, speakers and times come from the dataset's own word timings;
- a value is said at the end of its span's last word when the dataset has a span, else at the end of the caller's
  turn (``source: turn``), and it is never cut into pieces (a dictated number counts once it is complete);
- a service is said at the word that names it, else at the end of the turn whose dialogue acts name it.

Scores on these calls are therefore close to, but not the same as, what the official gold would give. They are
for comparing versions of our own extractor; the saved Speechineer baseline has no runs on them.

    python -B -m scripts.extra_spokenwoz data.json --n 24 --seed 1 --out extra

writes ``extra/processed/spokenwoz/test/`` and ``extra/config.yaml`` (the repository's config with only the
data, runs and results paths moved). Run and score with ``LFE_CONFIG=extra/config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "data" / "processed" / "spokenwoz" / "test"

SLOTS = {
    ("hotel", "area"): "hotel_area", ("hotel", "pricerange"): "hotel_price_range", ("hotel", "type"): "hotel_type",
    ("hotel", "stars"): "hotel_stars", ("hotel", "parking"): "hotel_parking", ("hotel", "internet"): "hotel_internet",
    ("hotel", "name"): "hotel_name", ("hotel", "day"): "hotel_day", ("hotel", "people"): "hotel_people",
    ("hotel", "stay"): "hotel_nights",
    ("restaurant", "area"): "restaurant_area", ("restaurant", "pricerange"): "restaurant_price_range",
    ("restaurant", "food"): "restaurant_food", ("restaurant", "name"): "restaurant_name",
    ("restaurant", "day"): "restaurant_day", ("restaurant", "people"): "restaurant_people",
    ("restaurant", "time"): "restaurant_time",
    ("train", "departure"): "train_departure", ("train", "destination"): "train_destination",
    ("train", "day"): "train_day", ("train", "leaveat"): "train_leave_at", ("train", "arriveby"): "train_arrive_by",
    ("train", "people"): "train_people",
    ("taxi", "departure"): "taxi_departure", ("taxi", "destination"): "taxi_destination",
    ("taxi", "leaveat"): "taxi_leave_at", ("taxi", "arriveby"): "taxi_arrive_by",
    ("attraction", "area"): "attraction_area", ("attraction", "type"): "attraction_type",
    ("attraction", "name"): "attraction_name", ("hospital", "department"): "hospital_department",
    ("profile", "name"): "profile_name", ("profile", "phonenumber"): "profile_phone",
    ("profile", "email"): "profile_email", ("profile", "idnumber"): "profile_id_number",
    ("profile", "platenumber"): "profile_plate_number",
}
# Span slot names in span_info differ a little from the state's ("bookday", "leaveAt").
SPAN_SLOT = {"bookday": "day", "bookpeople": "people", "bookstay": "stay", "booktime": "time", "leaveat": "leaveat",
             "arriveby": "arriveby", "price": "pricerange", "phone": "phonenumber", "id": "idnumber"}
EMPTY = {"", "none", "not mentioned", "dontcare", "dont care", "don't care", "do n't care", "?"}
SERVICE_WORDS = {
    "hotel": r"hotels?|guest ?houses?", "restaurant": r"restaurants?", "train": r"trains?", "taxi": r"taxis?|cabs?",
    "attraction": r"attractions?", "hospital": r"hospitals?", "police": r"police",
}
_PUNCT = re.compile(r"^[.,?!;:]+$")


def _words(turn: dict[str, Any], speaker: str) -> list[dict[str, Any]]:
    """The turn's timed words, with the transcript's punctuation attached as a speech-to-text vendor writes it."""
    timed = [w for w in turn.get("words", []) if str(w.get("Word", "")).strip()]
    tokens = turn.get("text", "").split()
    texts = [str(w["Word"]) for w in timed]
    plain = [t for t in tokens if not _PUNCT.match(t)]
    if len(plain) == len(texts):
        index = -1
        for token in tokens:
            if _PUNCT.match(token):
                if index >= 0:
                    texts[index] += token
            else:
                index += 1
                texts[index] = token
    elif texts and not texts[-1].endswith((".", "?", "!")):
        texts[-1] += "."
    return [{"text": text, "start": w["BeginTime"] / 1000, "end": w["EndTime"] / 1000, "speaker": speaker}
            for text, w in zip(texts, timed)]


def _state(metadata: dict[str, Any]) -> dict[tuple[str, str], str]:
    state: dict[tuple[str, str], str] = {}
    for domain, parts in metadata.items():
        for part in ("semi", "book"):
            for slot, value in (parts.get(part) or {}).items():
                if slot == "booked" or not isinstance(value, str):
                    continue
                state[(domain, slot.casefold())] = value.strip().casefold()
    return state


def _value(spec: dict[str, Any], raw: str) -> Any:
    """The raw value in the form's type, or None when the field cannot hold it."""
    if raw in EMPTY:
        return None
    kind = spec["type"]
    if kind == "select":
        return raw if raw in spec["options"] else None
    if kind == "checkbox":
        return {"yes": True, "free": True, "no": False}.get(raw)
    if kind == "integer":
        return int(raw) if raw.isdigit() else None
    if kind == "slider":
        return float(raw) if raw.isdigit() and 0 <= int(raw) <= 5 else None
    if kind == "time":
        match = re.fullmatch(r"(\d{1,2})[:.](\d{2})", raw)
        return f"{int(match[1]):02d}:{match[2]}" if match and int(match[1]) < 24 else None
    if kind == "phone":
        digits = "".join(ch for ch in raw if ch.isdigit())
        return digits or None
    return raw


def build(dialogue_id: str, dialogue: dict[str, Any], fields: dict[str, dict[str, Any]]):
    words: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    services: set[str] = set()
    last_user: dict[str, Any] | None = None
    last_user_words: list[dict[str, Any]] = []
    for index, turn in enumerate(dialogue["log"]):
        speaker = "caller" if turn["tag"] == "user" else "agent"
        turn_words = _words(turn, speaker)
        words += turn_words
        if speaker == "caller":
            last_user, last_user_words = turn, turn_words
            if not turn_words:
                continue
            for act in turn.get("dialog_act") or {}:
                domain = act.split("-")[0].casefold()
                if domain in SERVICE_WORDS and domain not in services:
                    services.add(domain)
                    named = next((w for w in turn_words
                                  if re.fullmatch(rf"(?:{SERVICE_WORDS[domain]})[.,?!]*", w["text"].casefold())), None)
                    items.append({"field_id": "services", "text": domain,
                                  "said": (named or turn_words[-1])["end"], "source": "rule" if named else "turn",
                                  "detail": {"turn": index - 0}})
            continue
        if not turn.get("metadata") or last_user is None or not last_user_words:
            continue
        current: dict[str, Any] = {}
        for key, raw in _state(turn["metadata"]).items():
            field = SLOTS.get(key)
            if field and field in fields and (value := _value(fields[field], raw)) is not None:
                current[field] = value
        turn_end = last_user_words[-1]["end"]
        spans = {}
        for act, slot, _value_text, start, end in last_user.get("span_info") or []:
            domain = act.split("-")[0].casefold()
            slot = SPAN_SLOT.get(slot.casefold(), slot.casefold())
            if SLOTS.get((domain, slot)) and 0 <= end < len(last_user_words):
                spans[SLOTS[(domain, slot)]] = last_user_words[end]["end"]
        for field in sorted(set(current) | set(previous)):
            if current.get(field) == previous.get(field):
                continue
            value = current.get(field)
            said, source = (spans[field], "span") if field in spans and value is not None else (turn_end, "turn")
            values.append({"field_id": field, "value": value, "said": said, "source": source,
                           "detail": {"turn": index - 1, "agent_turn": index}})
        snapshots.append({"t": turn_end, "values": dict(current)})
        previous = current
    words.sort(key=lambda w: w["start"])
    conversation = {"id": dialogue_id, "dataset": "spokenwoz", "split": "test", "words": words,
                    "end": max(w["end"] for w in words), "roles": {"caller": "caller", "agent": "agent"},
                    "meta": {"form_id": "spokenwoz", "source": "extra_spokenwoz.py"}}
    gold = {"conversation_id": dialogue_id, "values": values, "items": items, "texts": [], "facts": [],
            "accepted": [], "snapshots": snapshots}
    return conversation, gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="SpokenWOZ test data.json")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=240, help="skip longer calls (cost)")
    parser.add_argument("--ids", nargs="*", help="build exactly these calls (e.g. the 6 shipped ones, to compare)")
    parser.add_argument("--out", type=Path, default=REPO / "extra")
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    forms = json.loads((OFFICIAL / "forms.json").read_text())
    fields = {spec["field_id"]: spec for spec in forms["spokenwoz"]["fields"]}
    shipped = {json.loads(line)["id"] for line in (OFFICIAL / "conversations.jsonl").open()}
    built = []
    candidates = args.ids or sorted(set(data) - shipped)
    if not args.ids:
        random.Random(args.seed).shuffle(candidates)
    for dialogue_id in candidates:
        conversation, gold = build(dialogue_id, data[dialogue_id], fields)
        if not args.ids and (conversation["end"] > args.max_seconds or not gold["values"]):
            continue
        built.append((conversation, gold))
        if not args.ids and len(built) == args.n:
            break
    target = args.out / "processed" / "spokenwoz" / "test"
    target.mkdir(parents=True, exist_ok=True)
    (target / "forms.json").write_text(json.dumps(forms))
    with (target / "conversations.jsonl").open("w") as out:
        out.writelines(json.dumps(conversation) + "\n" for conversation, _ in built)
    with (target / "gold.jsonl").open("w") as out:
        out.writelines(json.dumps(gold) + "\n" for _, gold in built)
    config = (REPO / "config.yaml").read_text()
    for key, value in (("processed", "processed"), ("runs", "runs"), ("results", "results")):
        folder = (args.out / value).resolve()
        shown = folder.relative_to(REPO) if folder.is_relative_to(REPO) else folder
        config = re.sub(rf"(?m)^(  {key}:\s*)\S+", rf"\g<1>{shown}", config)
    (args.out / "config.yaml").write_text(config)
    ids = [conversation["id"] for conversation, _ in built]
    print(f"{len(ids)} calls, {sum(c['end'] for c, _ in built) / 60:.1f} min: {' '.join(ids)}")


if __name__ == "__main__":
    main()
