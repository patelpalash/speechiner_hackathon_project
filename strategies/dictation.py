"""Collect the digits a caller dictates for an ID or phone number, with no model call.

Callers read long numbers in groups ("0, 2, 3, 8." ... "9 2 2 1.") while the agent backchannels or echoes a
group. The model often splits such a dictation between the ID and the phone field, or rewrites a group. Here the
field is chosen from what the agent asked ("may I have your id number") or what the caller announces ("my
telephone number is"), every new caller group extends the number at once, and the dictation ends when someone
says something that is not a digit, a backchannel or an echo.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z]+|\d+")
_FIELDS = (
    ("profile_id_number", re.compile(r"\b(?:id|identification|identity) (?:number|card)\b")),
    ("profile_phone", re.compile(r"\b(?:phone|telephone|cellphone|cell phone|mobile|contact number)\b")),
)
_ASK_CUE = re.compile(r"\b(?:your|may i|can i|could i|need|provide|give me|what is|what s)\b")
_CALLER_STATES = re.compile(r"\bmy (?:\w+ ){0,2}(?:number|phone|telephone|id)\b")
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
}
_REPEAT = {"double": 2, "triple": 3}
_FILLER = frozenset({
    "uh", "um", "mm", "mhm", "hmm", "ah", "er", "and", "then", "it", "its", "s", "is", "my", "sure", "yes", "yeah",
    "yep", "okay", "ok", "alright", "right", "so", "sorry", "that", "the", "number", "id", "phone", "telephone",
    "cellphone", "mobile", "identification", "a", "b", "got", "correct", "go", "on", "please", "thank", "thanks",
    "you", "just", "to", "confirm", "said", "they", "she", "he", "your", "of",
    "i", "noted", "continue", "next", "great", "good", "fine", "perfect", "with", "me",
})


def _digits(text: str) -> tuple[str, bool]:
    """The digits in a piece of speech and whether it holds anything other than digits and fillers."""
    tokens = _WORD.findall(text.casefold().replace("'", ""))
    digits: list[str] = []
    other = False
    repeat = 1
    for token in tokens:
        if token.isdigit():
            digits.append(token[0] * repeat + token[1:] if repeat > 1 else token)
            repeat = 1
        elif token in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[token] * repeat)
            repeat = 1
        elif token in _REPEAT:
            repeat = _REPEAT[token]
        elif token not in _FILLER:
            other = True
    return "".join(digits), other


def _asked_field(text: str, fields: set[str]) -> str | None:
    words = " ".join(_WORD.findall(text.casefold()))
    for field_id, pattern in _FIELDS:
        if field_id in fields and pattern.search(words):
            return field_id
    return None


class DigitDictation:
    """Follow one conversation and return (field, digits) whenever the dictated number grows."""

    def __init__(self, fields: set[str], caller: str = "A") -> None:
        self.fields = fields
        self.caller = caller
        self.target: str | None = None
        self.digits = ""
        self.done: dict[str, str] = {}  # field -> the number dictated for it

    @property
    def active(self) -> bool:
        return self.target is not None

    def observe(self, speaker: str | None, text: str, *, partial: bool = False) -> tuple[str, str] | None:
        digits, other = _digits(text)
        if speaker == self.caller:
            if self.target is None:
                field = _asked_field(text, self.fields) if _CALLER_STATES.search(text.casefold()) else None
                if field is None or not digits:
                    return None
                self.target, self.digits = field, ""
            if partial:
                grown = self.digits + digits
                return (self.target, grown) if digits and len(grown) >= 2 else None
            if other and not digits:
                return self._stop()
            self.digits += digits
            return (self.target, self.digits) if digits and len(self.digits) >= 2 else None
        if partial:
            return None
        # The agent: an ask with no digits starts a dictation; an echo or a backchannel continues one.
        if (field := self._ask(text)) is not None:
            self._stop()
            self.target, self.digits = field, ""
            return None
        echo = len(digits) >= 3 and digits in self.digits
        if self.target is not None and other and not echo:
            return self._stop()
        return None

    def _ask(self, text: str) -> str | None:
        """The field an agent sentence asks for ("could i get your id number"), judged sentence by sentence."""
        for sentence in re.split(r"[.?!]+", text.casefold().replace("'", " ")):
            words = " ".join(_WORD.findall(sentence))
            if _ASK_CUE.search(words) and not _digits(sentence)[0]:
                if (field := _asked_field(words, self.fields)) is not None:
                    return field
        return None

    def _stop(self) -> None:
        if self.target is not None and len(self.digits) >= 2:
            self.done[self.target] = self.digits
        self.target, self.digits = None, ""
        return None

    def claims(self, field_id: str, value: object) -> bool:
        """Whether a model value for a digits field should give way to the dictation."""
        if self.target is not None and field_id in self.fields:
            return True  # During a dictation the local digits stand for both number fields.
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return any(field_id != other and digits and digits in number for other, number in self.done.items())
