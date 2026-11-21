"""Decode a name the caller spells after the agent asks, with no model call.

Callers spell with code words ("l like lima", "h as in hotel"). Speech recognition often garbles the letter
("age like hotel", "are like rojak") but keeps the code word, so each code word's first letter is the letter.
The spelling is published only when the agent moves on, so a half-spelled name is never shown.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z]+")
_ASK = re.compile(r"\bspell\b")
_LAST = re.compile(r"\b(?:last name|surname|family name|second name)\b")
_FIRST = re.compile(r"\bfirst name\b")
_FULL_NAME = re.compile(r"\b(?:my (?:full )?name is|full name is|name is) ([a-z]{2,}) ([a-z]{2,})\b")
_LAST_NAME = re.compile(r"\b(?:last name is|surname is|family name is) ([a-z]{2,})\b")
_FIRST_NAME = re.compile(r"\b(?:first name is|my name is|this is|you are speaking to|you re speaking to) ([a-z]{2,})\b")
_CUES = frozenset({"like", "likes", "for"})
# The agent asking for the surname alone ("thank you, your surname"), and the caller's one-word answer.
_ASK_LAST = re.compile(r"\b(?:your|the) (?:last name|surname|family name|second name)\b")
_FILLER = re.compile(r"^(?:(?:okay|ok|yes|yeah|sure|uh|um|oh|well|right)\s+)*")
_LAST_ANSWER = re.compile(
    r"^(?:(?:it s|its|it is|that s|my (?:last |sur|family )?name is|(?:last name|surname|family name) is)\s+)?"
    r"([a-z]{2,})(?:$|\s+(?:spelled|spells|spelt|spell|like|as)\b)"
)
# Letter names that recognition writes as words, keyed to the letter they stand for.
_HOMOPHONES = {
    "or": "o", "oh": "o", "are": "r", "age": "h", "you": "u", "why": "y", "see": "c", "sea": "c", "be": "b",
    "bee": "b", "tea": "t", "tee": "t", "pea": "p", "pee": "p", "queue": "q", "jay": "j", "kay": "k", "el": "l",
    "em": "m", "en": "n", "es": "s", "ex": "x", "eye": "i", "dee": "d", "gee": "g", "vee": "v", "ef": "f",
}
_NOT_CODE = frozenset({"a", "an", "the", "my", "your", "it", "that", "this", "me", "you", "is", "and", "so"})
_BACKCHANNEL = frozenset({
    "mhm", "mm", "hmm", "uh", "huh", "um", "yes", "yeah", "yep", "okay", "ok", "right", "alright", "all",
    "what", "sorry", "pardon", "go", "on", "got", "it", "and", "next", "again", "please", "sure", "correct",
})
_STOP = frozenset({"a", "i"})
_NOT_NAME = frozenset({"called", "spell", "spells", "spelled", "spelt", "spelling", "like", "okay", "ok", "yes", "and", "please", "thank", "thanks", "here", "so", "all", "fine", "good", "great", "my", "the", "a", "an", "is", "calling", "speaking", "from", "about", "it", "not", "just"})


def _plain(text: str) -> str:
    return text.casefold().replace("'", "").replace("\u2019", "")


def _letters(text: str) -> list[str]:
    tokens = _WORD.findall(_plain(text))
    letters: list[str] = []
    for index, word in enumerate(tokens):
        before = tokens[index - 1] if index else ""
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        if len(word) >= 2 and word not in _NOT_CODE and (
            before in _CUES or (index >= 2 and tokens[index - 2: index] == ["as", "in"])
            or _HOMOPHONES.get(before) == word[0]
        ):
            letters.append(word[0])
        elif len(word) == 1 and word not in _STOP and after not in _CUES | {"as"} and not after.startswith(word):
            letters.append(word)
    return letters


def _subsequence(letters: str, text: str) -> bool:
    remaining = iter(text)
    return all(letter in remaining for letter in letters)


class SpelledName:
    """Follow one conversation's turns and return a corrected name when a spelling finishes."""

    def __init__(self, caller: str = "A") -> None:
        self.caller = caller
        self.first_name: str | None = None
        self.last_name: str | None = None
        self.stated: str | None = None  # "my name is mary edwards", until taken with take_stated()
        self.active = False
        self.asked_last = False
        self.target = "name"
        self.letters: list[str] = []

    def observe(self, speaker: str | None, text: str, current: str | None) -> str | None:
        words = " ".join(_WORD.findall(_plain(text)))
        if speaker == self.caller:
            if self.active:
                self.letters.extend(_letters(text))
                return None
            if self.asked_last:
                self.asked_last = False
                if joined := self._join_last(words, current):
                    return joined
            if (match := _FIRST_NAME.search(words)) and match[1] not in _NOT_NAME:
                self.first_name = match[1]
            # Per sentence: "my name is gracious. grace spells like" states no full name.
            sentences = (" ".join(_WORD.findall(part)) for part in re.split(r"[.?!]", _plain(text)))
            for match in filter(None, map(_FULL_NAME.search, sentences)):
                if not {match[1], match[2]} & _NOT_NAME:
                    self.first_name, self.last_name = match[1], match[2]
                    self.stated = f"{match[1]} {match[2]}"
            if (match := _LAST_NAME.search(words)) and match[1] not in _NOT_NAME:
                self.last_name = match[1]
            return None
        self.asked_last = bool(_ASK_LAST.search(words)) and not _ASK.search(words)
        if _ASK.search(words):
            self.active, self.letters = True, []
            self.target = "last" if _LAST.search(words) else "first" if _FIRST.search(words) else "name"
            return None
        if not self.active or not words or set(words.split()) <= _BACKCHANNEL:
            return None
        return self.finish(current)

    def _join_last(self, words: str, current: str | None) -> str | None:
        """ "thank you, your surname." "my name is gibson." joins gibson to a lone first name."""
        match = _LAST_ANSWER.match(_FILLER.sub("", words))
        if match is None or match[1] in _NOT_NAME | _BACKCHANNEL:
            return None
        tokens = (current or "").split() or ([self.first_name] if self.first_name else [])
        if len(tokens) != 1 or tokens[0] == match[1]:
            return None
        self.last_name = match[1]
        return f"{tokens[0]} {match[1]}"

    def take_stated(self) -> str | None:
        """A full name the caller just stated in so many words; returned once."""
        stated, self.stated = self.stated, None
        return stated

    def finish(self, current: str | None) -> str | None:
        if not self.active:
            return None
        self.active = False
        if len(self.letters) < 3:
            return None
        spelled = "".join(self.letters)
        # Compare with the name on the form, else the name the caller said before spelling it.
        tokens = (current or "").split() or [name for name in (self.first_name, self.last_name) if name]
        if _subsequence(spelled, "".join(tokens)):
            return None  # Letters consistent with the name already heard confirm it, even if some were missed.
        if self.target == "first":
            return " ".join([spelled, *tokens[1:]]) if tokens else None
        if len(tokens) < 2:
            # Only an explicit last-name spelling is joined to a lone first name; otherwise it is ambiguous.
            return f"{tokens[0]} {spelled}" if tokens and self.target == "last" else None
        # Replace the part that starts with the same letter, else the last name.
        index = next((i for i in range(len(tokens) - 1, -1, -1) if tokens[i][0] == spelled[0]), len(tokens) - 1)
        if self.target == "last":
            index = max(index, 1)
        return " ".join([*tokens[:index], spelled, *tokens[index + 1:]])
