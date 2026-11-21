"""Offline replay of SNG1193 with scripted model replies, scored by the normal `lfe evaluate`.

The script stands in for Gemini. It proposes what a model plausibly proposes when evidence appears, including the
mistakes seen in the pilot: ASR wording in quotes ("center", "exclusive"), the reference day "thursday" and the
misheard surname. It measures the local checks and rules around the model, not the model's accuracy.

    uv run --locked --no-sync lfe run scripts.scripted_sng1193:Scripted --dataset spokenwoz --ids SNG1193 --speed 10
"""

from __future__ import annotations

import json
from typing import Any

from my_extractor import MyExtractor
from strategies.openrouter import Completion

# Each proposal fires once, as soon as its quote is in the evidence a request carries. Order matters within a reply.
SCRIPT: list[tuple[str, Any, str]] = [
    ("restaurant_food", "indian", "eat indian food"),
    ("services", ["restaurant"], "we can eat indian food"),
    ("restaurant_price_range", "expensive", "exclusive price range"),     # ASR word: rejected
    ("restaurant_area", "centre", "center of the city"),                  # US spelling: now accepted
    ("restaurant_price_range", "expensive", "expensive price range"),
    ("restaurant_name", "golden curry", "golden curry"),
    ("restaurant_people", 5, "myself and five other people"),              # companions miscounted: rejected
    ("restaurant_people", 6, "myself and five other people"),
    ("restaurant_time", "11:15", "quarter past 11"),
    ("restaurant_day", "thursday", "today is a thursday"),                # reference day: rejected
    ("restaurant_day", "saturday", "today is a thursday or two days from now"),
    ("profile_name", "rebecca louie", "my last name is louie"),
    ("profile_name", "rebecca louie", "he like echo"),                    # stale reply after spelling: locked
]


def _plain(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in text.casefold()).split())


class ScriptedBackend:
    def __init__(self) -> None:
        self.fired: set[int] = set()

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> Completion:
        evidence = json.loads(messages[-1]["content"])
        text = _plain(" ".join(str(evidence.get(key, "")) for key in (
            "recently_reviewed_turns", "new_finished_turns", "current_unfinished_turn")))
        operations = []
        for index, (field, value, quote) in enumerate(SCRIPT):
            if index not in self.fired and _plain(quote) in text:
                self.fired.add(index)
                operations.append({"field": field, "value": value, "quote": quote})
        reply = json.dumps({"set": operations, "facts": []})
        return Completion(reply, "scripted/offline", {"input_tokens": 1500, "output_tokens": 40},
                          {"input_tokens": 0.0, "output_tokens": 0.0}, 0.0)


class Scripted(MyExtractor):
    def __init__(self) -> None:
        super().__init__(backend=ScriptedBackend())


class EmptyBackend:
    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> Completion:
        return Completion('{"set":[],"facts":[]}', "scripted/offline", {"input_tokens": 1500, "output_tokens": 10},
                          {"input_tokens": 0.0, "output_tokens": 0.0}, 0.0)


class LocalOnly(MyExtractor):
    """Only the local rules publish: a check for false fills from the rules on any conversation."""

    def __init__(self) -> None:
        super().__init__(backend=EmptyBackend())
