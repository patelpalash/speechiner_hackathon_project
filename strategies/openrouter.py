"""OpenRouter chat completions and per-run spend accounting.

To opt in to paid extraction, set LFE_OPENROUTER_ENABLE_PAID=1 and provide
OPENROUTER_API_KEY. Set LFE_OPENROUTER_EUR_PER_USD and verify the model's
LFE_OPENROUTER_INPUT_USD_PER_M and LFE_OPENROUTER_OUTPUT_USD_PER_M rates.
LFE_OPENROUTER_CACHE_READ_USD_PER_M and LFE_OPENROUTER_CACHE_WRITE_USD_PER_M
are optional, but required for rate-based accounting if those units are used.

The local run budget defaults to $2 and resets with each extractor process.
Use a dedicated OpenRouter API key with a provider-side limit to enforce a cap
across processes. The local budget uses a conservative request estimate and
stops sending if a response's charge cannot be determined.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from strategies.delta import response_format

DEFAULT_MODEL = "google/gemini-2.5-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_PREFIX = "LFE_OPENROUTER_"
TIMEOUT_S = 7.0
MAX_UNRESOLVED = 3
RESPONSE_LIMIT = 512 * 1024
MODEL_ID = re.compile(r"[a-zA-Z0-9._-]+/[a-zA-Z0-9._:-]+")


@dataclass(frozen=True)
class Completion:
    text: str | None
    model: str
    usage: dict[str, int]
    prices: dict[str, float]
    latency_s: float
    error: str | None = None
    cost_usd: float | None = None
    cost_source: str | None = None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value < math.inf else None


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 10_000_000 else None


def _rate(name: str, default: str | None = None, *, optional: bool = False) -> float | None:
    key = ENV_PREFIX + name
    raw = os.environ.get(key, default)
    if raw is None and optional:
        return None
    try:
        rate = _number(float(raw)) if raw is not None else None
    except (ValueError, OverflowError):
        rate = None
    if rate is None:
        raise RuntimeError(f"Set {key} to a finite nonnegative number.")
    return rate


def _tokens(reported: dict[str, Any]) -> dict[str, int] | None:
    prompt = _count(reported.get("prompt_tokens"))
    output = _count(reported.get("completion_tokens"))
    details = reported.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    read = _count(details.get("cached_tokens") or 0)
    write = _count(details.get("cache_write_tokens") or 0)
    if prompt is None or output is None or read is None or write is None or read + write > prompt:
        return None
    return {unit: count for unit, count in (
        ("input_tokens", prompt - read - write), ("cache_read_tokens", read),
        ("cache_write_tokens", write), ("output_tokens", output),
    ) if count}


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # The Authorization header must stay on the fixed OpenRouter host.


class _RequestFailed(Exception):
    """Safe error text for the run log. Provider response bodies can contain sensitive data."""

    http_status: int | None = None


def _cacheable(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Mark the end of the stable system prompt as a cache breakpoint.

    Gemini's implicit cache covered about 1,630 of the prompt's 2,010 tokens and missed now and then. An explicit
    breakpoint caches the whole system prompt (about 1,970 tokens, read at a tenth of the input price), so only the
    short evidence message is billed at the full rate. The first request of a prompt pays one cache write.
    """
    if os.environ.get(ENV_PREFIX + "EXPLICIT_CACHE", "1") == "0" or not messages:
        return list(messages)
    first, *rest = messages
    if first.get("role") != "system" or not isinstance(first.get("content"), str):
        return list(messages)
    block = {"type": "text", "text": first["content"], "cache_control": {"type": "ephemeral"}}
    return [{"role": "system", "content": [block]}, *rest]


class OpenRouterClient:
    """One instance shares a local budget among all conversations of a run."""

    def __init__(
        self, *, api_key: str, model: str, usd_per_million: dict[str, float],
        eur_per_usd: float, budget_usd: float,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._rates = dict(usd_per_million)
        self._eur_per_usd = eur_per_usd
        self._budget_usd = budget_usd
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._blocked = False
        self._unresolved = 0  # failed requests in a row whose charge is unknown

    @classmethod
    def from_env(cls) -> OpenRouterClient:
        if os.environ.get(ENV_PREFIX + "ENABLE_PAID") != "1":
            raise RuntimeError(f"Paid OpenRouter extraction requires {ENV_PREFIX}ENABLE_PAID=1.")
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key or not key.isascii() or any(char.isspace() for char in key):
            raise RuntimeError("Set OPENROUTER_API_KEY to a valid API key.")
        model = os.environ.get(ENV_PREFIX + "MODEL", DEFAULT_MODEL)
        if not MODEL_ID.fullmatch(model) or model == "openrouter/auto":
            raise RuntimeError(f"Set {ENV_PREFIX}MODEL to one concrete provider/model ID.")
        rates = {
            "input_tokens": _rate("INPUT_USD_PER_M"),
            "output_tokens": _rate("OUTPUT_USD_PER_M"),
            "cache_read_tokens": _rate("CACHE_READ_USD_PER_M", optional=True),
            "cache_write_tokens": _rate("CACHE_WRITE_USD_PER_M", optional=True),
        }
        budget = _rate("RUN_BUDGET_USD", "2")
        exchange = _rate("EUR_PER_USD")
        if not budget or budget > 20:
            raise RuntimeError(f"{ENV_PREFIX}RUN_BUDGET_USD must be greater than 0 and at most 20.")
        if not exchange:
            raise RuntimeError(f"{ENV_PREFIX}EUR_PER_USD must be greater than 0.")
        return cls(api_key=key, model=model, usd_per_million={unit: rate for unit, rate in rates.items()
                                                           if rate is not None},
                   eur_per_usd=exchange, budget_usd=budget)

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 400) -> Completion | None:
        """Send at most one request. None means the local budget or a previous billing error blocked it."""
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be an integer from 1 to 2048.")
        input_rate = self._rates["input_tokens"]
        output_rate = self._rates["output_tokens"]
        # UTF-8 bytes overestimate tokens on these text prompts. Extra room covers chat framing.
        request_context = (messages, response_format(messages))
        input_upper_estimate = len(json.dumps(request_context, ensure_ascii=False).encode("utf-8")) + 1024
        largest_input_rate = max(input_rate, self._rates.get("cache_read_tokens", input_rate),
                                 self._rates.get("cache_write_tokens", input_rate))
        reserve = (input_upper_estimate * largest_input_rate + max_tokens * output_rate) / 1_000_000
        if self._blocked or self._spent_usd + self._reserved_usd + reserve > self._budget_usd:
            return None
        self._reserved_usd += reserve
        started = time.monotonic()
        result = None
        rejected = False
        try:
            async with asyncio.timeout(TIMEOUT_S):
                response = await asyncio.to_thread(self._request, messages, max_tokens)
            result = self._completion(response, time.monotonic() - started)
            return result
        except (TimeoutError, _RequestFailed) as error:
            # An HTTP error status (e.g. 503, no provider available) comes back before generation. A timeout may
            # still bill, but never more than its reserve: max_tokens and max_price bound it. The reserve is counted
            # as spent and later requests go ahead, until MAX_UNRESOLVED failures in a row block the client.
            rejected = getattr(error, "http_status", None) is not None
            reason = "OpenRouter request timed out." if isinstance(error, TimeoutError) else str(error)
            return Completion(None, self.model, {}, {}, time.monotonic() - started, reason)
        finally:
            self._reserved_usd -= reserve
            if result is None or result.cost_usd is None:
                self._spent_usd += reserve
                self._unresolved += not rejected
                self._blocked = self._blocked or self._unresolved >= MAX_UNRESOLVED
            else:
                self._unresolved = 0
                self._spent_usd += result.cost_usd
                if result.cost_usd > reserve:
                    self._blocked = True

    def _request(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": _cacheable(messages),
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format(messages),
            "reasoning": {"enabled": False},
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
                "max_price": {
                    "prompt": str(self._rates["input_tokens"]),
                    "completion": str(self._rates["output_tokens"]),
                    "request": "0",
                },
            },
        }
        request = Request(API_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        try:
            with build_opener(_RejectRedirect()).open(request, timeout=TIMEOUT_S) as response:
                payload = response.read(RESPONSE_LIMIT + 1)
            if len(payload) > RESPONSE_LIMIT:
                raise _RequestFailed("OpenRouter response exceeded the size limit.")
            response = json.loads(payload)
        except HTTPError as error:
            error.close()
            failure = _RequestFailed(f"OpenRouter returned HTTP {error.code}.")
            failure.http_status = error.code
            raise failure from None
        except (URLError, OSError):
            raise _RequestFailed("OpenRouter connection failed.") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _RequestFailed("OpenRouter returned invalid JSON.") from None
        if not isinstance(response, dict):
            raise _RequestFailed("OpenRouter returned an invalid response object.")
        return response

    def _completion(self, response: dict[str, Any], seconds: float) -> Completion:
        raw = response.get("usage")
        raw = raw if isinstance(raw, dict) else {}
        tokens = _tokens(raw)
        actual = _number(raw.get("cost"))
        if actual is not None:
            usage = {**(tokens or {}), "requests": 1}
            prices = {**dict.fromkeys(tokens or {}, 0.0), "requests": actual * self._eur_per_usd}
            cost, source = actual, "openrouter"
        elif raw.get("cost") is None and tokens is not None and tokens.keys() <= self._rates.keys():
            usage = tokens
            prices = {unit: self._rates[unit] * self._eur_per_usd / 1_000_000 for unit in tokens}
            cost = sum(count * self._rates[unit] for unit, count in tokens.items()) / 1_000_000
            source = "configured_rates"
        else:
            return Completion(None, self.model, {}, {}, seconds, "OpenRouter did not provide a usable charge.")
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if choice.get("finish_reason") == "length":
            return Completion(None, self.model, usage, prices, seconds, "OpenRouter output token limit reached.",
                              cost, source)
        if (response.get("error") or not isinstance(text, str) or not text.strip()
                or choice.get("finish_reason") not in (None, "stop")):
            return Completion(None, self.model, usage, prices, seconds, "OpenRouter did not complete the response.",
                              cost, source)
        return Completion(text, self.model, usage, prices, seconds, cost_usd=cost, cost_source=source)
