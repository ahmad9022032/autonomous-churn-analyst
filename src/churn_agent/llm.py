"""Provider-agnostic chat client (Groq/OpenRouter via OpenAI-compatible API).

Free-tier discipline lives here: honor Retry-After on 429s, exponential
backoff with jitter, a small attempt cap, and temperature 0 so manual testing
is reproducible. `FakeLLM` implements the same interface for offline tests —
the whole agent loop is testable without burning rate limit.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

from .config import AgentConfig


class LLMUnavailable(Exception):
    """Raised when the provider keeps failing after retries."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    parse_error: bool = False


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_json_object(text: str) -> dict | None:
    """json.loads with a balanced-brace rescue for chatty models."""
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        out = json.loads(text[start : i + 1])
                        return out if isinstance(out, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class LLMClient:
    MAX_ATTEMPTS = 5

    def __init__(self, config: AgentConfig):
        from openai import OpenAI

        if not config.api_key:
            raise LLMUnavailable(
                "no API key configured — copy .env.example to .env and set LLM_API_KEY "
                "(free key at https://console.groq.com)"
            )
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.calls_made = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import openai

        last_error: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools or openai.NOT_GIVEN,
                    temperature=0,
                    max_tokens=1024,
                )
                self.calls_made += 1
                return self._parse(response)
            except (openai.RateLimitError, openai.InternalServerError,
                    openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_error = exc
                self._sleep(exc, attempt)
            except (openai.NotFoundError, openai.AuthenticationError,
                    openai.PermissionDeniedError, openai.BadRequestError) as exc:
                # configuration problems don't improve with retries
                raise LLMUnavailable(
                    f"provider rejected the request ({type(exc).__name__}): {exc}. "
                    "Check LLM_MODEL / LLM_API_KEY in .env"
                ) from exc
            except openai.APIStatusError as exc:
                # any other HTTP status (413 payload too large, 409, ...) —
                # surface as an honest failure rather than an unhandled traceback
                raise LLMUnavailable(
                    f"provider error {getattr(exc, 'status_code', '?')}: {exc}"
                ) from exc
        raise LLMUnavailable(f"provider kept failing after {self.MAX_ATTEMPTS} attempts: {last_error}")

    @staticmethod
    def _sleep(exc: Exception, attempt: int) -> None:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                match = re.match(r"[\d.]+", header)
                retry_after = float(match.group()) if match else None
        delay = retry_after if retry_after else min(2 ** attempt + random.random(), 30)
        time.sleep(min(delay, 60))

    @staticmethod
    def _parse(response) -> LLMResponse:
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                parse_error = not isinstance(args, dict)
                args = args if isinstance(args, dict) else {}
            except json.JSONDecodeError:
                args, parse_error = {}, True
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args, parse_error=parse_error))
        return LLMResponse(content=message.content, tool_calls=calls)


class FakeLLM:
    """Scripted stand-in for offline tests. Same interface as LLMClient."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls_made = 0
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.seen_messages.append(list(messages))
        self.calls_made += 1
        if not self.script:
            raise LLMUnavailable("FakeLLM script exhausted")
        return self.script.pop(0)
