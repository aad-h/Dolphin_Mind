"""Minimal chat-model interface so the agent is backend-agnostic."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    def chat(self, messages: List[Message]) -> str: ...


class HFChatLLM:
    """Wrap a local Hugging Face model + tokenizer (e.g. from dolphinmind.models.load_qwen)."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 256, temperature: float = 0.2):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def chat(self, messages: List[Message]) -> str:
        from ..models import generate

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer, _ = generate(self.model, self.tokenizer, prompt, self.max_new_tokens, self.temperature)
        return answer


class ScriptedLLM:
    """Replays fixed responses in order. Used in tests and for demos without a GPU."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[List[Message]] = []

    def chat(self, messages: List[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


class LLMError(RuntimeError):
    """Raised when a remote chat backend fails permanently (bad request, auth, retries exhausted)."""


@dataclass
class Usage:
    """Running token usage across calls, as reported by the server."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: Optional[Dict[str, Any]]) -> None:
        self.calls += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)


class OpenAICompatLLM:
    """Chat backend for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with OpenAI, OpenRouter, Together, Groq, vLLM, llama.cpp server, Ollama, etc.
    The API key is read from the ``DOLPHIN_API_KEY`` environment variable (never pass it
    on the command line or commit it); the base URL may come from ``DOLPHIN_BASE_URL``.
    A missing key is allowed for local servers that need no auth.

    Transient failures (network errors, timeouts, HTTP 429 and 5xx) are retried with
    exponential backoff, honouring ``Retry-After``. Token usage accumulates in ``usage``
    and the usage of the most recent call is kept in ``last_usage``; ``chat_with_usage``
    also returns each call's own usage, which stays exact when threads share the client.
    """

    RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        import httpx

        self.model = model or os.environ.get("DOLPHIN_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("DOLPHIN_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = api_key if api_key is not None else os.environ.get("DOLPHIN_API_KEY")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_retries, self.backoff, self._sleep = max_retries, backoff, sleep
        self.usage = Usage()
        self.last_usage: Dict[str, int] = {}
        self._usage_lock = threading.Lock()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    def __repr__(self) -> str:  # never include the key
        return f"OpenAICompatLLM(model={self.model!r}, base_url={self.base_url!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatLLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.backoff * (2 ** attempt)

    def chat(self, messages: List[Message]) -> str:
        return self.chat_with_usage(messages)[0]

    def chat_with_usage(self, messages: List[Message]) -> Tuple[str, Tuple[int, int]]:
        """Reply plus ``(prompt_tokens, completion_tokens)`` reported for this call."""
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp)
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in self.RETRY_STATUS:
                    raise LLMError(last_err)
                retry_after = resp.headers.get("retry-after")
            if attempt < self.max_retries:
                self._sleep(self._delay(attempt, retry_after))
        raise LLMError(f"giving up after {self.max_retries + 1} attempts: {last_err}")

    def _parse(self, resp: Any) -> Tuple[str, Tuple[int, int]]:
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed response: {resp.text[:300]}") from e
        usage = data.get("usage") or {}
        last = {k: int(usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        with self._usage_lock:
            self.usage.add(usage)
            self.last_usage = last
        return content or "", (last["prompt_tokens"], last["completion_tokens"])
