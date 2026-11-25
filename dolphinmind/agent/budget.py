"""Per-call token metering and per-query token / cost budgets.

Backends that know their real usage expose ``chat_with_usage(messages)``, returning
``(reply, (prompt_tokens, completion_tokens))`` for *that* call. Reading usage per call
(instead of diffing a shared counter) stays correct when several workers call the same
backend concurrently. Backends without it are metered by an estimate of ~4 characters
per token.

:class:`BudgetedLLM` wraps any :class:`~dolphinmind.agent.llm.LLM` and charges every call
to a :class:`Budget`; once a call would overrun the budget it raises
:class:`BudgetExceeded` instead of calling the backend.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .llm import LLM, Message

TokenPair = Tuple[int, int]  # (prompt_tokens, completion_tokens)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for backends that report no usage."""
    return (len(text) + 3) // 4


def estimate_prompt_tokens(messages: List[Message]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def reports_usage(llm: LLM) -> bool:
    """Whether ``llm`` reports real per-call token usage via ``chat_with_usage``."""
    return callable(getattr(llm, "chat_with_usage", None))


def metered_chat(llm: LLM, messages: List[Message]) -> Tuple[str, TokenPair, bool]:
    """Call ``llm`` and return ``(reply, (prompt, completion), estimated)`` for this call only."""
    if reports_usage(llm):
        reply, usage = llm.chat_with_usage(messages)  # type: ignore[attr-defined]
        if usage is not None:
            return reply, (int(usage[0]), int(usage[1])), False
    else:
        reply = llm.chat(messages)
    return reply, (estimate_prompt_tokens(messages), estimate_tokens(reply)), True


class BudgetExceeded(Exception):
    """Raised instead of making an LLM call that the query budget cannot afford.

    Deliberately not a ``RuntimeError``: agents catch backend errors to recover, but a
    spent budget must stop the whole query.
    """


@dataclass
class Budget:
    """Token and/or USD limit for one query, safe to charge from several threads.

    ``prices`` is USD per 1M ``(input, output)`` tokens (the same convention as
    ``run_agent_eval.py --prices``) and is required for ``max_cost_usd``. A call is
    refused when the tokens already spent plus an estimate (~4 chars/token) of its
    prompt would exceed a limit. The reply's size is unknown beforehand, so the last
    accepted call can overshoot a limit by its reply (and any estimation error).
    """

    max_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    prices: Optional[Tuple[float, float]] = None
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)
    estimated: bool = field(default=False, init=False)  # some charges were ~4 chars/token estimates

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.max_cost_usd is not None:
            if self.max_cost_usd <= 0:
                raise ValueError("max_cost_usd must be > 0")
            if self.prices is None:
                raise ValueError("max_cost_usd needs prices (USD per 1M input, output tokens)")
        self._lock = threading.Lock()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def _cost(self, prompt: int, completion: int) -> Optional[float]:
        if self.prices is None:
            return None
        return (prompt * self.prices[0] + completion * self.prices[1]) / 1e6

    @property
    def cost_usd(self) -> Optional[float]:
        return self._cost(self.prompt_tokens, self.completion_tokens)

    @property
    def exhausted(self) -> bool:
        return self._over(0) is not None

    def _over(self, next_prompt: int) -> Optional[str]:
        """Why a call with ``next_prompt`` prompt tokens is unaffordable, or ``None``."""
        tokens = self.total_tokens + next_prompt
        if self.max_tokens is not None and (tokens > self.max_tokens or self.total_tokens >= self.max_tokens):
            return f"token budget {self.max_tokens} reached ({self.total_tokens} spent)"
        cost = self._cost(self.prompt_tokens + next_prompt, self.completion_tokens)
        if self.max_cost_usd is not None and cost is not None and cost > self.max_cost_usd:
            return f"cost budget ${self.max_cost_usd:g} reached (${self.cost_usd:.6f} spent)"
        return None

    def check(self, next_prompt: int = 0) -> None:
        with self._lock:
            reason = self._over(next_prompt)
        if reason:
            raise BudgetExceeded(reason)

    def charge(self, prompt: int, completion: int, estimated: bool = False) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1
            self.estimated = self.estimated or estimated

    def fresh(self) -> "Budget":
        """A new, unspent budget with the same limits (one per query)."""
        return Budget(self.max_tokens, self.max_cost_usd, self.prices)

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = self.completion_tokens = self.calls = 0
            self.estimated = False


class BudgetedLLM:
    """Charge every call of ``llm`` to ``budget``; refuse calls the budget cannot afford."""

    def __init__(self, llm: LLM, budget: Budget):
        self.llm, self.budget = llm, budget

    def chat_with_usage(self, messages: List[Message]) -> Tuple[str, Optional[TokenPair]]:
        self.budget.check(estimate_prompt_tokens(messages))
        reply, usage, estimated = metered_chat(self.llm, messages)
        self.budget.charge(*usage, estimated=estimated)
        return reply, None if estimated else usage

    def chat(self, messages: List[Message]) -> str:
        return self.chat_with_usage(messages)[0]
