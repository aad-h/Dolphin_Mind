"""Lightweight tracing: one span per agent run, LLM call and tool call.

:class:`Tracer` records nested :class:`Span` trees in memory with no dependencies, so
traces work offline, in tests and in the HTTP service (``GET /traces``). Parent/child
links follow :mod:`contextvars`, so spans opened in ``asyncio`` tasks or in
``asyncio.to_thread`` workers nest under the span that was current when they started.

Finished root spans are handed to ``exporters``. :class:`OpenTelemetryExporter`
replays each finished tree into OpenTelemetry (``opentelemetry-api``/``-sdk``, optional)
with the original start/end times, so any OTel backend (Jaeger, Langfuse, Honeycomb,
an OTLP collector...) can show the same trace.

Instrumentation points:

* :class:`TracedLLM` wraps any :class:`~dolphinmind.agent.llm.LLM` and opens an
  ``llm.chat`` span per call with token usage;
* :class:`~dolphinmind.agent.react.ReActAgent` (``tracer=...``) opens ``agent.run`` and
  one ``tool.<name>`` span per tool call;
* :class:`~dolphinmind.agent.multi.MultiAgent` (``tracer=...``) adds ``multi.run``,
  ``multi.plan``, ``multi.research``, ``multi.verify`` and ``multi.synthesize``.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

from .llm import LLM, Message

AttrValue = Any  # str | int | float | bool, other values are JSON-encoded on export


@dataclass
class Span:
    """One timed operation; ``children`` holds the spans opened inside it."""

    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    attributes: Dict[str, AttrValue] = field(default_factory=dict)
    start_ns: int = 0
    end_ns: Optional[int] = None
    status: str = "ok"  # ok | error
    error: Optional[str] = None
    children: List["Span"] = field(default_factory=list)
    _tracer: Optional["Tracer"] = field(default=None, repr=False, compare=False)

    @property
    def seconds(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.time_ns()
        return (end - self.start_ns) / 1e9

    def set(self, **attributes: AttrValue) -> "Span":
        """Add or overwrite attributes (``None`` values are skipped)."""
        self.attributes.update({k: v for k, v in attributes.items() if v is not None})
        return self

    def walk(self) -> Iterator["Span"]:
        """This span and all its descendants, depth first, in start order."""
        yield self
        for child in list(self.children):
            yield from child.walk()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "seconds": round(self.seconds, 6),
            "status": self.status,
            "error": self.error,
            "attributes": dict(self.attributes),
            "children": [c.to_dict() for c in list(self.children)],
        }


_current: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar("dolphin_span", default=None)

Exporter = Callable[[Span], None]


class Tracer:
    """Records span trees; keeps the ``max_traces`` most recent finished traces.

    Thread-safe. ``exporters`` are called with each finished *root* span (the whole
    tree is complete by then); an exporter that raises is ignored so tracing can never
    break a query.
    """

    def __init__(self, max_traces: int = 100, exporters: Optional[List[Exporter]] = None):
        if max_traces < 1:
            raise ValueError("max_traces must be >= 1")
        self.exporters: List[Exporter] = list(exporters or [])
        self._traces: Deque[Span] = deque(maxlen=max_traces)
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, **attributes: AttrValue) -> Iterator[Span]:
        """Open a span as a child of the current one (or a new trace). Exceptions mark it ``error``."""
        parent = _current.get()
        if parent is not None and parent._tracer is not self:
            parent = None  # a span of another tracer: start a separate trace
        s = Span(name, parent.trace_id if parent else uuid.uuid4().hex, uuid.uuid4().hex[:16],
                 parent.span_id if parent else None, start_ns=time.time_ns(), _tracer=self)
        s.set(**attributes)
        if parent is not None:
            with self._lock:
                parent.children.append(s)
        token = _current.set(s)
        try:
            yield s
        except BaseException as e:
            s.status, s.error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            s.end_ns = time.time_ns()
            _current.reset(token)
            if parent is None:
                self._finish(s)

    def _finish(self, root: Span) -> None:
        with self._lock:
            self._traces.append(root)
        for export in self.exporters:
            try:
                export(root)
            except Exception:  # noqa: BLE001 - never fail a query because of telemetry
                pass

    def traces(self, limit: Optional[int] = None) -> List[Span]:
        """Finished root spans, newest first."""
        with self._lock:
            out = list(reversed(self._traces))
        return out if limit is None else out[: max(0, limit)]

    def get(self, trace_id: str) -> Optional[Span]:
        with self._lock:
            return next((t for t in self._traces if t.trace_id == trace_id), None)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


def current_span() -> Optional[Span]:
    """The innermost open span in this context, if any."""
    return _current.get()


@contextmanager
def maybe_span(tracer: Optional[Tracer], name: str, **attributes: AttrValue) -> Iterator[Optional[Span]]:
    """``tracer.span(...)`` when tracing is on, otherwise a no-op yielding ``None``."""
    if tracer is None:
        yield None
    else:
        with tracer.span(name, **attributes) as s:
            yield s


class TracedLLM:
    """Wrap ``llm`` so every call opens an ``llm.chat`` span with its token usage.

    Exposes ``chat_with_usage`` (real usage when the backend reports it, else ``None``,
    like :class:`~dolphinmind.agent.budget.BudgetedLLM`), so it composes with budgets.
    Other attributes (``usage``, ``close``, ``calls``...) are forwarded to ``llm``.
    """

    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm, self.tracer = llm, tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)

    def chat_with_usage(self, messages: List[Message]) -> Tuple[str, Optional[Tuple[int, int]]]:
        from .budget import metered_chat

        model = getattr(self.llm, "model", None)
        with self.tracer.span("llm.chat", backend=type(self.llm).__name__,
                              model=model if isinstance(model, str) else None,
                              messages=len(messages)) as s:
            reply, (prompt, completion), estimated = metered_chat(self.llm, messages)
            s.set(prompt_tokens=prompt, completion_tokens=completion,
                  tokens_estimated=estimated, reply_chars=len(reply))
        return reply, None if estimated else (prompt, completion)

    def chat(self, messages: List[Message]) -> str:
        return self.chat_with_usage(messages)[0]


def traced(llm: LLM, tracer: Optional[Tracer]) -> LLM:
    """``llm`` wrapped in :class:`TracedLLM` for ``tracer`` (unchanged if already wrapped or no tracer)."""
    if tracer is None or (isinstance(llm, TracedLLM) and llm.tracer is tracer):
        return llm
    return TracedLLM(llm, tracer)


# -- OpenTelemetry ---------------------------------------------------------------------

def _otel_value(v: AttrValue) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    try:
        return json.dumps(v, default=str)
    except (TypeError, ValueError):
        return str(v)


class OpenTelemetryExporter:
    """Replay finished span trees into OpenTelemetry with their original timings.

    ``otel_tracer`` defaults to ``opentelemetry.trace.get_tracer("dolphinmind")``, i.e.
    whatever ``TracerProvider`` the application configured (OTLP, console, ...).
    Requires ``opentelemetry-api`` (and an SDK to actually export anything).
    """

    def __init__(self, otel_tracer: Any = None):
        from opentelemetry import trace

        self._trace = trace
        self.otel_tracer = otel_tracer or trace.get_tracer("dolphinmind")

    def __call__(self, root: Span) -> None:
        self._emit(root, None)

    def _emit(self, s: Span, parent: Any) -> None:
        ctx = self._trace.set_span_in_context(parent) if parent is not None else None
        o = self.otel_tracer.start_span(s.name, context=ctx, start_time=s.start_ns,
                                        attributes={k: _otel_value(v) for k, v in s.attributes.items()})
        o.set_attribute("dolphin.trace_id", s.trace_id)
        if s.status == "error":
            o.set_status(self._trace.Status(self._trace.StatusCode.ERROR, s.error or ""))
        for child in list(s.children):
            self._emit(child, o)
        o.end(end_time=s.end_ns if s.end_ns is not None else time.time_ns())


def format_trace(root: Span, indent: str = "  ") -> str:
    """Indented text tree: ``name  1.234s  key=value ...`` per span."""
    lines: List[str] = []

    def rec(s: Span, depth: int) -> None:
        attrs = " ".join(f"{k}={_short(v)}" for k, v in s.attributes.items())
        flag = f"  ERROR {s.error}" if s.status == "error" else ""
        lines.append(f"{indent * depth}{s.name}  {s.seconds:.3f}s  {attrs}".rstrip() + flag)
        for c in s.children:
            rec(c, depth + 1)

    rec(root, 0)
    return "\n".join(lines)


def _short(v: AttrValue, n: int = 60) -> str:
    text = v if isinstance(v, str) else json.dumps(v, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."
