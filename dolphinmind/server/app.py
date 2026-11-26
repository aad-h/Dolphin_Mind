"""FastAPI service for DolphinAgent: upload a document, ask questions, stream agent steps.

Endpoints
  GET  /health                      liveness + number of cached documents
  POST /docs                        upload text -> ``doc_id`` (content-addressed, cached in memory)
  GET  /docs/{doc_id}               chunk count and chunking parameters
  GET  /docs/{doc_id}/chunks/{i}    one chunk's text (for highlighting citations)
  POST /ask                         run the ReAct agent, return answer + citations + steps
  GET  /ask/stream                  same, as Server-Sent Events: one ``step`` event per tool
                                    call, then ``answer`` (or ``error``)

The chat backend comes from ``llm_factory``, called once per question so that stateful
backends (``ScriptedLLM``) and per-query token usage stay isolated. The default builds an
``OpenAICompatLLM`` from the ``DOLPHIN_*`` environment variables; no key is ever accepted
over HTTP.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable, Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..agent.cli import result_to_dict, step_to_dict
from ..agent.llm import LLM, LLMError, OpenAICompatLLM
from ..agent.react import AgentResult, ReActAgent, Step
from ..agent.tools import Toolbox
from .store import DocStore, StoredDoc

MAX_STEPS_LIMIT = 20


class DocIn(BaseModel):
    text: str = Field(min_length=1, description="UTF-8 document text")
    chunk_size: int = Field(256, ge=16, le=4096, description="words per chunk")
    overlap: int = Field(64, ge=0, description="overlapping words between chunks")

    @model_validator(mode="after")
    def _check(self) -> "DocIn":
        if not self.text.strip():
            raise ValueError("text is empty")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    doc_id: Optional[str] = Field(None, description="id from POST /docs")
    text: Optional[str] = Field(None, description="inline document (uploaded and cached on the fly)")
    max_steps: int = Field(6, ge=1, le=MAX_STEPS_LIMIT)

    @model_validator(mode="after")
    def _one_source(self) -> "AskIn":
        if (self.doc_id is None) == (self.text is None):
            raise ValueError("give exactly one of doc_id or text")
        return self


def _default_llm_factory() -> LLM:
    return OpenAICompatLLM()


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    llm_factory: Callable[[], LLM] = _default_llm_factory,
    store: Optional[DocStore] = None,
    max_doc_chars: int = 5_000_000,
) -> FastAPI:
    """Build the service. Pass a ``ScriptedLLM`` factory (and a ``DocStore``) in tests."""
    app = FastAPI(title="DolphinAgent", version="0.1.0",
                  description="Tool-using research agent over long documents.")
    docs = store if store is not None else DocStore()
    app.state.store = docs

    def add_doc(text: str, chunk_size: int = 256, overlap: int = 64) -> StoredDoc:
        if len(text) > max_doc_chars:
            raise HTTPException(413, f"document longer than {max_doc_chars} characters")
        return docs.add(text, chunk_size, overlap)

    def get_doc(doc_id: str) -> StoredDoc:
        stored = docs.get(doc_id)
        if stored is None:
            raise HTTPException(404, f"unknown doc_id {doc_id!r}; upload it with POST /docs")
        return stored

    def run_agent(stored: StoredDoc, question: str, max_steps: int,
                  on_step: Optional[Callable[[Step], None]] = None) -> dict:
        """Run one question with a fresh LLM and toolbox; raises LLMError/RuntimeError on backend failure."""
        llm = llm_factory()
        try:
            toolbox = Toolbox(stored.doc, stored.retriever)
            result: AgentResult = ReActAgent(llm, toolbox, max_steps=max_steps).run(question, on_step=on_step)
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        out = result_to_dict(result, stored.doc, getattr(llm, "usage", None))
        out["doc_id"] = stored.doc_id
        return out

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "docs": len(docs)}

    @app.post("/docs")
    def upload(body: DocIn) -> dict:
        return add_doc(body.text, body.chunk_size, body.overlap).info()

    @app.get("/docs/{doc_id}")
    def doc_info(doc_id: str) -> dict:
        return get_doc(doc_id).info()

    @app.get("/docs/{doc_id}/chunks/{chunk_id}")
    def chunk(doc_id: str, chunk_id: int) -> dict:
        stored = get_doc(doc_id)
        if not 0 <= chunk_id < len(stored.doc.chunks):
            raise HTTPException(404, f"chunk_id must be in [0, {len(stored.doc.chunks) - 1}]")
        return {"doc_id": doc_id, "chunk_id": chunk_id, "text": stored.doc.chunks[chunk_id]}

    @app.post("/ask")
    def ask(body: AskIn) -> dict:
        stored = get_doc(body.doc_id) if body.doc_id is not None else add_doc(body.text or "")
        try:
            return run_agent(stored, body.question, body.max_steps)
        except (LLMError, RuntimeError) as e:
            raise HTTPException(502, f"model backend error: {e}") from e

    @app.get("/ask/stream")
    def ask_stream(
        doc_id: str,
        question: str = Query(min_length=1),
        max_steps: int = Query(6, ge=1, le=MAX_STEPS_LIMIT),
    ) -> StreamingResponse:
        stored = get_doc(doc_id)  # 404 before the stream starts
        events: "queue.Queue[Optional[str]]" = queue.Queue()

        def worker() -> None:
            n = 0

            def on_step(step: Step) -> None:
                nonlocal n
                n += 1
                if step.tool:  # the final answer step arrives with the answer event
                    events.put(_sse("step", {"index": n, **step_to_dict(step)}))

            try:
                events.put(_sse("answer", run_agent(stored, question, max_steps, on_step)))
            except (LLMError, RuntimeError) as e:
                events.put(_sse("error", {"detail": f"model backend error: {e}"}))
            except Exception as e:  # noqa: BLE001 - report instead of hanging the stream
                events.put(_sse("error", {"detail": f"{type(e).__name__}: {e}"}))
            finally:
                events.put(None)

        def stream() -> Iterator[str]:
            yield _sse("start", {"doc_id": doc_id, "question": question, "max_steps": max_steps})
            threading.Thread(target=worker, daemon=True).start()
            while (item := events.get()) is not None:
                yield item

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
