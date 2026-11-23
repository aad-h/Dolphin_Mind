"""Tools the agent can call. Each tool works on a chunked Document and records which chunks it showed."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..retrieval import SemanticRetriever, chunk_sentences


@dataclass
class Document:
    """A long text split into numbered chunks the agent can search and cite."""

    text: str
    chunk_size: int = 256
    overlap: int = 64
    chunks: List[str] = field(init=False)

    def __post_init__(self):
        self.chunks = chunk_sentences(self.text, self.chunk_size, self.overlap) or [self.text]


@dataclass
class Tool:
    name: str
    description: str
    args: Dict[str, str]  # arg name -> type/description, shown to the model
    fn: Callable[..., str]

    def spec(self) -> str:
        arg_str = ", ".join(f"{k}: {v}" for k, v in self.args.items())
        return f"- {self.name}({arg_str}): {self.description}"


class Toolbox:
    """Builds the standard tool set over a document and tracks chunks the agent has seen."""

    def __init__(self, doc: Document, retriever: Optional[SemanticRetriever] = None, max_chars: int = 3000):
        self.doc = doc
        self.retriever = retriever
        self.max_chars = max_chars
        self.seen: set[int] = set()
        # reuse an index that already covers these chunks (e.g. shared across eval questions)
        if retriever is not None and (retriever.index is None or retriever.chunks != doc.chunks):
            retriever.build(doc.chunks)
        self.tools: Dict[str, Tool] = {t.name: t for t in self._build()}

    def reset(self) -> None:
        """Forget which chunks were shown, so the toolbox can serve a new question."""
        self.seen.clear()

    # -- helpers -------------------------------------------------------------
    def _show(self, ids: List[int]) -> str:
        """Render chunks for the model, truncating each one separately so every shown chunk is visible."""
        self.seen.update(ids)
        return "\n\n".join(f"[chunk {i}] {self.doc.chunks[i][: self.max_chars]}" for i in ids)

    # -- tools ---------------------------------------------------------------
    def search_doc(self, query: str, k: int = 3) -> str:
        if self.retriever is None:
            return "error: semantic search unavailable; use grep"
        ids = sorted(i for i, _ in self.retriever.search(query, top_k=int(k)))
        return self._show(ids) or "no results"

    def grep(self, pattern: str, k: int = 3) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        ids = [i for i, c in enumerate(self.doc.chunks) if rx.search(c)][: int(k)]
        return self._show(ids) if ids else f"no matches for {pattern!r}"

    def read_section(self, chunk_id: int) -> str:
        i = int(chunk_id)
        if not 0 <= i < len(self.doc.chunks):
            return f"error: chunk_id must be in [0, {len(self.doc.chunks) - 1}]"
        return self._show([i])

    def calculator(self, expression: str) -> str:
        try:
            return str(_safe_eval(expression))
        except Exception as e:  # noqa: BLE001 - surfaced to the model as an observation
            return f"error: {e}"

    def _build(self) -> List[Tool]:
        return [
            Tool("search_doc", "semantic search; returns the k most relevant chunks",
                 {"query": "str", "k": "int=3"}, self.search_doc),
            Tool("grep", "case-insensitive regex/keyword search; returns first k matching chunks",
                 {"pattern": "str", "k": "int=3"}, self.grep),
            Tool("read_section", f"read one chunk by id (0..{len(self.doc.chunks) - 1}), e.g. a neighbour of a hit",
                 {"chunk_id": "int"}, self.read_section),
            Tool("calculator", "evaluate arithmetic like '(47*2)+3.5'",
                 {"expression": "str"}, self.calculator),
        ]

    def call(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}; available: {', '.join(self.tools)}"
        try:
            return tool.fn(**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"

    def specs(self) -> str:
        return "\n".join(t.spec() for t in self.tools.values())


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str):
    """Arithmetic only: no names, calls or attributes, so model output can't execute code."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")

    return ev(ast.parse(expr, mode="eval"))
