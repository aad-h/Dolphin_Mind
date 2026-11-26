"""In-memory, content-addressed cache of chunked documents for the HTTP service."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..agent.tools import Document
from ..retrieval import SemanticRetriever


@dataclass
class StoredDoc:
    """A chunked document plus its (optional) prebuilt semantic index."""

    doc_id: str
    doc: Document
    retriever: Optional[SemanticRetriever] = None

    def info(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_chunks": len(self.doc.chunks),
            "chars": len(self.doc.text),
            "chunk_size": self.doc.chunk_size,
            "overlap": self.doc.overlap,
            "semantic": self.retriever is not None,
        }


def doc_key(text: str, chunk_size: int, overlap: int) -> str:
    """Stable id: the same text with the same chunking always maps to the same document."""
    h = hashlib.sha256(f"{chunk_size}:{overlap}:".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


class DocStore:
    """Thread-safe LRU cache of uploaded documents.

    Uploading identical text twice returns the cached entry, so chunking and indexing
    happen once. When more than ``max_docs`` are held the least recently used is dropped.
    ``embedder_factory`` (if given) returns an embedder used to build a ``search_doc``
    index for every new document; without it agents fall back to ``grep``.
    """

    def __init__(self, max_docs: int = 32, embedder_factory: Optional[Callable[[], Any]] = None):
        if max_docs < 1:
            raise ValueError("max_docs must be >= 1")
        self.max_docs = max_docs
        self._embedder_factory = embedder_factory
        self._embedder: Any = None
        self._docs: "OrderedDict[str, StoredDoc]" = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, text: str, chunk_size: int = 256, overlap: int = 64) -> StoredDoc:
        """Chunk (and index) ``text`` unless an identical document is already cached."""
        key = doc_key(text, chunk_size, overlap)
        with self._lock:
            if key in self._docs:
                self._docs.move_to_end(key)
                return self._docs[key]
        # chunk / embed outside the lock: it can be slow for long books
        stored = StoredDoc(key, Document(text, chunk_size=chunk_size, overlap=overlap))
        if self._embedder_factory is not None:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            stored.retriever = SemanticRetriever(self._embedder).build(stored.doc.chunks)
        with self._lock:
            stored = self._docs.setdefault(key, stored)
            self._docs.move_to_end(key)
            while len(self._docs) > self.max_docs:
                self._docs.popitem(last=False)
        return stored

    def get(self, doc_id: str) -> Optional[StoredDoc]:
        with self._lock:
            stored = self._docs.get(doc_id)
            if stored is not None:
                self._docs.move_to_end(doc_id)
            return stored
