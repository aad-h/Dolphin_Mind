"""Core DolphinMind pipeline: sentence-aware chunking + embedding retrieval."""

from __future__ import annotations

import re
from typing import List

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_sentences(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """Split text into ~chunk_size-word chunks on sentence boundaries.

    Consecutive chunks share roughly `overlap` words so a fact that straddles
    a boundary still appears whole in at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    length = 0

    for i, sent in enumerate(sentences):
        current.append(sent)
        length += len(sent.split())

        if length >= chunk_size:
            chunks.append(" ".join(current))
            # carry the last `overlap` words worth of sentences into the next chunk
            carry: List[str] = []
            carry_len = 0
            back = i
            while back >= 0 and carry_len < overlap:
                carry.insert(0, sentences[back])
                carry_len += len(sentences[back].split())
                back -= 1
            current, length = carry, carry_len

    # only emit the tail if it holds something not already in the last chunk
    if current and (not chunks or not chunks[-1].endswith(" ".join(current))):
        chunks.append(" ".join(current))
    return chunks


class SemanticRetriever:
    """Embed chunks into a FAISS inner-product index and fetch the top-k for a query."""

    def __init__(self, embedding_model=None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(model_name, device="cpu")
        self.embedder = embedding_model
        self.chunks: List[str] = []
        self.index = None

    def build(self, chunks: List[str]) -> "SemanticRetriever":
        import faiss

        if not chunks:
            raise ValueError("no chunks to index")
        emb = np.asarray(self.embedder.encode(chunks, show_progress_bar=False, batch_size=32), dtype="float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.chunks = chunks
        return self

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return the top-k chunks, re-sorted into document order."""
        import faiss

        if self.index is None:
            raise RuntimeError("call build() first")
        q = np.asarray(self.embedder.encode([query]), dtype="float32")
        faiss.normalize_L2(q)
        k = min(top_k, len(self.chunks))
        _, idx = self.index.search(q, k)
        return [self.chunks[i] for i in sorted(i for i in idx[0] if i >= 0)]

    def build_context(self, text: str, query: str, chunk_size: int = 512, overlap: int = 128, top_k: int = 5) -> str:
        """Full pipeline: chunk -> embed -> retrieve -> chronological reconstruction."""
        self.build(chunk_sentences(text, chunk_size, overlap))
        return "\n\n".join(self.retrieve(query, top_k))
