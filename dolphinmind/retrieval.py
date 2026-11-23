"""Core DolphinMind pipeline: sentence-aware chunking + embedding retrieval."""

from __future__ import annotations

import re
import zlib
from typing import List, Tuple

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have he her his how in is it its many much "
    "of on or she that the their they this to was were what when where which who whom why with".split()
)


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


def fixed_chunks(text: str, chunk_size: int = 512) -> List[str]:
    """Naive chunking: consecutive ``chunk_size``-word windows, no overlap, ignoring sentence boundaries.

    The "plain RAG" baseline; a fact that straddles a window edge is split across two chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


class HashingEmbedder:
    """Download-free bag-of-words embedder: hashed unigram counts with sublinear tf, stopwords dropped.

    A deterministic offline stand-in for sentence-transformers (``encode`` has the same shape).
    Strong on exact keyword / name overlap, blind to paraphrase, so it suits synthetic
    benchmarks with invented names, smoke tests and CI, not real semantic search.
    """

    def __init__(self, dim: int = 4096):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim

    def encode(self, texts: List[str], **_) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for w in _WORD.findall(text.lower()):
                if w not in _STOPWORDS:
                    out[row, zlib.crc32(w.encode("utf-8")) % self.dim] += 1.0
        return np.log1p(out)


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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return ``(chunk_id, cosine score)`` for the top-k chunks, best first."""
        import faiss

        if self.index is None:
            raise RuntimeError("call build() first")
        q = np.asarray(self.embedder.encode([query]), dtype="float32")
        faiss.normalize_L2(q)
        k = min(int(top_k), len(self.chunks))
        if k < 1:
            return []
        scores, idx = self.index.search(q, k)
        return [(int(i), float(sc)) for i, sc in zip(idx[0], scores[0]) if i >= 0]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return the top-k chunks, re-sorted into document order."""
        return [self.chunks[i] for i in sorted(i for i, _ in self.search(query, top_k))]

    def build_context(self, text: str, query: str, chunk_size: int = 512, overlap: int = 128, top_k: int = 5) -> str:
        """Full pipeline: chunk -> embed -> retrieve -> chronological reconstruction."""
        self.build(chunk_sentences(text, chunk_size, overlap))
        return "\n\n".join(self.retrieve(query, top_k))
