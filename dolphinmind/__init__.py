"""DolphinMind: long-context QA for small language models via semantic retrieval."""

from .retrieval import HashingEmbedder, SemanticRetriever, chunk_sentences, fixed_chunks
from .evaluation import insert_needle, check_answer, cohens_d, set_seed

__all__ = [
    "chunk_sentences",
    "fixed_chunks",
    "HashingEmbedder",
    "SemanticRetriever",
    "insert_needle",
    "check_answer",
    "cohens_d",
    "set_seed",
]

__version__ = "0.1.0"
