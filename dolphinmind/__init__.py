"""DolphinMind: long-context QA for small language models via semantic retrieval."""

from .retrieval import chunk_sentences, SemanticRetriever
from .evaluation import insert_needle, check_answer, cohens_d, set_seed

__all__ = [
    "chunk_sentences",
    "SemanticRetriever",
    "insert_needle",
    "check_answer",
    "cohens_d",
    "set_seed",
]

__version__ = "0.1.0"
