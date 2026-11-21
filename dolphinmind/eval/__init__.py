"""Evaluation harness for DolphinAgent: dataset schema and synthetic question generator."""

from .dataset import QUESTION_TYPES, UNANSWERABLE, EvalItem, check_evidence, load_jsonl, save_jsonl
from .generator import SyntheticBenchmark, generate, plant_sentences

__all__ = [
    "QUESTION_TYPES",
    "UNANSWERABLE",
    "EvalItem",
    "check_evidence",
    "load_jsonl",
    "save_jsonl",
    "SyntheticBenchmark",
    "generate",
    "plant_sentences",
]
