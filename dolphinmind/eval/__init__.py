"""Evaluation harness for DolphinAgent: dataset schema, synthetic question generator and metrics."""

from .dataset import QUESTION_TYPES, UNANSWERABLE, EvalItem, check_evidence, load_jsonl, save_jsonl
from .generator import SyntheticBenchmark, generate, plant_sentences
from .metrics import (METRICS, Estimate, QuestionScore, bootstrap_ci, citation_precision, evidence_recall,
                      exact_match, format_summary, fuzzy_match, is_abstention, normalize_answer, score_agent_result,
                      score_question, summarize, token_f1)

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
    "METRICS",
    "Estimate",
    "QuestionScore",
    "bootstrap_ci",
    "citation_precision",
    "evidence_recall",
    "exact_match",
    "format_summary",
    "fuzzy_match",
    "is_abstention",
    "normalize_answer",
    "score_agent_result",
    "score_question",
    "summarize",
    "token_f1",
]
