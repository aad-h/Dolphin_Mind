"""Needle-in-a-haystack benchmark helpers."""

from __future__ import annotations

import random
import re

import numpy as np

BASE_TEXT = (
    "Artificial intelligence has transformed numerous industries through machine learning algorithms that "
    "process vast amounts of data efficiently. Neural networks learn complex patterns through iterative "
    "training processes. Deep learning architectures achieve remarkable performance on tasks including "
    "image recognition and natural language processing. Researchers continuously push the boundaries of "
    "what is computationally possible. Applications span from healthcare diagnostics to autonomous vehicle "
    "systems. The field evolves rapidly with breakthrough discoveries. Computational resources become "
    "increasingly accessible to practitioners. Ethical considerations remain paramount in deployment."
)

NEEDLES = [
    {"text": "The authentication code is ALPHA7392BETA", "question": "What is the authentication code?",
     "answer": "ALPHA7392BETA", "category": "exact_match", "difficulty": "easy"},
    {"text": "Dr. Martinez discovered the cure using osmium and rhodium compounds",
     "question": "What elements did Dr. Martinez use?", "answer": "osmium and rhodium",
     "category": "multi_fact", "difficulty": "medium"},
    {"text": "The ancient prophecy will be fulfilled on the seventh moon",
     "question": "When will the prophecy be fulfilled?", "answer": "seventh moon",
     "category": "extractive", "difficulty": "medium"},
    {"text": "Project Phoenix requires exactly 47 minutes at 380 celsius with 2.5 atmospheres pressure",
     "question": "What are the Project Phoenix parameters?", "answer": "47 minutes, 380 celsius, 2.5 atmospheres",
     "category": "multi_fact", "difficulty": "hard"},
    {"text": "The encryption key is QE-9X7K-MM4P-ZZ82", "question": "What is the encryption key?",
     "answer": "QE-9X7K-MM4P-ZZ82", "category": "exact_match", "difficulty": "easy"},
    {"text": "According to Chen et al Nature study, the recommended dosage is 250 milligrams",
     "question": "What dosage did Chen recommend?", "answer": "250 milligrams",
     "category": "extractive", "difficulty": "medium"},
    {"text": "The coordinates are latitude 42.3601 north longitude 71.0589 west",
     "question": "What are the coordinates?", "answer": "42.3601 north, 71.0589 west",
     "category": "multi_fact", "difficulty": "hard"},
    {"text": "The sequence is: press blue button, wait 10 seconds, then enter code 8472",
     "question": "What is the activation sequence?", "answer": "press blue button, wait 10 seconds, enter 8472",
     "category": "multi_step", "difficulty": "hard"},
]


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch so sampled generations are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_haystack(n_words: int) -> str:
    words = []
    base = BASE_TEXT.split()
    while len(words) < n_words:
        words.extend(base)
    return " ".join(words[:n_words])


def insert_needle(haystack: str, needle: str, depth_percent: float) -> str:
    """Insert `needle` at depth_percent of the haystack, measured in words (same as the original benchmark)."""
    if not 0 <= depth_percent <= 100:
        raise ValueError("depth_percent must be in [0, 100]")
    words = haystack.split()
    words.insert(int(len(words) * depth_percent / 100), needle)
    return " ".join(words)


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower())


def check_answer(response: str, expected: str, category: str) -> bool:
    """Exact match for codes/keys; >=60% key-term overlap for multi-part answers."""
    resp, exp = _norm(response), _norm(expected)
    if category == "exact_match":
        return exp in resp
    terms = exp.split()
    if len(terms) > 1:
        key_terms = [t for t in terms if len(t) > 2]
        hits = sum(1 for t in key_terms if t in resp)
        return hits >= 0.6 * len(key_terms)
    return exp in resp


def cohens_d(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled else 0.0
