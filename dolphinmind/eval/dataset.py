"""Eval dataset schema and JSONL I/O.

One JSON object per line::

    {"id": "q0001", "question": "...", "answer": "...", "type": "multi_hop",
     "evidence": ["sentence that supports the answer", "..."],
     "doc_id": "book", "metadata": {"depths": [0.12, 0.81]}}

``evidence`` holds verbatim text from the document that a correct answer must
rely on; it is empty for ``unanswerable`` items, whose answer is always
:data:`UNANSWERABLE`. Citation-faithfulness metrics check that cited chunks
contain this evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

QUESTION_TYPES = ("single_hop", "multi_hop", "numeric", "unanswerable")
UNANSWERABLE = "not found"

_REQUIRED = ("id", "question", "answer", "type")


@dataclass
class EvalItem:
    """One benchmark question with its gold answer and supporting evidence."""

    id: str
    question: str
    answer: str
    type: str
    evidence: List[str] = field(default_factory=list)
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the item breaks the schema."""
        for name in ("id", "question", "answer", "type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name!r} must be a non-empty string")
        if self.type not in QUESTION_TYPES:
            raise ValueError(f"type must be one of {QUESTION_TYPES}, got {self.type!r}")
        if not isinstance(self.evidence, list) or not all(isinstance(e, str) and e for e in self.evidence):
            raise ValueError("evidence must be a list of non-empty strings")
        if self.type == "unanswerable":
            if self.evidence:
                raise ValueError("unanswerable items must have no evidence")
            if self.answer != UNANSWERABLE:
                raise ValueError(f"unanswerable items must have answer {UNANSWERABLE!r}")
        elif not self.evidence:
            raise ValueError(f"{self.type} items need at least one evidence string")
        if self.type == "multi_hop" and len(self.evidence) < 2:
            raise ValueError("multi_hop items need at least two evidence strings")
        if not isinstance(self.doc_id, str):
            raise ValueError("doc_id must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalItem":
        if not isinstance(data, dict):
            raise ValueError("record must be a JSON object")
        missing = [k for k in _REQUIRED if k not in data]
        if missing:
            raise ValueError(f"missing field(s): {', '.join(missing)}")
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
        return cls(**data)


PathLike = Union[str, Path]


def save_jsonl(items: Iterable[EvalItem], path: PathLike) -> int:
    """Write items one per line; returns the number written. Ids must be unique."""
    items = list(items)
    _check_unique(items)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return len(items)


def load_jsonl(path: PathLike) -> List[EvalItem]:
    """Read and validate a dataset; errors name the offending line. Blank lines are skipped."""
    items: List[EvalItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                items.append(EvalItem.from_dict(json.loads(line)))
            except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
                raise ValueError(f"{path}:{lineno}: {e}") from None
    _check_unique(items)
    return items


def check_evidence(items: Iterable[EvalItem], docs: Dict[str, str]) -> List[str]:
    """Return a list of problems: unknown ``doc_id`` or evidence not found verbatim in its document."""
    problems: List[str] = []
    for item in items:
        text = docs.get(item.doc_id)
        if text is None:
            problems.append(f"{item.id}: unknown doc_id {item.doc_id!r}")
            continue
        for ev in item.evidence:
            if ev not in text:
                problems.append(f"{item.id}: evidence not in document: {ev[:60]!r}")
    return problems


def _check_unique(items: List[EvalItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate id {item.id!r}")
        seen.add(item.id)
