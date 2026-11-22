"""Per-question scoring and aggregate metrics with bootstrap confidence intervals.

Answer accuracy
    * ``exact``  normalized string equality (lowercase, no punctuation/articles,
      collapsed whitespace); numbers compare by value ("1,200" == "1200.0").
    * ``fuzzy``  the normalized gold answer appears as a whole-token span in the
      prediction, or token F1 >= ``fuzzy_threshold``; for numeric gold answers the
      prediction must state exactly one number, equal to the gold value, so
      listing every candidate number doesn't count.
    * ``f1``     SQuAD-style token F1.

  For ``unanswerable`` items a prediction is correct iff it abstains
  ("not found", "unknown", "the document does not say", ...); for answerable
  items an abstention is always wrong.

Citations
    * ``evidence_recall``     fraction of gold evidence strings found in at least
      one cited chunk (the "is the claim in the cited chunk?" faithfulness score).
    * ``citation_precision``  fraction of cited chunks that contain any evidence.

  Both are ``None`` (excluded from averages) when undefined: no gold evidence, or
  no citations for precision.

Cost
    tool calls, prompt/completion tokens, latency and optional USD cost per
    question; :func:`summarize` reports the mean of every metric with a
    percentile-bootstrap 95% CI, overall and per question type.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import UNANSWERABLE, EvalItem

# -- answer normalization ---------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_NUMBER = re.compile(rf"(?<![\w.,])(?:{_NUM})(?![\w]|[.,]\d)")
_TOKEN = re.compile(rf"(?P<num>{_NUMBER.pattern})|(?P<word>[^\W_]+)")

_ABSTAIN_PATTERNS = [
    r"\bnot found\b", r"\bunanswerable\b", r"\bunknown\b", r"\bn/a\b", r"\bno answer\b",
    r"\b(?:i )?(?:do not|don't|cannot|can't) (?:know|find|determine|answer)\b",
    r"\bnot (?:mentioned|stated|specified|given|provided|in the (?:document|text))\b",
    r"\b(?:document|text|passage)s? (?:does|do) not (?:say|mention|state|contain|specify)\b",
    r"\bno (?:information|mention|evidence)\b", r"\binsufficient (?:information|evidence)\b",
]
_ABSTAIN = re.compile("|".join(_ABSTAIN_PATTERNS))


def _format_number(token: str) -> str:
    value = float(token.replace(",", ""))
    return str(int(value)) if value.is_integer() else repr(value)


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    for m in _TOKEN.finditer(text.lower()):
        if m.group("num"):
            out.append(_format_number(m.group("num")))
        elif m.group("word") not in _ARTICLES:
            out.append(m.group("word"))
    return out


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, canonicalize numbers ("1,200.0" -> "1200"), squeeze spaces."""
    return " ".join(_tokens(text))


def extract_numbers(text: str) -> List[float]:
    """All standalone numbers in ``text`` in order (thousands separators allowed; "ALPHA7392" is not one)."""
    return [float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)]


def is_abstention(text: str) -> bool:
    """True if the prediction declines to answer (empty, "not found", "the text does not say", ...)."""
    norm = " ".join(text.lower().replace("’", "'").split())
    return not norm or bool(_ABSTAIN.search(norm))


def _gold_number(gold: str) -> Optional[float]:
    """The value of a purely numeric gold answer, else ``None``."""
    nums = extract_numbers(gold)
    return nums[0] if len(nums) == 1 and len(_tokens(gold)) == 1 else None


# -- answer accuracy ----------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style bag-of-tokens F1 after :func:`normalize_answer`."""
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def answer_f1(prediction: str, gold: str) -> float:
    """Token F1, except unanswerable gold scores 1/0 on abstention and abstaining on answerable gold scores 0."""
    if gold == UNANSWERABLE:
        return float(is_abstention(prediction))
    return 0.0 if is_abstention(prediction) else token_f1(prediction, gold)


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized equality; numbers by value; abstention-aware for ``"not found"`` gold answers."""
    if gold == UNANSWERABLE:
        return is_abstention(prediction)
    if is_abstention(prediction):
        return False
    return normalize_answer(prediction) == normalize_answer(gold)


def fuzzy_match(prediction: str, gold: str, threshold: float = 0.8) -> bool:
    """Lenient accuracy: gold span contained in prediction, or token F1 >= ``threshold``."""
    if exact_match(prediction, gold):
        return True
    if gold == UNANSWERABLE or is_abstention(prediction):
        return False
    g = _gold_number(gold)
    if g is not None:
        nums = extract_numbers(prediction)
        return len(nums) == 1 and math.isclose(nums[0], g)
    pred_n, gold_n = normalize_answer(prediction), normalize_answer(gold)
    if gold_n and f" {gold_n} " in f" {pred_n} ":
        return True
    return token_f1(prediction, gold) >= threshold


# -- citation faithfulness ----------------------------------------------------------

def _squash(text: str) -> str:
    return " ".join(text.split())


def evidence_recall(evidence: Sequence[str], cited_chunks: Sequence[str]) -> Optional[float]:
    """Fraction of evidence strings that appear (whitespace-insensitive) in some cited chunk."""
    if not evidence:
        return None
    chunks = [_squash(c) for c in cited_chunks]
    return sum(any(_squash(e) in c for c in chunks) for e in evidence) / len(evidence)


def citation_precision(evidence: Sequence[str], cited_chunks: Sequence[str]) -> Optional[float]:
    """Fraction of cited chunks that contain at least one evidence string."""
    if not evidence or not cited_chunks:
        return None
    ev = [_squash(e) for e in evidence]
    return sum(any(e in _squash(c) for e in ev) for c in cited_chunks) / len(cited_chunks)


# -- per-question record ------------------------------------------------------------

@dataclass
class QuestionScore:
    """Everything measured for one (system, question) pair; one CSV row."""

    id: str
    type: str
    gold: str
    prediction: str
    exact: bool
    fuzzy: bool
    f1: float
    evidence_recall: Optional[float] = None
    citation_precision: Optional[float] = None
    citations: List[int] = field(default_factory=list)
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: Optional[float] = None
    system: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def cost_usd(prompt_tokens: int, completion_tokens: int,
             usd_per_m_input: float, usd_per_m_output: float) -> float:
    """Dollar cost given per-million-token prices."""
    return (prompt_tokens * usd_per_m_input + completion_tokens * usd_per_m_output) / 1e6


def score_question(
    item: EvalItem,
    prediction: str,
    cited_chunks: Sequence[str] = (),
    citations: Sequence[int] = (),
    *,
    tool_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_s: float = 0.0,
    prices: Optional[Tuple[float, float]] = None,
    system: str = "",
    fuzzy_threshold: float = 0.8,
) -> QuestionScore:
    """Score one prediction against its gold item. ``prices`` is (USD/1M input, USD/1M output)."""
    return QuestionScore(
        id=item.id, type=item.type, gold=item.answer, prediction=prediction,
        exact=exact_match(prediction, item.answer),
        fuzzy=fuzzy_match(prediction, item.answer, fuzzy_threshold),
        f1=answer_f1(prediction, item.answer),
        evidence_recall=evidence_recall(item.evidence, cited_chunks),
        citation_precision=citation_precision(item.evidence, cited_chunks),
        citations=list(citations), tool_calls=tool_calls,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, latency_s=latency_s,
        cost_usd=cost_usd(prompt_tokens, completion_tokens, *prices) if prices else None,
        system=system,
    )


def score_agent_result(
    item: EvalItem,
    result: Any,
    chunks: Sequence[str],
    *,
    usage: Optional[Dict[str, int]] = None,
    latency_s: Optional[float] = None,
    prices: Optional[Tuple[float, float]] = None,
    system: str = "react",
) -> QuestionScore:
    """Score a :class:`~dolphinmind.agent.react.AgentResult`.

    ``chunks`` are the document chunks the citation ids index into; ``usage`` holds
    ``prompt_tokens``/``completion_tokens`` for this question (e.g. the delta of
    ``OpenAICompatLLM.usage``); latency defaults to the sum of step times.
    """
    usage = usage or {}
    cited = [chunks[i] for i in result.citations if 0 <= i < len(chunks)]
    if latency_s is None:
        latency_s = sum(s.seconds for s in result.steps)
    return score_question(
        item, result.answer, cited, result.citations,
        tool_calls=result.tool_calls,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        latency_s=latency_s, prices=prices, system=system,
    )


# -- aggregation --------------------------------------------------------------------

@dataclass
class Estimate:
    """A mean with a bootstrap confidence interval over ``n`` defined values."""

    mean: float
    lo: float
    hi: float
    n: int

    def fmt(self, digits: int = 3, pct: bool = False) -> str:
        if self.n == 0:
            return "n/a"
        k = 100.0 if pct else 1.0
        return f"{self.mean * k:.{digits}f} [{self.lo * k:.{digits}f}, {self.hi * k:.{digits}f}]"


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> Estimate:
    """Percentile bootstrap CI of ``stat`` (default mean). ``None``/NaN values are dropped."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray([float(v) for v in values if v is not None and not _isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    point = float(stat(arr))
    if n == 1 or np.all(arr == arr[0]):
        return Estimate(point, point, point, n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.apply_along_axis(stat, 1, arr[idx]) if stat is not np.mean else arr[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return Estimate(point, float(lo), float(hi), n)


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


METRICS = ("exact", "fuzzy", "f1", "evidence_recall", "citation_precision",
           "tool_calls", "total_tokens", "latency_s", "cost_usd")


def summarize(
    scores: Sequence[QuestionScore],
    *,
    by_type: bool = True,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> Dict[str, Dict[str, Estimate]]:
    """Mean + bootstrap CI for every metric in :data:`METRICS`.

    Returns ``{"all": {metric: Estimate}, "<type>": {...}, ...}``; per-type groups
    appear in :data:`~dolphinmind.eval.dataset.QUESTION_TYPES` order when present.
    """
    from .dataset import QUESTION_TYPES

    groups: Dict[str, List[QuestionScore]] = {"all": list(scores)}
    if by_type:
        for t in QUESTION_TYPES:
            sub = [s for s in scores if s.type == t]
            if sub:
                groups[t] = sub
    out: Dict[str, Dict[str, Estimate]] = {}
    for name, group in groups.items():
        out[name] = {m: bootstrap_ci([_value(s, m) for s in group], n_boot=n_boot,
                                     confidence=confidence, seed=seed) for m in METRICS}
    return out


def _value(score: QuestionScore, metric: str) -> Optional[float]:
    v = getattr(score, metric)
    return None if v is None else float(v)


def format_summary(summary: Dict[str, Dict[str, Estimate]],
                   metrics: Sequence[str] = ("exact", "fuzzy", "evidence_recall", "tool_calls", "total_tokens",
                                             "latency_s")) -> str:
    """Markdown table: one row per group, each cell ``mean [lo, hi]``."""
    pct = {"exact", "fuzzy", "f1", "evidence_recall", "citation_precision"}
    header = "| group | n | " + " | ".join(metrics) + " |"
    rule = "|" + "---|" * (len(metrics) + 2)
    rows = [header, rule]
    for group, est in summary.items():
        n = est[metrics[0]].n if metrics else 0
        cells = [est[m].fmt(1 if m in pct else 2, pct=m in pct) for m in metrics]
        rows.append(f"| {group} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
