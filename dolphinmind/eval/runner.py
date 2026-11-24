"""Compare QA systems on an eval dataset: ``python run_agent_eval.py data/moby.jsonl``.

Systems (all share one chat backend and the same context budget):

* ``truncation``   the first ``context_words`` words of the document in one prompt
* ``rag``          plain RAG: fixed word windows (no overlap, sentence-blind), top-k by
                   embedding similarity, pasted best-first
* ``dolphinmind``  sentence-aware overlapping chunks, top-k, reassembled in document order
* ``react``        :class:`~dolphinmind.agent.ReActAgent` with ``search_doc`` / ``grep`` /
                   ``read_section`` / ``calculator``, citing the chunks it read
* ``multi``        :class:`~dolphinmind.agent.MultiAgent`: planner -> ReAct workers ->
                   verifier -> synthesis (opt-in via ``--systems``; several calls per question)

Single-pass systems "cite" the context they were given, so for them
``evidence_recall`` measures retrieval recall (did the evidence reach the model?);
for the agents it is citation faithfulness. ``context_words`` defaults to
``top_k * chunk_size`` so truncation gets the same word budget as retrieval.

Token counts come from the backend's own usage counter when it has one
(:class:`~dolphinmind.agent.OpenAICompatLLM`); otherwise they are estimated at
~4 characters per token and the summary says so. Questions whose backend call
fails count as wrong (prediction ``"[error]"``).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

from ..agent.llm import LLM, LLMError, Message, Usage
from ..agent.multi import MultiAgent
from ..agent.react import ReActAgent
from ..agent.tools import Document, Toolbox
from ..retrieval import HashingEmbedder, SemanticRetriever, fixed_chunks
from .dataset import QUESTION_TYPES, UNANSWERABLE, EvalItem, check_evidence, load_jsonl
from .metrics import QuestionScore, score_question, summarize

SYSTEMS = ("truncation", "rag", "dolphinmind", "react", "multi")
DEFAULT_SYSTEMS = SYSTEMS[:4]  # "multi" costs several LLM calls per question: opt in
_AGENTS = frozenset({"react", "multi"})
_NEEDS_EMBEDDER = frozenset({"rag", "dolphinmind"})
ERROR_ANSWER = "[error]"

ANSWER_SYSTEM = (
    "Answer the question using only the context. Reply with the short answer and nothing else. "
    f'If the context does not contain the answer, reply "{UNANSWERABLE}".'
)
ANSWER_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


@dataclass
class EvalConfig:
    """Knobs shared by every system so the comparison is like for like."""

    chunk_size: int = 256
    overlap: int = 64
    top_k: int = 5
    context_words: Optional[int] = None
    max_steps: int = 6
    prices: Optional[Tuple[float, float]] = None  # USD per 1M (input, output) tokens
    max_errors: int = 5

    def __post_init__(self) -> None:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if min(self.chunk_size, self.top_k, self.max_steps) < 1:
            raise ValueError("chunk_size, top_k and max_steps must be >= 1")
        if self.context_words is not None and self.context_words < 1:
            raise ValueError("context_words must be >= 1")

    @property
    def budget_words(self) -> int:
        return self.context_words or self.top_k * self.chunk_size


# -- token metering ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for backends that report no usage."""
    return (len(text) + 3) // 4


class MeteredLLM:
    """Wrap an :class:`~dolphinmind.agent.llm.LLM` and count calls and tokens.

    Uses the backend's ``usage`` counter when it has one, otherwise estimates tokens
    from message lengths (``estimated`` is then ``True``).
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.estimated = not isinstance(getattr(llm, "usage", None), Usage)
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def chat(self, messages: List[Message]) -> str:
        if self.estimated:
            reply = self.llm.chat(messages)
            self.prompt_tokens += sum(estimate_tokens(m.get("content", "")) for m in messages)
            self.completion_tokens += estimate_tokens(reply)
        else:
            usage: Usage = self.llm.usage  # type: ignore[attr-defined]
            before = (usage.prompt_tokens, usage.completion_tokens)
            reply = self.llm.chat(messages)
            self.prompt_tokens += usage.prompt_tokens - before[0]
            self.completion_tokens += usage.completion_tokens - before[1]
        self.calls += 1
        return reply

    def snapshot(self) -> Tuple[int, int]:
        return self.prompt_tokens, self.completion_tokens


# -- per-document indexes ------------------------------------------------------------

class DocIndex:
    """Chunkings and embedding indexes for one document, built lazily once and shared by all questions."""

    def __init__(self, text: str, cfg: EvalConfig, embedder: Any = None):
        self.text, self.cfg, self.embedder = text, cfg, embedder
        self.doc = Document(text, cfg.chunk_size, cfg.overlap)
        self.naive_chunks = fixed_chunks(text, cfg.chunk_size) or [text]
        self._semantic: Optional[SemanticRetriever] = None
        self._naive: Optional[SemanticRetriever] = None
        self._toolbox: Optional[Toolbox] = None
        self._truncated: Optional[str] = None

    def _need_embedder(self) -> Any:
        if self.embedder is None:
            raise ValueError("this system needs an embedder")
        return self.embedder

    @property
    def truncated(self) -> str:
        if self._truncated is None:
            self._truncated = " ".join(self.text.split()[: self.cfg.budget_words])
        return self._truncated

    @property
    def semantic(self) -> SemanticRetriever:
        if self._semantic is None:
            self._semantic = SemanticRetriever(embedding_model=self._need_embedder()).build(self.doc.chunks)
        return self._semantic

    @property
    def naive(self) -> SemanticRetriever:
        if self._naive is None:
            self._naive = SemanticRetriever(embedding_model=self._need_embedder()).build(self.naive_chunks)
        return self._naive

    def toolbox(self) -> Toolbox:
        """The agent's toolbox, reset for a new question (``search_doc`` only with an embedder)."""
        if self._toolbox is None:
            self._toolbox = Toolbox(self.doc, self.semantic if self.embedder is not None else None)
        self._toolbox.reset()
        return self._toolbox


# -- systems -------------------------------------------------------------------------

@dataclass
class Prediction:
    answer: str
    citations: List[int] = field(default_factory=list)
    cited_chunks: List[str] = field(default_factory=list)
    tool_calls: int = 0


def _ask(llm: LLM, context: str, question: str) -> str:
    messages = [{"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)}]
    return llm.chat(messages).strip()


def _truncation(llm: LLM, idx: DocIndex, question: str) -> Prediction:
    ctx = idx.truncated
    return Prediction(_ask(llm, ctx, question), [0], [ctx])


def _rag(llm: LLM, idx: DocIndex, question: str) -> Prediction:
    ids = [i for i, _ in idx.naive.search(question, idx.cfg.top_k)]  # best first
    chunks = [idx.naive_chunks[i] for i in ids]
    return Prediction(_ask(llm, "\n\n".join(chunks), question), ids, chunks)


def _dolphinmind(llm: LLM, idx: DocIndex, question: str) -> Prediction:
    ids = sorted(i for i, _ in idx.semantic.search(question, idx.cfg.top_k))  # document order
    chunks = [idx.doc.chunks[i] for i in ids]
    return Prediction(_ask(llm, "\n\n".join(chunks), question), ids, chunks)


def _react(llm: LLM, idx: DocIndex, question: str) -> Prediction:
    res = ReActAgent(llm, idx.toolbox(), max_steps=idx.cfg.max_steps).run(question)
    cited = [idx.doc.chunks[i] for i in res.citations if 0 <= i < len(idx.doc.chunks)]
    return Prediction(res.answer, list(res.citations), cited, res.tool_calls)


def _multi(llm: LLM, idx: DocIndex, question: str) -> Prediction:
    res = MultiAgent(llm, idx.toolbox(), max_steps=idx.cfg.max_steps).run(question)
    cited = [idx.doc.chunks[i] for i in res.citations if 0 <= i < len(idx.doc.chunks)]
    return Prediction(res.answer, list(res.citations), cited, res.tool_calls)


_RUNNERS: Dict[str, Callable[[LLM, DocIndex, str], Prediction]] = {
    "truncation": _truncation, "rag": _rag, "dolphinmind": _dolphinmind, "react": _react, "multi": _multi,
}


# -- evaluation loop -----------------------------------------------------------------

@dataclass
class EvalRow:
    """One (system, question) result: the score plus where it came from and any backend error."""

    score: QuestionScore
    doc_id: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.score.to_dict()
        d["citations"] = " ".join(map(str, d["citations"]))
        d["doc_id"], d["error"] = self.doc_id, self.error
        return {k: ("" if v is None else v) for k, v in d.items()}


@dataclass
class EvalRun:
    rows: List[EvalRow]
    tokens_estimated: bool = False

    def scores(self, system: Optional[str] = None) -> List[QuestionScore]:
        return [r.score for r in self.rows if system is None or r.score.system == system]

    @property
    def systems(self) -> List[str]:
        present = {r.score.system for r in self.rows}
        return [s for s in SYSTEMS if s in present]


def run_eval(
    items: Sequence[EvalItem],
    docs: Dict[str, str],
    llm: LLM,
    systems: Sequence[str] = DEFAULT_SYSTEMS,
    cfg: Optional[EvalConfig] = None,
    embedder: Any = None,
    progress: Optional[Callable[[int, int, EvalRow], None]] = None,
) -> EvalRun:
    """Run every system on every item; ``docs`` maps ``doc_id`` to document text.

    Raises ``ValueError`` for unknown systems, missing documents or a missing
    embedder, and :class:`~dolphinmind.agent.llm.LLMError` once more than
    ``cfg.max_errors`` backend calls have failed.
    """
    cfg = cfg or EvalConfig()
    if bad := [s for s in systems if s not in _RUNNERS]:
        raise ValueError(f"unknown system(s): {', '.join(bad)}; choose from {', '.join(SYSTEMS)}")
    if embedder is None and (need := _NEEDS_EMBEDDER.intersection(systems)):
        raise ValueError(f"system(s) {', '.join(sorted(need))} need an embedder")
    if missing := sorted({it.doc_id for it in items} - set(docs)):
        raise ValueError(f"no document text for doc_id(s): {', '.join(map(repr, missing))}")

    metered = MeteredLLM(llm)
    indexes: Dict[str, DocIndex] = {}
    rows: List[EvalRow] = []
    errors, total = 0, len(items) * len(systems)
    for item in items:
        idx = indexes.get(item.doc_id)
        if idx is None:
            idx = indexes[item.doc_id] = DocIndex(docs[item.doc_id], cfg, embedder)
        for system in systems:
            (p0, c0), t0 = metered.snapshot(), time.perf_counter()
            error = ""
            try:
                pred = _RUNNERS[system](metered, idx, item.question)
            except LLMError as e:
                errors += 1
                if errors > cfg.max_errors:
                    raise LLMError(f"aborting after {errors} backend errors; last: {e}") from e
                pred, error = Prediction(ERROR_ANSWER), str(e)
            latency = time.perf_counter() - t0
            p1, c1 = metered.snapshot()
            score = score_question(
                item, pred.answer, pred.cited_chunks, pred.citations, tool_calls=pred.tool_calls,
                prompt_tokens=p1 - p0, completion_tokens=c1 - c0, latency_s=latency,
                prices=cfg.prices, system=system,
            )
            rows.append(EvalRow(score, item.doc_id, error))
            if progress:
                progress(len(rows), total, rows[-1])
    return EvalRun(rows, tokens_estimated=metered.estimated)


# -- reporting -----------------------------------------------------------------------

CSV_FIELDS = ("system", "doc_id", "id", "type", "gold", "prediction", "exact", "fuzzy", "f1",
              "evidence_recall", "citation_precision", "citations", "tool_calls", "prompt_tokens",
              "completion_tokens", "total_tokens", "latency_s", "cost_usd", "error")


def write_csv(run: EvalRun, path: str | Path) -> None:
    """One row per (system, question)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in run.rows:
            w.writerow({k: row.to_dict()[k] for k in CSV_FIELDS})


_PCT = {"exact", "fuzzy", "f1", "evidence_recall", "citation_precision"}


def format_comparison(run: EvalRun, *, n_boot: int = 2000, seed: Optional[int] = 0) -> str:
    """Markdown: overall metrics per system, then fuzzy accuracy per question type (95% bootstrap CIs)."""
    metrics = ["exact", "fuzzy", "evidence_recall", "tool_calls", "total_tokens", "latency_s"]
    if any(s.cost_usd is not None for s in run.scores()):
        metrics.append("cost_usd")
    summaries = {s: summarize(run.scores(s), n_boot=n_boot, seed=seed) for s in run.systems}

    def cell(est: Any, m: str) -> str:
        if m == "cost_usd":
            return est.fmt(5)
        return est.fmt(1 if m in _PCT else 2, pct=m in _PCT)

    lines = ["| system | n | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 2)]
    for system, summ in summaries.items():
        est = summ["all"]
        lines.append(f"| {system} | {est['exact'].n} | " + " | ".join(cell(est[m], m) for m in metrics) + " |")

    types = [t for t in QUESTION_TYPES if any(t in summ for summ in summaries.values())]
    if types:
        lines += ["", "Fuzzy accuracy (%) by question type:", "",
                  "| system | " + " | ".join(types) + " |", "|" + "---|" * (len(types) + 1)]
        for system, summ in summaries.items():
            lines.append(f"| {system} | " + " | ".join(
                cell(summ[t]["fuzzy"], "fuzzy") if t in summ else "n/a" for t in types) + " |")

    notes = ["Cells are mean [95% bootstrap CI]; evidence_recall is retrieval recall for single-pass systems."]
    if run.tokens_estimated:
        notes.append("Token counts are estimated (~4 chars/token); the backend reported no usage.")
    if n_err := sum(1 for r in run.rows if r.error):
        notes.append(f"{n_err} backend error(s) counted as wrong answers.")
    return "\n".join(lines + [""] + [f"_{n}_" for n in notes])


# -- command line --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_agent_eval.py",
        description="Compare truncation, plain RAG, DolphinMind retrieval, the ReAct agent and (opt-in) the "
                    "multi-agent on a JSONL "
                    "benchmark. Writes a per-question CSV and a markdown summary with bootstrap CIs.",
    )
    p.add_argument("datasets", nargs="+", help="JSONL datasets; each item's document is <dir>/<doc_id>.txt")
    p.add_argument("--doc", help="use this text file for every item instead of <dir>/<doc_id>.txt")
    p.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                   help=f"comma-separated subset of {','.join(SYSTEMS)} (default: {','.join(DEFAULT_SYSTEMS)})")
    p.add_argument("--backend", choices=("openai", "hf"), default="openai",
                   help="openai: any OpenAI-compatible endpoint, key from DOLPHIN_API_KEY; hf: local 4-bit Qwen")
    p.add_argument("--model", help="model name (openai: DOLPHIN_MODEL or gpt-4o-mini)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: DOLPHIN_BASE_URL or OpenAI)")
    p.add_argument("--embedder", choices=("minilm", "hashing"), default="minilm",
                   help="minilm: all-MiniLM-L6-v2 (downloads once); hashing: offline bag-of-words")
    p.add_argument("--types", help="only these question types (comma-separated)")
    p.add_argument("--limit", type=int, help="only the first N questions (after --types)")
    p.add_argument("--chunk-size", type=int, default=256, help="words per chunk (default: 256)")
    p.add_argument("--overlap", type=int, default=64, help="DolphinMind/agent chunk overlap (default: 64)")
    p.add_argument("--top-k", type=int, default=5, help="retrieved chunks (default: 5)")
    p.add_argument("--context-words", type=int, help="truncation budget (default: top-k * chunk-size)")
    p.add_argument("--max-steps", type=int, default=6, help="agent tool-call budget (default: 6)")
    p.add_argument("--prices", help="USD per 1M tokens as IN,OUT (e.g. 0.15,0.60) to report cost")
    p.add_argument("--max-errors", type=int, default=5, help="abort after this many backend errors (default: 5)")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default: 2000)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    p.add_argument("--out", default="results/agent_eval.csv",
                   help="CSV path; the summary goes next to it as .md (default: results/agent_eval.csv)")
    p.add_argument("--quiet", action="store_true", help="no per-question progress on stderr")
    return p


def load_datasets(paths: Sequence[str], doc_override: Optional[str] = None) -> Tuple[List[EvalItem], Dict[str, str]]:
    """Load items from JSONL files and the document text each ``doc_id`` refers to."""
    items: List[EvalItem] = []
    docs: Dict[str, str] = {}
    override = Path(doc_override).read_text(encoding="utf-8") if doc_override else None
    seen: set[Tuple[str, str]] = set()
    for path in map(Path, paths):
        for item in load_jsonl(path):
            if not item.doc_id:
                item.doc_id = path.stem
            if (item.doc_id, item.id) in seen:
                raise ValueError(f"{path}: duplicate id {item.id!r} for doc {item.doc_id!r}")
            seen.add((item.doc_id, item.id))
            if item.doc_id not in docs:
                docs[item.doc_id] = override if override is not None else \
                    (path.parent / f"{item.doc_id}.txt").read_text(encoding="utf-8")
            items.append(item)
    return items, docs


def _parse_prices(text: str) -> Tuple[float, float]:
    try:
        a, b = (float(x) for x in text.split(","))
    except ValueError:
        raise ValueError("--prices must be IN,OUT, e.g. 0.15,0.60") from None
    return a, b


def _build_llm(args: argparse.Namespace) -> Any:
    if args.backend == "openai":
        from ..agent.llm import OpenAICompatLLM

        return OpenAICompatLLM(model=args.model, base_url=args.base_url)
    from ..agent.llm import HFChatLLM
    from ..models import QWEN, load_qwen

    model, tokenizer = load_qwen(args.model or QWEN)
    return HFChatLLM(model, tokenizer)


def _build_embedder(name: str) -> Any:
    if name == "hashing":
        return HashingEmbedder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def main(argv: Optional[List[str]] = None, *, llm: Optional[LLM] = None, embedder: Any = None,
         stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    """Run the comparison. ``llm``/``embedder`` override the CLI choices (tests, notebooks).

    Returns 0 on success, 1 on a backend failure, 2 on bad usage or input.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems or (bad := [s for s in systems if s not in SYSTEMS]):
        parser.error(f"--systems must be a subset of {','.join(SYSTEMS)}")
    try:
        cfg = EvalConfig(args.chunk_size, args.overlap, args.top_k, args.context_words, args.max_steps,
                         _parse_prices(args.prices) if args.prices else None, args.max_errors)
        items, docs = load_datasets(args.datasets, args.doc)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=stderr)
        return 2
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        if bad := wanted - set(QUESTION_TYPES):
            parser.error(f"unknown type(s): {', '.join(sorted(bad))}")
        items = [it for it in items if it.type in wanted]
    if args.limit is not None:
        items = items[: max(args.limit, 0)]
    if not items:
        print("error: no questions to evaluate", file=stderr)
        return 2
    if problems := check_evidence(items, docs):
        print(f"warning: {len(problems)} evidence problem(s), e.g. {problems[0]}", file=stderr)

    owns_llm = llm is None
    try:
        llm = llm or _build_llm(args)
        if embedder is None and (_NEEDS_EMBEDDER.intersection(systems) or _AGENTS.intersection(systems)):
            embedder = _build_embedder(args.embedder)
    except (OSError, ValueError, ImportError) as e:
        print(f"error: {e}", file=stderr)
        return 2

    def progress(i: int, n: int, row: EvalRow) -> None:
        s = row.score
        mark = "ERR" if row.error else ("ok " if s.fuzzy else "-- ")
        print(f"[{i}/{n}] {mark} {s.system:<11} {s.id}: {s.prediction[:60]!r}", file=stderr)

    try:
        run = run_eval(items, docs, llm, systems, cfg, embedder, None if args.quiet else progress)
    except LLMError as e:
        print(f"error: {e}", file=stderr)
        return 1
    finally:
        close = getattr(llm, "close", None)
        if owns_llm and callable(close):
            close()

    out = Path(args.out)
    write_csv(run, out)
    summary = format_comparison(run, n_boot=args.n_boot, seed=args.seed)
    out.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print(summary, file=stdout)
    print(f"\nwrote {out} and {out.with_suffix('.md')}", file=stderr)
    return 0
