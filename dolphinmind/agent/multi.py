"""Multi-agent research: planner -> workers -> verifier -> synthesis.

The :class:`MultiAgent` answers a question in four stages:

1. **Planner** decomposes the question into at most ``max_sub_questions`` ordered
   sub-questions (a simple question stays a single sub-question).
2. **Workers** research each sub-question with a :class:`~dolphinmind.agent.react.ReActAgent`
   over a shared :class:`~dolphinmind.agent.tools.Toolbox`. Each worker sees the verified
   findings of earlier sub-questions, so multi-hop chains can use a bridge entity.
3. **Verifier** checks every finding against the chunks it cites. An unsupported finding
   is researched again (up to ``max_retries`` times, with the verifier's reason as
   feedback) and is otherwise replaced by ``"not found"`` and flagged.
4. **Synthesis** combines the verified findings into the final answer; it may only cite
   chunks that back a verified finding.

Workers run sequentially here; parallel execution is a separate step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

from .llm import LLM, Message
from .react import AgentResult, ReActAgent, Step, parse_action
from .tools import Toolbox

NOT_FOUND = "not found"

PLANNER_PROMPT = """You plan research over a long document that only tools can read.
Break the user's question into at most {max_sub} short sub-questions, in the order they
must be answered. Each sub-question must be answerable by looking something up in the
document; a later sub-question may refer to the answer of an earlier one as "that person",
"that place", etc. A question that needs a single lookup stays one sub-question.

Reply with exactly one JSON object and nothing else:
  {{"sub_questions": ["...", "..."]}}"""

VERIFIER_PROMPT = """You check research findings against source text.
Decide whether the excerpts directly state the given answer to the question
(paraphrase is fine; outside knowledge is not).

Reply with exactly one JSON object and nothing else:
  {"supported": true or false, "reason": "<one sentence>"}"""

SYNTH_PROMPT = """You write the final answer to a question from verified research findings.
Use only the findings below; do any arithmetic the question needs. If the findings do not
answer the question, answer "not found" with no citations.

Reply with exactly one JSON object and nothing else:
  {"answer": "<short answer>", "citations": [<chunk ids from the findings>]}"""


def _is_not_found(answer: str) -> bool:
    from ..eval.metrics import is_abstention  # lazy: dolphinmind.eval imports the agent package

    return is_abstention(answer)


def _int_ids(values: object) -> List[int]:
    if not isinstance(values, list):
        return []
    return [int(v) for v in values if str(v).lstrip("-").isdigit()]


# -- verification --------------------------------------------------------------------

@dataclass
class Verdict:
    supported: bool
    reason: str


class Verifier(Protocol):
    def check(self, question: str, answer: str, citations: Sequence[int], toolbox: Toolbox) -> Verdict: ...


def _precheck(answer: str, citations: Sequence[int]) -> Optional[Verdict]:
    """Verdicts that need no model: abstentions pass, uncited claims fail."""
    if _is_not_found(answer):
        return Verdict(True, "worker abstained")
    if not citations:
        return Verdict(False, "the answer cites no chunks")
    return None


class LLMVerifier:
    """Ask a chat model whether the cited chunks state the answer.

    An unparseable verifier reply counts as *unsupported*: a claim is only accepted
    when the check positively succeeds.
    """

    def __init__(self, llm: LLM, max_chars: int = 3000):
        self.llm, self.max_chars = llm, max_chars
        self.calls = 0

    def check(self, question: str, answer: str, citations: Sequence[int], toolbox: Toolbox) -> Verdict:
        if (v := _precheck(answer, citations)) is not None:
            return v
        excerpts = "\n\n".join(f"[chunk {i}] {toolbox.doc.chunks[i][: self.max_chars]}" for i in citations)
        messages: List[Message] = [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\n\nExcerpts:\n{excerpts}"},
        ]
        self.calls += 1
        try:
            action = parse_action(self.llm.chat(messages))
        except ValueError as e:
            return Verdict(False, f"verifier reply unparseable ({e})")
        supported = action.get("supported")
        if isinstance(supported, str):
            supported = supported.strip().lower() in {"true", "yes"}
        return Verdict(bool(supported), str(action.get("reason", "")).strip())


class LexicalVerifier:
    """Model-free check: every content token of the answer occurs in the cited chunks.

    Cheap and deterministic (useful offline and as an ablation), but it cannot accept
    paraphrases or derived values such as sums.
    """

    def check(self, question: str, answer: str, citations: Sequence[int], toolbox: Toolbox) -> Verdict:
        from ..eval.metrics import normalize_answer

        if (v := _precheck(answer, citations)) is not None:
            return v
        cited = set(normalize_answer(" ".join(toolbox.doc.chunks[i] for i in citations)).split())
        missing = [t for t in normalize_answer(answer).split() if t not in cited]
        if missing:
            return Verdict(False, f"not in cited chunks: {', '.join(missing[:5])}")
        return Verdict(True, "all answer tokens appear in the cited chunks")


# -- results -------------------------------------------------------------------------

@dataclass
class Finding:
    """A worker's answer to one sub-question after verification."""

    question: str
    answer: str
    citations: List[int]
    verified: bool
    reason: str = ""
    attempts: int = 1
    flagged: bool = False  # verification failed on every attempt; answer replaced by "not found"

    @property
    def found(self) -> bool:
        return not _is_not_found(self.answer)


@dataclass
class MultiAgentResult:
    """Same surface as :class:`AgentResult` (answer, citations, steps, tool_calls, stopped) plus the plan."""

    answer: str
    citations: List[int]
    plan: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    stopped: str = "answered"  # answered | not_found | synth_error

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool)


# -- orchestration -------------------------------------------------------------------

class MultiAgent:
    """Planner -> workers -> verifier -> synthesis over one document.

    ``llm`` drives every role unless ``planner_llm`` / ``verifier`` override it, so a
    small model can do the research while a stronger one plans and checks.
    """

    def __init__(
        self,
        llm: LLM,
        toolbox: Toolbox,
        *,
        planner_llm: Optional[LLM] = None,
        verifier: Optional[Verifier] = None,
        max_sub_questions: int = 4,
        max_steps: int = 6,
        max_retries: int = 1,
    ):
        if max_sub_questions < 1:
            raise ValueError("max_sub_questions must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm, self.toolbox = llm, toolbox
        self.planner_llm = planner_llm or llm
        self.verifier: Verifier = verifier if verifier is not None else LLMVerifier(llm)
        self.max_sub_questions, self.max_steps, self.max_retries = max_sub_questions, max_steps, max_retries

    # -- stages --------------------------------------------------------------
    def plan(self, question: str) -> List[str]:
        """Sub-questions in answer order; falls back to ``[question]`` on a bad plan."""
        messages: List[Message] = [
            {"role": "system", "content": PLANNER_PROMPT.format(max_sub=self.max_sub_questions)},
            {"role": "user", "content": f"Question: {question}"},
        ]
        try:
            raw = parse_action(self.planner_llm.chat(messages)).get("sub_questions")
        except ValueError:
            return [question]
        subs: List[str] = []
        for q in raw if isinstance(raw, list) else []:
            q = " ".join(str(q).split())
            if q and q.lower() not in {s.lower() for s in subs}:
                subs.append(q)
        return subs[: self.max_sub_questions] or [question]

    def _worker_prompt(self, sub_q: str, original: str, known: Sequence[Finding], feedback: str) -> str:
        parts = [sub_q]
        if sub_q != original:
            parts.append(f"(This is one step towards answering: {original})")
        found = [f for f in known if f.found]
        if found:
            parts.append("Findings so far:\n" + "\n".join(
                f"- {f.question} -> {f.answer} (chunks {', '.join(map(str, f.citations))})" for f in found))
        if feedback:
            parts.append(feedback)
        return "\n\n".join(parts)

    def research(self, sub_q: str, original: str, known: Sequence[Finding], steps: List[Step]) -> Finding:
        """Run a worker on one sub-question, verify it, and retry with feedback if unsupported."""
        feedback, last = "", None
        for attempt in range(1, self.max_retries + 2):
            res: AgentResult = ReActAgent(self.llm, self.toolbox, max_steps=self.max_steps).run(
                self._worker_prompt(sub_q, original, known, feedback))
            steps.extend(res.steps)
            t0 = time.time()
            verdict = self.verifier.check(sub_q, res.answer, res.citations, self.toolbox)
            steps.append(Step(f"[verifier] {sub_q} -> {res.answer!r}: "
                              f"{'supported' if verdict.supported else 'UNSUPPORTED'} ({verdict.reason})",
                              None, {}, "", time.time() - t0))
            if verdict.supported:
                cites = [] if _is_not_found(res.answer) else list(res.citations)
                return Finding(sub_q, res.answer, cites, True, verdict.reason, attempt)
            last = (res.answer, verdict.reason)
            feedback = (f"A previous attempt answered {res.answer!r} but the verifier rejected it: "
                        f"{verdict.reason}. Search again and cite the chunks that state the answer, "
                        f'or answer "{NOT_FOUND}".')
        assert last is not None
        return Finding(sub_q, NOT_FOUND, [], False, f"rejected {last[0]!r}: {last[1]}",
                       self.max_retries + 1, flagged=True)

    def synthesize(self, question: str, findings: Sequence[Finding]) -> tuple[str, List[int], bool]:
        """Final ``(answer, citations, ok)``; citations limited to chunks backing verified findings."""
        allowed = [c for f in findings if f.found for c in f.citations]
        lines = [f"{i}. {f.question} -> {f.answer}"
                 + (f" (chunks {', '.join(map(str, f.citations))})" if f.citations else "")
                 for i, f in enumerate(findings, 1)]
        messages: List[Message] = [
            {"role": "system", "content": SYNTH_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nFindings:\n" + "\n".join(lines)},
        ]
        try:
            action = parse_action(self.llm.chat(messages))
        except ValueError:
            best = next((f for f in reversed(findings) if f.found), None)
            return (best.answer, best.citations, False) if best else (NOT_FOUND, [], False)
        answer = str(action.get("answer", NOT_FOUND)).strip() or NOT_FOUND
        if _is_not_found(answer):
            return answer, [], True
        cites = [c for c in dict.fromkeys(_int_ids(action.get("citations"))) if c in allowed]
        return answer, cites or list(dict.fromkeys(allowed)), True

    # -- entry point ---------------------------------------------------------
    def run(self, question: str) -> MultiAgentResult:
        steps: List[Step] = []
        t0 = time.time()
        plan = self.plan(question)
        steps.append(Step(f"[planner] {len(plan)} sub-question(s)", None, {}, json.dumps(plan),
                          time.time() - t0))

        findings: List[Finding] = []
        for sub_q in plan:
            findings.append(self.research(sub_q, question, findings, steps))

        if not any(f.found for f in findings):
            return MultiAgentResult(NOT_FOUND, [], plan, findings, steps, stopped="not_found")
        if len(findings) == 1:  # nothing to combine: the verified finding is the answer
            f = findings[0]
            return MultiAgentResult(f.answer, list(f.citations), plan, findings, steps)

        t0 = time.time()
        answer, cites, ok = self.synthesize(question, findings)
        steps.append(Step("[synthesis]" if ok else "[synthesis] unparseable reply; used last finding",
                          None, {}, answer, time.time() - t0))
        stopped = "not_found" if _is_not_found(answer) else ("answered" if ok else "synth_error")
        return MultiAgentResult(answer, cites, plan, findings, steps, stopped=stopped)
