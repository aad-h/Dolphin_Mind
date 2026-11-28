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

Workers run concurrently with :mod:`asyncio`. The planner may say which earlier
sub-questions each one needs (``depends_on``); a sub-question starts as soon as its
dependencies are verified, so independent lookups run side by side while a multi-hop
chain still waits for its bridge entity. Without ``depends_on`` every sub-question
depends on all earlier ones (the safe, sequential order). Blocking ``LLM.chat`` calls run
in worker threads, capped at ``max_workers`` at a time.

With a :class:`~dolphinmind.agent.tracing.Tracer`, each query is one ``multi.run`` trace
with ``multi.plan``, ``multi.research`` (per sub-question and attempt), ``multi.verify``
and ``multi.synthesize`` spans around the workers' ``agent.run`` / ``llm.chat`` /
``tool.*`` spans.

Every LLM call of a query is charged to a per-query :class:`~dolphinmind.agent.budget.Budget`.
With ``max_tokens`` / ``max_cost_usd`` set, a call the budget cannot afford raises
:class:`~dolphinmind.agent.budget.BudgetExceeded`: the affected sub-questions are flagged
``"not found"``, synthesis falls back to the best verified finding, and the result is
marked ``stopped="budget"``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .budget import Budget, BudgetedLLM, BudgetExceeded
from .llm import LLM, Message
from .react import AgentResult, ReActAgent, Step, parse_action
from .tools import Toolbox
from .tracing import Tracer, maybe_span, traced

NOT_FOUND = "not found"

PLANNER_PROMPT = """You plan research over a long document that only tools can read.
Break the user's question into at most {max_sub} short sub-questions, in the order they
must be answered. Each sub-question must be answerable by looking something up in the
document; a later sub-question may refer to the answer of an earlier one as "that person",
"that place", etc. A question that needs a single lookup stays one sub-question.
For each sub-question, list the indices (0-based) of the earlier sub-questions whose
answers it needs; independent sub-questions get [] and are researched in parallel.

Reply with exactly one JSON object and nothing else:
  {{"sub_questions": ["...", "..."], "depends_on": [[], [0]]}}"""

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
    """Same surface as :class:`AgentResult` (answer, citations, steps, tool_calls, stopped) plus
    the plan, its dependency graph and the query's token / cost usage."""

    answer: str
    citations: List[int]
    plan: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    stopped: str = "answered"  # answered | not_found | synth_error | budget
    depends_on: List[List[int]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Optional[float] = None  # set when the agent was given prices
    tokens_estimated: bool = False  # backend reported no usage: ~4 chars/token estimate

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool)


# -- orchestration -------------------------------------------------------------------

class MultiAgent:
    """Planner -> parallel workers -> verifier -> synthesis over one document.

    ``llm`` drives every role unless ``planner_llm`` / ``verifier`` override it, so a
    small model can do the research while a stronger one plans and checks. Workers share
    ``toolbox`` (a worker may cite chunks another worker showed, as its findings are
    passed along). ``budget`` sets per-query limits; a fresh copy is used for every query.
    Calls made by a custom ``verifier``'s own model are not charged to the budget.
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
        max_workers: int = 4,
        budget: Optional[Budget] = None,
        tracer: Optional[Tracer] = None,
    ):
        if max_sub_questions < 1:
            raise ValueError("max_sub_questions must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.llm, self.toolbox = llm, toolbox
        self.planner_llm = planner_llm or llm
        self.verifier: Optional[Verifier] = verifier  # None -> LLMVerifier over the (metered) llm
        self.max_sub_questions, self.max_steps, self.max_retries = max_sub_questions, max_steps, max_retries
        self.max_workers = max_workers
        self.budget = budget or Budget()
        self.tracer = tracer

    # -- stages --------------------------------------------------------------
    def plan(self, question: str) -> List[str]:
        """Sub-questions in answer order; falls back to ``[question]`` on a bad plan."""
        return self.plan_graph(question)[0]

    def plan_graph(self, question: str) -> Tuple[List[str], List[List[int]]]:
        """``(sub_questions, depends_on)``; ``depends_on[i]`` holds earlier indices only.

        A missing or malformed ``depends_on`` entry means "depends on every earlier
        sub-question", so a sloppy plan degrades to sequential research, never to a
        multi-hop step running before its bridge entity is known.
        """
        messages: List[Message] = [
            {"role": "system", "content": PLANNER_PROMPT.format(max_sub=self.max_sub_questions)},
            {"role": "user", "content": f"Question: {question}"},
        ]
        with maybe_span(self.tracer, "multi.plan", question=question) as span:
            subs, deps = self._parse_plan(question, messages)
            if span is not None:
                span.set(sub_questions=subs, depends_on=deps)
            return subs, deps

    def _parse_plan(self, question: str, messages: List[Message]) -> Tuple[List[str], List[List[int]]]:
        try:
            action = parse_action(self.planner_llm.chat(messages))
        except ValueError:
            return [question], [[]]
        raw, raw_deps = action.get("sub_questions"), action.get("depends_on")
        raw = raw if isinstance(raw, list) else []
        raw_deps = raw_deps if isinstance(raw_deps, list) and len(raw_deps) == len(raw) else None

        subs: List[str] = []
        deps: List[List[int]] = []
        where: Dict[int, int] = {}  # raw index -> kept index (duplicates map to the first copy)
        for j, q in enumerate(raw):
            q = " ".join(str(q).split())
            if not q:
                continue
            dup = next((k for k, s_ in enumerate(subs) if s_.lower() == q.lower()), None)
            if dup is not None:
                where[j] = dup
                continue
            if len(subs) == self.max_sub_questions:
                break
            entry = raw_deps[j] if raw_deps is not None else None
            if isinstance(entry, list) and all(str(d).lstrip("-").isdigit() for d in entry):
                mine = sorted({where[int(d)] for d in entry if int(d) in where})
            else:
                mine = list(range(len(subs)))
            where[j] = len(subs)
            subs.append(q)
            deps.append(mine)
        return (subs, deps) if subs else ([question], [[]])

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
        verifier = self.verifier if self.verifier is not None else LLMVerifier(self.llm)
        feedback, last = "", None
        for attempt in range(1, self.max_retries + 2):
            with maybe_span(self.tracer, "multi.research", sub_question=sub_q, attempt=attempt) as span:
                res: AgentResult = ReActAgent(self.llm, self.toolbox, max_steps=self.max_steps,
                                              tracer=self.tracer).run(
                    self._worker_prompt(sub_q, original, known, feedback))
                steps.extend(res.steps)
                t0 = time.time()
                with maybe_span(self.tracer, "multi.verify", answer=res.answer,
                                citations=list(res.citations)) as vspan:
                    verdict = verifier.check(sub_q, res.answer, res.citations, self.toolbox)
                    if vspan is not None:
                        vspan.set(supported=verdict.supported, reason=verdict.reason)
                if span is not None:
                    span.set(answer=res.answer, supported=verdict.supported)
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
        with maybe_span(self.tracer, "multi.synthesize", findings=len(findings)) as span:
            try:
                action = parse_action(self.llm.chat(messages))
            except ValueError:
                if span is not None:
                    span.set(parsed=False)
                return (*_fallback(findings), False)
        answer = str(action.get("answer", NOT_FOUND)).strip() or NOT_FOUND
        if _is_not_found(answer):
            return answer, [], True
        cites = [c for c in dict.fromkeys(_int_ids(action.get("citations"))) if c in allowed]
        return answer, cites or list(dict.fromkeys(allowed)), True

    # -- entry points --------------------------------------------------------
    def run(self, question: str) -> MultiAgentResult:
        """Answer ``question`` (blocking). Inside a running event loop use :meth:`arun`."""
        return asyncio.run(self.arun(question))

    async def arun(self, question: str) -> MultiAgentResult:
        """Answer ``question``, researching independent sub-questions concurrently."""
        budget = self.budget.fresh()
        agent = copy.copy(self)  # per-query view whose models charge this query's budget
        agent.llm = traced(BudgetedLLM(self.llm, budget), self.tracer)
        agent.planner_llm = (agent.llm if self.planner_llm is self.llm
                             else traced(BudgetedLLM(self.planner_llm, budget), self.tracer))
        with maybe_span(self.tracer, "multi.run", question=question) as span:
            res = await agent._execute(question)
            res.prompt_tokens, res.completion_tokens = budget.prompt_tokens, budget.completion_tokens
            res.cost_usd, res.tokens_estimated = budget.cost_usd, budget.estimated
            if span is not None:
                span.set(answer=res.answer, citations=list(res.citations), stopped=res.stopped,
                         sub_questions=len(res.plan), tool_calls=res.tool_calls,
                         prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                         cost_usd=res.cost_usd)
        return res

    async def _execute(self, question: str) -> MultiAgentResult:
        steps: List[Step] = []
        t0 = time.time()
        try:
            plan, deps = await asyncio.to_thread(self.plan_graph, question)
        except BudgetExceeded as e:
            steps.append(Step(f"[budget] planner: {e}", None, {}, "", time.time() - t0))
            return MultiAgentResult(NOT_FOUND, [], [], [], steps, stopped="budget")
        steps.append(Step(f"[planner] {len(plan)} sub-question(s)", None, {},
                          json.dumps({"sub_questions": plan, "depends_on": deps}), time.time() - t0))

        findings, budget_hit = await self._research_all(question, plan, deps, steps)

        if not any(f.found for f in findings):
            return MultiAgentResult(NOT_FOUND, [], plan, findings, steps,
                                    stopped="budget" if budget_hit else "not_found", depends_on=deps)
        if len(findings) == 1:  # nothing to combine: the verified finding is the answer
            f = findings[0]
            return MultiAgentResult(f.answer, list(f.citations), plan, findings, steps, depends_on=deps)

        t0 = time.time()
        try:
            answer, cites, ok = await asyncio.to_thread(self.synthesize, question, findings)
            note = "[synthesis]" if ok else "[synthesis] unparseable reply; used last finding"
        except BudgetExceeded as e:
            (answer, cites), ok, budget_hit = _fallback(findings), False, True
            note = f"[synthesis] budget exhausted ({e}); used last finding"
        steps.append(Step(note, None, {}, answer, time.time() - t0))
        if budget_hit:
            stopped = "budget"
        else:
            stopped = "not_found" if _is_not_found(answer) else ("answered" if ok else "synth_error")
        return MultiAgentResult(answer, cites, plan, findings, steps, stopped=stopped, depends_on=deps)

    async def _research_all(
        self, question: str, plan: List[str], deps: List[List[int]], steps: List[Step],
    ) -> Tuple[List[Finding], bool]:
        """Research every sub-question once its dependencies are done, ``max_workers`` at a time.

        Trace steps are appended in plan order, whatever order the workers finish in.
        """
        sem = asyncio.Semaphore(self.max_workers)
        per_step: List[List[Step]] = [[] for _ in plan]
        tasks: List["asyncio.Task[Finding]"] = []
        budget_hit = False

        async def solve(i: int) -> Finding:
            nonlocal budget_hit
            known = [await tasks[d] for d in deps[i]]
            async with sem:
                t0 = time.time()
                try:
                    return await asyncio.to_thread(self.research, plan[i], question, known, per_step[i])
                except BudgetExceeded as e:
                    budget_hit = True
                    per_step[i].append(Step(f"[budget] {plan[i]}: {e}", None, {}, "", time.time() - t0))
                    return Finding(plan[i], NOT_FOUND, [], False, f"budget exhausted: {e}", 0, flagged=True)

        for i in range(len(plan)):
            tasks.append(asyncio.create_task(solve(i)))
        findings = list(await asyncio.gather(*tasks))
        for s_ in per_step:
            steps.extend(s_)
        return findings, budget_hit


def _fallback(findings: Sequence[Finding]) -> Tuple[str, List[int]]:
    """The last verified finding, used when synthesis fails."""
    best = next((f for f in reversed(findings) if f.found), None)
    return (best.answer, list(best.citations)) if best else (NOT_FOUND, [])
