"""Parallel multi-agent workers and per-query token/cost budgets (no GPU, no network)."""

import asyncio
import json
import threading
import time

import httpx
import pytest

from dolphinmind.agent import (Budget, BudgetedLLM, BudgetExceeded, Document, LexicalVerifier, MultiAgent,
                               OpenAICompatLLM, ScriptedLLM, Toolbox)
from dolphinmind.agent.budget import estimate_prompt_tokens, metered_chat
from dolphinmind.agent.multi import PLANNER_PROMPT, SYNTH_PROMPT
from dolphinmind.eval.runner import MeteredLLM
from dolphinmind.evaluation import insert_needle, make_haystack

FACT_A = "The lighthouse of Varrow holds 47 lamps."
FACT_B = "The lighthouse of Qell holds 15 lamps."
QUESTION = "How many lamps do the lighthouses of Varrow and Qell hold together?"
SUB_A, SUB_B, SUB_SUM = ("How many lamps does the Varrow lighthouse hold?",
                         "How many lamps does the Qell lighthouse hold?",
                         "What is the sum of those two numbers?")


def act(**kw):
    return json.dumps(kw)


@pytest.fixture
def doc():
    text = insert_needle(make_haystack(4_000), FACT_A, 25)
    return Document(insert_needle(text, FACT_B, 75))


def chunk_with(doc, phrase):
    return next(i for i, c in enumerate(doc.chunks) if phrase in c)


class RoutedLLM:
    """Thread-safe fake: replies come from a per-route queue chosen by the conversation.

    Routes: "planner", "synth", or a sub-question (matched in the worker's first user turn).
    Each call sleeps briefly and the fake records the peak number of calls in flight.
    """

    def __init__(self, routes, delay=0.05, usage=None):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.delay, self.usage = delay, usage
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.log = []  # (route, user prompt) in call order

    def _route(self, messages):
        system, user = messages[0]["content"], messages[1]["content"]
        if system.startswith(PLANNER_PROMPT[:30]):
            return "planner"
        if system == SYNTH_PROMPT:
            return "synth"
        return next(k for k in self.routes if user.startswith(f"Question: {k}"))

    def chat(self, messages):
        route = self._route(messages)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.log.append((route, messages[1]["content"]))
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
            return self.routes[route].pop(0)


class UsageLLM(RoutedLLM):
    """Routed fake that reports a fixed per-call usage, like a real API backend."""

    def chat_with_usage(self, messages):
        return self.chat(messages), self.usage


def lamp_routes(doc, depends_on):
    ca, cb = chunk_with(doc, "Varrow holds"), chunk_with(doc, "Qell holds")
    return {
        "planner": [act(sub_questions=[SUB_A, SUB_B, SUB_SUM], depends_on=depends_on)],
        SUB_A: [act(tool="grep", args={"pattern": "Varrow holds"}), act(answer="47", citations=[ca])],
        SUB_B: [act(tool="grep", args={"pattern": "Qell holds"}), act(answer="15", citations=[cb])],
        SUB_SUM: [act(tool="calculator", args={"expression": "47+15"}), act(answer="not found", citations=[])],
        "synth": [act(answer="62", citations=[ca, cb])],
    }


# -- planning graph ----------------------------------------------------------------------

def test_plan_graph_parses_dependencies(doc):
    llm = ScriptedLLM([act(sub_questions=["A?", "B?", "C?"], depends_on=[[], [], [0, 1]])])
    assert MultiAgent(llm, Toolbox(doc)).plan_graph("q") == (["A?", "B?", "C?"], [[], [], [0, 1]])


@pytest.mark.parametrize("depends_on, expected", [
    (None, [[], [0], [0, 1]]),                    # missing -> sequential
    ([[], []], [[], [0], [0, 1]]),                # wrong length -> sequential
    ([[], "0", [0]], [[], [0], [0]]),             # malformed entry -> all earlier
    ([[2], [1], [0, 0, 5]], [[], [], [0]]),       # forward/self/unknown references dropped
])
def test_plan_graph_falls_back_to_sequential(doc, depends_on, expected):
    plan = {"sub_questions": ["A?", "B?", "C?"]}
    if depends_on is not None:
        plan["depends_on"] = depends_on
    assert MultiAgent(ScriptedLLM([json.dumps(plan)]), Toolbox(doc)).plan_graph("q")[1] == expected


def test_plan_graph_remaps_duplicates_and_cap(doc):
    llm = ScriptedLLM([act(sub_questions=["A?", "a?", "B?", "C?"], depends_on=[[], [], [1], [0, 2]])])
    subs, deps = MultiAgent(llm, Toolbox(doc), max_sub_questions=2).plan_graph("q")
    assert subs == ["A?", "B?"] and deps == [[], [0]]  # dependency on the duplicate -> its first copy
    assert MultiAgent(ScriptedLLM(["no json"]), Toolbox(doc)).plan_graph("q") == (["q"], [[]])
    with pytest.raises(ValueError):
        MultiAgent(llm, Toolbox(doc), max_workers=0)


# -- parallel execution -------------------------------------------------------------------

def test_independent_workers_run_in_parallel_and_dependants_wait(doc):
    llm = RoutedLLM(lamp_routes(doc, [[], [], [0, 1]]))
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier(), max_workers=4).run(QUESTION)
    assert llm.peak == 2  # the two lookups overlapped
    assert res.answer == "62" and res.stopped == "answered"
    assert res.depends_on == [[], [], [0, 1]]
    assert [f.answer for f in res.findings] == ["47", "15", "not found"]
    # the dependent worker started only after both lookups finished, and saw both findings
    order = [r for r, _ in llm.log]
    first_sum = order.index(SUB_SUM)
    assert order[:first_sum].count(SUB_A) == 2 and order[:first_sum].count(SUB_B) == 2
    sum_prompt = next(p for r, p in llm.log if r == SUB_SUM)
    assert "47" in sum_prompt and "15" in sum_prompt
    # the trace is in plan order regardless of finish order
    verifier_steps = [s.thought for s in res.steps if s.thought.startswith("[verifier]")]
    assert [SUB_A in verifier_steps[0], SUB_B in verifier_steps[1], SUB_SUM in verifier_steps[2]] == [True] * 3
    assert all(not r for r in llm.routes.values())


def test_max_workers_one_is_sequential(doc):
    llm = RoutedLLM(lamp_routes(doc, [[], [], [0, 1]]), delay=0.01)
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier(), max_workers=1).run(QUESTION)
    assert llm.peak == 1 and res.answer == "62"


def test_without_depends_on_workers_stay_sequential(doc):
    routes = lamp_routes(doc, None)
    routes["planner"] = [act(sub_questions=[SUB_A, SUB_B, SUB_SUM])]
    llm = RoutedLLM(routes, delay=0.01)
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier()).run(QUESTION)
    assert llm.peak == 1 and res.depends_on == [[], [0], [0, 1]] and res.answer == "62"


def test_arun_inside_an_event_loop(doc):
    llm = RoutedLLM(lamp_routes(doc, [[], [], [0, 1]]), delay=0.0)
    agent = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier())

    async def main():
        return await agent.arun(QUESTION)

    assert asyncio.run(main()).answer == "62"


# -- usage and budgets -----------------------------------------------------------------

def test_result_reports_usage_and_cost(doc):
    llm = UsageLLM(lamp_routes(doc, [[], [], [0, 1]]), delay=0.0, usage=(100, 10))
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier(), budget=Budget(prices=(1.0, 2.0))).run(QUESTION)
    calls = 1 + 2 + 2 + 2 + 1  # planner, 3 workers x 2 turns, synthesis
    assert (res.prompt_tokens, res.completion_tokens) == (100 * calls, 10 * calls)
    assert res.total_tokens == 110 * calls and not res.tokens_estimated
    assert res.cost_usd == pytest.approx((100 * calls * 1.0 + 10 * calls * 2.0) / 1e6)


def test_usage_is_estimated_without_backend_usage(doc):
    llm = RoutedLLM(lamp_routes(doc, [[], [], [0, 1]]), delay=0.0)
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier()).run(QUESTION)
    assert res.tokens_estimated and res.total_tokens > 0 and res.cost_usd is None


def test_token_budget_stops_research_and_falls_back(doc):
    # planner + 2 turns of SUB_A spend the 30k-token budget; every later call is refused
    routes = lamp_routes(doc, [[], [0], [0, 1]])
    llm = UsageLLM(routes, delay=0.0, usage=(10_000, 0))
    agent = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier(), max_workers=1,
                       budget=Budget(max_tokens=30_000))
    res = agent.run(QUESTION)
    assert res.stopped == "budget"
    assert res.findings[0].answer == "47" and res.findings[0].verified
    assert all(f.flagged and "budget exhausted" in f.reason for f in res.findings[1:])
    assert res.answer == "47"  # synthesis refused too -> best verified finding
    assert res.total_tokens == 30_000  # nothing beyond the budget was spent
    assert any(s.thought.startswith("[budget]") for s in res.steps)
    assert routes["synth"]  # the synthesis call never reached the model
    # the budget is per query: a second query starts unspent
    assert agent.budget.total_tokens == 0


def test_cost_budget_on_planner(doc):
    llm = UsageLLM({"planner": []}, delay=0.0, usage=(10, 0))
    budget = Budget(max_cost_usd=1e-9, prices=(1.0, 1.0))
    res = MultiAgent(llm, Toolbox(doc), budget=budget).run(QUESTION)
    assert res.stopped == "budget" and res.answer == "not found" and res.plan == []
    assert not llm.log  # refused before calling the model


def test_budget_validation_and_accounting():
    with pytest.raises(ValueError):
        Budget(max_tokens=0)
    with pytest.raises(ValueError):
        Budget(max_cost_usd=1.0)  # needs prices
    b = Budget(max_tokens=100, prices=(2.0, 4.0))
    b.check(100)
    with pytest.raises(BudgetExceeded):
        b.check(101)
    b.charge(60, 40, estimated=True)
    assert b.total_tokens == 100 and b.exhausted and b.estimated
    assert b.cost_usd == pytest.approx((60 * 2 + 40 * 4) / 1e6)
    with pytest.raises(BudgetExceeded, match="token budget 100"):
        b.check()
    f = b.fresh()
    assert f.total_tokens == 0 and f.max_tokens == 100 and not f.estimated
    b.reset()
    assert b.calls == 0 and not b.exhausted


def test_budget_is_thread_safe():
    b = Budget()

    def work():
        for _ in range(2000):
            b.charge(1, 1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (b.prompt_tokens, b.completion_tokens, b.calls) == (16000, 16000, 16000)


def test_budgeted_llm_refuses_before_calling():
    inner = ScriptedLLM(["abcd"])
    b = Budget(max_tokens=5)
    llm = BudgetedLLM(inner, b)
    msgs = [{"role": "user", "content": "x" * 12}]  # ~3 tokens
    assert llm.chat(msgs) == "abcd" and b.total_tokens == 4 and b.estimated
    with pytest.raises(BudgetExceeded):
        llm.chat(msgs)  # 4 spent + 3 estimated > 5
    assert len(inner.calls) == 1
    assert estimate_prompt_tokens(msgs) == 3


# -- exact per-call usage under concurrency --------------------------------------------

def _api(prompt=7, completion=3):
    def handler(request):
        time.sleep(0.005)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                          "usage": {"prompt_tokens": prompt, "completion_tokens": completion}})

    return OpenAICompatLLM(model="m", api_key="", transport=httpx.MockTransport(handler))


def test_openai_chat_with_usage_and_concurrent_metering():
    api = _api()
    assert api.chat_with_usage([{"role": "user", "content": "hi"}]) == ("ok", (7, 3))
    assert metered_chat(api, [{"role": "user", "content": "hi"}]) == ("ok", (7, 3), False)

    metered = MeteredLLM(_api())
    assert not metered.estimated
    threads = [threading.Thread(target=lambda: [metered.chat([{"role": "user", "content": "q"}])
                                                for _ in range(10)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (metered.calls, metered.prompt_tokens, metered.completion_tokens) == (80, 560, 240)
    assert metered.llm.usage.total_tokens == 800
