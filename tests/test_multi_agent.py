import json

import pytest

from dolphinmind.agent import (Document, LexicalVerifier, LLMVerifier, MultiAgent, MultiAgentResult, ScriptedLLM,
                               Toolbox)
from dolphinmind.eval import EvalItem, score_agent_result
from dolphinmind.evaluation import insert_needle, make_haystack

FACT_1 = "The chief archivist of Thisgate is Daendis Draemmork."
FACT_2 = "Daendis Draemmork was born in Tuholm."
QUESTION = "In which town was the chief archivist of Thisgate born?"


def act(**kw):
    return json.dumps(kw)


def ok(reason="stated"):
    return act(supported=True, reason=reason)


def bad(reason="not stated"):
    return act(supported=False, reason=reason)


@pytest.fixture
def doc():
    text = insert_needle(make_haystack(4_000), FACT_1, 20)
    return Document(insert_needle(text, FACT_2, 80))


def chunk_with(doc, phrase):
    return next(i for i, c in enumerate(doc.chunks) if phrase in c)


def test_multi_hop_plan_research_verify_synthesize(doc):
    c1, c2 = chunk_with(doc, "chief archivist of Thisgate is"), chunk_with(doc, "was born in Tuholm")
    llm = ScriptedLLM([
        act(sub_questions=["Who is the chief archivist of Thisgate?", "In which town was that person born?"]),
        act(tool="grep", args={"pattern": "chief archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        ok(),
        act(tool="grep", args={"pattern": "Draemmork was born"}),
        act(answer="Tuholm", citations=[c2]),
        ok(),
        act(answer="Tuholm", citations=[c2, 999]),  # 999 backs no finding -> dropped
    ])
    res = MultiAgent(llm, Toolbox(doc)).run(QUESTION)
    assert isinstance(res, MultiAgentResult)
    assert res.answer == "Tuholm" and res.citations == [c2]
    assert res.stopped == "answered" and res.tool_calls == 2
    assert [f.answer for f in res.findings] == ["Daendis Draemmork", "Tuholm"]
    assert all(f.verified and f.attempts == 1 for f in res.findings)
    assert not llm.responses  # every scripted reply consumed: no extra calls
    # the second worker is told what the first one found (the bridge entity)
    worker2_prompt = llm.calls[4][1]["content"]
    assert "Findings so far" in worker2_prompt and "Daendis Draemmork" in worker2_prompt
    # the verifier sees the cited chunk text; synthesis sees the findings
    assert FACT_1 in llm.calls[3][1]["content"]
    assert "Tuholm" in llm.calls[7][1]["content"]
    assert res.steps[0].thought.startswith("[planner]") and res.steps[-1].thought == "[synthesis]"


def test_single_sub_question_skips_synthesis_and_bad_plan_falls_back(doc):
    c1 = chunk_with(doc, "chief archivist of Thisgate is")
    llm = ScriptedLLM([
        "I would split this into steps.",  # unparseable plan -> [question]
        act(tool="grep", args={"pattern": "archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        ok(),
    ])
    q = "Who is the chief archivist of Thisgate?"
    res = MultiAgent(llm, Toolbox(doc)).run(q)
    assert res.plan == [q]
    assert res.answer == "Daendis Draemmork" and res.citations == [c1]
    assert not llm.responses
    # a single sub-question equal to the question gets no "one step towards" preamble
    assert "one step towards" not in llm.calls[1][1]["content"]


def test_rejected_finding_is_retried_with_feedback(doc):
    c1 = chunk_with(doc, "chief archivist of Thisgate is")
    llm = ScriptedLLM([
        act(sub_questions=["Who is the chief archivist of Thisgate?"]),
        act(tool="read_section", args={"chunk_id": 0}),
        act(answer="Ishmael", citations=[0]),
        bad("chunk 0 does not mention the archivist"),
        act(tool="grep", args={"pattern": "archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        ok(),
    ])
    res = MultiAgent(llm, Toolbox(doc), max_retries=1).run("Who is the chief archivist of Thisgate?")
    assert res.answer == "Daendis Draemmork" and res.findings[0].attempts == 2
    retry_prompt = llm.calls[4][1]["content"]
    assert "Ishmael" in retry_prompt and "chunk 0 does not mention the archivist" in retry_prompt
    assert res.tool_calls == 2


def test_unsupported_after_retries_is_flagged_not_found(doc):
    llm = ScriptedLLM([
        act(sub_questions=["Who is the mayor of Thisgate?", "Where was that person born?"]),
        act(tool="read_section", args={"chunk_id": 1}),
        act(answer="Bob", citations=[1]),
        bad(),
        act(answer="Bob", citations=[1]),
        bad(),
        act(answer="not found", citations=[]),  # second sub-question: abstains, no verifier call
    ])
    res = MultiAgent(llm, Toolbox(doc), max_retries=1).run("Where was the mayor of Thisgate born?")
    first = res.findings[0]
    assert first.flagged and not first.verified and first.answer == "not found" and first.attempts == 2
    assert "Bob" in first.reason
    assert res.answer == "not found" and res.citations == [] and res.stopped == "not_found"
    assert not llm.responses  # no synthesis call when nothing was found


def test_uncited_answer_fails_without_a_verifier_call(doc):
    verifier_llm = ScriptedLLM([])  # would raise if called
    v = LLMVerifier(verifier_llm)
    verdict = v.check("q", "Tuholm", [], Toolbox(doc))
    assert not verdict.supported and "cites no chunks" in verdict.reason
    assert v.check("q", "The document does not say.", [], Toolbox(doc)).supported
    assert v.calls == 0


def test_llm_verifier_parsing(doc):
    tb = Toolbox(doc)
    assert LLMVerifier(ScriptedLLM(['{"supported": "yes", "reason": "ok"}'])).check("q", "a", [0], tb).supported
    v = LLMVerifier(ScriptedLLM(["looks right to me"])).check("q", "a", [0], tb)
    assert not v.supported and "unparseable" in v.reason


def test_lexical_verifier(doc):
    tb = Toolbox(doc)
    c2 = chunk_with(doc, "was born in Tuholm")
    lv = LexicalVerifier()
    assert lv.check("q", "Tuholm", [c2], tb).supported
    rejected = lv.check("q", "Paris", [c2], tb)
    assert not rejected.supported and "paris" in rejected.reason


def test_offline_run_with_lexical_verifier(doc):
    c1, c2 = chunk_with(doc, "chief archivist of Thisgate is"), chunk_with(doc, "was born in Tuholm")
    llm = ScriptedLLM([
        act(sub_questions=["Who is the chief archivist of Thisgate?", "Where was that person born?"]),
        act(tool="grep", args={"pattern": "chief archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        act(tool="grep", args={"pattern": "was born in"}),
        act(answer="Tuholm", citations=[c2]),
        "Final answer: Tuholm",  # unparseable synthesis -> last verified finding
    ])
    res = MultiAgent(llm, Toolbox(doc), verifier=LexicalVerifier()).run(QUESTION)
    assert res.answer == "Tuholm" and res.citations == [c2] and res.stopped == "synth_error"
    item = EvalItem(id="t1", question=QUESTION, answer="Tuholm", type="multi_hop",
                    evidence=[FACT_1, FACT_2], doc_id="hay")
    score = score_agent_result(item, res, doc.chunks, system="multi")
    assert score.exact and score.tool_calls == 2 and score.system == "multi"


def test_planner_dedupes_and_caps_sub_questions(doc):
    llm = ScriptedLLM([act(sub_questions=["A?", " a? ", "", "B?", "C?", "D?"])])
    assert MultiAgent(llm, Toolbox(doc), max_sub_questions=2).plan("q") == ["A?", "B?"]
    assert MultiAgent(ScriptedLLM([act(sub_questions=[])]), Toolbox(doc)).plan("q") == ["q"]
    with pytest.raises(ValueError):
        MultiAgent(llm, Toolbox(doc), max_sub_questions=0)


def test_synthesis_cannot_cite_unverified_chunks(doc):
    c1, c2 = chunk_with(doc, "chief archivist of Thisgate is"), chunk_with(doc, "was born in Tuholm")
    llm = ScriptedLLM([
        act(sub_questions=["Who is the chief archivist of Thisgate?", "Where was that person born?"]),
        act(answer="Daendis Draemmork", citations=[c1]),  # c1 never shown -> citation dropped -> rejected
        act(tool="grep", args={"pattern": "chief archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        ok(),
        act(tool="grep", args={"pattern": "was born in"}),
        act(answer="Tuholm", citations=[c2]),
        ok(),
        act(answer="Tuholm", citations=[]),  # no citations given -> all verified chunks
    ])
    res = MultiAgent(llm, Toolbox(doc)).run(QUESTION)
    assert res.findings[0].attempts == 2
    assert res.answer == "Tuholm" and res.citations == [c1, c2]
