import json

import pytest

from dolphinmind.agent import Document, ReActAgent, ScriptedLLM, Toolbox, parse_action
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack


def _doc():
    return Document(insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60))


def _needle_chunk(doc):
    return next(i for i, c in enumerate(doc.chunks) if "ALPHA7392BETA" in c)


def act(**kw):
    return json.dumps(kw)


def test_parse_action_handles_fences_and_chatter():
    assert parse_action('```json\n{"tool": "grep", "args": {"pattern": "x"}}\n```')["tool"] == "grep"
    assert parse_action('Sure! {"answer": "42", "citations": [1]} hope that helps}')["answer"] == "42"
    with pytest.raises(ValueError):
        parse_action("no json here")


def test_agent_greps_then_answers_with_valid_citation():
    doc = _doc()
    cid = _needle_chunk(doc)
    llm = ScriptedLLM([
        act(thought="look for it", tool="grep", args={"pattern": "authentication code"}),
        act(thought="found", answer="ALPHA7392BETA", citations=[cid, 999]),
    ])
    res = ReActAgent(llm, Toolbox(doc)).run("What is the authentication code?")
    assert res.answer == "ALPHA7392BETA"
    assert res.citations == [cid]  # 999 was never seen -> dropped
    assert res.tool_calls == 1 and res.stopped == "answered"
    assert "ALPHA7392BETA" in res.steps[0].observation


def test_agent_recovers_from_bad_json_and_unknown_tool():
    doc = _doc()
    llm = ScriptedLLM([
        "I think I should search",  # not JSON
        act(tool="teleport", args={}),  # unknown tool
        act(answer="not found", citations=[]),
    ])
    res = ReActAgent(llm, Toolbox(doc)).run("q")
    assert res.answer == "not found"
    assert "unknown tool" in res.steps[0].observation
    assert any("Invalid reply" in m["content"] for m in llm.calls[-1])


def test_agent_stops_at_step_budget():
    doc = _doc()
    llm = ScriptedLLM([act(tool="read_section", args={"chunk_id": 0})] * 3 + [act(answer="x", citations=[0])])
    res = ReActAgent(llm, Toolbox(doc), max_steps=3).run("q")
    assert res.stopped == "max_steps" and res.tool_calls == 3 and res.citations == [0]


def test_tools_validate_inputs():
    tb = Toolbox(_doc())
    assert "error" in tb.call("read_section", {"chunk_id": 10_000})
    assert "bad arguments" in tb.call("grep", {"wrong": 1})
    assert "semantic search unavailable" in tb.call("search_doc", {"query": "x"})
    assert tb.call("calculator", {"expression": "(47*2)+3.5"}) == "97.5"
    assert "error" in tb.call("calculator", {"expression": "__import__('os').system('ls')"})
    assert "error" in tb.call("calculator", {"expression": "9**9999"})


def test_multi_chunk_results_truncate_per_chunk():
    doc = Document(" ".join(f"Sentence number {i} is here." for i in range(400)), chunk_size=60, overlap=10)
    tb = Toolbox(doc, max_chars=50)
    out = tb.call("grep", {"pattern": "sentence", "k": 3})
    # every returned chunk is visible, not just the first ones
    assert all(f"[chunk {i}]" in out for i in range(3))
    assert all(len(block) <= len("[chunk 0] ") + 50 for block in out.split("\n\n"))


def test_default_toolbox_shows_whole_default_chunks():
    doc = Document(make_haystack(5_000))
    tb = Toolbox(doc)
    assert all(tb.read_section(i) == f"[chunk {i}] {c}" for i, c in enumerate(doc.chunks))
