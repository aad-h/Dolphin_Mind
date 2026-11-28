"""Tracing tests: in-memory span trees, agent/LLM/tool instrumentation, OpenTelemetry replay (offline)."""

import asyncio
import json
import threading

import httpx
import pytest

from dolphinmind.agent import (Budget, BudgetedLLM, BudgetExceeded, Document, MultiAgent, OpenAICompatLLM,
                               ReActAgent, ScriptedLLM, Toolbox, TracedLLM, Tracer, cli, format_trace)
from dolphinmind.agent.tracing import current_span, maybe_span, traced
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack

BOOK = insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60)


def act(**kw):
    return json.dumps(kw)


@pytest.fixture
def doc():
    return Document(BOOK)


def needle_cid(doc):
    return next(i for i, c in enumerate(doc.chunks) if "ALPHA7392BETA" in c)


def names(span):
    return [s.name for s in span.walk()]


# -- Tracer ----------------------------------------------------------------------------

def test_spans_nest_time_and_record_errors():
    tracer = Tracer()
    with tracer.span("root", a=1) as root:
        assert current_span() is root
        with tracer.span("child", b="x", skipped=None) as child:
            child.set(c=[1, 2])
        with pytest.raises(ValueError):
            with tracer.span("boom"):
                raise ValueError("bad")
    assert current_span() is None
    assert names(root) == ["root", "child", "boom"]
    assert child.parent_id == root.span_id and child.trace_id == root.trace_id
    assert child.attributes == {"b": "x", "c": [1, 2]}
    assert root.children[1].status == "error" and root.children[1].error == "ValueError: bad"
    assert root.status == "ok" and root.end_ns >= child.end_ns >= child.start_ns >= root.start_ns
    assert tracer.traces() == [root] and tracer.get(root.trace_id) is root
    d = root.to_dict()
    assert d["children"][0]["attributes"]["c"] == [1, 2] and d["seconds"] >= 0
    json.dumps(d)  # JSON-ready


def test_keeps_recent_traces_newest_first_and_exporters_cannot_break_queries():
    seen = []

    def broken(_):
        raise RuntimeError("collector down")

    tracer = Tracer(max_traces=2, exporters=[broken, seen.append])
    for i in range(3):
        with tracer.span(f"q{i}"):
            pass
    assert [t.name for t in tracer.traces()] == ["q2", "q1"]
    assert [t.name for t in tracer.traces(limit=1)] == ["q2"]
    assert [t.name for t in seen] == ["q0", "q1", "q2"]  # only roots are exported
    tracer.clear()
    assert tracer.traces() == []
    with pytest.raises(ValueError):
        Tracer(max_traces=0)


def test_context_follows_asyncio_threads_but_not_other_tracers():
    tracer, other = Tracer(), Tracer()

    def work(i):
        with tracer.span("work", i=i):
            with other.span("foreign"):
                pass

    async def main():
        with tracer.span("root") as root:
            await asyncio.gather(*(asyncio.to_thread(work, i) for i in range(3)))
        return root

    root = asyncio.run(main())
    assert sorted(c.attributes["i"] for c in root.children) == [0, 1, 2]
    assert all(c.parent_id == root.span_id for c in root.children)
    assert [t.name for t in other.traces()] == ["foreign"] * 3  # separate traces, not mixed in

    # a plain thread starts with an empty context: its span is a new trace
    def solo():
        with tracer.span("solo"):
            pass

    with tracer.span("outer") as outer:
        t = threading.Thread(target=solo)
        t.start()
        t.join()
    assert outer.children == [] and tracer.traces()[1].name == "solo"


def test_maybe_span_is_a_noop_without_tracer():
    with maybe_span(None, "x") as s:
        assert s is None


# -- instrumentation -------------------------------------------------------------------

def test_react_agent_traces_run_llm_and_tool_calls(doc):
    cid = needle_cid(doc)
    llm = ScriptedLLM([act(thought="t", tool="grep", args={"pattern": "authentication code", "k": 1}),
                       act(tool="nope", args={}),
                       act(answer="ALPHA7392BETA", citations=[cid])])
    tracer = Tracer()
    res = ReActAgent(llm, Toolbox(doc), tracer=tracer).run("What is the code?")
    assert res.answer == "ALPHA7392BETA"
    (root,) = tracer.traces()
    assert names(root) == ["agent.run", "llm.chat", "tool.grep", "llm.chat", "tool.nope", "llm.chat"]
    assert root.attributes["question"] == "What is the code?"
    assert root.attributes["answer"] == "ALPHA7392BETA" and root.attributes["citations"] == [cid]
    assert root.attributes["stopped"] == "answered" and root.attributes["tool_calls"] == 2
    grep = root.children[1]
    assert grep.attributes["args"] == {"pattern": "authentication code", "k": 1}
    assert grep.attributes["new_chunks"] == [cid] and grep.attributes["tool_error"] is False
    assert root.children[3].attributes["tool_error"] is True
    chat = root.children[0]
    assert chat.attributes["backend"] == "ScriptedLLM" and chat.attributes["tokens_estimated"] is True
    assert chat.attributes["messages"] == 2 and chat.attributes["prompt_tokens"] > 0
    # the wrapped backend is still reachable
    assert len(llm.calls) == 3
    text = format_trace(root)
    assert text.splitlines()[0].startswith("agent.run") and "\n  tool.grep" in text


def test_untraced_agent_keeps_backend_unwrapped(doc):
    llm = ScriptedLLM([act(answer="x", citations=[])])
    agent = ReActAgent(llm, Toolbox(doc))
    assert agent.llm is llm
    tracer = Tracer()
    wrapped = traced(llm, tracer)
    assert traced(wrapped, tracer) is wrapped and traced(llm, None) is llm


def test_backend_error_marks_spans_failed(doc):
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        ReActAgent(ScriptedLLM([]), Toolbox(doc), tracer=tracer).run("q")
    (root,) = tracer.traces()
    assert root.status == "error" and root.children[0].status == "error"
    assert "ran out" in root.children[0].error


def test_traced_openai_backend_records_real_usage():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 3}})

    api = OpenAICompatLLM(model="m", base_url="http://fake/v1", api_key="", transport=httpx.MockTransport(handler))
    tracer = Tracer()
    llm = TracedLLM(api, tracer)
    assert llm.chat_with_usage([{"role": "user", "content": "x"}]) == ("hi", (11, 3))
    assert llm.usage.total_tokens == 14  # forwarded attribute
    span = tracer.traces()[0]
    assert span.attributes == {"backend": "OpenAICompatLLM", "model": "m", "messages": 1, "prompt_tokens": 11,
                               "completion_tokens": 3, "tokens_estimated": False, "reply_chars": 2}


def test_traced_llm_composes_with_budgets():
    tracer = Tracer()
    budget = Budget(max_tokens=5)
    llm = TracedLLM(BudgetedLLM(ScriptedLLM(["ok"] * 3), budget), tracer)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "ok"
    assert llm.chat_with_usage([{"role": "user", "content": "x"}])[1] is None  # estimate stays an estimate
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "x" * 400}])
    assert [t.status for t in tracer.traces()] == ["error", "ok", "ok"]


def test_multi_agent_trace_covers_every_stage():
    text = insert_needle(make_haystack(4_000), "The chief archivist of Thisgate is Daendis Draemmork.", 20)
    text = insert_needle(text, "Daendis Draemmork was born in Tuholm.", 80)
    doc = Document(text)
    c1 = next(i for i, c in enumerate(doc.chunks) if "chief archivist of Thisgate is" in c)
    c2 = next(i for i, c in enumerate(doc.chunks) if "born in Tuholm" in c)
    llm = ScriptedLLM([
        act(sub_questions=["Who is the chief archivist?", "Where was that person born?"], depends_on=[[], [0]]),
        act(tool="grep", args={"pattern": "chief archivist"}),
        act(answer="Daendis Draemmork", citations=[c1]),
        act(supported=True, reason="stated"),
        act(tool="grep", args={"pattern": "Draemmork was born"}),
        act(answer="Tuholm", citations=[c2]),
        act(supported=True, reason="stated"),
        act(answer="Tuholm", citations=[c2]),
    ])
    tracer = Tracer()
    res = MultiAgent(llm, Toolbox(doc), tracer=tracer).run("Where was the chief archivist born?")
    assert res.answer == "Tuholm"
    (root,) = tracer.traces()
    assert root.name == "multi.run" and root.attributes["stopped"] == "answered"
    assert root.attributes["sub_questions"] == 2 and root.attributes["tool_calls"] == 2
    top = [c.name for c in root.children]
    assert top == ["multi.plan", "multi.research", "multi.research", "multi.synthesize"]
    assert root.children[0].attributes["depends_on"] == [[], [0]]
    research = root.children[1]
    assert [c.name for c in research.children] == ["agent.run", "multi.verify"]
    verify = research.children[1]
    assert verify.attributes["supported"] is True and [c.name for c in verify.children] == ["llm.chat"]
    assert names(research.children[0]) == ["agent.run", "llm.chat", "tool.grep", "llm.chat"]
    assert sum(1 for s in root.walk() if s.name == "llm.chat") == 8  # one span per model call, no doubles
    assert {s.trace_id for s in root.walk()} == {root.trace_id}


# -- OpenTelemetry ---------------------------------------------------------------------

def test_opentelemetry_exporter_replays_tree():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from dolphinmind.agent import OpenTelemetryExporter

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = Tracer(exporters=[OpenTelemetryExporter(provider.get_tracer("test"))])
    with tracer.span("agent.run", question="q", citations=[1, 2]) as root:
        with tracer.span("tool.grep", args={"pattern": "x"}):
            pass
        with pytest.raises(KeyError):
            with tracer.span("llm.chat"):
                raise KeyError("k")
    spans = {s.name: s for s in memory.get_finished_spans()}
    assert set(spans) == {"agent.run", "tool.grep", "llm.chat"}
    run, grep, chat = spans["agent.run"], spans["tool.grep"], spans["llm.chat"]
    assert grep.parent.span_id == run.context.span_id == chat.parent.span_id
    assert grep.context.trace_id == run.context.trace_id
    assert run.attributes["question"] == "q" and run.attributes["citations"] == "[1, 2]"
    assert grep.attributes["args"] == '{"pattern": "x"}'
    assert run.attributes["dolphin.trace_id"] == root.trace_id
    assert run.start_time == root.start_ns and run.end_time == root.end_ns
    assert chat.status.status_code == StatusCode.ERROR


# -- CLI ------------------------------------------------------------------------------

def test_cli_spans_flag_prints_span_tree(tmp_path, doc):
    book = tmp_path / "book.txt"
    book.write_text(BOOK, encoding="utf-8")
    script = tmp_path / "r.json"
    script.write_text(json.dumps([{"tool": "grep", "args": {"pattern": "authentication code", "k": 1}},
                                  {"answer": "ALPHA7392BETA", "citations": [needle_cid(doc)]}]))
    import io

    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["q", "--doc", str(book), "--backend", "scripted", "--script", str(script), "--spans"],
                    stdout=out, stderr=err)
    assert code == 0
    assert "Answer: ALPHA7392BETA" in out.getvalue()
    lines = err.getvalue().splitlines()
    assert lines[0].startswith("agent.run") and any(line.startswith("  tool.grep") for line in lines)
