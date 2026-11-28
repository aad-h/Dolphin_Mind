"""Response cache + tracing in the HTTP service (TestClient + ScriptedLLM; no network or key)."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dolphinmind.agent import Document, ScriptedLLM, Tracer  # noqa: E402
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack  # noqa: E402
from dolphinmind.server import ResponseCache, create_app, normalize_question, response_key  # noqa: E402

BOOK = insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60)
CID = next(i for i, c in enumerate(Document(BOOK).chunks) if "ALPHA7392BETA" in c)
REPLIES = [
    json.dumps({"thought": "look", "tool": "grep", "args": {"pattern": "authentication code", "k": 1}}),
    json.dumps({"thought": "found", "answer": "ALPHA7392BETA", "citations": [CID]}),
]


class Factory:
    def __init__(self, replies):
        self.replies, self.made = replies, 0

    def __call__(self):
        self.made += 1
        return ScriptedLLM(self.replies)


def client(replies=REPLIES, **kw):
    factory = Factory(replies)
    return TestClient(create_app(llm_factory=factory, **kw)), factory


def upload(c):
    return c.post("/docs", json={"text": BOOK}).json()["doc_id"]


def events(body):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# -- ResponseCache ----------------------------------------------------------------------

def test_cache_lru_ttl_copies_and_stats():
    now = [0.0]
    cache = ResponseCache(max_entries=2, ttl=10, clock=lambda: now[0])
    cache.put(("a",), {"v": [1]})
    cache.put(("b",), {"v": [2]})
    got = cache.get(("a",))
    got["v"].append(99)  # callers get copies
    assert cache.get(("a",)) == {"v": [1]}
    cache.put(("c",), {"v": [3]})  # evicts b, the least recently used
    assert cache.get(("b",)) is None and len(cache) == 2
    now[0] = 10.5
    assert cache.get(("a",)) is None and len(cache) == 1  # expired and dropped
    assert cache.stats() == {"entries": 1, "max_entries": 2, "ttl": 10, "hits": 2, "misses": 2}
    cache.clear()
    assert len(cache) == 0
    for bad in ({"max_entries": 0}, {"ttl": 0}):
        with pytest.raises(ValueError):
            ResponseCache(**bad)


def test_key_normalizes_question_only():
    assert normalize_question("  Who   KILLED\nthe count? ") == "who killed the count?"
    assert response_key("d", "Who?", 6) == response_key("d", " who? ", 6)
    assert response_key("d", "Who?", 6) != response_key("d", "Who?", 5)
    assert response_key("d", "Who?", 6) != response_key("e", "Who?", 6)


# -- server cache -----------------------------------------------------------------------

def test_ask_is_answered_from_cache_without_calling_the_model():
    c, factory = client(cache=ResponseCache())
    doc_id = upload(c)
    first = c.post("/ask", json={"doc_id": doc_id, "question": "What is the code?"}).json()
    assert first["answer"] == "ALPHA7392BETA" and first["cached"] is False and factory.made == 1
    again = c.post("/ask", json={"doc_id": doc_id, "question": "  what is THE code? "}).json()
    assert again["cached"] is True and factory.made == 1
    assert {k: v for k, v in again.items() if k != "cached"} == {k: v for k, v in first.items() if k != "cached"}
    # a different step budget is a different question
    c.post("/ask", json={"doc_id": doc_id, "question": "What is the code?", "max_steps": 3})
    assert factory.made == 2
    # inline text maps to the same content-addressed doc, so it hits too
    assert c.post("/ask", json={"text": BOOK, "question": "What is the code?"}).json()["cached"] is True
    assert c.get("/health").json()["cache"]["hits"] == 2


def test_stream_replays_cached_steps():
    c, factory = client(cache=ResponseCache())
    doc_id = upload(c)
    url = f"/ask/stream?doc_id={doc_id}&question=What+is+the+code%3F"
    live = events(c.get(url).text)
    cached = events(c.get(url).text)
    assert factory.made == 1
    assert [e for e, _ in live] == [e for e, _ in cached] == ["start", "step", "answer"]
    assert live[1][1] == cached[1][1] and cached[1][1]["index"] == 1 and cached[1][1]["tool"] == "grep"
    assert live[2][1]["cached"] is False and cached[2][1]["cached"] is True


def test_unfinished_and_failed_runs_are_not_cached():
    grep = REPLIES[0]
    c, factory = client(replies=[grep, grep, "{}"], cache=ResponseCache())  # budget hit, then no answer
    doc_id = upload(c)
    for _ in range(2):
        r = c.post("/ask", json={"doc_id": doc_id, "question": "q", "max_steps": 1}).json()
        assert r["stopped"] == "max_steps" and r["cached"] is False
    assert factory.made == 2

    c, factory = client(replies=[], cache=ResponseCache())  # backend error
    doc_id = upload(c)
    assert c.post("/ask", json={"doc_id": doc_id, "question": "q"}).status_code == 502
    assert c.post("/ask", json={"doc_id": doc_id, "question": "q"}).status_code == 502
    assert len(c.app.state.cache) == 0


def test_cache_is_off_by_default():
    c, factory = client()
    doc_id = upload(c)
    for _ in range(2):
        assert c.post("/ask", json={"doc_id": doc_id, "question": "q"}).json()["cached"] is False
    assert factory.made == 2 and "cache" not in c.get("/health").json()


# -- server tracing ---------------------------------------------------------------------

def test_traces_endpoints_and_trace_id():
    c, _ = client(tracer=Tracer(), cache=ResponseCache())
    doc_id = upload(c)
    r = c.post("/ask", json={"doc_id": doc_id, "question": "What is the code?"}).json()
    hit = c.post("/ask", json={"doc_id": doc_id, "question": "What is the code?"}).json()
    traces = c.get("/traces").json()
    assert traces["enabled"] is True and len(traces["traces"]) == 2
    newest, oldest = traces["traces"]
    assert oldest["trace_id"] == r["trace_id"] and newest["trace_id"] == hit["trace_id"] != r["trace_id"]
    assert oldest["name"] == "server.ask" and oldest["attributes"]["cached"] is False
    run = oldest["children"][0]
    assert run["name"] == "agent.run"
    assert [s["name"] for s in run["children"]] == ["llm.chat", "tool.grep", "llm.chat"]
    assert newest["attributes"]["cached"] is True and newest["children"] == []  # no model work on a hit
    assert c.get(f"/traces/{r['trace_id']}").json()["span_id"] == oldest["span_id"]
    assert c.get("/traces/nope").status_code == 404
    assert len(c.get("/traces?limit=1").json()["traces"]) == 1


def test_stream_answer_carries_trace_id_and_tracing_off_by_default():
    c, _ = client(tracer=Tracer())
    doc_id = upload(c)
    answer = events(c.get(f"/ask/stream?doc_id={doc_id}&question=q").text)[-1][1]
    assert c.get(f"/traces/{answer['trace_id']}").status_code == 200

    c, _ = client()
    doc_id = upload(c)
    assert "trace_id" not in c.post("/ask", json={"doc_id": doc_id, "question": "q"}).json()
    assert c.get("/traces").json() == {"enabled": False, "traces": []}
    assert c.get("/traces/x").status_code == 404


def test_server_cli_flags(monkeypatch):
    import uvicorn

    from dolphinmind.server import __main__ as srv

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: seen.update(app=app, port=port))
    srv.main(["--port", "9", "--cache-size", "5", "--cache-ttl", "30", "--trace"])
    app = seen["app"]
    assert seen["port"] == 9 and app.state.cache.max_entries == 5 and app.state.cache.ttl == 30
    assert isinstance(app.state.tracer, Tracer)
    srv.main(["--cache-size", "0"])
    assert seen["app"].state.cache is None and seen["app"].state.tracer is None
    with pytest.raises(SystemExit):
        srv.main(["--cache-size", "-1"])
