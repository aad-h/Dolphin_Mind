"""HTTP service tests: FastAPI TestClient + ScriptedLLM / fake embedder (no GPU, network or key)."""

import json
import threading
from typing import List

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dolphinmind.agent import Document, ScriptedLLM  # noqa: E402
from dolphinmind.agent.react import ReActAgent  # noqa: E402
from dolphinmind.agent.tools import Toolbox  # noqa: E402
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack  # noqa: E402
from dolphinmind.retrieval import HashingEmbedder  # noqa: E402
from dolphinmind.server import DocStore, create_app, doc_key  # noqa: E402

BOOK = insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60)


def _needle_cid(text=BOOK, chunk_size=256, overlap=64):
    return next(i for i, c in enumerate(Document(text, chunk_size, overlap).chunks) if "ALPHA7392BETA" in c)


def _replies(cid) -> List[str]:
    return [
        json.dumps({"thought": "look for it", "tool": "grep", "args": {"pattern": "authentication code", "k": 1}}),
        json.dumps({"thought": "found it", "answer": "ALPHA7392BETA", "citations": [cid]}),
    ]


class Factory:
    """Hands each request its own ScriptedLLM and records how many were made."""

    def __init__(self, replies: List[str]):
        self.replies, self.made = replies, []

    def __call__(self):
        llm = ScriptedLLM(self.replies)
        self.made.append(llm)
        return llm


def _client(replies=None, **kw):
    factory = Factory(_replies(_needle_cid()) if replies is None else replies)
    return TestClient(create_app(llm_factory=factory, **kw)), factory


def _events(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health_upload_and_chunk_lookup():
    client, _ = _client()
    assert client.get("/health").json() == {"status": "ok", "docs": 0}
    r = client.post("/docs", json={"text": BOOK})
    assert r.status_code == 200
    info = r.json()
    assert info["doc_id"] == doc_key(BOOK, 256, 64) and info["n_chunks"] == len(Document(BOOK).chunks)
    assert info["semantic"] is False
    # identical upload is served from the cache
    assert client.post("/docs", json={"text": BOOK}).json()["doc_id"] == info["doc_id"]
    assert client.get("/health").json()["docs"] == 1
    assert client.get(f"/docs/{info['doc_id']}").json() == info
    cid = _needle_cid()
    chunk = client.get(f"/docs/{info['doc_id']}/chunks/{cid}").json()
    assert "ALPHA7392BETA" in chunk["text"]
    assert client.get(f"/docs/{info['doc_id']}/chunks/9999").status_code == 404
    assert client.get("/docs/nope").status_code == 404


def test_upload_validation():
    client, _ = _client()
    assert client.post("/docs", json={"text": "   "}).status_code == 422
    assert client.post("/docs", json={"text": "a.", "chunk_size": 32, "overlap": 32}).status_code == 422
    small, _ = _client(max_doc_chars=10)
    assert small.post("/docs", json={"text": "x" * 11}).status_code == 413


def test_ask_by_doc_id_returns_answer_citations_and_steps():
    client, factory = _client()
    doc_id = client.post("/docs", json={"text": BOOK}).json()["doc_id"]
    r = client.post("/ask", json={"question": "What is the authentication code?", "doc_id": doc_id})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "ALPHA7392BETA" and data["doc_id"] == doc_id
    assert data["citations"][0]["chunk_id"] == _needle_cid()
    assert data["tool_calls"] == 1 and data["steps"][0]["tool"] == "grep"
    assert len(factory.made) == 1


def test_ask_with_inline_text_caches_doc_and_validates_source():
    client, _ = _client()
    r = client.post("/ask", json={"question": "code?", "text": BOOK})
    assert r.status_code == 200 and r.json()["answer"] == "ALPHA7392BETA"
    assert client.get(f"/docs/{r.json()['doc_id']}").status_code == 200
    assert client.post("/ask", json={"question": "q"}).status_code == 422
    assert client.post("/ask", json={"question": "q", "doc_id": "x", "text": "y"}).status_code == 422
    assert client.post("/ask", json={"question": "q", "doc_id": "missing"}).status_code == 404
    assert client.post("/ask", json={"question": "q", "text": BOOK, "max_steps": 0}).status_code == 422


def test_each_request_gets_fresh_toolbox_and_llm():
    """Citations seen in one request must not validate citations in the next."""
    cid = _needle_cid()
    client, factory = _client([json.dumps({"answer": "ALPHA7392BETA", "citations": [cid]})])
    doc_id = client.post("/docs", json={"text": BOOK}).json()["doc_id"]
    for _ in range(2):
        data = client.post("/ask", json={"question": "q", "doc_id": doc_id}).json()
        assert data["citations"] == []  # never read in this request -> dropped
    assert len(factory.made) == 2


def test_backend_error_maps_to_502():
    client, _ = _client([])  # empty script -> ScriptedLLM raises RuntimeError
    r = client.post("/ask", json={"question": "q", "text": BOOK})
    assert r.status_code == 502 and "model backend error" in r.json()["detail"]


def test_stream_emits_start_steps_and_answer():
    client, _ = _client()
    doc_id = client.post("/docs", json={"text": BOOK}).json()["doc_id"]
    with client.stream("GET", "/ask/stream", params={"doc_id": doc_id, "question": "code?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = _events(body)
    assert [e for e, _ in events] == ["start", "step", "answer"]
    assert events[0][1]["question"] == "code?"
    step = events[1][1]
    assert step["index"] == 1 and step["tool"] == "grep" and "ALPHA7392BETA" in step["observation"]
    assert events[2][1]["answer"] == "ALPHA7392BETA" and events[2][1]["citations"][0]["chunk_id"] == _needle_cid()


def test_stream_errors():
    client, _ = _client([])
    assert client.get("/ask/stream", params={"doc_id": "missing", "question": "q"}).status_code == 404
    doc_id = client.post("/docs", json={"text": BOOK}).json()["doc_id"]
    assert client.get("/ask/stream", params={"doc_id": doc_id, "question": "q", "max_steps": 99}).status_code == 422
    body = client.get("/ask/stream", params={"doc_id": doc_id, "question": "q"}).text
    events = _events(body)
    assert [e for e, _ in events] == ["start", "error"]
    assert "ran out of responses" in events[1][1]["detail"]


def test_stream_steps_arrive_before_the_agent_finishes():
    """Over a real socket, the step event is readable while the model is still blocked on its next reply.

    (TestClient buffers whole responses, so this runs uvicorn on a loopback port.)
    """
    uvicorn = pytest.importorskip("uvicorn")
    import socket
    import time

    import httpx

    release = threading.Event()
    cid = _needle_cid()

    class SlowLLM(ScriptedLLM):
        def chat(self, messages):
            if self.calls:  # second call: wait until the client has seen the first step
                if not release.wait(5):
                    raise RuntimeError("step event was not delivered before the agent finished")
            return super().chat(messages)

    app = create_app(llm_factory=lambda: SlowLLM(_replies(cid)))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=10) as http:
            doc_id = http.post("/docs", json={"text": BOOK}).json()["doc_id"]
            seen = []
            with http.stream("GET", "/ask/stream", params={"doc_id": doc_id, "question": "q"}) as r:
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        seen.append(line[7:])
                        if line == "event: step":
                            release.set()
    finally:
        server.should_exit = True
        thread.join(10)
    assert seen == ["start", "step", "answer"]


def test_doc_store_lru_and_semantic_index():
    made = []

    def embedder():
        made.append(1)
        return HashingEmbedder(dim=256)

    store = DocStore(max_docs=2, embedder_factory=embedder)
    a, b = store.add("alpha text. more."), store.add("beta text. more.")
    store.get(a.doc_id)  # touch a -> b is least recently used
    store.add("gamma text. more.")
    assert len(store) == 2 and store.get(b.doc_id) is None and store.get(a.doc_id) is not None
    assert a.retriever is not None and len(made) == 1  # embedder built once, reused
    with pytest.raises(ValueError):
        DocStore(max_docs=0)

    # search_doc works through the API with the prebuilt index
    replies = [json.dumps({"tool": "search_doc", "args": {"query": "authentication code", "k": 1}}),
               json.dumps({"answer": "ALPHA7392BETA", "citations": [_needle_cid()]})]
    client = TestClient(create_app(llm_factory=lambda: ScriptedLLM(replies), store=DocStore(embedder_factory=embedder)))
    data = client.post("/ask", json={"question": "What is the authentication code?", "text": BOOK}).json()
    assert data["steps"][0]["tool"] == "search_doc" and "ALPHA7392BETA" in data["steps"][0]["observation"]
    assert data["citations"][0]["chunk_id"] == _needle_cid()


def test_react_on_step_callback():
    cid = _needle_cid()
    seen = []
    res = ReActAgent(ScriptedLLM(_replies(cid)), Toolbox(Document(BOOK))).run("q", on_step=seen.append)
    assert [s.tool for s in seen] == ["grep", None] and seen == res.steps
