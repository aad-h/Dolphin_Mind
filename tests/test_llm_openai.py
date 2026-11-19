"""OpenAICompatLLM tests against a mocked HTTP transport (no network, no real key)."""

import json

import httpx
import pytest

from dolphinmind.agent import Document, LLMError, OpenAICompatLLM, ReActAgent, Toolbox
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack


def _ok(content, prompt=10, completion=5):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    })


class Server:
    """Replays queued responses (or exceptions) and records requests."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _llm(server, **kw):
    sleeps = []
    llm = OpenAICompatLLM(model="test-model", base_url="http://fake/v1", transport=httpx.MockTransport(server),
                          sleep=sleeps.append, **kw)
    return llm, sleeps


def test_sends_openai_payload_and_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("DOLPHIN_API_KEY", "sk-test-not-real")
    server = Server(_ok("hello"))
    llm, _ = _llm(server, temperature=0.0, max_tokens=64)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
    req = server.requests[0]
    assert req.url == "http://fake/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test-not-real"
    body = json.loads(req.content)
    assert body == {"model": "test-model", "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.0, "max_tokens": 64}
    assert "sk-test" not in repr(llm)


def test_no_auth_header_without_key(monkeypatch):
    monkeypatch.delenv("DOLPHIN_API_KEY", raising=False)
    server = Server(_ok("x"))
    llm, _ = _llm(server)
    llm.chat([{"role": "user", "content": "hi"}])
    assert "authorization" not in server.requests[0].headers


def test_env_base_url_and_model(monkeypatch):
    monkeypatch.setenv("DOLPHIN_BASE_URL", "http://local:8000/v1/")
    monkeypatch.setenv("DOLPHIN_MODEL", "qwen")
    llm = OpenAICompatLLM(transport=httpx.MockTransport(Server()))
    assert llm.base_url == "http://local:8000/v1" and llm.model == "qwen"


def test_tracks_token_usage_across_calls():
    llm, _ = _llm(Server(_ok("a", 10, 5), _ok("b", 20, 7)))
    llm.chat([{"role": "user", "content": "1"}])
    llm.chat([{"role": "user", "content": "2"}])
    assert llm.usage.calls == 2
    assert (llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.total_tokens) == (30, 12, 42)
    assert llm.last_usage == {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}


def test_retries_transient_errors_with_backoff_and_retry_after():
    server = Server(
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.Response(429, headers={"Retry-After": "7"}, text="rate limited"),
        httpx.Response(503, text="busy"),
        _ok("finally"),
    )
    llm, sleeps = _llm(server, max_retries=4, backoff=0.5)
    assert llm.chat([{"role": "user", "content": "q"}]) == "finally"
    assert sleeps == [0.5, 1.0, 7.0, 4.0]
    assert len(server.requests) == 5


def test_gives_up_after_max_retries():
    llm, sleeps = _llm(Server(*[httpx.Response(500, text="boom")] * 3), max_retries=2)
    with pytest.raises(LLMError, match="3 attempts.*HTTP 500"):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(sleeps) == 2


def test_client_errors_fail_fast():
    server = Server(httpx.Response(401, text="bad key"))
    llm, sleeps = _llm(server)
    with pytest.raises(LLMError, match="HTTP 401"):
        llm.chat([{"role": "user", "content": "q"}])
    assert sleeps == [] and not server.replies


def test_malformed_response_raises():
    llm, _ = _llm(Server(httpx.Response(200, json={"choices": []})))
    with pytest.raises(LLMError, match="malformed"):
        llm.chat([{"role": "user", "content": "q"}])


def test_drives_react_agent_end_to_end():
    doc = Document(insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60))
    cid = next(i for i, c in enumerate(doc.chunks) if "ALPHA7392BETA" in c)
    server = Server(
        _ok(json.dumps({"thought": "search", "tool": "grep", "args": {"pattern": "authentication code"}})),
        _ok(json.dumps({"thought": "done", "answer": "ALPHA7392BETA", "citations": [cid]})),
    )
    llm, _ = _llm(server)
    res = ReActAgent(llm, Toolbox(doc)).run("What is the authentication code?")
    assert res.answer == "ALPHA7392BETA" and res.citations == [cid]
    assert llm.usage.calls == 2 and llm.usage.total_tokens == 30
    # second request carries the grep observation back to the model
    assert "ALPHA7392BETA" in json.loads(server.requests[1].content)["messages"][-1]["content"]
