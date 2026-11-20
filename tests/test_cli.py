"""CLI tests: scripted and mocked-HTTP backends only (no GPU, no network, no key)."""

import io
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from dolphinmind.agent import OpenAICompatLLM, cli
from dolphinmind.evaluation import NEEDLES, insert_needle, make_haystack

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "book.txt"
    p.write_text(insert_needle(make_haystack(3_000), NEEDLES[0]["text"], 60), encoding="utf-8")
    return p


def _script(tmp_path, replies):
    p = tmp_path / "replies.json"
    p.write_text(json.dumps(replies), encoding="utf-8")
    return p


def _replies(cid):
    """Grep for the needle, then answer citing its chunk."""
    return [
        {"thought": "look for it", "tool": "grep", "args": {"pattern": "authentication code", "k": 1}},
        {"thought": "found it", "answer": "ALPHA7392BETA", "citations": [cid]},
    ]


def _needle_cid(book_path, chunk_size=256, overlap=64):
    from dolphinmind.agent import Document

    doc = Document(book_path.read_text(), chunk_size, overlap)
    return next(i for i, c in enumerate(doc.chunks) if "ALPHA7392BETA" in c)


def run(argv, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_scripted_backend_prints_answer_citations_and_trace(tmp_path, book):
    cid = _needle_cid(book)
    script = _script(tmp_path, _replies(cid))
    code, out, err = run(["What is the authentication code?", "--doc", str(book),
                          "--backend", "scripted", "--script", str(script), "--trace"])
    assert code == 0, err
    assert "Answer: ALPHA7392BETA" in out
    assert f"[chunk {cid}]" in out and "Citations:" in out
    assert "Trace" in out and "grep(" in out and "thought: look for it" in out
    assert "Stopped: answered | tool calls: 1" in out


def test_trace_is_hidden_by_default_and_json_output(tmp_path, book):
    cid = _needle_cid(book)
    argv = ["q", "--doc", str(book), "--backend", "scripted"]
    code, out, _ = run(argv + ["--script", str(_script(tmp_path, _replies(cid)))])
    assert code == 0 and "Trace" not in out

    code, out, _ = run(argv + ["--script", str(_script(tmp_path, _replies(cid))), "--json"])
    data = json.loads(out)
    assert data["answer"] == "ALPHA7392BETA" and data["tool_calls"] == 1
    assert data["citations"][0]["chunk_id"] == cid and "ALPHA7392BETA" in data["citations"][0]["text"]
    assert data["steps"][0]["tool"] == "grep" and "usage" not in data


def test_doc_from_stdin_and_unseen_citation_dropped(tmp_path):
    script = _script(tmp_path, ['{"answer": "blue", "citations": [0]}'])  # chunk 0 never read
    code, out, _ = run(["sky colour?", "--doc", "-", "--backend", "scripted", "--script", str(script)],
                       stdin="The sky is blue. Grass is green.")
    assert code == 0
    assert "Answer: blue" in out and "Citations: none" in out


def test_openai_backend_via_mocked_transport_reports_tokens(tmp_path, book, monkeypatch):
    cid = _needle_cid(book)
    replies = [json.dumps(r) for r in _replies(cid)]
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": replies.pop(0)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    def fake(model=None, base_url=None):
        return OpenAICompatLLM(model=model, base_url=base_url or "http://fake/v1",
                               transport=httpx.MockTransport(handler))

    monkeypatch.setattr(cli, "OpenAICompatLLM", fake)
    monkeypatch.setenv("DOLPHIN_API_KEY", "sk-test-not-real")
    code, out, err = run(["q", "--doc", str(book), "--model", "m1"])
    assert code == 0, err
    assert "tokens: 240 (200 in / 40 out)" in out
    assert json.loads(seen[0].content)["model"] == "m1"
    assert "sk-test-not-real" not in out + err


def test_openai_backend_error_returns_exit_1(book, monkeypatch):
    def fake(model=None, base_url=None):
        return OpenAICompatLLM(base_url="http://fake/v1", max_retries=0,
                               transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))

    monkeypatch.setattr(cli, "OpenAICompatLLM", fake)
    code, out, err = run(["q", "--doc", str(book)])
    assert code == 1 and "HTTP 401" in err and out == ""


def test_hf_backend_uses_load_qwen(tmp_path, book, monkeypatch):
    import dolphinmind.models as models

    loaded = {}

    def fake_load(name):
        loaded["name"] = name
        return "model", "tokenizer"

    monkeypatch.setattr(models, "load_qwen", fake_load)

    class FakeHF:
        def __init__(self, model, tokenizer):
            self.replies = ['{"answer": "not found", "citations": []}']

        def chat(self, messages):
            return self.replies.pop(0)

    monkeypatch.setattr(cli, "HFChatLLM", FakeHF)
    code, out, _ = run(["q", "--doc", str(book), "--backend", "hf"])
    assert code == 0 and "Answer: not found" in out
    assert loaded["name"] == models.QWEN


def test_semantic_flag_enables_search_doc(tmp_path, book, monkeypatch):
    pytest.importorskip("faiss")
    from dolphinmind.retrieval import SemanticRetriever
    from test_core import _BagOfWords

    monkeypatch.setattr(cli, "SemanticRetriever",
                        lambda: SemanticRetriever(embedding_model=_BagOfWords(["authentication", "code", "neural"])))
    script = _script(tmp_path, [
        {"tool": "search_doc", "args": {"query": "authentication code", "k": 2}},
        {"answer": "ALPHA7392BETA", "citations": [_needle_cid(book)]},
    ])
    code, out, _ = run(["q", "--doc", str(book), "--backend", "scripted", "--script", str(script),
                        "--semantic", "--json"])
    data = json.loads(out)
    assert code == 0 and "ALPHA7392BETA" in data["steps"][0]["observation"]
    assert data["citations"][0]["chunk_id"] == _needle_cid(book)


@pytest.mark.parametrize("extra, msg", [
    (["--backend", "scripted"], "needs --script"),
    (["--backend", "scripted", "--script", "/no/such/file.json"], "error:"),
])
def test_bad_backend_config_exits_2(book, extra, msg):
    code, _, err = run(["q", "--doc", str(book)] + extra)
    assert code == 2 and msg in err


def test_bad_inputs_exit_2(tmp_path, book):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ")
    assert run(["q", "--doc", str(tmp_path / "missing.txt")])[0] == 2
    assert "empty" in run(["q", "--doc", str(empty)])[2]
    bad_script = _script(tmp_path, {"not": "a list"})
    assert "non-empty JSON list" in run(["q", "--doc", str(book), "--backend", "scripted",
                                         "--script", str(bad_script)])[2]
    for argv in (["--max-steps", "0"], ["--chunk-size", "10", "--overlap", "10"], ["--backend", "gpt"]):
        with pytest.raises(SystemExit) as e:
            run(["q", "--doc", str(book)] + argv)
        assert e.value.code == 2


def test_script_running_out_returns_exit_1(tmp_path, book):
    script = _script(tmp_path, [{"tool": "grep", "args": {"pattern": "x"}}])
    code, _, err = run(["q", "--doc", str(book), "--backend", "scripted", "--script", str(script)])
    assert code == 1 and "ran out" in err


def test_python_dash_m_entry_point(tmp_path, book):
    script = _script(tmp_path, [{"answer": "not found", "citations": []}])
    proc = subprocess.run(
        [sys.executable, "-m", "dolphinmind.agent", "q", "--doc", str(book), "--backend", "scripted",
         "--script", str(script)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Answer: not found" in proc.stdout
