"""run_agent_eval: system comparison with a rule-based fake reader (no GPU, no network, no downloads)."""

import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

pytest.importorskip("faiss")

from dolphinmind.agent import LLMError, OpenAICompatLLM, ScriptedLLM  # noqa: E402
from dolphinmind.agent.multi import PLANNER_PROMPT, VERIFIER_PROMPT  # noqa: E402
from dolphinmind.agent.tools import Document, Toolbox  # noqa: E402
from dolphinmind.eval import EvalConfig, EvalItem, generate, run_eval, save_jsonl, write_csv  # noqa: E402
from dolphinmind.eval.runner import (ANSWER_SYSTEM, CSV_FIELDS, MeteredLLM, estimate_tokens,  # noqa: E402
                                     format_comparison, load_datasets, main)
from dolphinmind.evaluation import make_haystack  # noqa: E402
from dolphinmind.retrieval import HashingEmbedder, SemanticRetriever, fixed_chunks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_CHUNK = re.compile(r"\[chunk (\d+)\] (.*?)(?=\n\n\[chunk \d+\] |\Z)", re.DOTALL)


def _squash(s):
    return " ".join(s.split())


class ReaderLLM:
    """An honest fake model: answers only when every gold evidence sentence is in what it was shown.

    Single-pass prompts: answer gold iff all evidence is in the context, else "not found".
    ReAct prompts: grep for each evidence sentence's opening words, then answer citing the
    chunks that contain the evidence (or "not found" if it never saw it all).
    """

    def __init__(self, items):
        self.by_q = {it.question: it for it in items}
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        question = re.search(r"Question: (.*?)(?:\n|$)", messages[1]["content"]).group(1)
        item = self.by_q[question]
        evidence = [_squash(e) for e in item.evidence]
        if messages[0]["content"] == ANSWER_SYSTEM:
            ctx = _squash(messages[1]["content"])
            return item.answer if evidence and all(e in ctx for e in evidence) else "not found"

        seen = {}
        for m in messages[2:]:
            if m["role"] == "user" and m["content"].startswith("Observation:"):
                for cid, text in _CHUNK.findall(m["content"]):
                    seen[int(cid)] = _squash(text)
        have = [e for e in evidence if any(e in t for t in seen.values())]
        tool_calls = sum(1 for m in messages if m["role"] == "assistant")
        if evidence and len(have) == len(evidence):
            cites = sorted(c for c, t in seen.items() if any(e in t for e in evidence))
            return json.dumps({"thought": "done", "answer": item.answer, "citations": cites})
        if tool_calls < len(evidence):
            pattern = re.escape(" ".join(item.evidence[tool_calls].split()[:4]))
            return json.dumps({"thought": "search", "tool": "grep", "args": {"pattern": pattern, "k": 2}})
        return json.dumps({"thought": "nothing", "answer": "not found", "citations": []})


@pytest.fixture(scope="module")
def bench():
    # facts planted all through a 6k-word doc; truncation sees only the first ~1k words
    return generate(make_haystack(6_000), n_questions=12, seed=3, doc_id="hay")


CFG = EvalConfig(chunk_size=128, overlap=32, top_k=4)


# -- retrieval helpers -------------------------------------------------------------------

def test_fixed_chunks_and_hashing_embedder():
    text = " ".join(f"w{i}" for i in range(10))
    assert fixed_chunks(text, 4) == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]
    assert fixed_chunks("", 4) == []
    with pytest.raises(ValueError):
        fixed_chunks(text, 0)

    emb = HashingEmbedder(dim=64)
    a, b = emb.encode(["The Raven of Tuholm", "raven tuholm"])
    np.testing.assert_allclose(a, b)  # case and stopwords ignored
    assert emb.encode(["x"]).shape == (1, 64) and emb.encode(["x"]).dtype == np.float32


def test_search_returns_scored_ids_best_first():
    chunks = ["apples and pears", "the tuholm raven", "raven raven tuholm mill", "nothing here"]
    r = SemanticRetriever(embedding_model=HashingEmbedder()).build(chunks)
    hits = r.search("raven of tuholm", top_k=2)
    assert {i for i, _ in hits} == {1, 2}
    assert hits[0][1] >= hits[1][1]
    assert r.retrieve("raven of tuholm", top_k=2) == [chunks[1], chunks[2]]  # document order
    assert len(r.search("q", top_k=99)) == 4


def test_toolbox_reuses_built_index_and_resets():
    doc = Document("Ann was born in Tuholm. Bob sells salt. " * 50, chunk_size=20, overlap=5)
    r = SemanticRetriever(embedding_model=HashingEmbedder()).build(doc.chunks)
    index = r.index
    tb = Toolbox(doc, r)
    assert r.index is index  # not rebuilt
    assert "[chunk" in tb.search_doc("Tuholm", k=2) and tb.seen
    tb.reset()
    assert not tb.seen


# -- metering ----------------------------------------------------------------------------

def test_metered_llm_estimates_or_uses_backend_usage():
    m = MeteredLLM(ScriptedLLM(["abcdefgh"]))
    assert m.estimated
    m.chat([{"role": "user", "content": "x" * 40}])
    assert m.snapshot() == (10, 2) and m.calls == 1
    assert estimate_tokens("") == 0

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}],
                                          "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    api = OpenAICompatLLM(model="m", api_key="", transport=httpx.MockTransport(handler))
    api.usage.prompt_tokens = 100  # earlier, unrelated usage must not be counted
    m = MeteredLLM(api)
    assert not m.estimated
    m.chat([{"role": "user", "content": "hello"}])
    m.chat([{"role": "user", "content": "hello"}])
    assert m.snapshot() == (14, 6)


# -- the comparison ----------------------------------------------------------------------

def test_run_eval_compares_all_systems(bench):
    llm = ReaderLLM(bench.items)
    run = run_eval(bench.items, {"hay": bench.text}, llm, cfg=CFG, embedder=HashingEmbedder())
    assert run.systems == ["truncation", "rag", "dolphinmind", "react"]
    assert len(run.rows) == 4 * len(bench.items) and run.tokens_estimated

    def acc(system):
        return np.mean([s.fuzzy for s in run.scores(system)])

    # every reader is honest, so unanswerable questions are always right
    for s in run.scores():
        if s.type == "unanswerable":
            assert s.fuzzy and s.evidence_recall is None
    # truncation only sees the first 512 words: it misses facts planted deeper
    assert acc("truncation") < acc("dolphinmind")
    assert acc("react") == 1.0  # grep finds every planted fact
    react = run.scores("react")
    assert all(s.evidence_recall == 1.0 for s in react if s.type != "unanswerable")
    assert any(s.tool_calls >= 2 for s in react if s.type == "multi_hop")
    single = [s for s in run.scores() if s.system != "react"]
    assert all(s.tool_calls == 0 and s.prompt_tokens > 0 for s in single)
    # a correct single-pass answer needs the evidence in the context; only plain RAG can
    # get it split across two retrieved windows (sentence-blind chunking), so recall < 1 there
    for s in single:
        if s.type != "unanswerable" and s.fuzzy:
            assert s.evidence_recall == 1.0 or (s.system == "rag" and s.evidence_recall > 0)
    assert all(s.citations == [0] for s in run.scores("truncation"))


def test_run_eval_validation(bench):
    llm = ReaderLLM(bench.items)
    with pytest.raises(ValueError, match="unknown system"):
        run_eval(bench.items, {"hay": bench.text}, llm, ["bm25"])
    with pytest.raises(ValueError, match="need an embedder"):
        run_eval(bench.items, {"hay": bench.text}, llm, ["rag"])
    with pytest.raises(ValueError, match="no document text"):
        run_eval(bench.items, {}, llm, ["truncation"])
    with pytest.raises(ValueError):
        EvalConfig(chunk_size=64, overlap=64)
    # react without an embedder falls back to grep-only tools
    run = run_eval(bench.items[:2], {"hay": bench.text}, llm, ["react"], CFG)
    assert len(run.rows) == 2


class MultiReaderLLM(ReaderLLM):
    """ReaderLLM plus the multi-agent roles: a one-step planner and an honest verifier."""

    def chat(self, messages):
        system = messages[0]["content"]
        if system == PLANNER_PROMPT.format(max_sub=4):
            self.calls += 1
            return json.dumps({"sub_questions": [messages[1]["content"].removeprefix("Question: ")]})
        if system == VERIFIER_PROMPT:
            self.calls += 1
            question = re.search(r"Question: (.*?)\n", messages[1]["content"]).group(1)
            excerpts = _squash(messages[1]["content"])
            ok = all(_squash(e) in excerpts for e in self.by_q[question].evidence)
            return json.dumps({"supported": ok, "reason": "evidence present" if ok else "missing"})
        return super().chat(messages)


def test_run_eval_multi_agent_system(bench):
    run = run_eval(bench.items, {"hay": bench.text}, MultiReaderLLM(bench.items), ["react", "multi"], CFG,
                   embedder=HashingEmbedder())
    assert run.systems == ["react", "multi"]
    multi = run.scores("multi")
    assert len(multi) == len(bench.items) and all(s.fuzzy for s in multi)
    assert all(s.evidence_recall == 1.0 for s in multi if s.type != "unanswerable")
    # planner + verifier calls cost tokens on top of the worker's
    react = {s.id: s for s in run.scores("react")}
    answerable = [s for s in multi if s.type != "unanswerable"]
    assert all(s.prompt_tokens > react[s.id].prompt_tokens for s in answerable)
    assert format_comparison(run, n_boot=50).count("| multi |") >= 1


class _Flaky:
    def __init__(self, fail_times):
        self.fail_times = fail_times

    def chat(self, messages):
        if self.fail_times:
            self.fail_times -= 1
            raise LLMError("HTTP 503")
        return "not found"


def test_backend_errors_count_as_wrong_then_abort(bench):
    items = [it for it in bench.items if it.type == "unanswerable"][:2]
    run = run_eval(items, {"hay": bench.text}, _Flaky(1), ["truncation"], CFG)
    first, second = run.scores()
    assert first.prediction == "[error]" and not first.fuzzy  # an error is not an abstention
    assert second.fuzzy and run.rows[0].error == "HTTP 503" and not run.rows[1].error
    with pytest.raises(LLMError, match="aborting after 2"):
        run_eval(items, {"hay": bench.text}, _Flaky(9), ["truncation"], EvalConfig(max_errors=1))


def test_csv_and_markdown(bench, tmp_path):
    run = run_eval(bench.items, {"hay": bench.text}, ReaderLLM(bench.items),
                   cfg=EvalConfig(chunk_size=128, overlap=32, top_k=4, prices=(1.0, 2.0)),
                   embedder=HashingEmbedder())
    path = tmp_path / "out" / "eval.csv"
    write_csv(run, path)
    rows = list(csv.DictReader(path.open()))
    assert tuple(rows[0]) == CSV_FIELDS and len(rows) == len(run.rows)
    react = next(r for r in rows if r["system"] == "react" and r["type"] == "multi_hop")
    assert react["citations"] and react["doc_id"] == "hay" and float(react["cost_usd"]) > 0
    unans = next(r for r in rows if r["type"] == "unanswerable")
    assert unans["evidence_recall"] == ""

    md = format_comparison(run, n_boot=200)
    lines = md.splitlines()
    assert lines[0] == ("| system | n | exact | fuzzy | evidence_recall | tool_calls | total_tokens | "
                        "latency_s | cost_usd |")
    assert [ln.split(" | ")[0] for ln in lines[2:6]] == ["| truncation", "| rag", "| dolphinmind", "| react"]
    assert "| system | single_hop | multi_hop | numeric | unanswerable |" in md
    assert "| react | 100.0 [100.0, 100.0] |" in md
    assert "Token counts are estimated" in md


# -- command line ------------------------------------------------------------------------

@pytest.fixture
def dataset(tmp_path, bench):
    (tmp_path / "hay.txt").write_text(bench.text, encoding="utf-8")
    save_jsonl(bench.items, tmp_path / "hay.jsonl")
    return tmp_path / "hay.jsonl"


def _main(argv, **kw):
    out, err = io.StringIO(), io.StringIO()
    return main(argv, stdout=out, stderr=err, **kw), out.getvalue(), err.getvalue()


def test_cli_end_to_end(dataset, bench, tmp_path):
    out_csv = tmp_path / "res" / "eval.csv"
    code, out, err = _main([str(dataset), "--embedder", "hashing", "--chunk-size", "128", "--overlap", "32",
                            "--top-k", "4", "--out", str(out_csv), "--n-boot", "100"],
                           llm=ReaderLLM(bench.items))
    assert code == 0, err
    assert "| dolphinmind |" in out and "[1/48]" in err
    assert out_csv.exists() and out_csv.with_suffix(".md").read_text().startswith("| system |")

    code, out, err = _main([str(dataset), "--systems", "react", "--types", "numeric", "--limit", "2",
                            "--embedder", "hashing", "--quiet", "--out", str(out_csv)],
                           llm=ReaderLLM(bench.items))
    assert code == 0 and err.count("[") == 0
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 2 and {r["type"] for r in rows} == {"numeric"}


def test_cli_doc_override_and_missing_doc(tmp_path, bench):
    items = [EvalItem(**{**it.to_dict(), "doc_id": ""}) for it in bench.items[:2]]
    save_jsonl(items, tmp_path / "qs.jsonl")
    code, _, err = _main([str(tmp_path / "qs.jsonl"), "--systems", "truncation"], llm=ReaderLLM(items))
    assert code == 2 and "qs.txt" in err  # doc_id falls back to the file stem
    (tmp_path / "book.txt").write_text(bench.text)
    code, out, err = _main([str(tmp_path / "qs.jsonl"), "--doc", str(tmp_path / "book.txt"),
                            "--systems", "truncation", "--out", str(tmp_path / "r.csv"), "--quiet"],
                           llm=ReaderLLM(items))
    assert code == 0, err
    loaded, docs = load_datasets([str(tmp_path / "qs.jsonl")], str(tmp_path / "book.txt"))
    assert {it.doc_id for it in loaded} == {"qs"} and docs["qs"] == bench.text


def test_cli_bad_usage(dataset, tmp_path):
    with pytest.raises(SystemExit) as e:
        _main([str(dataset), "--systems", "bm25"])
    assert e.value.code == 2
    code, _, err = _main([str(dataset), "--prices", "cheap"], llm=ScriptedLLM([]))
    assert code == 2 and "--prices" in err
    code, _, err = _main([str(dataset), "--types", "numeric", "--limit", "0"], llm=ScriptedLLM([]))
    assert code == 2 and "no questions" in err


def test_cli_backend_failure_exit_code(dataset, tmp_path):
    code, _, err = _main([str(dataset), "--systems", "truncation", "--max-errors", "0", "--quiet",
                          "--out", str(tmp_path / "x.csv")], llm=_Flaky(5))
    assert code == 1 and "aborting" in err and not (tmp_path / "x.csv").exists()


def test_openai_backend_through_cli(dataset, tmp_path, monkeypatch):
    """The real backend class over a mocked transport: tokens come from reported usage."""
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "not found"}}],
                                          "usage": {"prompt_tokens": 50, "completion_tokens": 2}})

    monkeypatch.delenv("DOLPHIN_API_KEY", raising=False)
    llm = OpenAICompatLLM(model="m", transport=httpx.MockTransport(handler))
    code, out, err = _main([str(dataset), "--systems", "truncation,rag", "--embedder", "hashing", "--quiet",
                            "--limit", "3", "--out", str(tmp_path / "o.csv")], llm=llm)
    assert code == 0, err
    rows = list(csv.DictReader((tmp_path / "o.csv").open()))
    assert {r["total_tokens"] for r in rows} == {"52"}
    assert "estimated" not in out


def test_script_help():
    res = subprocess.run([sys.executable, str(ROOT / "run_agent_eval.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0 and "--systems" in res.stdout
