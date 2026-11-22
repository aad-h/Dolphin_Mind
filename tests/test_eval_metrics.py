"""Eval metrics: answer matching, citation faithfulness, bootstrap CIs, summaries (offline)."""

import json
import math

import numpy as np
import pytest

from dolphinmind.agent import Document, ReActAgent, ScriptedLLM, Toolbox
from dolphinmind.eval import (UNANSWERABLE, EvalItem, QuestionScore, bootstrap_ci, citation_precision,
                              evidence_recall, exact_match, format_summary, fuzzy_match, generate, is_abstention,
                              normalize_answer, score_agent_result, score_question, summarize, token_f1)
from dolphinmind.eval.metrics import extract_numbers
from dolphinmind.evaluation import make_haystack


def _item(**kw):
    base = dict(id="q1", question="Where was Ann born?", answer="Tuholm", type="single_hop",
                evidence=["Ann was born in Tuholm."])
    base.update(kw)
    return EvalItem(**base)


# -- normalization ------------------------------------------------------------------

@pytest.mark.parametrize("text, norm", [
    ("The Tuholm.", "tuholm"),
    ("  An  OLD   mill! ", "old mill"),
    ("1,200.0 sacks", "1200 sacks"),
    ("ALPHA7392BETA", "alpha7392beta"),
    ("QE-9X7K-MM4P-ZZ82", "qe 9x7k mm4p zz82"),
    ("2.5 atmospheres", "2.5 atmospheres"),
])
def test_normalize_answer(text, norm):
    assert normalize_answer(text) == norm


def test_extract_numbers_ignores_codes():
    assert extract_numbers("made 1,250 then 3.5, code A7392") == [1250.0, 3.5]
    assert extract_numbers("812.") == [812.0]


@pytest.mark.parametrize("text", ["not found", "Not found.", "I don't know", "I don’t know",
                                  "The document does not say.", "Unknown", "", "  ",
                                  "This is not mentioned in the text"])
def test_abstentions(text):
    assert is_abstention(text)


@pytest.mark.parametrize("text", ["Tuholm", "812", "He was born in a small town"])
def test_non_abstentions(text):
    assert not is_abstention(text)


# -- accuracy -----------------------------------------------------------------------

def test_exact_match():
    assert exact_match("tuholm.", "Tuholm")
    assert exact_match("The Tuholm", "Tuholm")
    assert not exact_match("born in Tuholm", "Tuholm")
    assert exact_match("1,200", "1200") and exact_match("812.0", "812")
    assert not exact_match("813", "812")


def test_fuzzy_match():
    assert fuzzy_match("He was born in Tuholm.", "Tuholm")
    assert not fuzzy_match("He was born in Tuholmgate.", "Tuholm")  # whole tokens only
    assert fuzzy_match("osmium and rhodium compounds", "osmium and rhodium")
    assert not fuzzy_match("Tuholm", "Daendis Draemmork")


def test_numeric_fuzzy_needs_a_single_matching_number():
    assert fuzzy_match("They produced 812 sacks in total.", "812")
    assert not fuzzy_match("Either 400 or 812", "812")  # hedging across numbers doesn't count
    assert not fuzzy_match("8120", "812")


def test_unanswerable_and_abstention_scoring():
    assert exact_match("I cannot find this in the document", UNANSWERABLE)
    assert fuzzy_match("not found", UNANSWERABLE)
    assert not exact_match("Rex", UNANSWERABLE) and not fuzzy_match("Rex", UNANSWERABLE)
    # abstaining on an answerable question is always wrong
    assert not exact_match("not found", "Tuholm")
    assert not fuzzy_match("unknown", "Tuholm")


def test_token_f1():
    assert token_f1("osmium and rhodium", "osmium and rhodium") == 1.0
    assert token_f1("osmium", "osmium and rhodium") == pytest.approx(0.5)
    assert token_f1("lead", "osmium") == 0.0
    assert token_f1("", "") == 1.0


# -- citations ----------------------------------------------------------------------

def test_evidence_recall_and_precision():
    ev = ["A was born in X.", "A keeps a goat."]
    chunks = ["noise. A was  born\nin X. more", "unrelated text", "Here A keeps a goat. end"]
    assert evidence_recall(ev, chunks) == 1.0
    assert evidence_recall(ev, chunks[:2]) == 0.5
    assert evidence_recall(ev, []) == 0.0
    assert evidence_recall([], chunks) is None
    assert citation_precision(ev, chunks) == pytest.approx(2 / 3)
    assert citation_precision(ev, []) is None
    assert citation_precision([], chunks) is None


def test_score_question_fields_and_cost():
    s = score_question(_item(), "Tuholm", ["Ann was born in Tuholm. Then"], [4], tool_calls=2,
                       prompt_tokens=1000, completion_tokens=100, latency_s=1.5, prices=(0.15, 0.60), system="x")
    assert s.exact and s.fuzzy and s.f1 == 1.0
    assert s.evidence_recall == 1.0 and s.citation_precision == 1.0
    assert s.total_tokens == 1100 and s.cost_usd == pytest.approx(0.00021)
    row = s.to_dict()
    assert row["total_tokens"] == 1100 and row["citations"] == [4] and row["system"] == "x"
    json.dumps(row)  # CSV/JSON friendly

    u = score_question(_item(type="unanswerable", answer=UNANSWERABLE, evidence=[]), "not found")
    assert u.exact and u.f1 == 1.0 and u.evidence_recall is None and u.cost_usd is None
    wrong = score_question(_item(), "not found")
    assert not wrong.exact and wrong.f1 == 0.0 and wrong.evidence_recall == 0.0


def test_score_agent_result_end_to_end():
    """Synthetic benchmark -> scripted ReAct agent -> scored record with faithful citations."""
    bench = generate(make_haystack(4_000), n_questions=4, seed=3, doc_id="hay", types=["single_hop"])
    item = bench.items[0]
    doc = Document(bench.text)
    toolbox = Toolbox(doc)
    keyword = item.evidence[0].split()[-1].rstrip(".")
    cid = next(i for i, c in enumerate(doc.chunks) if item.evidence[0] in c)
    llm = ScriptedLLM([
        json.dumps({"thought": "search", "tool": "grep", "args": {"pattern": keyword, "k": 50}}),
        json.dumps({"thought": "done", "answer": item.answer, "citations": [cid]}),
    ])
    res = ReActAgent(llm, toolbox).run(item.question)
    s = score_agent_result(item, res, doc.chunks, usage={"prompt_tokens": 50, "completion_tokens": 5})
    assert s.exact and s.evidence_recall == 1.0 and s.citation_precision == 1.0
    assert s.tool_calls == 1 and s.total_tokens == 55 and s.system == "react"
    assert s.latency_s >= 0


# -- bootstrap & summary -------------------------------------------------------------

def test_bootstrap_ci_basic_properties():
    rng = np.random.default_rng(0)
    vals = rng.binomial(1, 0.7, size=400)
    est = bootstrap_ci(vals, seed=1)
    assert est.n == 400 and est.lo < est.mean < est.hi
    assert est.mean == pytest.approx(vals.mean())
    # ~ normal-approx half width 1.96*sqrt(p(1-p)/n) ~= 0.045
    assert 0.03 < (est.hi - est.lo) / 2 < 0.06
    assert bootstrap_ci(vals, seed=1) == est  # deterministic per seed
    narrow = bootstrap_ci(vals, seed=1, confidence=0.5)
    assert narrow.hi - narrow.lo < est.hi - est.lo


def test_bootstrap_ci_edge_cases():
    empty = bootstrap_ci([None, float("nan")])
    assert empty.n == 0 and math.isnan(empty.mean) and empty.fmt() == "n/a"
    one = bootstrap_ci([0.25])
    assert (one.mean, one.lo, one.hi, one.n) == (0.25, 0.25, 0.25, 1)
    const = bootstrap_ci([1, 1, 1])
    assert const.lo == const.hi == 1.0
    med = bootstrap_ci([1, 2, 3, 100], stat=np.median)
    assert med.mean == 2.5 and med.lo <= 2.5 <= med.hi
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], confidence=1.0)


def _score(id, type, exact, tokens=100, recall=1.0):
    return QuestionScore(id=id, type=type, gold="g", prediction="p", exact=exact, fuzzy=exact,
                         f1=float(exact), evidence_recall=recall, prompt_tokens=tokens)


def test_summarize_groups_and_table():
    scores = [_score("a", "single_hop", True), _score("b", "single_hop", False),
              _score("c", "numeric", True, tokens=300),
              _score("d", "unanswerable", True, recall=None)]
    summ = summarize(scores, n_boot=200)
    assert list(summ) == ["all", "single_hop", "numeric", "unanswerable"]
    assert summ["all"]["exact"].mean == 0.75 and summ["all"]["exact"].n == 4
    assert summ["all"]["evidence_recall"].n == 3  # unanswerable excluded
    assert summ["numeric"]["total_tokens"].mean == 300
    assert summ["all"]["cost_usd"].n == 0
    assert list(summarize(scores, by_type=False)) == ["all"]

    table = format_summary(summ)
    lines = table.splitlines()
    assert lines[0].startswith("| group | n | exact |") and len(lines) == 2 + 4
    assert "| all | 4 | 75.0 [" in table
    assert "| unanswerable | 1 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | n/a |" in table
