"""Eval dataset schema, JSONL I/O and synthetic generator (offline, no downloads)."""

import json
import random
import re
from collections import Counter

import pytest

from dolphinmind.agent.tools import Document, Toolbox
from dolphinmind.eval import (QUESTION_TYPES, UNANSWERABLE, EvalItem, check_evidence, generate, load_jsonl,
                              plant_sentences, save_jsonl)
from dolphinmind.eval import cli
from dolphinmind.evaluation import make_haystack


def _item(**kw):
    base = dict(id="q1", question="Who?", answer="Ann", type="single_hop", evidence=["Ann did it."])
    base.update(kw)
    return EvalItem(**base)


# -- schema ----------------------------------------------------------------------

def test_item_roundtrip_dict():
    it = _item(doc_id="book", metadata={"depths": [0.5]})
    assert EvalItem.from_dict(it.to_dict()) == it


@pytest.mark.parametrize("kw, msg", [
    (dict(type="trivia"), "type must be"),
    (dict(question=" "), "'question'"),
    (dict(evidence=[]), "at least one evidence"),
    (dict(type="multi_hop"), "two evidence"),
    (dict(type="unanswerable", answer=UNANSWERABLE), "no evidence"),
    (dict(type="unanswerable", evidence=[]), "must have answer"),
    (dict(evidence=["ok", ""]), "non-empty strings"),
])
def test_item_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _item(**kw)


def test_unanswerable_item_ok():
    assert _item(type="unanswerable", answer=UNANSWERABLE, evidence=[]).evidence == []


def test_from_dict_rejects_missing_and_unknown_fields():
    with pytest.raises(ValueError, match="missing field"):
        EvalItem.from_dict({"id": "x", "question": "q"})
    with pytest.raises(ValueError, match="unknown field"):
        EvalItem.from_dict(dict(_item().to_dict(), extra=1))


def test_jsonl_roundtrip_and_errors(tmp_path):
    items = [_item(id="a"), _item(id="b", type="numeric", answer="3", evidence=["1 and 2."])]
    p = tmp_path / "sub" / "ds.jsonl"
    assert save_jsonl(items, p) == 2
    p.write_text(p.read_text() + "\n", encoding="utf-8")  # trailing blank line is fine
    assert load_jsonl(p) == items

    with pytest.raises(ValueError, match="duplicate id"):
        save_jsonl([_item(), _item()], tmp_path / "dup.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(_item().to_dict()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_jsonl(bad)


def test_check_evidence():
    items = [_item(doc_id="d"), _item(id="q2", doc_id="nope"), _item(id="q3", doc_id="d", evidence=["Bob."])]
    problems = check_evidence(items, {"d": "Long text. Ann did it. More."})
    assert len(problems) == 2
    assert "unknown doc_id" in problems[0] and "q3" in problems[1]


# -- generator -------------------------------------------------------------------

def test_plant_sentences_preserves_text_and_whitespace():
    text = "One.\nTwo!  Three?\n\nFour."
    out, depths = plant_sentences(text, ["Fact A.", "Fact B."], random.Random(1))
    assert "Fact A." in out and "Fact B." in out
    assert re.sub(r" Fact [AB]\.", "", out) == text
    assert all(0 <= d <= 1 for d in depths)


def test_plant_sentences_without_boundaries_appends():
    out, depths = plant_sentences("no punctuation here", ["Fact."], random.Random(0))
    assert out == "no punctuation here Fact." and depths == [1.0]


def test_generate_balanced_valid_and_grounded():
    text = make_haystack(6_000)
    bench = generate(text, n_questions=40, seed=3, doc_id="hay")
    assert Counter(it.type for it in bench.items) == {t: 10 for t in QUESTION_TYPES}
    assert check_evidence(bench.items, {"hay": bench.text}) == []
    assert len({it.id for it in bench.items}) == 40
    assert len({it.question for it in bench.items}) == 40
    for it in bench.items:
        it.validate()
        if it.type == "unanswerable":
            assert it.answer == UNANSWERABLE
            assert all(d in bench.text for d in it.metadata["distractors"])
        else:
            # the answer is stated in the planted evidence, never in the original text
            assert it.answer.lower() not in text.lower()
        if it.type == "multi_hop":
            assert len(it.evidence) == 2 and it.metadata["bridge"] in it.evidence[0]
        if it.type == "numeric":
            a, b = it.metadata["operands"]
            assert int(it.answer) == (a + b if it.metadata["op"] == "sum" else abs(a - b))
            assert all(str(x) in " ".join(it.evidence) for x in (a, b))


def test_unanswerable_entity_is_mentioned_but_fact_absent():
    bench = generate(make_haystack(3_000), n_questions=12, types=["unanswerable"], seed=5)
    for it in bench.items:
        distractor = it.metadata["distractors"][0]
        # the entity the question names appears in the document, via its distractor sentence
        assert any(w in it.question for w in re.findall(r"[A-Z][a-z]+", distractor) if w not in ("In", "The"))
        assert distractor in bench.text


def test_generate_is_deterministic_and_seed_sensitive():
    text = make_haystack(2_000)
    a, b = generate(text, 8, seed=7), generate(text, 8, seed=7)
    assert a.text == b.text and a.items == b.items
    assert generate(text, 8, seed=8).items != a.items


def test_generate_validates_arguments():
    with pytest.raises(ValueError):
        generate("Some text.", 0)
    with pytest.raises(ValueError):
        generate("Some text.", 4, types=["trivia"])
    with pytest.raises(ValueError):
        generate("   ", 4)


def test_planted_facts_are_findable_by_agent_tools():
    """End-to-end sanity: evidence survives chunking and grep on the bridge entity finds both hops."""
    bench = generate(make_haystack(5_000), n_questions=4, types=["multi_hop"], seed=2)
    tb = Toolbox(Document(bench.text))
    for it in bench.items:
        obs = tb.grep(it.metadata["bridge"], k=5)
        assert all(ev in obs for ev in it.evidence)


# -- CLI -------------------------------------------------------------------------

def test_cli_writes_doc_and_dataset(tmp_path, capsys):
    book = tmp_path / "moby.txt"
    book.write_text(make_haystack(3_000), encoding="utf-8")
    out = tmp_path / "data"
    assert cli.main([str(book), "--haystack", "2000", "--out", str(out), "-n", "6",
                     "--types", "single_hop,numeric"]) == 0
    items = load_jsonl(out / "moby.jsonl")
    assert len(items) == 6 and {it.type for it in items} == {"single_hop", "numeric"}
    assert check_evidence(items, {"moby": (out / "moby.txt").read_text(encoding="utf-8")}) == []
    assert (out / "haystack2000.jsonl").exists()
    assert "moby: 6 questions" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--out", "x"], ["--haystack", "100", "--out", "x", "--types", "bogus"],
                                  ["missing.txt", "--out", "x"], ["--haystack", "100", "--out", "x", "-n", "0"]])
def test_cli_bad_usage(argv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
