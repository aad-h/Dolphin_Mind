import numpy as np
import pytest

from dolphinmind.evaluation import NEEDLES, check_answer, cohens_d, insert_needle, make_haystack
from dolphinmind.retrieval import SemanticRetriever, chunk_sentences


def test_haystack_length():
    assert len(make_haystack(10_000).split()) == 10_000


def test_insert_needle_depth():
    hay = make_haystack(1_000)
    doc = insert_needle(hay, "NEEDLE", 50)
    assert doc.split().index("NEEDLE") == 500
    with pytest.raises(ValueError):
        insert_needle(hay, "x", 120)


def test_chunks_cover_text_and_overlap():
    text = make_haystack(5_000)
    chunks = chunk_sentences(text, chunk_size=512, overlap=128)
    assert len(chunks) > 1
    assert all(len(c.split()) >= 512 for c in chunks[:-1])
    # consecutive chunks share text
    assert " ".join(chunks[0].split()[-20:]) in chunks[1]
    # needle survives chunking
    doc = insert_needle(text, NEEDLES[0]["text"], 50)
    assert any("ALPHA7392BETA" in c for c in chunk_sentences(doc))


def test_overlap_validation():
    with pytest.raises(ValueError):
        chunk_sentences("a. b.", chunk_size=10, overlap=10)


def test_check_answer():
    assert check_answer("The code is ALPHA7392BETA.", "ALPHA7392BETA", "exact_match")
    assert not check_answer("I don't know", "ALPHA7392BETA", "exact_match")
    assert check_answer("osmium and rhodium", "osmium and rhodium", "multi_fact")
    assert not check_answer("only iron", "osmium and rhodium", "multi_fact")


def test_cohens_d():
    assert cohens_d([1, 1, 0, 1], [0, 0, 1, 0]) > 0
    assert cohens_d([1, 1], [1, 1]) == 0.0


class _BagOfWords:
    """Tiny stand-in for SentenceTransformer so retrieval is testable without downloads."""

    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def encode(self, texts, **_):
        out = np.zeros((len(texts), len(self.vocab)), dtype="float32")
        for r, t in enumerate(texts):
            for w in t.lower().split():
                if w in self.vocab:
                    out[r, self.vocab[w]] += 1
        return out + 1e-3


def test_retriever_finds_needle_in_order():
    pytest.importorskip("faiss")
    text = insert_needle(make_haystack(4_000), NEEDLES[0]["text"], 70)
    r = SemanticRetriever(embedding_model=_BagOfWords(["authentication", "code", "neural"]))
    chunks = chunk_sentences(text, 256, 64)
    r.build(chunks)
    hits = r.retrieve("authentication code", top_k=3)
    assert any("ALPHA7392BETA" in h for h in hits)
    positions = [chunks.index(h) for h in hits]
    assert positions == sorted(positions)
