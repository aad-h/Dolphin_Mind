"""Synthetic long-document QA generator.

Plants made-up facts about invented people and places into a long text (a real
public-domain book or the needle haystack) at random sentence boundaries and
writes questions whose answers depend on them:

* ``single_hop``   one planted sentence answers the question
* ``multi_hop``    two sentences linked by a bridge entity, planted independently
* ``numeric``      two quantities that must be added or subtracted
* ``unanswerable`` the entity is mentioned, but the asked-for fact never is

Every name is invented and checked against the source text, and every entity is
used by one question only, so no planted fact can answer a different question.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import QUESTION_TYPES, UNANSWERABLE, EvalItem

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "qu", "r", "s", "t", "th", "v", "w", "z"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou"]
_CODAS = ["", "n", "r", "l", "s", "th", "m", "nd", "rk", "x"]
_PLACE_SUFFIXES = ["holm", "gate", "ford", "mere", "wick", "dale", "stead", "moor"]

_ROLES = ["lighthouse keeper", "harbour master", "chief archivist", "head brewer", "bridge warden", "royal cartographer"]
_ANIMALS = ["raven", "mule", "tortoise", "falcon", "goat", "ferret"]
_GOODS = [("sacks of flour", "mill"), ("barrels of cider", "press"), ("bolts of cloth", "loom house"),
          ("crates of salt", "salt works")]
_SEASONS = ["spring", "summer", "autumn", "winter"]


@dataclass
class _Plan:
    """A question plus the fact sentences to plant (evidence) and extra mentions (distractors)."""

    type: str
    question: str
    answer: str
    evidence: List[str]
    distractors: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class SyntheticBenchmark:
    """A document with planted facts and the questions about them."""

    doc_id: str
    text: str
    items: List[EvalItem]


class _Names:
    """Pronounceable invented names, unique within a benchmark and absent from the source text."""

    def __init__(self, rng: random.Random, source_text: str):
        self.rng = rng
        self.used: set[str] = set()  # every name and name part handed out so far
        self.source_words = set(re.findall(r"[a-z]+", source_text.lower()))

    def _syllables(self, n: int) -> str:
        return "".join(self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS) + self.rng.choice(_CODAS)
                       for _ in range(n))

    def _fresh(self, make: Callable[[], str]) -> str:
        for _ in range(1000):
            name = make()
            parts = name.split()
            if (len(name) > 3 and name not in self.used and not any(p in self.used for p in parts)
                    and not any(p.lower() in self.source_words for p in parts)):
                self.used.update([name, *parts])
                return name
        raise RuntimeError("could not generate a fresh name")

    def word(self) -> str:
        return self._fresh(lambda: self._syllables(2).capitalize())

    def person(self) -> str:
        return self._fresh(lambda: f"{self._syllables(1).capitalize()}{self.rng.choice(['a', 'o', 'is'])} "
                                   f"{self._syllables(2).capitalize()}")

    def place(self) -> str:
        return self._fresh(lambda: self._syllables(1).capitalize() + self.rng.choice(_PLACE_SUFFIXES))


# -- question templates ----------------------------------------------------------

def _single_hop(rng: random.Random, n: _Names) -> _Plan:
    kind = rng.randrange(3)
    if kind == 0:
        role, place, person = rng.choice(_ROLES), n.place(), n.person()
        return _Plan("single_hop", f"Who is the {role} of {place}?", person,
                     [f"The {role} of {place} is {person}."])
    if kind == 1:
        person, animal, pet = n.person(), rng.choice(_ANIMALS), n.word()
        return _Plan("single_hop", f"What is the name of {person}'s {animal}?", pet,
                     [f"{person} keeps a {animal} named {pet}."])
    person, town = n.person(), n.place()
    return _Plan("single_hop", f"In which town was {person} born?", town, [f"{person} was born in {town}."])


def _multi_hop(rng: random.Random, n: _Names) -> _Plan:
    role, place, person = rng.choice(_ROLES), n.place(), n.person()
    bridge = f"The {role} of {place} is {person}."
    if rng.random() < 0.5:
        town = n.place()
        return _Plan("multi_hop", f"In which town was the {role} of {place} born?", town,
                     [bridge, f"{person} was born in {town}."], metadata={"bridge": person})
    animal, pet = rng.choice(_ANIMALS), n.word()
    return _Plan("multi_hop", f"What is the name of the {animal} kept by the {role} of {place}?", pet,
                 [bridge, f"{person} keeps a {animal} named {pet}."], metadata={"bridge": person})


def _numeric(rng: random.Random, n: _Names) -> _Plan:
    goods, works = rng.choice(_GOODS)
    place = n.place()
    s1, s2 = rng.sample(_SEASONS, 2)
    a, b = rng.randint(20, 900), rng.randint(20, 900)
    while b == a:
        b = rng.randint(20, 900)
    facts = [f"In {s1}, the {works} of {place} produced {a} {goods}.",
             f"In {s2}, the {works} of {place} produced {b} {goods}."]
    if rng.random() < 0.5:
        q = f"How many {goods} did the {works} of {place} produce in {s1} and {s2} combined?"
        return _Plan("numeric", q, str(a + b), facts, metadata={"op": "sum", "operands": [a, b]})
    hi, lo = (s1, s2) if a > b else (s2, s1)
    q = f"How many more {goods} did the {works} of {place} produce in {hi} than in {lo}?"
    return _Plan("numeric", q, str(abs(a - b)), facts, metadata={"op": "difference", "operands": [a, b]})


def _unanswerable(rng: random.Random, n: _Names) -> _Plan:
    """Mention the entity (so keyword search finds something) but never plant the asked-for fact."""
    if rng.random() < 0.5:
        person, town, animal = n.person(), n.place(), rng.choice(_ANIMALS)
        return _Plan("unanswerable", f"What is the name of {person}'s {animal}?", UNANSWERABLE, [],
                     distractors=[f"{person} was born in {town}."])
    goods, works = rng.choice(_GOODS)
    place = n.place()
    s1, s2 = rng.sample(_SEASONS, 2)
    return _Plan("unanswerable", f"How many {goods} did the {works} of {place} produce in {s2}?", UNANSWERABLE, [],
                 distractors=[f"In {s1}, the {works} of {place} produced {rng.randint(20, 900)} {goods}."])


_TEMPLATES: Dict[str, Callable[[random.Random, _Names], _Plan]] = {
    "single_hop": _single_hop,
    "multi_hop": _multi_hop,
    "numeric": _numeric,
    "unanswerable": _unanswerable,
}


# -- planting ------------------------------------------------------------------

def plant_sentences(text: str, sentences: Sequence[str], rng: random.Random) -> Tuple[str, List[float]]:
    """Insert each sentence after a random sentence boundary of ``text``, keeping its original whitespace.

    Returns the new text and each sentence's depth (0..1, by character offset in the original text).
    Sentences are never inserted inside one another.
    """
    boundaries = [m.start() for m in _BOUNDARY.finditer(text)] or [len(text)]
    offsets = [rng.choice(boundaries) for _ in sentences]
    depths = [off / max(len(text), 1) for off in offsets]
    # insert from the back so earlier offsets stay valid; ties keep the given order
    out = text
    for off, sent in sorted(zip(offsets, sentences), key=lambda p: p[0], reverse=True):
        out = out[:off] + " " + sent + out[off:]
    return out, depths


def generate(
    text: str,
    n_questions: int = 40,
    types: Sequence[str] = QUESTION_TYPES,
    seed: int = 0,
    doc_id: str = "doc",
    id_prefix: Optional[str] = None,
) -> SyntheticBenchmark:
    """Plant facts into ``text`` and return the document plus ``n_questions`` items.

    Question types are assigned round-robin from ``types`` so the mix is balanced.
    The same ``seed`` and inputs always produce the same benchmark.
    """
    if n_questions < 1:
        raise ValueError("n_questions must be >= 1")
    unknown = [t for t in types if t not in _TEMPLATES]
    if unknown or not types:
        raise ValueError(f"types must be a non-empty subset of {QUESTION_TYPES}, got {list(types)}")
    if not text.strip():
        raise ValueError("source text is empty")

    rng = random.Random(seed)
    names = _Names(rng, text)
    plans = [_TEMPLATES[types[i % len(types)]](rng, names) for i in range(n_questions)]

    sentences = [s for p in plans for s in p.evidence + p.distractors]
    doc, depths = plant_sentences(text, sentences, rng)

    prefix = id_prefix if id_prefix is not None else f"{doc_id}-"
    items: List[EvalItem] = []
    pos = 0
    for i, p in enumerate(plans):
        k = len(p.evidence) + len(p.distractors)
        meta = dict(p.metadata, depths=[round(d, 4) for d in depths[pos: pos + k]])
        if p.distractors:
            meta["distractors"] = list(p.distractors)
        pos += k
        items.append(EvalItem(id=f"{prefix}{i:04d}", question=p.question, answer=p.answer, type=p.type,
                              evidence=list(p.evidence), doc_id=doc_id, metadata=meta))
    return SyntheticBenchmark(doc_id=doc_id, text=doc, items=items)
