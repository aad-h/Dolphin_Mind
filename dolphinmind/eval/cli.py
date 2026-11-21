"""Generate a synthetic benchmark: ``python -m dolphinmind.eval book.txt --out data/ -n 100``.

Writes ``<out>/<doc_id>.txt`` (the book with planted facts) and ``<out>/<doc_id>.jsonl``
(the questions). Several books give one document + dataset pair each.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, TextIO

from ..evaluation import make_haystack
from .dataset import QUESTION_TYPES, check_evidence, save_jsonl
from .generator import generate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dolphinmind.eval",
                                description="Plant facts into long texts and write a JSONL QA benchmark.")
    p.add_argument("docs", nargs="*", help="UTF-8 text files to plant facts into")
    p.add_argument("--haystack", type=int, metavar="WORDS",
                   help="use a synthetic haystack of this many words instead of (or as well as) files")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("-n", "--questions", type=int, default=40, help="questions per document (default: 40)")
    p.add_argument("--types", default=",".join(QUESTION_TYPES),
                   help=f"comma-separated question types (default: {','.join(QUESTION_TYPES)})")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return p


def main(argv: Optional[List[str]] = None, stdout: Optional[TextIO] = None) -> int:
    """Run the generator. Returns 0 on success; argparse exits with 2 on bad usage."""
    stdout = stdout or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    if not args.docs and not args.haystack:
        parser.error("give at least one document or --haystack WORDS")
    if bad := [t for t in types if t not in QUESTION_TYPES]:
        parser.error(f"unknown type(s): {', '.join(bad)}")
    if args.questions < 1:
        parser.error("--questions must be >= 1")

    sources = []
    for path in args.docs:
        try:
            sources.append((Path(path).stem, Path(path).read_text(encoding="utf-8")))
        except OSError as e:
            parser.error(f"cannot read {path}: {e.strerror or e}")
    if args.haystack:
        sources.append((f"haystack{args.haystack}", make_haystack(args.haystack)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (doc_id, text) in enumerate(sources):
        bench = generate(text, args.questions, types, seed=args.seed + i, doc_id=doc_id)
        problems = check_evidence(bench.items, {doc_id: bench.text})
        if problems:  # should be impossible; guards the generator itself
            print("\n".join(problems), file=sys.stderr)
            return 1
        (out / f"{doc_id}.txt").write_text(bench.text, encoding="utf-8")
        save_jsonl(bench.items, out / f"{doc_id}.jsonl")
        mix = ", ".join(f"{t}={c}" for t, c in sorted(Counter(it.type for it in bench.items).items()))
        print(f"{doc_id}: {len(bench.items)} questions ({mix}) -> {out / doc_id}.jsonl", file=stdout)
    return 0
