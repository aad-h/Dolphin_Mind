"""Reproduce the DolphinMind needle-in-a-haystack results.

    python run_benchmark.py                       # full run (Truncated, DolphinMind, SmolLM3 4k/64k)
    python run_benchmark.py --quick               # 2 needles, 10k words, 1 depth: smoke test
    python run_benchmark.py --methods dolphinmind # skip SmolLM3 (saves ~2.5 GB VRAM)

Requires a CUDA GPU (4-bit bitsandbytes). Writes results/needle_results.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dolphinmind.evaluation import NEEDLES, check_answer, cohens_d, insert_needle, make_haystack, set_seed
from dolphinmind.models import ask_smollm, generate, load_qwen, load_smollm, truncate_tokens
from dolphinmind.retrieval import SemanticRetriever

PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["truncated", "dolphinmind", "smollm"],
                   choices=["truncated", "dolphinmind", "smollm"])
    p.add_argument("--context-lengths", nargs="+", type=int, default=[10_000, 40_000], help="haystack sizes in words")
    p.add_argument("--depths", nargs="+", type=int, default=[10, 50, 90])
    p.add_argument("--smol-windows", nargs="+", type=int, default=[4_000, 64_000])
    p.add_argument("--max-context-tokens", type=int, default=3_500, help="token budget for retrieved context")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results/needle_results.csv")
    args = p.parse_args()

    import pandas as pd
    from scipy import stats

    needles = NEEDLES
    if args.quick:
        needles, args.context_lengths, args.depths = NEEDLES[:2], [10_000], [50]

    set_seed(args.seed)
    qwen, qtok = load_qwen()
    retriever = SemanticRetriever() if "dolphinmind" in args.methods else None
    smol = load_smollm() if "smollm" in args.methods else None

    rows = []

    def record(method, window, needle, ctx_len, depth, answer, tokens, elapsed):
        rows.append({
            "method": method, "window": window, "category": needle["category"],
            "difficulty": needle["difficulty"], "context_len": ctx_len, "depth": depth,
            "question": needle["question"], "expected": needle["answer"], "prediction": answer,
            "correct": check_answer(answer, needle["answer"], needle["category"]),
            "tokens": tokens, "time": elapsed,
        })

    for n_i, needle in enumerate(needles, 1):
        print(f"needle {n_i}/{len(needles)} ({needle['category']})")
        for ctx_len in args.context_lengths:
            haystack = make_haystack(ctx_len)
            for depth in args.depths:
                doc = insert_needle(haystack, needle["text"], depth)
                q = needle["question"]

                if "truncated" in args.methods:
                    t = time.time()
                    ctx = truncate_tokens(qtok, doc, 4_000)
                    ans, toks = generate(qwen, qtok, PROMPT.format(context=ctx, question=q), 30)
                    record("Truncated", "4k", needle, ctx_len, depth, ans, toks, time.time() - t)

                if "dolphinmind" in args.methods:
                    t = time.time()
                    ctx = retriever.build_context(doc, q, top_k=args.top_k)
                    ctx = truncate_tokens(qtok, ctx, args.max_context_tokens)
                    ans, toks = generate(qwen, qtok, PROMPT.format(context=ctx, question=q), 30)
                    record("DolphinMind", "semantic", needle, ctx_len, depth, ans, toks, time.time() - t)

                if smol is not None:
                    for window in args.smol_windows:
                        t = time.time()
                        ans, toks = ask_smollm(*smol, doc, q, window)
                        record("SmolLM3", f"{window // 1000}k", needle, ctx_len, depth, ans, toks, time.time() - t)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = (df.groupby(["method", "window"])
                 .agg(accuracy=("correct", "mean"), tokens=("tokens", "mean"), time_s=("time", "mean"), n=("correct", "size"))
                 .sort_values("accuracy", ascending=False))
    print("\n", summary.round(3).to_string(), sep="")

    dm = df[df.method == "DolphinMind"]["correct"].astype(float)
    if len(dm) > 1:
        for (m, w), grp in df[df.method != "DolphinMind"].groupby(["method", "window"]):
            other = grp["correct"].astype(float)
            if len(other) > 1:
                _, pval = stats.ttest_ind(dm, other)
                print(f"DolphinMind vs {m} ({w}): p = {pval:.4f}, Cohen's d = {cohens_d(dm, other):.3f}")
    print(f"\nsaved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
