"""Compare truncation, plain RAG, DolphinMind retrieval and the ReAct agent on a QA benchmark.

    python -m dolphinmind.eval --haystack 20000 --out data/ -n 40      # make a dataset
    python run_agent_eval.py data/haystack20000.jsonl --model gpt-4o-mini --prices 0.15,0.60

Writes results/agent_eval.csv (one row per system x question) and results/agent_eval.md
(markdown summary with bootstrap 95% CIs). The API key is read from DOLPHIN_API_KEY.
See ``dolphinmind/eval/runner.py`` for system definitions and all flags (``--help``).
"""

import sys

from dolphinmind.eval.runner import main

if __name__ == "__main__":
    sys.exit(main())
