"""Command-line entry point: ``python -m dolphinmind.agent "question" --doc book.txt``.

Backends:
  scripted  replay canned JSON replies from ``--script`` (offline demos and tests)
  openai    any OpenAI-compatible endpoint; key from ``DOLPHIN_API_KEY`` only
  hf        local Hugging Face model in 4-bit (needs a CUDA GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional, TextIO

from ..retrieval import SemanticRetriever
from .llm import HFChatLLM, LLMError, OpenAICompatLLM, ScriptedLLM
from .react import AgentResult, ReActAgent, Step
from .tools import Document, Toolbox
from .tracing import Tracer, format_trace

BACKENDS = ("scripted", "openai", "hf")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dolphinmind.agent",
        description="Ask a question about a long document with the DolphinAgent ReAct loop.",
    )
    p.add_argument("question", help="question to answer")
    p.add_argument("--doc", required=True, help="path to a UTF-8 text file, or '-' for stdin")
    p.add_argument("--backend", choices=BACKENDS, default="openai", help="chat backend (default: openai)")
    p.add_argument("--model", help="model name (openai: DOLPHIN_MODEL or gpt-4o-mini; hf: Qwen2.5-3B-Instruct)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: DOLPHIN_BASE_URL or OpenAI)")
    p.add_argument("--script", help="scripted backend: JSON file holding a list of model replies")
    p.add_argument("--max-steps", type=int, default=6, help="tool-call budget (default: 6)")
    p.add_argument("--chunk-size", type=int, default=256, help="words per chunk (default: 256)")
    p.add_argument("--overlap", type=int, default=64, help="overlapping words between chunks (default: 64)")
    p.add_argument("--semantic", action="store_true",
                   help="enable the search_doc tool (loads a sentence-transformers model)")
    p.add_argument("--trace", action="store_true", help="print every thought, tool call and observation")
    p.add_argument("--spans", action="store_true",
                   help="print the timed span tree (agent run, LLM and tool calls) to stderr")
    p.add_argument("--json", action="store_true", dest="as_json", help="print the result as JSON")
    return p


def _load_script(path: str) -> List[str]:
    """Read scripted replies: a JSON list whose items are strings or action objects."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("--script must contain a non-empty JSON list")
    return [r if isinstance(r, str) else json.dumps(r) for r in data]


def build_llm(args: argparse.Namespace) -> Any:
    """Instantiate the chat backend selected on the command line."""
    if args.backend == "scripted":
        if not args.script:
            raise ValueError("--backend scripted needs --script replies.json")
        return ScriptedLLM(_load_script(args.script))
    if args.backend == "openai":
        return OpenAICompatLLM(model=args.model, base_url=args.base_url)
    from ..models import QWEN, load_qwen

    model, tokenizer = load_qwen(args.model or QWEN)
    return HFChatLLM(model, tokenizer)


def _read_doc(path: str, stdin: TextIO) -> str:
    if path == "-":
        return stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def _snippet(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def format_result(result: AgentResult, doc: Document, trace: bool = False, usage: Any = None) -> str:
    """Human-readable report: optional step trace, then answer, citations and token usage."""
    lines: List[str] = []
    if trace:
        lines.append("Trace")
        for i, s in enumerate(result.steps, 1):
            head = f"  [{i}] {s.tool}({json.dumps(s.args)})" if s.tool else f"  [{i}] answer"
            lines.append(f"{head}  {s.seconds:.2f}s")
            if s.thought:
                lines.append(f"      thought: {_snippet(s.thought)}")
            if s.tool:
                lines.append(f"      observation: {_snippet(s.observation, 240)}")
        lines.append("")
    lines.append(f"Answer: {result.answer}")
    if result.citations:
        lines.append("Citations:")
        lines += [f"  [chunk {c}] {_snippet(doc.chunks[c])}" for c in result.citations]
    else:
        lines.append("Citations: none")
    summary = f"Stopped: {result.stopped} | tool calls: {result.tool_calls}"
    if usage is not None and getattr(usage, "calls", 0):
        summary += f" | tokens: {usage.total_tokens} ({usage.prompt_tokens} in / {usage.completion_tokens} out)"
    lines.append(summary)
    return "\n".join(lines)


def step_to_dict(step: Step) -> dict:
    """JSON-ready view of one agent step (shared by ``--json`` and the HTTP service)."""
    return {"thought": step.thought, "tool": step.tool, "args": step.args, "observation": step.observation,
            "seconds": round(step.seconds, 4)}


def result_to_dict(result: AgentResult, doc: Document, usage: Any = None) -> dict:
    out = {
        "answer": result.answer,
        "citations": [{"chunk_id": c, "text": doc.chunks[c]} for c in result.citations],
        "stopped": result.stopped,
        "tool_calls": result.tool_calls,
        "steps": [step_to_dict(s) for s in result.steps],
    }
    if usage is not None and getattr(usage, "calls", 0):
        out["usage"] = {"calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
    return out


def main(argv: Optional[List[str]] = None, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
         stderr: TextIO = sys.stderr) -> int:
    """Run the CLI. Returns a process exit code (0 ok, 1 runtime/backend error, 2 bad usage)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.overlap >= args.chunk_size:
        parser.error("--overlap must be smaller than --chunk-size")

    try:
        text = _read_doc(args.doc, stdin)
    except OSError as e:
        print(f"error: cannot read --doc: {e}", file=stderr)
        return 2
    if not text.strip():
        print("error: document is empty", file=stderr)
        return 2

    try:
        llm = build_llm(args)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=stderr)
        return 2

    doc = Document(text, chunk_size=args.chunk_size, overlap=args.overlap)
    retriever = SemanticRetriever() if args.semantic else None
    tracer = Tracer(max_traces=1) if args.spans else None
    try:
        result = ReActAgent(llm, Toolbox(doc, retriever), max_steps=args.max_steps,
                            tracer=tracer).run(args.question)
    except (LLMError, RuntimeError) as e:
        print(f"error: {e}", file=stderr)
        return 1
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()

    if tracer is not None and tracer.traces():
        print(format_trace(tracer.traces()[0]), file=stderr)
    usage = getattr(llm, "usage", None)
    if args.as_json:
        print(json.dumps(result_to_dict(result, doc, usage), indent=2), file=stdout)
    else:
        print(format_result(result, doc, trace=args.trace, usage=usage), file=stdout)
    return 0
