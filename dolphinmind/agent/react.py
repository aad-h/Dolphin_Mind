"""ReAct-style agent: think -> call a tool -> observe -> ... -> answer with citations."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .llm import LLM
from .tools import Toolbox
from .tracing import Tracer, maybe_span, traced

SYSTEM_PROMPT = """You answer questions about a long document by calling tools.

Tools:
{tools}

Reply with exactly ONE JSON object per turn and nothing else:
  {{"thought": "...", "tool": "<name>", "args": {{...}}}}
or, when you have enough evidence:
  {{"thought": "...", "answer": "<short answer>", "citations": [<chunk ids you read>]}}

Rules:
- Only cite chunk ids that appeared in tool results.
- If the document does not contain the answer, answer "not found" with no citations.
- You have at most {max_steps} tool calls."""


@dataclass
class Step:
    thought: str
    tool: Optional[str]
    args: dict
    observation: str
    seconds: float


@dataclass
class AgentResult:
    answer: str
    citations: List[int]
    steps: List[Step] = field(default_factory=list)
    stopped: str = "answered"  # answered | max_steps | parse_errors

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> dict:
    """Pull the JSON action out of a model reply, tolerating code fences and chatter around it."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError("no JSON object found")
    # try progressively shorter spans in case the model added trailing braces/text
    candidate = m.group(0)
    while candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        cut = candidate.rfind("}", 0, len(candidate) - 1)
        candidate = candidate[: cut + 1] if cut > 0 else ""
    raise ValueError("invalid JSON")


class ReActAgent:
    """Think -> tool call -> observe loop. With ``tracer`` every run, LLM call and tool call is a span."""

    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 6, max_parse_errors: int = 2,
                 tracer: Optional[Tracer] = None):
        self.tracer = tracer
        self.llm, self.toolbox = traced(llm, tracer), toolbox
        self.max_steps, self.max_parse_errors = max_steps, max_parse_errors

    def run(self, question: str, on_step: Optional[Callable[[Step], None]] = None) -> AgentResult:
        """Answer ``question``; ``on_step`` (if given) is called with each Step as soon as it finishes."""
        with maybe_span(self.tracer, "agent.run", question=question, max_steps=self.max_steps) as span:
            result = self._run(question, on_step)
            if span is not None:
                span.set(answer=result.answer, citations=list(result.citations), stopped=result.stopped,
                         tool_calls=result.tool_calls)
            return result

    def _call_tool(self, name: str, args: dict) -> str:
        with maybe_span(self.tracer, f"tool.{name}", tool=name, args=args) as span:
            before = set(self.toolbox.seen)
            obs = self.toolbox.call(name, args)
            if span is not None:
                span.set(observation_chars=len(obs), new_chunks=sorted(self.toolbox.seen - before),
                         tool_error=obs.startswith("error:"))
            return obs

    def _run(self, question: str, on_step: Optional[Callable[[Step], None]]) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=self.toolbox.specs(), max_steps=self.max_steps)},
            {"role": "user", "content": f"Question: {question}"},
        ]
        steps: List[Step] = []
        parse_errors = 0

        while len([s for s in steps if s.tool]) < self.max_steps:
            t0 = time.time()
            reply = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": reply})
            try:
                action = parse_action(reply)
            except ValueError as e:
                parse_errors += 1
                if parse_errors > self.max_parse_errors:
                    return AgentResult("not found", [], steps, stopped="parse_errors")
                messages.append({"role": "user", "content": f"Invalid reply ({e}). Reply with one JSON object."})
                continue

            thought = str(action.get("thought", ""))
            if "answer" in action:
                cited = [int(c) for c in action.get("citations", []) if str(c).lstrip("-").isdigit()]
                # drop citations the agent never actually saw: no hallucinated sources
                valid = [c for c in cited if c in self.toolbox.seen]
                steps.append(Step(thought, None, {}, "", time.time() - t0))
                if on_step is not None:
                    on_step(steps[-1])
                return AgentResult(str(action["answer"]).strip(), valid, steps)

            name, args = action.get("tool"), action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            obs = self._call_tool(str(name), args)
            steps.append(Step(thought, str(name), args, obs, time.time() - t0))
            if on_step is not None:
                on_step(steps[-1])
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})

        # budget exhausted: force a final answer
        messages.append({"role": "user", "content": "Tool budget used up. Give your final answer JSON now."})
        try:
            action = parse_action(self.llm.chat(messages))
            cited = [int(c) for c in action.get("citations", []) if str(c).isdigit()]
            return AgentResult(str(action.get("answer", "not found")),
                               [c for c in cited if c in self.toolbox.seen], steps, stopped="max_steps")
        except (ValueError, RuntimeError):
            return AgentResult("not found", [], steps, stopped="max_steps")
