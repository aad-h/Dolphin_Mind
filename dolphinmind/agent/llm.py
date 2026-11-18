"""Minimal chat-model interface so the agent is backend-agnostic."""

from __future__ import annotations

from typing import Dict, List, Protocol

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    def chat(self, messages: List[Message]) -> str: ...


class HFChatLLM:
    """Wrap a local Hugging Face model + tokenizer (e.g. from dolphinmind.models.load_qwen)."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 256, temperature: float = 0.2):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def chat(self, messages: List[Message]) -> str:
        from ..models import generate

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer, _ = generate(self.model, self.tokenizer, prompt, self.max_new_tokens, self.temperature)
        return answer


class ScriptedLLM:
    """Replays fixed responses in order. Used in tests and for demos without a GPU."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[List[Message]] = []

    def chat(self, messages: List[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)
