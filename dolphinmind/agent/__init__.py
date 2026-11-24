"""DolphinAgent: tool-using agent over long documents."""

from .llm import LLM, HFChatLLM, LLMError, OpenAICompatLLM, ScriptedLLM, Usage
from .multi import Finding, LexicalVerifier, LLMVerifier, MultiAgent, MultiAgentResult, Verdict
from .react import AgentResult, ReActAgent, parse_action
from .tools import Document, Toolbox

__all__ = [
    "LLM", "HFChatLLM", "LLMError", "OpenAICompatLLM", "ScriptedLLM", "Usage",
    "AgentResult", "ReActAgent", "parse_action", "Document", "Toolbox",
    "Finding", "LexicalVerifier", "LLMVerifier", "MultiAgent", "MultiAgentResult", "Verdict",
]
