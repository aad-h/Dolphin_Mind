"""DolphinAgent: tool-using agent over long documents."""

from .budget import Budget, BudgetedLLM, BudgetExceeded
from .llm import LLM, HFChatLLM, LLMError, OpenAICompatLLM, ScriptedLLM, Usage
from .multi import Finding, LexicalVerifier, LLMVerifier, MultiAgent, MultiAgentResult, Verdict
from .react import AgentResult, ReActAgent, parse_action
from .tools import Document, Toolbox
from .tracing import OpenTelemetryExporter, Span, TracedLLM, Tracer, format_trace

__all__ = [
    "LLM", "HFChatLLM", "LLMError", "OpenAICompatLLM", "ScriptedLLM", "Usage",
    "AgentResult", "ReActAgent", "parse_action", "Document", "Toolbox",
    "Budget", "BudgetedLLM", "BudgetExceeded",
    "Tracer", "Span", "TracedLLM", "OpenTelemetryExporter", "format_trace",
    "Finding", "LexicalVerifier", "LLMVerifier", "MultiAgent", "MultiAgentResult", "Verdict",
]
