"""Provider-side builtin tools: config parsing, pass-through on every protocol, results, and echoes."""

import json

import pytest
from agent_harness import session as agent_session
from model_harness import _AnthropicMockClientFactory, _AnthropicStreamClientFactory, _MockClientFactory, _session, _StreamClientFactory

from minacode.base import (
    PAUSED_TURN_KEY,
    SEARCH_SOURCES_KEY,
    ConfigError,
    ModelError,
    ToolCall,
    builtin_tool_label,
)
from minacode.config import (
    ConfigFile,
    ProviderConfig,
)
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.model import ModelClient
from minacode.render import search_sources_footer
from minacode.runner import ToolRunner
from minacode.skill import SkillLibrary

WEB_SEARCH = {"type": "web_search"}
FUNCTION_TOOL = {
    "type": "function",
    "function": {"name": "Bash", "description": "Run a command", "parameters": {"type": "object", "properties": {}}},
}


def _responses_body(status="completed", output=None):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-5",
        "output": output or [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
















































































def _chat_body(model="qwen3.8-max-preview"):
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    }
























