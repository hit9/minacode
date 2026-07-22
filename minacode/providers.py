"""Provider-specific protocol profiles and reasoning mappings."""

from typing import Any

DEEPSEEK_DEFAULT_MAX_TOKENS = 32_768

CHAT_REASONING_EFFORT_VALUES: dict[str, dict[str, str | int]] = {
    "thinking": {"minimal": "high", "low": "high", "medium": "high", "high": "max", "xhigh": "max"},
    "enable_thinking": {"minimal": 256, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384},
}

PROVIDER_PROFILES: dict[str, dict[str, Any]] = {
    "api.openai.com": {"chat_reasoning_rules": (("reasoning_effort", ("o1", "o3", "o4", "gpt-5")),), "strict_tools": True},
    "openrouter.ai": {"chat_reasoning": "reasoning"},
    "opencode.ai": {"api_rules": (("anthropic", ("claude-", "qwen3.")),), "chat_reasoning_rules": (("reasoning", ("deepseek-v4",)),)},
    "api.deepseek.com": {
        "chat_reasoning": "thinking",
        "max_tokens": DEEPSEEK_DEFAULT_MAX_TOKENS,
        "prompt_cache_key": False,
        "strict_tools": True,
        "strict_beta": True,
    },
    "aliyuncs.com": {
        "chat_reasoning_rules": (("reasoning_effort", ("qwen3.8-max-preview",)),),
        "reasoning_effort_off": "none",
    },
}
