"""Protocol-specific helpers shared by request construction and context accounting."""

from __future__ import annotations

import re
from typing import Any


ANTHROPIC_ADAPTIVE_MIN_VERSION = (4, 6)
ANTHROPIC_XHIGH_MIN_VERSION = (4, 7)
ANTHROPIC_ALWAYS_THINKING_FAMILIES = ("fable", "mythos")
ANTHROPIC_EFFORT_VALUES = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
# Why: manual thinking APIs use integer token budgets, while DeepSeek's thinking mode accepts
# only high/max. These tables map minacode's normalized effort without encoding a host profile.
# Evidence: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
#           https://api-docs.deepseek.com/guides/thinking_mode/
#           https://docs.qwencloud.com/api-reference/chat/openai-chat
CHAT_REASONING_EFFORT_VALUES: dict[str, dict[str, str | int]] = {
    # DeepSeek accepts only high/max and documents these compatibility folds.
    "thinking": {"minimal": "high", "low": "high", "medium": "high", "high": "max", "xhigh": "max"},
    # Manual thinking APIs use integer token budgets, with Anthropic's 1,024-token minimum.
    "enable_thinking": {"minimal": 1024, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384},
}


def anthropic_model_version(model: str) -> tuple[int, int] | None:
    """Return the first short numeric generation in a Claude model id, if present."""

    tokens = [token for token in re.split(r"[^0-9a-z]+", model.lower()) if token]
    for index, token in enumerate(tokens):
        if not (token.isdigit() and len(token) <= 2):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        minor = int(following) if following.isdigit() and len(following) <= 2 else 0
        return int(token), minor
    return None


def anthropic_thinking_params(model: str, reasoning: str, effort: str, budget_tokens: int) -> dict[str, Any]:
    """Build the documented thinking fields for a known Claude generation.

    Unknown aliases remain unconfigured. A gateway may point such a name at either side of the
    adaptive-thinking boundary, and guessing would turn a valid alias into a 400 response.
    """

    # Why: 4.5 and earlier require manual thinking; 4.6 recommends adaptive; 4.7+ rejects
    # manual thinking. Opus 4.5 uniquely combines manual thinking with output_config.effort.
    # Evidence: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
    #           https://platform.claude.com/docs/en/build-with-claude/effort
    version = anthropic_model_version(model)
    if version is None:
        return {}
    families = re.split(r"[^0-9a-z]+", model.lower())
    adaptive = version >= ANTHROPIC_ADAPTIVE_MIN_VERSION
    always_thinking = any(family in families for family in ANTHROPIC_ALWAYS_THINKING_FAMILIES)
    if reasoning == "off":
        return {"thinking": {"type": "disabled"}} if adaptive and not always_thinking else {}
    level = ANTHROPIC_EFFORT_VALUES.get(effort, "high")
    if not adaptive:
        params: dict[str, Any] = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
        if version == (4, 5) and "opus" in families:
            params["output_config"] = {"effort": level if level in ("low", "medium", "high") else "high"}
        return params
    if level == "xhigh" and version < ANTHROPIC_XHIGH_MIN_VERSION:
        level = "max"
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}


def anthropic_thinking_always_on(model: str) -> bool:
    families = re.split(r"[^0-9a-z]+", model.lower())
    return any(family in families for family in ANTHROPIC_ALWAYS_THINKING_FAMILIES)


def readable_provider_context(message: dict[str, Any], responses_key: str, anthropic_key: str) -> list[str]:
    """Readable provider state replayed in addition to the normalized assistant fields.

    Encrypted payloads and signatures are transport state rather than prompt text, so their byte
    length is not a token estimate. Text, summaries, and thinking blocks do occupy model context.
    """

    readable: list[str] = []
    responses = message.get(responses_key)
    if isinstance(responses, list):
        for item in responses:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            for key in ("content", "summary"):
                value = item.get(key)
                if value:
                    readable.append(str(value))
    anthropic = message.get(anthropic_key)
    if isinstance(anthropic, list):
        for block in anthropic:
            if not isinstance(block, dict) or block.get("type") not in ("thinking", "redacted_thinking"):
                continue
            if thinking := block.get("thinking"):
                readable.append(str(thinking))
    return readable
