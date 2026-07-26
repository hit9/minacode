"""Compatibility overrides for provider APIs that diverge from generic protocol defaults."""

import re
from typing import Any

OPENAI_REASONING_MODEL_FAMILIES = ("o", "gpt-5")
ZAI_THINKING_MODEL_FAMILIES = ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")

KIMI_PLATFORM_COMPATIBILITY: dict[str, Any] = {
    "chat_reasoning_rules": (
        ("reasoning_effort", ("kimi-k3",)),
        ("thinking_toggle", ("kimi-k2.5", "kimi-k2.6")),
        ("mandatory_thinking", ("kimi-k2.7-code",)),
    ),
    "reasoning_effort_values": {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max"},
    # K3 cannot disable thinking on the open platform, so /reason off selects its lowest valid tier.
    "reasoning_effort_off": "low",
    "strict_tools": True,
    "suppress_temperature": True,
}

ZAI_COMPATIBILITY: dict[str, Any] = {
    "chat_reasoning_rules": (
        ("thinking_effort", ("glm-5.2",)),
        ("thinking_toggle", ZAI_THINKING_MODEL_FAMILIES),
    ),
    "prompt_cache_key": False,
}

CHAT_REASONING_EFFORT_VALUES: dict[str, dict[str, str | int]] = {
    # Why: DeepSeek accepts only high/max and documents these compatibility folds.
    # Evidence: https://api-docs.deepseek.com/guides/thinking_mode/
    "thinking": {"minimal": "high", "low": "high", "medium": "high", "high": "max", "xhigh": "max"},
    # Why: manual thinking APIs require integer token budgets; these are minacode's normalized
    # tiers, with Anthropic's documented 1,024-token minimum as the floor.
    # Evidence: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
    #           https://docs.qwencloud.com/api-reference/chat/openai-chat
    "enable_thinking": {"minimal": 1024, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384},
}

COMPATIBILITY_OVERRIDES: dict[str, dict[str, Any]] = {
    # Why: Chat Completions accepts reasoning_effort only for reasoning model families,
    # while strict function schemas are an OpenAI capability rather than a generic default.
    # Those same reasoning families reject temperature outright, so it is suppressed for them
    # alone and stays available on sibling chat models like gpt-4o.
    # Evidence: https://developers.openai.com/api/docs/guides/reasoning
    #           https://developers.openai.com/api/docs/guides/function-calling#strict-mode
    "api.openai.com": {
        "chat_reasoning_rules": (("reasoning_effort", OPENAI_REASONING_MODEL_FAMILIES),),
        "strict_tools": True,
        "suppress_temperature": ("reasoning_effort",),
    },
    # Why: OpenRouter normalizes providers behind its own top-level reasoning object.
    # Evidence: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    "openrouter.ai": {"chat_reasoning": "reasoning"},
    # Why: one OpenCode base URL multiplexes wire protocols by model, so api=auto cannot infer
    # the protocol from the URL: Claude and Qwen are served by Messages, GPT by Responses, and
    # the rest by Chat Completions.
    # Evidence: https://opencode.ai/docs/zen
    "opencode.ai": {
        "api_rules": (
            ("anthropic", ("claude-", "qwen")),
            ("responses", ("gpt-",)),
        ),
    },
    # Why: DeepSeek uses thinking.type plus a reduced effort scale, does not define OpenAI's
    # prompt_cache_key, and requires the /beta endpoint for strict function schemas.
    # Evidence: https://api-docs.deepseek.com/guides/thinking_mode/
    #           https://api-docs.deepseek.com/api/create-chat-completion/
    #           https://api-docs.deepseek.com/guides/tool_calls
    "api.deepseek.com": {
        "chat_reasoning": "thinking",
        "prompt_cache_key": False,
        "strict_tools": True,
        "strict_beta": True,
    },
    # Why: Qwen Chat documents top-level reasoning_effort for Qwen3.8, including none as
    # the OpenAI-compatible spelling for disabling thinking; older families use other controls.
    # Evidence: https://docs.qwencloud.com/api-reference/chat/openai-chat
    "aliyuncs.com": {
        "chat_reasoning_rules": (("reasoning_effort", ("qwen3.8-",)),),
        "reasoning_effort_off": "none",
    },
    # Why: the international and China Kimi open platforms expose the same model controls
    # on different regional domains. K2.5/K2.6 use thinking.type, K2.7 is always-thinking,
    # and K3 accepts only low/high/max; its lowest tier is the only valid fallback for off.
    # Their temperature values are fixed, while prompt_cache_key is explicitly supported.
    # Evidence: https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.ai/docs/api/models-overview
    #           https://platform.kimi.ai/docs/api/chat
    "moonshot.ai": KIMI_PLATFORM_COMPATIBILITY,
    "moonshot.cn": KIMI_PLATFORM_COMPATIBILITY,
    # Why: Kimi Code is a separate subscription API with different model IDs and K3 off
    # semantics: k3 accepts low/high/max and maps none to disabled, while both
    # kimi-for-coding variants are always-thinking K2.7 models. Its official client exposes
    # request temperature, so it does not inherit the open platform's fixed-temperature rule.
    # Evidence: https://www.kimi.com/code/docs/kimi-code/models.html
    #           https://www.kimi.com/code/docs/
    #           https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/env-vars.html
    "kimi.com": {
        "chat_reasoning_rules": (
            ("reasoning_effort", ("k3",)),
            ("mandatory_thinking", ("kimi-for-coding",)),
        ),
        "reasoning_effort_values": {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max"},
        "reasoning_effort_off": "none",
    },
    # Why: both Z.AI regions use thinking.type for GLM-4.5+ and reasoning_effort for
    # GLM-5.2+. Their context caches are automatic and require no request cache key.
    # Evidence: https://docs.z.ai/guides/capabilities/thinking
    #           https://docs.z.ai/guides/overview/concept-param
    #           https://docs.z.ai/guides/capabilities/cache
    "z.ai": ZAI_COMPATIBILITY,
    # Why: China's BigModel endpoint documents the same thinking and automatic-cache contract.
    # Evidence: https://docs.bigmodel.cn/cn/guide/capabilities/thinking
    #           https://docs.bigmodel.cn/cn/guide/start/concept-param
    #           https://docs.bigmodel.cn/cn/guide/capabilities/cache
    "bigmodel.cn": ZAI_COMPATIBILITY,
}


# Anthropic's thinking configuration is generational rather than host-keyed, because the same
# model families are reachable through the first-party API, cloud platforms, and gateways.
#
# Why: extended thinking (thinking.type "enabled" with budget_tokens) is the only mode on
# Claude 4.5 and earlier, is deprecated but functional on the 4.6 generation, and is rejected
# with a 400 from Claude 4.7 onward, where adaptive thinking plus output_config.effort replaces
# it. Thinking is on by default on the newest models, and the always-thinking families reject
# thinking.type "disabled" outright, so /reason off has to omit the parameter for them instead.
# Effort levels are also generational: xhigh arrived after the 4.6 generation, which tops out at
# max, so minacode's highest normalized level maps to whichever of the two the model accepts.
# Evidence: https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting#supported-models
#           https://platform.claude.com/docs/en/build-with-claude/extended-thinking
#           https://platform.claude.com/docs/en/build-with-claude/effort
ANTHROPIC_ADAPTIVE_MIN_VERSION = (4, 6)
ANTHROPIC_XHIGH_MIN_VERSION = (4, 7)
ANTHROPIC_ALWAYS_THINKING_FAMILIES = ("fable", "mythos")
ANTHROPIC_EFFORT_VALUES: dict[str, str] = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}


def anthropic_model_version(model: str) -> tuple[int, int] | None:
    """The (major, minor) generation in a model id, or None when it carries no version.

    Ids arrive in many shapes across platforms — `claude-opus-4-5-20251101`, `claude-sonnet-5`,
    `anthropic.claude-sonnet-4-5-20250929-v1:0` — so the first one or two short numeric tokens
    are read as the version and long ones are left alone as release dates."""
    tokens = [token for token in re.split(r"[^0-9a-z]+", model.lower()) if token]
    for index, token in enumerate(tokens):
        if not (token.isdigit() and len(token) <= 2):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        minor = int(following) if following.isdigit() and len(following) <= 2 else 0
        return int(token), minor
    return None


def anthropic_thinking_params(model: str, reasoning: str, effort: str, budget_tokens: int) -> dict[str, Any]:
    """The thinking and effort fields for a Messages request, for the generation of `model`.

    An unversioned id is treated as current, since new models keep arriving while the
    extended-thinking-only ones are a closed, shrinking set."""
    version = anthropic_model_version(model)
    adaptive = version is None or version >= ANTHROPIC_ADAPTIVE_MIN_VERSION
    families = re.split(r"[^0-9a-z]+", model.lower())
    always_thinking = any(family in families for family in ANTHROPIC_ALWAYS_THINKING_FAMILIES)
    if reasoning == "off":
        # Extended-thinking models think only when asked, so omitting the parameter is off.
        return {"thinking": {"type": "disabled"}} if adaptive and not always_thinking else {}
    if not adaptive:
        return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
    level = ANTHROPIC_EFFORT_VALUES.get(effort, "high")
    if level == "xhigh" and version is not None and version < ANTHROPIC_XHIGH_MIN_VERSION:
        level = "max"
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
