"""Provider compatibility profiles and protocol adaptation helpers.

Profiles describe documented host/model differences. They do not build request bodies; the
Chat, Responses, and Anthropic protocol paths remain responsible for their own wire formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


OPENAI_REASONING_MODEL_FAMILIES = ("o", "gpt-5")
ZAI_THINKING_MODEL_FAMILIES = ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")


@dataclass(frozen=True)
class ModelRule:
    """A value selected by model-family prefixes or a documented version pattern."""

    value: str
    prefixes: tuple[str, ...] = ()
    pattern: str = ""

    def matches(self, model: str) -> bool:
        return any(model.startswith(prefix) for prefix in self.prefixes) or bool(self.pattern and re.match(self.pattern, model))


@dataclass(frozen=True)
class CompatibilityProfile:
    """Only the documented ways a host differs from generic protocol behavior."""

    api_rules: tuple[ModelRule, ...] = ()
    chat_reasoning: str | None = None
    chat_reasoning_rules: tuple[ModelRule, ...] = ()
    reasoning_effort_values: Mapping[str, str | int] = field(default_factory=dict)
    reasoning_effort_off_rules: tuple[ModelRule, ...] = ()
    responses_reasoning_effort_off_rules: tuple[ModelRule, ...] = ()
    prompt_cache_key: bool = True
    strict_tools: bool = False
    strict_beta: bool = False
    suppress_temperature: bool = False
    suppress_temperature_models: tuple[str, ...] = ()

    @staticmethod
    def rule_value(rules: tuple[ModelRule, ...], model: str) -> str | None:
        return next((rule.value for rule in rules if rule.matches(model)), None)


@dataclass(frozen=True)
class ResolvedProvider:
    """The effective transport policy after explicit config and compatibility are folded."""

    api: str
    base_url: str
    host: str
    chat_reasoning: str
    reasoning_effort: str | None
    suppress_temperature: bool
    prompt_cache_key: bool
    strict_tools_active: bool


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


KIMI_EFFORT_VALUES = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max"}

KIMI_PLATFORM_COMPATIBILITY = CompatibilityProfile(
    chat_reasoning_rules=(
        ModelRule("reasoning_effort", ("kimi-k3",)),
        ModelRule("thinking_toggle", ("kimi-k2.5", "kimi-k2.6")),
        ModelRule("mandatory_thinking", ("kimi-k2.7-code",)),
    ),
    reasoning_effort_values=KIMI_EFFORT_VALUES,
    # K3 cannot disable thinking on the open platform, so /reason off selects its lowest valid tier.
    reasoning_effort_off_rules=(ModelRule("low", ("kimi-k3",)),),
    strict_tools=True,
    suppress_temperature=True,
)

ZAI_COMPATIBILITY = CompatibilityProfile(
    chat_reasoning_rules=(
        ModelRule("thinking_effort", ("glm-5.2",)),
        ModelRule("thinking_toggle", ZAI_THINKING_MODEL_FAMILIES),
    ),
    prompt_cache_key=False,
)


COMPATIBILITY_PROFILES: dict[str, CompatibilityProfile] = {
    # Why: Chat Completions accepts reasoning_effort only for reasoning model families,
    # while strict function schemas are an OpenAI capability rather than a generic default.
    # Responses models from GPT-5.1 onward document `none` as the no-reasoning effort; the
    # original GPT-5 does not. These reasoning families reject temperature, while sibling models
    # such as gpt-4o retain it.
    # Evidence: https://developers.openai.com/api/docs/guides/reasoning
    #           https://developers.openai.com/api/docs/models/gpt-5
    #           https://developers.openai.com/api/docs/models/gpt-5.1
    #           https://developers.openai.com/api/docs/guides/function-calling#strict-mode
    "api.openai.com": CompatibilityProfile(
        chat_reasoning_rules=(ModelRule("reasoning_effort", OPENAI_REASONING_MODEL_FAMILIES),),
        responses_reasoning_effort_off_rules=(ModelRule("none", pattern=r"gpt-5\.(?:[1-9]\d*)(?:-|$)"),),
        strict_tools=True,
        suppress_temperature_models=OPENAI_REASONING_MODEL_FAMILIES,
    ),
    # Why: OpenRouter normalizes providers behind its own top-level reasoning object.
    # Evidence: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    "openrouter.ai": CompatibilityProfile(chat_reasoning="reasoning"),
    # Why: one OpenCode base URL multiplexes wire protocols by model, so api=auto cannot infer
    # the protocol from the URL: Claude and Qwen are served by Messages, GPT by Responses, and
    # the rest by Chat Completions.
    # Evidence: https://opencode.ai/docs/zen
    "opencode.ai": CompatibilityProfile(
        api_rules=(
            ModelRule("anthropic", ("claude-", "qwen")),
            ModelRule("responses", ("gpt-",)),
        )
    ),
    # Why: DeepSeek uses thinking.type plus a reduced effort scale, does not define OpenAI's
    # prompt_cache_key, and requires the /beta endpoint for strict function schemas.
    # Evidence: https://api-docs.deepseek.com/guides/thinking_mode/
    #           https://api-docs.deepseek.com/api/create-chat-completion/
    #           https://api-docs.deepseek.com/guides/tool_calls
    "api.deepseek.com": CompatibilityProfile(
        chat_reasoning="thinking",
        prompt_cache_key=False,
        strict_tools=True,
        strict_beta=True,
    ),
    # Why: Qwen Chat documents top-level reasoning_effort for Qwen3.8, including none as
    # the OpenAI-compatible spelling for disabling thinking; older families use other controls.
    # Evidence: https://docs.qwencloud.com/api-reference/chat/openai-chat
    "aliyuncs.com": CompatibilityProfile(
        chat_reasoning_rules=(ModelRule("reasoning_effort", ("qwen3.8-",)),),
        reasoning_effort_off_rules=(ModelRule("none", ("qwen3.8-",)),),
    ),
    # Why: the international and China Kimi open platforms expose the same model controls
    # on different regional domains. K2.5/K2.6 use thinking.type, K2.7 is always-thinking,
    # and K3 accepts only low/high/max. Their temperature values are fixed.
    # Evidence: https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.ai/docs/api/models-overview
    #           https://platform.kimi.ai/docs/api/chat
    "moonshot.ai": KIMI_PLATFORM_COMPATIBILITY,
    "moonshot.cn": KIMI_PLATFORM_COMPATIBILITY,
    # Why: Kimi Code is a separate subscription API with different model IDs and K3 off
    # semantics. Its official client exposes request temperature, so it does not inherit the
    # open platform's fixed-temperature rule.
    # Evidence: https://www.kimi.com/code/docs/kimi-code/models.html
    #           https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/env-vars.html
    "kimi.com": CompatibilityProfile(
        chat_reasoning_rules=(
            ModelRule("reasoning_effort", ("k3",)),
            ModelRule("mandatory_thinking", ("kimi-for-coding",)),
        ),
        reasoning_effort_values=KIMI_EFFORT_VALUES,
        reasoning_effort_off_rules=(ModelRule("none", ("k3",)),),
    ),
    # Why: both Z.AI regions use thinking.type for GLM-4.5+ and reasoning_effort for GLM-5.2+.
    # Their context caches are automatic and require no request cache key.
    # Evidence: https://docs.z.ai/guides/capabilities/thinking
    #           https://docs.z.ai/guides/capabilities/cache
    "z.ai": ZAI_COMPATIBILITY,
    # Why: China's BigModel endpoint documents the same thinking and automatic-cache contract.
    # Evidence: https://docs.bigmodel.cn/cn/guide/capabilities/thinking
    #           https://docs.bigmodel.cn/cn/guide/capabilities/cache
    "bigmodel.cn": ZAI_COMPATIBILITY,
}


def compatibility_for_host(host: str, profiles: Mapping[str, CompatibilityProfile] = COMPATIBILITY_PROFILES) -> CompatibilityProfile:
    """Return the most specific domain profile while respecting label boundaries."""

    matches = ((domain, profile) for domain, profile in profiles.items() if host == domain or host.endswith(f".{domain}"))
    return max(matches, key=lambda item: len(item[0]), default=("", CompatibilityProfile()))[1]
