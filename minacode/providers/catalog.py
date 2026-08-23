"""Declarative model capabilities and provider compatibility data.

The catalog is advisory rather than an allowlist. Unmatched providers and model names stay on
minacode's generic protocol path, while ``provider_compat`` folds these documented exceptions into
the resolved request policy.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ModelRuleData(TypedDict):
    """A value selected by model-family prefixes or a documented version pattern."""

    value: str
    prefixes: NotRequired[tuple[str, ...]]
    pattern: NotRequired[str]


class ModelEffortRuleData(TypedDict):
    """Supported normalized effort levels for a model family."""

    levels: tuple[str, ...]
    prefixes: NotRequired[tuple[str, ...]]
    pattern: NotRequired[str]


BuiltinToolRuleData = dict[str, object]


class CompatibilityData(TypedDict, total=False):
    """Data compiled into one provider compatibility profile."""

    model_capabilities: tuple[str, ...]
    api_rules: tuple[ModelRuleData, ...]
    chat_reasoning: str
    chat_reasoning_rules: tuple[ModelRuleData, ...]
    chat_reasoning_history: str
    chat_reasoning_history_rules: tuple[ModelRuleData, ...]
    reasoning_effort_levels: tuple[str, ...]
    reasoning_effort_level_rules: tuple[ModelEffortRuleData, ...]
    reasoning_effort_off_rules: tuple[ModelRuleData, ...]
    responses_reasoning_effort_off_rules: tuple[ModelRuleData, ...]
    responses_reasoning_models: tuple[str, ...] | None
    prompt_cache_key: bool
    json_response_format: bool
    strict_tools: bool
    strict_beta: bool
    suppress_temperature: bool
    suppress_temperature_models: tuple[str, ...]
    # Provider-side (builtin) tools are provider-native JSON passed through unchanged. Each rule
    # is a required JSON subset, so the catalog can distinguish entries that share a type but have
    # different lifecycles (for example Kimi's builtin_function/$web_search). ``None`` keeps
    # generic pass-through for unknown hosts; an empty mapping means this known provider has no
    # supported provider-side tools through the ``tools`` array.
    builtin_tools_by_wire: dict[str, tuple[BuiltinToolRuleData, ...]] | None


class ProviderData(CompatibilityData):
    """A named provider and the host domains on which its policy applies."""

    hosts: tuple[str, ...]


class AnthropicModelData(TypedDict):
    adaptive_min_version: tuple[int, int]
    xhigh_min_version: tuple[int, int]
    always_thinking_families: tuple[str, ...]
    effort_levels: tuple[str, ...]


REASONING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")

# Why: manual thinking APIs use integer token budgets, while model-native effort APIs expose
# subsets of minacode's normalized effort scale. ``provider_compat`` applies the shared nearest-
# level fallback; the budget table is a wire value rather than an effort compatibility rule.
# Evidence: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
#           https://api-docs.deepseek.com/guides/thinking_mode/
#           https://docs.qwencloud.com/api-reference/chat/openai-chat
THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}

ANTHROPIC_MODELS: AnthropicModelData = {
    "adaptive_min_version": (4, 6),
    "xhigh_min_version": (4, 7),
    "always_thinking_families": ("fable", "mythos"),
    "effort_levels": ("low", "medium", "high", "xhigh", "max"),
}

# OpenAI effort support varies by model generation. Unknown future models stay on the generic
# pass-through path; only documented families are folded to their nearest accepted level.
# Evidence: https://developers.openai.com/api/docs/guides/latest-model
#           https://developers.openai.com/api/docs/models/gpt-5.5
#           https://developers.openai.com/api/docs/models/gpt-5.4-pro
#           https://developers.openai.com/api/docs/models/gpt-5.3-codex
#           https://developers.openai.com/api/docs/models/gpt-5.1
#           https://developers.openai.com/api/docs/models/gpt-5
OPENAI_EFFORT_CAPABILITY: CompatibilityData = {
    "reasoning_effort_level_rules": (
        {"levels": ("low", "medium", "high", "xhigh", "max"), "pattern": r"gpt-5\.6(?:-|$)"},
        {"levels": ("medium", "high", "xhigh"), "pattern": r"gpt-5\.(?:2|4|5)-pro(?:-|$)"},
        {"levels": ("low", "medium", "high", "xhigh"), "pattern": r"gpt-5\.(?:2|3)-codex(?:-|$)"},
        {"levels": ("low", "medium", "high", "xhigh"), "pattern": r"gpt-5\.(?:2|4|5)(?:-|$)"},
        {"levels": ("low", "medium", "high"), "pattern": r"gpt-5\.1(?:-|$)"},
        {"levels": ("high",), "pattern": r"gpt-5-pro(?:-|$)"},
        {"levels": ("minimal", "low", "medium", "high"), "pattern": r"gpt-5(?:-|$)"},
        {"levels": ("low", "medium", "high"), "pattern": r"o[1-4](?:-|$)"},
    ),
    "reasoning_effort_off_rules": (
        {"value": "medium", "pattern": r"gpt-5\.(?:2|4|5)-pro(?:-|$)"},
        {"value": "low", "pattern": r"gpt-5\.(?:2|3)-codex(?:-|$)"},
        {"value": "high", "pattern": r"gpt-5-pro(?:-|$)"},
        {"value": "none", "pattern": r"gpt-5\.(?:[1-9]\d*)(?:-|$)"},
    ),
    "responses_reasoning_effort_off_rules": (
        {"value": "medium", "pattern": r"gpt-5\.(?:2|4|5)-pro(?:-|$)"},
        {"value": "low", "pattern": r"gpt-5\.(?:2|3)-codex(?:-|$)"},
        {"value": "high", "pattern": r"gpt-5-pro(?:-|$)"},
        {"value": "none", "pattern": r"gpt-5\.(?:[1-9]\d*)(?:-|$)"},
    ),
}

MODEL_CAPABILITIES: dict[str, CompatibilityData] = {
    "openai_effort": OPENAI_EFFORT_CAPABILITY,
    # DeepSeek V4 uses thinking.type and accepts low/high/max plus xhigh as a model-specific
    # compatibility level. The host folds medium/high/xhigh to high server-side, so only low and
    # max are distinct efforts; xhigh is kept in the level list because the endpoint accepts it.
    # Its tool-call replay rule travels with the model, not the endpoint: a
    # gateway serving the same model still requires reasoning on tool-call messages and ignores it
    # everywhere else.
    # Evidence: https://api-docs.deepseek.com/guides/thinking_mode/
    "deepseek_v4": {
        "chat_reasoning_rules": ({"value": "thinking", "prefixes": ("deepseek-v4-",)},),
        "chat_reasoning_history_rules": ({"value": "tool_calls", "prefixes": ("deepseek-v4-",)},),
        "reasoning_effort_level_rules": ({"levels": ("low", "high", "xhigh", "max"), "prefixes": ("deepseek-v4-",)},),
    },
    # K3 uses normalized effort, K2.5/K2.6 use thinking.type, and K2.7 is always-thinking. K3 and
    # K2.7 preserve thinking across turns; K3 cannot disable thinking on the open platform.
    # Evidence: https://platform.kimi.com/docs/guide/use-thinking-models
    "kimi_open": {
        "chat_reasoning_rules": (
            {"value": "reasoning_effort", "prefixes": ("kimi-k3",)},
            {"value": "thinking_toggle", "prefixes": ("kimi-k2.5", "kimi-k2.6")},
            {"value": "mandatory_thinking", "prefixes": ("kimi-k2.7-code",)},
        ),
        "chat_reasoning_history_rules": ({"value": "all", "prefixes": ("kimi-k3", "kimi-k2.7-code")},),
        "reasoning_effort_level_rules": ({"levels": ("low", "high", "max"), "prefixes": ("kimi-k3",)},),
        "reasoning_effort_off_rules": ({"value": "low", "prefixes": ("kimi-k3",)},),
    },
    # The standard Z.AI APIs use thinking.type for GLM-4.5+ and reasoning_effort for GLM-5.2+.
    # GLM-5.2 documents two effort levels and resolves anything other than "high" to max, so an
    # unfolded "low" buys the most expensive setting rather than the cheapest: its low end is high.
    # Evidence: https://docs.z.ai/guides/capabilities/thinking-mode
    #           https://docs.z.ai/guides/overview/migrate-to-glm-new
    "zai_standard": {
        "chat_reasoning_rules": (
            {"value": "thinking_effort", "prefixes": ("glm-5.2",)},
            {"value": "thinking_toggle", "prefixes": ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")},
        ),
        "reasoning_effort_level_rules": ({"levels": ("high", "max"), "prefixes": ("glm-5.2",)},),
    },
    # OpenCode currently exposes the same controls for its documented GLM-5 families.
    # Evidence: https://opencode.ai/docs/zen
    "zai_opencode": {
        "chat_reasoning_rules": (
            {"value": "thinking_effort", "prefixes": ("glm-5.2",)},
            {"value": "thinking_toggle", "prefixes": ("glm-5",)},
        ),
        "reasoning_effort_level_rules": ({"levels": ("high", "max"), "prefixes": ("glm-5.2",)},),
    },
    # Qwen3.8 Chat uses top-level reasoning_effort, including none to disable thinking. Its Max
    # models document low/medium/xhigh only: high and max are not accepted spellings there.
    # Evidence: https://docs.qwencloud.com/api-reference/chat/openai-chat
    "qwen3_8": {
        "chat_reasoning_rules": ({"value": "reasoning_effort", "prefixes": ("qwen3.8-",)},),
        "reasoning_effort_level_rules": ({"levels": ("low", "medium", "xhigh"), "prefixes": ("qwen3.8-",)},),
        "reasoning_effort_off_rules": ({"value": "none", "prefixes": ("qwen3.8-",)},),
    },
    # Kimi Code has distinct model IDs and K3 off semantics from the open platform.
    # Evidence: https://www.kimi.com/code/docs/kimi-code/models.html
    "kimi_code": {
        "chat_reasoning_rules": (
            {"value": "reasoning_effort", "prefixes": ("k3",)},
            {"value": "mandatory_thinking", "prefixes": ("kimi-for-coding",)},
        ),
        "chat_reasoning_history_rules": ({"value": "all", "prefixes": ("k3", "kimi-for-coding")},),
        "reasoning_effort_level_rules": ({"levels": ("low", "high", "max"), "prefixes": ("k3",)},),
        "reasoning_effort_off_rules": ({"value": "none", "prefixes": ("k3",)},),
    },
}


TEXT_ONLY_VALUE = "text_only"


# Documented text-only model families: a conservative negative list for image routing. Anything
# not matched stays unknown and is tried on the main model; the catalog is reviewed source data,
# never scraped at runtime. Rules are anchored to complete documented families or exact IDs, and
# matching is case-insensitive because configured model IDs are normalized to lower case.
# A gateway ID in canonical vendor/model form (e.g. `deepseek/deepseek-v4-pro`) is matched by
# its model part only when the vendor prefix is one of CANONICAL_VENDORS; a custom alias such as
# `production-model` stays unknown and is probed.
#
# Evidence:
#   DeepSeek (chat = V3, reasoner = R1, V4 Flash/Pro text; V4 Flash Vision is the documented
#   vision variant and must never match): https://api-docs.deepseek.com/quick_start/pricing
#   https://api-docs.deepseek.com/guides/vision
#   Z.AI / BigModel GLM families (GLM-5x text, GLM-4.7/4.6/4.5 and non-V variants, GLM-4-32B;
#   GLM-5V / GLM-4.6V / GLM-4.5V / GLM-OCR must never match):
#   https://docs.z.ai/guides/overview/overview
#   Qwen coding-plan models documented without direct vision (do not generalize to Qwen3.5/3.6/3.7
#   or Qwen-VL): https://help.aliyun.com/zh/model-studio/add-vision-skill
#   https://help.aliyun.com/zh/model-studio/vision-model/
#   Kimi / Moonshot text families (K3, K2.7 Code, pre-K2.5 K2, moonshot-v1 sizes; Kimi K2.5,
#   Kimi K2.6, and moonshot-v1-*-vision-preview must never match):
#   https://platform.kimi.com/docs/models
#   https://platform.kimi.com/docs/guide/use-kimi-vision-model
#   gpt-oss text models: https://developers.openai.com/api/docs/models/gpt-oss-120b
#   Claude stays unknown/main-first (never in the negative list):
#   https://platform.claude.com/docs/en/about-claude/models/overview
TEXT_ONLY_MODEL_RULES: tuple[ModelRuleData, ...] = (
    # DeepSeek ordinary text models.
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-chat(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-reasoner(?:-|$)"},
    # V4 text models, excluding the documented vision variant (`...-vision-exp` or any other
    # `-vision...` suffix).
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-v4-(?:flash|pro)(?:-(?!vision)|$)"},
    # DeepSeek V3 and R1 documented text families: `deepseek-v3`, `deepseek-v3.x`, `deepseek-r1`
    # and their suffixed variants. The documented vision families `deepseek-vl2` and
    # `deepseek-ocr` share only the `deepseek-` prefix and never match; `deepseek-*` stays unknown.
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-v3(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-v3\.\d+(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"deepseek-r1(?:-|$)"},
    # Z.AI / BigModel GLM text families. `glm-5`, `glm-5-turbo`, and numbered 5.x variants;
    # GLM-4.7/4.6/4.5 and their non-V variants; GLM-4-32B-0414-128K. `glm-5v`, `glm-4.6v`,
    # `glm-4.5v`, and `glm-ocr` never match (a `v` directly after the version is not `-`/end).
    {"value": TEXT_ONLY_VALUE, "pattern": r"glm-5(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"glm-5-turbo(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"glm-5\.\d+(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"glm-4\.[567](?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"glm-4-32b-0414-128k(?:-|$)"},
    # Qwen models explicitly documented as lacking direct vision in coding plans.
    {"value": TEXT_ONLY_VALUE, "pattern": r"qwen3-max-2026-01-23(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"qwen3-coder-(?:next|plus)(?:-|$)"},
    # Kimi K3 (open platform and Kimi Code), K2.7 Code, and pre-K2.5 K2 text IDs. Kimi K2.5/
    # K2.6 never match (`kimi-k2` followed by `.` is not `-`/end).
    {"value": TEXT_ONLY_VALUE, "pattern": r"kimi-k3(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"kimi-k2\.7-code(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"kimi-k2(?!\.5|\.6)(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"k3(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"kimi-for-coding(?:-|$)"},
    # Moonshot v1 text sizes; a `moonshot-v1-*-vision-preview` ID never matches.
    {"value": TEXT_ONLY_VALUE, "pattern": r"moonshot-v1-(?:8k|32k|128k)(?:-(?!vision)|$)"},
    # Other common explicit text models.
    {"value": TEXT_ONLY_VALUE, "pattern": r"minimax-m2\.5(?:-|$)"},
    {"value": TEXT_ONLY_VALUE, "pattern": r"gpt-oss-(?:20b|120b)(?:-|$)"},
)


# Canonical vendor slugs of the negative-list families, for the `vendor/model` gateway form.
# A different vendor prefix keeps the ID unknown even when its suffix would match a rule.
CANONICAL_VENDORS: tuple[str, ...] = (
    "deepseek",
    "z-ai",
    "bigmodel",
    "qwen",
    "moonshotai",
    "minimax",
    "openai",
)


PROVIDER_CATALOG: dict[str, ProviderData] = {
    # Why: Chat Completions accepts reasoning_effort only for reasoning model families,
    # while strict function schemas are an OpenAI capability rather than a generic default.
    # Responses models from GPT-5.1 onward document `none` as the no-reasoning effort; the
    # original GPT-5 does not. The optional Responses reasoning object is limited to these
    # reasoning families: GPT-4.1 supports Responses but is explicitly non-reasoning. Reasoning
    # families reject temperature, while sibling chat models such as gpt-4o retain it.
    # Evidence: https://developers.openai.com/api/docs/guides/reasoning
    #           https://developers.openai.com/api/docs/models/gpt-5
    #           https://developers.openai.com/api/docs/models/gpt-5.1
    #           https://developers.openai.com/api/docs/models/gpt-4.1
    #           https://developers.openai.com/api/docs/guides/function-calling#strict-mode
    "openai": {
        "hosts": ("api.openai.com",),
        "json_response_format": True,
        "model_capabilities": ("openai_effort",),
        "chat_reasoning_rules": ({"value": "reasoning_effort", "prefixes": ("o", "gpt-5")},),
        "responses_reasoning_models": ("o", "gpt-5"),
        "strict_tools": True,
        "suppress_temperature_models": ("o", "gpt-5"),
        # Why: OpenAI documents provider-side web_search on the Responses API; Chat Completions
        # rejects non-function tool entries. Only web_search is supported so far; the other
        # server tools need file/container/media approval lifecycles.
        # Evidence: https://developers.openai.com/api/docs/guides/tools-web-search
        "builtin_tools_by_wire": {"responses": ({"type": "web_search"},)},
    },
    # Why: OpenRouter normalizes providers behind its own top-level reasoning object.
    # Evidence: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    "openrouter": {
        "hosts": ("openrouter.ai",),
        "chat_reasoning": "reasoning",
        # Why: OpenRouter documents server tools as `openrouter:*` entries in the Chat or
        # Responses tools array. The legacy `plugins`/`:online` search config is deprecated.
        # Evidence: https://openrouter.ai/docs/guides/features/server-tools/overview
        "builtin_tools_by_wire": {
            "chat": (
                {"type": "openrouter:web_search"},
                {"type": "openrouter:web_fetch"},
                {"type": "openrouter:datetime"},
            ),
            "responses": (
                {"type": "openrouter:web_search"},
                {"type": "openrouter:web_fetch"},
                {"type": "openrouter:datetime"},
            ),
        },
    },
    # Why: one OpenCode base URL multiplexes wire protocols by model, so api=auto cannot infer
    # the protocol from the URL: Claude and Qwen are served by Messages, GPT and Grok by
    # Responses, and the rest by Chat Completions. Its model traits reuse the canonical effort
    # capabilities above; only routing remains an OpenCode provider overlay.
    # Evidence: https://opencode.ai/docs/zen
    "opencode": {
        "hosts": ("opencode.ai",),
        "model_capabilities": ("openai_effort", "deepseek_v4", "zai_opencode", "kimi_open"),
        "api_rules": (
            {"value": "anthropic", "prefixes": ("claude-", "qwen")},
            {"value": "responses", "prefixes": ("gpt-", "grok-")},
        ),
        # Why: Zen only documents endpoint routing; its websearch/webfetch client tools are not
        # Zen API server tools, so no provider-side tools are assumed.
        # Evidence: https://opencode.ai/docs/zen
        "builtin_tools_by_wire": {},
    },
    # Why: DeepSeek uses thinking.type plus a reduced effort scale, does not define OpenAI's
    # prompt_cache_key, and requires the /beta endpoint for strict function schemas. Ordinary
    # turns may omit reasoning, but every assistant tool-call message must retain it.
    # Evidence: https://api-docs.deepseek.com/guides/thinking_mode/
    #           https://api-docs.deepseek.com/api/create-chat-completion/
    #           https://api-docs.deepseek.com/guides/tool_calls
    #           https://api-docs.deepseek.com/guides/thinking_mode
    "deepseek": {
        "hosts": ("api.deepseek.com",),
        "json_response_format": True,
        "model_capabilities": ("deepseek_v4",),
        "chat_reasoning": "thinking",
        "chat_reasoning_history": "tool_calls",
        "reasoning_effort_levels": ("low", "high", "xhigh", "max"),
        "prompt_cache_key": False,
        "strict_tools": True,
        "strict_beta": True,
        # Why: DeepSeek's Chat schema only accepts function tools; it has no provider-side tools.
        # Evidence: https://api-docs.deepseek.com/api/create-chat-completion/
        "builtin_tools_by_wire": {},
    },
    # Why: Qwen ignores prior-turn reasoning by default, while tool loops should replay it.
    # Explicit preserve_thinking=true is folded at request time.
    # Evidence: https://platform.qianwenai.com/docs/developer-guides/text-generation/thinking
    "qwen": {
        "hosts": ("aliyuncs.com",),
        "json_response_format": True,
        "model_capabilities": ("qwen3_8",),
        "chat_reasoning_history": "current_turn",
        # Why: Qwen Responses documents web_search/web_extractor as provider-side tools, while
        # Qwen Chat Completions configures search in the request body. The remaining Responses
        # tools need output/resource lifecycle coverage first.
        # Evidence: https://help.aliyun.com/en/model-studio/web-search
        #           https://help.aliyun.com/en/model-studio/web-extractor
        "builtin_tools_by_wire": {"responses": ({"type": "web_search"}, {"type": "web_extractor"})},
    },
    # Why: the international and China Kimi open platforms expose the same model controls
    # on different regional domains. Their temperature values are fixed; explicit
    # thinking.keep="all" is folded at request time.
    # Evidence: https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model
    #           https://platform.kimi.ai/docs/api/models-overview
    #           https://platform.kimi.ai/docs/api/chat
    "kimi_open": {
        "hosts": ("moonshot.ai", "moonshot.cn"),
        "model_capabilities": ("kimi_open",),
        "chat_reasoning_history": "current_turn",
        "reasoning_effort_levels": ("low", "high", "max"),
        "strict_tools": True,
        "suppress_temperature": True,
        # Why: Kimi's builtin functions ($web_search) are Chat tool entries the model calls back.
        # Evidence: https://platform.kimi.ai/docs/guide/use-web-search
        "builtin_tools_by_wire": {"chat": ({"type": "builtin_function", "function": {"name": "$web_search"}},)},
    },
    # Why: Kimi Code is a separate subscription API whose official client tools (WebSearch,
    # FetchURL) run on the client; no coding-endpoint server-tool contract exists.
    # Evidence: https://platform.kimi.ai/docs/api/chat
    "kimi_code": {
        "hosts": ("kimi.com",),
        "model_capabilities": ("kimi_code",),
        "reasoning_effort_levels": ("low", "high", "max"),
        "builtin_tools_by_wire": {},
    },
    # Why: both Z.AI regions share thinking controls and automatic context caching.
    # Evidence: https://docs.z.ai/guides/capabilities/thinking
    #           https://docs.z.ai/guides/capabilities/cache
    "zai": {
        "hosts": ("z.ai",),
        "model_capabilities": ("zai_standard",),
        "chat_reasoning_history": "current_turn",
        "prompt_cache_key": False,
        # Why: Z.AI's web_search entry lives in the Chat tools array; retrieval and server MCP
        # need their own lifecycle handling before they can be offered.
        # Evidence: https://docs.z.ai/guides/tools/web-search
        "builtin_tools_by_wire": {"chat": ({"type": "web_search", "web_search": {}},)},
    },
    # Why: China's BigModel endpoint documents the same thinking and automatic-cache contract.
    # Evidence: https://docs.bigmodel.cn/cn/guide/capabilities/thinking
    #           https://docs.bigmodel.cn/cn/guide/capabilities/cache
    "bigmodel": {
        "hosts": ("bigmodel.cn",),
        "model_capabilities": ("zai_standard",),
        "chat_reasoning_history": "current_turn",
        "prompt_cache_key": False,
        "builtin_tools_by_wire": {"chat": ({"type": "web_search", "web_search": {}},)},
    },
    # Why: Anthropic server tools (web_search_20250305) are Messages tool definitions; only the
    # tested web search version is offered so far. OpenCode Zen documents endpoint routing only,
    # with no gateway server-tool contract, so no provider-side tools are assumed for it.
    # Evidence: https://platform.claude.com/docs/en/build-with-claude/tool-use
    "anthropic": {
        "hosts": ("api.anthropic.com",),
        "builtin_tools_by_wire": {"anthropic": ({"type": "web_search_20250305", "name": "web_search"},)},
    },
}
