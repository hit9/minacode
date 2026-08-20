"""Compile provider/model catalog data and resolve protocol compatibility.

The catalog describes documented host/model differences. This module applies generic matching and
effort fallback; Chat, Responses, and Anthropic paths remain responsible for their wire formats.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .catalog import (
    ANTHROPIC_MODELS,
    MODEL_CAPABILITIES,
    PROVIDER_CATALOG,
    REASONING_LEVELS,
    BuiltinToolRuleData,
    ModelEffortRuleData,
    ModelRuleData,
    ProviderData,
)


@dataclass(frozen=True)
class ModelMatch:
    """The model selector every catalog rule shares: family prefixes or a documented pattern."""

    prefixes: tuple[str, ...] = ()
    pattern: str = ""

    def matches(self, model: str) -> bool:
        return any(model.startswith(prefix) for prefix in self.prefixes) or bool(self.pattern and re.match(self.pattern, model))


@dataclass(frozen=True)
class ModelRule(ModelMatch):
    """A value selected by model-family prefixes or a documented version pattern."""

    value: str = ""


@dataclass(frozen=True)
class ModelEffortRule(ModelMatch):
    """Supported normalized efforts selected by model family."""

    levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatibilityProfile:
    """Only the documented ways a host differs from generic protocol behavior."""

    api_rules: tuple[ModelRule, ...] = ()
    chat_reasoning: str | None = None
    chat_reasoning_rules: tuple[ModelRule, ...] = ()
    chat_reasoning_history: str = "all"
    chat_reasoning_history_rules: tuple[ModelRule, ...] = ()
    reasoning_effort_levels: tuple[str, ...] = ()
    reasoning_effort_level_rules: tuple[ModelEffortRule, ...] = ()
    reasoning_effort_off_rules: tuple[ModelRule, ...] = ()
    responses_reasoning_effort_off_rules: tuple[ModelRule, ...] = ()
    responses_reasoning_models: tuple[str, ...] | None = None
    prompt_cache_key: bool = True
    # Chat-Completions `response_format={"type":"json_object"}`. Opt-in, not a default: an
    # OpenAI-compatible gateway that does not implement it answers 400, and the only caller is
    # compaction, whose failure is a silent downgrade to deterministic trimming. Off is the safe
    # unknown -- the prompt reminder and the retry still apply everywhere.
    json_response_format: bool = False
    strict_tools: bool = False
    strict_beta: bool = False
    suppress_temperature: bool = False
    suppress_temperature_models: tuple[str, ...] = ()
    # Provider-side tool policy: which resolved wires may carry which provider-native JSON
    # subsets. ``None`` keeps generic pass-through for unknown hosts; an empty mapping means no
    # wire accepts tools.
    builtin_tools_by_wire: dict[str, tuple[BuiltinToolRuleData, ...]] | None = None

    @staticmethod
    def rule_value(rules: tuple[ModelRule, ...], model: str) -> str | None:
        return next((rule.value for rule in rules if rule.matches(model)), None)

    def reasoning_effort_value(self, model: str, effort: str) -> str:
        levels = next((rule.levels for rule in self.reasoning_effort_level_rules if rule.matches(model)), self.reasoning_effort_levels)
        return nearest_reasoning_effort(effort, levels)


@dataclass(frozen=True)
class ResolvedProvider:
    """The effective transport policy after explicit config and compatibility are folded."""

    api: str
    base_url: str
    host: str
    chat_reasoning: str
    chat_reasoning_history: str
    reasoning_effort: str | None
    responses_reasoning: bool
    suppress_temperature: bool
    prompt_cache_key: bool
    strict_tools_active: bool
    builtin_tools_by_wire: dict[str, tuple[BuiltinToolRuleData, ...]] | None = None
    # Defaulted, and last: an unknown or hand-built provider must land on "no constrained
    # decoding" rather than fail to construct.
    json_response_format: bool = False


@dataclass(frozen=True)
class BuiltinToolsIssue:
    """One known-provider incompatibility, ready for request and command feedback."""

    reason: Literal["wire", "entry"]
    configured: tuple[str, ...]
    supported_wires: tuple[str, ...] = ()
    supported_entries: tuple[str, ...] = ()


def _builtin_tool_label(entry: Mapping[str, object]) -> str:
    tool_type = str(entry.get("type") or "?")
    function = entry.get("function")
    if tool_type == "builtin_function" and isinstance(function, Mapping):
        name = str(function.get("name") or "")
        if name:
            return f"{tool_type}/{name}"
    requirements = []
    for key, value in entry.items():
        if key == "type":
            continue
        requirements.append(f"{key} object" if isinstance(value, Mapping) else f"{key}={value}")
    if requirements:
        return f"{tool_type} ({', '.join(requirements)})"
    return tool_type


def _matches_builtin_tool_rule(entry: Mapping[str, object], rule: Mapping[str, object]) -> bool:
    """Whether entry contains every literal or nested field required by a catalog rule."""

    for key, expected in rule.items():
        if key not in entry:
            return False
        actual = entry[key]
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or not _matches_builtin_tool_rule(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def builtin_tools_issue(resolved: ResolvedProvider, entries: tuple[Mapping[str, object], ...]) -> BuiltinToolsIssue | None:
    """Return the known compatibility problem, while leaving unknown hosts on pass-through."""

    policy = resolved.builtin_tools_by_wire
    if policy is None or not entries:
        return None
    configured = tuple(_builtin_tool_label(entry) for entry in entries)
    rules = policy.get(resolved.api)
    if rules is None:
        return BuiltinToolsIssue("wire", configured, supported_wires=tuple(sorted(policy)))
    unsupported = tuple(_builtin_tool_label(entry) for entry in entries if not any(_matches_builtin_tool_rule(entry, rule) for rule in rules))
    if unsupported:
        return BuiltinToolsIssue("entry", unsupported, supported_entries=tuple(_builtin_tool_label(rule) for rule in rules))
    return None


def nearest_reasoning_effort(effort: str, supported: tuple[str, ...]) -> str:
    """Return the closest supported normalized effort, preferring the higher level on a tie."""

    if effort not in REASONING_LEVELS:
        return effort
    ranks = {level: rank for rank, level in enumerate(REASONING_LEVELS)}
    candidates = tuple(level for level in supported if level in ranks)
    if not candidates:
        return effort
    target = ranks[effort]
    return min(candidates, key=lambda level: (abs(ranks[level] - target), -ranks[level]))


def _model_rules(*groups: tuple[ModelRuleData, ...]) -> tuple[ModelRule, ...]:
    return tuple(ModelRule(rule.get("prefixes", ()), rule.get("pattern", ""), rule["value"]) for group in groups for rule in group)


def _effort_rules(*groups: tuple[ModelEffortRuleData, ...]) -> tuple[ModelEffortRule, ...]:
    return tuple(ModelEffortRule(rule.get("prefixes", ()), rule.get("pattern", ""), rule["levels"]) for group in groups for rule in group)


ModelRuleField = Literal[
    "api_rules",
    "chat_reasoning_rules",
    "chat_reasoning_history_rules",
    "reasoning_effort_off_rules",
    "responses_reasoning_effort_off_rules",
]


def _capability_model_rules(data: ProviderData, field: ModelRuleField) -> tuple[tuple[ModelRuleData, ...], ...]:
    groups: list[tuple[ModelRuleData, ...]] = []
    for name in data.get("model_capabilities", ()):
        capability = MODEL_CAPABILITIES[name]
        groups.append(cast(tuple[ModelRuleData, ...], capability.get(field, ())))
    return tuple(groups)


def _capability_effort_rules(data: ProviderData) -> tuple[tuple[ModelEffortRuleData, ...], ...]:
    return tuple(MODEL_CAPABILITIES[name].get("reasoning_effort_level_rules", ()) for name in data.get("model_capabilities", ()))


def _compatibility_profile(data: ProviderData) -> CompatibilityProfile:
    """Compile one provider overlay and its reusable model capability sets."""

    return CompatibilityProfile(
        api_rules=_model_rules(data.get("api_rules", ()), *_capability_model_rules(data, "api_rules")),
        chat_reasoning=data.get("chat_reasoning"),
        chat_reasoning_rules=_model_rules(data.get("chat_reasoning_rules", ()), *_capability_model_rules(data, "chat_reasoning_rules")),
        chat_reasoning_history=data.get("chat_reasoning_history", "all"),
        chat_reasoning_history_rules=_model_rules(data.get("chat_reasoning_history_rules", ()), *_capability_model_rules(data, "chat_reasoning_history_rules")),
        reasoning_effort_levels=data.get("reasoning_effort_levels", ()),
        reasoning_effort_level_rules=_effort_rules(data.get("reasoning_effort_level_rules", ()), *_capability_effort_rules(data)),
        reasoning_effort_off_rules=_model_rules(data.get("reasoning_effort_off_rules", ()), *_capability_model_rules(data, "reasoning_effort_off_rules")),
        responses_reasoning_effort_off_rules=_model_rules(
            data.get("responses_reasoning_effort_off_rules", ()),
            *_capability_model_rules(data, "responses_reasoning_effort_off_rules"),
        ),
        responses_reasoning_models=data.get("responses_reasoning_models"),
        prompt_cache_key=data.get("prompt_cache_key", True),
        json_response_format=data.get("json_response_format", False),
        strict_tools=data.get("strict_tools", False),
        strict_beta=data.get("strict_beta", False),
        suppress_temperature=data.get("suppress_temperature", False),
        suppress_temperature_models=data.get("suppress_temperature_models", ()),
        builtin_tools_by_wire=data.get("builtin_tools_by_wire"),
    )


def _compatibility_profiles(catalog: Mapping[str, ProviderData] = PROVIDER_CATALOG) -> dict[str, CompatibilityProfile]:
    profiles: dict[str, CompatibilityProfile] = {}
    for data in catalog.values():
        profile = _compatibility_profile(data)
        for host in data["hosts"]:
            if host in profiles:
                raise ValueError(f"duplicate provider compatibility host: {host}")
            profiles[host] = profile
    return profiles


COMPATIBILITY_PROFILES = _compatibility_profiles()
_FAMILY_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def anthropic_model_version(model: str) -> tuple[int, int] | None:
    """Return the first short numeric generation in a Claude model id, if present."""

    tokens = [token for token in _FAMILY_SPLIT_RE.split(model.lower()) if token]
    for index, token in enumerate(tokens):
        if not (token.isdigit() and len(token) <= 2):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        minor = int(following) if following.isdigit() and len(following) <= 2 else 0
        return int(token), minor
    return None


def anthropic_thinking_params(model: str, reasoning: str, effort: str, budget_tokens: int) -> dict[str, object]:
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
    families = _FAMILY_SPLIT_RE.split(model.lower())
    adaptive = version >= ANTHROPIC_MODELS["adaptive_min_version"]
    always_thinking = any(family in families for family in ANTHROPIC_MODELS["always_thinking_families"])
    if reasoning == "off":
        return {"thinking": {"type": "disabled"}} if adaptive and not always_thinking else {}
    level = nearest_reasoning_effort(effort, ANTHROPIC_MODELS["effort_levels"]) if effort in REASONING_LEVELS else "high"
    if not adaptive:
        params: dict[str, object] = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
        if version == (4, 5) and "opus" in families:
            params["output_config"] = {"effort": level if level in ("low", "medium", "high") else "high"}
        return params
    if level == "xhigh" and version < ANTHROPIC_MODELS["xhigh_min_version"]:
        level = "max"
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}


def anthropic_thinking_always_on(model: str) -> bool:
    families = _FAMILY_SPLIT_RE.split(model.lower())
    return any(family in families for family in ANTHROPIC_MODELS["always_thinking_families"])


def anthropic_keeps_prior_thinking(model: str) -> bool:
    """Whether Claude keeps earlier turns' thinking in its effective context."""

    # Opus 4.5 and all numbered 4.6+ models preserve and bill all prior thinking. Sonnet/Haiku
    # 4.5 and earlier models keep only the latest turn; unknown aliases stay conservative.
    # Current-turn thinking blocks are required for tool use regardless of this distinction.
    # Evidence: https://platform.claude.com/docs/en/build-with-claude/thinking
    version = anthropic_model_version(model)
    if version is None:
        return True
    families = _FAMILY_SPLIT_RE.split(model.lower())
    return version >= ANTHROPIC_MODELS["adaptive_min_version"] or (version == (4, 5) and "opus" in families)


def compatibility_for_host(host: str, profiles: Mapping[str, CompatibilityProfile] = COMPATIBILITY_PROFILES) -> CompatibilityProfile:
    """Return the most specific domain profile while respecting label boundaries."""

    matches = ((domain, profile) for domain, profile in profiles.items() if host == domain or host.endswith(f".{domain}"))
    return max(matches, key=lambda item: len(item[0]), default=("", CompatibilityProfile()))[1]
