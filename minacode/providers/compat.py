"""Compile provider/model catalog data and resolve protocol compatibility.

The catalog describes documented host/model differences. This module applies generic matching and
effort fallback; Chat, Responses, and Anthropic paths remain responsible for their wire formats.

Model facts and endpoint facts are matched separately and meet here. For every reasoning field the
order is:

    this host's model rules  >  the model's trait  >  this host's plain value

A host rule is the narrowest statement, so it wins. A trait beats the host's plain value because
that value is the host's fallback for models it has nothing specific to say about. And a host that
normalizes reasoning takes no traits at all -- see ``CompatibilityProfile.model_traits``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .catalog import (
    ANTHROPIC_MODELS,
    CANONICAL_VENDORS,
    MODEL_TRAITS,
    PROVIDER_CATALOG,
    REASONING_LEVELS,
    TEXT_ONLY_MODEL_RULES,
    BuiltinToolRuleData,
    ModelEffortRuleData,
    ModelRuleData,
    ModelTraitData,
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
class ModelTrait(ModelMatch):
    """How one model family takes reasoning, wherever it is served.

    These facts belong to the model, so they are matched on the model name alone and apply on
    every host: `deepseek-v4-flash` wants thinking.type and its own effort scale whether it comes
    from api.deepseek.com, a gateway, or a self-hosted proxy. Before this existed the same data sat
    inside host entries that each had to claim it by name, which meant an endpoint the catalog had
    never heard of got nothing at all — the common case for a model served by many gateways."""

    chat_reasoning: str = ""
    chat_reasoning_history: str = ""
    reasoning_effort_levels: tuple[str, ...] = ()
    reasoning_effort_off: str = ""


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
    # Documented text-only model families (static evidence, see catalog.TEXT_ONLY_MODEL_RULES).
    # A route matching one of these must never be sent a raw image; anything else stays unknown
    # and is tried on the main model.
    text_only_rules: tuple[ModelRule, ...] = ()
    # Model-keyed reasoning knowledge, carried by every profile including the empty one an unknown
    # host resolves to. Precedence against this host's own settings, per field, is:
    #
    #   this host's model rules  >  the model's trait  >  this host's plain value
    #
    # A host rule is the most specific statement available, so it wins. A trait beats the host's
    # plain value because that value is the host's fallback for models it has nothing specific to
    # say about -- `moonshot.ai` replays only the current turn, except for the K3 family, whose
    # own rule says otherwise. A host that re-encodes reasoning instead of passing each model's
    # native spelling through opts out entirely (`normalizes_reasoning`), because for it no
    # per-field precedence is right: OpenRouter would take DeepSeek's thinking.type and send it
    # to an endpoint that documents only its own reasoning object.
    model_traits: tuple[ModelTrait, ...] = ()

    def trait(self, model: str) -> ModelTrait | None:
        return next((trait for trait in self.model_traits if trait.matches(model)), None)

    @staticmethod
    def rule_value(rules: tuple[ModelRule, ...], model: str) -> str | None:
        return next((rule.value for rule in rules if rule.matches(model)), None)

    def supported_efforts(self, model: str) -> tuple[str, ...]:
        """The effort scale for this model: host rule, else the model's trait, else the host's own."""

        if rule := next((rule for rule in self.reasoning_effort_level_rules if rule.matches(model)), None):
            return rule.levels
        trait = self.trait(model)
        return (trait.reasoning_effort_levels if trait and trait.reasoning_effort_levels else ()) or self.reasoning_effort_levels

    def reasoning_effort_value(self, model: str, effort: str) -> str:
        return nearest_reasoning_effort(effort, self.supported_efforts(model))

    def reasoning_off_value(self, model: str, *, responses: bool = False) -> str | None:
        """The wire spelling of "do not think" for this model, or None when none is documented."""

        rules = self.responses_reasoning_effort_off_rules if responses else self.reasoning_effort_off_rules
        if value := self.rule_value(rules, model):
            return value
        trait = self.trait(model)
        return (trait.reasoning_effort_off or None) if trait else None

    def chat_reasoning_for(self, model: str) -> str:
        trait = self.trait(model)
        return self.rule_value(self.chat_reasoning_rules, model) or (trait.chat_reasoning if trait else "") or self.chat_reasoning or "off"

    def chat_reasoning_history_for(self, model: str) -> str:
        trait = self.trait(model)
        return self.rule_value(self.chat_reasoning_history_rules, model) or (trait.chat_reasoning_history if trait else "") or self.chat_reasoning_history


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
    # Static text-only evidence folded from the catalog. Learned session evidence lives in the
    # session image-routing state and is combined at delivery time; this is only the catalog half.
    text_only: bool = False


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


def nearest_supported_effort(effort: str, levels: tuple[str, ...]) -> str:
    """Move an effort onto a model's own scale, so a stored choice is never one it cannot take.

    This runs when the scale changes under a stored effort — a `/model` switch, a config written
    against a different model — and not on the way to the provider. What is offered is what a model
    accepts, so a chosen effort is already on the scale and reaches the wire as written; a request
    that quietly sent something other than the level on screen was the thing worth removing.

    A scale of names minacode does not recognize has no comparable order, so an effort off it lands
    on the middle entry rather than on a guessed rank."""

    if not levels or effort in levels:
        return effort
    nearest = nearest_reasoning_effort(effort, levels)
    return nearest if nearest in levels else levels[len(levels) // 2]


def _model_rules(*groups: tuple[ModelRuleData, ...]) -> tuple[ModelRule, ...]:
    return tuple(ModelRule(rule.get("prefixes", ()), rule.get("pattern", ""), rule["value"]) for group in groups for rule in group)


# The compiled static text-only negative list. Model evidence is global: a documented text-only
# model stays text-only on any host, including unknown hosts and the default compatibility
# profile, so the default rule set is this one rather than empty.
TEXT_ONLY_RULES = _model_rules(TEXT_ONLY_MODEL_RULES)


def _effort_rules(*groups: tuple[ModelEffortRuleData, ...]) -> tuple[ModelEffortRule, ...]:
    return tuple(ModelEffortRule(rule.get("prefixes", ()), rule.get("pattern", ""), rule["levels"]) for group in groups for rule in group)


def _model_traits(data: ProviderData, traits: tuple[ModelTraitData, ...] = MODEL_TRAITS) -> tuple[ModelTrait, ...]:
    """Compile the model-keyed traits this host may use — none, for a host that normalizes."""

    if data.get("normalizes_reasoning"):
        return ()
    return tuple(
        ModelTrait(
            prefixes=trait.get("prefixes", ()),
            pattern=trait.get("pattern", ""),
            chat_reasoning=trait.get("chat_reasoning", ""),
            chat_reasoning_history=trait.get("chat_reasoning_history", ""),
            reasoning_effort_levels=trait.get("reasoning_effort_levels", ()),
            reasoning_effort_off=trait.get("reasoning_effort_off", ""),
        )
        for trait in traits
    )


def _compatibility_profile(data: ProviderData) -> CompatibilityProfile:
    """Compile one provider overlay and its reusable model capability sets."""

    return CompatibilityProfile(
        api_rules=_model_rules(data.get("api_rules", ())),
        chat_reasoning=data.get("chat_reasoning"),
        chat_reasoning_rules=_model_rules(data.get("chat_reasoning_rules", ())),
        chat_reasoning_history=data.get("chat_reasoning_history", "all"),
        chat_reasoning_history_rules=_model_rules(data.get("chat_reasoning_history_rules", ())),
        # A host's own level list is its fallback vocabulary for models the trait table does not
        # name -- every thinking model on api.deepseek.com takes the same scale, whether or not the
        # catalog knows that model by name.
        reasoning_effort_levels=data.get("reasoning_effort_levels", ()),
        reasoning_effort_level_rules=_effort_rules(data.get("reasoning_effort_level_rules", ())),
        reasoning_effort_off_rules=_model_rules(data.get("reasoning_effort_off_rules", ())),
        responses_reasoning_effort_off_rules=_model_rules(data.get("responses_reasoning_effort_off_rules", ())),
        model_traits=_model_traits(data),
        responses_reasoning_models=data.get("responses_reasoning_models"),
        prompt_cache_key=data.get("prompt_cache_key", True),
        json_response_format=data.get("json_response_format", False),
        strict_tools=data.get("strict_tools", False),
        strict_beta=data.get("strict_beta", False),
        suppress_temperature=data.get("suppress_temperature", False),
        suppress_temperature_models=data.get("suppress_temperature_models", ()),
        builtin_tools_by_wire=data.get("builtin_tools_by_wire"),
        text_only_rules=_model_rules(TEXT_ONLY_MODEL_RULES),
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
# What an unknown endpoint resolves to: no endpoint policy, but the same model knowledge every
# known host gets. Model traits are matched on the model name, so a host the catalog has never
# seen is only missing facts about the host.
GENERIC_PROFILE = _compatibility_profile(cast(ProviderData, {"hosts": ()}))
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


def is_text_only_model(model: str, profile: CompatibilityProfile | None = None) -> bool:
    """Whether a configured model ID resolves to documented static text-only evidence.

    The full ID is matched first; a canonical `vendor/model` gateway form (OpenRouter/OpenCode)
    is matched by its model part only when the vendor prefix is one of the canonical family
    slugs, so a custom alias or an unknown host stays unknown and is probed on the main model.
    The rules are global model evidence, so the default/unknown-host profile falls back to the
    compiled list instead of matching nothing.
    """

    rules = profile.text_only_rules if profile is not None and profile.text_only_rules else TEXT_ONLY_RULES
    if CompatibilityProfile.rule_value(rules, model):
        return True
    if "/" in model:
        vendor, _, candidate = model.partition("/")
        if vendor in CANONICAL_VENDORS and CompatibilityProfile.rule_value(rules, candidate):
            return True
    return False


def compatibility_for_host(host: str, profiles: Mapping[str, CompatibilityProfile] = COMPATIBILITY_PROFILES) -> CompatibilityProfile:
    """Return the most specific domain profile while respecting label boundaries.

    An unmatched host resolves to GENERIC_PROFILE rather than an empty one: it still knows the
    models, it just has nothing to add about the endpoint. That is the ordinary case for a gateway
    or self-hosted proxy serving a well-known model, and matching nothing there is what made a
    catalogued model behave like an unknown one purely because of the domain it arrived from."""

    matches = ((domain, profile) for domain, profile in profiles.items() if host == domain or host.endswith(f".{domain}"))
    return max(matches, key=lambda item: len(item[0]), default=("", GENERIC_PROFILE))[1]
