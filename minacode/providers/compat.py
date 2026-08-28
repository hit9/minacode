"""Compile a catalog snapshot and resolve protocol compatibility.

``catalog.json`` describes documented host/model differences. This module applies generic matching
and effort fallback, then folds explicit config, provider overlays, and generic defaults into one
request policy (``ProviderPolicy.resolve``) and applies request recipes (``RequestRuleEngine``).
Chat, Responses, and Anthropic paths remain responsible for their wire formats; the policy only
answers what the request body must carry.

Nothing here names a provider or model; every fact lives in the JSON snapshot. The Python left
behind is the generic part: selector matching, effort fallback, precedence folding, and the
whitelisted recipe interpreter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlparse

from .catalog import decode_bundled
from .schema import (
    CatalogSnapshot,
    PolicyRule,
    ProviderRule,
    RecipeCondition,
    RecipeValue,
    RequestRecipe,
)

Json = dict[str, object]


def _raw_condition(when: Mapping[str, object]) -> RecipeCondition:
    """Compile a raw ``when`` mapping (step or case) into the schema's condition form."""

    eq: dict[str, object] = {}
    contains: dict[str, tuple[object, ...]] = {}
    present: dict[str, bool] = {}
    for key, condition in when.items():
        if isinstance(condition, dict):
            if "in" in condition:
                contains[key] = tuple(condition["in"])
            if "present" in condition:
                present[key] = bool(condition["present"])
            if "eq" in condition:
                eq[key] = condition["eq"]
        else:
            eq[key] = condition
    return RecipeCondition(eq=eq, contains=contains, present=present)


# ---------------------------------------------------------------------------
# Generic effort fallback (the only Python between a stored effort and the wire)
# ---------------------------------------------------------------------------


def nearest_reasoning_effort(effort: str, supported: tuple[str, ...], effort_order: tuple[str, ...]) -> str:
    """Return the closest supported normalized effort, preferring the higher level on a tie.

    ``effort_order`` is the catalog's normalized scale, so the rank comparison itself is data-driven.
    """

    if effort not in effort_order:
        return effort
    ranks = {level: rank for rank, level in enumerate(effort_order)}
    candidates = tuple(level for level in supported if level in ranks)
    if not candidates:
        return effort
    target = ranks[effort]
    return min(candidates, key=lambda level: (abs(ranks[level] - target), -ranks[level]))


def nearest_supported_effort(effort: str, levels: tuple[str, ...], effort_order: tuple[str, ...]) -> str:
    """Move an effort onto a model's own scale, so a stored choice is never one it cannot take.

    This runs when the scale changes under a stored effort -- a ``/model`` switch, a config written
    against a different model -- and not on the way to the provider. What is offered is what a model
    accepts, so a chosen effort is already on the scale and reaches the wire as written; a request
    that quietly sent something other than the level on screen was the thing worth removing.

    A scale of names minacode does not recognize has no comparable order, so an effort off it lands
    on the middle entry rather than on a guessed rank.
    """

    if not levels or effort in levels:
        return effort
    nearest = nearest_reasoning_effort(effort, levels, effort_order)
    return nearest if nearest in levels else levels[len(levels) // 2]


# ---------------------------------------------------------------------------
# Compiled resolution result and builtin-tool feedback
# ---------------------------------------------------------------------------


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
    builtin_tools_by_wire: Mapping[str, tuple[dict[str, object], ...]] | None = None
    # Defaulted, and last: an unknown or hand-built provider must land on "no constrained
    # decoding" rather than fail to construct.
    json_response_format: bool = False
    # Static text-only evidence folded from the catalog. Learned session evidence lives in the
    # session image-routing state and is combined at delivery time; this is only the catalog half.
    text_only: bool = False
    # The request recipe chosen for this model (see request_recipes), and whether the model is
    # documented as always reasoning. ``reasoning_mandatory`` gates recipe branches that must not
    # turn thinking off, and drives temperature suppression on wires that pin it.
    reasoning_recipe: str = "off"
    reasoning_mandatory: bool = False
    catalog_version: int = 0


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


# ---------------------------------------------------------------------------
# CompatibilityResolver: host lookup + per-field precedence
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from minacode.config import ProviderConfig


class CompatibilityResolver:
    """Compile a snapshot into host overlays and answer per-field policy questions.

    Precedence, per policy field, is fixed (see PROVIDER_CATALOG_SPEC.md 5.3):
    provider ``model_rules``, then top-level ``model_rules`` (unless the provider declares the
    namespace ``ignore``), then the provider's ``defaults``, then the generic
    ``defaults.provider_policy``. The algorithm contains no provider or model name.
    """

    def __init__(self, snapshot: CatalogSnapshot):
        self.snapshot = snapshot
        self._host_providers: dict[str, ProviderRule] = {}
        for provider in snapshot.providers:
            for host in provider.hosts:
                self._host_providers[host] = provider

    # -- host matching -------------------------------------------------------

    def provider_for_host(self, host: str) -> ProviderRule | None:
        """The most specific domain overlay while respecting DNS label boundaries."""

        matches = ((domain, provider) for domain, provider in self._host_providers.items() if host == domain or host.endswith(f".{domain}"))
        return max(matches, key=lambda item: len(item[0]), default=("", None))[1]

    # -- selector matching ---------------------------------------------------

    @staticmethod
    def _rule_matches(rule: PolicyRule, model: str) -> bool:
        return rule.selector is not None and rule.selector.matches(model)

    def _first_setting_rule(self, rules: tuple[PolicyRule, ...], model: str, path: str) -> PolicyRule | None:
        """First rule in JSON order that matches the model and sets ``path``.

        A rule may set several fields (a trait carries recipe + dialect + levels together), while
        an earlier rule may set an unrelated one (the claude mandatory rule sets only
        ``reasoning.mandatory``). The winner for one path is the first matching rule that sets that
        path, so the mandatory rule does not shadow the generation recipes that follow it.
        The full id is matched first; a canonical ``vendor/model`` gateway form is matched by its
        model part only when the vendor prefix is a canonical family slug.
        """

        for rule in rules:
            if self._rule_matches(rule, model) and path in rule.set:
                return rule
        for form in self.snapshot.model_id_forms:
            if form.separator not in model:
                continue
            vendor, _, candidate = model.partition(form.separator)
            if vendor not in form.vendors:
                continue
            for rule in rules:
                if self._rule_matches(rule, candidate) and path in rule.set:
                    return rule
        return None

    # -- field resolution ---------------------------------------------------

    def field_value(self, provider: ProviderRule | None, model: str, path: str) -> object | None:
        """The first value for ``path`` along the fixed precedence, or ``None`` when no source sets it."""

        namespace = path.split(".", 1)[0]
        if provider is not None:
            if (rule := self._first_setting_rule(provider.model_rules, model, path)) is not None:
                return rule.set[path]
            if not provider.ignores(namespace) and (rule := self._first_setting_rule(self.snapshot.model_rules, model, path)) is not None:
                return rule.set[path]
            if path in provider.defaults:
                return provider.defaults[path]
        else:
            if (rule := self._first_setting_rule(self.snapshot.model_rules, model, path)) is not None:
                return rule.set[path]
        if path in self.snapshot.defaults.provider_policy:
            return self.snapshot.defaults.provider_policy[path]
        return None

    def field_rules(self, provider: ProviderRule | None, model: str, path: str) -> tuple[str, tuple[str, ...]]:
        """(why, evidence) of the rule that won ``path`` for this model, or ("", ())."""

        if provider is not None:
            if (rule := self._first_setting_rule(provider.model_rules, model, path)) is not None:
                return rule.why, rule.evidence
            if not provider.ignores(path.split(".", 1)[0]) and (rule := self._first_setting_rule(self.snapshot.model_rules, model, path)) is not None:
                return rule.why, rule.evidence
        else:
            if (rule := self._first_setting_rule(self.snapshot.model_rules, model, path)) is not None:
                return rule.why, rule.evidence
        return "", ()

    # -- text-only model evidence (global, vendor-aware) --------------------

    def text_only(self, model: str) -> bool:
        """Whether a configured model ID resolves to documented static text-only evidence.

        The full ID is matched first; a canonical ``vendor/model`` gateway form (OpenRouter/OpenCode)
        is matched by its model part only when the vendor prefix is one of the canonical family
        slugs, so a custom alias or an unknown host stays unknown and is probed on the main model.
        """

        rule = self._first_setting_rule(self.snapshot.model_rules, model, "image.input")
        return bool(rule and rule.set.get("image.input"))


# ---------------------------------------------------------------------------
# RequestRuleEngine: the whitelisted recipe interpreter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestPolicyContext:
    """Everything a recipe may read. ``resolved_effort`` is the off spelling when reasoning is off."""

    wire: str
    reasoning_enabled: bool
    resolved_effort: str | None
    off_value: str | None = None
    max_tokens: int | None = None
    reasoning_mandatory: bool = False
    temperature: float | None = None
    model: str = ""

    def get(self, key: str) -> object:
        if key == "wire":
            return self.wire
        if key == "reasoning_enabled":
            return self.reasoning_enabled
        if key == "resolved_effort":
            return self.resolved_effort
        if key == "off_value":
            return self.off_value
        if key == "max_tokens":
            return self.max_tokens
        if key == "reasoning_mandatory":
            return self.reasoning_mandatory
        if key == "temperature":
            return self.temperature
        if key == "model":
            return self.model
        return None


class RequestRuleEngine:
    """Apply a request recipe to a constructed request body.

    The interpreter is deliberately tiny: conditions are ``eq``/``in``/``present`` on a fixed context,
    values are literal/source/case/lookup/bounded_budget, and actions are ``set``/``remove`` on
    explicit paths under the body or ``extra_body``. No headers, URLs, filesystem, SDK parameters,
    or callables are reachable.
    """

    def __init__(self, thinking_budgets: Mapping[str, int]):
        self._tables: dict[str, Mapping[str, object]] = {"thinking_budgets": thinking_budgets}

    # -- value evaluation ---------------------------------------------------

    def _lookup(self, raw: object, context: RequestPolicyContext) -> object | None:
        if not isinstance(raw, Mapping):
            return None
        table = self._tables.get(str(raw.get("table") or ""))
        if table is None:
            return None
        key = context.get(str(raw.get("key") or "")) if raw.get("key") else raw.get("default")
        value = table.get(str(key)) if key is not None else None
        return value if value is not None else table.get(str(raw.get("default") or ""))

    def _bounded_budget(self, raw: object, context: RequestPolicyContext) -> int | None:
        """Integer from a table, clamped to ``minimum <= value <= max_tokens - headroom``."""

        if not isinstance(raw, Mapping):
            return None
        table = self._tables.get(str(raw.get("table") or ""))
        if table is None:
            return None
        minimum = int(raw.get("minimum") or 0)
        headroom = int(raw.get("headroom") or 0)
        value = table.get(str(context.resolved_effort or "")) or table.get("medium")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = int(value)
        if context.max_tokens is not None:
            return max(minimum, min(max(0, context.max_tokens - headroom), value))
        return max(minimum, value)

    def _eval_case(self, raw: object, context: RequestPolicyContext) -> object:
        if not isinstance(raw, Mapping):
            return None
        cases = raw.get("case")
        if not isinstance(cases, list):
            return None
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            condition = case.get("when")
            if isinstance(condition, Mapping) and _raw_condition(condition).matches(context):
                return self._eval_value(case.get("then"), context)
        return self._eval_value(raw.get("else"), context)

    def _eval_value(self, value: object, context: RequestPolicyContext) -> object:
        if isinstance(value, Mapping):
            if "source" in value and len(value) == 1:
                return context.get(str(value["source"]))
            if "case" in value:
                return self._eval_case(value, context)
            if "lookup" in value and len(value) == 1:
                return self._lookup(value["lookup"], context)
            if "bounded_budget" in value and len(value) == 1:
                return self._bounded_budget(value["bounded_budget"], context)
            return {key: self._eval_value(item, context) for key, item in value.items()}
        if isinstance(value, list):
            return [self._eval_value(item, context) for item in value]
        return value

    def evaluate(self, value: RecipeValue, context: RequestPolicyContext) -> object:
        """Evaluate a compiled recipe value; nested special leaves inside literals resolve too."""

        if value.kind == "source":
            return context.get(str(value.raw))
        if value.kind == "lookup":
            return self._lookup(value.raw, context)
        if value.kind == "bounded_budget":
            return self._bounded_budget(value.raw, context)
        if value.kind == "case":
            return self._eval_case(value.raw, context)
        return self._eval_value(value.raw, context)

    # -- actions ------------------------------------------------------------

    @staticmethod
    def _set(params: dict[str, object], path: tuple[str, ...], value: object) -> None:
        if not path:
            return
        node: dict[str, object] = params
        for key in path[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        leaf = path[-1]
        existing = node.get(leaf)
        if isinstance(existing, dict) and isinstance(value, dict):
            # Shallow merge: user-configured extra_body fields stay authoritative on key conflicts
            # outside the managed path, while the recipe's own keys win where they both set one.
            node[leaf] = {**existing, **value}
        else:
            node[leaf] = value

    @staticmethod
    def _remove(params: dict[str, object], path: tuple[str, ...]) -> None:
        node: dict[str, object] | None = params
        for key in path[:-1]:
            if not isinstance(node, dict) or not isinstance(node.get(key), dict):
                return
            node = cast(dict[str, object], node[key])
        if isinstance(node, dict):
            node.pop(path[-1], None)

    def apply(self, params: dict[str, object], recipe: RequestRecipe, context: RequestPolicyContext) -> dict[str, object]:
        """Apply the recipe's steps in order, mutating and returning ``params``."""

        for step in recipe.steps:
            if step.when is not None and not step.when.matches(context):
                continue
            for action in step.set:
                self._set(params, action.path, self.evaluate(action.value, context))
            for path in step.remove:
                self._remove(params, path)
        return params


# ---------------------------------------------------------------------------
# ProviderPolicy: the public entry a session holds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Explicit-dialect escape hatch
# ---------------------------------------------------------------------------

# When the catalog names no recipe for a provider/model, an explicitly configured
# `chat_reasoning` dialect picks the wire format itself, so the request-recipe data is still the
# only description of how each format is built (see ProviderPolicy.resolve).
_DIALECT_RECIPE_FALLBACK: dict[str, str] = {
    "reasoning": "reasoning.object",
    "reasoning_effort": "reasoning.effort",
    "thinking": "thinking.with-effort",
    "thinking_toggle": "thinking.toggle",
    "thinking_effort": "thinking.effort",
    "enable_thinking": "enable-thinking",
    "mandatory_thinking": "mandatory",
    "off": "off",
}


class PolicyConfig(Protocol):
    """The config surface :class:`ProviderPolicy` reads; ``ProviderConfig`` and test fakes satisfy it."""

    url: str
    model: str
    api: str
    chat_reasoning: str
    reasoning: str
    max_tokens: int
    temperature: float | None
    strict_tools: bool

    def declared_levels(self, model: str = "") -> tuple[str, ...]: ...


class ProviderPolicy:
    """Compile one snapshot into resolver + engine and answer everything config/CLI/model need.

    ``resolve`` is the only place the catalog is queried for a request; ``apply_request`` runs the
    chosen recipe against a body the adapter already assembled. ``reasoning_choices``,
    ``supported_efforts``, ``effort_source``, ``normalized_reasoning`` and ``text_only`` are the
    read-side helpers shared by config, ``/reason``, image routing and tests.
    """

    def __init__(self, snapshot: CatalogSnapshot):
        self.snapshot = snapshot
        self._resolver = CompatibilityResolver(snapshot)
        self._engine = RequestRuleEngine(snapshot.defaults.thinking_budgets)
        self._effort_order = snapshot.defaults.effort_order

    @property
    def effort_order(self) -> tuple[str, ...]:
        return self._effort_order

    # -- helpers shared by config/CLI/tests ---------------------------------

    def _provider_for(self, config: PolicyConfig | None) -> ProviderRule | None:
        url = getattr(config, "url", "")
        host = (urlparse(str(url).rstrip("/")).hostname or "").lower()
        return self._resolver.provider_for_host(host)

    def supported_efforts(self, config: PolicyConfig, model: str = "") -> tuple[str, ...]:
        """The effort levels this model accepts -- what ``/reason`` offers, and all it offers.

        A configured declaration wins, then the catalog, and a model neither knows anything about
        keeps the full scale: unknown means unconstrained, not empty.
        """

        model = (model or str(getattr(config, "model", ""))).lower()
        if declared := config.declared_levels(model):
            return declared
        levels = self._resolver.field_value(self._provider_for(config), model, "reasoning.levels")
        if isinstance(levels, (list, tuple)):
            return tuple(levels)
        return self._effort_order

    def reasoning_mandatory(self, config: PolicyConfig, model: str = "") -> bool:
        model = (model or str(getattr(config, "model", ""))).lower()
        if config.declared_levels(model):
            return False
        return bool(self._resolver.field_value(self._provider_for(config), model, "reasoning.mandatory"))

    def reasoning_choices(self, config: PolicyConfig, model: str = "") -> tuple[str, ...]:
        """Everything ``/reason`` may offer for this model.

        ``off`` is among them unless the model documents that it always reasons -- Grok, Kimi K3,
        GLM-5.3. Listing it there would be the menu promising something the endpoint does not do.
        """

        model = (model or str(getattr(config, "model", ""))).lower()
        mandatory = not config.declared_levels(model) and self.reasoning_mandatory(config, model)
        return self.supported_efforts(config, model) if mandatory else ("off", *self.supported_efforts(config, model))

    def effort_source(self, config: PolicyConfig, model: str = "") -> tuple[str, str]:
        """Why ``/reason`` offers what it offers, as (one line, page) -- empty when default."""

        model = (model or str(getattr(config, "model", ""))).lower()
        if config.declared_levels(model):
            return "declared for this model in your config", ""
        why, evidence = self._resolver.field_rules(self._provider_for(config), model, "reasoning.levels")
        return (why, evidence[0] if evidence else "")

    def reasoning_effort_value(self, config: PolicyConfig) -> str:
        """The stored effort if it is a known normalized level or user-declared, else ``medium``."""

        reasoning = str(getattr(config, "reasoning", "off"))
        if reasoning in self._effort_order or reasoning in config.declared_levels():
            return reasoning
        return "medium"

    def normalized_reasoning(self, config: PolicyConfig, model: str = "") -> str:
        """This entry's effort, moved onto ``model``'s choices if it is not already among them."""

        choices = self.reasoning_choices(config, model)
        if getattr(config, "reasoning", "off") == "off":
            return "off" if "off" in choices else choices[0]
        return nearest_supported_effort(self.reasoning_effort_value(config), self.supported_efforts(config, model), self._effort_order)

    def text_only(self, config: PolicyConfig | None, model: str = "") -> bool:
        return self._resolver.text_only((model or str(getattr(config, "model", ""))).lower())

    # -- resolve ------------------------------------------------------------

    def resolve(self, config: PolicyConfig) -> ResolvedProvider:
        """Fold explicit configuration and documented compatibility into one request policy."""

        url = str(getattr(config, "url", "")).rstrip("/").removesuffix("/chat/completions").removesuffix("/responses").removesuffix("/messages")
        host = (urlparse(url).hostname or "").lower()
        provider = self._resolver.provider_for_host(host)
        model = str(getattr(config, "model", "")).lower()

        api = str(getattr(config, "api", "auto"))
        if api == "auto":
            path = urlparse(str(getattr(config, "url", "")).rstrip("/")).path
            suffix_api = next(
                (value for suffix, value in (("/responses", "responses"), ("/messages", "anthropic"), ("/chat/completions", "chat")) if path.endswith(suffix)),
                None,
            )
            api = str(suffix_api or self._resolver.field_value(provider, model, "api") or "chat")

        chat_reasoning = str(getattr(config, "chat_reasoning", "auto"))
        if chat_reasoning == "auto":
            chat_reasoning = str(self._resolver.field_value(provider, model, "reasoning.dialect") or "off")

        reasoning_recipe = str(self._resolver.field_value(provider, model, "reasoning.recipe") or "off")
        # The explicit `chat_reasoning` config is the escape hatch for gateways the catalog has no
        # evidence about: when the catalog names no recipe, the requested dialect picks the wire
        # format, so an unrecognized endpoint still gets the reasoning/thinking fields it takes.
        # A catalogued recipe wins (the catalog owns a known model's format), which is why this
        # fallback only fires on the generic "off" recipe.
        responses_models = self._resolver.field_value(provider, model, "responses.reasoning_models")
        responses_reasoning = not isinstance(responses_models, (list, tuple)) or any(model.startswith(str(prefix)) for prefix in responses_models)
        if reasoning_recipe == "off":
            explicit_dialect = str(getattr(config, "chat_reasoning", "auto"))
            if explicit_dialect != "auto":
                reasoning_recipe = _DIALECT_RECIPE_FALLBACK.get(explicit_dialect, "off")
            elif api == "responses" and responses_reasoning:
                # The generic Responses default for a model the catalog does not know: a stateless
                # reasoning request carries the effort in the top-level `reasoning` object, matching
                # what the Responses adapter used to add for unknown models.
                reasoning_recipe = "reasoning.effort"

        if getattr(config, "reasoning", "off") == "off":
            reasoning_effort = self._resolver.field_value(provider, model, "reasoning.off")
            if api == "responses":
                reasoning_effort = self._resolver.field_value(provider, model, "reasoning.off_responses") or reasoning_effort
            if reasoning_effort is not None:
                reasoning_effort = str(reasoning_effort)
        else:
            # Sent as chosen. ``/reason`` only offers what this model accepts and a model switch
            # moves a stored effort onto the new scale, so by the time a request is built the
            # effort is already one this model takes -- the last-resort move here is for an entry
            # constructed directly, never for a session that has been through either path.
            reasoning_effort = nearest_supported_effort(self.reasoning_effort_value(config), self.supported_efforts(config, model), self._effort_order)

        suppress_value = self._resolver.field_value(provider, model, "temperature.suppress")
        suppress_models = self._resolver.field_value(provider, model, "temperature.suppress_models")
        suppress_temperature = bool(suppress_value) or any(
            model.startswith(str(prefix)) for prefix in (suppress_models if isinstance(suppress_models, (list, tuple)) else ())
        )
        if not suppress_temperature:
            recipe = self.snapshot.request_recipes.get(reasoning_recipe)
            suppress_temperature = getattr(config, "reasoning", "off") != "off" and bool(recipe and recipe.pins_temperature)

        strict_tools_active = (
            bool(getattr(config, "strict_tools", False)) and bool(self._resolver.field_value(provider, model, "strict.tools")) and api in ("chat", "responses")
        )
        if strict_tools_active and self._resolver.field_value(provider, model, "strict.beta") and not url.endswith("/beta"):
            url += "/beta"

        history = self._resolver.field_value(provider, model, "history.reasoning")
        chat_reasoning_history = str(history or "all")

        return ResolvedProvider(
            api=api,
            base_url=url,
            host=host,
            chat_reasoning=chat_reasoning,
            chat_reasoning_history=chat_reasoning_history,
            reasoning_effort=reasoning_effort,
            responses_reasoning=responses_reasoning,
            suppress_temperature=suppress_temperature,
            prompt_cache_key=bool(self._resolver.field_value(provider, model, "cache.prompt_key")),
            strict_tools_active=strict_tools_active,
            builtin_tools_by_wire=provider.builtin_tools_by_wire if provider is not None else None,
            json_response_format=bool(self._resolver.field_value(provider, model, "json.response_format")),
            text_only=self._resolver.text_only(model),
            reasoning_recipe=reasoning_recipe,
            reasoning_mandatory=self.reasoning_mandatory(config, model),
            catalog_version=self.snapshot.version,
        )

    # -- request recipes ----------------------------------------------------

    def apply_request(self, params: Json, config: object, resolved: ResolvedProvider, *, wire: str) -> Json:
        """Run the resolved recipe against a request body the adapter already assembled.

        ``wire`` is ``chat``, ``responses`` or ``anthropic``; recipes gate their steps on it. The
        body is mutated and returned, so extra_body/thinking merges keep user-configured fields.
        """

        recipe = self.snapshot.request_recipes.get(resolved.reasoning_recipe)
        if recipe is None:
            return params
        reasoning_enabled = getattr(config, "reasoning", "off") != "off"
        # The adapter may already have resolved the effective output cap into params (the anthropic
        # path sets max_tokens = anthropic_output_cap() before this runs), so the budget clamp
        # must compare against the cap that will actually be sent, not just the raw config value.
        max_tokens = params.get("max_tokens")
        if not isinstance(max_tokens, int):
            max_tokens = int(getattr(config, "max_tokens", 0) or 0) or None
        context = RequestPolicyContext(
            wire=wire,
            reasoning_enabled=reasoning_enabled,
            resolved_effort=resolved.reasoning_effort,
            off_value=resolved.reasoning_effort if not reasoning_enabled else None,
            max_tokens=max_tokens,
            reasoning_mandatory=resolved.reasoning_mandatory,
            temperature=getattr(config, "temperature", None),
            model=str(getattr(config, "model", "")).lower(),
        )
        return cast(Json, self._engine.apply(params, recipe, context))

    # -- catalog-dependent config validation --------------------------------

    def validate_config(self, config: PolicyConfig, provider: ProviderConfig | None = None) -> None:
        """Second-phase config checks that need the catalog (reasoning scale, API values)."""

        from minacode.config import REASONING_CHOICES, ConfigError

        reasoning = getattr(config, "reasoning", "off")
        if reasoning == "off":
            return
        if reasoning in self.supported_efforts(config):
            return
        levels = REASONING_CHOICES
        raise ConfigError(f"reasoning must be one of {', '.join(levels)} for {getattr(config, 'model', '') or '(no model)'} on this provider")


# ---------------------------------------------------------------------------
# Bundled-policy accessor for tooling and tests
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def bundled_policy() -> ProviderPolicy:
    """The policy compiled from the bundled catalog.json; shared by tests and offline tooling.

    Sessions compile their own policy from the selected snapshot (see sync.CatalogRuntime); this
    accessor exists so config-free call sites -- tests, ``/catalog`` reporting, image fallback --
    see exactly what the package ships without constructing a runtime.
    """

    return ProviderPolicy(decode_bundled())


def is_text_only_model(model: str) -> bool:
    """Catalog-side text-only evidence for the bundled snapshot (tests and tooling)."""

    return bundled_policy().text_only(None, model)
