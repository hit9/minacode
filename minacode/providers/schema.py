"""Raw JSON catalog shape and the immutable compiled snapshot.

``schema.py`` is the dependency floor of the provider packages: it defines the JSON
``TypedDict`` raw shapes, the error taxonomy, and the immutable dataclasses that
``CatalogCodec`` compiles raw documents into. Nothing here does IO or knows a provider
or model by name; it only describes the shape a valid catalog may take.

Two layers, mirroring the JSON contract in PROVIDER_CATALOG_SPEC.md:

* ``Raw*`` TypedDicts — the parsed ``catalog.json`` document, exactly as written.
* the frozen dataclasses — the compiled snapshot ``CatalogSnapshot`` that codec and
  resolver pass around. Arrays are tuples, mappings are read-only views, and every
  nested value is immutable, so nothing can mutate the active snapshot in place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal, NotRequired, Protocol, TypedDict

Json = dict[str, object]
CatalogSource = Literal["bundled", "cached"]
CATALOG_MAINTENANCE_SCOPE = "well_known_and_necessary_specializations_only"


class RecipeContext(Protocol):
    """The read-only key lookup a recipe condition needs (a dict or ``RequestPolicyContext``)."""

    def get(self, key: str) -> object: ...


class CatalogError(Exception):
    """Base class for every catalog failure, so callers can catch one family."""


class CatalogFormatError(CatalogError):
    """The JSON document is malformed: types, dates, regex, references, or a semantic invariant."""


class CatalogSourceError(CatalogError):
    """The bundled or cached source could not be read."""


class CatalogSyncError(CatalogError):
    """A remote sync failed: timeout, HTTP, oversize, ETag, or the atomic write."""


class CatalogVersionConflict(CatalogSyncError):
    """Two remote/current documents share a version but differ in canonical content."""


# ---------------------------------------------------------------------------
# Raw JSON shapes (TypedDict). Keys are dotted policy paths or selector fields;
# the JSON document is the single source of provider/model knowledge.
# ---------------------------------------------------------------------------


class RawSelector(TypedDict, total=False):
    prefixes: NotRequired[list[str]]
    pattern: NotRequired[str]
    tokens_any: NotRequired[list[str]]
    tokens_all: NotRequired[list[str]]
    token_separator: NotRequired[str]
    version: NotRequired[RawVersionSelector]


class RawVersionSelector(TypedDict):
    pattern: str
    min_inclusive: NotRequired[list[int]]
    max_inclusive: NotRequired[list[int]]
    max_exclusive: NotRequired[list[int]]


class RawPolicyRule(TypedDict):
    id: str
    match: NotRequired[RawSelector]
    set: dict[str, object]
    description: NotRequired[str]
    why: NotRequired[str]
    evidence: NotRequired[list[str]]
    notes: NotRequired[list[str]]


class RawProvider(TypedDict):
    id: str
    hosts: list[str]
    model_rule_modes: NotRequired[dict[str, str]]
    defaults: NotRequired[dict[str, object]]
    model_rules: NotRequired[list[RawPolicyRule]]
    builtin_tools_by_wire: NotRequired[dict[str, list[Json]]]
    description: NotRequired[str]
    why: NotRequired[str]
    evidence: NotRequired[list[str]]
    notes: NotRequired[list[str]]


class RawModelIdForm(TypedDict):
    id: str
    separator: str
    vendors: list[str]
    description: NotRequired[str]
    why: NotRequired[str]
    evidence: NotRequired[list[str]]
    notes: NotRequired[list[str]]


class RawRecipeAction(TypedDict):
    path: list[str]
    value: object


class RawRecipeStep(TypedDict, total=False):
    when: NotRequired[dict[str, object]]
    set: NotRequired[list[RawRecipeAction]]
    remove: NotRequired[list[list[str]]]


class RawRequestRecipe(TypedDict):
    id: NotRequired[str]
    pins_temperature: NotRequired[bool]
    steps: list[RawRecipeStep]
    description: NotRequired[str]


class RawDefaults(TypedDict):
    effort_order: list[str]
    thinking_budgets: dict[str, int]
    reasoning_dialects: dict[str, str]
    wire_defaults: dict[str, Json]
    provider_policy: dict[str, object]


class RawCatalog(TypedDict):
    schema_version: int
    version: int
    updated_at: str
    maintenance_scope: str
    defaults: RawDefaults
    model_id_forms: list[RawModelIdForm]
    request_recipes: dict[str, RawRequestRecipe]
    model_rules: list[RawPolicyRule]
    providers: list[RawProvider]


# ---------------------------------------------------------------------------
# Compiled immutable snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionSelector:
    pattern: re.Pattern[str]
    min_inclusive: tuple[int, int] | None = None
    max_inclusive: tuple[int, int] | None = None
    max_exclusive: tuple[int, int] | None = None

    def extract(self, model: str) -> tuple[int, int] | None:
        match = self.pattern.search(model)
        if match is None:
            return None
        major = match.group("major")
        minor = match.group("minor") or "0"
        try:
            return int(major), int(minor)
        except ValueError:
            return None

    def matches(self, model: str) -> bool:
        version = self.extract(model)
        if version is None:
            return False
        if self.min_inclusive is not None and version < self.min_inclusive:
            return False
        if self.max_inclusive is not None and version > self.max_inclusive:
            return False
        return self.max_exclusive is None or version < self.max_exclusive


@dataclass(frozen=True)
class Selector:
    """The fixed, verifiable selector every policy rule shares.

    Fields within one selector are AND; ``prefixes``/``tokens_any`` arrays are OR.
    An empty selector is illegal (only a generic default rule may carry none).
    """

    prefixes: tuple[str, ...] = ()
    pattern: re.Pattern[str] | None = None
    tokens_any: tuple[str, ...] = ()
    tokens_all: tuple[str, ...] = ()
    token_separator: str = ""
    version: VersionSelector | None = None

    def matches(self, model: str) -> bool:
        lowered = model.lower()
        if self.prefixes and not any(lowered.startswith(prefix.lower()) for prefix in self.prefixes):
            return False
        if self.pattern is not None and self.pattern.match(lowered) is None:
            return False
        if self.tokens_any:
            tokens = set(lowered.split(self.token_separator))
            if not tokens.intersection(self.tokens_any):
                return False
        if self.tokens_all and not set(self.tokens_all).issubset(lowered.split(self.token_separator)):
            return False
        return self.version is None or self.version.matches(lowered)


@dataclass(frozen=True)
class PolicyRule:
    """One model rule: a selector and the policy paths it sets, plus its knowledge trail."""

    id: str
    selector: Selector | None
    set: Mapping[str, object]
    why: str = ""
    evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelIdForm:
    """How a canonical ``vendor/model`` gateway id is recognized."""

    id: str
    separator: str
    vendors: frozenset[str]


@dataclass(frozen=True)
class RecipeCondition:
    """Compiled ``when``: a mapping of context key to an eq value, ``in`` list, or ``present`` flag."""

    eq: Mapping[str, object]
    contains: Mapping[str, tuple[object, ...]]
    present: Mapping[str, bool]

    @classmethod
    def from_raw(cls, when: Mapping[str, object]) -> RecipeCondition:
        """Freeze a validated raw condition for top-level steps and nested ``case`` arms."""

        eq: dict[str, object] = {}
        contains: dict[str, tuple[object, ...]] = {}
        present: dict[str, bool] = {}
        for key, condition in when.items():
            if isinstance(condition, Mapping):
                if "in" in condition:
                    contains[key] = tuple(condition["in"])
                if "present" in condition:
                    present[key] = bool(condition["present"])
                if "eq" in condition:
                    eq[key] = condition["eq"]
            else:
                eq[key] = condition
        return cls(eq=eq, contains=contains, present=present)

    def matches(self, context: RecipeContext) -> bool:
        for key, expected in self.eq.items():
            if context.get(key) != expected:
                return False
        for key, values in self.contains.items():
            if context.get(key) not in values:
                return False
        for key, expected in self.present.items():
            if (context.get(key) is not None) != expected:
                return False
        return True


@dataclass(frozen=True)
class RecipeAction:
    path: tuple[str, ...]
    value: object


@dataclass(frozen=True)
class RecipeStep:
    when: RecipeCondition | None
    set: tuple[RecipeAction, ...] = ()
    remove: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RequestRecipe:
    id: str
    steps: tuple[RecipeStep, ...] = ()
    pins_temperature: bool = False


@dataclass(frozen=True)
class CatalogDefaults:
    effort_order: tuple[str, ...]
    thinking_budgets: Mapping[str, int]
    reasoning_dialects: Mapping[str, str]
    wire_defaults: Mapping[str, Mapping[str, object]]
    provider_policy: Mapping[str, object]


@dataclass(frozen=True)
class ProviderRule:
    """One provider overlay: the host domains it applies on and its policy deltas."""

    id: str
    hosts: tuple[str, ...]
    model_rule_modes: Mapping[str, str]
    defaults: Mapping[str, object]
    model_rules: tuple[PolicyRule, ...]
    builtin_tools_by_wire: Mapping[str, tuple[Mapping[str, object], ...]] | None
    why: str = ""
    evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def ignores(self, namespace: str) -> bool:
        return self.model_rule_modes.get(namespace) == "ignore"


@dataclass(frozen=True)
class CatalogSnapshot:
    """The immutable, compiled form of one whole catalog document."""

    schema_version: int
    version: int
    updated_at: date
    maintenance_scope: str
    defaults: CatalogDefaults
    model_id_forms: tuple[ModelIdForm, ...]
    request_recipes: Mapping[str, RequestRecipe]
    model_rules: tuple[PolicyRule, ...]
    providers: tuple[ProviderRule, ...]
    source: CatalogSource
    content_hash: str
