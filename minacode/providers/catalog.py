"""CatalogCodec: decode, validate, and compile one whole catalog document.

This module owns no provider/model facts — every fact lives in ``catalog.json`` and is
compiled here into the immutable :class:`CatalogSnapshot` the resolver consumes. It does
no IO (``sync.py`` reads files and fetches over the network) and defines no constants
that would change when the catalog updates.

The codec is strict by design: an invalid document is rejected with a
:class:`CatalogFormatError`, never repaired. A corrupted bundled catalog is a broken
install and fails fast; a corrupted cached copy is ignored by the repository and the
bundled snapshot keeps serving.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from typing import cast

from minacode.providers.schema import (
    CatalogDefaults,
    CatalogFormatError,
    CatalogSnapshot,
    CatalogSource,
    ModelIdForm,
    PolicyRule,
    ProviderRule,
    RawCatalog,
    RawPolicyRule,
    RawProvider,
    RawRecipeAction,
    RawRecipeStep,
    RawSelector,
    RecipeAction,
    RecipeCondition,
    RecipeStep,
    RecipeValue,
    RequestRecipe,
    Selector,
    VersionSelector,
)

# Safety bounds (spec 5.3 / 8): regex length, prose sizes, and the safe integer window
# for the document version.
MAX_REGEX_LENGTH = 512
MAX_PROSE_LENGTH = 1000
MAX_EVIDENCE_URL_LENGTH = 2048
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
MAX_VERSION = 2**53 - 1
MAX_POLICY_LEVELS = 32
MAX_RECIPE_STEPS = 64
MAX_RECIPE_PATHS = 16

# The request-body main fields a catalog recipe must never set or remove (spec 9): the
# user's own omit_body stays the last step and can only ever drop, never the recipe.
PROTECTED_BODY_ROOTS = frozenset({"model", "messages", "input"})

# Fixed enums the schema recognizes. Protocol wire names are Python's contract (what the
# client implements); everything else in this set is a closed catalog vocabulary.
POLICY_PATHS = frozenset(
    {
        "api",
        "reasoning.recipe",
        "reasoning.dialect",
        "reasoning.levels",
        "reasoning.off",
        "reasoning.off_responses",
        "reasoning.mandatory",
        "history.reasoning",
        "image.input",
        "responses.reasoning_models",
        "cache.prompt_key",
        "json.response_format",
        "strict.tools",
        "strict.beta",
        "temperature.suppress",
        "temperature.suppress_models",
        "tools.builtin_by_wire",
    }
)
HISTORY_MODES = frozenset({"all", "tool_calls", "current_turn"})
IMAGE_INPUTS = frozenset({"text_only", "auto"})
WIRES = frozenset({"chat", "responses", "anthropic"})
RULE_MODES = frozenset({"inherit", "ignore"})
RECIPE_CONTEXT_KEYS = frozenset({"wire", "reasoning_enabled", "resolved_effort", "off_value", "max_tokens", "reasoning_mandatory", "temperature", "model"})

_EFFORT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class CatalogCodec:
    """Decode raw bytes into a compiled, validated :class:`CatalogSnapshot`."""

    def decode(self, payload: bytes, source: CatalogSource) -> CatalogSnapshot:
        if not payload:
            raise CatalogFormatError("catalog document is empty")
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise CatalogFormatError(f"catalog document exceeds {MAX_DOCUMENT_BYTES} bytes")
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogFormatError(f"catalog is not valid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise CatalogFormatError("catalog top level must be an object")
        raw_catalog = cast(RawCatalog, raw)
        self._validate_document(raw_catalog)
        return self._compile(raw_catalog, source)

    def canonical_hash(self, raw: Mapping[str, object]) -> str:
        """The content hash that decides whether two same-version documents are identical."""
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_document(self, raw: RawCatalog) -> None:
        schema_version = raw.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise CatalogFormatError("catalog.schema_version must be an integer")
        if schema_version != 1:
            raise CatalogFormatError(f"unsupported catalog schema_version {schema_version}; this build supports 1")

        version = raw.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or not 0 < version <= MAX_VERSION:
            raise CatalogFormatError("catalog.version must be a positive integer within the safe range")
        updated_at = raw.get("updated_at")
        if not isinstance(updated_at, str):
            raise CatalogFormatError("catalog.updated_at must be a string date")
        try:
            date.fromisoformat(updated_at)
            if len(updated_at) != 10:
                raise ValueError(updated_at)
        except ValueError as error:
            raise CatalogFormatError(f"catalog.updated_at must be a valid UTC date YYYY-MM-DD, got {updated_at!r}") from error

        required = ("defaults", "model_id_forms", "request_recipes", "model_rules", "providers")
        for key in required:
            if key not in raw:
                raise CatalogFormatError(f"catalog is missing required field {key!r}")

        defaults = raw.get("defaults")
        if not isinstance(defaults, dict):
            raise CatalogFormatError("catalog.defaults must be an object")
        self._validate_defaults(defaults)

        seen_forms: set[str] = set()
        for form in raw.get("model_id_forms", []):
            self._require_unique_id(form, seen_forms, "model_id_forms")
            for key in ("id", "separator"):
                if not isinstance(form.get(key), str) or not form.get(key):
                    raise CatalogFormatError("model_id_forms entries need non-empty string id and separator")
            vendors = form.get("vendors")
            if not isinstance(vendors, list) or not vendors or not all(isinstance(v, str) and v for v in vendors):
                raise CatalogFormatError(f"model_id_forms.{form.get('id', '?')}.vendors must be a non-empty string list")
            if not form.get("separator"):
                raise CatalogFormatError(f"model_id_forms.{form.get('id')}.separator must be non-empty")
            self._validate_prose(form, "model_id_forms." + str(form.get("id")))

        recipes = raw.get("request_recipes")
        if not isinstance(recipes, dict):
            raise CatalogFormatError("catalog.request_recipes must be an object keyed by recipe id")
        seen_recipes: set[str] = set()
        for recipe_id, recipe in recipes.items():
            if not isinstance(recipe, dict):
                raise CatalogFormatError(f"request_recipes.{recipe_id} must be an object")
            rid = recipe.get("id") or recipe_id
            if rid in seen_recipes:
                raise CatalogFormatError(f"duplicate request recipe id {rid!r}")
            seen_recipes.add(rid)
            self._validate_recipe(rid, recipe)

        seen_rules: set[str] = set()
        for rule in raw.get("model_rules", []):
            self._validate_rule(rule, seen_rules, "catalog.model_rules", allow_empty_selector=False)

        seen_hosts: set[str] = set()
        seen_provider_ids: set[str] = set()
        for provider in raw.get("providers", []):
            if not isinstance(provider, dict) or not isinstance(provider.get("id"), str) or not provider["id"]:
                raise CatalogFormatError("catalog.providers entries need a non-empty string id")
            pid = provider["id"]
            if pid in seen_provider_ids:
                raise CatalogFormatError(f"duplicate provider id {pid!r}")
            seen_provider_ids.add(pid)
            hosts = provider.get("hosts")
            if not isinstance(hosts, list) or not hosts or not all(isinstance(h, str) and h for h in hosts):
                raise CatalogFormatError(f"{pid}.hosts must be a non-empty list of strings")
            for host in hosts:
                if host in seen_hosts:
                    raise CatalogFormatError(f"duplicate provider host {host!r}")
                seen_hosts.add(host)
            modes = provider.get("model_rule_modes") or {}
            if not isinstance(modes, dict) or not all(mode in RULE_MODES for mode in modes.values()):
                raise CatalogFormatError(f"{pid}.model_rule_modes must map namespaces to inherit|ignore")
            for namespace in modes:
                if namespace not in {path.split(".", 1)[0] for path in POLICY_PATHS}:
                    raise CatalogFormatError(f"{pid}.model_rule_modes has unknown namespace {namespace!r}")
            provider_defaults = provider.get("defaults")
            if provider_defaults is not None:
                self._validate_policy_values(pid + ".defaults", cast(dict, provider_defaults))
            for rule in provider.get("model_rules", []):
                self._validate_rule(rule, seen_rules, pid + ".model_rules", allow_empty_selector=False)
            tools = provider.get("builtin_tools_by_wire")
            if tools is not None:
                if not isinstance(tools, dict):
                    raise CatalogFormatError(f"{pid}.builtin_tools_by_wire must be an object keyed by wire")
                for wire, entries in tools.items():
                    if wire not in WIRES:
                        raise CatalogFormatError(f"{pid}.builtin_tools_by_wire has unknown wire {wire!r}")
                    self._validate_builtin_entries(f"{pid}.builtin_tools_by_wire.{wire}", entries)
            self._validate_prose(provider, pid)

        # All recipe references from policy rules must exist.
        for rule in raw.get("model_rules", []):
            self._validate_recipe_reference(rule, seen_recipes)
        for provider in raw.get("providers", []):
            for rule in provider.get("model_rules", []):
                self._validate_recipe_reference(rule, seen_recipes)
            for value in (provider.get("defaults") or {}).values():
                if isinstance(value, str) and value in seen_recipes and value != "off":
                    continue  # a recipe reference, already covered below
            recipe_value = (provider.get("defaults") or {}).get("reasoning.recipe")
            if isinstance(recipe_value, str) and recipe_value not in seen_recipes:
                raise CatalogFormatError(f"{provider.get('id')}.defaults.reasoning.recipe references unknown recipe {recipe_value!r}")

    def _validate_defaults(self, defaults: Mapping[str, object]) -> None:
        effort_order = defaults.get("effort_order")
        if effort_order is not None:
            if not isinstance(effort_order, list) or not effort_order:
                raise CatalogFormatError("catalog.defaults.effort_order must be a non-empty string list")
            if len(set(effort_order)) != len(effort_order):
                raise CatalogFormatError("catalog.defaults.effort_order must not repeat an effort")
            for effort in effort_order:
                if not isinstance(effort, str) or _EFFORT_RE.fullmatch(effort) is None:
                    raise CatalogFormatError(f"catalog.defaults.effort_order has invalid effort {effort!r}")
        budgets = defaults.get("thinking_budgets")
        if budgets is not None:
            if not isinstance(budgets, dict):
                raise CatalogFormatError("catalog.defaults.thinking_budgets must be an object")
            for effort, budget in budgets.items():
                if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                    raise CatalogFormatError(f"catalog.defaults.thinking_budgets.{effort} must be a positive integer")
                if effort_order and effort not in effort_order:
                    raise CatalogFormatError(f"catalog.defaults.thinking_budgets.{effort} is not in effort_order")
        policy = defaults.get("provider_policy")
        if policy is not None:
            self._validate_policy_values("catalog.defaults.provider_policy", cast(dict, policy))

    def _validate_policy_values(self, where: str, values: Mapping[str, object]) -> None:
        for path, value in values.items():
            if path not in POLICY_PATHS:
                raise CatalogFormatError(f"{where} has unknown policy path {path!r}")
            self._validate_policy_value(where, path, value)

    def _validate_policy_value(self, where: str, path: str, value: object) -> None:
        if path == "api":
            if value not in WIRES:
                raise CatalogFormatError(f"{where}.{path} must be one of {sorted(WIRES)}")
        elif path in ("reasoning.recipe", "reasoning.dialect"):
            if not isinstance(value, str) or not value:
                raise CatalogFormatError(f"{where}.{path} must be a non-empty string")
        elif path == "reasoning.levels":
            if not isinstance(value, list) or not value or len(value) > MAX_POLICY_LEVELS:
                raise CatalogFormatError(f"{where}.{path} must be a non-empty list of efforts")
            if len(set(value)) != len(value):
                raise CatalogFormatError(f"{where}.{path} must not repeat an effort")
            for effort in value:
                if not isinstance(effort, str) or _EFFORT_RE.fullmatch(effort) is None:
                    raise CatalogFormatError(f"{where}.{path} has invalid effort {effort!r}")
        elif path in ("reasoning.off", "reasoning.off_responses"):
            if not isinstance(value, str) or not value:
                raise CatalogFormatError(f"{where}.{path} must be a non-empty string")
        elif path in ("reasoning.mandatory", "cache.prompt_key", "json.response_format", "strict.tools", "strict.beta", "temperature.suppress"):
            if not isinstance(value, bool):
                raise CatalogFormatError(f"{where}.{path} must be a boolean")
        elif path == "history.reasoning":
            if value not in HISTORY_MODES:
                raise CatalogFormatError(f"{where}.{path} must be one of {sorted(HISTORY_MODES)}")
        elif path == "image.input":
            if value not in IMAGE_INPUTS:
                raise CatalogFormatError(f"{where}.{path} must be one of {sorted(IMAGE_INPUTS)}")
        elif path == "responses.reasoning_models" or path == "temperature.suppress_models":
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
                raise CatalogFormatError(f"{where}.{path} must be a list of model prefixes")
        elif path == "tools.builtin_by_wire" and not isinstance(value, dict):
            raise CatalogFormatError(f"{where}.{path} must be an object keyed by wire")

    def _validate_prose(self, entry: Mapping[str, object], where: str) -> None:
        why = entry.get("why")
        if why is not None and (not isinstance(why, str) or len(why) > MAX_PROSE_LENGTH or "\n" in why):
            raise CatalogFormatError(f"{where}.why must be a single line up to {MAX_PROSE_LENGTH} chars")
        for key in ("evidence", "notes"):
            items = entry.get(key)
            if items is None:
                continue
            if not isinstance(items, list):
                raise CatalogFormatError(f"{where}.{key} must be a list of strings")
            for item in items:
                if not isinstance(item, str) or not item or len(item) > MAX_PROSE_LENGTH:
                    raise CatalogFormatError(f"{where}.{key} entries must be non-empty strings up to {MAX_PROSE_LENGTH} chars")
        evidence = entry.get("evidence")
        if evidence is not None:
            if not isinstance(evidence, list):
                raise CatalogFormatError(f"{where}.evidence must be non-empty https:// URLs")
            for url in evidence:
                if not isinstance(url, str) or not url.startswith("https://") or len(url) > MAX_EVIDENCE_URL_LENGTH:
                    raise CatalogFormatError(f"{where}.evidence must be non-empty https:// URLs")

    def _require_unique_id(self, entry: Mapping[str, object], seen: set[str], where: str) -> str:
        rid = entry.get("id")
        if not isinstance(rid, str) or not rid:
            raise CatalogFormatError(f"{where} entries need a non-empty string id")
        if rid in seen:
            raise CatalogFormatError(f"duplicate id {rid!r}")
        seen.add(rid)
        return rid

    def _validate_rule(self, rule: Mapping[str, object], seen: set[str], where: str, *, allow_empty_selector: bool) -> None:
        if not isinstance(rule, dict) or "set" not in rule or not isinstance(rule["set"], dict) or not rule["set"]:
            raise CatalogFormatError(f"{where} rules need a non-empty set object")
        rid = self._require_unique_id(rule, seen, where)
        selector = rule.get("match")
        if selector is not None and not isinstance(selector, dict):
            raise CatalogFormatError(f"{where}.{rid}.match must be an object")
        if (selector is None or not selector) and not allow_empty_selector:
            raise CatalogFormatError(f"{where}.{rid} needs a non-empty selector")
        self._validate_selector(cast(dict, selector) if selector else None, f"{where}.{rid}.match")
        self._validate_policy_values(f"{where}.{rid}.set", cast(dict, rule["set"]))
        self._validate_prose(rule, f"{where}.{rid}")
        narrowing = any(path in rule["set"] for path in ("reasoning.levels", "reasoning.mandatory", "image.input", "reasoning.off", "reasoning.off_responses"))
        if narrowing and not (rule.get("why") and rule.get("evidence")):
            raise CatalogFormatError(f"{where}.{rid} narrows a capability or declares text-only and must carry why and evidence")

    def _validate_selector(self, selector: Mapping[str, object] | None, where: str) -> None:
        if selector is None:
            return
        known = {"prefixes", "pattern", "tokens_any", "tokens_all", "token_separator", "version"}
        for key in selector:
            if key not in known:
                raise CatalogFormatError(f"{where} has unknown selector field {key!r}")
        prefixes = selector.get("prefixes")
        if prefixes is not None and (not isinstance(prefixes, list) or not prefixes or not all(isinstance(p, str) and p for p in prefixes)):
            raise CatalogFormatError(f"{where}.prefixes must be a non-empty list of strings")
        pattern = selector.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_REGEX_LENGTH:
                raise CatalogFormatError(f"{where}.pattern must be a string up to {MAX_REGEX_LENGTH} chars")
            try:
                re.compile(pattern)
            except re.error as error:
                raise CatalogFormatError(f"{where}.pattern does not compile: {error}") from error
        version = selector.get("version")
        if version is not None:
            if not isinstance(version, dict) or not isinstance(version.get("pattern"), str) or not version["pattern"]:
                raise CatalogFormatError(f"{where}.version needs a pattern with named major/minor groups")
            try:
                compiled = re.compile(version["pattern"])
            except re.error as error:
                raise CatalogFormatError(f"{where}.version.pattern does not compile: {error}") from error
            if "major" not in compiled.groupindex:
                raise CatalogFormatError(f"{where}.version.pattern must define a named group major")
            for bound in ("min_inclusive", "max_inclusive", "max_exclusive"):
                value = version.get(bound)
                if value is not None and (
                    not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
                ):
                    raise CatalogFormatError(f"{where}.version.{bound} must be a [major, minor] integer pair")

    def _validate_recipe_reference(self, rule: Mapping[str, object], seen_recipes: set[str]) -> None:
        set_values = rule.get("set")
        recipe = set_values.get("reasoning.recipe") if isinstance(set_values, dict) else None
        if isinstance(recipe, str) and recipe not in seen_recipes:
            raise CatalogFormatError(f"{rule.get('id')}.set.reasoning.recipe references unknown recipe {recipe!r}")

    def _validate_builtin_entries(self, where: str, entries: object) -> None:
        if not isinstance(entries, list) or not entries:
            raise CatalogFormatError(f"{where} must be a non-empty list of tool entries")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("type"), str) or not entry["type"]:
                raise CatalogFormatError(f"{where} entries must each set a non-empty type")

    def _validate_recipe(self, rid: str, recipe: Mapping[str, object]) -> None:
        steps = recipe.get("steps")
        if not isinstance(steps, list) or len(steps) > MAX_RECIPE_STEPS:
            raise CatalogFormatError(f"request_recipes.{rid}.steps must be a list up to {MAX_RECIPE_STEPS}")
        pins = recipe.get("pins_temperature")
        if pins is not None and not isinstance(pins, bool):
            raise CatalogFormatError(f"request_recipes.{rid}.pins_temperature must be a boolean")
        for step_index, step in enumerate(steps):
            self._validate_recipe_step(rid, step_index, cast(dict, step))

    def _validate_recipe_step(self, rid: str, index: int, step: Mapping[str, object]) -> None:
        when = step.get("when")
        if when is not None:
            if not isinstance(when, dict) or not when:
                raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}].when must be a non-empty object")
            for key, condition in when.items():
                if key not in RECIPE_CONTEXT_KEYS:
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}].when has unknown input {key!r}")
                if isinstance(condition, dict):
                    for op in condition:
                        if op not in ("in", "present", "eq"):
                            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}].when.{key} has unknown operator {op!r}")
                        if op == "present" and not isinstance(condition[op], bool):
                            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}].when.{key}.present must be a boolean")
                        if op == "in" and not isinstance(condition[op], list):
                            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}].when.{key}.in must be a list")
        set_actions = step.get("set")
        if not isinstance(set_actions, list):
            set_actions = []
        remove = step.get("remove")
        if not isinstance(remove, list):
            remove = []
        if not set_actions and not remove:
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] must set or remove something")
        if len(set_actions) + len(remove) > MAX_RECIPE_PATHS:
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] touches too many paths")
        for action in set_actions:
            self._validate_recipe_action(rid, index, cast(dict, action))
        for path in remove:
            self._validate_recipe_path(rid, index, path)

    def _validate_recipe_path(self, rid: str, index: int, path: object) -> None:
        if not isinstance(path, list) or not path or not all(isinstance(part, str) and part for part in path):
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] has an invalid remove path")
        if path[0] in PROTECTED_BODY_ROOTS:
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] must not touch protected body field {path[0]!r}")

    def _validate_recipe_action(self, rid: str, index: int, action: Mapping[str, object]) -> None:
        path = action.get("path")
        self._validate_recipe_path(rid, index, path)
        if "value" not in action:
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] set actions need a value")
        self._validate_recipe_value(rid, index, action["value"])

    def _validate_recipe_value(self, rid: str, index: int, value: object) -> None:
        if isinstance(value, dict):
            if set(value) == {"source"} and isinstance(value["source"], str):
                if value["source"] not in RECIPE_CONTEXT_KEYS:
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] sources unknown context value {value['source']!r}")
                return
            if "case" in value:
                cases = value["case"]
                if not isinstance(cases, list) or not cases:
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] case must be a non-empty list")
                for case in cases:
                    if not isinstance(case, dict) or "when" not in case:
                        raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] case entries need when")
                if "else" not in value:
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] case needs an else")
                for case in cases:
                    self._validate_recipe_step_condition(rid, index, cast(dict, case["when"]))
                self._validate_recipe_value(rid, index, value["else"])
                for case in cases:
                    self._validate_recipe_value(rid, index, case.get("then"))
                return
            if set(value) == {"lookup"}:
                table = value["lookup"]
                if not isinstance(table, dict) or not isinstance(table.get("table"), str):
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] lookup needs a table name")
                self._validate_recipe_value(rid, index, table.get("key"))
                return
            if set(value) == {"bounded_budget"}:
                budget = value["bounded_budget"]
                if not isinstance(budget, dict) or not isinstance(budget.get("table"), str):
                    raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] bounded_budget needs a table name")
                for key in ("minimum", "headroom"):
                    if key in budget and (not isinstance(budget[key], int) or isinstance(budget[key], bool)):
                        raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] bounded_budget.{key} must be an integer")
                return
            # a literal object: validate any nested special forms recursively
            for nested in value.values():
                self._validate_recipe_value(rid, index, nested)
        elif isinstance(value, list):
            for item in value:
                self._validate_recipe_value(rid, index, item)

    def _validate_recipe_step_condition(self, rid: str, index: int, when: object) -> None:
        if not isinstance(when, dict) or not when:
            raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] when must be a non-empty object")
        for key, condition in when.items():
            if key not in RECIPE_CONTEXT_KEYS:
                raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] has unknown when input {key!r}")
            if isinstance(condition, dict) and set(condition) == {"in"} and not isinstance(condition["in"], list):
                raise CatalogFormatError(f"request_recipes.{rid}.steps[{index}] when.{key}.in must be a list")

    # ------------------------------------------------------------------
    # compilation
    # ------------------------------------------------------------------

    def _compile(self, raw: RawCatalog, source: CatalogSource) -> CatalogSnapshot:
        defaults_raw = raw["defaults"]
        effort_order = tuple(str(e) for e in defaults_raw.get("effort_order") or [])
        budgets_raw = defaults_raw.get("thinking_budgets") or {}
        thinking_budgets = {str(effort): int(budget) for effort, budget in budgets_raw.items()}
        provider_policy = defaults_raw.get("provider_policy") or {}
        defaults = CatalogDefaults(effort_order=effort_order, thinking_budgets=thinking_budgets, provider_policy=dict(provider_policy))

        forms = tuple(
            ModelIdForm(
                id=str(form["id"]),
                separator=str(form["separator"]),
                vendors=frozenset(str(v) for v in form["vendors"]),
            )
            for form in raw["model_id_forms"]
        )

        recipes: dict[str, RequestRecipe] = {}
        for recipe_id, recipe in raw["request_recipes"].items():
            recipes[recipe_id] = RequestRecipe(
                id=str(recipe.get("id") or recipe_id),
                steps=tuple(self._compile_step(step) for step in recipe["steps"]),
                pins_temperature=bool(recipe.get("pins_temperature", False)),
            )

        model_rules = tuple(self._compile_rule(rule) for rule in raw["model_rules"])
        providers = tuple(self._compile_provider(provider) for provider in raw["providers"])
        return CatalogSnapshot(
            schema_version=raw["schema_version"],
            version=raw["version"],
            updated_at=date.fromisoformat(raw["updated_at"]),
            defaults=defaults,
            model_id_forms=forms,
            request_recipes=recipes,
            model_rules=model_rules,
            providers=providers,
            source=source,
            content_hash=self.canonical_hash(raw),
        )

    def _compile_selector(self, match: RawSelector | None) -> Selector | None:
        if match is None:
            return None
        version_raw = match.get("version")
        version = None
        if isinstance(version_raw, dict):
            min_inclusive = version_raw.get("min_inclusive")
            max_inclusive = version_raw.get("max_inclusive")
            max_exclusive = version_raw.get("max_exclusive")
            version = VersionSelector(
                pattern=re.compile(str(version_raw["pattern"])),
                min_inclusive=cast(tuple[int, int], tuple(min_inclusive)) if min_inclusive is not None else None,
                max_inclusive=cast(tuple[int, int], tuple(max_inclusive)) if max_inclusive is not None else None,
                max_exclusive=cast(tuple[int, int], tuple(max_exclusive)) if max_exclusive is not None else None,
            )
        return Selector(
            prefixes=tuple(str(p) for p in match.get("prefixes") or ()),
            pattern=re.compile(str(match.get("pattern"))) if match.get("pattern") else None,
            tokens_any=tuple(str(t) for t in match.get("tokens_any") or ()),
            tokens_all=tuple(str(t) for t in match.get("tokens_all") or ()),
            token_separator=str(match.get("token_separator") or " "),
            version=version,
        )

    def _compile_rule(self, rule: RawPolicyRule) -> PolicyRule:
        return PolicyRule(
            id=str(rule["id"]),
            selector=self._compile_selector(rule.get("match")),
            set=dict(rule["set"]),
            why=str(rule.get("why") or ""),
            evidence=tuple(str(e) for e in rule.get("evidence") or ()),
            notes=tuple(str(n) for n in rule.get("notes") or ()),
        )

    def _compile_provider(self, provider: RawProvider) -> ProviderRule:
        tools = provider.get("builtin_tools_by_wire")
        builtin = None
        if tools is not None:
            builtin = {str(wire): tuple(dict(entry) for entry in entries) for wire, entries in tools.items()}
        return ProviderRule(
            id=str(provider["id"]),
            hosts=tuple(str(h) for h in provider["hosts"]),
            model_rule_modes=dict(provider.get("model_rule_modes") or {}),
            defaults=dict(provider.get("defaults") or {}),
            model_rules=tuple(self._compile_rule(rule) for rule in provider.get("model_rules") or ()),
            builtin_tools_by_wire=builtin,
            why=str(provider.get("why") or ""),
            evidence=tuple(str(e) for e in provider.get("evidence") or ()),
            notes=tuple(str(n) for n in provider.get("notes") or ()),
        )

    def _compile_step(self, step: RawRecipeStep) -> RecipeStep:
        when_raw = step.get("when")
        return RecipeStep(
            when=self._compile_condition(cast(Mapping[str, object], when_raw)) if when_raw else None,
            set=tuple(self._compile_action(action) for action in step.get("set") or ()),
            remove=tuple(tuple(str(part) for part in path) for path in step.get("remove") or ()),
        )

    def _compile_condition(self, when: Mapping[str, object]) -> RecipeCondition:
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

    def _compile_action(self, action: RawRecipeAction) -> RecipeAction:
        return RecipeAction(
            path=tuple(str(part) for part in action["path"]),
            value=self._compile_value(action["value"]),
        )

    def _compile_value(self, value: object) -> RecipeValue:
        if isinstance(value, dict):
            if set(value) == {"source"} and isinstance(value["source"], str):
                return RecipeValue("source", value["source"])
            if "case" in value:
                return RecipeValue("case", value)
            if set(value) == {"lookup"}:
                return RecipeValue("lookup", value["lookup"])
            if set(value) == {"bounded_budget"}:
                return RecipeValue("bounded_budget", value["bounded_budget"])
        return RecipeValue("literal", value)


def decode_bundled() -> CatalogSnapshot:
    """Decode the catalog shipped inside the package, failing fast on a corrupt install."""
    from importlib import resources

    payload = (resources.files("minacode.providers") / "catalog.json").read_bytes()
    return CatalogCodec().decode(payload, "bundled")
