"""Tool base class: schema generation, argument parsing, and shared file helpers."""

from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
from typing import Any, ClassVar

from minacode.base import Json, ToolArgs, ToolError
from minacode.session import Session, TurnDiff


class Tool:
    """One capability the model can invoke: its schema, its arguments, and a single call.

    A subclass declares itself through class attributes and implements `call`. Those attributes are
    not documentation — the runner reads them: `MUTATES` decides whether a call needs confirmation,
    `STORES_RESULT` whether its output is retained for recall, `PRODUCES_MODEL_OBSERVATION` whether it
    contributes more than text. `DESCRIPTION` and `EXAMPLE` are prompt surface and cost context on
    every request.

    The JSON Schema comes from `params_schema`, rewritten when the provider demands strict function
    calling, where every property is required and optionals become nullable. A schema containing a
    free-form object cannot be expressed that way and falls back to non-strict rather than being
    silently narrowed.

    An instance is per call, not per session: state read afterward, such as an edit's diff, describes
    that one invocation.
    """

    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    EXAMPLE: ClassVar[tuple[str, ...]] = ()
    RANGE_SCHEMA: ClassVar[Json] = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
    SKIP_DIRS: ClassVar[set[str]] = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}
    MUTATES: ClassVar[bool] = False
    PRODUCES_MODEL_OBSERVATION: ClassVar[bool] = False
    STORES_RESULT: ClassVar[bool] = True
    LOG_LEXER: ClassVar[str] = "tool-args"

    def __init__(self, session: Session, args: ToolArgs):
        self.session = session
        self.args = args

    def turn_diff(self) -> TurnDiff | None:
        """The file diff this tool produced on its last run, or None if it made no edit. Overridden
        by EditTool; the runner records it against the stored result for the /diff viewer."""
        return None

    def model_observation(self) -> Json | None:
        """A model-facing observation produced by the completed call, if any."""
        return None

    @classmethod
    def schema(cls, strict: bool = False) -> Json:
        description = "\n".join([cls.DESCRIPTION, *(("- " + item) for item in cls.EXAMPLE if item)])
        function: Json = {"name": cls.NAME, "description": description, "parameters": cls.params_schema()}
        if strict and cls._strictifiable(function["parameters"]):
            function["parameters"] = cls._strict_schema(function["parameters"])
            function["strict"] = True
        return {"type": "function", "function": function}

    @staticmethod
    def resolved_schemas(session: Session) -> list[Json]:
        """Return the tool schemas available for this session and provider."""

        from minacode.tools import TOOL_REGISTRY, MCPTool, SkillTool  # local import: the registry is built on top of every tool

        strict = session.config.provider.resolve().strict_tools_active
        # Optional tool families stay out of the model prefix until they have usable session state.
        has_skills = bool(session.skills and session.skills.skills)
        has_mcp = bool(session.mcp and (session.mcp.tools or session.mcp.resources))
        return [tool.schema(strict) for tool in TOOL_REGISTRY.values() if (tool is not SkillTool or has_skills) and (tool is not MCPTool or has_mcp)]

    @staticmethod
    def _strictifiable(schema: object) -> bool:
        """False if the schema contains a free-form object (an `object` with no `properties`),
        which strict function calling cannot represent — such tools fall back to non-strict."""
        if isinstance(schema, dict):
            if schema.get("type") == "object" and "properties" not in schema:
                return False
            return all(Tool._strictifiable(value) for value in schema.values())
        if isinstance(schema, list):
            return all(Tool._strictifiable(item) for item in schema)
        return True

    @staticmethod
    def _strict_schema(schema: Json) -> Json:
        """Rewrite a JSON Schema to satisfy strict function-calling (OpenAI / DeepSeek beta):
        every object property becomes required (genuine optionals turned nullable),
        additionalProperties is forced false, and unsupported keywords are dropped."""
        # Strict validators only allow scalar types in a `type` union; object/array nullability
        # must be expressed with anyOf instead (e.g. {"anyOf": [<array schema>, {"type": "null"}]}).
        scalars = ("string", "number", "integer", "boolean")

        def nullable(sub: Json) -> Json:
            kind = sub.get("type")
            if isinstance(kind, str) and kind in scalars:
                sub["type"] = [kind, "null"]
            elif isinstance(kind, list) and all(item in (*scalars, "null") for item in kind):
                if "null" not in kind:
                    sub["type"] = [*kind, "null"]
            else:
                return {"anyOf": [sub, {"type": "null"}]}
            # An enum must accept null too, otherwise strict validation rejects the "omitted" value.
            if isinstance(sub.get("enum"), list) and None not in sub["enum"]:
                sub["enum"] = [*sub["enum"], None]
            return sub

        # Json is intentionally shallow (dict[str, Any]); this recursive schema transform is one
        # of the places where preserving that dynamic value type is clearer than repeated casts.
        def transform(node: Any) -> Any:
            if isinstance(node, list):
                return [transform(item) for item in node]
            if not isinstance(node, dict):
                return node
            transformed = {key: transform(value) for key, value in node.items() if key not in ("minItems", "maxItems", "minLength", "maxLength")}
            if isinstance(transformed.get("properties"), dict):
                required = set(transformed.get("required") or [])
                for key, sub in transformed["properties"].items():
                    if key not in required and isinstance(sub, dict):
                        transformed["properties"][key] = nullable(sub)
                transformed["required"] = list(transformed["properties"].keys())
                transformed["additionalProperties"] = False
            return transformed

        return transform(copy.deepcopy(schema))

    @staticmethod
    def object_schema(properties: Json, required: list[str] | None = None) -> Json:
        schema: Json = {"type": "object", "properties": properties, "additionalProperties": False}
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({})

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload]

    def needs_confirmation(self) -> bool:
        return self.MUTATES

    @classmethod
    def log_lexer(cls, _: ToolArgs) -> str:
        return cls.LOG_LEXER

    def single_dict_arg(self, message: str) -> Json:
        if len(self.args) != 1 or not isinstance(self.args[0], dict):
            raise ToolError(message)
        return self.args[0]

    def preview(self) -> str:
        return f"{self.NAME}({', '.join(self.short_args())})"

    def short_args(self) -> list[str]:
        return [self.compact(arg) for arg in self.args]

    def call(self) -> str:
        raise NotImplementedError

    def strings(self, *, min_count: int = 0, max_count: int | None = None) -> list[str]:
        if len(self.args) < min_count or (max_count is not None and len(self.args) > max_count):
            limit = f"{min_count}" if max_count == min_count else f"{min_count}-{max_count or 'many'}"
            raise ToolError(f"{self.NAME} requires {limit} string args")
        if not all(isinstance(arg, str) for arg in self.args):
            raise ToolError(f"{self.NAME} args must be strings")
        return [str(arg) for arg in self.args]

    @staticmethod
    def line_range(value: object, label: str = "range") -> tuple[int, int]:
        if not isinstance(value, list) or len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ToolError(f"{label} must be [start,end] integers")
        start, end = value
        if start < 0 or end < 0:
            raise ToolError(f"{label} values must be >= 0")
        return int(start), int(end)

    @staticmethod
    def compact(value: Any, limit: int = 120) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def compile_regex(pattern: str, *, case_sensitive: bool = False, multiline: bool = False) -> re.Pattern[str]:
        try:
            flags = (0 if case_sensitive else re.IGNORECASE) | (re.MULTILINE if multiline else 0)
            return re.compile(pattern, flags)
        except re.error as error:
            raise ToolError(f"invalid regex: {error}") from error

    @staticmethod
    def process_result(tag: str, code: int, stdout: str, stderr: str) -> str:
        lines = [f"<{tag}>", f"* exit_code: {code}"]
        for name, text in (("stdout", stdout), ("stderr", stderr)):
            if text:
                lines.extend([f"<{name}>", text.rstrip(), f"</{name}>"])
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    @staticmethod
    def file_stat(path: str) -> str:
        stat = os.stat(path)
        return f'<file_stat mtime_ns="{stat.st_mtime_ns}" size="{stat.st_size}"/>'

    def gitignore_patterns(self, root: str) -> list[str]:
        patterns = []
        cache = self.session._gitignore_cache
        paths = [os.path.join(self.session.cwd, ".gitignore")]
        if os.path.isdir(root):
            paths.append(os.path.join(root, ".gitignore"))
        for path in dict.fromkeys(paths):
            try:
                mtime = os.stat(path).st_mtime_ns
                cached = cache.get(path)
                if cached is not None and cached[0] == mtime:
                    patterns.extend(cached[1])
                    continue
                with open(path, encoding="utf-8") as file:
                    pats = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")]
                cache[path] = (mtime, pats)
                patterns.extend(pats)
            except OSError:
                cache.pop(path, None)
        return patterns

    def ignored(self, path: str, patterns: list[str]) -> bool:
        rel = self.session.relpath(path).replace(os.sep, "/")
        name = os.path.basename(path)
        parts = [part for part in rel.split("/") if part and part != "."]
        for raw in patterns:
            directory = raw.endswith("/")
            pattern = raw.rstrip("/")
            if not pattern:
                continue
            if "/" in pattern:
                matched = fnmatch.fnmatch(rel, pattern) or (directory and (rel == pattern or rel.startswith(pattern + "/")))
            else:
                matched = any(fnmatch.fnmatch(part, pattern) for part in parts) or fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern)
            if matched:
                return True
        return False

    def default_ignored(self, path: str, patterns: list[str]) -> bool:
        rel = self.session.relpath(path).replace(os.sep, "/")
        hidden = rel not in {"", "."} and any(part.startswith(".") for part in rel.split("/") if part and part != ".")
        return hidden or self.ignored(path, patterns)
