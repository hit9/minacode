"""minacode tools: the built-in tool set exposed to the model."""

from __future__ import annotations

import codecs
import contextlib
import copy
import difflib
import fnmatch
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import code_symbol_index as csi

from minacode.base import Json, Text, ToolArgs, ToolError
from minacode.session import AgentState, BackgroundJob, HistorySegment, PlanItem, Session, TurnDiff


class Tool:
    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    SIGNATURE: ClassVar[str] = ""
    EXAMPLE: ClassVar[tuple[str, ...]] = ()
    RANGE_SCHEMA: ClassVar[Json] = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
    SKIP_DIRS: ClassVar[set[str]] = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}
    MUTATES: ClassVar[bool] = False
    STORES_RESULT: ClassVar[bool] = True
    LOG_LEXER: ClassVar[str] = "tool-args"

    def __init__(self, session: Session, args: ToolArgs):
        self.session = session
        self.args = args

    def turn_diff(self) -> TurnDiff | None:
        """The file diff this tool produced on its last run, or None if it made no edit. Overridden
        by EditTool; the runner records it against the stored result for the /diff viewer."""
        return None

    @classmethod
    def schema(cls, strict: bool = False) -> Json:
        description = "\n".join([cls.DESCRIPTION, "Signature: " + cls.SIGNATURE, *(("- " + item) for item in cls.EXAMPLE if item)])
        function: Json = {"name": cls.NAME, "description": description, "parameters": cls.params_schema()}
        if strict and cls._strictifiable(function["parameters"]):
            function["parameters"] = cls._strict_schema(function["parameters"])
            function["strict"] = True
        return {"type": "function", "function": function}

    @staticmethod
    def resolved_schemas(session: Session) -> list[Json]:
        """Return the tool schemas available for this session and provider."""

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


class ReadTool(Tool):
    NAME = "Read"
    DESCRIPTION = "Read UTF-8 file line ranges; returns file stat, total lines, and anchor=line:hash(line_content) text. Large outputs are bounded in conversation; use Recall(tr.N) for full stored output."
    SIGNATURE = "Read(path,ranges=[[start,end],...]) or Read(files=[{path,ranges}]); lines are 0-based, end-exclusive"
    # fmt: off
    EXAMPLE = (
        'Read ranges. Example: {"path":"src/app.py","ranges":[[0,80],[120,180]]}',
        'Read several files. Example: {"files":[{"path":"src/app.py","ranges":[[0,80]]},{"path":"README.md","ranges":[[0,40]]}]}',
    )
    # fmt: on

    @classmethod
    def arg_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "path": {"type": "string", "description": "File path to read"},
            "ranges": {"type": "array", "minItems": 1, "items": cls.RANGE_SCHEMA, "description": "Line ranges [[start,end],...], 0-based and end-exclusive; omit to read the whole file"},
        }, ["path"])
        # fmt: on

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "path": {"type": "string", "description": "File path to read (single-file form)"},
            "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1, "description": "Line ranges [[start,end],...], 0-based and end-exclusive; omit to read the whole file"},
            "files": {"type": "array", "items": cls.arg_schema(), "minItems": 1, "description": "Batch form: list of {path, ranges} to read several files in one call"},
        })
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return (
            payload["files"]
            if isinstance(payload.get("files"), list)
            else [{"path": payload.get("path", ""), "ranges": cls.ranges_arg(payload.get("ranges") or [[0, 0]])}]
        )

    @classmethod
    def ranges_arg(cls, value: object) -> object:
        return [value] if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in value) else value

    @staticmethod
    def line_hash(line: str) -> str:
        # Hash the visible content only. The trailing newline is stripped so the anchor matches the
        # line the model sees (anchor_line displays the stripped line), stays stable when only the
        # final newline changes, and is consistent with indexed_line_hash.
        return Text.base36(int(hashlib.sha1(line.rstrip("\n").encode("utf-8")).hexdigest()[:6], 16)).rjust(5, "0")

    @staticmethod
    def split_lines(text: str) -> list[str]:
        # Canonical line model shared by Read and Edit: split on "\n" only, keeping the newline
        # (like file.readlines()). str.splitlines(True) also breaks on \r, \v, \f, \x1c-\x1e, \x85,
        # \u2028, \u2029, which would number lines differently than Read and desync anchors.
        parts = text.split("\n")
        lines = [part + "\n" for part in parts[:-1]]
        if parts[-1]:
            lines.append(parts[-1])
        return lines

    @classmethod
    def anchor(cls, index: int, line: str) -> str:
        return f"{index}:{cls.line_hash(line)}"

    @classmethod
    def anchor_line(cls, index: int, line: str) -> str:
        return f"anchor={cls.anchor(index, line)} | {line.rstrip(chr(10))}"

    @staticmethod
    def indexed_line_hash(line: str) -> str:
        return hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def parse_anchor(value: str) -> tuple[int, str] | None:
        text = value.split("|", 1)[0].strip()
        if text.startswith("anchor="):
            text = text.removeprefix("anchor=").strip()
        match = re.fullmatch(r"(\d+):([0-9a-z]{5}|[0-9a-f]{8})", text)
        return (int(match.group(1)), match.group(2).lower()) if match else None

    @staticmethod
    def require_anchor(anchor: str) -> tuple[int, str]:
        """Parse an anchor or raise the standard ToolError guiding the model to a real one."""
        parsed = ReadTool.parse_anchor(anchor)
        if parsed is None:
            raise ToolError('invalid anchor; use the "anchor=line:hash" value from Read, Search, or InspectCode')
        return parsed

    @classmethod
    def anchor_matches(cls, line: str, expected: str) -> bool:
        return expected == cls.line_hash(line) or expected == cls.indexed_line_hash(line)

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(path) for path, _ in self.targets())

    def call(self) -> str:
        return "\n\n".join(self.read_one(path, ranges) for path, ranges in self.targets())

    def short_args(self) -> list[str]:
        return [self.session.relpath(path) + " " + ",".join(f"{start}:{end}" for start, end in ranges) for path, ranges in self.targets()]

    def targets(self) -> list[tuple[str, list[tuple[int, int]]]]:
        if not self.args:
            raise ToolError("Read requires at least one {path,ranges} object")
        targets = []
        for index, spec in enumerate(self.args):
            if not isinstance(spec, dict):
                raise ToolError("Read args must be {path,ranges} objects")
            if unexpected := sorted(set(spec) - {"path", "ranges"}):
                raise ToolError("Read unexpected field: " + ", ".join(unexpected))
            path = str(spec.get("path") or "").strip()
            raw_ranges = self.ranges_arg(spec.get("ranges") if "ranges" in spec else [[0, 0]])
            if not path:
                raise ToolError("Read requires non-empty path")
            if not isinstance(raw_ranges, list) or not raw_ranges:
                raise ToolError("Read requires non-empty ranges")
            ranges = [self.line_range(value, f"args[{index}].ranges") for value in raw_ranges]
            targets.append((self.session.resolve_path(path), ranges))
        return targets

    def read_one(self, path: str, ranges: list[tuple[int, int]]) -> str:
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()
        out = [f"<Read path={json.dumps(self.session.relpath(path))}>", self.file_stat(path), f"<total_lines>{len(lines)}</total_lines>"]
        for start, requested_end in ranges:
            start = min(start, len(lines))
            end = max(start, len(lines) if requested_end == 0 else min(len(lines), requested_end))
            out.append(f"<range>{start}:{end}</range>")
            out.append("<content hashline-numbered>")
            out.extend(self.anchor_line(i, lines[i]) for i in range(start, end))
            out.append("</content>")
        out.append("</Read>")
        return "\n".join(out)


class SearchTool(Tool):
    NAME = "Search"
    DESCRIPTION = "Search UTF-8 text files with case-insensitive regex; skips binary/hidden/gitignored files and returns path anchor=line:hash matches."
    SIGNATURE = "Search(pattern,path?,glob?,context?) or Search(queries=[...]); pattern is regex, A|B|C is ok"
    # fmt: off
    EXAMPLE = (
        'Search source with context. Example: {"pattern":"class .*Tool","path":"src","glob":"*.py","context":2}',
        'Search multiple queries. Example: {"queries":[{"pattern":"TODO","glob":"*.py"},{"pattern":"FIXME","path":"tests","glob":"*.py"}]}',
        'Batch regex terms. Example: {"queries":[{"pattern":"done in|elapsed|duration","glob":"*.py","context":2}]}',
    )
    # fmt: on
    MAX_FILE_BYTES = 2_000_000
    MAX_CONTEXT = 30

    @classmethod
    def arg_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "pattern": {"type": "string", "description": "Case-insensitive regex; alternation A|B|C is allowed"},
            "path": {"type": "string", "description": "File or directory to search under; defaults to repo root"},
            "glob": {"type": "string", "description": "Optional glob limiting which files are searched, e.g. *.py"},
            "context": {"type": "integer", "minimum": 0, "maximum": cls.MAX_CONTEXT, "description": f"Context lines around each match, 0..{cls.MAX_CONTEXT}"},
        }, ["pattern"])
        # fmt: on

    @classmethod
    def params_schema(cls) -> Json:
        props = dict(cls.arg_schema()["properties"])
        props["queries"] = {"type": "array", "items": cls.arg_schema(), "minItems": 1, "description": "Batch form: list of search queries to run in one call"}
        return cls.object_schema(props)

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return payload.get("queries") or [payload]

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(request["path"]) for request in self.requests())

    def call(self) -> str:
        return "\n\n".join(self.search(request) for request in self.requests())

    def short_args(self) -> list[str]:
        rows = []
        for request in self.requests():
            rel = self.session.relpath(str(request["path"]))
            rows.append(
                " ".join(
                    [
                        json.dumps(request["pattern"], ensure_ascii=False),
                        *(["path=" + rel] if rel != "." else []),
                        *(["glob=" + str(request["glob"])] if request["glob"] else []),
                        *(["C=" + str(request["context"])] if request["context"] else []),
                    ]
                )
            )
        return ["; ".join(rows)]

    def requests(self) -> list[Json]:
        if not self.args:
            raise ToolError("Search requires at least one query object")
        requests = []
        for item in self.args:
            if not isinstance(item, dict):
                raise ToolError("Search args must be query objects")
            if unexpected := sorted(set(item) - {"pattern", "path", "glob", "context"}):
                raise ToolError("Search unexpected field: " + ", ".join(unexpected))
            pattern = str(item.get("pattern") or "").replace("\\n", "\n")
            if not pattern:
                raise ToolError("Search requires pattern")
            context = item.get("context", 0)
            if isinstance(context, bool) or not isinstance(context, int) or context < 0 or context > self.MAX_CONTEXT:
                raise ToolError(f"Search context must be 0..{self.MAX_CONTEXT}")
            requests.append(
                {"pattern": pattern, "path": self.session.resolve_path(str(item.get("path") or ".")), "glob": str(item.get("glob") or ""), "context": context}
            )
        return requests

    def search(self, request: Json) -> str:
        patterns = self.gitignore_patterns(str(request["path"]))
        rows = [] if self.default_ignored(str(request["path"]), patterns) else None
        rows = rows if rows is not None or "\n" in str(request["pattern"]) else self.rg_matches(request)
        rows = rows if rows is not None else self.python_matches(request)
        header = f"<SearchToolResult pattern={json.dumps(request['pattern'])} matches={len(rows)}>"
        return "\n".join([header, *rows, "</SearchToolResult>"])

    def rg_matches(self, request: Json) -> list[str] | None:
        rg = shutil.which("rg")
        if not rg:
            return None
        cmd = [rg, "--json", "--line-number", "--with-filename", "--color=never", "--ignore-case", "--max-filesize", "2M"]
        if request["context"]:
            cmd.extend(["-C", str(request["context"])])
        if request["glob"]:
            cmd.extend(["--glob", str(request["glob"])])
        cmd.extend([str(request["pattern"]), str(request["path"])])
        proc = subprocess.run(cmd, cwd=self.session.cwd, text=True, capture_output=True, timeout=self.session.settings.shell_timeout, check=False)
        if proc.returncode == 2:
            proc = subprocess.run(
                [*cmd[:1], "--pcre2", *cmd[1:]], cwd=self.session.cwd, text=True, capture_output=True, timeout=self.session.settings.shell_timeout, check=False
            )
        if proc.returncode not in (0, 1):
            return None
        rows = []
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in {"match", "context"}:
                continue
            data = event.get("data") or {}
            path = data.get("path", {}).get("text")
            number = data.get("line_number")
            text = data.get("lines", {}).get("text", "")
            if not path or not isinstance(number, int):
                continue
            prefix = ">" if event["type"] == "match" else " "
            rows.append(self.match_line(prefix, path, number - 1, text))
        return rows

    def files(self, root: str, glob_pattern: str) -> list[str]:
        gitignore = self.gitignore_patterns(root)
        if self.default_ignored(root, gitignore):
            return []
        if os.path.isfile(root):
            return [root]
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in self.SKIP_DIRS and not name.startswith(".") and not self.ignored(os.path.join(dirpath, name), gitignore)
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = os.path.join(dirpath, filename)
                rel = self.session.relpath(path)
                if self.ignored(path, gitignore):
                    continue
                if glob_pattern and not (fnmatch.fnmatch(filename, glob_pattern) or fnmatch.fnmatch(rel, glob_pattern)):
                    continue
                found.append(path)
        return found

    def python_matches(self, request: Json) -> list[str]:
        regex = self.compile_regex(str(request["pattern"]), multiline=True)
        rows = []
        for path in self.files(str(request["path"]), str(request["glob"])):
            for row, _ in self.file_matches(path, regex, int(request["context"])):
                rows.append(row)
        return rows

    def file_matches(self, path: str, regex: re.Pattern[str], context: int) -> list[tuple[str, bool]]:
        try:
            if os.path.getsize(path) > self.MAX_FILE_BYTES:
                return []
            with open(path, encoding="utf-8") as file:
                lines = file.readlines()
        except (OSError, UnicodeDecodeError):
            return []
        rows: list[tuple[str, bool]] = []
        content = "".join(lines)
        starts = (
            [content.count("\n", 0, match.start()) for match in regex.finditer(content)]
            if "\n" in regex.pattern
            else [index for index, line in enumerate(lines) if regex.search(line)]
        )
        for index in starts:
            seen = set()
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            for line_index in range(start, end):
                if line_index in seen:
                    continue
                seen.add(line_index)
                prefix = ">" if line_index == index else " "
                rows.append((self.match_line(prefix, path, line_index, lines[line_index]), line_index == index))
        return rows

    def match_line(self, prefix: str, path: str, line_index: int, line: str) -> str:
        return f"{prefix} {self.session.relpath(path)} {ReadTool.anchor_line(line_index, line)}"


class CodeIndex:
    AUTO_UPDATE_LIMIT: ClassVar[int] = 20
    # fmt: off
    SYMBOLS: ClassVar[dict[str, str]] = {
        "ready": "✓", "synced": "✓", "stale": "*", "syncing": "~",
        "updating": "~", "missing": "?", "unavailable": "!", "error": "!",
    }
    # fmt: on

    def __init__(self, session: Session):
        self.session = session

    def available(self) -> bool:
        status, message = self.status()
        self.session.state.code_index_error = message if status == "error" else ""
        return status in {"ready", "stale"}

    def set_status(self, status: str, message: str = "") -> None:
        self.session.state.code_index_status = "synced" if status == "ready" else status
        self.session.state.code_index_error = message if status == "error" else ""

    @classmethod
    def label(cls, status: str) -> str:
        return cls.SYMBOLS.get(status, status)

    @classmethod
    def status_line(cls, status: str, message: str = "") -> str:
        status = "synced" if status == "ready" else status
        return f"index{cls.label(status)} {status}" + ((": " + message) if message else "")

    def notice(self, text: str = "", *, refreshing: bool = False) -> None:
        self.session.state.code_index_notice = text
        self.session.state.code_index_refreshing = refreshing
        if text:
            self.session.state.code_index_status = "syncing" if text in {"syncing", "updating"} else text

    def fail(self, error: object) -> str:
        self.session.state.code_index_error = str(error).strip()
        self.notice("error")
        return self.session.state.code_index_error

    def finish(self, status: str = "synced") -> None:
        self.notice("")
        self.session.state.code_index_error, self.session.state.code_index_status = "", status

    def status(self, *, check: bool = False, max_pending_files: int = 20) -> tuple[str, str]:
        try:
            data = csi.status(self.session.cwd, check=check, max_pending_files=max_pending_files)
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            self.set_status("error", str(error))
            return "error", str(error)
        status = str(getattr(data, "status", "") or "error")
        message = str(getattr(data, "message", None) or getattr(data, "reason", None) or "")
        pending = getattr(data, "pending_changes", None)
        files = getattr(data, "pending_files", ()) or ()
        if pending and pending != "unknown":
            sample = ", ".join(str(path) for path in (files or [])[:3])
            message = (message + "; " if message else "") + "pending " + str(pending) + ((" (" + sample + ")") if sample else "")
        preserves_stale = status == "ready" and pending == "unknown" and self.session.state.code_index_status == "stale"
        if not self.session.state.code_index_refreshing and not preserves_stale:
            self.set_status(status, message)
        return status, message

    def sync(self, *, force: bool = False) -> str:
        if self.session.state.code_index_refreshing:
            return "code_index: syncing"
        if force:
            csi.clean(self.session.cwd)
        self.notice("syncing", refreshing=True)
        try:
            csi.index(self.session.cwd)
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            return "code_index: error\n" + self.fail(error)
        self.finish()
        status, message = self.status(check=True)
        index_path = os.path.join(self.session.cwd, ".code-symbol-index", "index.sqlite")
        lines = ["code_index: " + ("rebuilt" if force else "synced"), "status: " + status, "path: " + index_path]
        if message:
            lines.append("note: " + message)
        return "\n".join(lines)

    def update(self, paths: list[str]) -> str:
        paths = self.update_paths(paths)
        if not paths or self.session.state.code_index_refreshing or not self.available():
            return ""
        self.notice("updating", refreshing=True)
        try:
            csi.update(paths, root=self.session.cwd)
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            return self.fail(error)
        self.finish()
        return "updated " + str(len(paths)) + " file(s)"

    def update_pending(self) -> str:
        if self.session.state.code_index_refreshing:
            return ""
        try:
            data = csi.status(self.session.cwd, check=True, max_pending_files=self.AUTO_UPDATE_LIMIT + 1)
        except Exception:  # noqa: BLE001 - background index freshness checks are best-effort.
            return ""
        self.set_status(str(getattr(data, "status", "") or "error"), str(getattr(data, "message", None) or getattr(data, "reason", None) or ""))
        if getattr(data, "status", "") != "stale":
            return ""
        pending = getattr(data, "pending_changes", None)
        files = [str(path) for path in getattr(data, "pending_files", ()) or () if path]
        if not files or len(files) > self.AUTO_UPDATE_LIMIT or (isinstance(pending, int) and pending > self.AUTO_UPDATE_LIMIT):
            return ""
        return self.update([self.session.resolve_path(path) for path in files])

    def update_pending_async(self) -> None:
        """Run the working-tree check (and any auto-update) off the UI critical path.

        ``update_pending`` does a ``check=True`` scan that walks/hashes the tree — slow on
        large repos. Running it inline blocks answer emission and /status, so spawn it in a
        daemon thread. Guarded so only one scan/update runs at a time.
        """
        if self.session.state.code_index_checking or self.session.state.code_index_refreshing:
            return
        self.session.state.code_index_checking = True

        def run() -> None:
            try:
                self.update_pending()
            finally:
                self.session.state.code_index_checking = False

        threading.Thread(target=run, daemon=True).start()

    def refresh_existing_async(self) -> bool:
        if self.session.state.code_index_refreshing:
            return False
        status = self.status()[0]
        if status not in {"ready", "stale"}:
            return False
        self.notice("syncing", refreshing=True)
        try:
            worker = csi.refresh_async(self.session.cwd)
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            self.fail(error)
            return False

        def finish() -> None:
            worker.join()
            try:
                self.session.state.code_index_refreshing = False
                self.session.state.code_index_notice = ""
                self.status(check=True)
            except Exception as error:  # noqa: BLE001 - isolate background code-index failures.
                self.fail(error)

        threading.Thread(target=finish, daemon=True).start()
        return True

    def update_paths(self, paths: list[str]) -> list[str]:
        paths = [self.session.resolve_path(path) for path in paths]
        return list(dict.fromkeys(path for path in paths if self.session.in_cwd(path) and os.path.isfile(path)))


class InspectCodeTool(Tool):
    NAME = "InspectCode"
    MAX_LIMIT: ClassVar[int] = 80
    MAX_OUTLINE_LIMIT: ClassVar[int] = 1000
    MAX_DEPTH: ClassVar[int] = 5
    MODES: ClassVar[tuple[str, ...]] = ("find", "inspect", "outline", "refs", "impls", "callers", "callees")
    SYMBOL_MODES: ClassVar[frozenset[str]] = frozenset({"find", "inspect", "refs", "impls", "callers", "callees"})
    RESOLVE_MODES: ClassVar[frozenset[str]] = frozenset({"inspect", "refs", "impls", "callers", "callees"})
    CHAIN_MODES: ClassVar[frozenset[str]] = frozenset({"callers", "callees"})
    OPTION_KEYS: ClassVar[tuple[str, ...]] = ("limit", "kind", "path", "symbol", "exact_only", "depth", "offset", "all_kinds", "ref_kind", "loose")
    DESCRIPTION = "Use the code index: find returns symbols; inspect returns anchors/members/references; outline returns a file symbol tree; refs lists classified references; impls lists implementors; callers/callees walk the call chain."
    SIGNATURE = "InspectCode(mode,target,kind?,path?,symbol?,limit?,exact_only?,depth?,offset?,all_kinds?,ref_kind?,loose?)"
    # fmt: off
    EXAMPLE = (
        'Find symbols; kind can be class|function|method|variable|constant|enum|struct|interface|module|type|trait|field|property|impl|namespace|dict_key, comma-ok. Example: {"mode":"find","target":"Tool","kind":"class,function","limit":20}',
        'Inspect one symbol; path narrows candidates. Example: {"mode":"inspect","target":"Tool","path":"src/app.py"}',
        'Outline one file; symbol narrows subtree. Example: {"mode":"outline","target":"src/app.py","symbol":"App","limit":300}',
        'List references; default hides import/attribute noise, ref_kind filters to call|read|write|inherit|type|import|attribute|usage (comma-ok), all_kinds shows everything, offset pages. Example: {"mode":"refs","target":"Tool","ref_kind":"call,write","offset":0}',
        'List implementors of an interface/base. Example: {"mode":"impls","target":"Tool","kind":"class"}',
        'Walk transitive callers/callees up to depth; callees loose includes ambiguous cross-module matches. Example: {"mode":"callers","target":"handle_job","depth":3}',
    )
    # fmt: on

    @classmethod
    def params_schema(cls) -> Json:
        props = {
            "mode": {"type": "string", "enum": list(cls.MODES), "description": "Query type: find|inspect|outline|refs|impls|callers|callees"},
            "target": {"type": "string", "description": "Symbol name (find/inspect/refs/impls/callers/callees) or file path (outline)"},
            "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_OUTLINE_LIMIT, "description": "Max results"},
            "kind": {"type": "string", "description": "Restrict to a symbol kind, e.g. function, class, method"},
            "path": {"type": "string", "description": "Restrict the search to this file or directory"},
            "symbol": {"type": "string", "description": "Disambiguate target when multiple symbols share a name"},
            "exact_only": {"type": "boolean", "description": "Match the target name exactly instead of fuzzily"},
            "depth": {"type": "integer", "minimum": 1, "maximum": cls.MAX_DEPTH, "description": "Call-chain depth for callers/callees"},
            "offset": {"type": "integer", "minimum": 0, "description": "Pagination offset for refs/impls"},
            "all_kinds": {"type": "boolean", "description": "Include all reference kinds, not just behavioral ones (refs)"},
            "ref_kind": {"type": "string", "description": "Restrict refs to a specific reference kind"},
            "loose": {"type": "boolean", "description": "Loosen call-chain matching (callees)"},
        }
        return cls.object_schema(props, ["mode", "target"])

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        options = {key: payload[key] for key in cls.OPTION_KEYS if key in payload}
        return [str(payload.get("mode") or ""), str(payload.get("target") or ""), *([options] if options else [])]

    def call(self) -> str:
        if len(self.args) not in (2, 3):
            raise ToolError("InspectCode requires mode, target[, options]")
        if not isinstance(self.args[0], str) or not isinstance(self.args[1], str):
            raise ToolError("InspectCode mode and target must be strings")
        mode, target = self.args[0].lower(), self.args[1].strip()
        if len(self.args) == 3 and not isinstance(self.args[2], dict):
            raise ToolError("InspectCode options must be an object")
        options = self.args[2] if len(self.args) == 3 else {}
        if unexpected := sorted(set(options) - set(self.OPTION_KEYS)):
            raise ToolError("InspectCode unexpected option: " + ", ".join(unexpected))
        if mode not in self.MODES:
            raise ToolError("InspectCode mode must be one of: " + ", ".join(self.MODES))
        if not target:
            raise ToolError("InspectCode target is required")
        if mode in self.SYMBOL_MODES and re.search(r"\s", target):
            # Models often repeat the kind inside the target, e.g. target "class Config" with
            # kind "class". When the first word duplicates a declared kind, drop it — that is the one
            # case we can strip deterministically (no guessing at per-language keywords).
            kinds = {token.strip().lower() for token in str(options.get("kind") or "").split(",") if token.strip()}
            first, _, rest = target.partition(" ")
            if kinds and first.lower() in kinds and rest.strip():
                target = rest.strip()
            if re.search(r"\s", target):
                raise ToolError("InspectCode symbol target must not contain whitespace")
        if mode in self.RESOLVE_MODES and (target.endswith(".py") or os.path.exists(self.session.resolve_path(target))):
            raise ToolError(f"InspectCode {mode} target must be a symbol, not a file")
        if mode == "outline" and not os.path.isfile(self.session.resolve_path(target)):
            raise ToolError("InspectCode outline target must be an existing file")
        limit = options.get("limit")
        max_limit = self.MAX_OUTLINE_LIMIT if mode == "outline" else self.MAX_LIMIT
        self._check_int_option(limit, 1, max_limit, f"InspectCode {mode} limit must be 1..{max_limit}")
        self._check_int_option(options.get("depth"), 1, self.MAX_DEPTH, f"InspectCode depth must be 1..{self.MAX_DEPTH}")
        self._check_int_option(options.get("offset"), 0, None, "InspectCode offset must be >= 0")
        ref_kind = options.get("ref_kind")
        if ref_kind is not None:
            if not isinstance(ref_kind, str):
                raise ToolError("InspectCode ref_kind must be a string")
            if options.get("all_kinds"):
                raise ToolError("InspectCode ref_kind and all_kinds are mutually exclusive")
            tokens = [token.strip() for token in ref_kind.split(",") if token.strip()]
            if unknown := sorted(set(tokens) - csi.REFERENCE_KINDS):
                raise ToolError("InspectCode unknown ref_kind: " + ", ".join(unknown) + "; valid: " + ", ".join(sorted(csi.REFERENCE_KINDS)))
        index = CodeIndex(self.session)
        if not index.available():
            raise ToolError("code index is not available; run /index")
        try:
            output = self.inspect_text(mode, target, options, limit)
        except csi.CodeSymbolIndexError as error:
            return self.process_result("InspectCodeToolResult", 1, "", str(error))
        return self.process_result("InspectCodeToolResult", 0, str(output), "")

    @staticmethod
    def _check_int_option(value: object, low: int, high: int | None, message: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
            raise ToolError(message)

    def inspect_text(self, mode: str, target: str, options: Json, limit: int | None) -> str:
        common = {
            "root": self.session.cwd,
            "kind": options.get("kind") or None,
            "path": options.get("path") or None,
            "exact_only": bool(options.get("exact_only")),
            "format": "text",
        }
        if mode == "find":
            return csi.search(target, limit=limit or csi.DEFAULT_SEARCH_LIMIT, **common)
        if mode == "inspect":
            return csi.inspect(target, limit=limit or csi.DEFAULT_PAGE_LIMIT, anchors=True, anchor_format="explicit", **common)
        if mode == "refs":
            ref_kinds = options.get("ref_kind") or ("all" if options.get("all_kinds") else "behavioral")
            return csi.refs(target, limit=limit or csi.DEFAULT_MAX_REFERENCES, offset=int(options.get("offset") or 0), ref_kinds=ref_kinds, **common)
        if mode == "impls":
            return csi.impls(target, limit=limit or csi.DEFAULT_MAX_IMPLEMENTORS, offset=int(options.get("offset") or 0), **common)
        if mode in self.CHAIN_MODES:
            depth = int(options.get("depth") or 3)
            if mode == "callees":
                return csi.callees(target, limit=limit or csi.DEFAULT_MAX_CALLEES, depth=depth, loose=bool(options.get("loose")), **common)
            return csi.callers(target, limit=limit or csi.DEFAULT_MAX_CALLERS, depth=depth, **common)
        symbol = options.get("symbol") or None
        return csi.outline(
            target, root=self.session.cwd, symbol=str(symbol) if symbol else None, max_symbols=limit or csi.DEFAULT_MAX_OUTLINE_SYMBOLS, format="text"
        )


@dataclass
class Edit:
    op: str
    start: str = ""
    end: str = ""
    content: str = ""
    old: str = ""
    new: str = ""


@dataclass
class EditApplyResult:
    content: str
    changes: list[tuple[int, int, int, int]]
    replacements: list[tuple[int, int, list[str]]]
    replace_all: bool = False


class EditTool(Tool):
    NAME = "Edit"
    DESCRIPTION = "Create or patch one UTF-8 file; op=create makes a new file; Edit start/end anchors are inclusive."
    SIGNATURE = "Edit(path, edits=[{op,start?,end?,content?,old?,new?}]); ops=create|replace|delete|insert_before|insert_after|replace_all"
    # fmt: off
    EXAMPLE = (
        'create file. Example: {"path":"src/app.py","edits":[{"op":"create","content":"print(1)\\n"}]}',
        'replace range. Example: {"path":"src/app.py","edits":[{"op":"replace","start":"10:1ab2c","end":"12:3de4f","content":"new_value = 1\\n"}]}',
        'delete range. Example: {"path":"src/app.py","edits":[{"op":"delete","start":"20:0aa11","end":"22:0bb22"}]}',
        'insert_before line. Example: {"path":"src/app.py","edits":[{"op":"insert_before","start":"30:0cc33","content":"setup()\\n"}]}',
        'insert_after line. Example: {"path":"src/app.py","edits":[{"op":"insert_after","start":"40:0dd44","content":"cleanup()\\n"}]}',
        'replace_all exact text; do not mix with anchored ops. Example: {"path":"src/app.py","edits":[{"op":"replace_all","old":"OldName","new":"NewName"}]}',
    )
    # fmt: on
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        edit = cls.object_schema({
            "op": {"type": "string", "description": "create|replace|delete|insert_before|insert_after|replace_all"},
            "start": {"type": "string", "description": "Start anchor line:hash (inclusive) for replace/delete/insert"},
            "end": {"type": "string", "description": "End anchor line:hash (inclusive) for replace/delete"},
            "content": {"type": "string", "description": "New text for create/replace/insert"},
            "old": {"type": "string", "description": "Text to find for replace_all"},
            "new": {"type": "string", "description": "Replacement text for replace_all"},
        }, ["op"])
        return cls.object_schema({
            "path": {"type": "string", "description": "File to create or patch"},
            "edits": {"type": "array", "items": edit, "minItems": 1, "description": "Ordered edit operations to apply"},
        }, ["path", "edits"])
        # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload.get("path", ""), payload.get("edits", [])]

    def call(self) -> str:
        path, original, created, result = self.build()
        if result.content == original and not created:
            raise ToolError(self.no_changes_error(original, result))
        if created:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(result.content)
        self.last_path = self.session.relpath(path)
        self.last_diff = self.diff(path, original, result.content)
        self.last_before = original
        self.last_after = result.content
        return "\n".join(
            [
                f"<Edit path={json.dumps(self.last_path)}>",
                self.file_stat(path),
                self.last_diff.rstrip(),
                self.edit_context(result.content, result.changes),
                "</Edit>",
            ]
        )

    def turn_diff(self) -> TurnDiff | None:
        path, diff = getattr(self, "last_path", ""), getattr(self, "last_diff", "")
        if not (path and diff):
            return None
        return TurnDiff(key="", turn=0, path=path, diff=diff, before=getattr(self, "last_before", ""), after=getattr(self, "last_after", ""))

    def preview(self) -> str:
        path, original, _, result = self.build()
        if result.content == original and os.path.exists(path):
            raise ToolError(self.no_changes_error(original, result))
        return self.diff(path, original, result.content) or f"Edit({path})"

    def short_args(self) -> list[str]:
        path = self.parse()[0]
        return [self.session.relpath(path)]

    def diff(self, path: str, original: str, new_content: str) -> str:
        relpath = self.session.relpath(path)
        return "".join(
            difflib.unified_diff(
                ReadTool.split_lines(original),
                ReadTool.split_lines(new_content),
                fromfile="/dev/null" if not original and not os.path.exists(path) else relpath,
                tofile=relpath,
            )
        )

    def parse(self) -> tuple[str, list[Edit]]:
        if len(self.args) != 2:
            raise ToolError("Edit requires path and edits")
        if not isinstance(self.args[0], str):
            raise ToolError("Edit path must be a string")
        path = self.session.resolve_path(str(self.args[0]))
        raw_edits = self.args[1]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ToolError("Edit edits must be a non-empty array")
        edits = []
        for item in raw_edits:
            if not isinstance(item, dict):
                raise ToolError("each edit must be an object")
            if unexpected := sorted(set(item) - {"op", "start", "end", "content", "old", "new"}):
                raise ToolError("Edit unexpected field: " + ", ".join(unexpected))
            op = str(item.get("op") or "")
            if op not in {"create", "replace", "delete", "insert_before", "insert_after", "replace_all"}:
                raise ToolError("unknown edit op")
            if op == "create" and len(raw_edits) != 1:
                raise ToolError("create cannot be mixed with other edits")
            if op in {"replace", "delete"} and (not item.get("start") or not item.get("end")):
                raise ToolError(f"{op} requires start and end anchors")
            if op in {"insert_before", "insert_after"} and not item.get("start"):
                raise ToolError(f"{op} requires start anchor")
            edits.append(
                Edit(
                    op=op,
                    start=str(item.get("start") or ""),
                    end=str(item.get("end") or ""),
                    content=self.content_text(str(item.get("content") or "")),
                    old=self.normalize_text(str(item.get("old") or "")),
                    new=self.content_text(str(item.get("new") or "")),
                )
            )
        return path, edits

    def _validate_target(self, path: str, creating: bool) -> bool:
        """Validate an edit/create target and return whether its current contents should be read."""

        if os.path.exists(path):
            if creating:
                raise ToolError("file already exists")
            if os.path.isdir(path):
                raise ToolError("path is a directory")
            return True
        if creating:
            parent = os.path.dirname(path) or "."
            if not self.session.in_cwd(parent):
                raise ToolError("refusing to create parent directories outside workspace")
            return False
        raise ToolError("file does not exist; use op=create to create it")

    def build(self) -> tuple[str, str, bool, EditApplyResult]:
        path, edits = self.parse()
        creating = edits[0].op == "create"
        if self._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.read()
            created = False
        else:
            original, created = "", True
        result = self.apply(original, edits)
        return path, original, created, result

    def apply(self, original: str, edits: list[Edit], anchor_resolver: Callable[[str], int] | None = None) -> EditApplyResult:
        if edits[0].op == "create":
            lines = self.content_lines(edits[0].content, False)
            return EditApplyResult("".join(lines), [(0, 0, 0, len(lines))], [])
        if any(edit.op == "replace_all" for edit in edits):
            if any(edit.op != "replace_all" for edit in edits):
                raise ToolError("replace_all cannot be mixed with anchored edits")
            content = original
            for edit in edits:
                if not edit.old and content:
                    raise ToolError("replace_all requires old")
                if edit.old and edit.old not in content:
                    raise ToolError("replace_all old text not found")
                content = content.replace(edit.old, edit.new)
            return EditApplyResult(content, [(0, 0, 0, len(ReadTool.split_lines(content)))], [], True)
        lines = ReadTool.split_lines(original)
        resolve_anchor = anchor_resolver or (lambda anchor: self.resolve_anchor(lines, anchor))
        replacements = []
        for edit in edits:
            start = resolve_anchor(edit.start)
            if edit.op in {"replace", "delete"}:
                end = resolve_anchor(edit.end)
                if end < start:
                    raise ToolError("end anchor is before start anchor")
                replacement = [] if edit.op == "delete" else self.content_lines(edit.content, end + 1 < len(lines))
                replacements.append((start, end + 1, replacement))
            elif edit.op in {"insert_before", "insert_after"}:
                index = start if edit.op == "insert_before" else start + 1
                replacements.append((index, index, self.content_lines(edit.content, index < len(lines))))
            else:
                raise ToolError("unknown edit op")
        previous = None
        for start, end, _ in sorted(replacements):
            if previous and (start < previous[1] or (start == previous[0] and end == previous[1])):
                raise ToolError(f"edits overlap or share an insertion point: {previous[0]}:{previous[1]} and {start}:{end}")
            previous = (start, end)
        new_lines = list(lines)
        for start, end, replacement in sorted(replacements, reverse=True):
            new_lines[start:end] = replacement
        changes = []
        delta = 0
        for start, end, replacement in sorted(replacements):
            new_start = start + delta
            new_end = new_start + len(replacement)
            clear_end = 0 if len(replacement) != end - start else new_start + (end - start)
            changes.append((new_start, clear_end, new_start, new_end))
            delta += len(replacement) - (end - start)
        return EditApplyResult("".join(new_lines), changes, replacements)

    def no_changes_error(self, original: str, result: EditApplyResult) -> str:
        return self.no_changes_error_from_lines(ReadTool.split_lines(original), result.replacements, result.replace_all)

    @classmethod
    def no_changes_error_from_lines(cls, lines: list[str], replacements: list[tuple[int, int, list[str]]], replace_all: bool) -> str:
        prefix = "edit produced no changes"
        if replace_all:
            return prefix + "; replace_all result is identical to current file"
        if not replacements:
            return prefix
        matching = [(start, end) for start, end, replacement in replacements if lines[start:end] == replacement]
        if len(matching) != len(replacements):
            return prefix + "; edits cancel out; check requested content"
        return prefix + "; requested content already matches target range\n" + cls.format_current_ranges(lines, matching)

    @classmethod
    def format_current_ranges(cls, lines: list[str], ranges: list[tuple[int, int]]) -> str:
        out = ["<current-target-ranges hashline-numbered>"]
        shown_lines = 0
        range_index = -1
        for range_index, (start, end) in enumerate(ranges[:3]):
            out.append(f"<target start={start} end={end}>")
            if start == end:
                out.append("(empty range)")
            else:
                for index in range(start, end):
                    if shown_lines >= 12:
                        out.append("...")
                        break
                    line = lines[index]
                    out.append(ReadTool.anchor_line(index, line))
                    shown_lines += 1
            out.append("</target>")
            if shown_lines >= 12:
                break
        if len(ranges) > range_index + 1:
            out.append("...")
        out.append("</current-target-ranges>")
        return "\n".join(out)

    def edit_context(self, content: str, changes: list[tuple[int, int, int, int]]) -> str:
        lines = ReadTool.split_lines(content)
        out = []
        for clear_start, clear_end, start, end in changes:
            out.append(f"<invalidate>{clear_start}:{clear_end}</invalidate>")
            shown = lines[start:end]
            if shown:
                out.append("<content hashline-numbered>")
                out.extend(ReadTool.anchor_line(start + index, line) for index, line in enumerate(shown))
                out.append("</content>")
        return "\n".join(out)

    def content_lines(self, content: str, followed_by_more: bool) -> list[str]:
        content = self.normalize_text(content)
        if content == "":
            return []
        lines = ReadTool.split_lines(content)
        if followed_by_more and lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def content_text(cls, value: str) -> str:
        value = cls.normalize_text(value)
        return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t") if "\n" not in value and "\\n" in value else value

    def resolve_anchor(self, lines: list[str], anchor: str) -> int:
        index, expected = ReadTool.require_anchor(anchor)
        if index >= len(lines):
            raise ToolError("anchor line out of range")
        if not ReadTool.anchor_matches(lines[index], expected):
            current = ReadTool.anchor_line(index, lines[index])
            raise ToolError(f"stale anchor {anchor}; current is {current}")
        return index


class BashTool(Tool):
    NAME = "Bash"
    LOG_LEXER = "bash"
    DESCRIPTION = "Run one bash shell invocation in the workspace; returns exit_code/stdout/stderr and shows live output. Avoid unbounded output; limit noisy commands with head/tail/sed/rg filters or command-specific limits, and inspect large outputs in chunks."
    SIGNATURE = "Bash(command)"
    # fmt: off
    EXAMPLE = (
        'Check environment. Example: {"command":"python3 --version"}',
        'Run a project command. Example: {"command":"python3 -m py_compile minacode.py"}',
    )
    # fmt: on
    MUTATES = True
    live_output: Callable[[str, str], None] | None = None

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        with self._process_lock:
            proc = self._process
        if proc is not None and proc.poll() is None:
            self.kill_process_group(proc)

    # Read-only executables that only inspect the filesystem/repo. A command built solely from these
    # (and safe git subcommands) auto-runs without a confirmation prompt in non-yolo mode, replacing
    # the dedicated List/Find/LineCount/read-only-Git tools that were removed in favour of Bash.
    # fmt: off
    SAFE_COMMANDS: ClassVar[frozenset[str]] = frozenset(
        {
            # Common read-only inspection commands. The obvious file-writing forms (`sort -o`,
            # `uniq IN OUT`, `sed -i`, `tree -o`) are guarded below; we do not chase exotic paths
            # like sed's `w` command — common sense over exhaustive safety.
            "ls", "cat", "head", "tail", "wc", "find", "grep", "egrep", "fgrep", "rg", "sort", "uniq",
            "sed", "tree", "cut", "tr", "nl", "comm", "column", "fold", "paste", "join", "echo", "printf", "pwd",
            "stat", "file", "basename", "dirname", "realpath", "readlink", "which", "type",
            "diff", "cmp", "date", "printenv", "du", "df", "jq", "true", "test", "uname", "hostname",
            # Benign builtin the model routinely prefixes (cd changes the subshell dir only).
            "cd",
        }
    )
    SAFE_GIT_SUBCOMMANDS: ClassVar[frozenset[str]] = frozenset(
        {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame", "describe",
         "shortlog", "cat-file", "ls-tree", "rev-list", "for-each-ref", "diff-tree"}
    )
    # fmt: on

    def needs_confirmation(self) -> bool:
        try:
            return not self.is_readonly(self.command())
        except ToolError:
            return True

    @classmethod
    def is_readonly(cls, command: str) -> bool:
        """Conservatively classify a command as safe to auto-run. Bias hard toward False: a false
        'safe' would run a mutating command without consent, while a false 'unsafe' only costs a
        confirmation prompt. Rejects anything that can write, execute arbitrary code, or background."""
        command = command.strip()
        if not command:
            return False
        # Normalize away the ubiquitous harmless redirections — discarding output to /dev/null and
        # merging stderr/stdout — so the common `cmd 2>/dev/null` / `cmd >/dev/null 2>&1` forms are
        # not treated as file writes.
        scan = re.sub(r"(?:\d*>>?|&>|<)\s*/dev/null(?![\w./])", " ", command)
        scan = scan.replace("2>&1", " ").replace(">&2", " ")
        # Anything still redirecting to/from a real path, or substituting a command, can write or
        # run arbitrary code.
        if any(ch in scan for ch in (">", "<", "`")) or "$(" in scan:
            return False
        # Reject a lone background & (detaches a process); && and || are allowed sequence operators.
        if re.search(r"(?<!&)&(?!&)", scan):
            return False
        # Split on every control operator (&& || | ; newline) and require EVERY stage to be a safe
        # read-only command — so `git log && rm x` is not auto-approved on the strength of `git log`.
        return all(cls._safe_segment(part) for part in re.split(r"&&|\|\||[|;\n]", scan) if part.strip())

    @classmethod
    def _safe_segment(cls, segment: str) -> bool:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            return False
        cmd = tokens[0]
        # Env assignments and wrapper commands can hide arbitrary execution — never auto-approve.
        # fmt: off
        if "=" in cmd or cmd in {"env", "sudo", "eval", "exec", "command", "xargs", "nohup", "time",
                                 "watch", "bash", "sh", "zsh", "tee", "awk", "python", "python3"}:
            return False
        # fmt: on
        if cmd == "git":
            return cls._safe_git(tokens)
        if cmd not in cls.SAFE_COMMANDS:
            return False
        # Flags/args that turn a read-only command into a writer.
        if cmd == "find" and any(t in {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"} for t in tokens):
            return False
        if cmd == "sed" and any(t.startswith(("-i", "--in-place")) for t in tokens):
            return False
        if cmd == "tree" and any(t.startswith(("-o", "--output")) for t in tokens):
            return False  # `tree -o FILE` writes the listing to a file
        if cmd == "sort" and any(t.startswith(("-o", "--output")) for t in tokens):
            return False  # `sort -o FILE` / `--output=FILE` writes to a file
        # `uniq INPUT OUTPUT` writes the second file operand.
        return not (cmd == "uniq" and cls._uniq_writes(tokens))

    @staticmethod
    def _uniq_writes(tokens: list[str]) -> bool:
        # uniq writes only in the two-operand form `uniq [OPTS] INPUT OUTPUT`. Count positional
        # operands, skipping the numeric argument that follows a value-taking short flag.
        value_flags = {"-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"}
        operands = 0
        skip_next = False
        for token in tokens[1:]:
            if skip_next:
                skip_next = False
            elif token in value_flags:
                skip_next = True
            elif not token.startswith("-"):
                operands += 1
        return operands >= 2

    @classmethod
    def _safe_git(cls, tokens: list[str]) -> bool:
        index = 1
        while index < len(tokens) and tokens[index] == "--no-pager":
            index += 1
        if index >= len(tokens):
            return False
        sub = tokens[index]
        if sub not in cls.SAFE_GIT_SUBCOMMANDS:
            return False
        args = tokens[index + 1 :]
        if any(t == "--output" or t.startswith("--output=") for t in args):
            return False
        return not (sub == "grep" and any(t.startswith(("-O", "--open-files-in-pager")) for t in args))

    # fmt: off
    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"command": {"type": "string", "minLength": 1, "pattern": "^.*\\S.*$", "description": "Bash command to run in the workspace; filter noisy output with head/tail/rg"}}, ["command"])
    # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        command = str(payload.get("command") or "")
        if not command.strip():
            raise ToolError("Bash command must be non-empty")
        return [command]

    def command(self) -> str:
        command = self.strings(min_count=1, max_count=1)[0]
        if not command.strip():
            raise ToolError("Bash command must be non-empty")
        return command

    def short_args(self) -> list[str]:
        return [self.command()]

    def call(self) -> str:
        command = self.command()
        bash = shutil.which("bash") or "bash"
        proc = None
        try:
            proc = subprocess.Popen(
                [bash, "-lc", command], cwd=self.session.cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
            )
            with self._process_lock:
                self._process = proc
            assert proc.stdout is not None and proc.stderr is not None
            return self.stream_process(proc)
        except KeyboardInterrupt:
            self.kill_and_collect(proc)
            raise
        finally:
            with self._process_lock:
                if self._process is proc:
                    self._process = None
            if self.live_output is not None:
                self.live_output("", "")

    def stream_process(self, proc: subprocess.Popen[bytes]) -> str:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        # Per-stream incremental decoders so a multibyte UTF-8 character split across two 4096-byte
        # reads is decoded once it is complete, instead of being mangled into replacement chars.
        self._decoders = {"stdout": codecs.getincrementaldecoder("utf-8")("replace"), "stderr": codecs.getincrementaldecoder("utf-8")("replace")}
        selector = selectors.DefaultSelector()
        stdout, stderr = proc.stdout, proc.stderr
        assert stdout is not None and stderr is not None
        selector.register(stdout, selectors.EVENT_READ, "stdout")
        selector.register(stderr, selectors.EVENT_READ, "stderr")
        timed_out = False
        started = time.monotonic()
        shell_deadline = started + self.session.settings.shell_timeout
        wait_budget = self.session.settings.bash_wait_timeout
        # Auto-promotion: if the command hasn't exited within bash_wait_timeout, hand the still-
        # running proc to the background jobs registry and return control to the model with a
        # partial-output payload. Disabled when the setting is 0 or the wait budget is already
        # >= shell_timeout (in which case we would kill on the same deadline anyway).
        promote_deadline = started + wait_budget if wait_budget and wait_budget < self.session.settings.shell_timeout else None
        try:
            while selector.get_map() or proc.poll() is None:
                now = time.monotonic()
                if promote_deadline is not None and now >= promote_deadline and proc.poll() is None:
                    # Don't drain here: drain_selector does BLOCKING os.reads, which would wait
                    # until bash produced more output (or exited) — defeating the whole point of
                    # promotion. Whatever data the streaming loop already read is the partial
                    # payload; anything still in-flight becomes the drainer thread's first read.
                    return self.promote_to_job(proc, selector, stdout_parts, stderr_parts)
                remaining = shell_deadline - now
                if remaining <= 0:
                    timed_out = True
                    self.kill_process_group(proc)
                    proc.wait()
                    self.drain_selector(selector, stdout_parts, stderr_parts)
                    break
                wait = min(0.2, remaining, promote_deadline - now if promote_deadline is not None else remaining)
                if selector.get_map():
                    for key, _ in selector.select(max(0.0, wait)):
                        self.read_stream_chunk(selector, key, stdout_parts, stderr_parts)
                else:
                    time.sleep(max(0.0, wait))
            if proc.returncode is None:
                proc.wait()
        finally:
            selector.close()
        stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
        if timed_out:
            stderr += ("\n" if stderr else "") + "timeout"
            return self.process_result("BashToolResult", -1, stdout, stderr)
        return self.process_result("BashToolResult", proc.returncode or 0, stdout, stderr)

    def promote_to_job(
        self,
        proc: subprocess.Popen[bytes],
        selector: selectors.BaseSelector,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> str:
        """Hand off a still-running Bash proc to the background job registry. Closes the streaming
        selector, starts a drainer thread that keeps reading proc.stdout/stderr into an in-memory
        tail buffer (bounded), and returns a partial-output payload for the model."""
        # Take pipe handles before closing the selector so the drainer can keep reading them.
        stdout_pipe, stderr_pipe = proc.stdout, proc.stderr
        with contextlib.suppress(OSError):
            selector.close()
        self.session.job_counter += 1
        job_id = f"job.{self.session.job_counter}"
        buffer: list[str] = []
        buffer_lock = threading.Lock()
        job = BackgroundJob(
            id=job_id,
            command=self.command(),
            process=proc,
            log_path="",
            started_at=time.monotonic() - self.session.settings.bash_wait_timeout,
            stream_buffer=buffer,
            stream_lock=buffer_lock,
        )
        self.session.jobs[job_id] = job

        def drain_pipe(pipe: Any) -> None:
            if pipe is None:
                return
            try:
                # read1 returns whatever is immediately available (line-buffered producers ship one
                # line per call), so a slow trickle of output lands in the tail buffer promptly
                # instead of blocking until a full 4KB is buffered.
                for chunk in iter(lambda: pipe.read1(4096), b""):
                    text = chunk.decode("utf-8", errors="replace")
                    with buffer_lock:
                        buffer.append(text)
                        # Trim from the front once we exceed the cap, keeping the tail intact.
                        total = sum(len(part) for part in buffer)
                        while total > BackgroundJob.BUFFER_LIMIT and len(buffer) > 1:
                            total -= len(buffer.pop(0))
            except (OSError, ValueError):
                return

        threading.Thread(target=drain_pipe, args=(stdout_pipe,), daemon=True).start()
        threading.Thread(target=drain_pipe, args=(stderr_pipe,), daemon=True).start()
        partial_stdout = "".join(stdout_parts)
        partial_stderr = "".join(stderr_parts)
        note = (
            f'backgrounded after {self.session.settings.bash_wait_timeout}s; still running as {job_id}. Use Job(action="wait"|"status"|"kill", job="{job_id}").'
        )
        partial_stderr = partial_stderr + ("\n" if partial_stderr else "") + note
        return self.process_result("BashToolResult", -1, partial_stdout, partial_stderr)

    def drain_selector(self, selector: selectors.BaseSelector, stdout_parts: list[str], stderr_parts: list[str]) -> None:
        for key in list(selector.get_map().values()):
            while self.read_stream_chunk(selector, key, stdout_parts, stderr_parts):
                pass

    def read_stream_chunk(
        self,
        selector: selectors.BaseSelector,
        key: selectors.SelectorKey,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> bool:
        try:
            data = os.read(cast(Any, key.fileobj).fileno(), 4096)
        except OSError:
            data = b""
        eof = not data
        if eof:
            with contextlib.suppress(Exception):
                selector.unregister(key.fileobj)
            with contextlib.suppress(Exception):
                cast(Any, key.fileobj).close()
        # final=True on EOF flushes any bytes still buffered in the decoder (e.g. a truncated
        # trailing character) so they are not silently dropped.
        text = self._decoders[key.data].decode(data, final=eof)
        if text:
            (stdout_parts if key.data == "stdout" else stderr_parts).append(text)
            if self.live_output is not None:
                self.live_output(str(key.data), text)
        return not eof

    @staticmethod
    def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                proc.kill()

    @classmethod
    def kill_and_collect(cls, proc: subprocess.Popen[bytes] | None) -> tuple[str, str]:
        if proc is None:
            return "", ""
        cls.kill_process_group(proc)
        stdout, stderr = proc.communicate()

        def decode(value: bytes | str | None) -> str:
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""

        return decode(stdout), decode(stderr)


class JobTool(Tool):
    NAME = "Job"
    DESCRIPTION = "Start, monitor, wait for, list, and kill background shell jobs. Processes run in their own process group and do not block the agent."
    SIGNATURE = 'Job(action="start"|"status"|"wait"|"list"|"kill", command?, job?, timeout?, limit?)'
    MUTATES = True
    ACTIONS: ClassVar[tuple[str, ...]] = ("start", "status", "wait", "list", "kill")
    MAX_JOBS: ClassVar[int] = 8
    DEFAULT_LIMIT: ClassVar[int] = 4096

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": list(cls.ACTIONS), "description": "Operation to perform"},
            "command": {"type": "string", "minLength": 1, "description": "Shell command to run for action=start"},
            "job": {"type": "string", "description": "Job id for action=status, wait, or kill"},
            "timeout": {"type": "integer", "minimum": 0, "description": "Seconds to wait for action=wait (0 means block until the process exits)"},
            "limit": {"type": "integer", "minimum": 1, "description": "Max characters of stdout/stderr to return; default 4096"},
        }, ["action"])
        # fmt: on

    def payload(self) -> Json:
        return self.single_dict_arg("Job requires a single object argument")

    def resolved_action(self, payload: Json) -> str:
        action = str(payload.get("action") or "").strip()
        if action not in self.ACTIONS:
            raise ToolError(f"unknown action: {action!r}")
        return action

    def needs_confirmation(self) -> bool:
        return self.resolved_action(self.payload()) in {"start", "kill", "wait"}

    def short_args(self) -> list[str]:
        payload = self.payload()
        action = self.resolved_action(payload)
        if action == "start":
            return [str(payload.get("command") or "")]
        if action == "list":
            return ["list"]
        return [action, str(payload.get("job") or "")]

    @classmethod
    def log_lexer(cls, args: ToolArgs) -> str:
        payload = args[0] if len(args) == 1 and isinstance(args[0], dict) else {}
        return "bash" if payload.get("action") == "start" else cls.LOG_LEXER

    def call(self) -> str:
        payload = self.payload()
        action = self.resolved_action(payload)
        if action == "start":
            return self._start(payload)
        if action == "status":
            return self._status(payload)
        if action == "wait":
            return self._wait(payload)
        if action == "list":
            return self._list()
        if action == "kill":
            return self._kill(payload)
        raise ToolError(f"unhandled action: {action!r}")

    def _start(self, payload: Json) -> str:
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ToolError("start requires a non-empty command")
        active = len(self.session.running_jobs())
        if active >= self.MAX_JOBS:
            raise ToolError(f"too many active jobs ({active}/{self.MAX_JOBS}); kill or wait for one first")
        self.session.job_counter += 1
        job_id = f"job.{self.session.job_counter}"
        # Log to disk (stdout+stderr merged) so we don't need a threaded drainer to keep the
        # subprocess's OS-level pipe buffers from filling. The command is wrapped in a `{ ...; }`
        # group so the redirection captures every stage of a compound command, not just the last
        # (`a; b && c` would otherwise leak its earlier stages to the inherited stdout).
        # `start_new_session` makes this shell its own process-group leader and the command inherits
        # that group, so killpg(pid) reaches the command and its children; running it directly (no
        # `exec`) keeps builtins like `cd` working.
        fd, log_path = tempfile.mkstemp(prefix=f"nc-{job_id}-", suffix=".log")
        os.close(fd)
        proc = subprocess.Popen(
            ["bash", "-lc", f"{{ {command}; }} > {shlex.quote(log_path)} 2>&1"],
            cwd=self.session.cwd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.session.jobs[job_id] = BackgroundJob(id=job_id, command=command, process=proc, log_path=log_path, started_at=time.monotonic())
        return f"Started {job_id}: {command}"

    def _status(self, payload: Json) -> str:
        job = self._resolve_job(payload)
        timeout = int(payload.get("timeout") or 0)
        if timeout > 0:
            with contextlib.suppress(subprocess.TimeoutExpired):
                job.process.wait(timeout=timeout)
        job.update_status()
        return self._format(job, payload)

    def _wait(self, payload: Json) -> str:
        job = self._resolve_job(payload)
        timeout = payload.get("timeout")
        with contextlib.suppress(subprocess.TimeoutExpired):
            # timeout omitted or 0 means block until the process exits (per the schema).
            job.process.wait(timeout=None if not timeout else max(1, int(timeout)))
        job.update_status()
        return self._format(job, payload)

    def _list(self) -> str:
        if not self.session.jobs:
            return "No jobs."
        self.session.running_jobs()
        rows = []
        for job in self.session.jobs.values():
            exit_code = job.exit_code if job.status != "running" else "-"
            rows.append(f"| {job.id} | {job.status} | {exit_code} | {job.command[:60]} |")
        return "Jobs:\n| id | status | exit | command |\n|---|---|---|---|\n" + "\n".join(rows)

    def _kill(self, payload: Json) -> str:
        job = self._resolve_job(payload)
        job.kill()
        return f"Killed {job.id} (status={job.status}, exit_code={job.exit_code})"

    def _resolve_job(self, payload: Json) -> BackgroundJob:
        job_id = str(payload.get("job") or "").strip()
        if not job_id:
            raise ToolError("job id required")
        # Allow bare numeric IDs as a shorthand for the canonical "job.N" form.
        if job_id not in self.session.jobs and not job_id.startswith("job.") and job_id.isdigit():
            job_id = f"job.{job_id}"
        job = self.session.jobs.get(job_id)
        if job is None:
            raise ToolError(f"unknown job: {job_id!r}")
        job.update_status()
        return job

    def _format(self, job: BackgroundJob, payload: Json) -> str:
        limit = max(1, int(payload.get("limit") or self.DEFAULT_LIMIT))
        output = job.tail(limit)
        lines = [
            f"Job: {job.id}",
            f"Status: {job.status}",
            f"Command: {job.command}",
            f"Elapsed: {job.elapsed():.1f}s",
        ]
        if job.exit_code is not None:
            lines.append(f"Exit code: {job.exit_code}")
        if output:
            lines.extend(["--- output ---", output])
        return "\n".join(lines)


class RecallTool(Tool):
    NAME = "Recall"
    DESCRIPTION = "Recall stored non-Recall tool results by tr.N key; ranges slice output lines to control context."
    SIGNATURE = "Recall(keys=[tr.N,...], ranges?); ranges are 0-based [start,end] output lines"
    # fmt: off
    EXAMPLE = (
        'Recall full result. Example: {"keys":["tr.1"]}',
        'Recall output line ranges. Example: {"keys":["tr.1","tr.2"],"ranges":[[0,80]]}',
    )
    # fmt: on
    STORES_RESULT = False

    # fmt: off
    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"keys": {"type": "array", "items": {"type": "string", "pattern": "^tr\\.\\d+$"}, "minItems": 1, "description": "Stored result keys to recall, e.g. [\"tr.3\",\"tr.5\"]"}, "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1, "description": "Optional 0-based [start,end] output-line slices to limit recalled context"}}, ["keys"])
    # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [{"keys": payload.get("keys", []), **({"ranges": payload["ranges"]} if "ranges" in payload else {})}]

    def call(self) -> str:
        requests = self.requests()
        if not requests:
            raise ToolError("Recall requires at least one key")
        chunks = ["<RecallToolResult>"]
        for key, ranges in requests:
            value = self.session.tool_results.get(key)
            if value is None:
                chunks.append(f"* {key}: missing")
                continue
            chunks.append(f"<Result key={json.dumps(key)}>")
            chunks.append(self.slice(value, ranges).rstrip())
            chunks.append("</Result>")
        chunks.append("</RecallToolResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        rows = []
        for key, ranges in self.requests():
            row = key
            if ranges:
                row += " " + ",".join(f"{start}:{end}" for start, end in ranges)
            rows.append(row)
        return ["; ".join(rows)]

    def requests(self) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
        payload = self.single_dict_arg("Recall requires keys")
        if unexpected := sorted(set(payload) - {"keys", "ranges"}):
            raise ToolError("Recall unexpected field: " + ", ".join(unexpected))
        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ToolError("Recall requires keys")
        ranges = self.parse_ranges(payload)
        keys = []
        for item in raw_keys:
            key = str(item).strip()
            if not re.fullmatch(r"tr\.\d+", key):
                raise ToolError("Recall key must look like tr.N")
            keys.append(key)
        return list(dict.fromkeys((key, ranges) for key in keys))

    def parse_ranges(self, payload: Json) -> tuple[tuple[int, int], ...]:
        raw = payload.get("ranges")
        if raw is None:
            return ()
        if not isinstance(raw, list) or not raw:
            raise ToolError("Recall ranges must be a non-empty array")
        return tuple(self.line_range(item, "Recall range") for item in raw)

    @staticmethod
    def slice(value: str, ranges: tuple[tuple[int, int], ...]) -> str:
        if not ranges:
            return value
        lines = value.splitlines()
        return "\n".join("\n".join(lines[start:end]) for start, end in ranges if end > start)


class RecallContextTool(Tool):
    NAME = "RecallContext"
    DESCRIPTION = "Recall stored compacted-conversation excerpts by seg.N key, or regex-search their titles and text; query alternation A|B|C is allowed."
    SIGNATURE = "RecallContext(keys=[seg.N,...]) or RecallContext(query=REGEX,keys?,case_sensitive?,limit?)"
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    MAX_QUERY_LENGTH = 500
    # fmt: off
    EXAMPLE = (
        'Recall one segment. Example: {"keys":["seg.1"]}',
        'Recall several segments. Example: {"keys":["seg.1","seg.3"]}',
        'Search all segments. Example: {"query":"cache prefix|task memory","limit":10}',
        'Search selected segments. Example: {"keys":["seg.1","seg.3"],"query":"compaction","case_sensitive":false}',
    )
    # fmt: on
    STORES_RESULT = False

    # fmt: off
    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({
            "keys": {"type": "array", "items": {"type": "string", "pattern": "^seg\\.\\d+$"}, "minItems": 1, "description": "Segment keys to retrieve, or to restrict query search"},
            "query": {"type": "string", "maxLength": cls.MAX_QUERY_LENGTH, "description": "Case-insensitive regex over segment titles and text; A|B|C is allowed"},
            "case_sensitive": {"type": "boolean", "description": "Make query matching case-sensitive; default false"},
            "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_LIMIT, "description": f"Maximum matching lines to return; default {cls.DEFAULT_LIMIT}"},
        })
    # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload]

    def call(self) -> str:
        request = self.request()
        if request["query"]:
            return self.search(request)
        keys = request["keys"]
        segments = {segment.key: segment for segment in self.session.history}
        chunks = ["<RecallContextResult>"]
        for key in keys:
            segment = segments.get(key)
            if segment is None:
                chunks.append(f"* {key}: missing")
                continue
            chunks.append(f"<Segment key={json.dumps(key)} title={json.dumps(segment.title)}>")
            chunks.append(segment.text.rstrip())
            chunks.append("</Segment>")
        chunks.append("</RecallContextResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        request = self.request()
        if request["query"]:
            scope = " in " + ",".join(request["keys"]) if request["keys"] else ""
            return [json.dumps(request["query"], ensure_ascii=False) + scope]
        return ["; ".join(request["keys"])]

    def request(self) -> Json:
        payload = self.single_dict_arg("RecallContext requires keys or query")
        if unexpected := sorted(set(payload) - {"keys", "query", "case_sensitive", "limit"}):
            raise ToolError("RecallContext unexpected field: " + ", ".join(unexpected))
        raw_keys = payload.get("keys")
        if raw_keys is not None and (not isinstance(raw_keys, list) or not raw_keys):
            raise ToolError("RecallContext keys must be a non-empty array")
        keys = []
        for item in raw_keys or []:
            key = str(item).strip()
            if not re.fullmatch(r"seg\.\d+", key):
                raise ToolError("RecallContext key must look like seg.N")
            keys.append(key)
        query = payload.get("query")
        if query is not None and (not isinstance(query, str) or not query.strip()):
            raise ToolError("RecallContext query must be a non-empty regex")
        query = query.strip() if isinstance(query, str) else ""
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ToolError(f"RecallContext query must be at most {self.MAX_QUERY_LENGTH} characters")
        case_sensitive = payload.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise ToolError("RecallContext case_sensitive must be boolean")
        limit = payload.get("limit", self.DEFAULT_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > self.MAX_LIMIT:
            raise ToolError(f"RecallContext limit must be 1..{self.MAX_LIMIT}")
        if not keys and not query:
            raise ToolError("RecallContext requires keys or query")
        if not query and ("case_sensitive" in payload or "limit" in payload):
            raise ToolError("RecallContext case_sensitive and limit require query")
        return {"keys": list(dict.fromkeys(keys)), "query": query, "case_sensitive": case_sensitive, "limit": limit}

    def search(self, request: Json) -> str:
        regex = self.compile_regex(request["query"], case_sensitive=request["case_sensitive"])
        by_key = {segment.key: segment for segment in self.session.history}
        segments = [by_key[key] for key in request["keys"] if key in by_key] if request["keys"] else self.session.history
        rows = []
        for segment in segments:
            if regex.search(segment.title):
                rows.append(self.match_row(segment, "title", segment.title))
            for line_number, line in enumerate(segment.text.splitlines(), 1):
                if regex.search(line):
                    rows.append(self.match_row(segment, str(line_number), line))
                if len(rows) >= request["limit"]:
                    break
            if len(rows) >= request["limit"]:
                break
        rows = rows[: request["limit"]]
        header = f"<RecallContextSearchResult query={json.dumps(request['query'])} matches={len(rows)}>"
        return "\n".join([header, *rows, "</RecallContextSearchResult>"])

    @staticmethod
    def match_row(segment: HistorySegment, location: str, text: str) -> str:
        return f"- {segment.key} {location} title={json.dumps(segment.title)}: {Tool.compact(text, 300)}"


class NoteTool(Tool):
    NAME = "Note"
    DESCRIPTION = "Maintain durable working notes; set_goal, replace_plan, and set_check replace current values, append_known appends, replace_known replaces all known facts. Plan items are objects with status todo|doing|done|blocked and text."
    SIGNATURE = "Note(set_goal?, replace_plan?, append_known?, replace_known?, set_check?)"
    # fmt: off
    EXAMPLE = (
        'Set memory. Example: {"set_goal":"ship parser fix","replace_plan":[{"status":"doing","text":"inspect parser"},{"status":"todo","text":"patch bug"}],"append_known":["tests use pytest"],"set_check":"pytest -q passed"}',
    )
    # fmt: on
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        plan_item = cls.object_schema({
            "status": {"type": "string", "enum": list(PlanItem.STATUSES), "description": "todo|doing|done|blocked"},
            "text": {"type": "string", "description": "Plan step description"},
        }, ["status", "text"])
        return cls.object_schema({
            "set_goal": {"type": "string", "description": "Replace the current goal"},
            "replace_plan": {"type": "array", "items": plan_item, "description": "Replace the plan with these status/text items"},
            "append_known": {"type": "array", "items": {"type": "string"}, "description": "Append these facts to known"},
            "replace_known": {"type": "array", "items": {"type": "string"}, "description": "Replace all known facts with these"},
            "set_check": {"type": "string", "description": "Replace the success/verification criteria"},
        })
        # fmt: on

    def call(self) -> str:
        data = self.single_dict_arg("Note requires named fields")
        if unexpected := sorted(set(data) - {"set_goal", "replace_plan", "append_known", "replace_known", "set_check"}):
            raise ToolError("Note unexpected field: " + ", ".join(unexpected))
        changed = []
        goal = self.session.state.goal
        plan = list(self.session.state.plan)
        known = list(self.session.state.known)
        check = self.session.state.check
        if "set_goal" in data:
            goal = str(data["set_goal"]).strip()
            changed.append("set_goal")
        if "set_check" in data:
            check = str(data["set_check"]).strip()
            changed.append("set_check")
        if "replace_plan" in data:
            if not isinstance(data["replace_plan"], list):
                raise ToolError('Note replace_plan must be an array of plan items, e.g. {"replace_plan":[{"status":"doing","text":"inspect"}]}')
            for item in data["replace_plan"]:
                if isinstance(item, dict):
                    if str(item.get("status") or "").strip().lower() not in PlanItem.STATUSES:
                        raise ToolError("Note replace_plan status must be one of: " + ", ".join(PlanItem.STATUSES))
                    if not str(item.get("text") or "").strip():
                        raise ToolError("Note replace_plan text is required")
            plan = cast(list[PlanItem | Json | str], AgentState.plan_items(data["replace_plan"]))
            changed.append("replace_plan")
        if "append_known" in data:
            if not isinstance(data["append_known"], list):
                raise ToolError('Note append_known must be an array of strings, e.g. {"append_known":["tests use pytest"]}')
            known = list(dict.fromkeys([*known, *(str(item).strip() for item in data["append_known"] if str(item).strip())]))
            changed.append("append_known")
        if "replace_known" in data:
            if not isinstance(data["replace_known"], list):
                raise ToolError('Note replace_known must be an array of strings, e.g. {"replace_known":["fact"]}')
            known = [str(item).strip() for item in data["replace_known"] if str(item).strip()]
            changed.append("replace_known")
        if not changed:
            raise ToolError("Note requires set_goal, replace_plan, append_known, replace_known, or set_check")
        self.session.state.goal = goal
        self.session.state.plan = plan
        self.session.state.known = known
        self.session.state.check = check
        return "Updated memory: " + ", ".join(changed)

    def short_args(self) -> list[str]:
        data = self.args[0] if self.args and isinstance(self.args[0], dict) else {}
        lines = []
        if goal := str(data.get("set_goal") or "").strip():
            lines.append("goal: " + Tool.compact(goal, 120))
        if check := str(data.get("set_check") or "").strip():
            lines.append("check: " + Tool.compact(check, 120))
        if isinstance(data.get("replace_plan"), list):
            lines.extend(["plan:", *(f"  {row}" for row in AgentState.plan_rows_for(data["replace_plan"], status=True, style="symbol") if row != "- (empty)")])
        if isinstance(data.get("append_known"), list):
            known = [Tool.compact(item, 120) for item in data["append_known"] if str(item).strip() and str(item).strip() not in self.session.state.known]
            if known:
                lines.extend(["known:", *(f"  + {item}" for item in known)])
        if isinstance(data.get("replace_known"), list):
            known = [Tool.compact(item, 120) for item in data["replace_known"] if str(item).strip()]
            if known:
                lines.extend(["known:", *(f"  {item}" for item in known)])
        return ["\n".join(lines) or "{}"]


@dataclass(frozen=True)
class AskSpec:
    """One validated question the model wants to ask the user."""

    question: str
    choices: list[str] | None = None
    previews: list[str] | None = None
    recommended: int | None = None


class AskTool(Tool):
    NAME = "Ask"
    DESCRIPTION = "Ask the user one or more questions (asked in sequence) and wait for their answers. Use when intent is genuinely ambiguous, a choice affects the codebase's external shape (module layout, public API, naming), or you need prioritization; prefer offering choices with previews, and optionally a recommended index when one option is clearly best. Do NOT ask about trivial internal details or anything determinable from context (Read/InspectCode/Bash) or already specified; if a reasonable default exists, proceed."
    SIGNATURE = "Ask(questions=[{question, choices?, previews?, recommended?}, ...])"
    # fmt: off
    EXAMPLE = (
        'One question, recommending a choice. Example: {"questions":[{"question":"Which approach?","choices":["Refactor","Rewrite"],"previews":["auth/\\n  session.py  (new, +87)\\n  views.py    (-12)","auth.py -> deleted\\nauth/*      all new (+430)"],"recommended":0}]}',
        'Batch related questions. Example: {"questions":[{"question":"Target runtime?","choices":["Node","Deno"]},{"question":"Name the module?"}]}',
    )
    # fmt: on
    MUTATES = False
    STORES_RESULT = True
    question_fn: Callable[[AskSpec, str], str] | None = None

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        question = cls.object_schema({
            "question": {"type": "string", "description": "The question to ask the user"},
            "choices": {"type": "array", "items": {"type": "string"}, "description": "Optional predefined choices the user can pick from"},
            "previews": {"type": "array", "items": {"type": "string"}, "description": "Optional preview per choice, shown as the user navigates. Make it graphic and concrete, not a restatement of the label: a short code/diff snippet, an ASCII layout or tree, or a file/API shape. Multi-line is fine (use \\n); keep under ~10 lines"},
            "recommended": {"type": "integer", "minimum": 0, "description": "Optional 0-based index of the recommended choice; pre-selected and marked"},
        }, ["question"])
        return cls.object_schema({
            "questions": {"type": "array", "minItems": 1, "description": "Questions to ask, one after another", "items": question},
        }, ["questions"])
        # fmt: on

    def call(self) -> str:
        questions = self.single_dict_arg(f"{self.NAME} requires named fields").get("questions")
        if not isinstance(questions, list) or not questions:
            raise ToolError(f"{self.NAME} requires a non-empty 'questions' list")
        # Validate the whole batch up front, so a malformed later question never strands the
        # user after they have already answered earlier ones.
        prepared: list[AskSpec] = []
        for item in questions:
            if not isinstance(item, dict):
                raise ToolError("each question must be an object with a 'question' field")
            question = str(item.get("question", "")).strip()
            if not question:
                raise ToolError("each question requires a 'question' field")
            choices = item.get("choices")
            previews = item.get("previews")
            recommended = item.get("recommended")
            if choices is not None:
                if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
                    raise ToolError(f"{self.NAME} choices must be a list of strings")
                if previews is not None:
                    if not isinstance(previews, list) or not all(isinstance(p, str) for p in previews):
                        raise ToolError(f"{self.NAME} previews must be a list of strings")
                    if len(previews) != len(choices):
                        raise ToolError(f"{self.NAME} previews must match choices length")
            if recommended is not None and (
                isinstance(recommended, bool) or not isinstance(recommended, int) or not choices or not 0 <= recommended < len(choices)
            ):
                raise ToolError(f"{self.NAME} recommended must be a valid 0-based choice index")
            prepared.append(AskSpec(question, choices, previews, recommended))
        total = len(prepared)
        answers: list[tuple[str, str]] = []
        for index, spec in enumerate(prepared):
            position = f"{index + 1}/{total}" if total > 1 else ""
            answers.append((spec.question, self.question_fn(spec, position) if self.question_fn else spec.question))
        if len(answers) == 1:
            return answers[0][1]
        return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in answers)

    def short_args(self) -> list[str]:
        questions = self.args[0].get("questions") if self.args and isinstance(self.args[0], dict) else None
        if not isinstance(questions, list) or not questions:
            return [""]
        first = str((questions[0] or {}).get("question", "") or "").strip() if isinstance(questions[0], dict) else ""
        label = Tool.compact(first, 80)
        return [label + (f" (+{len(questions) - 1} more)" if len(questions) > 1 else "")]


class MCPTool(Tool):
    NAME = "MCP"
    DESCRIPTION = "Call/describe external MCP server tools, and list/read MCP resources"
    SIGNATURE = 'MCP(action="call"|"describe"|"list_resources"|"read_resource", server, tool?, arguments?, uri?)'
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        # fmt: off
        return cls.object_schema({
            "action": {"type": "string", "enum": ["call", "describe", "list_resources", "read_resource"], "description": '"call" invokes a tool; "describe" returns a tool\'s schema; "list_resources" lists a server\'s resources; "read_resource" reads one by uri'},
            "server": {"type": "string", "description": "MCP server name from config"},
            "tool": {"type": "string", "description": "Remote MCP tool name (required for call/describe)"},
            "arguments": {"type": "object", "description": "Arguments for the remote tool (required for call)"},
            "uri": {"type": "string", "description": "Resource URI (required for read_resource), e.g. scheme://path"},
        }, ["action", "server"])
        # fmt: on

    def payload(self) -> Json:
        return self.single_dict_arg("MCP requires named fields")

    ACTIONS: ClassVar[tuple[str, ...]] = ("call", "describe", "list_resources", "read_resource")

    @classmethod
    def resolved_action(cls, payload: Json) -> str:
        """Effective action; defaults to "call" when omitted but the envelope looks like an invocation."""
        action = str(payload.get("action") or "").strip()
        if action:
            return action
        if payload.get("tool") or payload.get("arguments") is not None:
            return "call"
        return action

    def needs_confirmation(self) -> bool:
        payload = self.payload()
        if self.resolved_action(payload) != "call":
            return False
        if self.session.mcp is None:
            return False
        return self.session.mcp.tool_needs_confirmation(str(payload.get("server") or ""), str(payload.get("tool") or ""))

    def short_args(self) -> list[str]:
        payload = self.payload()
        action = self.resolved_action(payload)
        server = str(payload.get("server") or "")
        tool_name = str(payload.get("tool") or "")
        target = (server + " " + str(payload.get("uri") or "")).strip() if action == "read_resource" else (server + "." + tool_name).strip(".")
        parts = [part for part in (action, target) if part]
        arguments = payload.get("arguments")
        if action == "call" and isinstance(arguments, dict) and arguments:
            parts.append(self.format_call_args(arguments))
        return parts

    @staticmethod
    def format_call_args(arguments: Json) -> str:
        rendered = ", ".join(f"{key}={Tool.compact(value, 60)}" for key, value in arguments.items())
        return "(" + rendered + ")"

    def call(self) -> str:
        payload = self.payload()
        action = self.resolved_action(payload)
        server = payload.get("server", "")
        tool_name = payload.get("tool", "")
        arguments = payload.get("arguments", {})
        if action == "call" and not isinstance(arguments, dict):
            raise ToolError("MCP arguments must be an object")

        mcp = self.session.mcp
        if mcp is None:
            raise ToolError("MCP not configured")

        if action == "describe":
            return mcp.describe_tool(server, tool_name)
        if action == "call":
            prefix = mcp.auto_read_prefix(server, tool_name)
            try:
                output = mcp.call_tool(server, tool_name, arguments)
            except ToolError as error:
                raise ToolError(f"{error}\n\n{prefix}" if prefix else str(error)) from error
            return prefix + output if prefix else output
        if action == "list_resources":
            return mcp.list_resources(server)
        if action == "read_resource":
            return mcp.read_resource(server, str(payload.get("uri") or ""))
        raise ToolError(
            f"unknown MCP action {action!r}. Valid actions: {', '.join(self.ACTIONS)}. "
            f'To invoke a remote tool named {action!r}, use action="call", tool={action!r}.'
        )


class SkillTool(Tool):
    NAME = "Skill"
    DESCRIPTION = "Load a skill's full instructions by name (skills are listed in the SKILLS section). Follow the returned steps, running any bundled scripts it references via Bash."
    SIGNATURE = "Skill(name)"
    EXAMPLE = ('Load a skill. Example: {"name":"release-notes"}',)

    # fmt: off
    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema({"name": {"type": "string", "description": "Skill name from the SKILLS section"}}, ["name"])
    # fmt: on

    @classmethod
    def payload_args(cls, payload: Json) -> ToolArgs:
        return [payload.get("name", "")]

    def call(self) -> str:
        (name,) = self.strings(min_count=1, max_count=1)
        library = self.session.skills
        skill = library.get(name) if library else None
        if skill is None:
            available = ", ".join(item.name for item in library.all()) if library else ""
            raise ToolError(f"unknown skill {name!r}" + (f"; available: {available}" if available else "; no skills are installed"))
        assert library is not None
        return f"<Skill name={json.dumps(skill.name)}>\n{library.expand(skill)}\n</Skill>"


# fmt: off
TOOLS: tuple[type[Tool], ...] = (
    MCPTool, SkillTool, ReadTool, InspectCodeTool, SearchTool, EditTool,
    BashTool, JobTool, RecallTool, RecallContextTool, NoteTool, AskTool,
)
# fmt: on
TOOL_REGISTRY: dict[str, type[Tool]] = {tool.NAME: tool for tool in TOOLS}
