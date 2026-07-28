"""Search tools: text search and code symbol inspection."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import threading
from typing import ClassVar

import code_symbol_index as csi

from minacode.base import Json, ToolArgs, ToolError
from minacode.session import Session
from minacode.tools.base import Tool
from minacode.tools.files import ReadTool


class SearchTool(Tool):
    NAME = "Search"
    DESCRIPTION = "Search UTF-8 text files with case-insensitive regex; skips binary/hidden/gitignored files and returns path anchor=line:hash matches."
    EXAMPLE = (
        'Search source with context. Example: {"pattern":"class .*Tool","path":"src","glob":"*.py","context":2}',
        'Search multiple queries. Example: {"queries":[{"pattern":"TODO","glob":"*.py"},{"pattern":"FIXME","path":"tests","glob":"*.py"}]}',
    )
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
    EXAMPLE = (
        'Find symbols; kind can be class|function|method|variable|constant|enum|struct|interface|module|type|trait|field|property|impl|namespace|dict_key, comma-ok. Example: {"mode":"find","target":"Tool","kind":"class,function","limit":20}',
        'Inspect one symbol; path narrows candidates. Example: {"mode":"inspect","target":"Tool","path":"src/app.py"}',
        'Outline one file; symbol narrows subtree. Example: {"mode":"outline","target":"src/app.py","symbol":"App","limit":300}',
    )

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
