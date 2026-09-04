"""Search tools: text search and code symbol inspection."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import re
import shutil
import threading
from typing import ClassVar

import code_symbol_index as csi

from wizolt.base import Json, ToolArgs, ToolError, run_blocking
from wizolt.session import Session
from wizolt.source import INSPECT, SEARCH, SourceBlock, SourceSpan, SourceViewDraft, ToolOutput
from wizolt.tools.base import Tool


class SearchTool(Tool):
    NAME = "Search"
    DESCRIPTION = (
        "Search UTF-8 text files with case-insensitive regex, through ripgrep where it is installed; skips binary/hidden/gitignored "
        "files and returns grouped source views. Prefer it over grep or rg in a shell: matches come back as editable numbered "
        "source (source=view.N) you can Edit directly, where shell output carries none and costs you a re-read. The batch form "
        "runs several unrelated queries -- each with its own pattern, path, and glob -- in one call, so a whole investigation is "
        "one round trip."
    )
    EXAMPLE = (
        'Search source with context. Example: {"pattern":"class .*Tool","path":"src","glob":"*.py","context":2}',
        'A whole investigation in one call, each query labelled by its pattern in the result. Example: {"queries":[{"pattern":"class ChatBubble","glob":"*.tsx"},{"pattern":"ChatBubble","path":"src/views","context":2},{"pattern":"assistantName","glob":"*.ts"}]}',
    )
    MAX_FILE_BYTES = 2_000_000
    MAX_CONTEXT = 30
    SKIP_DIRS: ClassVar[set[str]] = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}

    def __init__(self, session: Session, args: ToolArgs):
        super().__init__(session, args)
        # A batched search over a large tree spends its time inside ripgrep, so the turn's
        # cancellation has to reach that process rather than wait out shell_timeout.
        self._process_lock = threading.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stopped = False

    def request_stop(self) -> None:
        """Kill the ripgrep child, and refuse to start another one; `call()` reaps what it started."""
        with self._process_lock:
            self._stopped = True
            proc = self._process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(OSError):
                proc.kill()

    async def _run_rg(self, cmd: list[str]) -> tuple[int, str, str] | None:
        """Run one ripgrep invocation, tracked so request_stop can kill it. None once stopped.

        `subprocess.run` would give the same output but no handle to kill, so the process is owned
        here: started under the lock unless a stop already arrived, always waited for, and always
        cleared -- a killed child is still reaped before this returns."""
        with self._process_lock:
            if self._stopped:
                return None
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.session.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        with self._process_lock:
            if self._stopped:
                proc.kill()
            self._process = proc
        try:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), self.session.settings.shell_timeout)
            except TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                raise
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
            await proc.communicate()
            raise
        finally:
            with self._process_lock:
                if self._process is proc:
                    self._process = None
        assert proc.returncode is not None
        return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")

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
        return any(not (self.session.in_cwd(request["path"]) or self.session.owns_asset(request["path"])) for request in self.requests())

    async def call(self) -> ToolOutput:
        """Run every query, then hydrate each candidate file once and render source views.

        Ripgrep (or the Python fallback) only discovers candidate paths and lines; the visible
        rows are re-derived by running the requested regex over the file's current content, so a
        file that changed between discovery and capture still yields current results. All visible
        spans for one path across all queries are unioned into one view shared by every query
        result that mentions the path.
        """
        requests = self.requests()
        candidates: dict[str, set[int]] = {}  # path -> query indices that matched it
        for index, request in enumerate(requests):
            for path, _ in await self.find_candidates(request):
                candidates.setdefault(path, set()).add(index)
        counts = [0] * len(requests)
        blocks: dict[str, SourceBlock] = {}
        for path, query_indices in candidates.items():
            lines = self.read_current(path)
            if lines is None:
                continue
            visible: dict[int, bool] = {}
            for index in query_indices:
                request = requests[index]
                regex = self.compile_regex(str(request["pattern"]), multiline=True)
                context = int(request["context"])
                match_indices = self.match_indices(lines, regex)
                counts[index] += len(match_indices)
                for match in match_indices:
                    visible[match] = True
                    for line_index in range(max(0, match - context), min(len(lines), match + context + 1)):
                        visible.setdefault(line_index, False)
            blocks[path] = self.build_block(path, lines, visible)
        parts: list[str | SourceBlock] = []
        for index, request in enumerate(requests):
            parts.append(f"<Search pattern={json.dumps(request['pattern'])} matches={counts[index]}>")
            for path in sorted(path for path, query_indices in candidates.items() if index in query_indices):
                if path in blocks:
                    parts.append(blocks[path])
            parts.append("</Search>")
        return ToolOutput.rendered(parts)

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

    def gitignore_patterns(self, root: str) -> list[str]:
        patterns = []
        paths = [os.path.join(self.session.cwd, ".gitignore")]
        if os.path.isdir(root):
            paths.append(os.path.join(root, ".gitignore"))
        for path in dict.fromkeys(paths):
            try:
                mtime = os.stat(path).st_mtime_ns
                with self.session._gitignore_lock:
                    cached = self.session._gitignore_cache.get(path)
                if cached is not None and cached[0] == mtime:
                    patterns.extend(cached[1])
                    continue
                with open(path, encoding="utf-8") as file:
                    pats = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")]
                with self.session._gitignore_lock:
                    self.session._gitignore_cache[path] = (mtime, pats)
                patterns.extend(pats)
            except OSError:
                with self.session._gitignore_lock:
                    self.session._gitignore_cache.pop(path, None)
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

    async def find_candidates(self, request: Json) -> list[tuple[str, int]]:
        """Discover (path, 0-based match line) pairs for one request.

        Prefers ripgrep; falls back to a Python scan. Both backends are candidate finders only:
        matches are re-derived from the current file content during hydration.
        """
        patterns = self.gitignore_patterns(str(request["path"]))
        if self.default_ignored(str(request["path"]), patterns):
            return []
        if "\n" not in str(request["pattern"]):
            rows = await self.rg_candidates(request)
            if rows is not None:
                return rows
        return await self._python_candidates(request)

    async def _python_candidates(self, request: Json) -> list[tuple[str, int]]:
        """Run the filesystem fallback off-loop and do not abandon it on cancellation."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self.python_candidates, request)
        cancelled: asyncio.CancelledError | None = None
        while not future.done():
            try:
                await asyncio.wait({future})
            except asyncio.CancelledError as error:
                cancelled = cancelled or error
                self.request_stop()
        try:
            result = future.result()
        except BaseException:
            if cancelled is None:
                raise
            raise cancelled from None
        if cancelled is not None:
            raise cancelled
        return result

    async def rg_candidates(self, request: Json) -> list[tuple[str, int]] | None:
        rg = shutil.which("rg")
        if not rg:
            return None
        cmd = [rg, "--json", "--line-number", "--with-filename", "--color=never", "--ignore-case", "--max-filesize", "2M"]
        if request["glob"]:
            cmd.extend(["--glob", str(request["glob"])])
        cmd.extend([str(request["pattern"]), str(request["path"])])
        proc = await self._run_rg(cmd)
        if proc is not None and proc[0] == 2:
            proc = await self._run_rg([*cmd[:1], "--pcre2", *cmd[1:]])
        if proc is None or proc[0] not in (0, 1):
            return None
        rows: list[tuple[str, int]] = []
        for line in proc[1].splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            path = data.get("path", {}).get("text")
            number = data.get("line_number")
            if not path or not isinstance(number, int):
                continue
            # ripgrep counts from 1; hydration indexes from 0.
            rows.append((path, number - 1))
        return rows

    def files(self, root: str, glob_pattern: str) -> list[str]:
        gitignore = self.gitignore_patterns(root)
        if self.default_ignored(root, gitignore):
            return []
        if os.path.isfile(root):
            return [root]
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            if self._stopped:
                break
            dirnames[:] = [
                name
                for name in dirnames
                if name not in self.SKIP_DIRS and not name.startswith(".") and not self.ignored(os.path.join(dirpath, name), gitignore)
            ]
            for filename in filenames:
                if self._stopped:
                    break
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

    def python_candidates(self, request: Json) -> list[tuple[str, int]]:
        regex = self.compile_regex(str(request["pattern"]), multiline=True)
        rows: list[tuple[str, int]] = []
        for path in self.files(str(request["path"]), str(request["glob"])):
            if self._stopped:
                break
            lines = self.read_current(path)
            if lines is None:
                continue
            for match in self.match_indices(lines, regex):
                rows.append((path, match))
        return rows

    def read_current(self, path: str) -> list[str] | None:
        try:
            if os.path.getsize(path) > self.MAX_FILE_BYTES:
                return None
            with open(path, encoding="utf-8") as file:
                return file.readlines()
        except (OSError, UnicodeDecodeError):
            return None

    def match_indices(self, lines: list[str], regex: re.Pattern[str]) -> list[int]:
        if "\n" in regex.pattern:
            content = "".join(lines)
            return [content.count("\n", 0, match.start()) for match in regex.finditer(content)]
        return [index for index, line in enumerate(lines) if regex.search(line)]

    def build_block(self, path: str, lines: list[str], visible: dict[int, bool]) -> SourceBlock:
        """One numbered source view for a path; `>` marks a match line, a space marks context."""
        ranges = [(line_index + 1, line_index + 1) for line_index in sorted(visible)]
        spans = SourceSpan.build(lines, ranges)
        markers = []
        for span in spans:
            for offset in range(len(span.lines)):
                line_index = span.start - 1 + offset
                markers.append("> " if visible.get(line_index) else "  ")
        draft = SourceViewDraft(path, self.session.relpath(path), len(lines), spans, SEARCH)
        return SourceBlock(draft, tuple(markers))


class CodeIndex:
    """Keep the symbol index useful without ever making the user wait for it.

    An optional accelerator: absent, stale, or broken, symbol lookups fall back to ordinary search
    rather than failing, which is why every integration failure here becomes a status instead of an
    exception. That status is published for the status bar to display.

    Freshness is opportunistic. Checking the working tree hashes files and is slow on a large
    repository, so it runs on a background thread after a turn, never in the path of an answer, and a
    flag keeps scans from stacking up. A few changed files are re-indexed automatically; beyond that
    the index is marked stale for an explicit sync, because a large rebuild is the user's time to
    spend.
    """

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
        """Read and publish index status on the calling thread."""

        return self._publish_status(*self._read_status(check=check, max_pending_files=max_pending_files))

    def _publish_status(self, status: str, message: str, pending: object) -> tuple[str, str]:
        """Apply one status read to Session on its owning thread."""

        preserves_stale = status == "ready" and pending == "unknown" and self.session.state.code_index_status == "stale"
        if not self.session.state.code_index_refreshing and not preserves_stale:
            self.set_status(status, message)
        return status, message

    def _read_status(self, *, check: bool = False, max_pending_files: int = 20) -> tuple[str, str, object]:
        """Blocking third-party status read. It returns data and never mutates Session."""

        try:
            data = csi.status(self.session.cwd, check=check, max_pending_files=max_pending_files)
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            return "error", str(error), None
        status = str(getattr(data, "status", "") or "error")
        message = str(getattr(data, "message", None) or getattr(data, "reason", None) or "")
        pending = getattr(data, "pending_changes", None)
        files = getattr(data, "pending_files", ()) or ()
        if pending and pending != "unknown":
            sample = ", ".join(str(path) for path in (files or [])[:3])
            message = (message + "; " if message else "") + "pending " + str(pending) + ((" (" + sample + ")") if sample else "")
        return status, message, pending

    async def sync(self, *, force: bool = False) -> str:
        """Index or rebuild the whole tree, keeping the prompt live while it runs.

        The third-party clean/index pair is a long synchronous walk of the workspace; it goes to a
        worker whole, and the status the reader sees is written back here on the loop."""

        if self.session.state.code_index_refreshing:
            return "code_index: syncing"
        self.notice("syncing", refreshing=True)
        try:
            await run_blocking(lambda: self._sync_worker(force))
        except asyncio.CancelledError:
            await self._settle_cancelled_operation()
            raise
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            return "code_index: error\n" + self.fail(error)
        self.finish()
        status, message = self._publish_status(*await run_blocking(lambda: self._read_status(check=True)))
        index_path = os.path.join(self.session.cwd, ".code-symbol-index", "index.sqlite")
        lines = ["code_index: " + ("rebuilt" if force else "synced"), "status: " + status, "path: " + index_path]
        if message:
            lines.append("note: " + message)
        return "\n".join(lines)

    def _sync_worker(self, force: bool) -> None:
        """The blocking third-party half of `sync`. Runs on a worker; touches no session state."""
        if force:
            csi.clean(self.session.cwd)
        csi.index(self.session.cwd)

    async def update(self, paths: list[str]) -> str:
        """Update edited paths without blocking the loop or publishing state from a worker."""

        paths = self.update_paths(paths)
        if not paths or self.session.state.code_index_refreshing:
            return ""
        status, _ = self._publish_status(*await run_blocking(self._read_status))
        if status not in {"ready", "stale"}:
            return ""
        return await self._update(paths)

    async def update_pending(self) -> str:
        """Check the working tree for drift and apply a small update, off the answer path.

        The `check=True` scan walks and hashes the tree, so it and the update it may trigger go to
        a worker; the coalescing flag and every status field are set here, on the loop."""

        if self.session.state.code_index_checking or self.session.state.code_index_refreshing:
            return ""
        # The flag is the coalescing gate: several triggers can arrive per turn (the turn end,
        # /status, a queued command) and only the first of them should walk the tree.
        self.session.state.code_index_checking = True
        try:
            try:
                data = await run_blocking(lambda: csi.status(self.session.cwd, check=True, max_pending_files=self.AUTO_UPDATE_LIMIT + 1))
            except Exception:  # noqa: BLE001 - background index freshness checks are best-effort.
                return ""
            self.set_status(str(getattr(data, "status", "") or "error"), str(getattr(data, "message", None) or getattr(data, "reason", None) or ""))
            if getattr(data, "status", "") != "stale":
                return ""
            pending = getattr(data, "pending_changes", None)
            files = [str(path) for path in getattr(data, "pending_files", ()) or () if path]
            if not files or len(files) > self.AUTO_UPDATE_LIMIT or (isinstance(pending, int) and pending > self.AUTO_UPDATE_LIMIT):
                return ""
            return await self._update(self.update_paths([self.session.resolve_path(path) for path in files]))
        finally:
            self.session.state.code_index_checking = False

    async def _update(self, paths: list[str]) -> str:
        """`update` for a caller on the loop: the third-party call on a worker, the status here."""
        if not paths or self.session.state.code_index_refreshing:
            return ""
        self.notice("updating", refreshing=True)
        try:
            await run_blocking(lambda: csi.update(paths, root=self.session.cwd))
        except asyncio.CancelledError:
            await self._settle_cancelled_operation()
            raise
        except Exception as error:  # noqa: BLE001 - isolate failures from the optional code-index integration.
            return self.fail(error)
        self.finish()
        return "updated " + str(len(paths)) + " file(s)"

    async def _settle_cancelled_operation(self) -> None:
        """Publish the real index state after a cancelled await has quiesced its worker."""

        self.notice("")
        self._publish_status(*await run_blocking(lambda: self._read_status(check=True)))

    def update_paths(self, paths: list[str]) -> list[str]:
        paths = [self.session.resolve_path(path) for path in paths]
        return list(dict.fromkeys(path for path in paths if self.session.in_cwd(path) and os.path.isfile(path)))


class InspectCodeTool(Tool):
    _WHITESPACE_RE: ClassVar[re.Pattern] = re.compile(r"\s")
    NAME = "InspectCode"
    MAX_LIMIT: ClassVar[int] = 80
    MAX_OUTLINE_LIMIT: ClassVar[int] = 1000
    MAX_DEPTH: ClassVar[int] = 5
    MODES: ClassVar[tuple[str, ...]] = ("find", "inspect", "outline", "refs", "impls", "callers", "callees")
    SYMBOL_MODES: ClassVar[frozenset[str]] = frozenset({"find", "inspect", "refs", "impls", "callers", "callees"})
    RESOLVE_MODES: ClassVar[frozenset[str]] = frozenset({"inspect", "refs", "impls", "callers", "callees"})
    CHAIN_MODES: ClassVar[frozenset[str]] = frozenset({"callers", "callees"})
    OPTION_KEYS: ClassVar[tuple[str, ...]] = ("limit", "kind", "path", "symbol", "exact_only", "depth", "offset", "all_kinds", "ref_kind", "loose")
    DESCRIPTION = (
        "Use the code index: find returns symbols; inspect returns members/references plus a current source block; outline returns a file symbol tree; "
        "refs lists classified references; impls lists implementors; callers/callees walk the call chain. Source shown is hydrated from the current file, "
        "so an index with stale metadata falls back to a stale note instead of fake source."
    )
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

    def call(self) -> ToolOutput:
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
        if mode in self.SYMBOL_MODES and self._WHITESPACE_RE.search(target):
            # Models often repeat the kind inside the target, e.g. target "class Config" with
            # kind "class". When the first word duplicates a declared kind, drop it — that is the one
            # case we can strip deterministically (no guessing at per-language keywords).
            kinds = {token.strip().lower() for token in str(options.get("kind") or "").split(",") if token.strip()}
            first, _, rest = target.partition(" ")
            if kinds and first.lower() in kinds and rest.strip():
                target = rest.strip()
            if self._WHITESPACE_RE.search(target):
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
            text = self.process_result("InspectCodeToolResult", 1, "", str(error))
            return ToolOutput.of(text)
        hydrated, blocks = self.hydrate(output)
        return ToolOutput.rendered([hydrated, *blocks])

    def hydrate(self, text: str) -> tuple[str, list[SourceBlock]]:
        """Hydrate the indexed definition from the current file as an editable source block.

        The index may identify a stale path or range; it never mints evidence itself. The
        definition's current lines are emitted only when the file exists and the indexed
        signature still matches; otherwise the block is omitted with an explicit stale note.
        """
        match = re.search(r"^  file: (.+)$\n^  range: (\d+):(\d+)$\n^  signature: (.*)$", text, re.MULTILINE)
        if not match:
            return text, []
        path, start, end, signature = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
        resolved = self.session.resolve_path(path)
        try:
            with open(resolved, encoding="utf-8") as file:
                lines = file.readlines()
        except OSError:
            note = f"\n\n<stale-index> definition file {path} no longer exists; Read it again for current source</stale-index>"
            return text + note, []
        if start < 1 or start > len(lines) or lines[start - 1].rstrip("\n") != signature:
            note = f"\n\n<stale-index> index is stale for {path}; Read it again for current source</stale-index>"
            return text + note, []
        spans = SourceSpan.build(lines, [(start, min(end, len(lines)))])
        if not spans:
            return text, []
        return text, [SourceBlock.plain(SourceViewDraft(resolved, self.session.relpath(resolved), len(lines), spans, INSPECT))]

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
            return csi.inspect(target, limit=limit or csi.DEFAULT_PAGE_LIMIT, **common)
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
