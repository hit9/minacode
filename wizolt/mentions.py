"""Mention grammar plus @file candidate discovery and turn-context expansion."""

from __future__ import annotations

import heapq
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from wizolt.session import Session


MentionKind = Literal["bare", "file", "mcp", "skill"]
_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_IDENTIFIER = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_BARE_FILE = re.compile(r"^[A-Za-z0-9_./:+@%=-]+$")


@dataclass(frozen=True)
class MentionSpan:
    """One syntactically recognized mention and its exact source span."""

    start: int
    end: int
    kind: MentionKind
    payload: str
    complete: bool = True


def encode_file_mention(path: str) -> str:
    """Return the canonical, round-trippable @file form for a display path."""
    payload = path if _BARE_FILE.fullmatch(path) else json.dumps(path, ensure_ascii=False)
    return "@file:" + payload


def scan_mentions(text: str) -> list[MentionSpan]:
    """Scan file/MCP/skill mentions without allowing namespace or email collisions."""
    spans: list[MentionSpan] = []
    index = 0
    while index < len(text):
        marker = text[index]
        if marker not in "@$" or (index and text[index - 1] in _WORD):
            index += 1
            continue
        span = _scan_at(text, index) if marker == "@" else _scan_dollar(text, index)
        if span is None:
            index += 1
            continue
        spans.append(span)
        index = max(index + 1, span.end)
    return spans


def active_mention(text_before_cursor: str) -> MentionSpan | None:
    """Return the editable mention ending at the cursor, if any."""
    spans = scan_mentions(text_before_cursor)
    return spans[-1] if spans and spans[-1].end == len(text_before_cursor) else None


def _scan_at(text: str, start: int) -> MentionSpan | None:
    payload_start = start + 1
    for namespace in ("file", "mcp", "skill"):
        prefix = namespace + ":"
        if text.startswith(prefix, payload_start):
            value_start = payload_start + len(prefix)
            if namespace == "file":
                return _scan_file(text, start, value_start)
            end = _identifier_end(text, value_start, dot=namespace == "mcp")
            return MentionSpan(start, end, namespace, text[value_start:end], end > value_start)
    end = _identifier_end(text, payload_start, dot=True)
    if end == payload_start:
        return MentionSpan(start, end, "bare", "", False)
    return MentionSpan(start, end, "bare", text[payload_start:end])


def _scan_dollar(text: str, start: int) -> MentionSpan | None:
    payload_start = start + 1
    end = _identifier_end(text, payload_start)
    if end == payload_start:
        return MentionSpan(start, end, "skill", "", False)
    return MentionSpan(start, end, "skill", text[payload_start:end])


def _scan_file(text: str, start: int, payload_start: int) -> MentionSpan:
    if payload_start >= len(text):
        return MentionSpan(start, payload_start, "file", "", False)
    if text[payload_start] != '"':
        end = payload_start
        while end < len(text) and not text[end].isspace():
            end += 1
        return MentionSpan(start, end, "file", text[payload_start:end], end > payload_start)
    try:
        payload, consumed = json.JSONDecoder().raw_decode(text[payload_start:])
    except json.JSONDecodeError:
        return MentionSpan(start, len(text), "file", text[payload_start + 1 :], False)
    if not isinstance(payload, str):
        return MentionSpan(start, payload_start + consumed, "file", "", False)
    return MentionSpan(start, payload_start + consumed, "file", payload)


def _identifier_end(text: str, start: int, *, dot: bool = False) -> int:
    allowed = _IDENTIFIER | ({"."} if dot else set())
    end = start
    while end < len(text) and text[end] in allowed:
        end += 1
    return end


def _has_git_component(path: str) -> bool:
    return ".git" in path.replace("\\", "/").split("/")


class FileMentions:
    """Discover selectable paths and expand explicit file mentions for a turn."""

    CACHE_TTL = 5.0
    MAX_INLINE_LINES = 400
    MAX_INLINE_BYTES = 64 * 1024
    MAX_INLINE_FILES = 10
    MAX_REFERENCES = 50
    GIT_TIMEOUT = 10

    def __init__(self, session: Session) -> None:
        self.session = session
        self._paths_cache: tuple[float, tuple[tuple[str, str], ...]] | None = None
        self._cache_lock = threading.Lock()
        self._refreshing = False
        self._refresh_callbacks: list[Callable[[], None]] = []
        self._generation = 0
        self._last_match: tuple[int, str, tuple[str, ...]] | None = None
        self._match_request: tuple[int, str, Callable[[], None]] | None = None
        self._match_worker_running = False
        self.picker = FzfPicker(self)

    def paths(self) -> tuple[tuple[str, str], ...]:
        """Return a fresh immutable (lowercase, display) candidate snapshot."""
        now = time.monotonic()
        with self._cache_lock:
            cached = self._paths_cache
            if cached is not None and now - cached[0] < self.CACHE_TTL:
                return cached[1]
        pairs = self._collect()
        with self._cache_lock:
            self._paths_cache = (time.monotonic(), pairs)
            self._generation += 1
            self._last_match = None
        return pairs

    def cached_paths(self) -> tuple[tuple[str, str], ...] | None:
        """Return the immutable snapshot without starting filesystem work."""
        with self._cache_lock:
            return self._paths_cache[1] if self._paths_cache is not None else None

    def cached_matches(self, query: str) -> tuple[str, ...]:
        with self._cache_lock:
            match = self._last_match
            return match[2] if match is not None and match[:2] == (self._generation, query) else ()

    def schedule_completion(self, query: str, callback: Callable[[], None]) -> None:
        """Compute the latest literal fallback query on one worker, then notify the TUI."""

        def candidates_ready() -> None:
            with self._cache_lock:
                self._match_request = (self._generation, query, callback)
                if self._match_worker_running:
                    return
                self._match_worker_running = True

            def match() -> None:
                while True:
                    with self._cache_lock:
                        request, self._match_request = self._match_request, None
                        paths = self._paths_cache[1] if self._paths_cache is not None else ()
                    if request is None:
                        with self._cache_lock:
                            if self._match_request is None:
                                self._match_worker_running = False
                                return
                        continue
                    generation, requested, ready = request
                    matches = self._literal_matches(paths, requested)
                    with self._cache_lock:
                        if generation == self._generation:
                            self._last_match = (generation, requested, matches)
                    with suppress(Exception):
                        ready()

            threading.Thread(target=match, name="file-matches", daemon=True).start()

        self.schedule_refresh(candidates_ready)

    @staticmethod
    def _literal_matches(paths: tuple[tuple[str, str], ...], query: str) -> tuple[str, ...]:
        lowered = query.lower()

        def ranked():
            for lower, path in paths:
                if lowered not in lower:
                    continue
                basename = lower.rsplit("/", 1)[-1]
                score = 0 if basename.startswith(lowered) else 1 if lowered in basename else 2
                yield score, len(lower), lower, path

        return tuple(path for _, _, _, path in heapq.nsmallest(50, ranked()))

    def schedule_refresh(self, callback: Callable[[], None] | None = None) -> None:
        """Refresh once on a worker and notify all waiters; never block the caller."""
        notify_now = False
        start = False
        with self._cache_lock:
            cached = self._paths_cache
            if cached is not None and time.monotonic() - cached[0] < self.CACHE_TTL:
                notify_now = callback is not None
            else:
                if callback is not None:
                    self._refresh_callbacks.append(callback)
                if not self._refreshing:
                    self._refreshing = True
                    start = True
        if notify_now:
            assert callback is not None
            callback()
            return
        if not start:
            return

        def refresh() -> None:
            try:
                pairs = self._collect()
                with self._cache_lock:
                    self._paths_cache = (time.monotonic(), pairs)
                    self._generation += 1
                    self._last_match = None
            except Exception:  # noqa: BLE001, S110 - optional completion cannot fail the TUI.
                pass
            finally:
                with self._cache_lock:
                    self._refreshing = False
                    callbacks, self._refresh_callbacks = self._refresh_callbacks, []
                for ready in callbacks:
                    with suppress(Exception):
                        ready()

        threading.Thread(target=refresh, name="file-candidates", daemon=True).start()

    def _collect(self) -> tuple[tuple[str, str], ...]:
        rels = self._git_paths()
        if rels is None:
            rels = self._rg_paths()
        if rels is None:
            rels = self._walk_paths()
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for rel in rels:
            rel = rel.replace(os.sep, "/").removeprefix("./")
            if not rel or rel in seen or _has_git_component(rel):
                continue
            path = os.path.join(self.session.cwd, *rel.split("/"))
            try:
                if not os.path.isfile(path):
                    continue
            except OSError:
                continue
            seen.add(rel)
            pairs.append((rel.lower(), rel))
        pairs.sort()
        return tuple(pairs)

    def _git_paths(self) -> list[str] | None:
        """Git-authoritative candidates, including the tracked-but-ignored subtraction."""
        # These queries only share Git's read-only index. Running them together removes one full
        # process latency from cold picker preparation in large parent worktrees.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="file-git") as pool:
            included_future = pool.submit(self._git_ls_files, ["--cached", "--others", "--exclude-standard"])
            ignored_future = pool.submit(self._git_ls_files, ["--cached", "--ignored", "--exclude-standard"])
            included = included_future.result()
            ignored = ignored_future.result()
        if included is None or ignored is None:
            return None
        excluded = set(ignored)
        return [path for path in included if path not in excluded]

    def _git_ls_files(self, flags: list[str]) -> list[str] | None:
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z", *flags, "--", "."],
                cwd=self.session.cwd,
                capture_output=True,
                timeout=self.GIT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]

    def _rg_paths(self) -> list[str] | None:
        executable = shutil.which("rg")
        if executable is None:
            return None
        try:
            result = subprocess.run(
                [executable, "--files", "--hidden", "--no-require-git", "--glob", "!**/.git/**", "--null"],
                cwd=self.session.cwd,
                capture_output=True,
                timeout=self.GIT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode not in (0, 1):
            return None
        return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]

    def _walk_paths(self) -> list[str]:
        """Correct non-Git fallback using nested GitIgnoreSpec scopes."""
        from pathspec import GitIgnoreSpec

        root = self.session.cwd
        found: list[str] = []

        def walk(directory: str, scopes: tuple[tuple[str, GitIgnoreSpec], ...]) -> None:
            rel_dir = os.path.relpath(directory, root)
            rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
            ignore_file = os.path.join(directory, ".gitignore")
            try:
                with open(ignore_file, encoding="utf-8") as handle:
                    local = GitIgnoreSpec.from_lines(handle)
            except OSError:
                local = None
            if local is not None:
                scopes = (*scopes, (rel_dir, local))
            try:
                entries = list(os.scandir(directory))
            except OSError:
                return
            for entry in entries:
                rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
                if entry.name == ".git" or _ignored_by_scopes(rel, entry.is_dir(follow_symlinks=False), scopes):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        walk(entry.path, scopes)
                    elif entry.is_file(follow_symlinks=True):
                        found.append(rel)
                except OSError:
                    continue

        walk(root, ())
        return found

    def resolve_mentions(self, text: str) -> str:
        """Build one bounded FILE MENTIONS context block from canonical scanner spans."""
        entries: list[str] = []
        seen: set[str] = set()
        inline_count = 0
        omitted = 0
        for span in scan_mentions(text):
            if span.kind != "file" or not span.complete or not span.payload:
                continue
            raw = span.payload
            path = self.session.resolve_path(raw)
            identity = os.path.normcase(os.path.realpath(path))
            if identity in seen:
                continue
            seen.add(identity)
            if len(entries) >= self.MAX_REFERENCES:
                omitted += 1
                continue
            block, inlined = self._file_block(raw, path, allow_inline=inline_count < self.MAX_INLINE_FILES)
            entries.append(block)
            inline_count += int(inlined)
        if omitted:
            entries.append(f"[{omitted} additional file mention(s) omitted at the {self.MAX_REFERENCES}-reference cap]")
        if not entries:
            return ""
        header = [
            "--- FILE MENTIONS ---",
            "The user explicitly referenced these files. Treat them as the subject of the request.",
            "",
        ]
        return "\n".join([*header, *entries]).strip()

    def _file_block(self, raw: str, path: str, *, allow_inline: bool) -> tuple[str, bool]:
        try:
            if os.path.isdir(path):
                return f"[{raw}] directory; Read it to see its contents", False
            if not os.path.isfile(path):
                return f"[{raw}] not found", False
            if not self.session.in_cwd(path):
                return f"[{raw}] outside the workspace; Read it", False
            size = os.path.getsize(path)
            if not allow_inline:
                return f"[{raw}] not inlined - the {self.MAX_INLINE_FILES}-file inline cap is reached; Read it if relevant", False
            if size > self.MAX_INLINE_BYTES:
                lines = self._line_count(path)
                return f"[{raw}] {lines} lines, {self._size_label(size)} - too large to inline; Read the part you need", False
            with open(path, "rb") as handle:
                data = handle.read(self.MAX_INLINE_BYTES + 1)
            if self._is_binary(data):
                return f"[{raw}] binary file, {self._size_label(size)}; Read it with an appropriate tool", False
            content = data.decode("utf-8")
            lines = self._line_count_bytes(data)
            if lines > self.MAX_INLINE_LINES:
                return f"[{raw}] {lines} lines, {self._size_label(size)} - too large to inline; Read the part you need", False
            return f"[{raw}] {lines} lines\n{content}", True
        except (OSError, UnicodeError):
            return f"[{raw}] unreadable or no longer exists", False

    @staticmethod
    def _is_binary(data: bytes) -> bool:
        if b"\0" in data:
            return True
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    @staticmethod
    def _line_count_bytes(data: bytes) -> int:
        return data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))

    @classmethod
    def _line_count(cls, path: str) -> int:
        count = 0
        total = 0
        last = b""
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                total += len(chunk)
                count += chunk.count(b"\n")
                last = chunk[-1:]
        return count + int(total > 0 and last != b"\n")

    @staticmethod
    def _size_label(size: int) -> str:
        return f"{size // 1024} KB" if size >= 1024 else f"{size} B"


@dataclass(frozen=True)
class FilePick:
    selection: str | None = None
    unavailable: bool = False


class FzfPicker:
    """Isolated one-process interactive fzf adapter; matching stays inside fzf."""

    def __init__(self, mentions: FileMentions, executable: str = "fzf") -> None:
        self.mentions = mentions
        self.name = executable
        self._executable: str | None = None
        self._resolved = False
        self._failed = False

    def available(self) -> bool:
        if self._failed:
            return False
        if not self._resolved:
            self._executable = shutil.which(self.name)
            self._resolved = True
        return self._executable is not None

    def pick(self, query: str) -> FilePick:
        if not self.available():
            return FilePick(unavailable=True)
        environment = os.environ.copy()
        for name in ("FZF_DEFAULT_COMMAND", "FZF_DEFAULT_OPTS", "FZF_DEFAULT_OPTS_FILE"):
            environment.pop(name, None)
        argv = [
            self._executable or self.name,
            "--read0",
            "--print0",
            "--scheme=path",
            "--no-multi",
            "--no-multi-line",
            "--layout=reverse",
            "--height=~60%",
            "--border",
            "--prompt=files> ",
            "--header=Ctrl-N/P or ↑/↓ move · Enter select · Esc close",
            "--bind=ctrl-n:down,ctrl-p:up",
            "--query",
            query,
        ]
        try:
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment)
        except OSError:
            self._failed = True
            return FilePick(unavailable=True)

        candidates: list[tuple[tuple[str, str], ...]] = []
        feed_failed = threading.Event()

        def feed() -> None:
            assert process.stdin is not None
            try:
                snapshot = self.mentions.cached_paths()
                if snapshot is None:
                    ready = threading.Event()
                    self.mentions.schedule_refresh(ready.set)
                    while not ready.wait(0.05):
                        if process.poll() is not None:
                            return
                    snapshot = self.mentions.cached_paths()
                    if snapshot is None:
                        feed_failed.set()
                        return
                else:
                    # A stale snapshot is still safe to display: the selected path is revalidated
                    # below. Refresh it for the next picker without delaying this one's first row.
                    self.mentions.schedule_refresh()
                candidates.append(snapshot)
                for _, path in snapshot:
                    process.stdin.write(os.fsencode(path) + b"\0")
            except (BrokenPipeError, OSError):
                pass
            except Exception:  # noqa: BLE001 - optional picker must fail closed
                feed_failed.set()
            finally:
                with suppress(OSError):
                    process.stdin.close()

        feeder = threading.Thread(target=feed, name="fzf-candidates", daemon=True)
        feeder.start()
        assert process.stdout is not None
        try:
            output = process.stdout.read(1 << 20)
            returncode = process.wait()
        except OSError:
            process.kill()
            with suppress(OSError):
                process.wait()
            self._failed = True
            return FilePick(unavailable=True)
        finally:
            feeder.join(timeout=1)
        if returncode == 1 and feed_failed.is_set():
            return FilePick(unavailable=True)
        if returncode in (1, 130):
            return FilePick()
        if returncode != 0:
            self._failed = True
            return FilePick(unavailable=True)
        selected = os.fsdecode(output.split(b"\0", 1)[0]) if output else ""
        snapshot = candidates[0] if candidates else ()
        if not selected or not any(path == selected for _, path in snapshot) or _has_git_component(selected):
            return FilePick()
        path = self.mentions.session.resolve_path(selected)
        if not self.mentions.session.in_cwd(path) or not os.path.isfile(path):
            return FilePick()
        return FilePick(selected)


def _ignored_by_scopes(rel: str, is_dir: bool, scopes) -> bool:
    ignored = False
    for base, spec in scopes:
        if base and not (rel == base or rel.startswith(base + "/")):
            continue
        local = rel[len(base) + 1 :] if base else rel
        result = spec.check_file(local + ("/" if is_dir else ""))
        if result.include is not None:
            ignored = result.include
    return ignored
