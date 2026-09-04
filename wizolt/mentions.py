"""Mention grammar plus @file candidate discovery and turn-context expansion."""

from __future__ import annotations

import asyncio
import heapq
import json
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from wizolt.base import run_blocking

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
        self._generation = 0
        self._last_match: tuple[int, str, tuple[str, ...]] | None = None
        # Plain request state, never a task/future: FileMentions outlives event loops. One admitted
        # coroutine drains the latest request, so rapid keystrokes replace queued ranking work
        # instead of filling the executor with obsolete queries.
        self._match_request = 0
        self._pending_match: tuple[int, str, Callable[[], None] | None] | None = None
        self._match_worker_running = False
        # Installed by CommandLoop while its background owner is open, and cleared when it closes.
        # A callable, never a task: the coalescing task is loop-bound and this object outlives
        # loops, so a future kept here would be a handle onto a loop the next run cannot await.
        self.refresh_owner: Callable[[], Awaitable[tuple[tuple[str, str], ...]] | None] | None = None
        self.picker = FzfPicker(self)

    def cached_paths(self) -> tuple[tuple[str, str], ...] | None:
        """Return the immutable snapshot without starting filesystem work."""
        return self._paths_cache[1] if self._paths_cache is not None else None

    def cached_matches(self, query: str) -> tuple[str, ...]:
        match = self._last_match
        return match[2] if match is not None and match[:2] == (self._generation, query) else ()

    def schedule_refresh(self) -> Awaitable[tuple[tuple[str, str], ...]] | None:
        """Ask the owner for its one coalesced scan, without waiting for it.

        Returns the shared task so a caller that does want the result can await it, and None when
        no owner is admitting work -- a session that is shutting down, or one with no frontend."""

        owner = self.refresh_owner
        return owner() if owner is not None else None

    async def candidates(self) -> tuple[tuple[str, str], ...]:
        """The candidate snapshot for a picker or a completion.

        A stale snapshot is handed back immediately and refreshed behind the caller: the picker
        revalidates whatever is selected against the exact snapshot it displayed, so a five-second
        old list is safe, while making every invocation wait for a full scan is not. Only a cold
        cache waits."""

        cached = self._paths_cache
        if cached is not None:
            if time.monotonic() - cached[0] >= self.CACHE_TTL:
                self.schedule_refresh()
            return cached[1]
        scan = self.schedule_refresh()
        # A caller waiting for the owner's shared scan does not own it. In particular, Escape from
        # a cold fzf picker cancels that picker, not the startup scan another waiter may still use.
        return await asyncio.shield(scan) if scan is not None else await self.refresh()

    async def complete(self, query: str, ready: Callable[[], None] | None = None) -> None:
        """Rank the literal fallback candidates for one query and publish the newest result.

        Ranking is bounded but not free over a large worktree, so it happens on a worker; the
        cache and the ready callback are touched here, on the loop. A query superseded while it
        ranked publishes nothing: the menu must show what the user is typing now, not what they
        had typed when the scan started."""

        self._match_request += 1
        self._pending_match = (self._match_request, query, ready)
        if self._match_worker_running:
            return
        self._match_worker_running = True
        try:
            while self._pending_match is not None:
                request, requested, callback = self._pending_match
                self._pending_match = None
                paths = await self.candidates()
                generation = self._generation
                matches = await run_blocking(lambda paths=paths, requested=requested: self._literal_matches(paths, requested))
                # Anything queued while this query was ranking supersedes it. Skip both its cache
                # publication and callback, then compute only the newest request next.
                if self._pending_match is not None or request != self._match_request:
                    continue
                if generation == self._generation:
                    self._last_match = (generation, requested, matches)
                if callback is not None:
                    with suppress(Exception):
                        callback()
        finally:
            self._pending_match = None
            self._match_worker_running = False

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

    async def refresh(self) -> tuple[tuple[str, str], ...]:
        """Rediscover the candidate paths and publish the new immutable snapshot.

        The two subprocess sources are native asyncio; only the Python walk and the normalize/stat
        pass are blocking, and they are one worker call together -- a 50k-path result is 50k stats,
        which is not something to hand back to the loop one at a time.

        A failure keeps the previous snapshot. File completion is an optional convenience and must
        never be the reason a prompt breaks."""

        try:
            rels = await self._git_paths()
            if rels is None:
                rels = await self._rg_paths()
            pairs = await run_blocking(lambda: self._collect(rels))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - optional completion cannot fail the TUI.
            return self.cached_paths() or ()
        self._paths_cache = (time.monotonic(), pairs)
        self._generation += 1
        self._last_match = None
        return pairs

    def _collect(self, rels: list[str] | None) -> tuple[tuple[str, str], ...]:
        """Normalize, deduplicate, stat, and sort one discovery result. Runs on a worker."""
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

    async def _git_paths(self) -> list[str] | None:
        """Git-authoritative candidates, including the tracked-but-ignored subtraction."""
        # These queries only share Git's read-only index. Running them together removes one full
        # process latency from cold picker preparation in large parent worktrees -- two coroutines
        # now, so overlapping them costs no threads.
        queries = [
            asyncio.ensure_future(self._git_ls_files(["--cached", "--others", "--exclude-standard"])),
            asyncio.ensure_future(self._git_ls_files(["--cached", "--ignored", "--exclude-standard"])),
        ]
        try:
            included, ignored = await asyncio.gather(*queries)
        except asyncio.CancelledError:
            # `gather` has already cancelled both queries and reports the first ending; it does not
            # wait for them to unwind. Wait here instead, so a cancelled refresh cannot return
            # while two `git ls-files` are still walking the worktree behind it. Cancelling them a
            # second time would interrupt exactly the reap that is being waited for.
            await asyncio.wait(queries)
            raise
        except BaseException:
            # One query failed on its own; the other is still running and is now pointless.
            for query in queries:
                query.cancel()
            await asyncio.wait(queries)
            raise
        if included is None or ignored is None:
            return None
        excluded = set(ignored)
        return [path for path in included if path not in excluded]

    async def _git_ls_files(self, flags: list[str]) -> list[str] | None:
        stdout, returncode = await self._read_null_separated(["git", "ls-files", "-z", *flags, "--", "."])
        if returncode != 0:
            return None
        return [os.fsdecode(item) for item in stdout.split(b"\0") if item]

    async def _rg_paths(self) -> list[str] | None:
        executable = shutil.which("rg")
        if executable is None:
            return None
        argv = [executable, "--files", "--hidden", "--no-require-git", "--glob", "!**/.git/**", "--null"]
        stdout, returncode = await self._read_null_separated(argv)
        if returncode not in (0, 1):
            return None
        return [os.fsdecode(item) for item in stdout.split(b"\0") if item]

    async def _read_null_separated(self, argv: list[str]) -> tuple[bytes, int | None]:
        """Run one discovery command in the workspace and read its NUL-separated output.

        Returns (stdout, returncode), with a None return code standing for "this source did not
        answer" -- it could not be launched, or it ran past the timeout. Either way the process is
        killed and reaped before this returns: a refresh that is cancelled or times out must not
        leave a `git ls-files` walking a large worktree behind it."""

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.session.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return b"", None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), self.GIT_TIMEOUT)
        except TimeoutError:
            await self._reap(process)
            return b"", None
        except BaseException:
            # Cancellation, or anything else on the way out: the child is this call's to end.
            await self._reap(process)
            raise
        return stdout, process.returncode

    @staticmethod
    async def _reap(process: asyncio.subprocess.Process) -> None:
        """Kill one discovery child and wait for it, so nothing is left running behind a refresh."""
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(process.wait())

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

    # One selected path, plus a sentinel byte: anything longer is not an answer fzf could have
    # produced from a single-selection picker, so it is refused rather than parsed.
    OUTPUT_LIMIT = 1 << 20

    async def pick(self, query: str) -> FilePick:
        """Open fzf over the candidate snapshot and return what the reader chose.

        The process is the runtime's own child from launch to reap: cancelling this -- a shutdown
        while the picker is up -- kills fzf and quiesces the candidate work before it returns,
        rather than leaving a child owning the terminal the application is about to reclaim."""

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
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
            )
        except OSError:
            self._failed = True
            return FilePick(unavailable=True)

        feed = asyncio.ensure_future(self._feed(process))
        try:
            output = await self._read_selection(process)
        except BaseException:
            await self._end(process, feed)
            raise
        if output is None:
            # Past the bound. Not something a single-selection picker produces, so the answer is
            # refused and the process it came from is ended rather than waited on.
            await self._end(process, feed)
            return FilePick()
        returncode = await process.wait()
        snapshot, feed_failed = await self._settle_feed(feed)
        if returncode == 1 and feed_failed:
            return FilePick(unavailable=True)
        if returncode in (1, 130):
            return FilePick()
        if returncode != 0:
            self._failed = True
            return FilePick(unavailable=True)
        return self._selection(output, snapshot)

    async def _feed(self, process: asyncio.subprocess.Process) -> tuple[tuple[str, str], ...]:
        """Write the candidate snapshot into fzf, and return the exact snapshot that was fed.

        That snapshot is what the selection is revalidated against, so it is the task's result
        rather than shared state: the list the reader saw is the list their answer must be in.

        On a cold cache the scan is raced against fzf's own exit, so Escape closes the picker
        instead of waiting for a large worktree to be walked first."""

        stdin = process.stdin
        assert stdin is not None
        try:
            snapshot = self.mentions.cached_paths()
            if snapshot is None:
                snapshot = await self._scan_or_exit(process)
                if snapshot is None:
                    return ()
            else:
                # A stale snapshot is still safe to display: the selected path is revalidated
                # below. Refresh it for the next picker without delaying this one's first row.
                self.mentions.schedule_refresh()
            for _, path in snapshot:
                stdin.write(os.fsencode(path) + b"\0")
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # fzf closed early -- a selection or an Escape, both handled by the exit code.
        finally:
            with suppress(OSError, BrokenPipeError, ConnectionResetError):
                stdin.close()
        return snapshot or ()

    async def _scan_or_exit(self, process: asyncio.subprocess.Process) -> tuple[tuple[str, str], ...] | None:
        """Wait for a cold-cache scan, unless fzf exits first. None means the picker is already gone."""

        scan = asyncio.ensure_future(self.mentions.candidates())
        exiting = asyncio.ensure_future(process.wait())
        try:
            await asyncio.wait({scan, exiting}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not exiting.done():
                exiting.cancel()
            if not scan.done():
                # Only this wrapper is cancelled; a scan coalesced by the loop owner keeps running
                # for whoever else asked for it.
                scan.cancel()
                await asyncio.wait({scan})
        return scan.result() if not scan.cancelled() else None

    async def _read_selection(self, process: asyncio.subprocess.Process) -> bytes | None:
        """Read fzf's answer up to the bound. None means it went past it and is not an answer."""
        stdout = process.stdout
        assert stdout is not None
        data = b""
        while len(data) <= self.OUTPUT_LIMIT:
            chunk = await stdout.read(self.OUTPUT_LIMIT + 1 - len(data))
            if not chunk:
                return data
            data += chunk
        return None

    async def _end(self, process: asyncio.subprocess.Process, feed: asyncio.Task) -> None:
        """Kill and reap fzf, then quiesce the candidate work, before this call returns."""
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.shield(process.wait())
        feed.cancel()
        await asyncio.wait({feed})

    @staticmethod
    async def _settle_feed(feed: asyncio.Task) -> tuple[tuple[tuple[str, str], ...], bool]:
        """The snapshot fed, and whether preparing it failed -- which routes to the fallback menu."""
        await asyncio.wait({feed})
        if feed.cancelled() or feed.exception() is not None:
            return (), True
        return feed.result(), False

    def _selection(self, output: bytes, snapshot: tuple[tuple[str, str], ...]) -> FilePick:
        """Revalidate fzf's answer against the exact snapshot it was shown."""
        selected = os.fsdecode(output.split(b"\0", 1)[0]) if output else ""
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
