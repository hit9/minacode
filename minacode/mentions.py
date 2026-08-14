"""@file: mentions: the file source (cached path list) and the FILE MENTIONS turn resolver.

One class owns both halves of the feature. The *source* feeds the prompt completer with a cached
list of workspace-relative paths (git ls-files when the cwd is a git repo, otherwise the
same gitignore-aware walk the Search tool uses), stored lowercase so no keystroke pays for
.lower(). The *resolver* turns a turn's @file: mentions into the FILE MENTIONS block appended
after the user's own message, inlining small in-workspace files and pointing at everything else.

The path list is rebuilt only when it is stale at the moment a completion starts - never on a
timer, never in the background: a stale entry costs one wrong row, a background thread costs a
race with the event loop.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minacode.session import Session


class FileMentions:
    """The @file: mention source and resolver."""

    # The cache is stale once a file could have appeared since it was built; completion only
    # reads the list at the moment it starts, so a short window costs nothing.
    CACHE_TTL = 5.0
    MAX_CANDIDATES = 50_000  # non-git walk bound
    MAX_INLINE_LINES = 400  # DECISION D1: bounded inline; larger files become pointers.
    MAX_INLINE_BYTES = 64 * 1024
    MAX_FILES = 10  # cap per turn; further mentions become pointers with a note
    # `@` preceded by a word character is not a mention (protects email addresses); a mention
    # ends at whitespace, so a path containing a space cannot be mentioned.
    MENTION_PATTERN = re.compile(r"(?:^|(?<=[^A-Za-z0-9_]))@file:([^\s]+)")

    def __init__(self, session: Session) -> None:
        self.session = session
        self._paths_cache: tuple[float, list[tuple[str, str]]] | None = None

    def paths(self) -> list[tuple[str, str]]:
        """Cached (lowercase, original) workspace-relative POSIX paths.

        Rebuilt on first use and whenever older than CACHE_TTL at call time.
        """
        now = time.monotonic()
        cached = self._paths_cache
        if cached is None or now - cached[0] >= self.CACHE_TTL:
            cached = (now, self._collect())
            self._paths_cache = cached
        return cached[1]

    def _collect(self) -> list[tuple[str, str]]:
        rels = self._git_paths()
        if rels is None:
            rels = self._walk_paths()
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for rel in rels:
            if rel in seen:
                continue
            seen.add(rel)
            pairs.append((rel.lower(), rel))
        pairs.sort()
        return pairs

    def _git_paths(self) -> list[str] | None:
        """Tracked plus untracked-not-ignored files, or None when cwd is not a git repo."""
        try:
            tracked = subprocess.run(["git", "ls-files"], cwd=self.session.cwd, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if tracked.returncode != 0:
            return None
        rows = tracked.stdout.splitlines()
        try:
            untracked = subprocess.run(
                ["git", "ls-files", "-o", "--exclude-standard"],
                cwd=self.session.cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return rows
        if untracked.returncode == 0:
            rows.extend(untracked.stdout.splitlines())
        return rows

    def _walk_paths(self) -> list[str]:
        # Local import: tools.search imports the session package at module scope.
        from minacode.tools.search import SearchTool

        rels: list[str] = []
        for path in SearchTool(self.session, []).files(self.session.cwd, ""):
            rels.append(self.session.relpath(path).replace(os.sep, "/"))
            if len(rels) >= self.MAX_CANDIDATES:
                break
        return rels

    def resolve_mentions(self, text: str) -> str:
        """The FILE MENTIONS block for a turn's user text, or "" when none mention a file.

        Mentions resolve in text order; each unique path appears once. Small in-workspace files
        are inlined, anything else becomes a pointer, and the MAX_FILES cap turns further
        mentions into pointers with a note. Missing paths report themselves - never silently
        dropped, since the user meant something by them.
        """
        blocks: list[str] = []
        seen: set[str] = set()
        for raw in self.MENTION_PATTERN.findall(text):
            if raw in seen:
                continue
            seen.add(raw)
            if len(blocks) >= self.MAX_FILES:
                blocks.append(self._overflow_pointer(raw))
                continue
            blocks.append(self._file_block(raw))
        if not blocks:
            return ""
        header = [
            "--- FILE MENTIONS ---",
            "The user explicitly referenced these files. Treat them as the subject of the request.",
            "",
        ]
        return "\n".join([*header, *blocks]).strip()

    def _file_block(self, raw: str) -> str:
        path = raw if os.path.isabs(raw) else os.path.join(self.session.cwd, raw)
        if os.path.isdir(path):
            return f"[{raw}] directory; Read it to see its contents"
        if not os.path.isfile(path):
            return f"[{raw}] not found"
        if not self.session.in_cwd(path):
            # Never inlined, whatever the size: Read keeps the out-of-workspace confirmation.
            return f"[{raw}] outside the workspace; Read it"
        size = os.path.getsize(path)
        if size > self.MAX_INLINE_BYTES:
            return f"[{raw}] {self._line_count(path)} lines, {self._size_label(size)} - too large to inline; Read the part you need"
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except OSError:
            return f"[{raw}] not found"
        lines = len(content.splitlines())
        if lines > self.MAX_INLINE_LINES:
            return f"[{raw}] {lines} lines, {self._size_label(size)} - too large to inline; Read the part you need"
        return f"[{raw}] {lines} lines\n{content}"

    def _overflow_pointer(self, raw: str) -> str:
        return f"[{raw}] not included - the {self.MAX_FILES}-file cap is reached; Read it if relevant"

    @staticmethod
    def _line_count(path: str) -> int:
        """Newline count for a file too large to read whole; streams in bounded memory."""
        count = 0
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                count += chunk.count(b"\n")
        return count

    @staticmethod
    def _size_label(size: int) -> str:
        return f"{size // 1024} KB" if size >= 1024 else f"{size} B"
