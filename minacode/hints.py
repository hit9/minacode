"""Idle-input hint mechanism.

One centralized place for the low-noise tips shown in the empty input placeholder. Each hint
declares its text, a weight, and an optional applicability predicate over a small Context that
describes the session's current situation. HintPicker draws a weighted random hint among the
applicable ones and caches it per Context, so a frequently re-rendered placeholder stays stable
and only re-rolls when the situation changes (a new editing round, leaving the early phase, ...).

Adding a tip is one line in HINTS; adding a new kind of situation is one field on Context plus a
predicate. The selection logic itself never changes.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Context:
    """The session situation that decides which hints apply and when to re-roll."""

    early: bool  # the session has done no work yet; navigation tips are welcome
    edited_round: int | None  # the round that just edited files, or None
    skills_available: bool = False  # at least one skill is installed
    mcp_connected: bool = False  # at least one MCP server is connected
    jobs_running: bool = False  # a background job is still running


def _when_early(ctx: Context) -> bool:
    return ctx.early


def _when_edited(ctx: Context) -> bool:
    return ctx.edited_round is not None


def _when_skills(ctx: Context) -> bool:
    return ctx.skills_available


def _when_mcp(ctx: Context) -> bool:
    return ctx.mcp_connected


def _when_jobs(ctx: Context) -> bool:
    return ctx.jobs_running


@dataclass(frozen=True)
class Hint:
    """One candidate tip: its text, selection weight, and when it applies (None = always)."""

    text: str
    weight: int = 1
    when: Callable[[Context], bool] | None = None


HINTS: tuple[Hint, ...] = (
    Hint("Esc then Enter inserts a newline"),
    Hint("Tab completes commands and @mentions"),
    Hint("↑ or Ctrl-P recalls earlier prompts"),
    Hint("Ctrl-R searches prompt history"),
    Hint("$skill loads a skill inline", when=_when_skills),
    Hint("@server.tool mentions an MCP tool", when=_when_mcp),
    Hint("Ctrl-X Ctrl-E opens $EDITOR"),
    Hint("Ctrl-U clears the line"),
    Hint("Paste an image path to attach it"),
    Hint("Type / for commands"),
    Hint("/sessions resumes a past session", when=_when_early),
    # Right after editing, /diff is the most useful tip: weight it high, but keep it a random
    # pick so repeated edits do not show the same line every time.
    Hint("/diff reviews recent edits", weight=3, when=_when_edited),
    # While a background job runs, remind the user it can be listed; clears once none run.
    Hint("/ps lists background jobs", weight=2, when=_when_jobs),
)


class HintPicker:
    """Weighted, per-Context cached selection over HINTS (see module docstring).

    The pick is cached per Context so a frequently re-rendered placeholder does not flicker; it
    re-rolls only when the Context changes. Inject `choice` for deterministic tests.
    """

    def __init__(self, choice: Callable[[Sequence[str]], str] = random.choice) -> None:
        self._choice = choice
        self._cache: tuple[Context, str] | None = None

    def pick(self, ctx: Context) -> str:
        if self._cache is None or self._cache[0] != ctx:
            self._cache = (ctx, self._select(ctx))
        return self._cache[1]

    def _select(self, ctx: Context) -> str:
        pool = [hint.text for hint in HINTS if hint.when is None or hint.when(ctx) for _ in range(hint.weight)]
        return self._choice(pool)
