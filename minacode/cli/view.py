"""TUI fragment and style supply for the interactive shell.

The view-facing half of the interactive loop: dividers, follow-up markers, model stream
previews, the input hint, and the prompt-toolkit style map. Reads the live loop state through
its `loop` reference; it renders, it does not own behavior.
"""

from __future__ import annotations

import heapq
import re
import shutil
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, ClassVar

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from minacode.base import LogBlock, LogEdge, Text, TurnBox
from minacode.cli.commands import SET_KEYS, SET_VALUES
from minacode.cli.hints import Context as HintContext
from minacode.cli.hints import HintPicker
from minacode.cli.runtime import RESUME_STATUS_LABEL
from minacode.cli.worker import WORKER_SUBCOMMANDS
from minacode.config import (
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
)
from minacode.mentions import MentionSpan, active_mention, encode_file_mention
from minacode.render import LiveSpark, Theme, UiPrinter
from minacode.session import QueuedInput
from minacode.tui import TuiApp

if TYPE_CHECKING:
    from minacode.cli import CommandLoop


class CommandCompleter(Completer):
    """Prompt-toolkit completer for slash commands, their arguments, and @/$ mentions."""

    # The three kinds offered on a bare "@", each with its one-line meta (SPEC 4.2).
    KINDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("file:", "files in this repo"),
        ("mcp:", "MCP servers and tools"),
        ("skill:", "installed skills"),
    )
    MAX_ROWS = 50  # SPEC R4: cap the menu.

    def __init__(
        self,
        providers: Callable[[], tuple[str, ...]] = tuple,
        models: Callable[[], tuple[str, ...]] = tuple,
        worker_models: Callable[[], tuple[str, ...]] = tuple,
        # The active model's own scale, so completion offers exactly what `/reason` accepts.
        reasoning_choices: Callable[[], tuple[str, ...]] = lambda: REASONING_CHOICES,
        # Workers can target another provider/model, so offer the catalog-wide vocabulary here.
        worker_reasoning_choices: Callable[[], tuple[str, ...]] = lambda: REASONING_CHOICES,
        mcp_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_connected_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_tools: Callable[[str], tuple[str, ...]] = lambda _server: (),
        skills: Callable[[], tuple[str, ...]] = tuple,
        files: Callable[[], tuple[tuple[str, str], ...]] = tuple,
        file_matches: Callable[[str], tuple[str, ...]] | None = None,
    ):
        self.providers = providers
        self.models = models
        self.worker_models = worker_models
        self.reasoning_choices = reasoning_choices
        self.worker_reasoning_choices = worker_reasoning_choices
        self.mcp_servers = mcp_servers
        self.mcp_connected_servers = mcp_connected_servers
        self.mcp_tools = mcp_tools
        self.skills = skills
        # (lowercase, original) workspace-relative paths from the session's cached path list.
        self.files = files
        self.file_matches = file_matches

    def get_completions(self, document, complete_event):
        del complete_event
        text = document.text_before_cursor
        if text.startswith("/set "):
            tail = text[len("/set ") :]
            if " " not in tail:
                yield from self.matches(SET_KEYS, tail)
                return
            key, _, value = tail.partition(" ")
            yield from self.matches(SET_VALUES.get(key, ()), value)
            return
        if text.startswith("/worker "):
            tail = text[len("/worker ") :]
            if " " not in tail:
                yield from self.matches(WORKER_SUBCOMMANDS, tail)
                return
            sub, _, value = tail.partition(" ")
            if sub == "provider":
                yield from self.matches(tuple(dict.fromkeys((*self.providers(), "off"))), value)
                return
            if sub == "model":
                yield from self.matches(self.worker_models(), value)
                return
            if sub == "reason":
                yield from self.matches((*self.worker_reasoning_choices(), "default"), value)
                return
            if sub == "api":
                yield from self.matches((*PROVIDER_API_CHOICES, "default"), value)
                return
        for command, values in (
            ("/model ", self.models),
            ("/provider ", self.providers),
            ("/reason ", self.reasoning_choices),
            ("/effort ", self.reasoning_choices),
            ("/api ", lambda: PROVIDER_API_CHOICES),
            ("/strict ", lambda: ("on", "off")),
            ("/compact ", lambda: ("log",)),
        ):
            if text.startswith(command):
                yield from self.matches(values(), text[len(command) :])
                return
        if text.startswith("/mcp "):
            tail = text[len("/mcp ") :]
            if " " not in tail:
                yield from self.matches(("connect", "disconnect", "tools"), tail)
                return
            sub, _, value = tail.partition(" ")
            if sub == "connect":
                completed, _, prefix = value.rpartition(" ")
                selected = set(completed.split())
                yield from self.matches((name for name in self.mcp_servers() if name not in selected), prefix)
                return
            if sub == "disconnect":
                yield from self.matches(self.mcp_servers(), value)
                return
            if sub == "tools":
                yield from self.matches(self.mcp_connected_servers(), value)
                return

        span = active_mention(text)
        if span is not None:
            yield from self._mention_completions(span, span.start - len(text))
            return

        if text.startswith("/") and " " not in text:
            # CommandLoop.COMMANDS is populated at the end of cli/__init__.py, after this module
            # is imported; resolve it lazily to avoid the import cycle.
            from minacode.cli import CommandLoop

            yield from self.matches(CommandLoop.COMMANDS, text)

    @staticmethod
    def matches(values, prefix: str):
        return (Completion(value, start_position=-len(prefix)) for value in values if value.startswith(prefix))

    def _mention_completions(self, span: MentionSpan, start: int) -> Iterator[Completion]:
        """Complete one scanner-owned span and always insert canonical namespace forms."""
        if span.kind == "file":
            yield from self._file_completions(span.payload, start)
        elif span.kind == "mcp":
            yield from self._mcp_completions(span.payload, start)
        elif span.kind == "skill":
            yield from self._skill_completions(span.payload, start)
        else:
            yield from self._merged_completions(span.payload, start)

    def _merged_completions(self, raw: str, start: int) -> Iterator[Completion]:
        """The bare "@" menu: kinds while they prefix-match, then all three sources (SPEC 4.2).

        Repository files deliberately stay out of this menu: selecting @file: opens the dedicated
        picker without scanning the repository merely because the user typed "@".
        """
        if not raw:
            for kind, meta in self.KINDS:
                yield Completion("@" + kind, start_position=start, display_meta=meta)
            return
        lower = raw.lower()
        kind_items = [Completion("@" + kind, start_position=start, display_meta=meta) for kind, meta in self.KINDS if kind.startswith(lower)]

        server_part, dot, tool_part = raw.partition(".")
        if dot:
            server = self._known_server(server_part)
            mcp_items = (
                [
                    Completion(f"@mcp:{server}.{name}", start_position=start, display_meta="mcp")
                    for name in self._matching_names(self.mcp_tools(server), tool_part)
                ]
                if server is not None
                else []
            )
        else:
            mcp_items = [Completion(f"@mcp:{name}", start_position=start, display_meta="mcp") for name in self._matching_names(self.mcp_servers(), raw)]
        skill_items = [Completion(f"@skill:{name}", start_position=start, display_meta="skill") for name in self._matching_names(self.skills(), raw)]
        yield from [*kind_items, *mcp_items, *skill_items][: self.MAX_ROWS]

    def _file_completions(self, query: str, start: int) -> Iterator[Completion]:
        for path in self._matching_files(query):
            yield Completion(encode_file_mention(path), start_position=start)

    def _matching_files(self, query: str) -> list[str]:
        """Case-insensitive substring over the whole workspace-relative path (SPEC M3), ranked
        by prefix of basename, then substring of basename, then substring of path (R1); ties by
        shorter path, then alphabetical (R3). Deterministic - no recency, no MRU."""
        if self.file_matches is not None:
            return list(self.file_matches(query))
        q = query.lower()

        def ranked():
            for lower, path in self.files():
                if q not in lower:
                    continue
                base = lower.rsplit("/", 1)[-1]
                if base.startswith(q):
                    score = 0
                elif q in base:
                    score = 1
                else:
                    score = 2
                yield score, len(lower), lower, path

        return [path for _, _, _, path in heapq.nsmallest(self.MAX_ROWS, ranked())]

    def _mcp_completions(self, query: str, start: int) -> Iterator[Completion]:
        """After "@mcp:": servers, then "server." expands to that server's tools."""
        server_part, dot, tool_part = query.partition(".")
        if dot:
            server = self._known_server(server_part)
            if server is not None:
                for name in self._matching_names(self.mcp_tools(server), tool_part):
                    yield Completion(f"@mcp:{server}.{name}", start_position=start)
            return
        for name in self._matching_names(self.mcp_servers(), query):
            yield Completion(f"@mcp:{name}", start_position=start)

    def _skill_completions(self, query: str, start: int) -> Iterator[Completion]:
        for name in self._matching_names(self.skills(), query):
            yield Completion(f"@skill:{name}", start_position=start)

    @staticmethod
    def _matching_names(values, query: str) -> list[str]:
        """Servers, skills, tools, kinds: prefix match, then substring (SPEC M2), alphabetical."""
        q = query.lower()
        prefix, substring = [], []
        for name in dict.fromkeys(values):
            low = name.lower()
            if low.startswith(q):
                prefix.append(name)
            elif q in low:
                substring.append(name)
        return [*sorted(prefix), *sorted(substring)][: CommandCompleter.MAX_ROWS]

    def _known_server(self, name: str) -> str | None:
        for candidate in self.mcp_servers():
            if candidate == name or candidate.lower() == name.lower():
                return candidate
        return None


class View:
    """Fragments and style for the TUI, fed from a live CommandLoop."""

    # Breathing green dot shown on the divider while a model request is in flight. The label moves
    # from working to thinking/responding as stream events arrive; the pulse remains until completion.
    WAITING_PULSE_STYLES: ClassVar[tuple[str, ...]] = (
        "fg:#0a3d0a",
        "fg:#146114",
        "fg:#1f8a1f",
        "fg:#2dbf2d bold",
        "fg:#43e043 bold",
        "fg:#7bff7b bold",
    )
    WAITING_PULSE_PERIOD: ClassVar[float] = 1.6

    # One cell per frame. A head that advances further than its own glow between redraws stops
    # reading as motion and starts reading as a dash blinking at scattered positions.
    QUEUE_SWEEP_CELLS_PER_SEC: ClassVar[float] = 1.0 / TuiApp.ANIMATION_INTERVAL
    # A comet: a soft head with a tail fading into the dim rule, by distance from the head. The ramp
    # is finer than one shade per cell, so a head between two cells lights both partially instead of
    # snapping onto the nearer one. The divider is only drawn while working; there is no idle look.
    GLOW_REACH: ClassVar[float] = 4.0
    GLOW_STEPS: ClassVar[int] = 12

    QUEUE_EMPTY_HINT = "Enter queues follow-up · Ctrl-C interrupts"
    QUEUE_PENDING_HINT = "↑ recalls queued · Ctrl-C interrupts"

    # Line-level markdown tokens the live stream preview styles. Block constructs (headings,
    # lists, fenced code) are deliberately not parsed: the preview shows partial streaming text,
    # and only tokens that close on one line can render without flickering as the stream grows.
    # Each token must hold a non-blank character, so whitespace-only runs (`** **`, `* *`) stay
    # literal; the italic branch's `(?<!\*) ... (?!\*)` guards keep a lone star from borrowing
    # itself out of a `**` run.
    STREAM_INLINE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(\*\*[^*\n]*[^*\s\n][^*\n]*\*\*|`[^`\n]*[^`\s\n][^`\n]*`|(?<!\*)\*[^*\n]*[^*\s\n][^*\n]*\*(?!\*))"
    )

    def __init__(self, loop: CommandLoop) -> None:
        self.loop = loop
        self._hint_picker = HintPicker()  # idle-placeholder tips; see minacode/cli/hints.py

    def waiting_pulse_fragments(self) -> StyleAndTextTuples:
        if self.loop.session.state.current_model_call_started_at <= 0:
            return []
        # Triangular breath: 0 → 1 → 0 over WAITING_PULSE_PERIOD seconds, mapped onto the palette.
        phase = (time.monotonic() % self.WAITING_PULSE_PERIOD) / self.WAITING_PULSE_PERIOD
        intensity = 1.0 - abs(2.0 * phase - 1.0)
        idx = min(len(self.WAITING_PULSE_STYLES) - 1, int(intensity * len(self.WAITING_PULSE_STYLES)))
        return [(self.WAITING_PULSE_STYLES[idx], "● ")]

    def sweep_divider_fragments(self, label: str, width: int | None = None, prefix: StyleAndTextTuples | None = None) -> StyleAndTextTuples:
        prefix = prefix or []
        prefix_len = sum(len(fragment[1]) for fragment in prefix)
        cols = shutil.get_terminal_size((80, 20)).columns
        width = width if width is not None else max(20, min(52, cols - 2))
        lead = 3
        # A comet needs a track long enough to read as motion. When the label is long (the
        # worker's `[worker]` + status + elapsed + rate + queued), widen the rule so both sides
        # keep at least MIN_TRAIL dashes instead of clipping the label; the terminal width is
        # the ceiling, and the trail shrinks only when even that cannot fit.
        min_trail = 12
        body_len = prefix_len + get_cwidth(label) + 2  # prefix + " label "
        width = min(max(cols - 2, 20), max(width, lead + body_len + min_trail))
        trail = max(3, width - lead - body_len)
        dash_count = lead + trail
        # The comet head bounces over the horizontal rule only. The label stays stable and readable
        # while the glow appears to pass through the dash track on either side.
        span = max(1, dash_count - 1)
        # A short track (a narrow terminal squeezing the trail) must not make the head bounce
        # faster than the eye can follow: hold each round trip to at least a second by lowering
        # the sweep rate as the span shrinks; normal-length tracks keep the full frame-per-cell
        # rate.
        sweep = min(self.QUEUE_SWEEP_CELLS_PER_SEC, 2 * span / 1.0)
        phase = time.monotonic() * sweep % (2 * span)
        head = phase if phase <= span else 2 * span - phase

        def dashes(offset: int, count: int) -> StyleAndTextTuples:
            fragments: StyleAndTextTuples = []
            for i in range(count):
                step = int(abs(offset + i - head) / self.GLOW_REACH * self.GLOW_STEPS)
                fragments.append((f"class:divider.glow{step}" if step < self.GLOW_STEPS else "class:queue.rule", "-"))
            return fragments

        return [
            *dashes(0, lead),
            ("class:queue.rule", " "),
            *prefix,
            ("class:divider.working", label),
            ("class:queue.rule", " "),
            *dashes(lead, trail),
        ]

    def queue_divider_fragments(self, queued: int = 0) -> StyleAndTextTuples:
        tui = self.loop.tui
        status = tui.status_label if tui is not None and tui.status_label else "working"
        if status in {"working", "retrying", "compacting context"}:
            retry_status = self.loop.status_bar.retry_status()
            attempt_status = self.loop.status_bar.model_attempt_status()
            with self.loop.model_stream_lock:
                phase = self.loop.model_stream_kind
            activity = retry_status or (
                ({"reasoning": "thinking", "output": "responding"}.get(phase, phase) or status) + (" · " + attempt_status if attempt_status else "")
            )
            # The elapsed time says how long the wait has been; the rate says whether it is moving.
            # Empty whenever nothing is streaming (between requests, non-streaming providers, a
            # summary), so the divider never claims a speed it is not currently seeing.
            rate = self.loop.status_bar.output_rate()
            label = f"{activity} ({Text.elapsed_since(self.loop.status_bar.started_at)}{' · ' + rate if rate else ''})"
        elif status == "running script":
            # Timed like the working phases, but never relabelled from the stream phase: nothing is
            # streaming while a script runs, so the last kind seen would be stale, not current.
            label = f"{status} ({Text.elapsed_since(self.loop.status_bar.started_at)})"
        elif status == RESUME_STATUS_LABEL:
            # A quiet gray lead-in to the restored transcript: nothing streams and nothing sweeps
            # while the replay is being prepared, so no pulse pretends there is activity.
            return [("ansibrightblack", status)]
        else:
            label = status
        if queued:
            label = f"{label} [ {queued} queued ]"
        prefix = self.waiting_pulse_fragments()
        worker = self.loop.session.worker
        if worker is not None and worker._active_turn_messages:
            # The same in-flight predicate as the status bar's worker marker.
            prefix = [("class:divider.worker", "[worker] "), *prefix]
        return self.sweep_divider_fragments(label, prefix=prefix)

    def followup_fragments(self) -> tuple[StyleAndTextTuples, StyleAndTextTuples]:
        with self.loop.session._queue_lock:
            pending = list(self.loop.session.pending_user_inputs)

        def render(items: list[QueuedInput], marker: str, marker_style: str) -> StyleAndTextTuples:
            fragments: StyleAndTextTuples = []
            for item in items:
                for index, line in enumerate(item.text.splitlines()):
                    fragments.extend([("", "\n"), (marker_style, marker if index == 0 else "  "), (UiPrinter.user_log_style(), line)])
            return fragments

        sent = [item for item in pending if item.inflight]
        queued = [item for item in pending if not item.inflight]
        transcript = render(sent, UiPrinter.USER_LOG_PREFIX, "class:prompt")
        # The divider is a standing boundary for the whole turn. Only messages that have not entered
        # a model request remain below it; sent messages render above it until the request commits them.
        waiting = self.queue_divider_fragments(len(queued))
        waiting.extend(render(queued, "+ ", UiPrinter.user_log_style()))
        return transcript, waiting

    def tui_activity_fragments(self) -> StyleAndTextTuples:
        sent, waiting = self.followup_fragments()
        fragments = sent
        stream = self.model_stream_fragments()
        if fragments:
            fragments.append(("", "\n"))
            # A blank row lifts the echoed follow-up off whatever follows it: the streamed reply,
            # or the standing divider when no stream exists yet. The divider always sits below,
            # so the gap can never leave a hanging blank row at the end of the activity region.
            fragments.append(("", "\n"))
        fragments.extend(stream)
        if stream:
            fragments.append(("", "\n"))
        with self.loop.live_preview.lock:
            rows = self.loop.live_preview.frame_rows() if self.loop.live_preview.active else []
        for row in rows:
            fragments.extend([*row, ("", "\n")])
        if rows:
            fragments.append(("", "\n"))
        fragments.extend(waiting)
        return fragments

    def model_stream_fragments(self) -> StyleAndTextTuples:
        with self.loop.model_stream_lock:
            text = self.loop.model_stream_text
            kind = self.loop.model_stream_kind
        if not text:
            return []
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        # Drawn with LogBlock's own rail so it cannot drift from the tree every tool call draws.
        # The rows carry CONTINUE (`│`) and nothing carries BRANCH: `├` is a T-junction, and there
        # is no line above the block for one to join. Nor does a `└` close it - the stream is still
        # arriving, and an end cap would say it had finished.
        rail = LogBlock.prefix(TurnBox.CONTENT_LEVEL + 1, LogEdge.CONTINUE)
        rows = [Text.clip_width(line.expandtabs(4), max(1, width - len(rail) - 1)) for line in text.replace("\r", "\n").splitlines()[-6:]]
        # The spark's row is the region's own, never the text's: a gray word beside the spark names
        # the phase (the same wording the divider below uses), and the first streamed line can arrive
        # on the next row down instead of racing for whatever room the spark leaves. The rows are
        # what the reader is actually reading, and breathing those would be a strobe rather than a
        # sign of life.
        spark = LogBlock.margin(TurnBox.CONTENT_LEVEL + 1) + LiveSpark.glyph(self.loop.session.state.stream_started_at)
        label = {"reasoning": "thinking", "output": "responding"}.get(kind)
        fragments: StyleAndTextTuples = [
            # Anchored to the stream this preview is showing, which ModelClient stamps on the
            # first chunk of each request and clears between them, so every response opens the
            # spark at its crest instead of wherever the wall clock happened to be.
            (LiveSpark.style(self.loop.session.state.stream_started_at), spark),
            *([("ansibrightblack", label)] if label else []),
            ("", "\n"),
            # A blank row keeps the spark off the rail: the star caps the region, it does not sit
            # on it. The running-command frame draws the same gap whenever it has output.
            ("", "\n"),
        ]
        for row in rows:
            fragments.append(("ansibrightblack", rail))
            fragments.extend(self._stream_inline_fragments(row))
            fragments.append(("", "\n"))
        return fragments

    @classmethod
    def _stream_inline_fragments(cls, row: str) -> StyleAndTextTuples:
        """One preview row as fragments: plain gray text with closed inline markdown tokens
        marked in the same gray tones (weight and underline, no color), so the region stays
        uniformly gray. A marker without its closing partner stays literal, so a stream that has
        not finished a token never toggles its style from frame to frame."""
        fragments: StyleAndTextTuples = []
        cursor = 0
        for match in cls.STREAM_INLINE_RE.finditer(row):
            if match.start() > cursor:
                fragments.append(("ansibrightblack", row[cursor : match.start()]))
            token = match.group(1)
            if token.startswith("**"):
                fragments.append(("ansibrightblack bold", token[2:-2]))
            elif token.startswith("`"):
                fragments.append(("ansibrightblack underline", token[1:-1]))
            else:
                fragments.append(("ansibrightblack italic", token[1:-1]))
            cursor = match.end()
        if cursor < len(row):
            fragments.append(("ansibrightblack", row[cursor:]))
        return fragments or [("ansibrightblack", row)]

    def tui_input_hint(self) -> str:
        tui = self.loop.tui
        if tui is None:
            return ""
        if tui.input_mode == "running":
            with self.loop.session._queue_lock:
                has_pending = any(not item.inflight for item in self.loop.session.pending_user_inputs)
            return self.QUEUE_PENDING_HINT if has_pending else self.QUEUE_EMPTY_HINT
        if tui.input_mode == "chat":
            return self._hint_picker.pick(self._hint_context(), self.loop.session.state.round_count)
        return ""

    def _hint_context(self) -> HintContext:
        """Project the session into the small situation the hint mechanism selects on.

        round_count only advances at the start of the next turn, so at idle it still names the
        round that just finished; edited_round therefore clears on its own once a later round
        makes no edits.
        """
        session = self.loop.session
        round_count = session.state.round_count
        edited = any((diff.round or diff.turn) == round_count for diff in session.turn_diffs)
        return HintContext(
            early=not session.tool_records,
            edited_round=round_count if edited else None,
            skills_available=bool(session.skills and session.skills.skills),
            mcp_connected=bool(session.mcp and session.mcp.tools),
            jobs_running=any(job.status == "running" for job in session.jobs.values()),
        )

    def style(self) -> Style:
        rule = Theme.style("divider.rule")
        return Style.from_dict(
            {
                "prompt": "ansicyan bold",
                # The comet fades into the rule it travels over, so both come from the palette.
                "queue.rule": rule,
                **{f"divider.glow{step}": color for step, color in enumerate(Theme.ramp("divider.glow", "divider.rule", self.GLOW_STEPS))},
                "queue.hint": "ansibrightblack",
                "quickhint": "ansicyan",
                "quickhint.focused": "reverse",
                "quickhint.sep": "ansibrightblack",
                "image.attachment": "ansicyan bold",
                "input.error": "ansired",
                "divider.working": "ansimagenta bold",
                "divider.worker": "ansiyellow bold",
                "approval": "ansiyellow",
                "approval.wait": "ansimagenta",
                "approval.action": "ansiyellow",
                "approval.action.focused": "reverse ansiyellow",
                "approval.action.dim": "ansibrightblack",
                "choice.title": "ansicyan bold",
                "choice.selected": "reverse",
                "choice.disabled": "ansibrightblack",
                "choice.preview": "ansigreen italic",
                # The preview's user messages take the same tone as the transcript's `• ` lines.
                "choice.user": Theme.style("user.log"),
                # The Ctrl-O browser's rows, coloured as the transcript colours the same call: the
                # tr.N key dim, the tool name in the tool green, the arguments plain. A row that is
                # still running takes the working divider's magenta instead of the dim key.
                "choice.meta": "ansibrightblack",
                "choice.tool": "ansigreen",
                "choice.live": "ansimagenta bold",
                "choice.output.ok": "ansigreen bold",
                "choice.output.fail": "ansired bold",
                "choice.status.connected": "ansigreen bold",
                "choice.status.connecting": "ansigreen bold",
                "choice.status.disconnected": "ansiyellow bold",
                "choice.status.disconnecting": "ansiyellow bold",
                "choice.status.error": "ansired bold",
                "choice.status.skipped": "ansibrightblack",
                "tab.active": "bold reverse ansicyan",
                "tab.inactive": "ansicyan",
                "completion-menu": "noreverse bg:default",
                "completion-menu.completion": "noreverse bg:default fg:default",
                "completion-menu.completion.current": "noreverse bg:default fg:ansicyan bold",
                "completion-menu.meta.completion": "noreverse bg:default fg:ansibrightblack",
                "completion-menu.meta.completion.current": "noreverse bg:default fg:ansicyan",
                "bottom-toolbar": "noreverse bg:default fg:default",
                "bottom-toolbar.text": "noreverse bg:default fg:default",
                "search-toolbar": "noreverse bg:default fg:default",
                "search-toolbar.prompt": "ansicyan",
                "search-toolbar.text": "fg:default",
            }
        )
