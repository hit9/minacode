"""TUI fragment and style supply for the interactive shell.

The view-facing half of the interactive loop: dividers, follow-up markers, model stream
previews, the input hint, and the prompt-toolkit style map. Reads the live loop state through
its `loop` reference; it renders, it does not own behavior.
"""

from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING, ClassVar

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.styles import Style

from minacode.base import Text
from minacode.hints import Context as HintContext
from minacode.hints import HintPicker
from minacode.render import Theme, UiPrinter
from minacode.session import QueuedInput
from minacode.tui import TuiApp

if TYPE_CHECKING:
    from minacode.cli import CommandLoop


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

    def __init__(self, loop: CommandLoop) -> None:
        self.loop = loop
        self._hint_picker = HintPicker()  # idle-placeholder tips; see minacode/hints.py

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
        body_len = prefix_len + len(label) + 2  # prefix + " label "
        lead = 3
        trail = max(3, width - lead - body_len)
        dash_count = lead + trail
        # The comet head bounces over the horizontal rule only. The label stays stable and readable
        # while the glow appears to pass through the dash track on either side.
        span = max(1, dash_count - 1)
        phase = time.monotonic() * self.QUEUE_SWEEP_CELLS_PER_SEC % (2 * span)
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
            label = f"{activity} ({Text.elapsed_since(self.loop.status_bar.started_at)})"
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
        if fragments:
            fragments.append(("", "\n"))
        stream = self.model_stream_fragments()
        fragments.extend(stream)
        if stream:
            fragments.append(("", "\n"))
        with self.loop.live_preview.lock:
            lines = self.loop.live_preview.frame_lines() if self.loop.live_preview.active else []
        for line in lines:
            fragments.extend([("ansibrightblack", line), ("", "\n")])
        if lines:
            fragments.append(("", "\n"))
        fragments.extend(waiting)
        return fragments

    def model_stream_fragments(self) -> StyleAndTextTuples:
        with self.loop.model_stream_lock:
            kind, text = self.loop.model_stream_kind, self.loop.model_stream_text
        if not text:
            return []
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        label = "thinking" if kind == "reasoning" else "responding"
        rows = [Text.clip_width(line.expandtabs(4), max(1, width - 4)) for line in text.replace("\r", "\n").splitlines()[-6:]]
        lines = [f"├─ {label}", *(f"│  {row}" for row in rows)]
        fragments: StyleAndTextTuples = []
        for line in lines:
            fragments.extend([("ansibrightblack", line), ("", "\n")])
        return fragments

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
                "choice.title": "ansicyan bold",
                "choice.selected": "reverse",
                "choice.disabled": "ansibrightblack",
                "choice.preview": "ansigreen italic",
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
