"""minacode command loop and interactive session runtime."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from prompt_toolkit import print_formatted_text
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from minacode.base import (
    DISMISSED,
    HTTP_USER_AGENT,
    IMAGE_INPUT_CHOICES,
    PROVIDER_API_CHOICES,
    REASONING_CHOICES,
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    ConfigError,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    MalformedToolCallError,
    MinacodeError,
    ProviderConfig,
    Text,
    ToolCall,
    ToolError,
    TurnBox,
    __version__,
)
from minacode.engine import Agent
from minacode.hints import Context as HintContext
from minacode.hints import HintPicker
from minacode.image import ImageInputs, UserInput
from minacode.model import ModelClient
from minacode.prompts import PREVIOUS_CONTEXT_TRIMMED, SYSTEM_PROMPT
from minacode.provider_compat import builtin_tools_issue
from minacode.render import BashLivePreview, StatusBar, Theme, UiPrinter, markdown_table, search_sources_footer
from minacode.runner import ToolDisplay
from minacode.session import QueuedInput, SessionEntry, SessionSnapshotCodec, SessionSnapshotStore, ToolResultRecord
from minacode.tools import TOOL_REGISTRY, AskSpec, CodeIndex
from minacode.tui import TUI_MODAL_PENDING, ChoiceViewState, DiffViewState, TabbedViewState, TuiApp
from minacode.update import UpdateChecker

SetHandler = tuple[str, str, Callable[[str], int | float | None] | None]
# fmt: off
SET_HANDLERS: dict[str, SetHandler] = {
    "provider.temperature": ("provider", "temperature", lambda v: None if v == "off" else float(v)),
    "provider.max_tokens": ("provider", "max_tokens", lambda v: max(0, int(v))),
    "provider.timeout": ("provider", "timeout", lambda v: max(1, int(v))),
    "provider.response_timeout": ("provider", "response_timeout", lambda v: max(0, int(v))),
    "provider.stream": ("provider", "stream", lambda v: v == "on"),
    "provider.image_input": ("provider", "image_input", None),
    "runtime.max_agent_steps": ("settings", "max_steps", lambda v: max(1, int(v))),
    "runtime.max_context_tokens": ("settings", "max_context_tokens", lambda v: max(1, int(v))),
    "runtime.max_parallel_tools": ("settings", "max_parallel_tools", lambda v: max(1, int(v))),
    "runtime.shell_timeout": ("settings", "shell_timeout", lambda v: max(1, int(v))),
    "runtime.bash_wait_timeout": ("settings", "bash_wait_timeout", lambda v: max(0, int(v))),
}
SET_KEYS = tuple(SET_HANDLERS)
# Keys whose values are a closed set: rejected by /set when unknown, and offered whole as completions.
SET_CHOICES: dict[str, tuple[str, ...]] = {
    "provider.stream": ("on", "off"),
    "provider.image_input": IMAGE_INPUT_CHOICES,
}
SET_VALUES: dict[str, tuple[str, ...]] = {
    "provider.temperature": ("off",),
    **SET_CHOICES,
}
# fmt: on


class CommandCompleter(Completer):
    MCP_MENTION_RE: ClassVar[re.Pattern] = re.compile(r"@([A-Za-z0-9_.-]*)$")
    SKILL_MENTION_RE: ClassVar[re.Pattern] = re.compile(r"(?<![A-Za-z0-9_])\$([A-Za-z0-9_-]*)$")

    def __init__(
        self,
        providers: Callable[[], tuple[str, ...]] = tuple,
        models: Callable[[], tuple[str, ...]] = tuple,
        mcp_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_connected_servers: Callable[[], tuple[str, ...]] = tuple,
        mcp_tools: Callable[[str], tuple[str, ...]] = lambda _server: (),
        skills: Callable[[], tuple[str, ...]] = tuple,
    ):
        self.providers = providers
        self.models = models
        self.mcp_servers = mcp_servers
        self.mcp_connected_servers = mcp_connected_servers
        self.mcp_tools = mcp_tools
        self.skills = skills

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
        for command, values in (
            ("/model ", self.models),
            ("/provider ", self.providers),
            ("/reason ", lambda: REASONING_CHOICES),
            ("/effort ", lambda: REASONING_CHOICES),
            ("/api ", lambda: PROVIDER_API_CHOICES),
            ("/strict ", lambda: ("on", "off")),
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

        at_match = CommandCompleter.MCP_MENTION_RE.search(text)
        if at_match:
            server_part, dot, tool_part = at_match.group(1).partition(".")
            if dot:
                yield from self.matches(self.mcp_tools(server_part), tool_part)
            else:
                yield from self.matches(self.mcp_servers(), server_part)
            return

        skill_match = CommandCompleter.SKILL_MENTION_RE.search(text)
        if skill_match:
            yield from self.matches(self.skills(), skill_match.group(1))
            return

        if text.startswith("/") and " " not in text:
            yield from self.matches(CommandLoop.COMMANDS, text)

    @staticmethod
    def matches(values, prefix: str):
        return (Completion(value, start_position=-len(prefix)) for value in values if value.startswith(prefix))


class CommandLoop:
    """Own session behavior: read input, dispatch commands, drive turns, and route output.

    Slash commands are handled here and never reach the model. The agent runs on this thread while
    prompt-toolkit runs on another, which is why output has two destinations: completed user,
    assistant, and tool output goes to native scrollback, while drafts, previews, queue state, and
    selectors belong to the TUI. Anything transient the terminal leaves in scrollback is an artifact,
    not history — the transcript is always rebuilt from semantic records.

    Input entered mid-turn is queued, and only an allowlist of read-only commands may run against a
    busy session; anything that mutates configuration would change the meaning of a turn already in
    flight.

    The same object serves the non-interactive path, where there is no TUI and input and output are
    plain callables — which is also how the tests drive it.
    """

    HUNK_HEADER_RE: ClassVar[re.Pattern] = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
    HELP_HEADING_RE: ClassVar[re.Pattern] = re.compile(r"^### (.+)$", re.MULTILINE)
    HELP_ENTRY_RE: ClassVar[re.Pattern] = re.compile(r"^- (.+?) — ", re.MULTILINE)
    QUEUE_EMPTY_HINT = "Enter queues follow-up · Ctrl-C interrupts"
    QUEUE_PENDING_HINT = "↑ recalls queued · Ctrl-C interrupts"
    TRANSCRIPT_DIFF_LINES: ClassVar[int] = 40
    EDITOR_CONTEXT_MAX_LINES: ClassVar[int] = 200
    INPUT_HISTORY_BYTES: ClassVar[int] = 512 * 1024
    # fmt: off
    COMMAND_HANDLERS: ClassVar[dict[str, str]] = {
        "/help": "help", "/status": "status", "/ps": "ps_command", "/diff": "diff_command",
        "/skills": "skills_command", "/config": "config",
        "/compact": "compact", "/index": "index", "/provider": "provider", "/model": "model",
        "/reason": "reason", "/effort": "reason", "/api": "api", "/set": "set_value", "/yolo": "yolo", "/strict": "strict", "/hints": "hints",
        "/mcp": "mcp_command", "/resend": "resend_command", "/name": "name_command", "/sessions": "sessions_command", "/resume": "sessions_command",
    }
    COMMANDS: ClassVar[tuple[str, ...]] = tuple(COMMAND_HANDLERS) + ("/exit", "/quit")
    # fmt: on

    # Commands safe to run from the follow-up input while the agent works: read-only
    # views plus /yolo, whose single atomic flag flip the agent simply reads at the next approval.
    QUEUE_RUN_COMMANDS: ClassVar[frozenset[str]] = frozenset({"/help", "/status", "/skills", "/ps", "/mcp", "/diff", "/yolo", "/hints", "/resend"})
    MODEL_CONFIGURED_LABEL = "---- Configured models ----"
    MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
    MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))
    MCP_COMMANDS: ClassVar[dict[str, tuple[int, int, str]]] = {
        "connect": (1, sys.maxsize, "Usage: /mcp connect <server> [server ...]"),
        "disconnect": (1, 1, "Usage: /mcp disconnect <server>"),
        "tools": (0, 1, "Usage: /mcp tools [server]"),
    }
    MCP_HELP = "Try /mcp, /mcp connect <server> [server ...], /mcp disconnect <server>, or /mcp tools [server]"

    HELP = """### Commands

- `/help` — Show this help.
- `/status` — Show runtime status.
- `/ps` — Show active background jobs.
- `/diff` — Show latest edits and overall session diff.
- `/skills` — List installed skills (load with `Skill(name)` or reference inline with `$name`).
- `/config` — Show active config.
- `/compact` — Compact context now.
- `/name [TEXT]` — Name this session for later, or show the current name.
- `/sessions [all]` — Browse saved sessions and re-enter one (alias: `/resume`; `all` widens
  past this project).
- `/resend` — Resend the in-flight model request (type it while a turn is working).
- `/index [force]` — Sync or rebuild code symbol index.
- `/provider [NAME]` — Select or show the active provider.
- `/model [MODEL]` — Select or set the active model.
- `/reason [EFFORT]` — Select or set reasoning effort (alias: `/effort`).
- `/api [API]` — Select or set the request protocol used to reach the model.
- `/set KEY VALUE` — Set `provider.*` and `runtime.*`.
- `/yolo` — Toggle tool confirmations.
- `/hints` — Toggle next-step quick hints.
- `/strict` — Toggle strict tool-call schemas (OpenAI / DeepSeek).
- `/mcp` — Manage MCP server connections.
- `/exit`, `/quit` — Exit.

### Mentions

- `@server[.tool]` — Point the agent at an MCP server/tool in your message (tab-completes).
- `$skill` — Reference a skill in your message to load its instructions for that turn (tab-completes).

### CLI

- `-c`, `--last`, `--latest` — Resume the latest session in the current project.
- `--resume [UID]` — Resume a saved session by uid, name, or uid prefix; defaults to latest
  (`last` also works).

### Tools

Read, ViewImage, InspectCode, Search, Edit, Bash, Job, Recall, Note, Ask, MCP, Skill.

`Skill(name)` loads a skill's full instructions on demand (see the SKILLS section / `$skill`).
"""

    DIFF_MAX_BYTES: ClassVar[int] = 50_000
    DIFF_MAX_LINES: ClassVar[int] = 1_200

    @classmethod
    def bounded_diff(cls, text: str) -> tuple[str, bool]:
        if len(text.encode("utf-8")) <= cls.DIFF_MAX_BYTES and text.count("\n") <= cls.DIFF_MAX_LINES:
            return text, False
        clipped: list[str] = []
        length = 0
        for line in text.splitlines():
            line_bytes = len(line.encode("utf-8")) + 1
            if length + line_bytes > cls.DIFF_MAX_BYTES or len(clipped) >= cls.DIFF_MAX_LINES:
                break
            clipped.append(line)
            length += line_bytes
        return "\n".join(clipped), True

    @staticmethod
    def diff_counts(text: str) -> tuple[int, int]:
        added = removed = 0
        old_remaining = new_remaining = 0
        for line in text.splitlines():
            if match := CommandLoop.HUNK_HEADER_RE.match(line):
                old_remaining = int(match.group(1) or 1)
                new_remaining = int(match.group(2) or 1)
            elif line.startswith("+") and new_remaining:
                added += 1
                new_remaining -= 1
            elif line.startswith("-") and old_remaining:
                removed += 1
                old_remaining -= 1
            elif line.startswith(" "):
                old_remaining = max(0, old_remaining - 1)
                new_remaining = max(0, new_remaining - 1)
        return added, removed

    def __init__(self, agent: Agent, input_fn=input, output_fn=print):
        self.agent = agent
        self.session = agent.session
        self._hint_picker = HintPicker()  # idle-placeholder tips; see minacode/hints.py
        self.input_fn = input_fn
        self.ui = UiPrinter(output_fn)
        self.status_bar = StatusBar(self.session)
        self.live_preview = BashLivePreview()
        self.model_stream_lock = threading.Lock()
        self.model_stream_kind = ""
        self.model_stream_text = ""
        self.model_stream_promoted_text = ""
        self.live_status_paused = False
        # Set to the uid this run should hand over to. `main` reads it after run() returns and
        # builds the next CommandLoop around that session.
        self.resume_request = ""
        self.background_output_lock = threading.Lock()
        self.background_output_open = True
        self.interactive_input = input_fn is input and sys.stdin.isatty()
        # Set by run_tui() while the full-TUI shell is active; tool_input reroutes through it so
        # approval prompts land in the same input widget the user is already typing in.
        self.tui: TuiApp | None = None
        if self.interactive_input:
            history_path = self.session.data_path("history.txt")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            self.trim_input_history(history_path)
            self.input_history = FileHistory(history_path)
        else:
            self.input_history = None
        self.input_completer = CommandCompleter(
            providers=lambda: tuple(sorted(self.session.config.providers)),
            models=lambda: self.session.config.provider.available_models,
            mcp_servers=lambda: tuple(config.name for config in self.session.mcp.parse_configs()) if self.session.mcp else (),
            mcp_connected_servers=lambda: (
                tuple(config.name for config in self.session.mcp.parse_configs() if self.session.mcp.connected(config.name)) if self.session.mcp else ()
            ),
            mcp_tools=lambda server: tuple(tool.name for tool in self.session.mcp.tools.get(server, [])) if self.session.mcp else (),
            skills=lambda: tuple(skill.name for skill in self.session.skills.all()) if self.session.skills else (),
        )
        self.agent.output_fn = self.agent_output
        self.agent.model.on_stream = self.model_stream_output
        self.agent.model.on_builtin_call = self.builtin_call_output
        self.agent.on_queue_flush = self.flush_queued_to_log
        self.agent.context.on_compaction = self.automatic_compaction_status
        self.agent.tools.output_fn = self.tool_output
        self.agent.tools.input_fn = self.tool_input
        self.agent.tools.live_start = self.tool_live_start
        self.agent.tools.live_output = self.tool_live_output
        self.agent.tools.question_fn = self.question_interaction

    def automatic_compaction_status(self, active: bool) -> None:
        """Show automatic context compaction as a distinct phase of the running turn."""
        if self.tui is not None:
            self.tui.set_running("compacting context" if active else "working")

    @classmethod
    def trim_input_history(cls, path: str) -> None:
        """Bound the input history file, which prompt_toolkit only ever appends to.

        Keeps the newest entries that fit in `INPUT_HISTORY_BYTES` and drops the rest. The cut is
        made at an entry header rather than at a byte offset, so what survives is always loadable:
        a header is written as "\n# <timestamp>\n" and content lines are "+"-prefixed, which is why
        a user line beginning with "#" cannot be mistaken for one. The replacement is atomic, so an
        interrupted trim cannot leave a truncated history behind, and every failure is ignored —
        recall is a convenience and must never keep the session from starting.
        """
        try:
            if os.path.getsize(path) <= cls.INPUT_HISTORY_BYTES:
                return
            with open(path, "rb") as file:
                file.seek(-cls.INPUT_HISTORY_BYTES, os.SEEK_END)
                tail = file.read()
            start = tail.find(b"\n# ")
            if start < 0:
                return  # a single entry larger than the budget; keep it rather than cut inside it
            temp = path + ".tmp"
            with open(temp, "wb") as file:
                file.write(tail[start + 1 :])
            os.replace(temp, path)
        except OSError:
            return

    def flush_queued_to_log(self, texts: list[str]) -> None:
        # Move flushed queued messages from the live activity region into terminal scrollback.
        texts = [text for text in texts if text.strip()]
        if not texts:
            return
        fragments: list[tuple[str, str]] = [("", "\n")]
        for index, text in enumerate(texts):
            if index:
                fragments.append(("", "\n"))
            fragments.extend([("class:prompt", UiPrinter.USER_LOG_PREFIX), (UiPrinter.user_log_style(), text), ("", "\n")])
        fragments.append(("", "\n"))
        print_formatted_text(FormattedText(fragments), style=self.style(), end="", flush=True)

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

    def waiting_pulse_fragments(self) -> StyleAndTextTuples:
        if self.session.state.current_model_call_started_at <= 0:
            return []
        # Triangular breath: 0 → 1 → 0 over WAITING_PULSE_PERIOD seconds, mapped onto the palette.
        phase = (time.monotonic() % self.WAITING_PULSE_PERIOD) / self.WAITING_PULSE_PERIOD
        intensity = 1.0 - abs(2.0 * phase - 1.0)
        idx = min(len(self.WAITING_PULSE_STYLES) - 1, int(intensity * len(self.WAITING_PULSE_STYLES)))
        return [(self.WAITING_PULSE_STYLES[idx], "● ")]

    # One cell per frame. A head that advances further than its own glow between redraws stops
    # reading as motion and starts reading as a dash blinking at scattered positions.
    QUEUE_SWEEP_CELLS_PER_SEC: ClassVar[float] = 1.0 / TuiApp.ANIMATION_INTERVAL
    # A comet: a soft head with a tail fading into the dim rule, by distance from the head. The ramp
    # is finer than one shade per cell, so a head between two cells lights both partially instead of
    # snapping onto the nearer one. The divider is only drawn while working; there is no idle look.
    GLOW_REACH: ClassVar[float] = 4.0
    GLOW_STEPS: ClassVar[int] = 12

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
        status = self.tui.status_label if self.tui is not None and self.tui.status_label else "working"
        if status == "working":
            retry_status = self.status_bar.retry_status()
            attempt_status = self.status_bar.model_attempt_status()
            with self.model_stream_lock:
                phase = self.model_stream_kind
            activity = retry_status or (
                ({"reasoning": "thinking", "output": "responding"}.get(phase, phase) or "working") + (" · " + attempt_status if attempt_status else "")
            )
            label = f"{activity} ({Text.elapsed_since(self.status_bar.started_at)})"
        else:
            label = status
        if queued:
            label = f"{label} [ {queued} queued ]"
        return self.sweep_divider_fragments(label, prefix=self.waiting_pulse_fragments())

    def followup_fragments(self) -> tuple[StyleAndTextTuples, StyleAndTextTuples]:
        with self.session._queue_lock:
            pending = list(self.session.pending_user_inputs)

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
        with self.live_preview.lock:
            lines = self.live_preview.frame_lines() if self.live_preview.active else []
        for line in lines:
            fragments.extend([("ansibrightblack", line), ("", "\n")])
        if lines:
            fragments.append(("", "\n"))
        fragments.extend(waiting)
        return fragments

    def model_stream_fragments(self) -> StyleAndTextTuples:
        with self.model_stream_lock:
            kind, text = self.model_stream_kind, self.model_stream_text
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
        if self.tui is None:
            return ""
        if self.tui.input_mode == "running":
            with self.session._queue_lock:
                has_pending = any(not item.inflight for item in self.session.pending_user_inputs)
            return self.QUEUE_PENDING_HINT if has_pending else self.QUEUE_EMPTY_HINT
        if self.tui.input_mode == "chat":
            return self._hint_picker.pick(self._hint_context(), self.session.state.round_count)
        return ""

    def _hint_context(self) -> HintContext:
        """Project the session into the small situation the hint mechanism selects on.

        round_count only advances at the start of the next turn, so at idle it still names the
        round that just finished; edited_round therefore clears on its own once a later round
        makes no edits.
        """
        session = self.session
        round_count = session.state.round_count
        edited = any((diff.round or diff.turn) == round_count for diff in session.turn_diffs)
        return HintContext(
            early=not session.tool_records,
            edited_round=round_count if edited else None,
            skills_available=bool(session.skills and session.skills.skills),
            mcp_connected=bool(session.mcp and session.mcp.tools),
            jobs_running=any(job.status == "running" for job in session.jobs.values()),
        )

    def editor_context(self) -> str:
        """The agent's most recent reply, restated as read-only reference inside the external
        editor (Ctrl-X Ctrl-E / Ctrl-G): the full-screen editor hides the terminal scrollback
        the reply is printed into. Long replies keep only their most recent lines so the
        editor's temp file stays small."""
        for message in reversed(self.session.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                lines = content.strip().splitlines()
                if len(lines) > self.EDITOR_CONTEXT_MAX_LINES:
                    drop = len(lines) - self.EDITOR_CONTEXT_MAX_LINES
                    lines = ["# [... earlier lines of the reply omitted ...]"] + lines[drop:]
                return "\n".join(lines)
        return ""

    def run_queued_command(self, text: str) -> None:
        """Dispatch a read-only slash command while an agent turn is running."""
        name = text.partition(" ")[0]
        if name not in self.QUEUE_RUN_COMMANDS:
            self.emit(f"{name} is unavailable while the agent is working; press Ctrl-C to run it.")
            return
        if name == "/mcp":
            sub = text.partition(" ")[2].split()
            if sub and sub[0] != "tools":
                self.emit("Only read-only /mcp (status, tools) is available while the agent is working.")
                return
        self.command(text)

    def take_pending_inputs(self) -> list[UserInput]:
        """Remove and return queued inputs that are not currently being flushed."""
        with self.session._queue_lock:
            texts = [item.user_input() for item in self.session.pending_user_inputs if not item.inflight]
            self.session.pending_user_inputs = [item for item in self.session.pending_user_inputs if item.inflight]
        return texts

    def recall_pending_input(self, on_inflight: Callable[[], None]) -> str | UserInput:
        """Move the newest queued input back to the editor, retrying if it was already claimed."""
        with self.session._queue_lock:
            item = next(reversed(self.session.pending_user_inputs), None)
            if item is None:
                return ""
            self.session.pending_user_inputs.remove(item)
            was_inflight = item.inflight
            if was_inflight:
                for pending_item in self.session.pending_user_inputs:
                    pending_item.inflight = False
        if was_inflight:
            on_inflight()
        self.session.images.retain(item.images)
        self.session.save_snapshot()
        return item.user_input()

    def run(self) -> int:
        # Interactive terminals use the full TUI; injected/non-TTY callers use the simple REPL.
        if self.interactive_input:
            return self.run_tui()
        self.session.settings.quick_hints = False  # the simple REPL has no hint UI; don't invite the model to offer them
        self.start_session()
        while True:
            try:
                entered = self.take_pending_inputs()
                initial_input = UserInput(
                    "\n".join(str(item) for item in entered),
                    tuple(image for item in entered for image in item.images),
                )
                user_input = self.read_input(initial_text=initial_input)
            except EOFError:
                self.emit(TurnBox.SEPARATOR)
                self.save_and_emit_resume()
                return 0
            except KeyboardInterrupt:
                continue
            if not user_input.strip():
                continue
            handled, exit_now = self.command(user_input.strip())
            if exit_now:
                return 0
            if handled:
                continue
            self.emit("")
            started = time.monotonic()
            malformed_tool_call = False
            try:
                self.status_bar.start()
                try:
                    answer = self.agent.run(user_input)
                except KeyboardInterrupt:
                    self.emit("Cancelled")
                    continue
                except MalformedToolCallError as error:
                    answer = str(error)
                    malformed_tool_call = True
                except MinacodeError as error:
                    answer = f"Error: {error}"
            finally:
                CodeIndex(self.session).update_pending_async()
                self.status_bar.stop()
            if self.ui.color and answer.strip():
                self.emit()
            self.ui.emit_answer(answer, rule=False)
            if footer := search_sources_footer(self.agent.turn_sources):
                self.ui.emit_answer(footer, rule=False)
            if not malformed_tool_call:
                self.ui.emit_turn_end(started)
            self.session.save_snapshot()

    def start_session(self) -> None:
        """Initialize output and background services shared by both command-loop frontends."""
        self.emit(f"minacode {__version__}. /help for commands.")
        UpdateChecker(self.session).start()
        if self.session.update.newer_than(__version__):
            self.emit(f"update available: {__version__} -> {self.session.update.latest}. upgrade with `{' '.join(UpdateChecker.upgrade_command())}`.")
        self.clean_expired_sessions_async()
        self.render_resumed_session()
        # Publish existing availability without scanning the working tree while the user is
        # starting to type. The bounded freshness check already runs after each completed turn.
        CodeIndex(self.session).status()
        # Discover auto_connect servers in the background so an unreachable one cannot block the
        # prompt for the discovery timeout; the tools index picks them up as they connect.
        mcp = self.session.mcp
        if mcp is not None:
            threading.Thread(target=mcp.discover_auto, name="mcp-discover", daemon=True).start()

    def clean_expired_sessions_async(self) -> None:
        """Sweep expired sessions off the startup path.

        The sweep stats every session file in every project directory. That is microseconds on a
        local disk, but a home directory on a network filesystem pays a round trip per file and can
        turn it into seconds — spent before the prompt accepts a keystroke. Nothing about a first
        keystroke depends on retention having run, so it runs on a daemon thread like the code
        index and MCP discovery beside it, and reports through the background channel that stays
        quiet once this loop no longer owns the terminal.
        """

        def sweep() -> None:
            with contextlib.suppress(Exception):
                removed = SessionSnapshotStore.clean_expired(self.session)
                if removed:
                    self.emit_background(self.expired_sessions_notice(removed))

        threading.Thread(target=sweep, name="session-cleanup", daemon=True).start()

    def expired_sessions_notice(self, removed: int) -> str:
        """Word the retention notice.

        Retention removes work the user cannot get back, so it is reported rather than done
        silently, and naming the setting turns the notice into the one moment that knob is worth
        knowing about.
        """
        days = self.session.settings.session_retention_days
        sessions = "session" if removed == 1 else "sessions"
        return f"removed {removed} saved {sessions} inactive for over {days} {'day' if days == 1 else 'days'} (runtime.session_retention_days)"

    def run_tui(self) -> int:
        return TuiRuntime(self).run()

    def render_resumed_session(self) -> None:
        # Transcript reconstruction owns historical call/result matching and ordering invariants.
        if not self.session.resumed:
            return
        self.session.resumed = False
        # The percent is derived, not persisted, so a resumed session carries a full history with a
        # zeroed reading. Recompute it now or the status bar reports 0% until the first turn.
        self.agent.context.update_current_tokens(SYSTEM_PROMPT)
        messages = [message for message in self.session.messages if not SessionSnapshotCodec.is_internal_message(message) and message.get("role") != "tool"]
        self.emit(f"Restored session: {self.session.uid}")
        if not messages:
            return
        diffs = {diff.key: diff.diff for diff in self.session.turn_diffs if diff.key and diff.diff}
        tool_record_index = 0
        for index, turn in enumerate(TurnBox.group(messages)):
            if index:
                self.emit("")
            for message in turn.messages:
                tool_record_index = self.render_transcript_message(message, tool_record_index, diffs)
        self.render_remaining_tool_records(tool_record_index, diffs)

    def render_transcript_message(self, message: Json, tool_record_index: int = 0, diffs: dict[str, str] | None = None) -> int:
        role = str(message.get("role") or "")
        content = ImageInputs.label_text(message).strip()
        raw_calls = message.get("tool_calls")
        has_tool_calls = isinstance(raw_calls, list) and bool(raw_calls)
        if role == "assistant" and content:
            self.ui.emit_answer(content, role=role, rule=False, indent=TurnBox.CONTENT_LEVEL if has_tool_calls else TurnBox.ROOT_LEVEL)
        if role == "assistant":
            return self.render_transcript_tool_calls(message, tool_record_index, diffs or {})
        if role == "user" and content and not ImageInputs.is_tool_observation(message):
            self.ui.emit_answer(content, role=role, rule=False)
        return tool_record_index

    def render_transcript_tool_calls(self, message: Json, tool_record_index: int, diffs: dict[str, str]) -> int:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            return tool_record_index
        for raw in raw_calls:
            call = self.transcript_tool_call(raw)
            if call is None:
                continue
            record, tool_record_index = self.transcript_tool_record(call, tool_record_index)
            self.emit_transcript_tool(call, record.key if record else "", diffs)
        return tool_record_index

    def render_remaining_tool_records(self, tool_record_index: int, diffs: dict[str, str]) -> None:
        for record in self.session.tool_records[tool_record_index:]:
            call = ToolCall(id="", name=record.name, args=record.args)
            self.emit_transcript_tool(call, record.key, diffs)

    def emit_transcript_tool(self, call: ToolCall, key: str, diffs: dict[str, str]) -> None:
        """An Edit shows the diff it made, the way it did when the edit ran live. Live, that preview
        comes from the approval block; here the stored diff text is the same string, so replaying it
        needs no reconstruction."""
        preview = diffs.get(key, "") if call.name == "Edit" else ""
        if not preview:
            self.emit(self.agent.tools.finish_display(call, key, "", failed=False))
            return
        # The preview block carries the call line, so the result collapses to its trailing marker
        # underneath it — the same nesting the live approval block produces.
        self.emit(self.transcript_edit_preview(call, preview))
        self.emit(self.agent.tools.finish_display(call, key, "", failed=False, d=ToolDisplay(nested_display=True)))

    def transcript_edit_preview(self, call: ToolCall, preview: str) -> LogBlock:
        tools = self.agent.tools
        lines = preview.rstrip().splitlines()
        # A long replay would bury the prompt under diffs, so each one is trimmed to a readable
        # window; `/diff` still holds the full text.
        hidden = max(0, len(lines) - self.TRANSCRIPT_DIFF_LINES)
        if hidden:
            lines = lines[: self.TRANSCRIPT_DIFF_LINES]
        children = [LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH)]
        children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in lines)
        if hidden:
            children.append(LogLine("", f"… {hidden} more lines, see /diff", LogRole.META, LogEdge.CONTINUE))
        return LogBlock.hierarchy(tools.log_root(tools.short_call(call), LogRole.AUTO, "", call), children)

    @staticmethod
    def transcript_tool_call(raw: object) -> ToolCall | None:
        if not isinstance(raw, dict):
            return None
        raw_function = raw.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        name = str(function.get("name") or "")
        if not name:
            return None
        arguments = function.get("arguments")
        try:
            # strict=False tolerates literal newlines in argument strings (e.g. multi-line
            # git commit messages) that would otherwise be rejected as invalid JSON.
            payload = json.loads(arguments, strict=False) if isinstance(arguments, str) else (arguments or {})
        except json.JSONDecodeError:
            payload = {}
        try:
            args = ModelClient.tool_payload(name, payload)
        except ToolError:
            # A malformed historical call (e.g. tool args that fail validation) must not crash
            # the resume; render it without parsed args.
            args = [payload] if payload else []
        return ToolCall(id=str(raw.get("id") or ""), name=name, args=args)

    def transcript_tool_record(self, call: ToolCall, tool_record_index: int) -> tuple[ToolResultRecord | None, int]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is not None and not tool_class.STORES_RESULT:
            return None, tool_record_index
        records = self.session.tool_records
        while tool_record_index < len(records):
            record = records[tool_record_index]
            tool_record_index += 1
            if record.name == call.name:
                return record, tool_record_index
        return None, tool_record_index

    def save_and_emit_resume(self) -> None:
        uid = self.session.save_snapshot()
        if uid:
            # The name goes in the sentence, never in the command: the line below is meant to be
            # pasted, and only the uid is guaranteed to still mean this session tomorrow.
            name = self.session.name
            self.emit(f"Resume {name!r} with:\nminacode --resume {uid}" if name else f"Resume with:\nminacode --resume {uid}")

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

    def read_input(
        self,
        prompt_text: str = UiPrinter.PROMPT_PREFIX,
        *,
        initial_text: str = "",
    ) -> str:
        """Read from the injected/non-TTY input path; interactive terminals use TuiApp."""
        return initial_text or self.input_fn(prompt_text)

    def emit(self, text: str | LogBlock = "") -> None:
        self.ui.emit(text)

    def emit_background(self, text: str) -> None:
        """Emit from a daemon worker only while this loop still owns terminal output."""
        with self.background_output_lock:
            if self.background_output_open:
                self.emit(text)

    def close_background_output(self, final_output: Callable[[], None] | None = None) -> None:
        with self.background_output_lock:
            self.background_output_open = False
            if final_output is not None:
                final_output()

    def with_status_paused(self, action):
        # Only quiet the standalone status-bar thread used by the simple/non-TTY path. The full TUI
        # renders status and output together, so it never needs this terminal-level coordination.
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.stop()
        try:
            return action()
        finally:
            if was_running:
                self.status_bar.start(reset=False)

    def tool_output(self, text: str | LogBlock = "") -> None:
        def output() -> None:
            if self.ui.color and (isinstance(text, str) or (text.items and isinstance(text.items[0], LogLine))):
                self.emit()
            self.emit(text)

        self.with_status_paused(output)

    def builtin_call_output(self, label: str, detail: str) -> None:
        """Log a tool the provider ran for itself, so the transcript shows it like any other call.

        A provider-side search leaves no local tool call to log, and the running status label is gone
        the moment the turn ends. Without this line the transcript would credit the model with
        knowledge it went and looked up."""
        self.tool_output(LogBlock([LogLine(label, Text.clip_width(detail, 120), LogRole.TOOL, LogEdge.BRANCH)]))

    @staticmethod
    def unpromoted_text(text: str, promoted: str) -> str:
        """What is left to publish after an early promotion already wrote `promoted` to scrollback.

        A local tool call ends the response, so its promoted text is the whole of it. A provider-side
        tool runs inside the response and the model keeps writing afterwards, so there the promotion
        is only a prefix: re-emitting the whole text would repeat it, and skipping it would drop
        everything the model wrote after the search."""
        answer = text.strip()
        if promoted and answer.startswith(promoted):
            return answer[len(promoted) :].strip()
        return answer

    def agent_output(self, text: str = "") -> None:
        # An early promotion is presentation-only: Agent still publishes the same semantic text
        # after ModelClient returns. Consume the one-shot marker instead of printing it twice.
        with self.model_stream_lock:
            promoted = self.model_stream_promoted_text
            self.model_stream_promoted_text = ""
        if promoted:
            remaining = self.unpromoted_text(text, promoted)
            if not remaining:
                return
            text = remaining
        self.with_status_paused(lambda: self.emit_agent_output(text))

    def model_stream_output(self, kind: str, text: str) -> None:
        """Update the dim preview or permanently promote a protocol-complete response.

        `output_done` is internal and emitted only when ModelClient has seen both completed text and
        a tool call. The scrollback write is synchronous so prompt-toolkit cannot batch it with the
        immediately following ToolRunner output and leave the `responding` preview covering it.
        """
        promote = ""
        tui = self.tui
        if kind == "output_done" and self.session.has_inflight_user_inputs():
            # A request that carried live follow-ups logs them to scrollback only once it returns,
            # so promoting here would place the response above the message it answers. Leave the
            # preview standing and let the ordinary post-request output keep the transcript ordered.
            return
        with self.model_stream_lock:
            if kind == "output_done":
                promote = text.strip()
                self.model_stream_kind = self.model_stream_text = ""
                if promote and tui is not None:
                    self.model_stream_promoted_text = promote
            elif not kind:
                self.model_stream_kind = self.model_stream_text = ""
            elif not text:
                self.model_stream_kind, self.model_stream_text = kind, ""
            elif text:
                if kind != self.model_stream_kind:
                    self.model_stream_kind, self.model_stream_text = kind, ""
                self.model_stream_text = (self.model_stream_text + text)[-8000:]
        if tui is not None:
            tui.invalidate_frame()
            if promote:
                self.with_status_paused(lambda: tui.write_to_scrollback(lambda: self.emit_agent_output(promote)))

    def tool_input(self, prompt: str = "") -> str:
        # When the TUI is running, route agent approvals through TuiApp's
        # own input widget so the user answers inline in the persistent shell instead of a
        # separate pt Application (which would fail because pt does not nest).
        if self.tui is not None:
            return self.tui.request_input(prompt)

        return self.with_status_paused(lambda: self.input_fn(prompt))

    def emit_agent_output(self, text: str) -> None:
        if self.ui.color and text.strip():
            self.emit()
        self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)

    def _begin_cli_preview(self) -> None:
        """Pause the status bar if running and start the CLI Bash live-preview line."""
        self.live_status_paused = self.status_bar.is_running()
        if self.live_status_paused:
            self.status_bar.stop()
        self.live_preview.start()

    def tool_live_start(self) -> None:
        if not self.ui.color:
            return
        if self.tui is not None:
            with self.live_preview.lock:
                self.live_preview.active = True
                self.live_preview.text = ""
                self.live_preview.started_at = time.monotonic()
            self.tui.invalidate()
            return
        self._begin_cli_preview()

    def tool_live_output(self, _stream: str, text: str) -> None:
        if not self.ui.color:
            return
        if self.tui is not None:
            with self.live_preview.lock:
                if text:
                    self.live_preview.active = True
                    self.live_preview.text = (self.live_preview.text + text)[-self.live_preview.MAX_CHARS :]
                else:
                    self.live_preview.active = False
                    self.live_preview.text = ""
            self.tui.invalidate()
            return
        if text:
            if not self.live_preview.active:
                self._begin_cli_preview()
            self.live_preview.update(text)
            return
        if self.live_preview.active:
            self.live_preview.finish()
        if self.live_status_paused:
            self.status_bar.start(reset=False)
            self.live_status_paused = False

    def command(self, text: str) -> tuple[bool, bool]:
        if text in {"/exit", "/quit", "exit", "quit"}:
            self.save_and_emit_resume()
            return True, True
        if not text.startswith("/"):
            return False, False
        name, _, args = text.partition(" ")
        method_name = self.COMMAND_HANDLERS.get(name)
        handler = getattr(self, method_name, None) if method_name else None
        output = handler(args.strip()) if handler else f"Unknown command: {name}"
        # A None result means the handler already rendered its own UI (e.g. /diff's viewer).
        if output is not None:
            if name == "/status":
                self.ui.emit_answer(output, rule=False)
            else:
                (self.ui.emit_answer if name in {"/help", "/ps", "/mcp", "/skills", "/diff"} else self.emit)(output)
        # A handler that asked to switch sessions ends this run the way /exit does; `main` starts
        # the next one on the session it named.
        return True, bool(self.resume_request)

    def resend_command(self, _args: str) -> str | None:
        """Resend the in-flight model request. Available only in the running queue-input region:
        typed while a turn works, it re-requests the current model call (same path as on_retry)."""
        if self.tui is None or self.tui.input_mode != "running":
            return "/resend re-requests the current model request — type it while a turn is working."
        if self.session.state.current_model_call_started_at <= 0:
            return "Nothing to resend right now; /resend works while the model is generating."
        self.tui.on_retry()
        return None

    def mcp_command(self, args: str) -> str | None:
        mcp = self.session.mcp
        if mcp is None:
            return "MCP not configured"

        parts = args.split()
        if not parts:
            if self.tui is not None and self.tui.input_mode != "running":
                return self.mcp_manager()
            return mcp.render_server_status()

        sub = parts[0]
        rest = parts[1:]
        command = self.MCP_COMMANDS.get(sub)
        if command is None:
            return f"Unknown /mcp subcommand: {sub}. {self.MCP_HELP}"
        min_args, max_args, usage = command
        if not min_args <= len(rest) <= max_args:
            return usage

        if sub == "connect":
            return mcp.connect_servers(rest, interactive=self.interactive_input, notify=self.emit)
        if sub == "disconnect":
            return mcp.disconnect_server(rest[0])
        if sub == "tools":
            return mcp.render_tool_listing(rest[0] if rest else None)
        raise AssertionError("unreachable MCP subcommand")

    def mcp_manager(self) -> None:
        mcp = self.session.mcp
        tui = self.tui
        if mcp is None or tui is None:
            return
        configs = tuple(mcp.parse_configs())
        if not configs:
            self.ui.emit_answer(mcp.render_server_status())
            return

        state = ChoiceViewState(tuple(config.name for config in configs), {}, set())
        transitions: dict[str, str] = {}
        errors: dict[str, str] = {}
        state_lock = threading.Lock()
        modal_open = threading.Event()
        modal_open.set()

        def server_labels() -> dict[str, str]:
            with state_lock:
                changing = dict(transitions)
                failed = dict(errors)
            server_rows = []
            for config in configs:
                if transition := changing.get(config.name):
                    status = mcp.STATUS_MARKER + " " + transition
                elif config.name in failed:
                    status = mcp.STATUS_MARKER + " error"
                elif issue := mcp.server_issue(config.name):
                    status = mcp.STATUS_MARKER + " " + issue[0]
                elif mcp.connected(config.name):
                    status = mcp.STATUS_MARKER + " connected"
                else:
                    status = mcp.STATUS_MARKER + " disconnected"
                mode = "auto" if config.auto_connect else "manual"
                count = len(mcp.tools.get(config.name, []))
                server_rows.append((config.name, status, mode, count))
            name_width = max(len(name) for name, *_rest in server_rows)
            status_width = max(len(mcp.STATUS_MARKER + " disconnecting"), *(len(status) for _name, status, _mode, _count in server_rows))
            return {name: f"{name:<{name_width}}  {status:<{status_width}}  {mode:<6}  {count:>3} tools" for name, status, mode, count in server_rows}

        def preview(name: str) -> str:
            with state_lock:
                if message := errors.get(name):
                    return message
            if issue := mcp.server_issue(name):
                return issue[1]
            return ""

        def fragments() -> StyleAndTextTuples:
            state.labels = server_labels()
            return state.fragments("MCP servers · Enter toggles connection", preview)

        def toggle(name: str, connect: bool) -> None:
            try:
                if connect:
                    result = mcp.connect_server(name, interactive=True, notify=self.emit)
                else:
                    result = mcp.disconnect_server(name)
            except Exception as error:  # noqa: BLE001 - keep background MCP failures visible in the selector.
                result = f"MCP server error: {name}: {error}"

            succeeded = mcp.connected(name) == connect
            with state_lock:
                transitions.pop(name, None)
                if succeeded:
                    errors.pop(name, None)
                else:
                    errors[name] = result
            if modal_open.is_set():
                tui.invalidate()
            else:
                self.emit_background(result)

        def handle_key(key: str, data: str = "") -> Any:
            result = state.handle_key(key, data)
            if not isinstance(result, str):
                return result
            with state_lock:
                if result in transitions:
                    return TUI_MODAL_PENDING
                connect = not mcp.connected(result)
                errors.pop(result, None)
                transitions[result] = "connecting" if connect else "disconnecting"
            threading.Thread(target=toggle, args=(result, connect), name="mcp-toggle-" + result, daemon=True).start()
            return TUI_MODAL_PENDING

        try:
            tui.show_modal(fragments, handle_key)
        finally:
            modal_open.clear()

    def select_choice(
        self,
        title: str,
        choices: tuple[str, ...],
        *,
        labels: dict[str, str] | None = None,
        current: str = "",
        disabled: set[str] | frozenset[str] = frozenset(),
    ) -> str | object | None:
        labels = labels or {}
        if not choices or not self.interactive_input:
            return None
        enabled = tuple(choice for choice in choices if choice not in disabled)
        if len(enabled) == 1:
            return enabled[0]
        try:
            return self.choice_application(title, choices, labels, current, set(disabled))
        except (EOFError, KeyboardInterrupt):
            self.emit("Cancelled")
            return None

    def choice_application(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str],
        current: str,
        disabled: set[str],
        *,
        preview_fn: Callable[[str], str] | None = None,
        free_text: bool = False,
    ) -> str | object | None:
        if free_text and self.interactive_input:
            choices = (*choices, ChoiceViewState.FREE_TEXT)
            labels = {**labels, ChoiceViewState.FREE_TEXT: "Type freely..."}
        state = ChoiceViewState(choices, labels, disabled)
        options = state.enabled()
        state.selected = options.index(current) if current in options else 0
        if self.tui is None:
            return None
        result = self.tui.show_modal(lambda: state.fragments(title, preview_fn), state.handle_key)
        if isinstance(result, KeyboardInterrupt):
            raise result
        return result

    def question_application(self, spec: AskSpec, position: str = "") -> str:
        """Ask via the shared choice selector, with dynamic previews and a free-text fallback."""
        choices = spec.choices
        # Prefix the position (e.g. "(1/3) ...") into the question text so it renders as plain
        # markdown — no separate styled line, hence no ANSI escapes to mangle.
        prompt = f"({position}) {spec.question}" if position else spec.question
        if not choices:
            return self.tui.request_input("\n" + prompt) if self.tui is not None else self.read_input("\n" + prompt)
        if not self.interactive_input:
            return self.read_input("\n" + prompt)

        # Blank separator line before each question so multi-question prompts don't run together.
        if self.ui.color:
            self.emit("")
            self.ui.emit_markdown(prompt)
        else:
            self.emit("\n" + prompt + "\n")

        # An optional recommended choice is pre-selected (via current) and marked (via labels),
        # reusing the selector's existing machinery.
        labels, current = {}, ""
        if spec.recommended is not None and 0 <= spec.recommended < len(choices):
            current = choices[spec.recommended]
            labels = {current: current + " (recommended)"}
        previews = spec.previews
        preview_map = {c: previews[i] for i, c in enumerate(choices) if previews and i < len(previews) and previews[i]}
        result = self.choice_application(
            "Select:",
            tuple(choices),
            labels,
            current,
            set(),
            preview_fn=lambda choice: preview_map.get(choice, ""),
            free_text=True,
        )
        if result is SELECTION_FREE_TEXT:
            # The question was already rendered before the choice selector; do not repeat a long
            # raw prompt when the user switches to free text.
            self.emit("")
            return self.tui.request_input("> ") if self.tui is not None else self.read_input("> ")
        if isinstance(result, str):
            return result
        return DISMISSED  # SELECTION_BACK (Esc) — user declined to answer

    def question_interaction(self, spec: AskSpec, position: str = "") -> str:
        """Entry point for Ask; the final tool log renders the returned answer."""
        return self.question_application(spec, position)

    def select_reasoning(self) -> str | object | None:
        current = self.session.config.provider.reasoning
        labels = {"off": "off - disable reasoning"}
        labels[current] = labels.get(current, current) + " (current)"
        return self.select_choice("Reasoning effort", REASONING_CHOICES, labels=labels, current=current)

    def select_api(self, model: str) -> str | object | None:
        # An endpoint that lists several model families rarely serves them all over one protocol, and
        # a /models listing does not say which. Confirm the wire alongside the model that needs it.
        provider = self.session.config.provider
        current = provider.api
        inferred = replace(provider, api="auto", model=model).resolve().api
        labels = {"auto": f"auto - infer from the endpoint URL and model ({inferred})"}
        labels[current] = labels.get(current, current) + " (current)"
        return self.select_choice("Request API", PROVIDER_API_CHOICES, labels=labels, current=current)

    def help(self, args: str) -> str:
        text = self.HELP.rstrip()
        if self.ui.color:
            return text
        text = text.replace("`", "")
        text = self.HELP_HEADING_RE.sub(r"\1:", text)
        return self.HELP_ENTRY_RE.sub(r"  \1  ", text)

    def status(self, args: str) -> str:
        def progress_bar(value: int, total: int, width: int = 14) -> str:
            ratio = min(1.0, max(0.0, value / total)) if total else 0.0
            eighths = int(ratio * width * 8 + 0.5)
            full, partial = divmod(eighths, 8)
            partials = "▏▎▍▌▋▊▉"
            return "[" + "█" * full + (partials[partial - 1] if partial else "") + "░" * (width - full - bool(partial)) + "]"

        def token_count(value: int) -> str:
            if value >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            if value >= 1_000:
                return f"{value / 1_000:.1f}K"
            return str(value)

        usage = self.session.usage
        provider = self.session.config.provider
        resolved = provider.resolve()
        context_tokens = self.agent.context.update_current_tokens(SYSTEM_PROMPT)
        context_budget = self.agent.context.request_token_budget()
        index = CodeIndex(self.session)
        index_status, index_message = index.status(check=False)
        index.update_pending_async()
        if self.session.state.code_index_refreshing:
            index_status, index_message = self.session.state.code_index_notice or "syncing", ""
        elif self.session.state.code_index_error:
            index_status, index_message = "error", self.session.state.code_index_error
        if index_status in {"missing", "unavailable", "error"} and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index"
        elif index_status == "stale" and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index or wait for auto update"
        cache_ratio = (usage.cached_prompt_tokens * 100 / usage.prompt_tokens) if usage.prompt_tokens else 0
        last_cache_ratio = (usage.last_cached_prompt_tokens * 100 / usage.last_prompt_tokens) if usage.last_prompt_tokens else 0
        connected_mcp = sum(self.session.mcp.connected(config.name) for config in self.session.mcp.parse_configs()) if self.session.mcp else 0
        activity: list[tuple[str, int | str]] = [
            ("history", len(self.session.messages)),
            ("turn", self.session.state.turn_messages),
            ("tools", len(self.session.tool_results)),
            ("mcp", connected_mcp),
            ("skills", len(self.session.skills.skills) if self.session.skills else 0),
            ("known", len(self.session.state.known)),
            ("compactions", self.session.state.compaction_count),
        ]
        running_jobs = len(self.session.running_jobs())
        if self.session.jobs:
            activity.append(("jobs", f"{running_jobs}/{len(self.session.jobs)}"))
        rows = [
            ("workspace", "`" + self.session.cwd + "`"),
            ("session", "`" + self.session.uid + "`"),
            (
                "model",
                f"`{self.session.config.active_provider}/{provider.model or '(empty)'}`; api `{resolved.api}`; reasoning `{provider.reasoning}`",
            ),
            (
                "context",
                f"`{progress_bar(context_tokens, context_budget)}` `~{token_count(context_tokens)} / {token_count(context_budget)}` (`{self.session.state.context_percent}%`)",
            ),
            (
                "cache",
                f"`{progress_bar(usage.last_cached_prompt_tokens, usage.last_prompt_tokens)}` last read `{token_count(usage.last_cached_prompt_tokens)} / {token_count(usage.last_prompt_tokens)} ({last_cache_ratio:.1f}%)`, write `{token_count(usage.last_cache_write_prompt_tokens)}`; "
                f"session read `{token_count(usage.cached_prompt_tokens)} / {token_count(usage.prompt_tokens)} ({cache_ratio:.1f}%)`, write `{token_count(usage.cache_write_prompt_tokens)}`"
                if usage.prompt_tokens
                else "(no requests yet)",
            ),
        ]
        if self.session.state.goal:
            rows.append(("goal", self.session.state.goal))
        visible_activity = [(name, value) for name, value in activity if value]
        if visible_activity:
            rows.append(("activity", "; ".join(f"{name} `{value}`" for name, value in visible_activity)))
        if usage.calls:
            rows.append(("usage", f"calls `{usage.calls}`; total `{token_count(usage.total_tokens)}`"))
        runtime = [
            f"yolo {'on' if self.session.settings.yolo else 'off'}",
            f"steps {self.session.settings.max_steps}",
            CodeIndex.status_line(index_status, index_message),
        ]
        update = UpdateChecker(self.session).status_line().removeprefix("update: ")
        if update not in {"current", "unknown"}:
            runtime.append("update " + update)
        rows.append(("runtime", "; ".join(f"`{value}`" for value in runtime)))
        return "\n".join(
            [
                "| status | value |",
                "| --- | --- |",
                *(f"| {name} | {Text.clean(str(value)).replace(chr(10), ' ').replace('|', chr(92) + '|')} |" for name, value in rows),
            ]
        )

    def skills_command(self, args: str) -> str:
        library = self.session.skills
        skills = library.all() if library else []
        if not skills:
            return "No skills installed. Add `<name>/SKILL.md` under `.minacode/skills/` (project) or `~/.minacode/skills/` (user)."
        table = markdown_table(
            ["skill", "source", "description"],
            [(f"`{skill.name}`", skill.source, skill.description or "(no description)") for skill in skills],
        )
        return "\n".join([f"### Skills · {len(skills)}", "", "Load with `Skill(name)` or reference inline with `$name`.", "", table])

    def ps_command(self, args: str) -> str:
        if args.strip():
            return "Usage: /ps"
        running = self.session.running_jobs()
        if not running:
            total = len(self.session.jobs)
            return f"No active jobs ({total} total)."
        rows = [(job.id, job.status, f"{job.elapsed():.1f}s", job.command[:80]) for job in running]
        table = markdown_table(["id", "status", "elapsed", "command"], rows)
        return f"### Active jobs · {len(running)}\n\n{table}"

    def bash_output_viewer(self) -> None:
        """Browse recent completed Bash previews without copying them into scrollback."""
        if self.tui is None:
            return
        records = []
        for record in reversed(self.session.tool_records):
            if record.name != "Bash":
                continue
            preview = self.agent.tools.bash_result_preview(record.output)
            if preview:
                records.append((record, preview))
            if len(records) == 10:
                break
        if not records:
            return
        width = max(20, shutil.get_terminal_size((120, 20)).columns - 12)
        labels = {}
        calls = {}
        for index, (record, _preview) in enumerate(records):
            call = self.agent.tools.short_call(ToolCall("", "Bash", record.args))
            choice = str(index)
            calls[choice] = call
            labels[choice] = Text.clip_width(f"{record.key}  {call}", width)
        choices = tuple(labels)
        state = ChoiceViewState(choices, labels, set())
        opened: str | None = None

        def rule(label: str) -> StyleAndTextTuples:
            cols = shutil.get_terminal_size((80, 20)).columns
            rule_width = max(20, min(72, cols - 2))
            lead = "──── "
            trail = " " + "─" * max(3, rule_width - get_cwidth(lead + label) - 1)
            return [("", "\n"), ("class:choice.disabled", lead + label + trail + "\n")]

        def fragments() -> StyleAndTextTuples:
            if opened is None:
                list_fragments = state.fragments("")
                return [*rule(f"Bash outputs · latest {len(records)}"), *list_fragments[1:]]
            record, preview = records[int(opened)]
            detail_width = max(20, shutil.get_terminal_size((120, 20)).columns - 6)
            parts: StyleAndTextTuples = [*rule(f"Bash output · {record.key}"), ("ansibrightblack", f"  {Text.clip_width(calls[opened], detail_width)}\n\n")]
            parts.extend(("ansibrightblack", f"  {Text.clip_width(line, detail_width)}\n") for line in preview.splitlines())
            parts.append(("class:choice.disabled", "\n  Esc / ← back · Ctrl-O / q closes\n"))
            return parts

        def handle_key(key: str, data: str) -> Any:
            nonlocal opened
            if key in {"c-o", "q"}:
                return None
            if opened is not None:
                if key in {"escape", "left", "h"}:
                    opened = None
                return TUI_MODAL_PENDING
            result = state.handle_key(key, data)
            if result is SELECTION_BACK:
                return None
            if isinstance(result, str):
                opened = result
            return TUI_MODAL_PENDING

        self.tui.show_modal(fragments, handle_key)

    def diff_command(self, args: str) -> str | None:
        if args.strip():
            return "Usage: /diff"
        if self.interactive_input and self.ui.color and (self.tui is None or self.tui.alternate_screen_available()):
            self.diff_viewer()
            return None
        latest = self.agent.session.latest_round_diff_sections()
        session = self.agent.session.session_diff_sections()
        groups: list[tuple[str, list[tuple[str, str, str]]]] = []
        if latest is not None and latest[1]:
            round, sections = latest
            groups.append((f"Latest · Round {round}", sections))
        if session:
            groups.append(("Session", session))
        if not groups:
            return "No changes"
        lines: list[str] = []
        for title, sections in groups:
            lines.append("### " + title)
            for _status, path, diff in sections:
                lines.append(f"#### {path}")
                bounded, truncated = CommandLoop.bounded_diff(diff)
                lines.append(f"```diff\n{bounded}\n```")
                if truncated:
                    lines.append("\n*Diff truncated. Full edit output is stored in the session.*")
        return "\n".join(lines)

    def diff_viewer(self) -> None:
        """Interactive diff viewer. First shows a file list; open a file to see its diff.

        List mode: ↑/↓ or j/k move, h/l or ←/→ switches tabs, Enter opens the selected file,
        r refreshes, q/Esc closes.
        Diff mode: ↑/↓ scroll one line, Ctrl-U/Ctrl-D half a page, PgUp/PgDn a page,
        Esc/← returns to list, r refreshes, q closes.
        """
        state = DiffViewState(TabbedViewState(("Latest", "Session")))

        def build_model() -> list[list[tuple[str, str, str]]]:
            latest = self.agent.session.latest_round_diff_sections()
            return [latest[1] if latest is not None else [], self.agent.session.session_diff_sections()]

        model = build_model()

        def viewport() -> int:
            return max(3, shutil.get_terminal_size().lines - 7)

        def active_sections() -> list[tuple[str, str, str]]:
            return model[state.view.tab]

        def list_fragments(parts: StyleAndTextTuples, sections: list[tuple[str, str, str]]) -> None:
            parts.append(("", "\n"))
            counts = [CommandLoop.diff_counts(diff) for _status, _path, diff in sections]
            added_width = max(len(str(added)) for added, _removed in counts)
            removed_width = max(len(str(removed)) for _added, removed in counts)
            for index, ((_status, path, _diff), (added, removed)) in enumerate(zip(sections, counts)):
                selected = index == state.file
                marker = "> " if selected else "  "
                style = "ansicyan" if selected else "class:choice.disabled"
                parts.extend(
                    [
                        (style, marker),
                        ("ansigreen", f"+{added:>{added_width}}"),
                        ("", " "),
                        ("ansired", f"-{removed:>{removed_width}}"),
                        (style, f" {path}\n"),
                    ]
                )
            parts.append(("", "\n"))

        def file_fragments(parts: StyleAndTextTuples, sections: list[tuple[str, str, str]]) -> None:
            state.clamp_file(len(sections))
            status, path, diff = sections[state.file]
            parts.append(("", "\n"))
            parts.append(("ansicyan", f"  {status.title()} · {path}\n"))
            lines = self.ui.segment_lines(self.ui.diff_segments_live(diff))
            visible = state.view.visible(lines, viewport())
            for line in visible:
                parts.extend(line)
            if not visible or not visible[-1] or not visible[-1][-1][1].endswith("\n"):
                parts.append(("", "\n"))

        def fragments() -> StyleAndTextTuples:
            parts: StyleAndTextTuples = [("", "\n")]
            parts.extend(self.ui.tab_segments(state.view.titles, state.view.tab))
            parts.append(("", "\n"))

            sections = active_sections()
            if not sections:
                parts.append(("class:choice.disabled", "  No diffs\n"))
            elif state.mode is DiffViewState.Mode.LIST:
                list_fragments(parts, sections)
            else:
                file_fragments(parts, sections)
            mode_hint = "list" if state.mode is DiffViewState.Mode.LIST else "diff"
            if state.mode is DiffViewState.Mode.LIST:
                hint = "↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close"
            else:
                hint = "↑/↓ scroll · Ctrl-U/D half-page · PgUp/PgDn page · Esc/← back · r refresh · q close"
            position = f"{state.file + 1 if sections else 0}/{len(sections)}"
            parts.append(("class:choice.disabled", f"\n  [{mode_hint}] {hint} [{position}]\n"))
            return parts

        if self.tui is None:
            return

        def modal_key(key: str, _data: str) -> Any:
            nonlocal model
            result = state.handle_key(key, len(active_sections()), viewport())
            if result is DiffViewState.REFRESH:
                model = build_model()
                return TUI_MODAL_PENDING
            return result

        self.tui.show_modal(fragments, modal_key, exclusive=True)

    def config(self, args: str) -> str:
        provider = self.session.config.provider
        resolved = provider.resolve()
        configured_builtin_tools = ", ".join(str(entry.get("type") or "?") for entry in provider.builtin_tools) or "(off)"
        builtin_issue = builtin_tools_issue(resolved, provider.builtin_tools)
        if not provider.builtin_tools:
            resolved_builtin_tools = "(off)"
        elif builtin_issue is None:
            resolved_builtin_tools = "active: " + configured_builtin_tools
        elif builtin_issue.reason == "wire":
            resolved_builtin_tools = f"inactive on {resolved.api}: {configured_builtin_tools}"
        else:
            resolved_builtin_tools = "invalid: " + ", ".join(builtin_issue.configured)
        return "\n".join(
            [
                f"provider.active: {self.session.config.active_provider}",
                f"provider.available: {', '.join(sorted(self.session.config.providers))}",
                f"provider.url: {provider.url or '(empty)'}",
                f"provider.key: {'(set)' if provider.key else '(empty)'}",
                f"provider.model: {provider.model or '(empty)'}",
                f"provider.api: {provider.api}",
                f"provider.stream: {'on' if provider.stream else 'off'}",
                f"provider.image_input: {provider.image_input}",
                f"provider.resolved_api: {resolved.api}",
                f"provider.prompt_cache_key: {provider.prompt_cache_key}",
                f"provider.available_models: {', '.join(provider.available_models) or '(empty)'}",
                f"provider.reasoning: {provider.reasoning}",
                f"provider.resolved_reasoning_effort: {resolved.reasoning_effort or '(off)'}",
                f"provider.resolved_chat_reasoning: {resolved.chat_reasoning}",
                f"provider.chat_reasoning: {provider.chat_reasoning}",
                f"provider.temperature: {provider.temperature if provider.temperature is not None else '(off)'}",
                f"provider.max_tokens: {provider.max_tokens or '(server default)'}",
                f"provider.strict_tools: {provider.strict_tools} (active {resolved.strict_tools_active})",
                f"provider.extra_body: {json.dumps(provider.extra_body, ensure_ascii=False, sort_keys=True) if provider.extra_body else '(off)'}",
                f"provider.builtin_tools: {configured_builtin_tools}",
                f"provider.resolved_builtin_tools: {resolved_builtin_tools}",
                f"provider.timeout: {provider.timeout}",
                f"provider.response_timeout: {provider.response_timeout or '(off)'}",
                f"paths.data_dir: {self.session.data_path()}",
                f"runtime.shell_timeout: {self.session.settings.shell_timeout}",
                f"runtime.max_agent_steps: {self.session.settings.max_steps}",
                f"runtime.max_context_tokens: {self.session.settings.max_context_tokens}",
                f"runtime.max_parallel_tools: {self.session.settings.max_parallel_tools}",
                f"runtime.session_retention_days: {self.session.settings.session_retention_days}",
                f"runtime.yolo: {'on' if self.session.settings.yolo else 'off'}",
            ]
        )

    def sessions_command(self, args: str) -> str | None:
        """Browse saved sessions and re-enter one. `/sessions all` widens past this project."""
        argument = args.strip().lower()
        if argument not in {"", "all"}:
            return "Usage: /sessions [all]"
        entries = SessionSnapshotStore.list_sessions(self.session.config.data_dir, self.session.cwd, all_projects=argument == "all")
        if not entries:
            return "No saved sessions yet."
        labels = {entry.uid: self.session_label(entry, all_projects=argument == "all") for entry in entries}
        if self.tui is None or not self.interactive_input:
            return "\n".join(f"{entry.uid}  {labels[entry.uid]}" for entry in entries)
        title = "Sessions" + (" · all projects" if argument == "all" else "")
        # The preview renders on every frame, so it reads the list already in hand, never the store.
        by_uid = {entry.uid: entry for entry in entries}
        chosen = self.choice_application(
            title, tuple(entry.uid for entry in entries), labels, self.session.uid, set(), preview_fn=lambda uid: self.session_preview(by_uid.get(uid))
        )
        if not isinstance(chosen, str) or chosen == self.session.uid:
            return None
        self.resume_request = chosen
        self.save_and_emit_resume()
        return None

    def session_label(self, entry: SessionEntry, *, all_projects: bool = False) -> str:
        rounds = f"{entry.rounds} round" + ("s" if entry.rounds > 1 else "") if entry.rounds else "no turns"
        parts = [Text.age(time.time() - entry.updated_at), rounds]
        if all_projects and entry.cwd:
            parts.append(os.path.basename(entry.cwd.rstrip(os.sep)) or entry.cwd)
        if entry.uid == self.session.uid:
            parts.append("current")
        return f"{entry.label()}  ·  " + " · ".join(parts)

    def session_preview(self, entry: SessionEntry | None) -> str:
        if entry is None:
            return ""
        return "\n".join([f"uid   {entry.uid}", f"start {entry.opening or '(no message)'}", f"where {entry.cwd or '(unknown)'}"])

    def name_command(self, args: str) -> str:
        """Show or set the session's name, the label a later `--resume` can be given instead of a uid."""
        text = args.strip()
        if not text:
            current = self.session.name
            source = {"user": "set by you", "goal": "from the current goal", "input": "from the opening message"}
            described = source.get(self.session.state.name_source, "")
            return f"Session name: {current} ({described})" if current and described else f"Session name: {current or '(unnamed)'}"
        name = self.session.rename(text)
        self.session.save_snapshot()
        return f"Session named: {name}\nResume with: minacode --resume {shlex.quote(name)}"

    def compact(self, args: str) -> str:
        if args.strip():
            return "Usage: /compact"
        before = len(self.session.messages)
        compacted, keep = self.agent.context.compaction_parts()
        if not compacted:
            return "No prior conversation to compact"
        fallback = False
        self.status_bar.begin()
        if self.tui is not None:
            self.tui.set_running("compacting context")
        else:
            self.status_bar.start(reset=False)
        try:
            data = self.agent.model.compact(self.agent.context.compaction_input(compacted))
        except KeyboardInterrupt:
            return "Cancelled"
        except Exception:  # noqa: BLE001 - manual compaction uses the same deterministic fallback as automatic compaction.
            self.agent.context.apply_compaction(None, keep, fallback_note=PREVIOUS_CONTEXT_TRIMMED, compacted=compacted)
            fallback = True
            data = None
        finally:
            if self.tui is not None:
                self.tui.set_dispatching()
            else:
                self.status_bar.stop()
        if data is not None:
            self.agent.context.apply_compaction(data, keep, compacted=compacted)
        self.agent.context.update_current_tokens(SYSTEM_PROMPT)
        # Compaction rewrites the history in place. Persist it now: leaving the session without
        # running another turn would otherwise resume from the log's pre-compaction state.
        self.session.save_snapshot()
        fallback_note = " (fallback)" if fallback else ""
        return (
            f"Compacted context: messages {before} -> {len(self.session.messages)}, "
            f"prior summary inserted, ctx {self.session.state.context_percent}%{fallback_note}"
        )

    def index(self, args: str) -> str:
        value = args.strip()
        if value not in {"", "force"}:
            return "Usage: /index [force]"
        try:
            self.status_bar.start()
            return CodeIndex(self.session).sync(force=value == "force")
        finally:
            self.status_bar.stop()

    def provider(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /provider [NAME]"
        if parts:
            return self.set_provider(parts[0])
        choices = tuple(sorted(self.session.config.providers))
        summary = "provider: " + self.session.config.active_provider + "\nproviders: " + ", ".join(choices)
        current = self.session.config.active_provider
        choice = self.select_choice("Provider", choices, labels={current: current + " (current)"}, current=current)
        if not isinstance(choice, str):
            return "No change" if choice is SELECTION_BACK else summary
        provider_result = self.set_provider(choice)
        model_result = self.model("")
        return provider_result + ("\n" + model_result if model_result else "")

    def set_provider(self, name: str) -> str:
        if name not in self.session.config.providers:
            return "Unknown provider: " + name
        self.session.config.active_provider = name
        return "Set provider = " + name

    def model(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /model [MODEL]"
        if parts:
            result = self.set_model(parts[0])
            return "No change" if result is SELECTION_BACK else str(result)
        provider = self.session.config.provider
        configured = tuple(dict.fromkeys(provider.available_models))
        tui = self.tui
        show_loading = tui is not None and bool(provider.url and provider.key)
        if show_loading and tui is not None:
            tui.set_dispatching("Loading models...")
        try:
            remote = tuple(model for model in self.remote_models(provider) if model not in configured)
        finally:
            if show_loading and tui is not None:
                tui.set_dispatching()
        choices: list[str] = []
        if configured:
            choices.extend((self.MODEL_CONFIGURED_LABEL, *configured))
        if remote:
            choices.extend((self.MODEL_DISCOVERED_LABEL, *remote))
        choice_values = tuple(choices)
        if not choice_values:
            return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
        while True:
            current = self.session.config.provider.model
            labels = {label: label for label in self.MODEL_LABELS if label in choice_values}
            labels.update({current: current + " (current)"} if current in choice_values else {})
            choice = self.select_choice("Model", choice_values, labels=labels, current=current, disabled=self.MODEL_LABELS)
            if choice is SELECTION_BACK:
                return "No change"
            if not isinstance(choice, str):
                return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
            if choice in self.MODEL_LABELS:
                continue
            result = self.set_model(choice, back_to_model=True)
            if result is SELECTION_BACK:
                continue
            return str(result)

    def remote_models(self, provider: ProviderConfig) -> tuple[str, ...]:
        if not provider.url or not provider.key:
            return ()
        try:
            # lazy import: /model discovery is the only OpenAI use here, so the SDK stays off the startup path
            from openai import OpenAI

            page = OpenAI(
                api_key=provider.key,
                base_url=provider.resolve().base_url,
                timeout=min(provider.timeout, 10),
                max_retries=0,
                default_headers={"User-Agent": HTTP_USER_AGENT},
            ).models.list()
        except Exception:  # noqa: BLE001 - remote model discovery is optional and provider SDKs expose varied failures.
            return ()
        names = []
        for item in getattr(page, "data", page) or []:
            name = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(sorted(dict.fromkeys(names)))

    def set_model(self, model: str, *, back_to_model: bool = False) -> str | object:
        while True:
            api = self.select_api(model)
            if api is SELECTION_BACK:
                return SELECTION_BACK if back_to_model else "No change"
            reasoning = self.select_reasoning()
            if reasoning is not SELECTION_BACK:
                break
        provider = self.session.config.provider
        provider.model = model
        lines = ["Set provider.model = " + model]
        if isinstance(api, str):
            lines.append(self.set_api(api))
        if isinstance(reasoning, str):
            provider.reasoning = reasoning
            lines.append("Set provider.reasoning = " + reasoning)
        return "\n".join(lines)

    def reason(self, args: str) -> str:
        value = args.strip()
        if value:
            if value not in REASONING_CHOICES:
                return "Usage: /reason " + "|".join(REASONING_CHOICES)
            self.session.config.provider.reasoning = value
            return "Set provider.reasoning = " + value
        choice = self.select_reasoning()
        if isinstance(choice, str):
            self.session.config.provider.reasoning = choice
            return "Set provider.reasoning = " + choice
        return "No change"

    def api(self, args: str) -> str:
        value = args.strip()
        provider = self.session.config.provider
        if value:
            if value not in PROVIDER_API_CHOICES:
                return "Usage: /api " + "|".join(PROVIDER_API_CHOICES)
            return self.set_api(value)
        choice = self.select_api(provider.model)
        return self.set_api(choice) if isinstance(choice, str) else "No change"

    def set_api(self, value: str) -> str:
        provider = self.session.config.provider
        provider.api = value
        # "auto" is the usual choice, so name the wire it resolved to rather than echoing the setting back.
        resolved = provider.resolve()
        result = f"Set provider.api = {value} (wire: {resolved.api})"
        issue = builtin_tools_issue(resolved, provider.builtin_tools)
        if issue is not None:
            if issue.reason == "wire":
                result += f"; builtin_tools inactive on {resolved.api}"
            else:
                result += "; unsupported builtin_tools: " + ", ".join(issue.configured)
        return result

    def yolo(self, args: str) -> str:
        self.session.settings.yolo = not self.session.settings.yolo
        return "yolo: " + ("on" if self.session.settings.yolo else "off")

    def hints(self, args: str) -> str:
        self.session.settings.quick_hints = not self.session.settings.quick_hints
        return "quick hints: " + ("on" if self.session.settings.quick_hints else "off")

    def strict(self, args: str) -> str:
        if args:
            return "Usage: /strict"
        provider = self.session.config.provider
        provider.strict_tools = not provider.strict_tools
        state = "on" if provider.strict_tools else "off"
        if provider.strict_tools:
            resolved = provider.resolve()
            if not resolved.strict_tools_active:
                return f"strict_tools: {state} (inactive: {resolved.host or 'this provider'} does not support strict tool calling)"
        return f"strict_tools: {state}"

    def set_value(self, args: str) -> str:
        key, _, value = args.partition(" ")
        if not key or not value:
            return "Usage: /set KEY VALUE"
        handler = SET_HANDLERS.get(key)
        if handler is None:
            return "Unknown config key: " + key
        target_name, attr, coerce = handler
        choices = SET_CHOICES.get(key)
        if choices is not None and value not in choices:
            return "Invalid value for " + key
        obj = self.session.config.provider if target_name == "provider" else self.session.settings
        try:
            if coerce is not None:
                value = coerce(value)
            setattr(obj, attr, value)
        except (ConfigError, ValueError):
            return "Invalid value for " + key
        return "Set " + key


class TuiRuntime:
    """Own the interactive session timeline while CommandLoop owns session behavior."""

    def __init__(self, command_loop: CommandLoop):
        self.loop = command_loop
        self.pending: queue.Queue[UserInput] = queue.Queue()
        self.stop = threading.Event()
        self.cancel_pending = threading.Event()
        self.main_busy = threading.Event()
        self.force_exit_timer: threading.Timer | None = None
        self.error: BaseException | None = None

    @property
    def tui(self) -> TuiApp:
        assert self.loop.tui is not None
        return self.loop.tui

    def _interrupt_active(self, cancel: Callable[[], None]) -> None:
        threading.Thread(target=cancel, daemon=True).start()
        if self.main_busy.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    def interrupt(self) -> None:
        if self.cancel_pending.is_set():
            return
        self.cancel_pending.set()
        self.tui.set_running("cancelling")
        self._interrupt_active(self.loop.agent.cancel)

    def _request_model_retry(self) -> None:
        state = self.loop.session.state
        if state.current_model_call_started_at <= 0 or state.manual_model_retry_requested:
            return
        state.manual_model_retry_requested = True
        state.model_retry_count += 1
        self.tui.invalidate()
        self._interrupt_active(self.loop.agent.model.cancel)

    def submit_running(self, value: str | UserInput) -> None:
        value = value if isinstance(value, UserInput) else UserInput(value)
        text = str(value).strip()
        if not text:
            return
        if not value.images and "\n" not in text and text.startswith("/"):
            threading.Thread(target=self.loop.run_queued_command, args=(text,), daemon=True).start()
        else:
            self.loop.session.enqueue_user_input(value)
            self.loop.session.save_snapshot()
        self.tui.invalidate()

    def recall(self) -> str | UserInput:
        return self.loop.recall_pending_input(self._request_model_retry)

    def expand_output(self) -> None:
        threading.Thread(target=self.loop.bash_output_viewer, name="bash-output", daemon=True).start()

    def request_exit(self) -> None:
        self.stop.set()
        self.loop.save_and_emit_resume()

    def force_exit(self) -> None:
        self.stop.set()
        threading.Thread(target=self.loop.agent.cancel, daemon=True).start()
        self.force_exit_timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
        self.force_exit_timer.daemon = True
        self.force_exit_timer.start()
        os.kill(os.getpid(), signal.SIGINT)

    def build_tui(self) -> TuiApp:
        return TuiApp(
            on_chat_submit=self.pending.put,
            on_running_submit=self.submit_running,
            on_exit_request=self.request_exit,
            on_force_exit=self.force_exit,
            on_interrupt=self.interrupt,
            on_retry=self._request_model_retry,
            on_recall=self.recall,
            on_expand_output=self.expand_output,
            status_fragments_fn=lambda: self.loop.status_bar.display_fragments(active=self.tui.input_mode == "running"),
            activity_fragments_fn=self.loop.tui_activity_fragments,
            input_hint_fn=self.loop.tui_input_hint,
            quick_hints_fn=lambda: self.loop.session.quick_hints if self.loop.session.settings.quick_hints else (),
            editor_context_fn=self.loop.editor_context,
            images=self.loop.session.images,
            history=self.loop.input_history,
            completer=self.loop.input_completer,
        )

    def submit_next(self, entered: Sequence[str | UserInput]) -> None:
        if not entered:
            return
        first = entered[0] if isinstance(entered[0], UserInput) else UserInput(entered[0])
        self.pending.put(first)
        for text in entered[1:]:
            self.loop.session.enqueue_user_input(text)

    def reset_turn(self) -> None:
        self.loop.model_stream_output("", "")
        # A request can fail after permanent promotion but before Agent re-publishes the text and
        # consumes its marker. Never let that stale marker suppress an identical later response.
        with self.loop.model_stream_lock:
            self.loop.model_stream_promoted_text = ""
        self.tui.set_idle()
        self.cancel_pending.clear()
        self.main_busy.clear()

    def dispatch(self, user_input: str | UserInput) -> bool:
        """Dispatch one input. Return true when it was fully handled as a command."""
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.ui.emit_answer(user_input.display_text(), role="user", rule=False)
        try:
            handled, exit_now = self.loop.command(user_input.strip())
        except (KeyboardInterrupt, MinacodeError) as error:
            self.loop.emit("Cancelled" if isinstance(error, KeyboardInterrupt) else f"Error: {error}")
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        if exit_now:
            self.stop.set()
            self.main_busy.clear()
            self.tui.exit()
            return True
        if handled:
            # A command must not strand queued follow-ups: flush them as run_agent_turn does, so
            # they keep chaining once the command completes (e.g. /compact then queued input).
            # Submit before restoring the idle prompt, where newer input can enter `pending`.
            self.submit_next(self.loop.take_pending_inputs())
            self.reset_turn()
            return True
        return False

    def run_agent_turn(self, user_input: str | UserInput) -> None:
        user_input = user_input if isinstance(user_input, UserInput) else UserInput(user_input)
        self.loop.emit("")
        self.loop.status_bar.begin()
        self.tui.set_running("working")
        started = time.monotonic()
        cancelled = False
        malformed_tool_call = False
        promoted_answer = ""
        try:
            answer = self.loop.agent.run(user_input)
        except KeyboardInterrupt:
            self.submit_next(self.loop.take_pending_inputs())
            answer = ""
            cancelled = True
        except MalformedToolCallError as error:
            answer = str(error)
            malformed_tool_call = True
        except MinacodeError as error:
            answer = f"Error: {error}"
        finally:
            # Snapshot the stream-promotion marker before reset_turn clears it: a terminal NextHints
            # batch promotes its answer into scrollback like any tool batch, but nothing re-publishes
            # it through agent_output, so without this the final emit below would print it again.
            with self.loop.model_stream_lock:
                promoted_answer = self.loop.model_stream_promoted_text
            self.reset_turn()
            self.loop.session.state.manual_model_retry_requested = False
            CodeIndex(self.loop.session).update_pending_async()
        if cancelled:
            self.loop.emit("Cancelled")
            return
        if remaining := self.loop.unpromoted_text(answer, promoted_answer):
            if self.loop.ui.color:
                self.loop.emit()
            self.loop.ui.emit_answer(remaining, rule=False)
        # Emitted outside the promotion check: a promoted answer is already in scrollback without
        # its sources, so skipping the footer there would drop them exactly when a search ran.
        if footer := search_sources_footer(self.loop.agent.turn_sources):
            self.loop.ui.emit_answer(footer, rule=False)
        if not malformed_tool_call:
            self.loop.ui.emit_turn_end(started)
        self.loop.session.save_snapshot()
        self.submit_next(self.loop.take_pending_inputs())

    def run_agent_loop(self) -> None:
        while not self.stop.is_set():
            try:
                user_input = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            self.main_busy.set()
            self.loop.session.clear_quick_hints()  # the user acted; drop last turn's offerings (also covers slash commands, which skip Agent.run)
            if self.cancel_pending.is_set():
                self.loop.emit("Cancelled")
                self.reset_turn()
                continue
            if not self.dispatch(user_input):
                self.run_agent_turn(user_input)

    def run_tui_app(self) -> None:
        try:
            self.tui.run(style=self.loop.style())
        except BaseException as error:  # noqa: BLE001 - propagate every TUI-thread failure on the main thread.
            self.error = error
            self.stop.set()

    def run(self) -> int:
        """Run the agent on the main thread and prompt-toolkit on one joined UI thread."""
        self.loop.tui = self.build_tui()
        tui_thread = threading.Thread(target=self.run_tui_app, name="tui")
        tui_thread.start()
        try:
            self.tui.ready.wait()
            if self.error is not None:
                raise self.error
            # Emit startup and restored transcript lines only after patch_stdout owns the terminal,
            # so the primary-screen application places them in native terminal/tmux scrollback.
            self.loop.start_session()
            self.submit_next(self.loop.take_pending_inputs())
            self.run_agent_loop()
        finally:
            self.stop.set()
            if self.force_exit_timer is not None:
                self.force_exit_timer.cancel()
            self.tui.exit()
            # Do not let interpreter finalization race a TUI thread flushing stdout. The emergency
            # force-exit timer remains responsible for terminating a genuinely wedged application.
            tui_thread.join()
            try:
                self.loop.close_background_output()
            finally:
                self.loop.tui = None
        if self.error is not None:
            raise self.error
        return 0
