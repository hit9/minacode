"""minacode command loop and interactive session runtime."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from minacode.base import (
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    MalformedToolCallError,
    MinacodeError,
    Text,
    ToolCall,
    ToolError,
    TurnBox,
    __version__,
)
from minacode.cli import commands, worker
from minacode.cli.modals import approval_text_viewer, question_interaction
from minacode.cli.runtime import TuiRuntime
from minacode.cli.update import UpdateChecker
from minacode.cli.view import CommandCompleter, View
from minacode.engine import Agent
from minacode.image import ImageInputs, UserInput
from minacode.model import ModelClient
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.render import BashLivePreview, StatusBar, UiPrinter, search_sources_footer
from minacode.runner import ToolDisplay
from minacode.session import SessionSnapshotCodec, SessionSnapshotStore, ToolResultRecord
from minacode.tools import TOOL_REGISTRY, CodeIndex
from minacode.tui import TuiApp


@dataclass(frozen=True)
class Command:
    name: str  # "/status"
    # A LogBlock result is the structured form of `render="plain"`: it goes to the log renderer as
    # tool output does, so a handler with rows to show does not have to pre-format them as text.
    handler: Callable[[CommandLoop, str], str | LogBlock | None]
    aliases: tuple[str, ...] = ()
    queue_safe: bool = False  # may run from the follow-up input while a turn works
    render: str = "plain"  # "plain" | "answer" | "compact"


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
    TRANSCRIPT_DIFF_LINES: ClassVar[int] = 40
    # Resume redraws at most this many recent turns. A long session would otherwise flood the
    # terminal with the whole transcript and push the prompt out of reach; the earlier turns stay
    # in the session, so the next request still sees them.
    MAX_REDRAWN_TURNS: ClassVar[int] = 20
    EDITOR_CONTEXT_MAX_LINES: ClassVar[int] = 200
    EDITOR_CONTEXT_ELLIPSIS: ClassVar[str] = "# [... earlier lines of this reply omitted ...]"
    EDITOR_CONTEXT_SEPARATOR: ClassVar[str] = "# --- (earlier reply) ---"
    INPUT_HISTORY_BYTES: ClassVar[int] = 512 * 1024
    # The command registry (`COMMANDS` below, after the class) drives dispatch, the completer's
    # name tuple, and the queue-safe allowlist. `CommandLoop.COMMANDS` is derived from it and
    # assigned right after the registry.
    COMMANDS: ClassVar[tuple[str, ...]]

    HELP = """### Commands

- `/help` — Show this help.
- `/status` — Show runtime status.
- `/ps` — Show active background jobs.
- `/diff` — Show latest edits and overall session diff.
- `/skills` — List installed skills (load with `Skill(name)` or reference inline with `$name`).
- `/config` — Show active config.
- `/compact` — Compact context now; `/compact log [seg.N]` reviews what compaction evicted.
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
- `/language [NAME]` — Force or show the reply language; auto follows your messages.
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
        self.view = View(self)
        self.input_fn = input_fn
        self.ui = UiPrinter(output_fn)
        self.status_bar = StatusBar(self.session)
        self.live_preview = BashLivePreview()
        self.model_stream_lock = threading.Lock()
        self.model_stream_kind = ""
        self.model_stream_text = ""
        self.model_stream_promoted_text = ""
        self.live_status_paused = False
        self.compaction_active = False
        self.script_active = False
        # The source of the ToolScript body running right now, so Ctrl-O can offer it before it
        # finishes and becomes a stored record. Empty whenever no script is running.
        self.script_running_code = ""
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
            worker_models=lambda: tuple(
                dict.fromkeys(
                    (*self.session.config.providers[self.session.config.worker_provider or self.session.config.active_provider].available_models, "default")
                )
            ),
            mcp_servers=lambda: tuple(config.name for config in self.session.mcp.parse_configs()) if self.session.mcp else (),
            mcp_connected_servers=lambda: (
                tuple(config.name for config in self.session.mcp.parse_configs() if self.session.mcp.connected(config.name)) if self.session.mcp else ()
            ),
            mcp_tools=lambda server: tuple(tool.name for tool in self.session.mcp.tools.get(server, [])) if self.session.mcp else (),
            skills=lambda: tuple(skill.name for skill in self.session.skills.all()) if self.session.skills else (),
            file_matches=self.session.mentions.cached_matches if self.session.mentions else None,
        )
        self.agent.output_fn = self.agent_output
        self.agent.model.on_stream = self.model_stream_output
        self.agent.model.on_builtin_call = self.builtin_call_output
        self.agent.on_queue_flush = self.flush_queued_to_log
        self.agent.context.on_compaction = self.automatic_compaction_status
        self.agent.model.on_retry_wait = self.model_retry_wait_status
        self.agent.tools.output_fn = self.tool_output
        self.agent.tools.input_fn = self.tool_input
        self.agent.tools.live_start = self.tool_live_start
        self.agent.tools.live_output = self.tool_live_output
        self.agent.tools.model_stream = self.model_stream_output
        self.agent.tools.question_fn = lambda specs: question_interaction(self, specs)
        self.agent.tools.worker_rule = self.ui.emit_worker_rule
        self.agent.tools.worker_answer = self.worker_answer_output
        self.agent.tools.worker_config_picker = worker.WorkerFlow(self).run_worker_config
        self.agent.tools.text_viewer = lambda view: approval_text_viewer(self, view)
        self.agent.tools.approval_form = self.set_approval_form
        # Worker agent lifecycle callbacks: delegate.py wires these onto the worker agent when set,
        # so a worker's retry backoff, provider-side builtin calls, and compaction show in this TUI.
        self.agent.tools.retry_wait = self.model_retry_wait_status
        self.agent.tools.builtin_call = self.builtin_call_output
        self.agent.tools.compaction = self.automatic_compaction_status
        self.agent.tools.script_status = self.toolscript_run_status

    def automatic_compaction_status(self, active: bool, error: str = "") -> None:
        """Show automatic context compaction as a distinct phase of the running turn."""
        self.compaction_active = active
        self.set_running_phase()
        if error:
            self.tool_output(LogBlock([LogLine("compaction fallback", error, LogRole.ERROR, LogEdge.END)]))

    def model_retry_wait_status(self, active: bool) -> None:
        """Show a retry backoff wait as a distinct phase instead of claiming the agent is working."""
        self.set_running_phase(retrying=active)

    def toolscript_run_status(self, active: bool, code: str = "") -> None:
        """Show a running ToolScript body as a distinct phase of the turn.

        A script is the one stretch where the model is idle and no single tool line is pending, so
        without this the divider claims "working" from approval until the whole batch is done. The
        source is held for the same reason: a long batch is exactly when the reader wants to see
        what is running, and until it returns there is no stored record to open."""
        self.script_active = active
        self.script_running_code = code if active else ""
        self.set_running_phase()

    def set_running_phase(self, retrying: bool = False) -> None:
        """Put the running divider on the innermost phase currently active."""
        if self.tui is None:
            return
        self.tui.set_running(
            "retrying" if retrying else "compacting context" if self.compaction_active else "running script" if self.script_active else "working"
        )

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
        for i, text in enumerate(texts):
            if i:
                fragments.append(("", "\n"))
            fragments.extend([("class:prompt", UiPrinter.USER_LOG_PREFIX), (UiPrinter.user_log_style(), text), ("", "\n")])
        fragments.append(("", "\n"))
        print_formatted_text(FormattedText(fragments), style=self.view.style(), end="", flush=True)

    def editor_context(self) -> str:
        """The agent's recent replies, newest first, restated as read-only reference for the
        external editor (Ctrl-X Ctrl-E / Ctrl-G), accumulated under a line budget so the
        editor's temp file stays small."""
        parts: list[str] = []
        for message in reversed(self.session.messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            lines = content.strip().splitlines()
            if len(lines) > self.EDITOR_CONTEXT_MAX_LINES:
                # Keep the newest lines and say so: a headless reply that reads as complete is
                # worse reference than a shorter one that admits where it was cut.
                drop = len(lines) - self.EDITOR_CONTEXT_MAX_LINES + 1
                lines = [self.EDITOR_CONTEXT_ELLIPSIS] + lines[drop:]
            if parts and len(parts) + 1 + len(lines) > self.EDITOR_CONTEXT_MAX_LINES:
                break  # an earlier reply would push the line budget; it adds little recent context
            if parts:
                parts.append(self.EDITOR_CONTEXT_SEPARATOR)
            parts.extend(lines)
        if not parts:
            return ""
        return "\n".join(parts)

    def run_queued_command(self, text: str) -> None:
        """Dispatch a read-only slash command while an agent turn is running."""
        name = text.partition(" ")[0]
        entry = COMMAND_LOOKUP.get(name)
        if entry is None or not entry.queue_safe:
            self.emit_turn(f"{name} is unavailable while the agent is working; press Ctrl-C to run it.")
            return
        if name == "/mcp":
            sub = text.partition(" ")[2].split()
            if sub and sub[0] != "tools":
                self.emit_turn("Only read-only /mcp (status, tools) is available while the agent is working.")
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
            answered = False
            try:
                self.status_bar.start()
                try:
                    self.agent.run(user_input)
                    answered = True
                except KeyboardInterrupt:
                    self.emit_turn("Cancelled")
                    continue
                except MalformedToolCallError as error:
                    answer = str(error)
                    malformed_tool_call = True
                except MinacodeError as error:
                    answer = f"Error: {error}"
            finally:
                CodeIndex(self.session).update_pending_async()
                self.status_bar.stop()
            # Same rule as TuiRuntime.run_agent_turn: the engine publishes its own final answer
            # through output_fn, so only an error it raised before publishing prints here.
            if not answered:
                if self.ui.color and answer.strip():
                    self.emit()
                self.ui.emit_answer(answer, rule=False, indent=TurnBox.CONTENT_LEVEL)
            if footer := search_sources_footer(self.agent.turn_sources):
                self.ui.emit_answer(footer, rule=False, indent=TurnBox.CONTENT_LEVEL)
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
        # Publish existing availability without scanning the tree; the freshness check already
        # runs after each completed turn.
        CodeIndex(self.session).status()
        # Discover auto_connect servers in the background so an unreachable one cannot block the
        # prompt; the tools index picks them up as they connect.
        mcp = self.session.mcp
        if mcp is not None:
            threading.Thread(target=mcp.discover_auto, name="mcp-discover", daemon=True).start()

    def clean_expired_sessions_async(self) -> None:
        """Run the retention sweep off the startup path: on a network filesystem it can cost
        seconds before the prompt accepts a keystroke, and nothing depends on it having run first.
        Runs on a daemon thread and reports through the background channel."""

        def sweep() -> None:
            with contextlib.suppress(Exception):
                removed = SessionSnapshotStore.clean_expired(self.session)
                if removed:
                    self.emit_background(self.expired_sessions_notice(removed))

        threading.Thread(target=sweep, name="session-cleanup", daemon=True).start()

    def expired_sessions_notice(self, removed: int) -> str:
        """Word the retention notice: retention removes unrecoverable work, so report it rather
        than deleting silently, and name the setting that controls it."""
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
        # The percent is derived, not persisted; recompute it or the status bar reads 0% until
        # the first turn.
        self.agent.context.update_current_tokens(self.agent.session.system_prompt)
        transcript = self.session.transcript_messages or self.session.messages
        tool_results = {
            str(message.get("tool_call_id") or ""): message for message in transcript if message.get("role") == "tool" and message.get("tool_call_id")
        }
        semantic_tool_results = any("status" in message for message in tool_results.values())
        messages = [message for message in transcript if not SessionSnapshotCodec.is_internal_message(message) and message.get("role") != "tool"]
        self.emit(f"Restored session: {self.session.uid}")
        if self.session.transcript_incomplete:
            self.emit("Warning: this transcript may omit turns written by an older minacode version.")
        if not messages:
            return
        transcript_diffs = self.session.transcript_turn_diffs or self.session.turn_diffs
        diffs = {diff.key: diff.diff for diff in transcript_diffs if diff.key and diff.diff}
        tool_record_index = 0
        turns = TurnBox.group(messages)
        hidden = len(turns) - self.MAX_REDRAWN_TURNS
        if hidden > 0:
            # The earliest turns are not redrawn: on a long session they would flood the terminal
            # and the prompt would scroll out of reach. They stay in the session, so the next
            # request still sees them; only the redraw is skipped. Tool records still advance
            # through them so the visible turns pair with their own results.
            for turn in turns[:hidden]:
                for message in turn.messages:
                    tool_record_index = self.render_transcript_message(message, tool_record_index, diffs, tool_results, dry_run=True)
            self.emit(f"… {hidden} earlier turn{'s' if hidden > 1 else ''} not redrawn (still in context)")
            turns = turns[hidden:]
        for i, turn in enumerate(turns):
            if i:
                self.emit("")
            for message in turn.messages:
                tool_record_index = self.render_transcript_message(message, tool_record_index, diffs, tool_results)
        if not semantic_tool_results:
            self.render_remaining_tool_records(tool_record_index, diffs)

    def render_transcript_message(
        self,
        message: Json,
        tool_record_index: int = 0,
        diffs: dict[str, str] | None = None,
        tool_results: dict[str, Json] | None = None,
        *,
        dry_run: bool = False,
    ) -> int:
        role = str(message.get("role") or "")
        content = ImageInputs.label_text(message).strip()
        if role == "assistant" and content and not dry_run:
            # Every assistant message sits in the content column, final answer included, so a
            # resumed session reads exactly like the live one. The turn's own text all shares that
            # column with the user's message, whose `• ` bullet hangs in the same two-space margin.
            self.ui.emit_answer(content, role=role, rule=False, indent=TurnBox.CONTENT_LEVEL)
        if role == "assistant":
            return self.render_transcript_tool_calls(message, tool_record_index, diffs or {}, tool_results or {}, dry_run=dry_run)
        if role == "user" and content and not ImageInputs.is_tool_observation(message) and not dry_run:
            # The follow-up marker is model-facing context, part of history because it was sent.
            # The scrollback shows what the user typed, exactly as it looked when they typed it.
            self.ui.emit_answer(content.removeprefix(LIVE_FOLLOWUP_PREFIX.strip()).lstrip(), role=role, rule=False)
        return tool_record_index

    def render_transcript_tool_calls(
        self,
        message: Json,
        tool_record_index: int,
        diffs: dict[str, str],
        tool_results: dict[str, Json] | None = None,
        *,
        dry_run: bool = False,
    ) -> int:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            return tool_record_index
        for raw in raw_calls:
            call = self.transcript_tool_call(raw)
            if call is None:
                continue
            result = (tool_results or {}).get(call.id)
            if result is not None and "status" in result:
                if not dry_run:
                    self.emit_transcript_tool(call, str(result.get("result_key") or ""), diffs, failed=result.get("status") != "ok")
                continue
            record, tool_record_index = self.transcript_tool_record(call, tool_record_index)
            if not dry_run:
                self.emit_transcript_tool(call, record.key if record else "", diffs)
        return tool_record_index

    def render_remaining_tool_records(self, tool_record_index: int, diffs: dict[str, str]) -> None:
        records = self.session.transcript_tool_records or self.session.tool_records
        for record in records[tool_record_index:]:
            call = ToolCall(id="", name=record.name, args=record.args)
            self.emit_transcript_tool(call, record.key, diffs)

    def emit_transcript_tool(self, call: ToolCall, key: str, diffs: dict[str, str], *, failed: bool = False) -> None:
        """An Edit shows the diff it made, the way it did when the edit ran live. Live, that preview
        comes from the approval block; here the stored diff text is the same string, so replaying it
        needs no reconstruction."""
        preview = diffs.get(key, "") if call.name == "Edit" else ""
        if not preview:
            self.emit(self.agent.tools.finish_display(call, key, "failed in saved session" if failed else "", failed=failed))
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
        records = self.session.transcript_tool_records or self.session.tool_records
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

    def read_input(
        self,
        prompt_text: str = UiPrinter.PROMPT_PREFIX,
        *,
        initial_text: str = "",
    ) -> str:
        """Read from the injected/non-TTY input path; interactive terminals use TuiApp."""
        return initial_text or self.input_fn(prompt_text)

    def emit(self, text: str | LogBlock = "", indent: int = 0) -> None:
        self.ui.emit(text, indent)

    def emit_turn(self, text: str = "") -> None:
        """A line that belongs to the exchange rather than to the session around it: a turn
        outcome, a command's reply, a refusal to run one. Those sit in the content column with
        the model's text and the tool lines; session chrome (the banner, the restored-session
        notice, the resume line) stays at column 0 and frames them."""
        self.emit(text, TurnBox.CONTENT_LEVEL)

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

    def set_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        # The selectable action row exists only in the TUI. Headless and piped runs report False so
        # the approval brief keeps advertising the typed protocol they do have.
        return self.tui is not None and self.tui.set_approval_form(actions)

    def tool_input(self, prompt: str = "") -> str | None:
        # Under the TUI, route agent approvals through TuiApp's input widget instead of a separate
        # pt Application (pt does not nest). None propagates the TUI's cancel signal; the headless
        # `input` path can only return a string.
        if self.tui is not None:
            return self.tui.request_input(prompt)

        return self.with_status_paused(lambda: self.input_fn(prompt))

    def emit_agent_output(self, text: str) -> None:
        if self.ui.color and text.strip():
            self.emit()
        self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)

    def worker_answer_output(self, text: str) -> None:
        """The worker's final report, rendered like an agent answer (markdown) rather than the
        plain log lines its interim messages print as."""
        self.with_status_paused(lambda: self.emit_agent_output(text))

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
        entry = COMMAND_LOOKUP.get(name)
        output = entry.handler(self, args.strip()) if entry else f"Unknown command: {name}"
        # None means the handler already rendered its own UI (e.g. /diff's viewer).
        if output is not None:
            if isinstance(output, LogBlock):
                self.emit(output)
            elif entry is not None and entry.render == "compact":
                self.ui.emit_answer(output, rule=False, compact=True, indent=TurnBox.CONTENT_LEVEL)
            elif entry is not None and entry.render == "answer":
                self.ui.emit_answer(output, indent=TurnBox.CONTENT_LEVEL)
            else:
                self.emit_turn(output)
        # A session switch ends this run the way /exit does; `main` starts the next one.
        return True, bool(self.resume_request)


# fmt: off
COMMANDS: tuple[Command, ...] = (
    Command("/help", commands.help, render="answer"),
    Command("/status", commands.status, queue_safe=True, render="compact"),
    Command("/ps", commands.ps_command, queue_safe=True, render="answer"),
    Command("/diff", commands.diff_command, queue_safe=True, render="answer"),
    Command("/skills", commands.skills_command, queue_safe=True, render="answer"),
    Command("/config", commands.config),
    Command("/compact", commands.compact),
    Command("/index", commands.index),
    Command("/provider", commands.provider),
    Command("/model", commands.model),
    Command("/reason", commands.reason, aliases=("/effort",)),
    Command("/api", commands.api),
    Command("/set", commands.set_value),
    Command("/yolo", commands.yolo, queue_safe=True),
    Command("/strict", commands.strict),
    Command("/hints", commands.hints, queue_safe=True),
    Command("/mcp", commands.mcp_command, queue_safe=True, render="answer"),
    Command("/resend", commands.resend_command, queue_safe=True),
    Command("/name", commands.name_command),
    Command("/sessions", commands.sessions_command, aliases=("/resume",)),
    Command("/worker", worker.worker_command),
    Command("/language", commands.language_command),
)
# fmt: on

CommandLoop.COMMANDS = tuple(dict.fromkeys(name for command in COMMANDS for name in (command.name, *command.aliases))) + ("/exit", "/quit")
COMMAND_LOOKUP = {name: command for command in COMMANDS for name in (command.name, *command.aliases)}
QUEUE_SAFE_COMMANDS = frozenset(command.name for command in COMMANDS if command.queue_safe)
