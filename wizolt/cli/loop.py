"""The command loop: `Command`, `CommandLoop`, and the slash-command registry.

`wizolt.cli` re-exports these at the package root, so callers keep importing them from
`wizolt.cli`. The registry below drives dispatch, the completer's name tuple, and the
queue-safe allowlist.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, ClassVar

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from wizolt.base import (
    ImageRouteNotice,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    MalformedToolCallError,
    Text,
    ToolCall,
    ToolError,
    TurnBox,
    WizoltError,
    __version__,
    run_blocking,
)
from wizolt.cli import commands, worker
from wizolt.cli.modals import approval_text_viewer, question_interaction
from wizolt.cli.runtime import ScrollbackWriter, TuiRuntime
from wizolt.cli.update import UpdateChecker
from wizolt.cli.view import CommandCompleter, View
from wizolt.engine import Agent
from wizolt.image import ImageInputs, UserInput
from wizolt.mentions import FilePick
from wizolt.prompts import LIVE_FOLLOWUP_PREFIX
from wizolt.render import BashLivePreview, StatusBar, UiPrinter, search_sources_footer
from wizolt.session import SessionSnapshotCodec, SessionSnapshotStore, ToolResultRecord
from wizolt.tools import TOOL_REGISTRY, CodeIndex, tool_payload, toolblocks, tooloutput
from wizolt.tools.delegate import worker_provider_config
from wizolt.tools.toolblocks import ToolDisplay
from wizolt.tui import TuiApp


@dataclass(frozen=True)
class Command:
    name: str  # "/status"
    # A LogBlock result is the structured form of `render="plain"`: it goes to the log renderer as
    # tool output does, so a handler with rows to show does not have to pre-format them as text.
    # A handler that reaches the network returns an awaitable instead, which dispatch awaits on the
    # session's own loop; every other handler is local and bounded and returns its result directly.
    handler: Callable[[CommandLoop, str], str | LogBlock | None | Awaitable[str | LogBlock | None]]
    aliases: tuple[str, ...] = ()
    queue_safe: bool = False  # may run from the follow-up input while a turn works
    render: str = "plain"  # "plain" | "answer" | "compact"


class CommandLoop:
    """Own session behavior: read input, dispatch commands, drive turns, and route output.

    Slash commands are handled here and never reach the model. The agent and prompt-toolkit share
    the runtime loop; completed user, assistant, and tool output goes to native scrollback, while
    drafts, previews, queue state, and selectors belong to the TUI. Anything transient the terminal
    leaves in scrollback is an artifact, not history — the transcript is always rebuilt from
    semantic records.

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
- `/catalog [status|sync]` — Show the provider catalog in use, or force a sync.
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
- `/strict` — Toggle strict tool-call schemas where supported.
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

### Documentation

Full documentation: https://wizolt.readthedocs.io
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
        self.model_stream_kind = ""
        self.model_stream_text = ""
        self.model_stream_promoted_text = ""
        self.live_status_paused = False
        self.compaction_active = False
        self.script_active = False
        # Tool batches the agent worked through in silence since it last said anything. A long run
        # of silent batches closes with a phase rule; one narration resets the count.
        self._silent_batches = 0
        # The source of the ToolScript body running right now, so Ctrl-O can offer it before it
        # finishes and becomes a stored record. Empty whenever no script is running.
        self.script_running_code = ""
        # Set to the uid this run should hand over to. `main` reads it after run() returns and
        # builds the next CommandLoop around that session.
        self.resume_request = ""
        self.background_output_lock = threading.Lock()
        self.background_output_open = True
        # Session-scoped background work this loop owns: startup maintenance, mention discovery,
        # code-index freshness, the interactive picker. Loop-bound state, so it is opened once the
        # frontend has entered its loop and closed before that invocation returns -- a task must
        # never outlive the loop that can settle it, and none of it survives into a later run.
        self._background: set[asyncio.Task] = set()
        self._background_open = False
        # The one in-flight mention scan, coalescing every caller onto it. Kept here rather than on
        # FileMentions because a task is loop-bound and the session outlives loops.
        self._mention_refresh: asyncio.Task | None = None
        # The runtime's ordered scrollback queue while the TUI is live. Set by TuiRuntime, which
        # also lends its admission lock to the background-output gate above: closing that gate and
        # scheduling a write are the same race, so one lock decides both.
        self.scrollback: ScrollbackWriter | None = None
        self.interactive_input = input_fn is input and sys.stdin.isatty()
        # Bytes already read from the default non-TTY stdin after the first newline. Keeping the
        # remainder here lets the loop use non-blocking os.read() without losing a following line.
        self._stdin_buffer = bytearray()
        # Set by TuiRuntime while the full-TUI shell is active; tool_input reroutes through it so
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
            reasoning_choices=lambda: self.session.policy.reasoning_choices(self.session.config.provider),
            worker_reasoning_choices=lambda: self.session.policy.reasoning_choices(
                worker_provider_config(
                    self.session.config,
                    self.session.config.worker_provider or self.session.config.active_provider,
                )
            ),
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
        self.agent.final_output_fn = self.agent_answer_output
        self.agent.model.on_stream = self.model_stream_output
        self.agent.model.on_builtin_call = self.builtin_call_output
        self.agent.on_queue_flush = self.flush_queued_to_log
        self.agent.context.on_compaction = self.automatic_compaction_status
        self.agent.model.on_retry_wait = self.model_retry_wait_status
        self.agent.on_image_route_notice = self.image_route_notice
        self.agent.on_tool_batch = self.tool_batch_output
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
        self.agent.tools.cancel_input = self.cancel_tool_input
        # Worker agent lifecycle callbacks: delegate.py wires these onto the worker agent when set,
        # so a worker's retry backoff, provider-side builtin calls, and compaction show in this TUI.
        self.agent.tools.retry_wait = self.model_retry_wait_status
        self.agent.tools.builtin_call = self.builtin_call_output
        self.agent.tools.compaction = self.automatic_compaction_status
        self.agent.tools.script_status = self.toolscript_run_status

    def image_route_notice(self, notice: ImageRouteNotice) -> None:
        """Show the one gray routing notice for a text-only image delivery decision.

        The root line names the observation and carries the routing reason as a gray meta
        suffix; the described-by entry is its single child with a tree edge, mirroring the
        ViewImage tool's rendering. The engine emits this only after the observation
        succeeded, so a failed vision call never shows a fake described-by. Presentation
        only; never enters model context.
        """

        children = [LogLine("described by", notice.described_by, LogRole.TOOL, LogEdge.END)] if notice.described_by else []
        count = len(notice.images)
        label = "Image" if count <= 1 else "Images"
        text = notice.images[0] if count == 1 else (f"{count} attachments" if count else "")
        root = LogLine(label, text, LogRole.META, LogEdge.NONE, meta=" · " + notice.reason)
        self.tool_output(LogBlock.hierarchy(root, children))

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

    async def run_queued_command(self, text: str) -> None:
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
        await self.command(text)

    def take_pending_inputs(self) -> list[UserInput]:
        """Remove and return queued inputs that are not currently being flushed."""
        texts = [item.user_input() for item in self.session.pending_user_inputs if not item.inflight]
        self.session.pending_user_inputs = [item for item in self.session.pending_user_inputs if item.inflight]
        return texts

    def recall_pending_input(self, on_inflight: Callable[[], None]) -> str | UserInput:
        """Move the newest queued input back to the editor, retrying if it was already claimed.

        The mutation only; persisting it is the caller's, because this runs inside a prompt-toolkit
        key handler that has to answer with the recalled text and cannot await a file write."""

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
        return item.user_input()

    def run(self, *, show_banner: bool = True) -> int:
        """Synchronous entry point for the CLI. Both frontends run on one loop from here."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("CommandLoop.run() cannot be called from a running event loop; await the frontend coroutine")
        return asyncio.run(self._run_frontend(show_banner=show_banner))

    async def _run_frontend(self, *, show_banner: bool = True) -> int:
        """Select the frontend inside the CLI's single event-loop entry."""
        if self.interactive_input:
            # The primary-screen renderer can spend a second probing cursor position on a slow
            # terminal. Put the static banner in native scrollback before that probe; restored
            # history and every later write still wait for the TUI's ordered output path.
            if show_banner:
                self.emit_banner()
            return await TuiRuntime(self).run(show_banner=False)
        return await self.run_simple(show_banner=show_banner)

    async def run_simple(self, *, show_banner: bool = True) -> int:
        """The non-TTY frontend on the CLI-owned loop.

        The same loop as the TUI frontend gives the turn: one owner for startup discovery, the
        model client, and MCP, so everything this session opened is closed before it closes."""

        self.session.next_hints_available = False  # the simple REPL has no chip UI; don't offer an invisible terminal tool
        self.open_background()
        self.start_session(show_banner=show_banner)
        discovery = asyncio.ensure_future(self.discover_mcp())
        try:
            return await self._simple_loop()
        finally:
            discovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await discovery
            await self.close_background()
            await self.close_resources()

    def open_background(self) -> None:
        """Start admitting session-scoped background work on the loop that is now running."""
        self._background = set()
        self._background_open = True
        self._mention_refresh = None
        if self.session.mentions is not None:
            self.session.mentions.refresh_owner = self.refresh_mentions

    def schedule_index_freshness(self) -> None:
        """Admit the post-turn code-index check. Both frontends go through here.

        Never awaited on the answer path: the check walks and hashes the working tree, and a turn's
        answer must not wait behind it. `update_pending` coalesces repeated triggers itself."""

        self.spawn_background(CodeIndex(self.session).update_pending(), name="code-index-freshness")

    def refresh_mentions(self) -> asyncio.Task | None:
        """One mention-candidate scan at a time, owned here.

        Every caller -- the startup warm-up, the picker on a cold cache, a completion behind a
        keystroke -- joins the scan already running instead of starting a competing one. None means
        admission is closed, and the caller falls back to whatever snapshot it already has."""

        mentions = self.session.mentions
        if mentions is None:
            return None
        task = self._mention_refresh
        if task is not None and not task.done():
            return task
        self._mention_refresh = task = self.spawn_background(mentions.refresh(), name="mention-candidates")
        return task

    async def pick_file(self, query: str) -> FilePick:
        """Run the interactive picker as work owned and settled by this session."""

        mentions = self.session.mentions
        if mentions is None:
            return FilePick(unavailable=True)
        task = self.spawn_background(mentions.picker.pick(query), name="file-picker")
        if task is None:
            return FilePick(unavailable=True)
        result = await task
        assert isinstance(result, FilePick)
        return result

    def spawn_background(self, coroutine: Coroutine[Any, Any, object], *, name: str) -> asyncio.Task | None:
        """Admit one background coroutine, and keep it until its outcome has been observed.

        Refusing after close is what makes shutdown mean something: a maintenance sweep scheduled
        while the loop is unwinding would mutate the session after close returned. A refused
        coroutine is closed here rather than dropped, so it never surfaces as a never-awaited
        warning."""

        if not self._background_open:
            coroutine.close()
            return None
        task = asyncio.get_running_loop().create_task(coroutine, name=name)
        self._background.add(task)
        task.add_done_callback(self._background_done)
        return task

    def _background_done(self, task: asyncio.Task) -> None:
        self._background.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        # Expected maintenance failures are contained by the feature coroutines themselves and
        # published as their own status. Anything reaching here is an invariant break: make it
        # visible rather than leaving "Task exception was never retrieved" to the loop's handler.
        self.emit_background(f"background task {task.get_name()} failed: {error}")

    async def close_background(self) -> None:
        """Stop admitting background work, then cancel and quiesce what was already accepted."""

        self._background_open = False
        if self.session.mentions is not None:
            self.session.mentions.refresh_owner = None
        self._mention_refresh = None
        pending, self._background = self._background, set()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def close_resources(self) -> None:
        """Close what this session opened, on the loop that opened it.

        MCP is the one exception left: it still owns a private loop of its own, so its synchronous
        close runs on the managed executor rather than here.
        """

        with contextlib.suppress(Exception):
            await self.agent.model.close()
        mcp = self.session.mcp
        if mcp is not None:
            with contextlib.suppress(Exception):
                await mcp.close()

    async def _simple_loop(self) -> int:
        while True:
            try:
                entered = self.take_pending_inputs()
                initial_input = UserInput(
                    "\n".join(str(item) for item in entered),
                    tuple(image for item in entered for image in item.images),
                )
                user_input = await self.read_input(initial_text=initial_input)
            except EOFError:
                self.emit(TurnBox.SEPARATOR)
                await self.save_and_emit_resume()
                return 0
            except KeyboardInterrupt:
                continue
            if not user_input.strip():
                continue
            handled, exit_now = await self.command(user_input.strip())
            if exit_now:
                return 0
            if handled:
                continue
            self.user_turn_rule()
            started = time.monotonic()
            malformed_tool_call = False
            answered = False
            try:
                self.status_bar.start()
                try:
                    await self.agent.run(user_input)
                    answered = True
                except (asyncio.CancelledError, KeyboardInterrupt):
                    self.emit_turn("Cancelled")
                    continue
                except MalformedToolCallError as error:
                    answer = str(error)
                    malformed_tool_call = True
                except WizoltError as error:
                    answer = f"Error: {error}"
            finally:
                self.schedule_index_freshness()
                self.status_bar.stop()
            # Same rule as TuiRuntime.run_agent_turn: the engine publishes its own final answer
            # through output_fn, so only an error it raised before publishing prints here.
            if not answered:
                if answer.strip():
                    self.ui.separate()
                self.ui.emit_answer(answer, rule=False, indent=TurnBox.CONTENT_LEVEL)
            if footer := search_sources_footer(self.agent.turn_sources):
                self.ui.emit_answer(footer, rule=False, indent=TurnBox.CONTENT_LEVEL)
            if not malformed_tool_call:
                self.ui.emit_turn_end(started)
            await self.session.save_snapshot()

    async def discover_mcp(self) -> None:
        """Connect the auto_connect servers, as a task the caller's runtime owns.

        A task, not a wait: an unreachable server must not hold the prompt, and the tools index
        picks servers up as they connect. It is owned rather than detached so shutdown can cancel
        it -- a discovery still opening clients when the loop closes is exactly the thing the
        private MCP thread used to hide."""

        mcp = self.session.mcp
        if mcp is not None:
            await mcp.discover_auto()

    def emit_banner(self) -> None:
        """Write the one static line that can safely precede interactive terminal setup."""
        self.emit(f"wizolt {__version__}. /help for commands.")

    def start_session(self, *, show_banner: bool = True) -> None:
        """Initialize output and background services shared by both command-loop frontends."""
        if show_banner:
            self.emit_banner()
        # Cached state is read synchronously -- it is small, local, and the first status display
        # needs it -- and only the remote half is scheduled. Nothing here may hold the prompt: a
        # slow index, a slow filesystem, or an unreachable PyPI is not a reason to wait to type.
        checker = UpdateChecker(self.session)
        update_due = checker.load_cached()
        if self.session.update.newer_than(__version__):
            self.emit(f"update available: {__version__} -> {self.session.update.latest}. upgrade with `{' '.join(UpdateChecker.upgrade_command())}`.")
        self.render_resumed_session()
        # Publish existing availability without scanning the tree; the freshness check already
        # runs after each completed turn.
        CodeIndex(self.session).status()
        if update_due:
            self.spawn_background(checker.check(), name="update-check")
        self.spawn_background(self.clean_expired_sessions(), name="session-cleanup")
        # The provider catalog refresh runs off the startup path after the first screen, gated to
        # once per 72h (see sync.CatalogRuntime); a failure only shows through /catalog.
        catalog = self.session.catalog
        if catalog is not None and catalog.refresh_due():
            self.spawn_background(catalog.refresh(), name="catalog-refresh")

    async def clean_expired_sessions(self) -> None:
        """Run the retention sweep off the startup path: on a network filesystem it can cost
        seconds before the prompt accepts a keystroke, and nothing depends on it having run first.

        The traversal and the deletions are one blocking pass. Cancelling this waits for that pass
        to finish rather than abandoning it half-deleted -- retention removes unrecoverable work,
        so the one thing it may not do is stop in the middle. The notice is emitted here, on the
        loop, once the pass has returned its count."""

        data_dir = self.session.config.data_dir
        current_uid = self.session.uid
        days = self.session.settings.session_retention_days
        with contextlib.suppress(Exception):
            removed = await run_blocking(lambda: SessionSnapshotStore.clean_expired(data_dir, current_uid, days))
            if removed:
                self.emit_background(self.expired_sessions_notice(removed))

    def expired_sessions_notice(self, removed: int) -> str:
        """Word the retention notice: retention removes unrecoverable work, so report it rather
        than deleting silently, and name the setting that controls it."""
        days = self.session.settings.session_retention_days
        sessions = "session" if removed == 1 else "sessions"
        return f"removed {removed} saved {sessions} inactive for over {days} {'day' if days == 1 else 'days'} (runtime.session_retention_days)"

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
        # The replay is a burst of independent emits; batch them into a single print_formatted_text
        # call so the whole session restores in one flush (and one TUI coordination) instead of one
        # per line.
        with self.ui.batched():
            self.emit(f"Restored session: {self.session.uid}")
            if self.session.transcript_incomplete:
                self.emit("Warning: this transcript may omit turns written by an older wizolt version.")
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
                # The notice is its own paragraph, like the blank line between turns.
                self.emit("")
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
            self.ui.separate()  # the same gap the live narration and answer open with
            # An assistant message that carries tool calls is interim narration, not the answer:
            # the resumed session draws the same phase rule above it the live turn did (the rule
            # opens the text), skipping it only when it would land too close to the rule above.
            if message.get("tool_calls") and self.ui.rule_due(self.MIN_ROWS_BETWEEN_RULES):
                self.ui.emit_phase_rule()
            self.ui.emit_answer(content, role=role, rule=False, indent=TurnBox.CONTENT_LEVEL)
        if role == "assistant":
            tool_record_index = self.render_transcript_tool_calls(message, tool_record_index, diffs or {}, tool_results or {}, dry_run=dry_run)
            if not dry_run and message.get("tool_calls"):
                # Each tool-bearing assistant message is one batch in the replay: a voiced one
                # (it carried narration) restarts the silent count, a silent run of four closes
                # with the same batch rule the live turn drew.
                if content:
                    self._silent_batches = 0
                else:
                    self._silent_batches += 1
                    if self._silent_batches >= self.TOOL_RUN_RULE_BATCHES:
                        self.ui.emit_phase_rule()
                        self._silent_batches = 0
            return tool_record_index
        if role == "user" and content and not ImageInputs.is_tool_observation(message) and not dry_run:
            # The follow-up marker is model-facing context, part of history because it was sent.
            # The scrollback shows what the user typed, exactly as it looked when they typed it.
            self.ui.emit_answer(content.removeprefix(LIVE_FOLLOWUP_PREFIX.strip()).lstrip(), role=role, rule=False)
            # The user's message opens its turn with the same rule the live turn opened with: a
            # blank line, then the rule under the message, and the silent-batch count restarts.
            self._silent_batches = 0
            self.ui.emit_phase_rule()
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
        # Through `tool_output`, like the live call: a replayed call opens its own group with a
        # blank row above it, and its result stays attached underneath. Emitted directly, every
        # call in a turn ran into the one above it and into the narration that introduced them.
        if not preview:
            self.tool_output(toolblocks.finish_display(self.session, call, key, "failed in saved session" if failed else "", failed=failed))
            return
        # The preview block carries the call line, so the result collapses to its trailing marker
        # underneath it — the same nesting the live approval block produces.
        self.tool_output(self.transcript_edit_preview(call, preview))
        self.tool_output(toolblocks.finish_display(self.session, call, key, "", failed=False, d=ToolDisplay(nested_display=True)))

    def transcript_edit_preview(self, call: ToolCall, preview: str) -> LogBlock:
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
        return LogBlock.hierarchy(toolblocks.log_root(tooloutput.short_call(self.session, call), LogRole.AUTO, "", call), children)

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
            args = tool_payload(name, payload)
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

    async def save_and_emit_resume(self) -> None:
        self.emit_resume_line(await self.session.save_snapshot())

    def emit_resume_line(self, uid: str) -> None:
        """The paste-ready resume line for a session that has just been persisted."""
        if uid:
            # The name goes in the sentence, never in the command: the line below is meant to be
            # pasted, and only the uid is guaranteed to still mean this session tomorrow.
            name = self.session.name
            self.emit(f"Resume {name!r} with:\nwizolt --resume {uid}" if name else f"Resume with:\nwizolt --resume {uid}")

    def read_input_sync(
        self,
        prompt_text: str = UiPrinter.PROMPT_PREFIX,
        *,
        initial_text: str = "",
    ) -> str:
        """Read from the injected/non-TTY input path; interactive terminals use TuiApp."""
        return initial_text or self.input_fn(prompt_text)

    async def invoke_input(self, action: Callable[[], Any]) -> Any:
        """Run an injected synchronous input callback without owning its blocking lifetime.

        Python cannot cancel an arbitrary callback. A daemon adapter lets cancellation release the
        CLI runtime immediately; the embedding still owns unblocking its callback if it wants the
        thread itself to finish. The default executor cannot be used here because `asyncio.run()`
        waits for that executor during shutdown.
        """

        loop = asyncio.get_running_loop()
        result: asyncio.Future[Any] = loop.create_future()

        def publish(value: Any = None, error: BaseException | None = None) -> None:
            if result.done():
                return
            if error is None:
                result.set_result(value)
            else:
                result.set_exception(error)

        def invoke() -> None:
            try:
                value, error = action(), None
            except BaseException as caught:  # noqa: BLE001 - reproduce the callback's outcome on its caller.
                value, error = None, caught
            with contextlib.suppress(RuntimeError):  # the cancelled runtime may already be closed.
                loop.call_soon_threadsafe(publish, value, error)

        threading.Thread(target=invoke, name="input-callback", daemon=True).start()
        return await result

    async def read_input(
        self,
        prompt_text: str = UiPrinter.PROMPT_PREFIX,
        *,
        initial_text: str | UserInput = "",
    ) -> str | UserInput:
        """Read one non-TTY line without parking the default executor on POSIX stdin.

        An injected synchronous reader remains an embedding boundary and runs through a daemon
        adapter; its owner is responsible for unblocking it. The process stdin path is driven
        directly by fd readiness.
        """

        if initial_text:
            return initial_text
        if self.input_fn is not input:
            return await self.invoke_input(lambda: self.read_input_sync(prompt_text))

        if prompt_text:
            sys.stdout.write(prompt_text)
            sys.stdout.flush()

        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        result: asyncio.Future[bytes] = loop.create_future()
        was_blocking = os.get_blocking(fd)

        def finish_line() -> bool:
            newline = self._stdin_buffer.find(b"\n")
            if newline < 0:
                return False
            line = bytes(self._stdin_buffer[:newline])
            del self._stdin_buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if not result.done():
                result.set_result(line)
            return True

        def readable() -> None:
            if finish_line():
                return
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                return
            except OSError as error:
                if not result.done():
                    result.set_exception(error)
                return
            if chunk:
                self._stdin_buffer.extend(chunk)
                finish_line()
                return
            if self._stdin_buffer:
                line = bytes(self._stdin_buffer)
                self._stdin_buffer.clear()
                if not result.done():
                    result.set_result(line)
            elif not result.done():
                result.set_exception(EOFError())

        os.set_blocking(fd, False)
        loop.add_reader(fd, readable)
        readable()
        try:
            raw = await result
        finally:
            loop.remove_reader(fd)
            os.set_blocking(fd, was_blocking)
        return raw.decode(sys.stdin.encoding or "utf-8", errors=sys.stdin.errors or "strict")

    def emit(self, text: str | LogBlock = "", indent: int = 0) -> None:
        self.ui.emit(text, indent)

    def emit_turn(self, text: str = "") -> None:
        """A line that belongs to the exchange rather than to the session around it: a turn
        outcome, a command's reply, a refusal to run one. Those sit in the content column with
        the model's text and the tool lines; session chrome (the banner, the restored-session
        notice, the resume line) stays at column 0 and frames them."""
        self.emit(text, TurnBox.CONTENT_LEVEL)

    def emit_background(self, text: str) -> None:
        """Emit from a daemon worker only while this loop still owns terminal output.

        The gate and the scrollback writer's admission share one lock while the TUI is live, so a
        worker cannot pass this check and then race the writer closing behind it."""
        with self.background_output_lock:
            if self.background_output_open:
                self.emit(text)

    def close_background_output(self, final_output: Callable[[], None] | None = None) -> None:
        with self.background_output_lock:
            self.background_output_open = False
            if final_output is not None:
                final_output()

    def with_status_paused(self, action):
        # Only quiet the standalone status row used by the simple frontend. The full TUI renders
        # status and output together, so it never needs this terminal-level coordination.
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
            # The blank line parts each block from the one above; it is skipped when the block
            # sits directly under a rule just drawn (the turn's opening rule, or a batch rule),
            # which already provides the seam.
            if isinstance(text, str) or (text.items and isinstance(text.items[0], LogLine)):
                self.ui.separate()
            self.emit(text)

        self.with_status_paused(output)

    def builtin_call_output(self, label: str, detail: str) -> None:
        """Log a tool the provider ran for itself, so the transcript shows it like any other call.

        A provider-side search leaves no local tool call to log, and the running status label is gone
        the moment the turn ends. Without this line the transcript would credit the model with
        knowledge it went and looked up. No edge: a standalone row, nothing above it to join (a
        branch glyph would dangle."""
        self.tool_output(LogBlock([LogLine(label, Text.clip_width(detail, 120), LogRole.TOOL, LogEdge.NONE)]))

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

    def agent_output(self, text: str = "", *, interim: bool = True) -> None:
        """A turn's interim narration or its final answer, whichever the engine is publishing:
        `output_fn` feeds interim text here, and `final_output_fn` routes the answer through the
        same promotion handling with the flag flipped. Only the flag differs; the answer takes no
        phase rule below it, because the turn-end rule already closes the turn and two rules in a
        row would read as a box."""
        # An early promotion is presentation-only: Agent still publishes the same semantic text
        # after ModelClient returns. Consume the one-shot marker instead of printing it twice.
        promoted = self.model_stream_promoted_text
        self.model_stream_promoted_text = ""
        if promoted:
            remaining = self.unpromoted_text(text, promoted)
            if not remaining:
                return
            text = remaining
        emit = self.emit_agent_output if interim else self.emit_agent_answer
        self.with_status_paused(lambda: emit(text))

    def agent_answer_output(self, text: str = "") -> None:
        """The turn's final answer: the same markdown rendering as interim narration, but no phase
        rule below it -- the turn-end rule is already there to close the turn."""
        self.agent_output(text, interim=False)

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
                # Queued, not written here: this runs on the runtime loop (a stream callback), and
                # printing above the application means suspending it. The turn awaits the writer's
                # barrier before its tool batch, so the promoted answer is on screen first.
                self.write_scrollback(lambda: self.with_status_paused(lambda: self.emit_agent_output(promote)))

    def write_scrollback(self, callback: Callable[[], None]) -> None:
        """Publish one completed write into the runtime's ordered queue, or print it directly.

        There is no queue outside the interactive runtime -- headless output is synchronous and
        already in order -- so the callback simply runs."""

        writer = self.scrollback
        if writer is not None:
            writer.submit(callback)
            return
        tui = self.tui
        if tui is not None:
            tui.write_to_scrollback_sync(callback)
        else:
            callback()

    def set_approval_form(self, actions: list[tuple[str, str]]) -> bool:
        # The selectable action row exists only in the TUI. Headless and piped runs report False so
        # the approval brief keeps advertising the typed protocol they do have.
        return self.tui is not None and self.tui.set_approval_form(actions)

    def cancel_tool_input(self) -> None:
        """Resolve a pending approval or Ask prompt as cancelled, so whoever is parked on the user
        returns. Headless runs have no prompt to resolve; their injected input owns its own end."""
        if self.tui is not None:
            self.tui.cancel_input()

    async def tool_input(self, prompt: str = "") -> str | None:
        """Await one line of user input for a tool: an approval, an Ask free-text page.

        Under the TUI this is the application's own input row -- prompt-toolkit does not nest, so a
        second application is not an option -- awaited on the loop that runs it. None propagates the
        TUI's cancel signal.

        Without a TUI the injected `input_fn` blocks, so it runs through the same daemon adapter as
        non-TTY input. That contract belongs to whoever injected it: it should still be unblocked,
        but cancellation never holds the CLI runtime open around it."""

        if self.tui is not None:
            return await self.tui.request_input(prompt)
        if self.input_fn is input:
            return await self.read_input(prompt)
        return await self.invoke_input(lambda: self.with_status_paused(lambda: self.input_fn(prompt)))

    # How close a phase rule may come to the one above it, in rendered rows. Under this the rule
    # is skipped: two rules a few rows apart part nothing, they just add lines to what is already
    # a short stretch. The agent saying two things in quick succession is one phase, not two.
    MIN_ROWS_BETWEEN_RULES: ClassVar[int] = 6
    # How many tool batches in a row the agent can work in silence before a phase rule closes the
    # stretch. Rendered rows would punish one big output and reward many small ones; what matters
    # is that the model keeps calling tools without ever saying anything back, so the count is of
    # batches, not lines. Fired after the batch's output is out, so a batch is never cut in half.
    TOOL_RUN_RULE_BATCHES: ClassVar[int] = 4

    def emit_agent_output(self, text: str) -> None:
        """A turn's interim narration, opened by the same full-width rule the turn ends with minus
        the label: the rule lands above the text, so it announces the new phase instead of closing
        the old one, and the narration's own text is the label, so the rule carries none. A rule
        that would land within MIN_ROWS_BETWEEN_RULES of the one above it is skipped -- two rules a
        few rows apart part nothing, they just add lines to an already short stretch. The agent
        saying two things in quick succession is one phase, not two."""
        # The narration breaks a run of silent tool batches: the agent spoke, so the count starts
        # over (the batch itself is reported voiced through on_tool_batch, not by this flag). The
        # blank line above parts it from the previous block unless it sits directly under a rule
        # just drawn, which already provides the seam.
        if text.strip():
            self.ui.separate()
        self._silent_batches = 0
        # The rule opens the text, so the distance check runs before it is drawn; the blank line
        # above already counts, the text's own rows count toward the next rule.
        if self.ui.rule_due(self.MIN_ROWS_BETWEEN_RULES):
            self.ui.emit_phase_rule()
        self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)

    def emit_agent_answer(self, text: str) -> None:
        """The turn's final answer: the one block of model text the turn-end rule closes, so it
        takes no phase rule of its own."""
        if text.strip():
            self.ui.separate()
        self.ui.emit_answer(text, rule=False, indent=TurnBox.CONTENT_LEVEL)

    def user_turn_rule(self) -> None:
        """Open the turn with the same full-width rule under the user's message: the seam between
        what the user said and everything the agent does in reply. It always draws -- the user's
        message is the top boundary of the turn, so every later rule measures its distance from
        it -- and it restarts the silent-batch count for the turn. The blank line above it is the
        turn-opening line the loop emits first."""
        self._silent_batches = 0
        self.ui.emit_phase_rule()

    def tool_batch_output(self, silent: bool) -> None:
        """Close a run of tool calls that has gone on long enough without the agent saying
        anything -- the model not recovering is exactly when the transcript needs the seam most,
        because nothing else is about to provide one. Fires once per batch, after the batch's
        output is out, so it can never cut a batch in half. `silent` is whether the batch carried
        no narration; a batch that spoke restarts nothing and counts nothing."""

        def output() -> None:
            if not silent:
                return
            self._silent_batches += 1
            if self._silent_batches >= self.TOOL_RUN_RULE_BATCHES:
                self.ui.emit_phase_rule()
                self._silent_batches = 0

        self.with_status_paused(output)

    def worker_answer_output(self, text: str) -> None:
        """The worker's interim and final model text, rendered like an agent answer (markdown) rather than the
        plain log lines tool execution prints as."""
        self.with_status_paused(lambda: self.emit_agent_output(text))

    def _begin_cli_preview(self) -> None:
        """Pause the status bar if running and start the CLI Bash live-preview line."""
        self.live_status_paused = self.status_bar.is_running()
        if self.live_status_paused:
            self.status_bar.stop()
        self.live_preview.start()

    def tool_live_start(self, budget: float | None = None) -> None:
        if not self.ui.color:
            return
        if self.tui is not None:
            with self.live_preview.lock:
                self.live_preview.active = True
                self.live_preview.text = ""
                self.live_preview.started_at = time.monotonic()
                self.live_preview.deadline = (time.monotonic() + budget) if budget else None
            self.tui.invalidate()
            return
        self._begin_cli_preview()
        # The preview region is reused across calls, so a call without a budget (Bash) must clear
        # any deadline a previous Job wait left behind.
        with self.live_preview.lock:
            self.live_preview.deadline = (time.monotonic() + budget) if budget else None

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

    async def command(self, text: str) -> tuple[bool, bool]:
        """Dispatch one slash command, on the loop that owns this session.

        A handler that needs the network -- `/compact` is the one -- is a coroutine and is awaited
        here, so its request lives on the same loop as everything else the session opened. Every
        other handler is local and bounded, and runs directly."""

        if text in {"/exit", "/quit", "exit", "quit"}:
            await self.save_and_emit_resume()
            return True, True
        if not text.startswith("/"):
            return False, False
        name, _, args = text.partition(" ")
        entry = COMMAND_LOOKUP.get(name)
        output = entry.handler(self, args.strip()) if entry else f"Unknown command: {name}"
        if inspect.isawaitable(output):
            output = await output
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
    Command("/catalog", commands.catalog_command, render="answer"),
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
