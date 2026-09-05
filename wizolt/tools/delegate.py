"""Delegate: hand a bounded task to the worker, a second in-process wizolt session.

The worker is a full wizolt session — compaction, Recall, tr.N, Job, Skill, MCP, image, diff,
confirmation, snapshots, protocol adapters — projected from the parent's state with its own system
prompt and a reduced tool list. Serial, one worker at a time, one-way: the worker has no tool that
points back at the parent. See DESIGN.md's worker-handoff section.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from wizolt.base import MEMORY_PREFIXES, ApprovalView, Json, LogBlock, LogLine, LogRole, ToolError, oneline, run_blocking
from wizolt.prompts import WORKER_PROMPT
from wizolt.session import Session, SessionSnapshotStore
from wizolt.tools.base import Tool

if TYPE_CHECKING:
    from wizolt.config import (
        Config,
        ProviderConfig,
    )
    from wizolt.engine import Agent
    from wizolt.runner import ToolRunner

# The worker's tool set. Exclusions, and why: Delegate (would recurse), Ask (blocks on user input
# while the parent turn is inside a tool call), NextHints (no idle prompt; blurs the turn-ending
# rule). ViewImage and ToolScript joined: ViewImage reads any local image path, not just user
# attachments, and ToolScript batches repetitive same-shape calls. Skill and MCP stay: a worker
# that cannot load skills or call external tools cannot do real work.
WORKER_TOOLS: tuple[str, ...] = (
    "Read",
    "ViewImage",
    "Search",
    "InspectCode",
    "Edit",
    "Bash",
    "Job",
    "ToolScript",
    "Recall",
    "RecallContext",
    "Note",
    "Skill",
    "MCP",
)

# Agent and worker session live and die together; rebuilt on every send would reopen the HTTP client
# each time. The Agent is held by the worker Session itself (Session._agent), so a fresh worker object
# — e.g. after /resume re-enters the same parent — always gets a fresh Agent: the old one dies with
# the old worker it was bound to, instead of lingering in a module-level dict keyed by uid.


def _worker_output(runner: ToolRunner):
    """Wrap the worker Agent's output for the parent's log stream.

    The engine publishes LogLine items on the runner paths but bare strings for the model's own
    text (engine.py's output_fn calls), so the wrapper cannot assume a LogLine: LogBlock.walk
    only recognizes LogLine and LogBlock items and crashes on a str.

    A memory-shaped string (a Note body, MEMORY_PREFIXES) is passed through unwrapped so the
    parent renders it with segments() -> memory_segments() and gets the same per-line colors as
    the main agent's Note display.
    """

    def emit(block) -> None:
        if isinstance(block, str) and block.startswith(MEMORY_PREFIXES):
            runner.output_fn(block)
            return
        if isinstance(block, str):
            block = LogLine("", block, LogRole.AUTO)
        runner.output_fn(LogBlock([block]))

    return emit


def _worker_stream(runner: ToolRunner):
    """Wrap the worker Agent's model stream for the parent's live preview.

    The worker's own output_fn already writes completed text into the parent's
    scrollback, and the parent loop's `output_done` promotion would write it a
    second time: the promoted-text marker (model_stream_promoted_text) is consumed
    only by the parent's own agent_output path, never by the worker's
    _worker_output path. So `output_done` must only clear the preview, never
    promote; everything else forwards unchanged. Non-TUI behavior is unchanged:
    without a TUI there is no promotion to skip, and the ("", "") clear is what
    model_stream_output would have done with the forwarded kind anyway."""

    def stream(kind: str, text: str) -> None:
        # Read through a local so the None check narrows: the attribute is Optional on the
        # runner, and on_stream is only wired when it is set, but the checker cannot see that here.
        model_stream = runner.model_stream
        if model_stream is None:
            return
        if kind == "output_done":
            model_stream("", "")
            return
        model_stream(kind, text)

    return stream


def _wire_worker_agent(agent: Agent, runner: ToolRunner) -> None:
    """Bind a persistent worker agent to the runner driving this send.

    The worker session keeps its Agent between sends, while presentation belongs to the current
    ToolRunner. Rebind every time so a worker first used headlessly does not stay headless after it
    is attached to a TUI, and a detached TUI callback is never retained by a later runner.
    """
    worker_output = _worker_output(runner)
    agent.output_fn = runner.worker_answer or worker_output
    agent.final_output_fn = runner.worker_answer
    agent.model.on_stream = _worker_stream(runner) if runner.model_stream is not None else None
    agent.model.on_retry_wait = runner.retry_wait
    agent.model.on_builtin_call = runner.builtin_call
    agent.context.on_compaction = runner.compaction
    agent.tools.input_fn = runner.input_fn
    agent.tools.output_fn = worker_output
    agent.tools.live_start = runner.live_start
    agent.tools.live_output = runner.live_output
    agent.tools.approval_form = runner.approval_form
    agent.tools.text_viewer = runner.text_viewer
    agent.tools.cancel_input = runner.cancel_input
    agent.tools.script_status = runner.script_status


def worker_provider_config(config: Config, provider_name: str) -> ProviderConfig:
    """The detached provider entry a worker should run on, with [worker] overrides applied.

    A worker must never share a ProviderConfig object with its parent: the /worker
    provider|model|reason|api live-switch path replaces the worker's active entry in place, and
    dataclasses.replace is shallow, so a shared object would leak worker-only changes into the
    parent's provider. This helper is the single place that copies the entry and folds in the
    [worker] model/reasoning/api overrides (empty string = inherit the entry's own value); both
    _spawn_worker and the live-switch path in loop.py call it."""
    provider = replace(config.providers[provider_name])
    if config.worker_model:
        provider.model = config.worker_model
    if config.worker_reasoning:
        provider.reasoning = config.worker_reasoning
    if config.worker_api:
        provider.api = config.worker_api
    return provider


def refresh_worker_entry(config: Config, worker: Session | None, provider_name: str | None = None) -> None:
    """Rebuild a live worker's active provider entry from the parent's current [worker] overrides.

    Copy-on-write so the worker never shares the parent's providers dict; provider_name switches
    the worker's active entry (the /worker provider path) instead of rewriting the current one.
    A no-op when no worker session exists. Shared by the /worker live switches in loop.py and the
    Delegate confirm-time configuration loop in runner.py."""
    if worker is None:
        return
    if worker.config.providers is config.providers:
        worker.config.providers = dict(worker.config.providers)
    target = provider_name or worker.config.active_provider
    if provider_name is not None:
        worker.config.active_provider = provider_name
    worker.config.providers[target] = worker_provider_config(config, target)


class DelegateTool(Tool):
    NAME = "Delegate"
    runner: ToolRunner | None = None  # injected by ToolRunner.call_tool; the runner owns the cancel wiring
    MUTATES = True  # the delegation itself needs confirmation; the worker's own tools still confirm per call
    DESCRIPTION = (
        "Send one bounded, verifiable task to a serial worker with separate context and provider. The worker sees only the standalone order and its own history. "
        "Use it for well-specified work worth isolating, not small or exploratory work. Include the goal, files, known facts, constraints, boundaries, and verification. "
        "Judge the result from the actual diff and rerun decisive checks through the affected public boundary; the worker report is not proof. "
        "Reset when changing tasks, after the spec changes or repeated failure, or before substantial work at high context. Reset keeps file changes."
    )

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema(
            {
                "action": {"type": "string", "enum": ["send", "reset", "status"], "description": "Worker operation"},
                "order": {
                    "type": "string",
                    "description": "Standalone send order: goal, files, known facts, constraints, boundaries, and verification",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional step cap for this send",
                },
                "language": {
                    "type": "string",
                    "description": "Worker output language, including its live stream",
                },
                "title": {
                    "type": "string",
                    "description": "Short title shown in start/done dividers",
                },
            },
            ["action"],
        )

    def always_confirms(self) -> bool:
        """A send is confirmed even under yolo; status and reset are not.

        What the prompt shows is the order text, and the order is what decides whether a delegation
        succeeds -- the parent model writes it, and it is also judging its own spec when it decides
        to delegate at all. Seeing the order before it goes is the only cheap check on that; a bad
        one costs a whole worker round to discover. Editing files under yolo fails visibly and at
        once, which is what that flag actually buys."""
        payload = self.payload_dict()
        return str(payload.get("action") or "").strip() == "send"

    def short_args(self) -> list[str]:
        """The root log line is just the action: `Delegate send`, never the order blob."""
        payload = self.payload_dict()
        action = str(payload.get("action") or "").strip()
        return [action] if action else [""]

    def payload_dict(self) -> Json:
        """The single-dict argument of this call, or {} when the shape differs."""
        return self.args[0] if len(self.args) == 1 and isinstance(self.args[0], dict) else {}

    def approval_view(self) -> ApprovalView | None:
        """The whole order, as prose: what `v` opens at the send prompt and what Ctrl-O reopens
        afterwards. No lexer -- an order is written for a reader, so it renders as markdown."""
        payload = self.payload_dict()
        order = payload.get("order")
        if not isinstance(order, str) or not order.strip():
            return None  # nothing to view
        return ApprovalView("order", order, "", self.header_rows())

    def header_rows(self, order_row: tuple[str, str] | None = None) -> list[tuple[str, str]]:
        """(label, value) rows for a send: title, explicit send parameters, and the worker
        configuration it runs under. The order itself is excluded, because the viewer shows it in
        full below these rows; the approval brief passes its one-line excerpt as `order_row` and
        this decides where it sits, so callers never have to splice a row into this list."""
        payload = self.payload_dict()
        rows: list[tuple[str, str]] = []
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            rows.append(("title", oneline(title.strip(), 120)))
        if order_row is not None:
            rows.append(order_row)
        language = payload.get("language")
        if isinstance(language, str) and language.strip():
            rows.append(("language", oneline(language.strip(), 60)))
        if payload.get("max_steps") is not None:
            rows.append(("max_steps", str(payload["max_steps"])))
        rows.extend(self.worker_config_rows(self.session.config))
        return rows

    @staticmethod
    def worker_config_rows(config: Config) -> list[tuple[str, str]]:
        """The effective worker provider/model/effort/api as (label, value) rows; a field that
        inherits the provider entry's own value is shown as `(inherit) <value>`."""
        provider_name = config.worker_provider or config.active_provider
        entry = config.providers[provider_name]
        return [
            ("provider", config.worker_provider or f"(inherit) {provider_name}"),
            ("model", config.worker_model or f"(inherit) {entry.model or '(no model)'}"),
            ("effort", config.worker_reasoning or f"(inherit) {entry.reasoning}"),
            ("api", config.worker_api or f"(inherit) {entry.api}"),
        ]

    async def call(self) -> str:
        payload = self.single_dict_arg("Delegate requires named fields")
        action = str(payload.get("action") or "").strip()
        if action == "send":
            return await self._send(payload)
        if action == "reset":
            return await self._reset()
        if action == "status":
            return self._status()
        raise ToolError(f"unknown action: {action!r}")

    def _send_args(self, payload: Json) -> tuple[str, int, str]:
        """Validate one send and return (order, max_steps, title). Pure: no worker is spawned.

        Separate from the send itself so the arguments are checked the same way whichever entry
        point was used -- validation is the tool's own business and needs no runner."""

        raw_order = payload.get("order")
        if not isinstance(raw_order, str) or not (order := raw_order.strip()):
            raise ToolError("Delegate send requires a non-empty string order")
        raw_max_steps = payload.get("max_steps")
        if raw_max_steps is None:
            max_steps = self.session.settings.max_steps
        elif isinstance(raw_max_steps, bool) or not isinstance(raw_max_steps, int) or raw_max_steps < 1:
            raise ToolError("Delegate max_steps must be an integer >= 1")
        else:
            max_steps = raw_max_steps
        raw_language = payload.get("language")
        language = raw_language.strip() if isinstance(raw_language, str) else ""
        if raw_language is not None and not language:
            raise ToolError("Delegate language must be a non-empty string")
        if language:
            # The user watches the worker live and reads its report; the worker prompt lets an
            # explicit language request override its defaults, so one directive in the order covers
            # the stream, the interim messages, and the final answer alike.
            order += f"\n\nReply language: {language}. The human user reads this terminal and sees everything you output -- the live stream while you work, your interim messages, and your final report -- so think and write in {language} for all of it; keep code, identifiers, paths, and commands verbatim."
        raw_title = payload.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else ""
        if raw_title is not None and not title:
            raise ToolError("Delegate title must be a non-empty string")
        return order, max_steps, title

    async def _send(self, payload: Json) -> str:
        order, max_steps, title = self._send_args(payload)
        runner = getattr(self, "runner", None)
        if runner is None:
            raise ToolError("Delegate requires a tool runner")
        parent = self.session
        worker = parent.worker
        if worker is None:
            worker = await self._spawn_worker(parent)
            parent.worker = worker
        # Rebuild the settings copy on every send: sharing the parent's RuntimeSettings object would
        # let a per-call max_steps override leak into the parent's budget, and a one-time copy would
        # miss runtime changes (/yolo, /set) between delegations.
        worker.settings = replace(parent.settings, max_steps=max_steps)
        agent = worker._agent
        if agent is None:
            # Local import: engine imports wizolt.tools at module level (engine.py:30), so a
            # module-level `from wizolt.engine import Agent` would cycle tools -> engine -> tools.
            # Same pattern as Tool.resolved_schemas and Session's local imports.
            from wizolt.engine import Agent

            agent = Agent(
                worker,
                input_fn=runner.input_fn,
                output_fn=lambda _text: None,
            )
            worker._agent = agent
        # The Agent is persistent but the runner/UI driving it need not be. This also wires a
        # resumed or previously headless worker before its first streamed token.
        _wire_worker_agent(agent, runner)
        started = time.monotonic()
        before_diffs = len(worker.turn_diffs)
        before_in = worker.usage.prompt_tokens
        before_out = worker.usage.completion_tokens
        # A visible boundary where the delegation starts: until the finish block there is nothing in
        # the scrollback that says who is talking, and the worker's own streamed lines do not. A
        # full-width rule whose yellow label reads the worker's live config and a one-line order
        # summary; without a wired worker_rule, the yellow [worker] line below stands in.
        config = worker.config
        if runner.worker_rule is not None:
            runner.worker_rule(f"worker start · {config.active_provider}/{config.provider.model or '(no model)'} · {title or oneline(order, 60)}")
        else:
            runner.output_fn(
                LogBlock(
                    [
                        LogLine(
                            "[worker]",
                            f"▶ {config.active_provider}/{config.provider.model or '(no model)'} · {title or oneline(order, 200)}",
                            LogRole.WORKER,
                        )
                    ]
                )
            )
        failure: Exception | None = None
        try:
            # Awaited directly: the worker's turn is a task of this one, so cancelling the
            # parent cancels the worker's own model request and tool batch by propagation --
            # no second cancellation channel, and no bridge back to the parent's loop.
            # The engine publishes the final answer through the agent's output_fn, so a
            # worker's report lands in the parent scrollback like its interim messages.
            answer = await agent.run(order)
        except Exception as error:  # noqa: BLE001 - the worker's failure becomes a ToolError envelope below, after the finally block merged its diffs
            failure = error
        finally:
            # Merge diffs even when interrupted, or the user never sees what the worker did.
            self._merge_diffs(worker, parent, before_diffs)
        if failure is not None:
            # Folded to one bounded, quote-free line at the source rather than where it is read.
            # `status` renders it as an attribute of the envelope the model parses, and a provider
            # error routinely carries both a quote and a newline -- an unescaped one closes the
            # attribute early and the rest of the tag reads as garbage. The full text is in the
            # failure report the parent got when it happened; this is the reminder, not the record.
            worker.state.last_error = oneline(str(failure).replace('"', "'"), 200)
            worker.state.last_error_round = worker.state.round_count
            raise ToolError(self._failure_report(worker, failure, started, before_diffs)) from failure
        worker.state.last_error, worker.state.last_error_round = "", 0
        elapsed, files, percent = self._send_facts(worker, started, before_diffs)
        in_tokens = worker.usage.prompt_tokens - before_in
        out_tokens = worker.usage.completion_tokens - before_out
        return "\n".join(
            [
                f'<Delegate action="send" steps="{worker.state.turn_step}" elapsed="{elapsed:.1f}s" files="{files}" stopped_at_max_steps="{str(agent.stopped_at_max_steps).lower()}" tokens="{in_tokens}/{out_tokens}" rounds="{worker.state.round_count}" context_percent="{percent}">',
                "<worker>",
                answer.rstrip(),
                "</worker>",
                "</Delegate>",
            ]
        )

    def _send_facts(self, worker: Session, started: float, before_diffs: int) -> tuple[float, str, int]:
        """Elapsed time, changed-file list, and context fill for one delegation: the facts the
        success envelope and the failure report both carry, computed from the same sources so the
        two paths cannot drift. `files` only reflects diffs merged so far, which is why the
        failure report is built after the `finally` block ran `_merge_diffs`."""
        elapsed = time.monotonic() - started
        files = ", ".join(sorted({diff.path for diff in worker.turn_diffs[before_diffs:]})) or "(none)"
        percent = worker.usage.context_percent(worker.state.context_percent)
        return elapsed, files, percent

    def _failure_report(self, worker: Session, error: Exception, started: float, before_diffs: int) -> str:
        """The ToolError message for a failed delegation. Still raised, not returned: the runner
        must mark the call failed (red line + status="failed"), and the envelope lives in the
        exception text. Answers the parent's three questions: what the worker did (`files`),
        whether it is still alive, and what to do next."""
        elapsed, files, percent = self._send_facts(worker, started, before_diffs)
        return "\n".join(
            [
                f"worker failed after {worker.state.turn_step} steps ({elapsed:.1f}s): {error}",
                f'alive="true" rounds="{worker.state.round_count}" context_percent="{percent}" files="{files}"',
                "Its context is kept: answer the problem and send again, or reset to discard this worker's process.",
                "Files listed above were already changed and merged.",
            ]
        )

    @staticmethod
    def _merge_diffs(worker: Session, parent: Session, start: int) -> None:
        for diff in worker.turn_diffs[start:]:
            parent.store_turn_diff(diff.key, diff.turn, diff.path, diff.diff, before=diff.before, after=diff.after, round=parent.state.round_count)

    async def _spawn_worker(self, parent: Session) -> Session:
        uid = parent.uid + ".w"
        provider_name = parent.config.worker_provider or parent.config.active_provider
        # replace() is shallow, so the worker needs its own providers dict with a detached copy of
        # the entry it runs on: the /worker live-switch mutates that entry in place, and the
        # snapshot-load path below receives this same config so a resumed worker picks up the
        # current provider/model/reasoning overrides. Every other entry is still the parent's
        # object, so a worker snapshot that carried provider_overrides for such an entry would
        # leak a restore into the parent via apply_provider_overrides. It cannot today: a worker
        # runs no slash commands, so its overrides are always empty.
        provider = worker_provider_config(parent.config, provider_name)
        config = replace(parent.config, active_provider=provider_name, providers={**parent.config.providers, provider_name: provider})
        settings = replace(parent.settings)
        cwd, system_info, created_at = parent.cwd, parent.system_info, parent.created_at
        skills, mcp, catalog = parent.skills, parent.mcp, parent.catalog

        def restore() -> Session:
            try:
                return SessionSnapshotStore.load(uid, config=config, settings=settings, cwd=cwd)
            except Exception:  # noqa: BLE001 - missing or corrupt snapshot means "no worker yet", never an error.
                return Session(
                    cwd=cwd,
                    system_info=system_info,  # shared: skip a SystemInfo.detect
                    config=config,
                    settings=settings,
                    created_at=created_at,  # cache-critical: the Environment layer must stay identical across spawns
                    uid=uid,
                    skills=skills,  # shared objects, never re-discovered
                    mcp=mcp,
                    catalog=catalog,
                )

        worker = await run_blocking(restore)
        # load only honors the snapshot's keys, so these return to their defaults; re-set them
        # after load or construction, never before. skills/mcp are shared objects from the parent
        # (never re-discovered or re-connected): a snapshot-loaded worker that kept its own would
        # spawn duplicate MCP processes and break its own cache-prefix reuse across delegations.
        worker.system_prompt = WORKER_PROMPT
        worker.tool_names = WORKER_TOOLS
        worker.listed = False
        worker.skills = parent.skills
        worker.mcp = parent.mcp
        worker.catalog = parent.catalog
        return worker

    async def _reset(self) -> str:
        parent = self.session
        worker = parent.worker
        uid = worker.uid if worker is not None else parent.uid + ".w"
        directory = SessionSnapshotStore.project_dir(parent.config.data_dir, parent.cwd)
        jobs = tuple(worker.jobs.values()) if worker is not None else ()

        def reset_transaction() -> bool:
            snapshot_path = os.path.join(directory, uid + ".jsonl")
            meta_path = os.path.join(directory, uid + SessionSnapshotStore.META_SUFFIX)
            assets_path = os.path.join(directory, uid + ".assets")
            if worker is None and not any(os.path.exists(path) for path in (snapshot_path, meta_path, assets_path)):
                return False

            # Reset owns the worker runtime, including background processes. Dropping the Session
            # while one of its Job handles is live would leave an unmanageable process behind.
            for job in jobs:
                try:
                    job.kill()
                except Exception as error:
                    raise ToolError(f"worker reset failed to stop {job.id}: {error}") from error

            # The delete path derives entirely from the parent's own uid; the model's arguments
            # carry no path, so the worst case is deleting this session's own worker, nothing else.
            try:
                os.unlink(snapshot_path)
            except FileNotFoundError:
                pass
            except OSError as error:
                raise ToolError(f"worker reset failed to delete its snapshot: {error}") from error
            with contextlib.suppress(OSError):
                os.unlink(meta_path)
            shutil.rmtree(assets_path, ignore_errors=True)
            return True

        cleared = await run_blocking(reset_transaction, commit=lambda cleared: setattr(parent, "worker", None) if cleared else None)
        return f'<Delegate action="reset" uid="{uid}"/>' if cleared else '<Delegate action="reset" alive="false"/>'

    def _status(self) -> str:
        worker = self.session.worker
        if worker is None:
            return '<Delegate action="status" alive="false"/>'
        percent = worker.usage.context_percent(worker.state.context_percent)
        last = next((str(message.get("content") or "") for message in reversed(worker.messages) if message.get("role") == "assistant"), "")
        # The last failure, so the parent can confirm why it stopped without relying on memory;
        # absent once a send succeeded since then.
        error_attr = f' last_error="{worker.state.last_error}" last_error_round="{worker.state.last_error_round}"' if worker.state.last_error else ""
        return "\n".join(
            [
                f'<Delegate action="status" alive="true" provider="{worker.config.active_provider}" model="{worker.config.provider.model}" rounds="{worker.state.round_count}" context_percent="{percent}"{error_attr}>',
                f"<last>{(last.strip()[:400] or '(no answer yet)')}</last>",
                "</Delegate>",
            ]
        )
