"""Delegate: hand a bounded task to the worker, a second in-process minacode session.

The worker is a full minacode session — compaction, Recall, tr.N, Job, Skill, MCP, image, diff,
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

from minacode.base import MEMORY_PREFIXES, ApprovalView, Json, LogBlock, LogLine, LogRole, ToolError
from minacode.prompts import WORKER_PROMPT
from minacode.session import Session, SessionSnapshotStore
from minacode.tools.base import Tool

if TYPE_CHECKING:
    from minacode.config import (
        Config,
        ProviderConfig,
    )
    from minacode.runner import ToolRunner

# The worker's tool set. Exclusions, and why: Delegate (would recurse), Ask (blocks on user input
# while the parent turn is inside a tool call), NextHints (no idle prompt; blurs the turn-ending
# rule), ViewImage (no user images). Skill and MCP stay: a worker that cannot load skills or call
# external tools cannot do real work.
WORKER_TOOLS: tuple[str, ...] = (
    "Read",
    "Search",
    "InspectCode",
    "Edit",
    "Bash",
    "Job",
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
    DESCRIPTION = """Hand a bounded task to the worker: a second in-process minacode session on its own provider, with its own system prompt and tool set. The worker cannot see this session's history -- only the order text and its own prior history -- so the order must stand alone.

Delegate bounded, verifiable work you can spec in one order and judge from its diff or test output; it buys context hygiene and the worker's model, never speed (delegation is serial). Do small work yourself, explore yourself until the task is bounded, and keep the heart of the current request here: writing the order and reviewing the result cost about as much as doing the work.

Accept the result against the order's `how to verify` by looking at actual evidence -- the diff, the test output, the recorded command output -- never the worker's own report: it is the worker's summary and rationalizes in its favor. Say what you accepted on and name it (which hunk, which test, which command's output); "the worker reports the tests pass" is not acceptance, and neither is a green test run on tests the worker also wrote. Prefer a rerunnable black-box test of the external contract (one you can read and rerun yourself) over your own judgment over a diff, and for complex or error-prone implementations have the order ask the worker to write that test before implementing; do not force test-first on tasks it does not fit (wording changes, config additions, open-ended exploration) -- decide per task in the order.

Write the order to state: the goal, the files it touches, the constraints, how to verify, the boundaries (what not to touch), and the facts you already established -- exact signatures, `path:line` anchors, the approach chosen and the ones already ruled out. That last part is what keeps the delegation cheap: the worker starts on an empty context, so every fact the order leaves out it re-derives and every gap it re-decides, and its decision is the one you then have to review or undo. Keep one delegation small enough that you can re-derive its semantics in a single read; when in doubt, split it into several delegations. Spell out what \"correct\" means: the direction of the effect, edge cases, and the exact extent of terms (e.g. writing \"CJK\" must say whether kana and hangul are included). The worker stops and ends its turn (no tool call) with a written question when the order conflicts with reality; answer it and send again. Set `language` to the user's reply language (e.g. \"Chinese\"): they watch the worker's live output and read its report, so the worker must speak their language; omit `language` only when the user works in English. Also set a short `title` while writing the spec -- a few words capturing the intent (e.g. \"fix /status blank line\") -- used as the human-readable label of the delegation's start and done dividers; omit `title` to fall back to the order's first line.

Reset the worker when switching tasks, when the spec changed, or after it failed twice in a row (its context has accumulated wrong beliefs). Each send's result reports `rounds` (how many orders this worker has already taken) and `context_percent` (its live context fill); when the fill is high and the next task is still substantial, reset before re-delegating to give the worker a clean context. Reset discards the worker's process, not its products: file changes and merged diffs stay."""

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema(
            {
                "action": {"type": "string", "enum": ["send", "reset", "status"], "description": "Operation to perform"},
                "order": {
                    "type": "string",
                    "description": "Complete work order for action=send. Must stand alone (the worker cannot see this session's history) and state the goal, the files, the constraints, the facts already established (signatures, path:line anchors, decisions already made), how to verify, and what not to touch",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional step cap for this single delegation; defaults to the parent's runtime.max_agent_steps",
                },
                "language": {
                    "type": "string",
                    "description": "Reply language for all of the worker's visible output, live stream included (e.g. \"Chinese\"). Pass the user's language unless they work in English",
                },
                "title": {
                    "type": "string",
                    "description": "Short human-readable title summarizing the task (shown in the start/done dividers). Omit to fall back to the order's first line.",
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
        from minacode.runner import ToolRunner  # local: runner imports minacode.tools at module level

        payload = self.payload_dict()
        rows: list[tuple[str, str]] = []
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            rows.append(("title", ToolRunner.oneline(title.strip(), 120)))
        if order_row is not None:
            rows.append(order_row)
        language = payload.get("language")
        if isinstance(language, str) and language.strip():
            rows.append(("language", ToolRunner.oneline(language.strip(), 60)))
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

    def call(self) -> str:
        payload = self.single_dict_arg("Delegate requires named fields")
        action = str(payload.get("action") or "").strip()
        if action == "send":
            return self._send(payload)
        if action == "reset":
            return self._reset()
        if action == "status":
            return self._status()
        raise ToolError(f"unknown action: {action!r}")

    def _send(self, payload: Json) -> str:
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
        runner = getattr(self, "runner", None)
        if runner is None:
            raise ToolError("Delegate requires a tool runner")
        parent = self.session
        worker = parent.worker
        if worker is None:
            worker = self._spawn_worker(parent)
            parent.worker = worker
        # Rebuild the settings copy on every send: sharing the parent's RuntimeSettings object would
        # let a per-call max_steps override leak into the parent's budget, and a one-time copy would
        # miss runtime changes (/yolo, /set) between delegations.
        worker.settings = replace(parent.settings, max_steps=max_steps)
        agent = worker._agent
        if agent is None:
            # Local import: engine imports minacode.tools at module level (engine.py:30), so a
            # module-level `from minacode.engine import Agent` would cycle tools -> engine -> tools.
            # Same pattern as Tool.resolved_schemas and Session's local imports.
            from minacode.engine import Agent

            agent = Agent(worker, input_fn=runner.input_fn, output_fn=_worker_output(runner))
            if runner.model_stream is not None:
                # The parent loop's stream display; see ToolRunner.model_stream.
                # _worker_stream swallows `output_done`: the worker's own output_fn
                # already writes completed text into the parent scrollback, and the
                # loop's promote would write it a second time.
                agent.model.on_stream = _worker_stream(runner)
            # Reuse the parent's live region: serial delegation means only one stream at a time.
            agent.tools.live_start = runner.live_start
            agent.tools.live_output = runner.live_output
            # Same None-guarded wiring style as model_stream: the worker is a full session, so its
            # retry backoff, provider-side builtin calls, and automatic compaction surface in the
            # parent TUI like the main agent's. on_queue_flush is deliberately not wired (a worker
            # does not receive live follow-ups).
            if runner.retry_wait is not None:
                agent.model.on_retry_wait = runner.retry_wait
            if runner.builtin_call is not None:
                agent.model.on_builtin_call = runner.builtin_call
            if runner.compaction is not None:
                agent.context.on_compaction = runner.compaction
            worker._agent = agent
        started = time.monotonic()
        before_diffs = len(worker.turn_diffs)
        before_in = worker.usage.prompt_tokens
        before_out = worker.usage.completion_tokens
        # A visible boundary where the delegation starts: until the finish block there is nothing in
        # the scrollback that says who is talking, and the worker's own streamed lines do not. A
        # full-width rule whose yellow label reads the worker's live config and a one-line order
        # summary; without a wired worker_rule, the yellow [worker] line below stands in.
        from minacode.runner import ToolRunner  # local: runner imports minacode.tools at module level, same cycle as Agent above

        config = worker.config
        if runner.worker_rule is not None:
            runner.worker_rule(f"worker start · {config.active_provider}/{config.provider.model or '(no model)'} · {title or ToolRunner.oneline(order, 60)}")
        else:
            runner.output_fn(
                LogBlock(
                    [
                        LogLine(
                            "[worker]",
                            f"▶ {config.active_provider}/{config.provider.model or '(no model)'} · {title or ToolRunner.oneline(order, 200)}",
                            LogRole.WORKER,
                        )
                    ]
                )
            )
        try:
            with runner._active_worker.track(agent):
                answer = agent.run(order)
        except Exception as error:
            raise ToolError(f"worker failed: {error}") from error
        finally:
            # Merge diffs even when interrupted, or the user never sees what the worker did.
            self._merge_diffs(worker, parent, before_diffs)
        elapsed = time.monotonic() - started
        in_tokens = worker.usage.prompt_tokens - before_in
        out_tokens = worker.usage.completion_tokens - before_out
        files = ", ".join(sorted({diff.path for diff in worker.turn_diffs[before_diffs:]})) or "(none)"
        percent = worker.usage.context_percent(worker.state.context_percent)
        return "\n".join(
            [
                f'<Delegate action="send" steps="{worker.state.turn_step}" elapsed="{elapsed:.1f}s" files="{files}" stopped_at_max_steps="{str(agent.stopped_at_max_steps).lower()}" tokens="{in_tokens}/{out_tokens}" rounds="{worker.state.round_count}" context_percent="{percent}">',
                "<worker>",
                answer.rstrip(),
                "</worker>",
                "</Delegate>",
            ]
        )

    @staticmethod
    def _merge_diffs(worker: Session, parent: Session, start: int) -> None:
        for diff in worker.turn_diffs[start:]:
            parent.store_turn_diff(diff.key, diff.turn, diff.path, diff.diff, before=diff.before, after=diff.after, round=parent.state.round_count)

    def _spawn_worker(self, parent: Session) -> Session:
        uid = parent.uid + ".w"
        provider_name = parent.config.worker_provider or parent.config.active_provider
        # replace() is shallow, so the worker needs its own providers dict with a detached copy of
        # the entry it runs on: the /worker live-switch mutates that entry in place, and the
        # snapshot-load path below receives this same config so a resumed worker picks up the
        # current provider/model/reasoning overrides.
        provider = worker_provider_config(parent.config, provider_name)
        config = replace(parent.config, active_provider=provider_name, providers={**parent.config.providers, provider_name: provider})
        settings = replace(parent.settings)
        try:
            worker = SessionSnapshotStore.load(uid, config=config, settings=settings, cwd=parent.cwd)
        except Exception:  # noqa: BLE001 - missing or corrupt snapshot means "no worker yet", never an error.
            worker = Session(
                cwd=parent.cwd,
                system_info=parent.system_info,  # shared: skip a SystemInfo.detect
                config=config,
                settings=settings,
                created_at=parent.created_at,  # cache-critical: the Environment layer must stay identical across spawns
                uid=uid,
                skills=parent.skills,  # shared objects, never re-discovered
                mcp=parent.mcp,
            )
        # load only honors the snapshot's keys, so these return to their defaults; re-set them
        # after load or construction, never before. skills/mcp are shared objects from the parent
        # (never re-discovered or re-connected): a snapshot-loaded worker that kept its own would
        # spawn duplicate MCP processes and break its own cache-prefix reuse across delegations.
        worker.system_prompt = WORKER_PROMPT
        worker.tool_names = WORKER_TOOLS
        worker.listed = False
        worker.skills = parent.skills
        worker.mcp = parent.mcp
        return worker

    def _reset(self) -> str:
        parent = self.session
        worker = parent.worker
        uid = worker.uid if worker is not None else parent.uid + ".w"
        directory = SessionSnapshotStore.project_dir(parent.config.data_dir, parent.cwd)
        snapshot_path = os.path.join(directory, uid + ".jsonl")
        meta_path = os.path.join(directory, uid + SessionSnapshotStore.META_SUFFIX)
        assets_path = os.path.join(directory, uid + ".assets")
        if worker is None and not any(os.path.exists(path) for path in (snapshot_path, meta_path, assets_path)):
            return '<Delegate action="reset" alive="false"/>'

        # Reset owns the worker runtime, including background processes. Dropping the Session while
        # one of its Job handles is live would leave an unmanageable process behind.
        if worker is not None:
            for job in worker.jobs.values():
                try:
                    job.kill()
                except Exception as error:
                    raise ToolError(f"worker reset failed to stop {job.id}: {error}") from error

        # The delete path derives entirely from the parent's own uid; the model's arguments carry no
        # path, so the worst case is deleting this session's own worker, nothing else.
        try:
            os.unlink(snapshot_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ToolError(f"worker reset failed to delete its snapshot: {error}") from error
        with contextlib.suppress(OSError):
            os.unlink(meta_path)
        shutil.rmtree(assets_path, ignore_errors=True)

        # The worker Session owns its Agent (Session._agent); drop both only after durable context is
        # gone. Otherwise a failed unlink would report success and the next send would reload it.
        parent.worker = None
        return f'<Delegate action="reset" uid="{uid}"/>'

    def _status(self) -> str:
        worker = self.session.worker
        if worker is None:
            return '<Delegate action="status" alive="false"/>'
        percent = worker.usage.context_percent(worker.state.context_percent)
        last = next((str(message.get("content") or "") for message in reversed(worker.messages) if message.get("role") == "assistant"), "")
        return "\n".join(
            [
                f'<Delegate action="status" alive="true" provider="{worker.config.active_provider}" model="{worker.config.provider.model}" rounds="{worker.state.round_count}" context_percent="{percent}">',
                f"<last>{(last.strip()[:400] or '(no answer yet)')}</last>",
                "</Delegate>",
            ]
        )
