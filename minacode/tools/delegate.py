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

from minacode.base import Json, LogBlock, ToolError
from minacode.prompts import WORKER_PROMPT
from minacode.session import Session, SessionSnapshotStore
from minacode.tools.base import Tool

if TYPE_CHECKING:
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


class DelegateTool(Tool):
    NAME = "Delegate"
    runner: ToolRunner | None = None  # injected by ToolRunner.call_tool; the runner owns the cancel wiring
    MUTATES = True  # the delegation itself needs confirmation; the worker's own tools still confirm per call
    STORES_RESULT = True  # results can be long; keep them in tr.N for Recall
    PRODUCES_MODEL_OBSERVATION = False
    DESCRIPTION = """Hand a bounded task to the worker: a second in-process minacode session on its own provider, with its own system prompt and tool set. The worker cannot see this session's history -- only the order text and its own prior history -- so the order must stand alone.

Write the order to state: the goal, the files it touches, the constraints, how to verify, and the boundaries (what not to touch). Keep one delegation small enough that you can re-derive its semantics in a single read; when in doubt, split it into several delegations. Spell out what \"correct\" means: the direction of the effect, edge cases, and the exact extent of terms (e.g. writing \"CJK\" must say whether kana and hangul are included). The worker stops and ends its turn (no tool call) with a written question when the order conflicts with reality; answer it and send again.

Reset the worker when switching tasks, when the spec changed, or after it failed twice in a row (its context has accumulated wrong beliefs). Reset discards the worker's process, not its products: file changes and merged diffs stay."""

    @classmethod
    def params_schema(cls) -> Json:
        return cls.object_schema(
            {
                "action": {"type": "string", "enum": ["send", "reset", "status"], "description": "Operation to perform"},
                "order": {
                    "type": "string",
                    "description": "Complete work order for action=send; must stand alone, the worker cannot see this session's history",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional step cap for this single delegation; defaults to the parent's runtime.max_agent_steps",
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
        payload = self.args[0] if len(self.args) == 1 and isinstance(self.args[0], dict) else {}
        return str(payload.get("action") or "").strip() == "send"

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
        order = str(payload.get("order") or "").strip()
        if not order:
            raise ToolError("Delegate send requires an order")
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
        worker.settings = replace(parent.settings, max_steps=int(payload.get("max_steps") or parent.settings.max_steps))
        agent = worker._agent
        if agent is None:
            # Local import: engine imports minacode.tools at module level (engine.py:30), so a
            # module-level `from minacode.engine import Agent` would cycle tools -> engine -> tools.
            # Same pattern as Tool.resolved_schemas and Session's local imports.
            from minacode.engine import Agent

            agent = Agent(worker, input_fn=runner.input_fn, output_fn=lambda block: runner.output_fn(LogBlock([block])))
            # Reuse the parent's live region: serial delegation means only one stream at a time.
            agent.tools.live_start = runner.live_start
            agent.tools.live_output = runner.live_output
            worker._agent = agent
        started = time.monotonic()
        before_diffs = len(worker.turn_diffs)
        try:
            with runner._active_worker.track(agent):
                answer = agent.run(order)
        except Exception as error:
            raise ToolError(f"worker failed: {error}") from error
        finally:
            # Merge diffs even when interrupted, or the user never sees what the worker did.
            self._merge_diffs(worker, parent, before_diffs)
        elapsed = time.monotonic() - started
        files = ", ".join(sorted({diff.path for diff in worker.turn_diffs[before_diffs:]})) or "(none)"
        return "\n".join(
            [
                f'<Delegate action="send" steps="{worker.state.turn_step}" elapsed="{elapsed:.1f}s" files="{files}" stopped_at_max_steps="{str(agent.stopped_at_max_steps).lower()}">',
                "<worker>",
                answer.rstrip(),
                "</worker>",
                "</Delegate>",
            ]
        )

    @staticmethod
    @staticmethod
    def _merge_diffs(worker: Session, parent: Session, start: int) -> None:
        for diff in worker.turn_diffs[start:]:
            parent.store_turn_diff(diff.key, diff.turn, diff.path, diff.diff, before=diff.before, after=diff.after, round=parent.state.round_count)

    def _spawn_worker(self, parent: Session) -> Session:
        uid = parent.uid + ".w"
        provider_name = parent.config.worker_provider or parent.config.active_provider
        config = replace(parent.config, active_provider=provider_name)
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
        if worker is None:
            return '<Delegate action="reset" alive="false"/>'
        parent.worker = None
        # The worker Session owns its Agent (Session._agent); dropping the handle releases both.
        # The delete path derives entirely from the parent's own uid; the model's arguments carry no
        # path, so the worst case is deleting this session's own worker, nothing else.
        directory = SessionSnapshotStore.project_dir(parent.config.data_dir, parent.cwd)
        for suffix in (".jsonl", SessionSnapshotStore.META_SUFFIX):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(directory, worker.uid + suffix))
        shutil.rmtree(os.path.join(directory, worker.uid + ".assets"), ignore_errors=True)
        return f'<Delegate action="reset" uid="{worker.uid}"/>'

    def _status(self) -> str:
        worker = self.session.worker
        if worker is None:
            return '<Delegate action="status" alive="false"/>'
        usage = worker.usage
        percent = min(100, usage.last_prompt_tokens * 100 // usage.last_prompt_budget) if usage.last_prompt_budget else worker.state.context_percent
        last = next((str(message.get("content") or "") for message in reversed(worker.messages) if message.get("role") == "assistant"), "")
        return "\n".join(
            [
                f'<Delegate action="status" alive="true" provider="{worker.config.active_provider}" model="{worker.config.provider.model}" rounds="{worker.state.round_count}" context_percent="{percent}">',
                f"<last>{(last.strip()[:400] or '(no answer yet)')}</last>",
                "</Delegate>",
            ]
        )
