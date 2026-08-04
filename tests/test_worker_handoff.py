"""Worker handoff: the second in-process session a parent delegates to (see DESIGN.md).

Coverage follows WORKER_HANDOFF_PLAN.txt section 9; each numbered test maps to that list.
"""

import json
import os
import time

import pytest
from agent_harness import call, session

from minacode.base import SESSION_EVENT_KEY
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.prompts import SYSTEM_PROMPT
from minacode.tools import TOOL_REGISTRY, Tool


def _requested_system(tmp_path, custom=None):
    s = session(tmp_path)
    if custom is not None:
        s.system_prompt = custom
    agent = Agent(s, output_fn=lambda text: None)
    request = agent.prepare_request([{"role": "user", "content": "hi"}])
    return s, request.messages[0]["content"]


# 1. tool_names filtering: only the whitelisted schemas, in TOOL_REGISTRY order; empty tuple is
#    exactly the unfiltered behavior.
def test_tool_names_filter_resolved_schemas_and_keep_registry_order(tmp_path):
    s = session(tmp_path)
    names = ("Read", "Edit", "Bash")
    s.tool_names = names
    resolved = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    assert resolved == [name for name in TOOL_REGISTRY if name in names]

    # Empty tuple = no filtering; identical to a session that never set tool_names.
    s.tool_names = ()
    unfiltered = [schema["function"]["name"] for schema in Tool.resolved_schemas(s)]
    plain = [schema["function"]["name"] for schema in Tool.resolved_schemas(session(tmp_path))]
    assert unfiltered == plain
    assert "Read" in unfiltered and "Edit" in unfiltered and "Bash" in unfiltered


# 2. system_prompt comes from the session: the request payload's system content changes with it,
#    and the parent default is unchanged.
def test_system_prompt_comes_from_session(tmp_path):
    _, system = _requested_system(tmp_path, custom="CUSTOM WORKER ROLE")
    assert system == "CUSTOM WORKER ROLE"

    _, parent_system = _requested_system(tmp_path)
    assert parent_system == SYSTEM_PROMPT.strip()


def test_system_prompt_default_matches_prompts_module(tmp_path):
    _, system = _requested_system(tmp_path)
    assert system == SYSTEM_PROMPT.strip()
    assert ContextManager(session(tmp_path)).model_messages(SYSTEM_PROMPT)[0]["content"] == SYSTEM_PROMPT.strip()


# 3. workers stay out of listings and never claim the latest pointer.
def test_worker_snapshot_hidden_from_listing_and_latest(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()  # latest -> parent.uid
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()

    assert worker.uid.endswith(".w")
    assert SessionSnapshotStore.read_latest(SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))) == parent.uid
    assert not os.path.exists(SessionSnapshotStore.meta_path(parent.config.data_dir, str(tmp_path), worker.uid))
    entries = SessionSnapshotStore.list_sessions(parent.config.data_dir, cwd=str(tmp_path))
    assert all(entry.uid != worker.uid for entry in entries)
    assert any(entry.uid == parent.uid for entry in entries)
    # `-c` still resolves to the parent even though the worker log is newer on disk.
    assert SessionSnapshotStore.latest_uid(parent.config.data_dir, cwd=str(tmp_path)) == parent.uid


def test_clean_expired_removes_worker_when_parent_expires_later_in_scan(tmp_path, monkeypatch):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.settings.session_retention_days = 1
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()  # create first: the worker is visited before its parent below
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    directory = SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))
    parent_path = os.path.join(directory, parent.uid + ".jsonl")
    worker_path = os.path.join(directory, worker.uid + ".jsonl")
    old = time.time() - 3 * 86400
    os.utime(parent_path, (old, old))  # parent expired; worker remains fresh

    real_scandir = os.scandir

    def worker_first(path):
        entries = list(real_scandir(path))
        return iter(sorted(entries, key=lambda entry: (not entry.name.endswith(".w.jsonl"), entry.name)))

    monkeypatch.setattr("minacode.session.os.scandir", worker_first)
    cleaner = session(tmp_path)
    cleaner.settings.session_retention_days = 1

    assert SessionSnapshotStore.clean_expired(cleaner) >= 2
    assert not os.path.isfile(parent_path)
    assert not os.path.isfile(worker_path)


# 10. two registration gates: Delegate appears only when both [worker] provider is configured and
#     runtime.worker is on; closing is not reset (the snapshot stays).
def test_delegate_registration_gates(tmp_path):
    s = session(tmp_path)

    def names():
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    assert "Delegate" not in names()
    s.settings.worker = True
    assert "Delegate" not in names()  # no [worker] provider yet
    s.config.worker_provider = "default"
    assert "Delegate" in names()
    s.settings.worker = False
    assert "Delegate" not in names()  # the gate is per-session stable: worker off drops the schema
    s.settings.worker = True
    assert "Delegate" in names()


def test_worker_config_parsing_and_validation(tmp_path):
    from minacode.base import Config, ConfigError, RuntimeSettings

    config = Config.from_dict({"worker": {"provider": "fast"}, "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m"}}})
    assert config.worker_provider == "fast"
    assert RuntimeSettings.from_dict({"runtime": {"worker": True}}).worker is True
    assert RuntimeSettings.from_dict({}).worker is False
    with pytest.raises(ConfigError, match="worker.provider"):
        Config.from_dict({"worker": {"provider": "nope"}})


# --- Delegation (steps 4-5): the worker is driven through DelegateTool with a scripted model. ---


class FakeModelClient:
    """Stands in for minacode.engine.ModelClient: records every request and replays a script of
    (assistant, tool_calls, content) triples, so the worker's loop is exercised without HTTP."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.received_tools = []

    def request(self, messages, request_tools=None):
        self.requests.append(messages)
        self.received_tools.append(request_tools)
        return self.script.pop(0)

    def estimated_request_tokens(self, messages, tools=None):
        return sum(len(str(message)) for message in messages) // 4

    def cancel(self):
        pass


def _delegate_session(tmp_path):
    parent = session(tmp_path)
    parent.config.worker_provider = "default"
    parent.settings.worker = True
    return parent


def _delegate_call(parent, runner, **args):
    from minacode.tools.delegate import DelegateTool

    tool = DelegateTool(parent, [args])
    tool.runner = runner
    return tool.call()


def _delegate_runner(parent):
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    return ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)


# 4. context continuity: the second delegation's request carries the first order and its answer.
def test_delegate_context_continuity(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="order one")
    _delegate_call(parent, runner, action="send", order="order two")

    assert len(model.requests) == 2
    assert parent.worker is not None
    second = json.dumps(model.requests[1])
    assert "order one" in second and "answer one" in second
    assert "order two" in second
    assert model.requests[1][0] == model.requests[0][0]  # same system prompt across delegations


# 5. reset: after reset, the next send carries no prior history, and the snapshot file is gone.
def test_delegate_reset_clears_context_and_snapshot(tmp_path, monkeypatch):
    from minacode.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
            ({"role": "assistant", "content": "answer fresh"}, [], "answer fresh"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="order one")
    worker_uid = parent.worker.uid
    _delegate_call(parent, runner, action="send", order="order two")
    assert "order one" in json.dumps(model.requests[1])

    result = _delegate_call(parent, runner, action="reset")
    assert 'action="reset"' in result
    assert parent.worker is None
    directory = SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))
    assert not os.path.exists(os.path.join(directory, worker_uid + ".jsonl"))

    _delegate_call(parent, runner, action="send", order="fresh start")
    fresh = json.dumps(model.requests[-1])
    assert "order one" not in fresh and "order two" not in fresh
    assert "fresh start" in fresh


def test_delegate_reset_stops_worker_jobs_before_dropping_runtime(tmp_path):
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    parent.worker = worker

    class Job:
        id = "job.1"
        killed = False

        def kill(self):
            self.killed = True

    job = Job()
    worker.jobs[job.id] = job

    result = _delegate_call(parent, _delegate_runner(parent), action="reset")

    assert 'action="reset"' in result
    assert job.killed is True
    assert parent.worker is None


def test_delegate_reset_keeps_worker_when_snapshot_delete_fails(tmp_path, monkeypatch):
    from minacode.base import ToolError
    from minacode.session import Session, SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    parent.worker = worker
    snapshot = SessionSnapshotStore.session_path(parent.config.data_dir, str(tmp_path), worker.uid)
    real_unlink = os.unlink

    def fail_snapshot(path):
        if os.fspath(path) == snapshot:
            raise PermissionError("read only")
        return real_unlink(path)

    monkeypatch.setattr("minacode.tools.delegate.os.unlink", fail_snapshot)

    with pytest.raises(ToolError, match="failed to delete its snapshot"):
        _delegate_call(parent, _delegate_runner(parent), action="reset")
    assert parent.worker is worker
    assert os.path.isfile(snapshot)


def test_delegate_reset_deletes_disk_only_worker_after_parent_resume(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    snapshot = SessionSnapshotStore.session_path(parent.config.data_dir, str(tmp_path), worker.uid)
    assert parent.worker is None and os.path.isfile(snapshot)

    result = _delegate_call(parent, _delegate_runner(parent), action="reset")

    assert 'action="reset"' in result
    assert not os.path.exists(snapshot)


@pytest.mark.parametrize("max_steps", [0, -1, True, "3"])
def test_delegate_rejects_invalid_max_steps(tmp_path, max_steps):
    from minacode.base import ToolError
    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    with pytest.raises(ToolError, match="integer >= 1"):
        DelegateTool(parent, [{"action": "send", "order": "work", "max_steps": max_steps}]).call()


# 6. diff reflux: an Edit inside the worker shows up in the parent's turn_diffs.
def test_delegate_merges_worker_diffs_into_parent(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [
                    call(
                        "Edit",
                        [
                            "f.txt",
                            [
                                {
                                    "op": "create",
                                    "content": "x\
",
                                }
                            ],
                        ],
                    )
                ],
                "editing",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    result = _delegate_call(parent, runner, action="send", order="create f.txt")

    assert "f.txt" in result
    assert (tmp_path / "f.txt").read_text() == "x"
    assert any(diff.path == "f.txt" for diff in parent.turn_diffs)


# 7. interrupt: cancellation lands on the worker, its turn settles with every tool call matched,
#    and the diff merge still runs.
def test_delegate_interrupt_settles_and_merges_diffs(tmp_path, monkeypatch):
    import threading

    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    started = threading.Event()
    cancelled = threading.Event()

    class SlowModel(FakeModelClient):
        def __init__(self):
            super().__init__([])
            self.requests = []

        def request(self, messages, request_tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return (
                    {"role": "assistant", "content": "editing"},
                    [
                        call(
                            "Edit",
                            [
                                "f.txt",
                                [
                                    {
                                        "op": "create",
                                        "content": "x\
",
                                    }
                                ],
                            ],
                        )
                    ],
                    "editing",
                )
            started.set()
            cancelled.wait(5)
            raise KeyboardInterrupt

        def cancel(self):
            pass

    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: SlowModel())
    runner = _delegate_runner(parent)
    tool = DelegateTool(parent, [{"action": "send", "order": "create f.txt"}])
    tool.runner = runner
    raised = []

    def run():
        try:
            tool.call()
        except BaseException as error:  # noqa: BLE001 - the delegate must surface the interrupt.
            raised.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    runner.cancel()  # fans out to the worker agent
    cancelled.set()
    thread.join(5)

    assert not thread.is_alive()
    assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
    assert (tmp_path / "f.txt").read_text() == "x"
    assert any(diff.path == "f.txt" for diff in parent.turn_diffs)
    # Every tool call the model issued has a matched result in the settled turn.
    worker_messages = json.dumps(parent.worker.messages)
    assert "f.txt" in worker_messages
    assert '"role": "tool"' in worker_messages


# 8. cache prefix: system + tools + Environment are byte-identical across delegations.
def test_worker_cache_prefix_stable_across_delegations(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="order one")
    _delegate_call(parent, runner, action="send", order="order two")
    assert model.requests[0][:3] == model.requests[1][:3]


# 12. settings isolation: a per-call max_steps override never touches the parent's budget, and the
#     worker sees the parent's current settings on every send.
def test_delegate_settings_isolated_and_fresh(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    parent.settings.max_steps = 7
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o", max_steps=3)
    assert parent.settings.max_steps == 7  # the parent's budget is untouched
    assert parent.worker.settings.max_steps == 3
    assert parent.worker.settings is not parent.settings

    parent.settings.yolo = True
    _delegate_call(parent, runner, action="send", order="o")
    assert parent.worker.settings.yolo is True  # fresh copy sees the runtime change


# 11. user reset: /worker reset appends a SESSION_EVENT_KEY message to the parent's history tail, and
#     the message reaches the next request (render-hidden, never filtered from the model history).
def test_worker_reset_appends_event_message(tmp_path):
    from minacode.engine import Agent
    from minacode.loop import CommandLoop
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    parent.worker = worker
    parent.save_snapshot()
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    loop.command("/worker reset")

    assert parent.worker is None
    assert parent.messages[-1].get(SESSION_EVENT_KEY) == "worker_reset"
    assert parent.messages[-1].get("role") == "user"
    request = agent.prepare_request([{"role": "user", "content": "continue"}])
    assert any(message.get("role") == "user" and "starts from scratch" in str(message.get("content")) for message in request.messages)


def test_status_bar_shows_worker_segment(tmp_path):
    from minacode.render import StatusBar
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    bar = StatusBar(parent)
    texts = [text for text, _ in bar.entries(show_elapsed=False)]
    parent_lead = parent.config.active_provider + "/" + (parent.config.provider.model.rsplit("/", 1)[-1] or "(no model)")
    assert parent_lead in texts and "worker" not in texts

    # A live worker leads with the marker, keeps the worker's provider/model unsuffixed, and
    # ctx/cache come from the worker's usage.
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker
    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    worker.usage.last_cached_prompt_tokens = 25
    texts = [text for text, _ in bar.entries(show_elapsed=False)]
    assert texts[0] == "worker"
    assert parent_lead in texts and not any(text.endswith("·worker") for text in texts)
    assert "ctx 50% · cache 50%" in texts


def test_working_divider_marks_inflight_worker(tmp_path):
    from minacode.engine import Agent
    from minacode.loop import CommandLoop
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    def label():
        return "".join(text for _style, text in loop.queue_divider_fragments())

    assert "· worker" not in label()
    worker._active_turn_messages.append({"role": "user", "content": "order"})
    assert "· worker" in label()


# The engine publishes the model's own text as bare strings (content beside tool calls), so the
# worker output wrapper must wrap them into LogLine items: LogBlock.walk crashes on a str item.
def test_worker_output_wraps_model_text_for_the_log_stream(tmp_path, monkeypatch):
    from minacode.base import LogBlock, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    tool_call = ToolCall(id="call1", name="Note", args={"action": "view"})
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "thinking out loud"}, [tool_call], "thinking out loud"),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    _delegate_call(parent, runner, action="send", order="o")

    assert outputs, "the worker turn produced no output"
    rendered = [str(block) for block in outputs if isinstance(block, LogBlock)]  # str items raised before the fix
    assert any("thinking out loud" in text for text in rendered)


# 12. the Agent lives on the worker Session, not in a module-level dict: a fresh worker object
#     (after /resume re-enters the same parent) always gets a fresh Agent bound to itself.
def test_agent_lives_on_worker_and_is_rebuilt_with_it(tmp_path, monkeypatch):
    from minacode.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "one"}, [], "one")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o")

    first_worker = parent.worker
    first_agent = first_worker._agent
    assert first_agent is not None
    assert first_agent.session is first_worker

    # /resume re-enters the same parent: a fresh parent object, worker rebuilt from the snapshot.
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    assert fresh.worker is None
    runner = _delegate_runner(fresh)
    _delegate_call(fresh, runner, action="send", order="o")

    second_worker = fresh.worker
    assert second_worker is not first_worker
    assert second_worker._agent is not first_agent  # the old Agent died with the old worker object
    assert second_worker._agent.session is second_worker  # and the new one is bound to the new object


# 13. a snapshot-restored worker shares the parent's skills/mcp objects (review point 2): load
#     rebuilds its own copies, so the delegate caller must re-attach the shared ones.
def test_snapshot_restored_worker_shares_parent_skills_and_mcp(tmp_path, monkeypatch):
    from minacode.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "one"}, [], "one")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o")
    parent.worker.messages.append({"role": "user", "content": "worker request"})
    parent.worker.save_snapshot()

    # Resume: the worker now comes back through SessionSnapshotStore.load, not the fresh-branch.
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    runner = _delegate_runner(fresh)
    _delegate_call(fresh, runner, action="send", order="o")
    worker = fresh.worker
    assert worker.skills is fresh.skills
    assert worker.mcp is fresh.mcp


# 14. stopped_at_max_steps in the envelope is a runtime fact from the Agent, never the answer's wording.
def test_delegate_envelope_reports_max_steps_from_runtime_fact(tmp_path, monkeypatch):
    from minacode.engine import Agent

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)

    def run_stopped(self, order):
        self.stopped_at_max_steps = True
        return "done"

    def run_normal(self, order):
        self.stopped_at_max_steps = False
        return "Stopped after max_agent_steps=3 (cosmetic wording only)"

    monkeypatch.setattr(Agent, "run", run_stopped)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'stopped_at_max_steps="true"' in result

    monkeypatch.setattr(Agent, "run", run_normal)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'stopped_at_max_steps="false"' in result  # the words are irrelevant; the fact is not set


# 15. resolve_uid prefix search never resolves to a worker snapshot (review point 5): the parent's
#     uid prefix must resolve to the parent alone, without ambiguity.
def test_resolve_uid_prefix_skips_worker_snapshot(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()

    resolved = SessionSnapshotStore.resolve_uid(parent.uid[:12], parent.config.data_dir, str(tmp_path))
    assert resolved == parent.uid

    # And an exact worker uid is not reachable through the prefix path either: it would require
    # typing the full uid, which the user never sees in listings.
    resolved_worker = SessionSnapshotStore.resolve_uid(worker.uid, parent.config.data_dir, str(tmp_path))
    assert resolved_worker == worker.uid  # explicit full uid still works, by design


# 16. the two blocks that must never drift between SYSTEM_PROMPT and WORKER_PROMPT are spliced from
#     the same module-level constants, so a wording change in one is a change in both (this is a
#     composition contract, not a prompt-literal test).
def test_worker_prompt_shares_language_and_secret_rules_with_parent():
    from minacode.prompts import LANGUAGE_RULES, SECRET_RULES, SYSTEM_PROMPT, WORKER_PROMPT

    assert LANGUAGE_RULES in SYSTEM_PROMPT
    assert LANGUAGE_RULES in WORKER_PROMPT
    assert SECRET_RULES in SYSTEM_PROMPT
    assert SECRET_RULES in WORKER_PROMPT


# 17. The two readable role prompts keep their role-specific behavior without exposing the
#     implementation as a collection of positional fragments.
def test_worker_prompt_does_not_inherit_parent_review_or_terminal_output():
    from minacode.prompts import SYSTEM_PROMPT, WORKER_PROMPT

    assert "REVIEW:" in SYSTEM_PROMPT and "REVIEW:" not in WORKER_PROMPT
    assert "terminal scrollback" in SYSTEM_PROMPT and "terminal scrollback" not in WORKER_PROMPT
    assert "You write for the delegator" in WORKER_PROMPT and "You write for the delegator" not in SYSTEM_PROMPT
    for unavailable in ("Ask", "NextHints", "ViewImage"):
        assert unavailable not in WORKER_PROMPT


# 18. The worker prompt may name only tools in its reduced tool set.
def test_prompts_never_name_tools_outside_their_toolset():
    import re

    from minacode.prompts import WORKER_PROMPT
    from minacode.tools import TOOL_REGISTRY
    from minacode.tools.delegate import WORKER_TOOLS

    def mentioned(prompt):
        return {name for name in TOOL_REGISTRY if re.search(rf"\b{re.escape(name)}\b", prompt)}

    worker_mentioned = mentioned(WORKER_PROMPT)
    assert worker_mentioned <= set(WORKER_TOOLS), worker_mentioned - set(WORKER_TOOLS)


# 19. refactor-stability sentinel for the parent prompt: pure refactors of the prompt composition
#     must not change SYSTEM_PROMPT's text (its cache-prefix stability and this contract depend on
#     it). A deliberate, release-level edit to the parent prompt updates this hash in the same
#     commit and records the change in the changelog.
def test_system_prompt_stable_across_refactors():
    import hashlib

    from minacode.prompts import SYSTEM_PROMPT

    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == "6cbb412a08515bfa24ec8da873af3a62d33493042c0419ba55bb386e20cc0154"


# 20. yolo covers editing files and running commands: those mistakes show up in the diff or the
#     command output at once. A delegation's mistake is the order text, and it only surfaces a whole
#     worker round later, so send is confirmed even under yolo. status and reset stay under it.
def test_delegate_send_is_confirmed_even_under_yolo(tmp_path):
    from minacode.tools import DelegateTool

    s = session(tmp_path)
    s.settings.yolo = True

    send = DelegateTool(s, [{"action": "send", "order": "do the thing"}])
    assert send.needs_confirmation() is True
    assert send.always_confirms() is True

    for action in ("status", "reset"):
        other = DelegateTool(s, [{"action": action}])
        assert other.always_confirms() is False, action

    # Every other mutating tool keeps yolo's meaning: only Delegate opts out.
    from minacode.tools import EditTool

    assert EditTool(s, ["a.py", []]).always_confirms() is False
