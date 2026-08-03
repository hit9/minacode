"""Worker handoff: the second in-process session a parent delegates to (see DESIGN.md).

Coverage follows WORKER_HANDOFF_PLAN.txt section 9; each numbered test maps to that list.
"""

import json
import os

import pytest
from agent_harness import call, session

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
    entries = SessionSnapshotStore.list_sessions(parent.config.data_dir, cwd=str(tmp_path))
    assert all(entry.uid != worker.uid for entry in entries)
    assert any(entry.uid == parent.uid for entry in entries)
    # `-c` still resolves to the parent even though the worker log is newer on disk.
    assert SessionSnapshotStore.latest_uid(parent.config.data_dir, cwd=str(tmp_path)) == parent.uid


def test_clean_expired_removes_orphaned_worker_with_parent(tmp_path):
    from minacode.session import Session, SessionSnapshotStore

    parent = session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.settings.session_retention_days = 1
    parent.save_snapshot()
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    directory = SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))

    # Simulate the parent having expired: drop its log, keep the worker's fresh mtime.
    parent_path = os.path.join(directory, parent.uid + ".jsonl")
    worker_path = os.path.join(directory, worker.uid + ".jsonl")
    assert os.path.isfile(parent_path) and os.path.isfile(worker_path)
    os.unlink(parent_path)

    assert SessionSnapshotStore.clean_expired(parent) >= 1
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


# 6. diff reflux: an Edit inside the worker shows up in the parent's turn_diffs.
def test_delegate_merges_worker_diffs_into_parent(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "editing"}, [call("Edit", ["f.txt", [{"op": "create", "content": "x\
"}]])], "editing"),
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
                return ({"role": "assistant", "content": "editing"}, [call("Edit", ["f.txt", [{"op": "create", "content": "x\
"}]])], "editing")
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
