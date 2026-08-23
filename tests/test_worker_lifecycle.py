"""worker lifecycle (split from tests/test_worker_handoff.py)."""
import json
import os
import time
import pytest
from agent_harness import call, session
from minacode.base import SESSION_EVENT_KEY
from minacode.cli.worker import worker_command
from minacode.context import ContextManager
from minacode.engine import Agent
from minacode.prompts import SYSTEM_PROMPT, WORKER_PROMPT
from minacode.tools import TOOL_REGISTRY, Tool
from test_worker_handoff import FakeModelClient, _delegate_call, _delegate_runner, _delegate_session

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

def test_worker_agent_wires_lifecycle_callbacks(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "answer"}, [], "answer")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    retry_wait = lambda active: None
    builtin_call = lambda label, detail: None
    compaction = lambda active: None
    runner.retry_wait = retry_wait
    runner.builtin_call = builtin_call
    runner.compaction = compaction

    _delegate_call(parent, runner, action="send", order="work")

    agent = parent.worker._agent
    assert agent is not None
    assert agent.model.on_retry_wait is retry_wait
    assert agent.model.on_builtin_call is builtin_call
    assert agent.context.on_compaction is compaction

    # None-guard: without injected callbacks the worker's hooks stay unset.
    parent2 = _delegate_session(tmp_path)
    model2 = FakeModelClient([({"role": "assistant", "content": "answer"}, [], "answer")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model2)
    runner2 = _delegate_runner(parent2)
    _delegate_call(parent2, runner2, action="send", order="work")
    agent2 = parent2.worker._agent
    assert getattr(agent2.model, "on_retry_wait", None) is None
    assert getattr(agent2.model, "on_builtin_call", None) is None
    assert agent2.context.on_compaction is None

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

def test_delegate_failure_reports_envelope_and_settles_worker_history(tmp_path, monkeypatch):
    from minacode.base import ToolError
    from minacode.prompts import FAILED_TOOL_CALL_RESULT
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    # Step 1 edits a real file (its diff lands before the failure), step 2 dies inside tools.run,
    # step 3 answers the follow-up delegation.
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [call("Edit", ["f.txt", [{"op": "create", "content": "x"}]])],
                "editing",
            ),
            (
                {"role": "assistant", "content": ""},
                [call("Read", [{"path": "missing.txt", "ranges": [[0, 1]]}])],
                "",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run
    batches = {"n": 0}

    def run_then_fail(self, tool_calls, **kwargs):
        batches["n"] += 1
        if batches["n"] == 2:
            raise RuntimeError("provider timeout")
        return real_run(self, tool_calls, **kwargs)

    monkeypatch.setattr(ToolRunner, "run", run_then_fail)

    with pytest.raises(ToolError) as excinfo:
        _delegate_call(parent, runner, action="send", order="create f.txt then die")
    message = str(excinfo.value)
    assert "worker failed after 2 steps" in message
    assert 'alive="true"' in message
    assert 'rounds="1"' in message
    assert 'context_percent="0"' in message  # no usage budget with the fake model: the state fallback
    assert 'files="f.txt"' in message
    assert "Its context is kept: answer the problem and send again, or reset to discard this worker's process." in message

    # The diff was still merged: the parent sees the file, and the failure report names it.
    assert (tmp_path / "f.txt").read_text() == "x"
    assert any(diff.path == "f.txt" for diff in parent.turn_diffs)

    # The settled history has no unanswered tool call: the Read got a Failed result and the turn
    # ends with the failure marker (the same check settle_interrupted_turn runs).
    worker = parent.worker
    worker_messages = json.dumps(worker.messages)
    assert FAILED_TOOL_CALL_RESULT in worker_messages
    assert '"[This turn ended early: provider timeout]"' in worker_messages
    answered = {message.get("tool_call_id") for message in worker.messages if message.get("role") == "tool"}
    dangling = [
        call_id
        for message in worker.messages
        if message.get("role") == "assistant"
        for call_id in [call.get("id") for call in (message.get("tool_calls") or [])]
        if call_id and call_id not in answered
    ]
    assert dangling == []

    # The next delegation on the same worker goes out normally on the settled history.
    result = _delegate_call(parent, runner, action="send", order="continue")
    assert 'rounds="2"' in result
    assert "done" in result

def test_delegate_failure_after_a_call_ran_in_the_dying_batch(tmp_path, monkeypatch):
    from minacode.base import ToolCall, ToolError
    from minacode.prompts import FAILED_TOOL_CALL_RESULT
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    # Step 1 edits f.txt (answered), step 2 carries two calls and dies with the first one already
    # executed, step 3 answers the follow-up delegation.
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [call("Edit", ["f.txt", [{"op": "create", "content": "x"}]])],
                "editing",
            ),
            (
                {"role": "assistant", "content": ""},
                [
                    ToolCall("edit-2", "Edit", ["g.txt", [{"op": "create", "content": "y"}]]),
                    call("Read", [{"path": "missing.txt", "ranges": [[0, 1]]}]),
                ],
                "",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run
    batches = {"n": 0}

    def run_then_die(self, tool_calls, **kwargs):
        batches["n"] += 1
        if batches["n"] == 2:
            # The first call of the dying batch really runs against the worker's session (file on
            # disk, diff recorded); the second never starts. The batch then dies before any result
            # is committed to the turn.
            real_run(self, tool_calls[:1], **kwargs)
            raise RuntimeError("provider timeout")
        return real_run(self, tool_calls, **kwargs)

    monkeypatch.setattr(ToolRunner, "run", run_then_die)

    with pytest.raises(ToolError) as excinfo:
        _delegate_call(parent, runner, action="send", order="create f.txt then die mid-batch")
    message = str(excinfo.value)
    assert "worker failed after 2 steps" in message
    # The edit from the dying batch is not lost: its diff was merged and the envelope names it.
    assert 'files="f.txt, g.txt"' in message

    assert (tmp_path / "f.txt").read_text() == "x"
    assert (tmp_path / "g.txt").read_text() == "y"

    # Every call of the dying batch is answered with the Failed result — the executed one included,
    # since its output died with the crash — and none dangles for the next request to reject.
    worker = parent.worker
    worker_messages = json.dumps(worker.messages)
    assert '"[This turn ended early: provider timeout]"' in worker_messages
    assert worker_messages.count(FAILED_TOOL_CALL_RESULT) == 2
    answered = {message.get("tool_call_id") for message in worker.messages if message.get("role") == "tool"}
    dangling = [
        call_id
        for message in worker.messages
        if message.get("role") == "assistant"
        for call_id in [call.get("id") for call in (message.get("tool_calls") or [])]
        if call_id and call_id not in answered
    ]
    assert dangling == []

    # The next delegation on the same worker goes out normally on the settled history.
    result = _delegate_call(parent, runner, action="send", order="continue")
    assert 'rounds="2"' in result and "done" in result

def test_delegate_status_reports_last_failure_until_a_success(tmp_path, monkeypatch):
    from minacode.base import ToolError
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": ""},
                [call("Read", [{"path": "missing.txt", "ranges": [[0, 1]]}])],
                "",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run
    monkeypatch.setattr(ToolRunner, "run", lambda self, tool_calls, **kwargs: (_ for _ in ()).throw(RuntimeError("provider timeout")))
    with pytest.raises(ToolError):
        _delegate_call(parent, runner, action="send", order="first")

    status = _delegate_call(parent, runner, action="status")
    assert 'last_error="provider timeout"' in status
    assert 'last_error_round="1"' in status
    assert 'rounds="1"' in status
    assert 'alive="true"' in status

    # A successful send clears the remembered failure.
    monkeypatch.setattr(ToolRunner, "run", real_run)
    _delegate_call(parent, runner, action="send", order="second")
    assert parent.worker.state.last_error == "" and parent.worker.state.last_error_round == 0
    status = _delegate_call(parent, runner, action="status")
    assert "last_error" not in status
    assert 'rounds="2"' in status

def test_delegate_failure_bounds_and_sanitizes_the_error_text(tmp_path, monkeypatch):
    from minacode.base import ToolError
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": ""},
                [call("Read", [{"path": "missing.txt", "ranges": [[0, 1]]}])],
                "",
            ),
        ]
    )
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    body = 'HTTP 400: {"error": "bad "quoted" thing"}\nsecond line\n' + "x" * 3000
    monkeypatch.setattr(ToolRunner, "run", lambda self, tool_calls, **kwargs: (_ for _ in ()).throw(RuntimeError(body)))
    with pytest.raises(ToolError):
        _delegate_call(parent, runner, action="send", order="first")

    # The status tag stays parseable: one line, no bare double quote inside the attribute value.
    status = _delegate_call(parent, runner, action="status")
    head = status.splitlines()[0]
    assert head.endswith(">") and head.count('"') % 2 == 0
    value = head.split('last_error="', 1)[1].split('"', 1)[0]
    assert value.startswith("HTTP 400: {'error': 'bad 'quoted' thing'}")
    assert "\n" not in value and len(value) <= 200

    # The marker in permanent history is bounded, so a whole HTTP body does not ride every later
    # request; it still names where the turn stopped.
    marker = next(
        message["content"]
        for message in parent.worker.messages
        if isinstance(message.get("content"), str) and message["content"].startswith("[This turn ended early:")
    )
    assert len(marker) <= 330 and "\n" not in marker
    assert "HTTP 400" in marker and marker.endswith("]")

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

def test_worker_reset_appends_event_message(tmp_path):
    from minacode.cli import CommandLoop
    from minacode.engine import Agent
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
