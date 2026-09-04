"""worker lifecycle (split from tests/test_worker_handoff.py)."""
import json
import os

import pytest
from agent_harness import call
from test_worker_handoff import FakeModelClient, _delegate_call, _delegate_runner, _delegate_session

from wizolt.base import SESSION_EVENT_KEY, ToolError


async def test_delegate_context_continuity(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="order one")
    await _delegate_call(parent, runner, action="send", order="order two")

    assert len(model.requests) == 2
    assert parent.worker is not None
    second = json.dumps(model.requests[1])
    assert "order one" in second and "answer one" in second
    assert "order two" in second
    assert model.requests[1][0] == model.requests[0][0]  # same system prompt across delegations

async def test_worker_agent_wires_lifecycle_callbacks(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "answer"}, [], "answer")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    retry_wait = lambda active: None
    builtin_call = lambda label, detail: None
    compaction = lambda active: None
    runner.retry_wait = retry_wait
    runner.builtin_call = builtin_call
    runner.compaction = compaction

    await _delegate_call(parent, runner, action="send", order="work")

    agent = parent.worker._agent
    assert agent is not None
    assert agent.model.on_retry_wait is retry_wait
    assert agent.model.on_builtin_call is builtin_call
    assert agent.context.on_compaction is compaction

    # None-guard: without injected callbacks the worker's hooks stay unset.
    parent2 = _delegate_session(tmp_path)
    model2 = FakeModelClient([({"role": "assistant", "content": "answer"}, [], "answer")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model2)
    runner2 = _delegate_runner(parent2)
    await _delegate_call(parent2, runner2, action="send", order="work")
    agent2 = parent2.worker._agent
    assert getattr(agent2.model, "on_retry_wait", None) is None
    assert getattr(agent2.model, "on_builtin_call", None) is None
    assert agent2.context.on_compaction is None

async def test_delegate_reset_clears_context_and_snapshot(tmp_path, monkeypatch):
    from wizolt.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
            ({"role": "assistant", "content": "answer fresh"}, [], "answer fresh"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="order one")
    worker_uid = parent.worker.uid
    await _delegate_call(parent, runner, action="send", order="order two")
    assert "order one" in json.dumps(model.requests[1])

    result = await _delegate_call(parent, runner, action="reset")
    assert 'action="reset"' in result
    assert parent.worker is None
    directory = SessionSnapshotStore.project_dir(parent.config.data_dir, str(tmp_path))
    assert not os.path.exists(os.path.join(directory, worker_uid + ".jsonl"))

    await _delegate_call(parent, runner, action="send", order="fresh start")
    fresh = json.dumps(model.requests[-1])
    assert "order one" not in fresh and "order two" not in fresh
    assert "fresh start" in fresh

async def test_delegate_reset_stops_worker_jobs_before_dropping_runtime(tmp_path):
    from wizolt.session import Session

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

    result = await _delegate_call(parent, _delegate_runner(parent), action="reset")

    assert 'action="reset"' in result
    assert job.killed is True
    assert parent.worker is None

async def test_delegate_reset_keeps_worker_when_snapshot_delete_fails(tmp_path, monkeypatch):
    from wizolt.base import ToolError
    from wizolt.session import Session, SessionSnapshotStore

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

    monkeypatch.setattr("wizolt.tools.delegate.os.unlink", fail_snapshot)

    with pytest.raises(ToolError, match="failed to delete its snapshot"):
        await _delegate_call(parent, _delegate_runner(parent), action="reset")
    assert parent.worker is worker
    assert os.path.isfile(snapshot)

async def test_delegate_reset_deletes_disk_only_worker_after_parent_resume(tmp_path):
    from wizolt.session import Session, SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    snapshot = SessionSnapshotStore.session_path(parent.config.data_dir, str(tmp_path), worker.uid)
    assert parent.worker is None and os.path.isfile(snapshot)

    result = await _delegate_call(parent, _delegate_runner(parent), action="reset")

    assert 'action="reset"' in result
    assert not os.path.exists(snapshot)

@pytest.mark.parametrize("max_steps", [0, -1, True, "3"])
def test_delegate_rejects_invalid_max_steps(tmp_path, max_steps):
    from wizolt.base import ToolError
    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    with pytest.raises(ToolError, match="integer >= 1"):
        DelegateTool(parent, [{"action": "send", "order": "work", "max_steps": max_steps}]).call()

async def test_delegate_merges_worker_diffs_into_parent(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [
                    call(
                        "Edit",
                        ["f.txt", "", [{"op": "create", "content": "x"}]],
                    )
                ],
                "editing",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    result = await _delegate_call(parent, runner, action="send", order="create f.txt")

    assert "f.txt" in result
    assert (tmp_path / "f.txt").read_text() == "x"
    assert any(diff.path == "f.txt" for diff in parent.turn_diffs)

async def test_delegate_interrupt_settles_and_merges_diffs(tmp_path, monkeypatch):
    """Cancelling the parent's Delegate call cancels the worker's turn by propagation.

    There is no second cancellation channel: the worker's turn is a child of the parent's, so the
    parent's cancellation reaches the worker's own model request. The worker still settles its
    history, and its diffs still merge -- otherwise the user never sees what it did."""
    import asyncio

    from wizolt.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    started = asyncio.Event()

    class SlowModel(FakeModelClient):
        def __init__(self):
            super().__init__([])
            self.requests = []

        async def request_async(self, messages, request_tools=None):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return (
                    {"role": "assistant", "content": "editing"},
                    [
                        call(
                            "Edit",
                            ["f.txt", "", [{"op": "create", "content": "x"}]],
                        )
                    ],
                    "editing",
                )
            started.set()
            await asyncio.sleep(30)
            raise AssertionError("the worker's second request must not complete in this test")

    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: SlowModel())
    runner = _delegate_runner(parent)
    tool = DelegateTool(parent, [{"action": "send", "order": "create f.txt"}])
    tool.runner = runner

    send = asyncio.ensure_future(tool.call_async())
    await asyncio.wait_for(started.wait(), 5)
    send.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send

    assert (tmp_path / "f.txt").read_text() == "x"
    assert any(diff.path == "f.txt" for diff in parent.turn_diffs)
    # Every tool call the model issued has a matched result in the settled turn.
    worker_messages = json.dumps(parent.worker.messages)
    assert "f.txt" in worker_messages
    assert '"role": "tool"' in worker_messages

async def test_delegate_failure_reports_envelope_and_settles_worker_history(tmp_path, monkeypatch):
    from wizolt.base import ToolError
    from wizolt.prompts import FAILED_TOOL_CALL_RESULT
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    # Step 1 edits a real file (its diff lands before the failure), step 2 dies inside tools.run,
    # step 3 answers the follow-up delegation.
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [call("Edit", ["f.txt", "", [{"op": "create", "content": "x"}]])],
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
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run_async
    batches = {"n": 0}

    async def run_then_fail(self, tool_calls, **kwargs):
        batches["n"] += 1
        if batches["n"] == 2:
            raise RuntimeError("provider timeout")
        return await real_run(self, tool_calls, **kwargs)

    monkeypatch.setattr(ToolRunner, "run_async", run_then_fail)

    with pytest.raises(ToolError) as excinfo:
        await _delegate_call(parent, runner, action="send", order="create f.txt then die")
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
    result = await _delegate_call(parent, runner, action="send", order="continue")
    assert 'rounds="2"' in result
    assert "done" in result

async def test_delegate_failure_after_a_call_ran_in_the_dying_batch(tmp_path, monkeypatch):
    from wizolt.base import ToolCall, ToolError
    from wizolt.prompts import FAILED_TOOL_CALL_RESULT
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    # Step 1 edits f.txt (answered), step 2 carries two calls and dies with the first one already
    # executed, step 3 answers the follow-up delegation.
    model = FakeModelClient(
        [
            (
                {"role": "assistant", "content": "editing"},
                [call("Edit", ["f.txt", "", [{"op": "create", "content": "x"}]])],
                "editing",
            ),
            (
                {"role": "assistant", "content": ""},
                [
                    ToolCall("edit-2", "Edit", ["g.txt", "", [{"op": "create", "content": "y"}]]),
                    call("Read", [{"path": "missing.txt", "ranges": [[0, 1]]}]),
                ],
                "",
            ),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run_async
    batches = {"n": 0}

    async def run_then_die(self, tool_calls, **kwargs):
        batches["n"] += 1
        if batches["n"] == 2:
            # The first call of the dying batch really runs against the worker's session (file on
            # disk, diff recorded); the second never starts. The batch then dies before any result
            # is committed to the turn.
            await real_run(self, tool_calls[:1], **kwargs)
            raise RuntimeError("provider timeout")
        return await real_run(self, tool_calls, **kwargs)

    monkeypatch.setattr(ToolRunner, "run_async", run_then_die)

    with pytest.raises(ToolError) as excinfo:
        await _delegate_call(parent, runner, action="send", order="create f.txt then die mid-batch")
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
    result = await _delegate_call(parent, runner, action="send", order="continue")
    assert 'rounds="2"' in result and "done" in result

async def test_delegate_status_reports_last_failure_until_a_success(tmp_path, monkeypatch):
    from wizolt.base import ToolError
    from wizolt.runner import ToolRunner

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
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)

    real_run = ToolRunner.run_async

    async def fail(self, tool_calls, **kwargs):
        raise RuntimeError("provider timeout")

    monkeypatch.setattr(ToolRunner, "run_async", fail)
    with pytest.raises(ToolError):
        await _delegate_call(parent, runner, action="send", order="first")

    status = await _delegate_call(parent, runner, action="status")
    assert 'last_error="provider timeout"' in status
    assert 'last_error_round="1"' in status
    assert 'rounds="1"' in status
    assert 'alive="true"' in status

    # A successful send clears the remembered failure.
    monkeypatch.setattr(ToolRunner, "run_async", real_run)
    await _delegate_call(parent, runner, action="send", order="second")
    assert parent.worker.state.last_error == "" and parent.worker.state.last_error_round == 0
    status = await _delegate_call(parent, runner, action="status")
    assert "last_error" not in status
    assert 'rounds="2"' in status

async def test_delegate_failure_bounds_and_sanitizes_the_error_text(tmp_path, monkeypatch):
    from wizolt.base import ToolError
    from wizolt.runner import ToolRunner

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
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    body = 'HTTP 400: {"error": "bad "quoted" thing"}\nsecond line\n' + "x" * 3000
    async def fail(self, tool_calls, **kwargs):
        raise RuntimeError(body)

    monkeypatch.setattr(ToolRunner, "run_async", fail)
    with pytest.raises(ToolError):
        await _delegate_call(parent, runner, action="send", order="first")

    # The status tag stays parseable: one line, no bare double quote inside the attribute value.
    status = await _delegate_call(parent, runner, action="status")
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

async def test_worker_cache_prefix_stable_across_delegations(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="order one")
    await _delegate_call(parent, runner, action="send", order="order two")
    assert model.requests[0][:3] == model.requests[1][:3]

async def test_delegate_settings_isolated_and_fresh(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    parent.settings.max_steps = 7
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "answer one"}, [], "answer one"),
            ({"role": "assistant", "content": "answer two"}, [], "answer two"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o", max_steps=3)
    assert parent.settings.max_steps == 7  # the parent's budget is untouched
    assert parent.worker.settings.max_steps == 3
    assert parent.worker.settings is not parent.settings

    parent.settings.yolo = True
    await _delegate_call(parent, runner, action="send", order="o")
    assert parent.worker.settings.yolo is True  # fresh copy sees the runtime change

async def test_worker_reset_appends_event_message(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.messages.append({"role": "user", "content": "worker request"})
    worker.save_snapshot()
    parent.worker = worker
    parent.save_snapshot()
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    await loop.command("/worker reset")

    assert parent.worker is None
    assert parent.messages[-1].get(SESSION_EVENT_KEY) == "worker_reset"
    assert parent.messages[-1].get("role") == "user"
    request = await agent.prepare_request_async([{"role": "user", "content": "continue"}])
    assert any(message.get("role") == "user" and "starts from scratch" in str(message.get("content")) for message in request.messages)

async def test_agent_lives_on_worker_and_is_rebuilt_with_it(tmp_path, monkeypatch):
    from wizolt.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "one"}, [], "one")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")

    first_worker = parent.worker
    first_agent = first_worker._agent
    assert first_agent is not None
    assert first_agent.session is first_worker

    # /resume re-enters the same parent: a fresh parent object, worker rebuilt from the snapshot.
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    assert fresh.worker is None
    runner = _delegate_runner(fresh)
    await _delegate_call(fresh, runner, action="send", order="o")

    second_worker = fresh.worker
    assert second_worker is not first_worker
    assert second_worker._agent is not first_agent  # the old Agent died with the old worker object
    assert second_worker._agent.session is second_worker  # and the new one is bound to the new object

async def test_snapshot_restored_worker_shares_parent_skills_and_mcp(tmp_path, monkeypatch):
    from wizolt.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "one"}, [], "one")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")
    parent.worker.messages.append({"role": "user", "content": "worker request"})
    parent.worker.save_snapshot()

    # Resume: the worker now comes back through SessionSnapshotStore.load, not the fresh-branch.
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    runner = _delegate_runner(fresh)
    await _delegate_call(fresh, runner, action="send", order="o")
    worker = fresh.worker
    assert worker.skills is fresh.skills
    assert worker.mcp is fresh.mcp

async def test_worker_and_parent_source_views_do_not_cross(tmp_path, monkeypatch):
    """A worker has its own Session, so it has its own view namespace. Both sides start at view.1
    for different files, and a view id a worker mentions in its prose means nothing to the parent:
    the parent's Edit resolves that id against its own registry or refuses it as missing."""
    from wizolt.tools import EditTool, ReadTool

    parent = _delegate_session(tmp_path)
    (tmp_path / "parent.txt").write_text("parent line\n", encoding="utf-8")
    (tmp_path / "worker.txt").write_text("worker line\n", encoding="utf-8")
    model = FakeModelClient([({"role": "assistant", "content": "used view.1"}, [], "used view.1")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    await _delegate_call(parent, runner, action="send", order="o")
    worker = parent.worker

    parent_key = parent.register_source_drafts(list(ReadTool(parent, [{"path": "parent.txt"}]).call().drafts))[0]
    worker_key = worker.register_source_drafts(list(ReadTool(worker, [{"path": "worker.txt"}]).call().drafts))[0]

    assert parent_key == worker_key == "view.1"
    assert parent.get_source_view("view.1").display_path == "parent.txt"
    assert worker.get_source_view("view.1").display_path == "worker.txt"

    # The parent's view.1 is its own file, so the worker's id cannot reach across to worker.txt.
    with pytest.raises(ToolError, match="source path mismatch"):
        EditTool(parent, ["worker.txt", worker_key, [{"op": "replace", "start": 1, "end": 1, "content": "x\n"}]]).call()
    assert (tmp_path / "worker.txt").read_text(encoding="utf-8") == "worker line\n"
