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
from minacode.prompts import SYSTEM_PROMPT, WORKER_PROMPT
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


# 10. two registration gates: Delegate appears only when [worker] provider was set at session
#     start AND runtime.worker is on. The provider half is frozen per session, so a runtime
#     /worker provider change never flips the tool block; settings.worker stays the live half.
#     Closing is not reset (the snapshot stays).
def test_delegate_registration_gates(tmp_path):
    from minacode.session import Session

    def names(s):
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    # Gate off at session start (no [worker] provider): runtime.worker alone cannot register it,
    # and setting the provider mid-session is frozen out.
    off = session(tmp_path)
    off.settings.worker = True
    assert "Delegate" not in names(off)
    off.config.worker_provider = "default"
    assert "Delegate" not in names(off)

    # Gate on at session start: both halves are required, and a runtime provider change never
    # flips the tool block.
    on = Session(cwd=str(tmp_path), config=off.config)
    assert "Delegate" not in names(on)  # runtime.worker off
    on.settings.worker = True
    assert "Delegate" in names(on)
    on.config.worker_provider = ""
    assert "Delegate" in names(on)  # the frozen half is unchanged by runtime changes
    on.settings.worker = False
    assert "Delegate" not in names(on)  # the live half still drops the schema
    on.settings.worker = True
    assert "Delegate" in names(on)


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
    from minacode.base import Config, ProviderConfig
    from minacode.render import StatusBar
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    parent.usage.last_prompt_tokens = 200
    parent.usage.last_prompt_budget = 400
    parent.usage.last_cached_prompt_tokens = 50
    bar = StatusBar(parent)
    texts = [text for text, _ in bar.entries(show_elapsed=False)]
    parent_lead = parent.config.active_provider + "/" + (parent.config.provider.model.rsplit("/", 1)[-1] or "(no model)")
    assert parent_lead in texts and "[worker]" not in texts
    assert "ctx 50% · cache 25%" in texts

    # A live but idle worker does not take over the bar: marker, provider/model, and usage all
    # apply only while a delegation is in flight (the engine clears _active_turn_messages in
    # finish_turn), so an idle worker leaves the parent's values exactly as before it existed.
    worker_config = Config()
    worker_config.providers["default"] = ProviderConfig(model="worker-model")
    worker = Session(cwd=str(tmp_path), config=worker_config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    worker.usage.last_cached_prompt_tokens = 25
    parent.worker = worker
    texts = [text for text, _ in bar.entries(show_elapsed=False)]
    assert "[worker]" not in texts
    assert parent_lead in texts and "default/worker-model" not in texts
    assert "ctx 50% · cache 25%" in texts  # the parent's usage stays the source while idle
    assert "ctx 50% · cache 50%" not in texts  # the worker's usage is not shown while idle

    # In flight: the bar leads with the marker and reads the worker's provider/model and usage.
    worker._active_turn_messages.append({"role": "user", "content": "order"})
    texts = [text for text, _ in bar.entries(show_elapsed=False)]
    assert texts[0] == "[worker]"
    assert "default/worker-model" in texts and parent_lead not in texts
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

    assert "[worker]" not in label()
    worker._active_turn_messages.append({"role": "user", "content": "order"})
    assert "[worker]" in label()


def test_worker_model_stream_is_wired_from_the_runner(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    calls = []
    runner.model_stream = lambda kind, text: calls.append((kind, text))
    _delegate_call(parent, runner, action="send", order="o")
    on_stream = parent.worker._agent.model.on_stream
    assert on_stream is not runner.model_stream  # wrapped: `output_done` must not promote
    assert callable(on_stream)
    on_stream("output", "x")
    on_stream("output_done", "t")
    assert calls == [("output", "x"), ("", "")]


def test_status_reports_worker_delegation_state(tmp_path):
    from minacode.engine import Agent
    from minacode.loop import CommandLoop
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    outputs: list = []
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=outputs.append)

    def status_text():
        outputs.clear()
        loop.command("/status")
        return "\n".join(str(text) for text in outputs)

    # No worker session: one `worker` row naming the configured [worker] provider. Everything is
    # one flat table — the session's own rows, the parent's, then the worker's under `worker*`.
    text = status_text()
    assert text.startswith("| field | value |")
    assert "###" not in text
    assert "[worker] provider" in text and "default" in text

    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    # A fresh worker has no requests yet: the worker context row says so instead of inventing
    # tokens, and the model row mirrors the parent's (provider/model, api, reasoning).
    text = status_text()
    assert "| worker | `default/" in text
    assert "| worker ctx | (no requests yet); `idle`, rounds `0` |" in text

    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    worker.usage.last_cached_prompt_tokens = 25
    worker.usage.prompt_tokens = 50
    worker.usage.cached_prompt_tokens = 25
    text = status_text()
    # Scope to the worker rows: the parent's own cache row also says "(no requests yet)".
    worker_section = text.split("| worker |", 1)[1]
    assert "~50 / 100" in worker_section and "(no requests yet)" not in worker_section
    assert "| worker cache | " in worker_section and "last `50.0%`; session `50.0%`" in worker_section

    worker._active_turn_messages.append({"role": "user", "content": "order"})
    text = status_text()
    assert "`delegating`, rounds `0`" in text


# /worker's own status branch returns readable text for the human (the model-facing envelope stays
# in DelegateTool): no-live-worker, one line per fact, and the usage/state-context-percent values
def test_worker_status_command_is_human_readable(tmp_path):
    from minacode.engine import Agent
    from minacode.loop import CommandLoop
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.worker_command("") == chr(10).join(["worker: no active session", "worker provider: default"])
    assert loop.worker_command("status") == loop.worker_command("")

    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker
    worker.config.provider.model = "worker-model-x"
    worker.config.provider.reasoning = "high"
    worker.state.round_count = 3
    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    text = loop.worker_command("status")
    assert "worker: default/worker-model-x" in text
    assert "worker reasoning: high" in text
    assert "worker state: idle" in text
    assert "worker rounds: 3" in text
    assert "worker context: 50%" in text
    assert "<Delegate" not in text

    worker._active_turn_messages.append({"role": "user", "content": "order"})
    assert "worker state: delegating" in loop.worker_command("status")

    # Without provider-reported usage the state estimate is the fallback, like the envelope.
    worker.usage.last_prompt_budget = 0
    worker.state.context_percent = 42
    assert "worker context: 42%" in loop.worker_command("status")


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


# 11b. a send opens with a visible start marker: one yellow [worker] line naming the worker's live
# provider/model and the one-line order summary, so the scrollback has a boundary before the
# finish block. This is the fallback when no worker_rule is wired; the wired path emits a
# full-width yellow rule label instead (see the test below).
def test_delegate_send_logs_a_worker_start_marker(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start marker, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    _delegate_call(parent, runner, action="send", order=order)

    blocks = [block for block in outputs if isinstance(block, LogBlock)]
    marker = next(block for block in blocks if any(item.role is LogRole.WORKER for item, _ in block.walk()))
    rendered = str(marker)
    assert "[worker]" in rendered
    assert "▶" in rendered
    assert "default/worker-model-x" in rendered
    assert ToolRunner.oneline(order, 200) in rendered


def test_delegate_send_worker_rule_start_label(tmp_path, monkeypatch):
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order)

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert ToolRunner.oneline(order, 60) in labels[0]
    assert not any("[worker]" in rendered for rendered in labels)  # the rule label replaces the [worker] ▶ line


# 11b2. the start divider uses the send's optional `title` when given: the human-readable label
# replaces the order-first-line summary on both the wired rule and the fallback [worker] ▶ line.
def test_delegate_send_worker_rule_start_label_with_title(tmp_path, monkeypatch):
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order, title="fix /status blank line")

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert "fix /status blank line" in labels[0]
    assert ToolRunner.oneline(order, 60) not in labels[0]


def test_delegate_send_worker_start_marker_with_title(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start marker, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    _delegate_call(parent, runner, action="send", order=order, title="fix /status blank line")

    blocks = [block for block in outputs if isinstance(block, LogBlock)]
    marker = next(block for block in blocks if any(item.role is LogRole.WORKER for item, _ in block.walk()))
    rendered = str(marker)
    assert "[worker]" in rendered and "▶" in rendered
    assert "default/worker-model-x" in rendered
    assert "fix /status blank line" in rendered
    assert ToolRunner.oneline(order, 200) not in rendered


def test_delegate_send_worker_rule_start_label_falls_back_to_order(tmp_path, monkeypatch):
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "worker-model-x"
    order = "Rewrite the worker handoff plan to cover the start rule, then check it. " * 8
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    _delegate_call(parent, runner, action="send", order=order)

    assert labels, "the worker_rule callback never fired"
    assert labels[0].startswith("worker start · default/worker-model-x · ")
    assert ToolRunner.oneline(order, 60) in labels[0]


def test_delegate_rejects_empty_title(tmp_path):
    from minacode.base import ToolError
    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    with pytest.raises(ToolError, match="non-empty string"):
        DelegateTool(parent, [{"action": "send", "order": "work", "title": "   "}]).call()


# 11c. language is a send parameter, not a setting: it lands in the order the worker receives as
#      an explicit language request covering the live stream and interim messages, not just the end.
def test_delegate_send_language_directive_is_injected_into_the_order(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="fix the parser", language="Chinese")

    worker_order = model.requests[0][-1]["content"]
    assert worker_order.startswith("fix the parser")
    assert "Reply language: Chinese" in worker_order and "live stream" in worker_order


def test_delegate_send_rejects_a_blank_language(tmp_path):
    from minacode.base import ToolError

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)
    with pytest.raises(ToolError, match="language"):
        _delegate_call(parent, runner, action="send", order="o", language="   ")


# 11d. a forced runtime language is inherited: the worker rebuilds its settings from the parent on
#      every send, so the parent's /language value lands in the worker's system prompt too, while
#      the per-send `language` parameter stays an order-text directive (see test above).
def test_worker_inherits_forced_reply_language_from_parent(tmp_path, monkeypatch):
    from minacode.context import ContextManager

    parent = _delegate_session(tmp_path)
    parent.settings.language = "Chinese"
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="fix the parser")

    worker = parent.worker
    assert worker.settings.language == "Chinese"
    system = model.requests[0][0]["content"]
    assert system.startswith(WORKER_PROMPT.strip())
    assert "LANGUAGE OVERRIDE:" in system and "Chinese" in system
    # the projection the worker uses matches the request the fake model received
    assert ContextManager(worker).model_messages(worker.system_prompt)[0]["content"] == system


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


# 14b. The send envelope also carries the worker's token spend for this delegation: the program
#      subtracts worker.usage before/after (the fake model updates nothing, so 0/0), and the finish
#      summary renders it.
def test_delegate_envelope_reports_token_spend_and_summary_renders(tmp_path, monkeypatch):
    from minacode.engine import Agent

    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)

    def run_quiet(self, order):
        self.stopped_at_max_steps = False
        return "done"

    monkeypatch.setattr(Agent, "run", run_quiet)
    result = _delegate_call(parent, runner, action="send", order="o")
    assert 'tokens="' in result
    summary = runner.delegate_result_summary(result)
    assert " in / " in summary and " out" in summary
    assert "0 in / 0 out" in summary


# 14c. delegate_result_summary formats the raw integer token counts like /status does, and keeps
#      parsing envelopes written before the tokens attribute existed.
def test_delegate_summary_formats_tokens_and_tolerates_old_envelopes(tmp_path):
    parent = _delegate_session(tmp_path)
    runner = _delegate_runner(parent)

    summary = runner.delegate_result_summary(
        '<Delegate action="send" steps="3" elapsed="2.5s" files="a.txt, b.txt" stopped_at_max_steps="false" tokens="8200/1300">'
    )
    assert "8.2K in / 1.3K out" in summary

    legacy = runner.delegate_result_summary(
        '<Delegate action="send" steps="3" elapsed="2.5s" files="a.txt, b.txt" stopped_at_max_steps="false">'
    )
    assert "steps 3" in legacy
    assert "2.5s" in legacy
    assert "a.txt, b.txt" in legacy
    assert " in / " not in legacy

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

    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == "3f3881d320ed14fc5d6d493e689f9312b3cf892aee72113d051f11d115c7d10f"

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


# 20b. Delegate send confirmation: Y/Enter approve, n refuses without a reason, any other input is
# a refusal reason passed back to the model. `a` is an ordinary reason now (the always key is
# retired), and only a whole-line "c"/"config" opens the worker configuration loop.
def test_delegate_send_confirmation_prompt_and_reasons(tmp_path, monkeypatch):
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner
    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    for answer, expected in [("y", (True, "")), ("", (True, "")), ("n", (False, "")), ("a", (False, "a"))]:
        runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda _prompt, a=answer: a, output_fn=lambda text: None)
        confirmed, reason = runner.confirm(
            ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
        )
        assert (confirmed, reason) == expected, answer

    # A c-prefixed sentence is an ordinary reason, never the config key (whole-line exact match only).
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "cost too high", output_fn=lambda text: None)
    confirmed, reason = runner.confirm(
        ToolCall("delegate-2", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}])
    )
    assert (confirmed, reason) == (False, "cost too high")


def test_delegate_approval_brief_lists_send_and_worker_details(tmp_path):
    from minacode.base import LogRole, ProviderConfig, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner
    from minacode.tools import EditTool
    from minacode.tools.delegate import DelegateTool

    parent = _delegate_session(tmp_path)
    parent.config.providers["fast"] = ProviderConfig(model="worker-model", reasoning="high", api="responses")
    parent.config.worker_provider = "fast"
    parent.config.worker_model = "override-model"
    parent.config.worker_api = ""
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "y", output_fn=lambda text: None)

    order_lines = [f"line {i}" for i in range(1, 16)]
    args = {"action": "send", "order": "\n".join(order_lines), "title": "fix things", "language": "Chinese", "max_steps": 7}
    tool = DelegateTool(parent, [args])
    block = runner.approval_display(ToolCall("delegate-1", "Delegate", [args]), tool, "confirm")
    rows = [(item.label, item.text) for item, _ in block.walk()]
    labels = [label for label, _ in rows]
    texts = [text for _, text in rows]
    assert "title" in labels and "fix things" in texts
    assert "order" in labels
    assert all(f"line {i}" in texts for i in range(1, 13))  # the first 12 order lines
    assert "line 13" not in texts and "line 14" not in texts
    assert any("3 more lines" in text for text in texts)  # 15 - 12 = 3 overflow
    assert "language" in labels and "Chinese" in texts
    assert "max_steps" in labels and "7" in texts
    # The worker config is four rows, one per knob, with inherited values marked explicitly.
    assert "provider" in labels and "model" in labels and "effort" in labels and "api" in labels
    assert "worker" not in labels  # no combined single-line row anymore
    assert next(text for label, text in rows if label == "provider") == "fast"  # explicit override
    assert next(text for label, text in rows if label == "model") == "override-model"
    assert next(text for label, text in rows if label == "effort") == "(inherit) high"
    assert next(text for label, text in rows if label == "api") == "(inherit) responses"  # worker_api empty
    # Every brief row is WORKER-role: yellow label, default-foreground value, never the gray META.
    assert all(item.role is LogRole.WORKER for item, _ in list(block.walk())[1:])

    # An explicit worker_api override wins over the entry's api.
    parent.config.worker_api = "chat"
    block = runner.approval_display(ToolCall("delegate-2", "Delegate", [args]), tool, "confirm")
    rows = [(item.label, item.text) for item, _ in block.walk()]
    api_row = next(text for label, text in rows if label == "api")
    assert api_row == "chat"
    assert "(inherit)" not in api_row

    # Non-send Delegate calls keep the plain display; Edit keeps its preview children.
    status_tool = DelegateTool(parent, [{"action": "status"}])
    block = runner.approval_display(ToolCall("delegate-3", "Delegate", [{"action": "status"}]), status_tool, "confirm")
    assert not block.has_children
    edit_tool = EditTool(parent, ["a.py", [{"op": "replace_all", "old": "x", "new": "y"}]])
    (tmp_path / "a.py").write_text("x\n")
    block = runner.approval_display(ToolCall("edit-1", "Edit", ["a.py", []]), edit_tool, "confirm")
    assert block.has_children


def test_delegate_config_cycle_changes_worker_knobs_and_refreshes_live_worker(tmp_path):
    from dataclasses import replace

    from minacode.base import LogBlock, ProviderConfig, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner
    from minacode.session import Session
    from minacode.tools.delegate import DelegateTool, refresh_worker_entry

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="alt-model", reasoning="low", api="anthropic")
    worker = Session(cwd=str(tmp_path), config=replace(parent.config), settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    # The injected picker loop (CommandLoop.run_worker_config in production) drives the changes and
    # writes the config / refreshes the live worker itself; the runner only triggers it on `c`.
    def picker():
        parent.config.worker_provider = "alt"
        parent.config.worker_model = "worker-m"
        parent.config.worker_reasoning = "off"
        parent.config.worker_api = "responses"
        refresh_worker_entry(parent.config, worker, "alt")

    calls = []
    answers = iter(["c", "y"])
    prompts = []
    outputs = []

    def input_fn(prompt):
        prompts.append(prompt)
        return next(answers)

    runner = ToolRunner(parent, ContextManager(parent), input_fn=input_fn, output_fn=outputs.append)
    runner.worker_config_picker = lambda: calls.append(1) or picker()
    confirmed, reason = runner.confirm(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}]))
    assert (confirmed, reason) == (True, "")
    assert calls == [1]  # the `c` key drove the picker loop exactly once
    assert parent.config.worker_provider == "alt"
    assert parent.config.worker_model == "worker-m"
    assert parent.config.worker_reasoning == "off"
    assert parent.config.worker_api == "responses"
    # The live worker's active entry carries the overrides (copy-on-write, never shared).
    assert worker.config.active_provider == "alt"
    entry = worker.config.providers["alt"]
    assert (entry.model, entry.reasoning, entry.api) == ("worker-m", "off", "responses")
    assert worker.config.providers is not parent.config.providers
    # The config block printed, and the approval brief was redrawn after the picker returned.
    assert any("worker config" in str(out) for out in outputs if isinstance(out, LogBlock))
    assert sum(1 for prompt in prompts if "[Y/n/c or reason] " in prompt) == 2
    assert len([out for out in outputs if isinstance(out, LogBlock) and any(item.label == "order" for item, _ in out.walk())]) == 2

    # Without an injected picker (headless / non-CommandLoop) the `c` key prints the config block
    # and re-asks without crashing.
    prompts = []
    outputs = []
    answers = iter(["c", "y"])
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: prompts.append(prompt) or next(answers), output_fn=outputs.append)
    confirmed, reason = runner.confirm(ToolCall("delegate-2", "Delegate", [{"action": "send", "order": "o"}]), DelegateTool(parent, [{"action": "send", "order": "o"}]))
    assert (confirmed, reason) == (True, "")
    assert any("worker config" in str(out) for out in outputs if isinstance(out, LogBlock))
    assert parent.config.worker_provider == "alt"  # untouched without a picker
    assert parent.config.worker_api == "responses"  # untouched without a picker


def test_delegate_yolo_without_authorization_still_confirms(tmp_path, monkeypatch):
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    parent.settings.yolo = True
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    prompts = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: prompts.append(prompt) or "y", output_fn=lambda text: None)

    status, _message, _observation = runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"
    assert len(prompts) == 1  # yolo alone does not skip a Delegate send


def test_delegate_send_refused_does_not_run(tmp_path, monkeypatch):
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda prompt: "n", output_fn=lambda text: None)

    status, message, _observation = runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "refused"
    assert "refused" in message
    assert not model.requests  # the worker never ran


# 21. [worker] model/reasoning/api parse like [worker] provider; reasoning and api validate their choices.
def test_worker_config_parses_model_and_reasoning(tmp_path):
    from minacode.base import Config

    config = Config.from_dict(
        {
            "worker": {"provider": "fast", "model": "m-x", "reasoning": "high", "api": "responses"},
            "provider": {"active": "default", "default": {"model": "d"}, "fast": {"model": "m"}},
        }
    )
    assert config.worker_provider == "fast"
    assert config.worker_model == "m-x"
    assert config.worker_reasoning == "high"
    assert config.worker_api == "responses"

    # Defaults: no [worker] model/reasoning/api means "inherit the entry's value" at spawn time.
    plain = Config.from_dict({"provider": {"default": {"model": "d"}}})
    assert plain.worker_model == "" and plain.worker_reasoning == "" and plain.worker_api == ""


def test_worker_config_rejects_invalid_worker_reasoning(tmp_path):
    from minacode.base import Config, ConfigError

    with pytest.raises(ConfigError, match="worker.reasoning"):
        Config.from_dict({"worker": {"reasoning": "turbo"}, "provider": {"default": {}}})


def test_worker_config_rejects_invalid_worker_api(tmp_path):
    from minacode.base import Config, ConfigError

    with pytest.raises(ConfigError, match="worker.api"):
        Config.from_dict({"worker": {"api": "oai"}, "provider": {"default": {}}})


def test_worker_provider_config_applies_api_override(tmp_path):
    """worker_provider_config folds an explicit worker.api into the detached entry; an empty
    worker_api inherits the entry's own protocol (the worker never shares the parent's object)."""
    from minacode.base import Config
    from minacode.tools.delegate import worker_provider_config

    config = Config.from_dict(
        {
            "worker": {"provider": "fast", "api": "chat"},
            "provider": {"active": "default", "default": {"model": "d", "api": "auto"}, "fast": {"model": "m", "api": "anthropic"}},
        }
    )
    entry = worker_provider_config(config, "fast")
    assert entry.api == "chat"  # the [worker] api override wins
    assert entry is not config.providers["fast"]
    assert config.providers["fast"].api == "anthropic"  # the parent's entry is untouched

    config.worker_api = ""
    entry = worker_provider_config(config, "fast")
    assert entry.api == "anthropic"  # empty override inherits the entry's api

    entry = worker_provider_config(config, "default")
    assert entry.api == "auto"


# 22. The Delegate registration gate is frozen per session: /worker provider stores the config for
#     the next spawn (and live-applies to a live worker) but never flips the tool block mid-
#     session, whether delegation was on or off at session start. A freshly constructed session
#     over the same config re-evaluates the gate (simulating a restart), and an unknown name is
#     rejected without touching the config.
def test_worker_provider_command_does_not_flip_registration_gate(tmp_path):
    from minacode.base import ProviderConfig
    from minacode.engine import Agent
    from minacode.loop import CommandLoop
    from minacode.session import Session
    from minacode.tools import Tool

    parent = session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    def names(s):
        return {schema["function"]["name"] for schema in Tool.resolved_schemas(s)}

    parent.settings.worker = True
    assert parent.worker_tool_enabled is False
    assert "Delegate" not in names(parent)
    # Frozen off: the command stores the value for the next spawn and says a restart is needed;
    # the tool block is unchanged mid-session.
    assert loop.worker_command("provider alt") == "Set worker provider = alt (delegation is off this session; takes effect after a restart)"
    assert parent.config.worker_provider == "alt"
    assert "Delegate" not in names(parent)
    # "off" clears quietly when the gate is frozen off.
    assert loop.worker_command("provider off") == "worker provider: off"
    assert parent.config.worker_provider == ""

    before = parent.config.worker_provider
    assert loop.worker_command("provider nope") == "Unknown provider: nope"
    assert parent.config.worker_provider == before

    # Simulating a restart: a freshly constructed session over the same config re-evaluates the
    # frozen gate, so the stored value registers Delegate...
    parent.config.worker_provider = "alt"
    fresh = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings)
    fresh.settings.worker = True
    assert fresh.worker_tool_enabled is True
    assert "Delegate" in names(fresh)
    # ...and the frozen-on gate stays registered across runtime changes, including clearing the
    # provider; only the next session re-evaluates it.
    fresh_agent = Agent(fresh, output_fn=lambda text: None)
    fresh_loop = CommandLoop(fresh_agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    assert fresh_loop.worker_command("provider off") == "worker provider: off"
    assert fresh.config.worker_provider == ""
    assert "Delegate" in names(fresh)


# 23. "off" is the clearing word unless a provider entry is literally named "off": existence in
#     config.providers wins, so /worker provider off selects that entry.
def test_worker_provider_off_selects_literal_off_entry(tmp_path):
    from minacode.base import ProviderConfig
    from minacode.engine import Agent
    from minacode.loop import CommandLoop

    parent = session(tmp_path)
    parent.config.providers["off"] = ProviderConfig(model="m")
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.worker_command("provider off") == "Set worker provider = off (delegation is off this session; takes effect after a restart)"
    assert parent.config.worker_provider == "off"


# 24. /worker model and /worker reason store overrides, reject an invalid effort, and "default"
#     clears; "off" is a valid reasoning effort, never the clearing word.
def test_worker_model_and_reason_overrides(tmp_path):
    from minacode.base import REASONING_CHOICES
    from minacode.engine import Agent
    from minacode.loop import CommandLoop

    parent = session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.worker_command("model") == "worker model: (inherit)"
    assert loop.worker_command("model gpt-5.2") == "Set worker.model = gpt-5.2"
    assert parent.config.worker_model == "gpt-5.2"
    assert loop.worker_command("model") == "worker model: gpt-5.2"
    assert loop.worker_command("model default") == "worker model: (inherit)"
    assert parent.config.worker_model == ""

    assert loop.worker_command("reason high") == "Set worker.reasoning = high"
    assert parent.config.worker_reasoning == "high"
    assert loop.worker_command("reason off") == "Set worker.reasoning = off"  # a valid effort
    assert parent.config.worker_reasoning == "off"
    assert loop.worker_command("reason default") == "worker reasoning: (inherit)"
    assert parent.config.worker_reasoning == ""

    assert loop.worker_command("reason turbo") == "Usage: /worker reason " + "|".join(REASONING_CHOICES)
    assert loop.worker_command("provider a b") == "Usage: /worker provider [NAME]"
    assert loop.worker_command("model a b") == "Usage: /worker model [MODEL]"
    assert loop.worker_command("reason a b") == "Usage: /worker reason [EFFORT]"


# 25. spawn isolation: the worker's active ProviderConfig is a detached copy (never `is` the
#     parent's), [worker] model/reasoning overrides are applied to it, and mutating it does not
#     leak into the parent's providers entry. A snapshot-resumed worker picks up the overrides the
#     same way, because the load path receives the same freshly built config.
def test_delegate_spawn_isolates_provider_and_applies_overrides(tmp_path, monkeypatch):
    from minacode.base import ProviderConfig
    from minacode.session import SessionSnapshotStore

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    parent.config.worker_provider = "alt"
    parent.config.worker_model = "worker-model"
    parent.config.worker_reasoning = "high"
    parent.messages.append({"role": "user", "content": "parent request"})
    parent.save_snapshot()
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o")

    worker_provider = parent.worker.config.provider
    assert worker_provider is not parent.config.providers["alt"]
    assert parent.worker.config.providers is not parent.config.providers
    assert worker_provider.model == "worker-model"
    assert worker_provider.reasoning == "high"
    assert parent.config.providers["alt"].model == "m"

    # Mutating the worker's active entry never leaks into the parent's providers entry.
    worker_provider.model = "mutated"
    assert parent.config.providers["alt"].model == "m"

    # Resume: the worker comes back through SessionSnapshotStore.load with the same freshly built
    # config, so a current override applies to the restored worker too.
    parent.worker.messages.append({"role": "user", "content": "worker request"})
    parent.worker.save_snapshot()
    model.script.append(({"role": "assistant", "content": "two"}, [], "two"))
    parent.config.worker_model = "resumed-model"
    fresh = SessionSnapshotStore.load(parent.uid, config=parent.config, settings=parent.settings, cwd=str(tmp_path))
    runner = _delegate_runner(fresh)
    _delegate_call(fresh, runner, action="send", order="o")
    assert fresh.worker.config.provider.model == "resumed-model"


# 26. live switch: with a live worker, /worker model X replaces the worker's active entry
#     immediately while the parent's providers entry is untouched; "default" restores the
#     underlying entry's model on the live worker.
def test_worker_model_switch_applies_to_live_worker(tmp_path, monkeypatch):
    from minacode.engine import Agent
    from minacode.loop import CommandLoop

    parent = _delegate_session(tmp_path)
    parent.config.providers["default"].model = "parent-model"
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o")

    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = parent.worker
    assert worker.config.provider.model == "parent-model"

    loop.worker_command("model worker-model")
    assert worker.config.provider.model == "worker-model"
    assert parent.config.providers["default"].model == "parent-model"  # untouched

    loop.worker_command("model default")
    assert worker.config.provider.model == "parent-model"  # restores the entry's model
    assert parent.config.providers["default"].model == "parent-model"


# 27. a live worker also takes /worker provider NAME immediately: its active entry is replaced with
#     a detached copy and the parent's entry is untouched.
def test_worker_provider_switch_applies_to_live_worker(tmp_path, monkeypatch):
    from minacode.base import ProviderConfig
    from minacode.engine import Agent
    from minacode.loop import CommandLoop

    parent = _delegate_session(tmp_path)
    parent.config.providers["alt"] = ProviderConfig(model="m")
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    _delegate_call(parent, runner, action="send", order="o")

    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = parent.worker

    loop.worker_command("provider alt")
    assert worker.config.active_provider == "alt"
    assert worker.config.provider is not parent.config.providers["alt"]
    assert worker.config.provider.model == "m"
    assert parent.config.providers["alt"].model == "m"  # untouched


# 28. a finished Delegate send renders as a proper log block: the confirmation root line is just
#     `Delegate send` (no argument blob), the finish block carries a steps/elapsed/files summary
#     and the worker's answer as an OUTPUT preview, and the raw envelope tags never reach the log.
#     This is the fallback when no worker_rule is wired; the wired path replaces the summary child
#     line with a yellow rule label (see the test below).
def test_delegate_send_finish_display_summary_and_preview(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    status, _message, _observation = runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    # The confirmation line shows the short root (`Delegate send`, not the order blob); the finish
    # block is the one with OUTPUT children (the worker's answer preview).
    assert any(block.items and block.items[0].label == "Delegate" and block.items[0].text == "send" for block in blocks)
    finish = next(block for block in blocks if any(item.role is LogRole.OUTPUT for item, _ in block.walk()))
    # The finish block is the closing marker of the delegation bracket, so it carries the same
    # yellow [worker] identity as the start marker: a root line whose label is the bracket tag.
    assert finish.items[0].label == "[worker]" and finish.items[0].text == "◀"
    rendered = str(finish)
    assert "steps 1" in rendered and "(none)" in rendered
    assert "the worker answer" in rendered
    assert "<Delegate" not in rendered and "<worker>" not in rendered and "</worker>" not in rendered
    assert any(item.label == "stored" for item, _ in finish.walk())


def test_delegate_send_finish_worker_rule_label_and_preview(tmp_path, monkeypatch):
    from minacode.base import LogBlock, LogRole, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    outputs = []
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    runner.worker_rule = lambda label: labels.append(label)
    status, _message, _observation = runner.run_one(ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o"}]))
    assert status == "ok"

    done = [label for label in labels if label.startswith("worker done · ")]
    assert done, "the finish worker_rule callback never fired"
    assert "worker done · steps 1" in done[0] and " in / " in done[0]
    assert "(none)" not in done[0]  # no files touched: the files segment is omitted, not '(none)'

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    finish = next(block for block in blocks if any(item.role is LogRole.OUTPUT for item, _ in block.walk()))
    rendered = str(finish)
    assert "the worker answer" in rendered
    assert any(item.label == "stored" for item, _ in finish.walk())
    # The done summary lives in the rule label now, not as a child line of the finish block.
    assert not any(item.label == "done" and item.text.startswith("steps ") for item, _ in finish.walk())


# 28b. a send with `title` carries the same human-readable label onto the done divider: the title
# is the first part of the `worker done` rule label, ahead of steps/elapsed/tokens/files.
def test_delegate_send_finish_worker_rule_label_carries_title(tmp_path, monkeypatch):
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "the worker answer"}, [], "the worker answer")])
    monkeypatch.setattr("minacode.engine.ModelClient", lambda session: model)
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=lambda text: None)
    runner.worker_rule = lambda label: labels.append(label)
    status, _message, _observation = runner.run_one(
        ToolCall("delegate-1", "Delegate", [{"action": "send", "order": "o", "title": "fix /status blank line"}])
    )
    assert status == "ok"

    done = [label for label in labels if label.startswith("worker done · ")]
    assert done, "the finish worker_rule callback never fired"
    assert done[0].startswith("worker done · fix /status blank line · steps 1")


# 29. a Delegate reset is a one-shot tool call, not a bracket: it keeps its ordinary tool root
#     and adds a plain done child stating what was cleared and what survives. No worker_rule rule
#     and no [worker] ◀ root.
def test_delegate_reset_finish_display_worker_root_and_cleared_notice(tmp_path):
    from minacode.base import LogBlock, ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.save_snapshot()
    parent.worker = worker
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)

    status, _message, _observation = runner.run_one(ToolCall("delegate-r", "Delegate", [{"action": "reset"}]))
    assert status == "ok"

    blocks = [item for item in outputs if isinstance(item, LogBlock)]
    finish = next(block for block in blocks if any(item.label == "done" for item, _ in block.walk()))
    # Reset keeps its ordinary tool root (the short_call, not [worker] ◀) and a done child.
    assert finish.items[0].label != "[worker]"
    assert "worker context cleared" in str(finish)


def test_delegate_reset_finish_worker_rule_label(tmp_path):
    from minacode.base import ToolCall
    from minacode.context import ContextManager
    from minacode.runner import ToolRunner
    from minacode.session import Session

    parent = _delegate_session(tmp_path)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    worker.save_snapshot()
    parent.worker = worker
    outputs = []
    labels = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    runner.worker_rule = lambda label: labels.append(label)

    status, _message, _observation = runner.run_one(ToolCall("delegate-r", "Delegate", [{"action": "reset"}]))
    assert status == "ok"

    # Reset is a one-shot tool call, not a delegation bracket: no full-width worker_rule rule
    # fires; the reset shows as an ordinary tool root with a plain done child.
    assert labels == [], "reset must not emit a worker_rule divider"
    from minacode.base import LogBlock
    done = [item for block in outputs if isinstance(block, LogBlock) for item, _ in block.walk() if item.label == "done"]
    assert done, "reset should keep its ordinary tool root with a done child"
    assert "worker context cleared" in next(item.text for block in outputs if isinstance(block, LogBlock) for item, _ in block.walk() if item.label == "done")


# The worker's model stream forwards to the parent loop's live display, except
# `output_done`: the parent's promote would write the completed text a second
# time on top of what the worker's own output_fn already put in the scrollback,
# and the worker path never consumes the promoted-text marker. `output_done` is
# downgraded to a plain ("", "") preview clear; everything else forwards
# unchanged.
def test_worker_stream_forwards_output_and_suppresses_output_done_promote():
    from minacode.tools.delegate import _worker_stream

    calls: list[tuple[str, str]] = []

    class StubRunner:
        def __init__(self):
            self.model_stream = lambda kind, text: calls.append((kind, text))

    stream = _worker_stream(StubRunner())

    stream("output", "x")
    stream("output_done", "t")
    stream("", "")
    stream("tool", "Bash")

    assert calls == [("output", "x"), ("", ""), ("", ""), ("tool", "Bash")]
    assert all(kind != "output_done" for kind, _ in calls)
