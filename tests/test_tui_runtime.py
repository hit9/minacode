"""TuiRuntime behavior: command dispatch, the follow-up queue, streamed response promotion,
resume, and session housekeeping at startup."""

import os
import threading
import time
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tui_harness import ResizableOutput, loop, run_interactive_tui, session, wait_until

import minacode.render as render_module
import minacode.tui as tui_module
from minacode.base import (
    Config,
    MalformedToolCallError,
    MinacodeError,
    ToolCall,
)
from minacode.engine import Agent
from minacode.loop import CommandLoop, TuiRuntime
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TuiApp
from minacode.update import UpdateChecker


def history_file(path, entries, line="x" * 200):
    """Write a prompt_toolkit history file with `entries` numbered entries."""
    with open(path, "wb") as file:
        file.writelines(f"\n# 2026-01-01 00:00:{index:02d}\n+{index}-{line}\n".encode() for index in range(entries))
    return path


class TextRecordingOutput(ResizableOutput):
    def __init__(self, rows=24, columns=80):
        super().__init__(rows, columns)
        self.writes = []
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.writes.append(data)

    def write_raw(self, data):
        with self.lock:
            self.writes.append(data)

    def text(self):
        with self.lock:
            return "".join(self.writes)


def test_tui_emits_resumed_history_after_primary_screen_starts(tmp_path, monkeypatch):
    scenario_session = session(tmp_path)
    scenario_session.resumed = True
    scenario_session.messages.extend(
        [
            {"role": "user", "content": "restored question"},
            {"role": "assistant", "content": "restored answer"},
        ]
    )
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    command_loop.ui.color = True
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    real_application = Application
    emitted_while_running = []
    history_emitted = threading.Event()

    def print_formatted(value, *args, **kwargs):
        text = fragment_list_to_text(to_formatted_text(value))
        if "restored answer" in text:
            emitted_while_running.append(command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            history_emitted.set()

    monkeypatch.setattr(render_module, "print_formatted_text", print_formatted)

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(tui_module, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            assert history_emitted.wait(timeout=1)
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert not driver.is_alive()
    assert emitted_while_running == [True]


@pytest.mark.parametrize("entered", [" /help", "exit "])
def test_tui_runtime_strips_input_before_command_dispatch(tmp_path, entered):
    command_loop = loop(tmp_path)
    dispatched = []
    command_loop.command = lambda text: dispatched.append(text) or (True, False)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)

    assert runtime.dispatch(entered)
    assert dispatched == [entered.strip()]


def test_tui_dispatch_compact_flushes_queued_followups(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")
    command_loop.session.enqueue_user_input("followup B")

    # Empty history makes /compact return early (no model) yet still exercise the command path.
    assert runtime.dispatch("/compact")

    # The queued follow-ups flush exactly as they do after a model turn: the first is ready to
    # run and the rest stay queued, instead of being stranded behind the command.
    assert runtime.pending.qsize() == 1
    assert runtime.pending.get_nowait() == "followup A"
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup B"]


def test_tui_dispatch_command_flushes_single_followup_completely(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = lambda text: (True, False)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("only followup")

    assert runtime.dispatch("/compact")

    assert runtime.pending.qsize() == 1
    assert runtime.pending.get_nowait() == "only followup"
    assert command_loop.session.pending_user_inputs == []


def test_tui_dispatch_failed_command_still_flushes_followup(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup after error")
    command_loop.command = lambda _text: (_ for _ in ()).throw(MinacodeError("command failed"))

    assert runtime.dispatch("/broken")

    assert runtime.pending.get_nowait() == "followup after error"
    assert command_loop.session.pending_user_inputs == []


def test_tui_dispatch_queues_older_followup_before_restoring_idle(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = lambda _text: (True, False)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("older followup")
    events = []
    real_submit_next = runtime.submit_next
    real_reset_turn = runtime.reset_turn

    def submit_next(entered):
        events.append("submit")
        real_submit_next(entered)

    def reset_turn():
        events.append("idle")
        real_reset_turn()

    runtime.submit_next = submit_next
    runtime.reset_turn = reset_turn

    assert runtime.dispatch("/slow-command")

    assert events == ["submit", "idle"]
    assert runtime.pending.get_nowait() == "older followup"


def test_tui_dispatch_command_with_empty_queue_stays_idle(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = lambda text: (True, False)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)

    assert runtime.dispatch("/help")

    assert runtime.pending.qsize() == 0
    assert command_loop.session.pending_user_inputs == []


def test_tui_dispatch_non_command_leaves_followups_for_agent_turn(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")

    # A plain message is not a command: dispatch returns False and must not flush the queue,
    # because run_agent_turn owns follow-up dispatch for model turns (no double dispatch).
    assert not runtime.dispatch("answer me")

    assert runtime.pending.qsize() == 0
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup A"]


def test_tui_dispatch_exit_does_not_flush_queued_followups(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.command = lambda text: (True, True)
    command_loop.tui = TuiApp()
    command_loop.tui.exit = lambda: None
    runtime = TuiRuntime(command_loop)
    command_loop.session.enqueue_user_input("followup A")

    assert runtime.dispatch("/exit")

    assert runtime.pending.qsize() == 0
    assert [item.text for item in command_loop.session.pending_user_inputs] == ["followup A"]


def test_tui_runtime_keeps_space_around_user_input_before_working(tmp_path, monkeypatch):
    output = []
    scenario_session = session(tmp_path)
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=output.append),
        input_fn=lambda prompt="": "",
        output_fn=output.append,
    )
    runtime = TuiRuntime(command_loop)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running = lambda label: output.append("set_running:" + label)
    command_loop.command = lambda _text: (False, False)
    command_loop.agent.run = lambda _text: "done"
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    assert not runtime.dispatch("answer me")
    runtime.run_agent_turn("answer me")

    assert output[:3] == ["\n• answer me", "", "set_running:working"]


def test_tui_runtime_does_not_reemit_a_stream_promoted_answer(tmp_path, monkeypatch):
    # A terminal NextHints batch promotes its answer into scrollback the way any tool batch does,
    # but unlike an ordinary batch nothing re-publishes it through agent_output. The post-turn emit
    # must therefore skip an answer that was already promoted, or it shows up twice.
    scenario_session = session(tmp_path)
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    runtime = TuiRuntime(command_loop)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running = lambda label: None
    command_loop.agent.run = lambda _text: "the final answer"
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)
    emitted: list[tuple] = []
    command_loop.ui.emit_answer = lambda *args, **kwargs: emitted.append(args)

    command_loop.model_stream_promoted_text = "the final answer"  # already permanent scrollback
    runtime.run_agent_turn("do it")

    assert emitted == []


def test_tui_runtime_emits_answer_when_not_stream_promoted(tmp_path, monkeypatch):
    # A plain final answer is never promoted, so the post-turn emit is its only path to scrollback.
    scenario_session = session(tmp_path)
    command_loop = CommandLoop(
        Agent(scenario_session, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    runtime = TuiRuntime(command_loop)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running = lambda label: None
    command_loop.agent.run = lambda _text: "the final answer"
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)
    emitted: list[tuple] = []
    command_loop.ui.emit_answer = lambda *args, **kwargs: emitted.append(args)

    runtime.run_agent_turn("do it")

    assert emitted == [("the final answer",)]


def test_automatic_compaction_replaces_working_divider_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.settings.max_context_tokens = 1
    command_loop.session.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest request"},
    ]
    command_loop.tui = TuiApp()
    divider_during_compaction = []

    def compact(_context):
        divider_during_compaction.append(fragment_list_to_text(command_loop.queue_divider_fragments()))
        return {"summary": "compact summary"}

    command_loop.agent.model.compact = compact

    command_loop.agent.context.prepare_messages(command_loop.agent.model, "system")

    assert "compacting context (" in divider_during_compaction[0]
    assert command_loop.tui.status_label == "working"


def test_compaction_retry_returns_to_compacting_and_reports_fallback(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    output = []
    command_loop.tool_output = output.append

    command_loop.automatic_compaction_status(True)
    command_loop.model_retry_wait_status(True)
    assert command_loop.tui.status_label == "retrying"

    command_loop.model_retry_wait_status(False)
    assert command_loop.tui.status_label == "compacting context"

    command_loop.automatic_compaction_status(False, "provider timed out")
    assert command_loop.tui.status_label == "working"
    assert output[0].items[0].label == "compaction fallback"
    assert output[0].items[0].text == "provider timed out"


def test_tui_runtime_clears_thinking_before_cancelled_output(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    emitted = []

    def interrupt(_user_input):
        command_loop.model_stream_output("reasoning", "private reasoning")
        raise KeyboardInterrupt

    command_loop.agent.run = interrupt
    command_loop.emit = lambda text: emitted.append((text, command_loop.model_stream_fragments()))
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    runtime.run_agent_turn("question")

    assert emitted[-1] == ("Cancelled", [])


def test_responses_stream_promotes_text_before_blocked_tool_arguments(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.config.provider.api = "responses"
    command_loop.session.config.provider.model = "gpt-5"
    command_loop.session.config.provider.url = "http://test"
    command_loop.session.config.provider.key = "sk-test"
    command_loop.ui.color = True
    app = TuiApp(activity_fragments_fn=command_loop.tui_activity_fragments)
    command_loop.tui = app
    output = TextRecordingOutput()
    arguments_blocked = threading.Event()
    release_arguments = threading.Event()
    request_finished = threading.Event()
    worker_errors = []
    timeline = []
    response = "I am editing the files."
    terminal = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "Bash",
                "arguments": '{"command":"echo hi"}',
            },
        ],
    }

    def events():
        yield {"type": "response.output_text.delta", "delta": response}
        yield {"type": "response.output_text.done"}
        yield {"type": "response.output_item.added", "item": {"type": "function_call"}}
        timeline.append("tool arguments")
        arguments_blocked.set()
        assert release_arguments.wait(timeout=2)
        yield {"type": "response.function_call_arguments.delta", "delta": '{"args"'}
        yield {"type": "response.completed", "response": terminal}

    responses = SimpleNamespace(create=lambda **_params: events())
    monkeypatch.setattr(command_loop.agent.model, "client", lambda: SimpleNamespace(responses=responses))
    real_emit = command_loop.emit_agent_output

    def emit_promoted(text):
        real_emit(text)
        timeline.append("white response")

    monkeypatch.setattr(command_loop, "emit_agent_output", emit_promoted)

    def request():
        try:
            _assistant, _calls, content = command_loop.agent.model.request([{"role": "user", "content": "make the change"}], [])
            command_loop.agent_output(content)
        except Exception as error:  # noqa: BLE001 - harness collects every worker-thread failure
            worker_errors.append(error)
        finally:
            request_finished.set()

    def drive(_pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            wait_until(lambda: arguments_blocked.is_set() or request_finished.is_set(), timeout=2)
            assert arguments_blocked.is_set(), worker_errors
            assert timeline[:2] == ["white response", "tool arguments"]
            assert command_loop.model_stream_fragments() == []
            assert response in output.text()
            assert not request_finished.is_set()
        finally:
            release_arguments.set()
        assert request_finished.wait(timeout=2)
        worker.join(timeout=1)
        assert not worker.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output)

    assert worker_errors == []
    assert timeline.count("white response") == 1


def test_provider_tool_stream_promotes_answer_once_into_tui_scrollback(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.api = "responses"
    provider.model = "gpt-5"
    provider.url = "http://test"
    provider.key = "sk-test"
    command_loop.tui = TuiApp()  # no running application: scrollback writes run inline
    answer = "The searched answer."
    terminal = {
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": answer}]},
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "minacode"},
            },
        ],
    }
    events = [
        {"type": "response.output_text.delta", "delta": answer},
        {"type": "response.output_text.done"},
        {"type": "response.output_item.added", "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "minacode"},
            },
        },
        {"type": "response.completed", "response": terminal},
    ]
    responses = SimpleNamespace(create=lambda **_params: iter(events))
    monkeypatch.setattr(command_loop.agent.model, "client", lambda: SimpleNamespace(responses=responses))
    emitted = []
    monkeypatch.setattr(command_loop, "emit_agent_output", emitted.append)

    _assistant, _calls, content = command_loop.agent.model.request([{"role": "user", "content": "search"}], None)
    command_loop.agent_output(content)

    assert emitted == [answer]
    assert command_loop.model_stream_promoted_text == ""
    assert command_loop.model_stream_fragments() == []


def test_provider_tool_stream_publishes_only_the_text_written_after_the_search(tmp_path, monkeypatch):
    """A provider-side tool sits inside one response, so the promotion is a prefix of the answer."""
    command_loop = loop(tmp_path)
    provider = command_loop.session.config.provider
    provider.api = "responses"
    provider.model = "gpt-5"
    provider.url = "http://test"
    provider.key = "sk-test"
    command_loop.tui = TuiApp()  # no running application: scrollback writes run inline
    lead, rest = "Let me look that up.", "The searched answer."
    call = {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "minacode"}}
    terminal = {
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": lead}]},
            call,
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": rest}]},
        ],
    }
    events = [
        {"type": "response.output_text.delta", "delta": lead},
        {"type": "response.output_text.done"},
        {"type": "response.output_item.added", "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}},
        {"type": "response.output_item.done", "item": call},
        {"type": "response.output_text.delta", "delta": rest},
        {"type": "response.output_text.done"},
        {"type": "response.completed", "response": terminal},
    ]
    responses = SimpleNamespace(create=lambda **_params: iter(events))
    monkeypatch.setattr(command_loop.agent.model, "client", lambda: SimpleNamespace(responses=responses))
    emitted = []
    monkeypatch.setattr(command_loop, "emit_agent_output", emitted.append)

    _assistant, _calls, content = command_loop.agent.model.request([{"role": "user", "content": "search"}], None)
    command_loop.agent_output(content)

    assert emitted == [lead, rest]
    assert command_loop.model_stream_promoted_text == ""


def test_turn_end_answer_drops_the_prefix_already_promoted_into_scrollback(tmp_path, monkeypatch):
    """The final answer is published once even when a mid-response promotion wrote its opening."""
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    emitted = []
    command_loop.ui.emit_answer = lambda text, **_kwargs: emitted.append(text)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    def answer(_user_input):
        with command_loop.model_stream_lock:
            command_loop.model_stream_promoted_text = "Let me look that up."
        return "Let me look that up.\n\nThe searched answer."

    command_loop.agent.run = answer

    runtime.run_agent_turn("question")

    assert emitted == ["The searched answer."]


def test_non_tui_stream_completion_keeps_normal_agent_output(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    emitted = []
    monkeypatch.setattr(command_loop, "emit_agent_output", emitted.append)

    command_loop.model_stream_output("output_done", "completed response")
    command_loop.agent_output("completed response")

    assert emitted == ["completed response"]


def test_stream_promotion_waits_for_the_follow_up_it_answers(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()  # no running application: scrollback writes run inline
    timeline = []
    monkeypatch.setattr(command_loop, "emit_agent_output", lambda text: timeline.append(("assistant", text)))
    monkeypatch.setattr(command_loop, "flush_queued_to_log", lambda texts: timeline.append(("user", list(texts))))
    command_loop.agent.on_queue_flush = command_loop.flush_queued_to_log
    command_loop.session.enqueue_user_input("also update the README")

    class FakeModel:
        on_stream = None

        def __init__(self):
            self.calls = 0

        def request(self, messages, tools=None):
            self.calls += 1
            if self.calls > 1:
                return {"role": "assistant", "content": "done"}, [], "done"
            # The follow-up rides along with this request, so its answer must not reach scrollback
            # before the request returns and logs the message that prompted it.
            command_loop.model_stream_output("output_done", "Sure, editing both files.")
            return {}, [ToolCall("call_1", "Bash", ["ls"])], "Sure, editing both files."

        def estimated_request_tokens(self, messages, tools=None):
            return 10

    command_loop.agent.model = FakeModel()
    command_loop.agent.context.model = None

    assert command_loop.agent.run("update the code") == "done"

    assert timeline == [("user", ["also update the README"]), ("assistant", "Sure, editing both files.")]


def test_tui_turn_reset_clears_unconsumed_stream_promotion(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.model_stream_promoted_text = "stale response"
    runtime = TuiRuntime(command_loop)

    runtime.reset_turn()

    assert command_loop.model_stream_promoted_text == ""


def test_tui_runtime_reports_repeated_textual_tool_call_without_done_marker(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    answers = []
    turns_ended = []

    def fail(_user_input):
        raise MalformedToolCallError("Model emitted Bash as text 6 times; none of the textual calls were executed.")

    command_loop.agent.run = fail
    command_loop.ui.emit_answer = lambda text, **_kwargs: answers.append(text)
    command_loop.ui.emit_turn_end = turns_ended.append
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    runtime.run_agent_turn("continue")

    assert answers == ["Model emitted Bash as text 6 times; none of the textual calls were executed."]
    assert turns_ended == []


def test_resumed_tui_auto_dispatches_persisted_queue_as_one_request(tmp_path, monkeypatch):
    saved = session(tmp_path)
    saved.enqueue_user_input("queued one")
    saved.enqueue_user_input("queued two")
    saved.save_snapshot()
    restored = Session.load_snapshot(saved.uid, config=saved.config)
    command_loop = CommandLoop(
        Agent(restored, output_fn=lambda _text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda _text: None,
    )
    requests = []

    class RecordingModel:
        def request(self, messages, tools=None):
            requests.append([message.get("content") for message in messages if message.get("role") == "user"])
            return {"role": "assistant", "content": "done"}, [], "done"

        def cancel(self):
            pass

    command_loop.agent.model = RecordingModel()
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    real_application = Application

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(tui_module, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            wait_until(lambda: len(requests) == 1)
            wait_until(lambda: command_loop.tui.input_mode == "chat")
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert len(requests) == 1
    assert "queued one" in requests[0]
    marked_followup = LIVE_FOLLOWUP_PREFIX + "queued two"
    assert marked_followup in requests[0]
    assert requests[0].index("queued one") < requests[0].index(marked_followup)
    assert restored.pending_user_inputs == []


def test_processed_queued_message_does_not_return_to_input(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    first_request = threading.Event()
    release_first = threading.Event()
    requests = []

    class RecordingModel:
        def request(self, messages, tools=None):
            requests.append([message.get("content") for message in messages if message.get("role") == "user"])
            if len(requests) == 1:
                first_request.set()
                assert release_first.wait(timeout=1)
            return {"role": "assistant", "content": "done"}, [], "done"

        def cancel(self):
            pass

    command_loop.agent.model = RecordingModel()
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    real_application = Application

    with create_pipe_input() as pipe_input:
        monkeypatch.setattr(tui_module, "Application", lambda **kwargs: real_application(input=pipe_input, **(kwargs | {"output": DummyOutput()})))

        def drive():
            wait_until(lambda: command_loop.tui is not None and command_loop.tui.app is not None and command_loop.tui.app.is_running)
            pipe_input.send_text("first task\r")
            assert first_request.wait(timeout=1)
            pipe_input.send_text("queued task\r")
            wait_until(lambda: [item.text for item in command_loop.session.pending_user_inputs] == ["queued task"])
            release_first.set()
            wait_until(lambda: len(requests) == 2)
            wait_until(lambda: command_loop.tui.input_mode == "chat")
            assert command_loop.tui.input_buffer.text == ""
            pipe_input.send_text("\x04")

        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        assert command_loop.run_tui() == 0
        driver.join(timeout=1)

    assert not driver.is_alive()
    assert "queued task" in requests[1]


def test_resend_command_only_resends_while_running(tmp_path):
    command_loop = loop(tmp_path)
    retried = []
    command_loop.tui = TuiApp(on_retry=lambda: retried.append(True))

    # Reachable from the running follow-up input (queue region), not just the idle prompt.
    assert "/resend" in CommandLoop.QUEUE_RUN_COMMANDS

    # Idle chat: no-op with guidance.
    command_loop.tui.set_idle()
    command_loop.command("/resend")
    assert retried == []

    # Running but no model call in flight: still a no-op.
    command_loop.tui.set_running("working")
    command_loop.session.state.current_model_call_started_at = 0.0
    command_loop.command("/resend")
    assert retried == []

    # Backoff countdown: there is no request in flight to resend.
    command_loop.session.state.current_model_call_started_at = 1.0
    command_loop.session.state.model_retry_until = 2.0
    command_loop.command("/resend")
    assert retried == []

    # Running with a model call in flight: resends via on_retry.
    command_loop.session.state.model_retry_until = 0.0
    command_loop.command("/resend")
    assert retried == [True]


def test_manual_resend_preserves_stream_driven_status(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("working")
    command_loop.session.state.current_model_call_started_at = 1.0
    runtime = TuiRuntime(command_loop)
    monkeypatch.setattr(runtime, "_interrupt_active", lambda _cancel: None)

    command_loop.session.state.model_retry_until = 2.0
    runtime._request_model_retry()
    assert command_loop.session.state.manual_model_retry_requested is False
    assert command_loop.session.state.model_retry_count == 0

    command_loop.session.state.model_retry_until = 0.0
    runtime._request_model_retry()

    assert command_loop.tui.status_label == "working"
    assert command_loop.session.state.manual_model_retry_requested is True
    assert command_loop.session.state.model_retry_count == 1


def test_recalling_sent_input_does_not_leave_revising_status(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("working")
    command_loop.session.enqueue_user_input("revise me")
    command_loop.session.claim_user_inputs()
    command_loop.session.state.current_model_call_started_at = 1.0
    runtime = TuiRuntime(command_loop)
    monkeypatch.setattr(runtime, "_interrupt_active", lambda _cancel: None)

    assert runtime.recall() == "revise me"
    command_loop.model_stream_output("output", "updated response")

    retrying = "".join(text for _style, text in command_loop.queue_divider_fragments())
    assert "retrying" in retrying
    assert "revising" not in retrying

    command_loop.status_bar.retry_notice_until = 0
    responding = "".join(text for _style, text in command_loop.queue_divider_fragments())
    assert "responding" in responding
    assert "revising" not in responding
    assert command_loop.session.state.manual_model_retry_requested is True


def test_retry_divider_keeps_pulse_and_elapsed_then_returns_to_working(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("working")
    command_loop.status_bar.started_at = 90.0
    command_loop.session.state.current_model_call_started_at = 99.0
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    command_loop.session.state.current_model_attempt = 2
    command_loop.session.state.model_retry_reason = "timeout"
    command_loop.session.state.model_retry_count += 1
    retrying = command_loop.queue_divider_fragments()
    retrying_text = "".join(text for _style, text in retrying)
    assert "retrying 2/6 · timeout (10s)" in retrying_text
    assert any(text == "● " for _style, text in retrying)
    assert ("retrying 2/6 · timeout", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    now[0] = 102.1
    working = command_loop.queue_divider_fragments()
    working_text = "".join(text for _style, text in working)
    assert "working · attempt 2/6 (12s)" in working_text
    assert "retrying" not in working_text
    assert any(text == "● " for _style, text in working)
    assert ("attempt 2/6", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    command_loop.session.state.current_model_call_started_at = 0.0
    assert all(text != "● " for _style, text in command_loop.queue_divider_fragments())


def test_retry_divider_shows_full_retry_text_while_waiting(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("retrying")
    command_loop.status_bar.started_at = 90.0
    command_loop.session.state.current_model_call_started_at = 99.0
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    command_loop.session.state.current_model_attempt = 3
    command_loop.session.state.model_retry_reason = "server error"
    command_loop.session.state.model_retry_count += 1
    command_loop.session.state.model_retry_until = now[0] + 20.0
    # Sync the notice tracker with the fresh retry count the way the render thread would, then
    # let the two-second notice expire while the wait itself is still in progress.
    command_loop.status_bar.retry_notice_active()
    command_loop.status_bar.retry_notice_until = 0

    waiting = command_loop.queue_divider_fragments()
    waiting_text = "".join(text for _style, text in waiting)
    assert "retrying 3/6 · server error · 20s" in waiting_text

    # Core fix: an in-flight wait keeps the full text even after the two-second notice window
    # expired, because a long backoff wait can outlast that window entirely.
    still_waiting = command_loop.queue_divider_fragments()
    still_text = "".join(text for _style, text in still_waiting)
    assert "retrying 3/6 · server error · 20s" in still_text

    # Once the wait ends, the divider falls back to the retrying phase label with the attempt
    # suffix (no reason, no countdown) and never claims the agent is working.
    command_loop.session.state.model_retry_until = 0
    after = command_loop.queue_divider_fragments()
    after_text = "".join(text for _style, text in after)
    assert "retrying · attempt 3/6" in after_text
    assert "working" not in after_text


def test_tui_activity_uses_transient_cancelling_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("cancelling")

    text = "".join(fragment for _style, fragment in command_loop.queue_divider_fragments())

    assert "cancelling" in text
    assert "working" not in text


def test_resume_history_prints_before_tui_starts(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.session.resumed = True
    command_loop.session.messages.extend(
        [
            {"role": "user", "content": "most recent question"},
            {"role": "assistant", "content": "most recent answer"},
        ]
    )
    command_loop.ui.color = True
    printed = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, *args, **kwargs: printed.append(fragment_list_to_text(to_formatted_text(value))))

    command_loop.render_resumed_session()

    text = "".join(printed)
    assert "most recent question" in text
    assert "most recent answer" in text


def test_tui_commands_print_output_immediately(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.ui.color = True
    monkeypatch.setattr(command_loop, "status", lambda _args: "status marker")
    printed = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, *args, **kwargs: printed.append(fragment_list_to_text(to_formatted_text(value))))

    assert command_loop.command("/help") == (True, False)
    assert command_loop.command("/status") == (True, False)
    assert command_loop.command("/skills") == (True, False)

    assert len(printed) == 3
    text = "".join(printed)
    assert "/provider" in text
    assert "status marker" in text
    assert "minacode-help" in text


def test_background_output_is_closed_before_final_output(tmp_path):
    command_loop = loop(tmp_path)
    emitted = []
    command_loop.emit = emitted.append

    command_loop.close_background_output(lambda: emitted.append("final"))
    command_loop.emit_background("late worker output")

    assert emitted == ["final"]


def test_start_session_does_not_scan_or_refresh_code_index(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    status_checks = []
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)
    monkeypatch.setattr(CommandLoop, "clean_expired_sessions_async", lambda _loop: None)
    monkeypatch.setattr(CommandLoop, "render_resumed_session", lambda _loop: None)
    monkeypatch.setattr(command_loop.session.mcp, "discover_auto", lambda: None)
    monkeypatch.setattr(
        CodeIndex,
        "status",
        lambda _index, *, check=False, max_pending_files=20: (status_checks.append(check) or ("ready", "")),
    )
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: pytest.fail("startup refreshed the code index"))

    command_loop.start_session()

    assert status_checks == [False]


def test_start_session_discovers_mcp_off_the_main_thread(tmp_path, monkeypatch):
    """start_session must dispatch auto_connect MCP discovery in the background: an unreachable
    server otherwise blocks the prompt for the whole discovery timeout. Regression guard for the
    lifecycle refactor that had briefly made discover_auto a synchronous startup call."""
    config = Config.from_dict(
        {
            "provider": {"active": "d", "d": {"url": "u", "key": "k", "model": "m"}},
            "mcp": {"slow": {"url": "http://unreachable/mcp", "auto_connect": True}},
            "paths": {"data_dir": str(tmp_path / "data")},
        }
    )
    s = Session(cwd=str(tmp_path), config=config)
    command_loop = CommandLoop(
        Agent(s, output_fn=lambda text: None),
        input_fn=lambda prompt="": "",
        output_fn=lambda text: None,
    )

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    monkeypatch.setattr(CodeIndex, "refresh_existing_async", lambda _index: False)
    monkeypatch.setattr(UpdateChecker, "start", lambda _checker: None)

    discover_started = threading.Event()
    allow_finish = threading.Event()
    ran_on: list[threading.Thread] = []

    def blocking_discover() -> None:
        ran_on.append(threading.current_thread())
        discover_started.set()
        allow_finish.wait(timeout=5)

    monkeypatch.setattr(s.mcp, "discover_auto", blocking_discover)

    try:
        command_loop.start_session()
        # Discovery was dispatched, but start_session returned while it is still blocked —
        # i.e. it ran on a background thread rather than blocking the main (prompt) thread.
        assert discover_started.wait(timeout=2), "discover_auto was never dispatched"
        assert not allow_finish.is_set()
        assert ran_on and ran_on[0] is not threading.main_thread()
    finally:
        allow_finish.set()


def test_input_history_is_trimmed_to_a_bounded_size(tmp_path):
    path = history_file(tmp_path / "history.txt", 5000)
    assert os.path.getsize(path) > CommandLoop.INPUT_HISTORY_BYTES

    CommandLoop.trim_input_history(str(path))

    assert os.path.getsize(path) <= CommandLoop.INPUT_HISTORY_BYTES
    # The newest entries survive, the oldest are the ones dropped, and what remains still loads.
    kept = list(FileHistory(str(path)).load_history_strings())
    assert kept[0].startswith("4999-")
    assert not any(entry.startswith("0-") for entry in kept)
    assert all(entry.split("-")[0].isdigit() for entry in kept)


def test_input_history_trim_cuts_only_at_an_entry_boundary(tmp_path):
    path = history_file(tmp_path / "history.txt", 4000)

    CommandLoop.trim_input_history(str(path))

    # A cut inside an entry would leave a partial first line; the survivor must start with a header.
    with open(path, "rb") as file:
        assert file.read(2) == b"# "
    with open(path, encoding="utf-8") as file:
        text = file.read()
    assert all(line.startswith(("#", "+")) for line in text.splitlines() if line)


def test_input_history_under_the_cap_is_left_alone(tmp_path):
    path = history_file(tmp_path / "history.txt", 10)
    with open(path, "rb") as file:
        before = file.read()

    CommandLoop.trim_input_history(str(path))

    with open(path, "rb") as file:
        assert file.read() == before


def test_input_history_trim_survives_a_missing_or_odd_file(tmp_path):
    CommandLoop.trim_input_history(str(tmp_path / "absent.txt"))  # must not raise

    # One entry larger than the whole budget is kept rather than cut in half.
    path = tmp_path / "huge.txt"
    path.write_bytes(b"\n# 2026-01-01 00:00:00\n+" + b"y" * (CommandLoop.INPUT_HISTORY_BYTES + 1000) + b"\n")
    before = path.read_bytes()

    CommandLoop.trim_input_history(str(path))

    assert path.read_bytes() == before


def test_expired_session_cleanup_reports_without_blocking_startup(monkeypatch, tmp_path):
    """The sweep runs on a daemon thread, so the notice arrives through the background channel."""
    command_loop = loop(tmp_path)
    command_loop.session.settings.session_retention_days = 7
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 3)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="": lines.append(str(text)))

    command_loop.clean_expired_sessions_async()
    for _ in range(200):
        if lines:
            break
        time.sleep(0.01)

    assert len(lines) == 1
    # Says what was lost and which setting governs it, so the knob is discoverable when it acts.
    assert "removed 3 saved sessions" in lines[0]
    assert "7 days" in lines[0]
    assert "session_retention_days" in lines[0]


def test_no_notice_when_nothing_expired(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", lambda _session: 0)
    lines = []
    monkeypatch.setattr(command_loop, "emit", lambda text="": lines.append(str(text)))

    command_loop.clean_expired_sessions_async()
    time.sleep(0.1)

    assert lines == []


def test_expired_session_sweep_never_breaks_startup(monkeypatch, tmp_path):
    """A failing sweep must not escape the thread; retention is not worth a broken session."""
    command_loop = loop(tmp_path)

    def boom(_session):
        raise OSError("data dir unreadable")

    monkeypatch.setattr(SessionSnapshotStore, "clean_expired", boom)

    command_loop.clean_expired_sessions_async()
    time.sleep(0.1)


def test_expired_session_notice_reads_correctly_when_singular(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.settings.session_retention_days = 1

    assert "removed 1 saved session inactive for over 1 day " in command_loop.expired_sessions_notice(1) + " "
