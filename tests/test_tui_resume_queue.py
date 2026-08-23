"""tui resume queue (split from tests/test_tui_runtime.py)."""
import threading
import time
from dataclasses import replace

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tui_harness import loop, session, wait_until

import minacode.cli as loop_module
import minacode.render as render_module
import minacode.tui.app as tui_module
from minacode.cli import QUEUE_SAFE_COMMANDS, CommandLoop, TuiRuntime
from minacode.cli.update import UpdateChecker
from minacode.engine import Agent
from minacode.prompts import LIVE_FOLLOWUP_PREFIX
from minacode.session import Session, SessionSnapshotStore
from minacode.tools import CodeIndex
from minacode.tui import TuiApp


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
    assert "/resend" in QUEUE_SAFE_COMMANDS

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

    retrying = "".join(text for _style, text in command_loop.view.queue_divider_fragments())
    assert "retrying" in retrying
    assert "revising" not in retrying

    command_loop.status_bar.retry_notice_until = 0
    responding = "".join(text for _style, text in command_loop.view.queue_divider_fragments())
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
    retrying = command_loop.view.queue_divider_fragments()
    retrying_text = "".join(text for _style, text in retrying)
    assert "retrying 2/6 · timeout (10s)" in retrying_text
    assert any(text == "● " for _style, text in retrying)
    assert ("retrying 2/6 · timeout", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    now[0] = 102.1
    working = command_loop.view.queue_divider_fragments()
    working_text = "".join(text for _style, text in working)
    assert "working · attempt 2/6 (12s)" in working_text
    assert "retrying" not in working_text
    assert any(text == "● " for _style, text in working)
    assert ("attempt 2/6", "warn") in command_loop.status_bar.entries(show_elapsed=True)

    command_loop.session.state.current_model_call_started_at = 0.0
    assert all(text != "● " for _style, text in command_loop.view.queue_divider_fragments())

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

    waiting = command_loop.view.queue_divider_fragments()
    waiting_text = "".join(text for _style, text in waiting)
    assert "retrying 3/6 · server error · 20s" in waiting_text

    # Core fix: an in-flight wait keeps the full text even after the two-second notice window
    # expired, because a long backoff wait can outlast that window entirely.
    still_waiting = command_loop.view.queue_divider_fragments()
    still_text = "".join(text for _style, text in still_waiting)
    assert "retrying 3/6 · server error · 20s" in still_text

    # Once the wait ends, the divider falls back to the retrying phase label with the attempt
    # suffix (no reason, no countdown) and never claims the agent is working.
    command_loop.session.state.model_retry_until = 0
    after = command_loop.view.queue_divider_fragments()
    after_text = "".join(text for _style, text in after)
    assert "retrying · attempt 3/6" in after_text
    assert "working" not in after_text

def test_tui_activity_uses_transient_cancelling_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running("cancelling")

    text = "".join(fragment for _style, fragment in command_loop.view.queue_divider_fragments())

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
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *values, **kwargs: printed.extend(fragment_list_to_text(to_formatted_text(value)) for value in values))

    command_loop.render_resumed_session()

    text = "".join(printed)
    assert "most recent question" in text
    assert "most recent answer" in text

def test_resume_redraws_only_the_recent_turns_and_says_so(tmp_path, monkeypatch):
    """A long session does not flood the terminal on resume: only the newest turns are redrawn,
    with a line saying the earlier ones stayed in context."""
    command_loop = loop(tmp_path)
    command_loop.session.resumed = True
    messages = []
    for index in range(23):
        messages.append({"role": "user", "content": f"question {index}"})
        messages.append({"role": "assistant", "content": f"answer {index}"})
    command_loop.session.messages.extend(messages)
    command_loop.ui.color = True
    printed = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *values, **kwargs: printed.extend(fragment_list_to_text(to_formatted_text(value)) for value in values))

    command_loop.render_resumed_session()

    text = "".join(printed)
    assert "3 earlier turns not redrawn (still in context)" in text
    assert "question 22" in text and "answer 22" in text  # the newest turn is redrawn
    assert "question 3" in text  # the first visible turn is redrawn
    assert "question 0" not in text and "answer 0" not in text  # the earliest turns are not

def test_resume_redraw_keeps_tool_pairing_after_truncation(tmp_path, monkeypatch):
    """Truncating the redraw still pairs the visible turns with their own tool results: the
    folded turns advance the record cursor without rendering anything."""
    command_loop = loop(tmp_path)
    command_loop.session.resumed = True
    command_loop.session.store_tool_result("Bash", ["printf old"], "old out")
    command_loop.session.store_tool_result("Bash", ["printf new"], "new out")
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "Bash", "arguments": '["printf old"]'}}]},
        {"role": "assistant", "content": "old answer"},
    ]
    for index in range(20):
        messages.append({"role": "user", "content": f"question {index}"})
        messages.append({"role": "assistant", "content": f"answer {index}"})
    messages.append({"role": "user", "content": "new question"})
    messages.append({"role": "assistant", "tool_calls": [{"id": "call-2", "type": "function", "function": {"name": "Bash", "arguments": '["printf new"]'}}]})
    messages.append({"role": "assistant", "content": "new answer"})
    command_loop.session.messages.extend(messages)
    command_loop.ui.color = True
    printed = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *values, **kwargs: printed.extend(fragment_list_to_text(to_formatted_text(value)) for value in values))

    command_loop.render_resumed_session()

    text = "".join(printed)
    assert "2 earlier turns not redrawn (still in context)" in text
    assert "tr.2" in text  # the newest call pairs with its own record
    assert "tr.1" not in text  # the folded turn's record is not rendered

def test_tui_commands_print_output_immediately(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    command_loop.ui.color = True
    # Dispatch calls the registry's callable directly, so patch the registry entry (not the
    # instance method) to keep the /status handler deterministic.
    status_entry = replace(loop_module.COMMAND_LOOKUP["/status"], handler=lambda _loop, _args: "status marker")
    monkeypatch.setattr(loop_module, "COMMAND_LOOKUP", {**loop_module.COMMAND_LOOKUP, "/status": status_entry})
    printed = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *values, **kwargs: printed.extend(fragment_list_to_text(to_formatted_text(value)) for value in values))

    assert command_loop.command("/help") == (True, False)
    assert command_loop.command("/status") == (True, False)
    assert command_loop.command("/skills") == (True, False)

    assert len(printed) == 3
    text = "".join(printed)
    assert "/provider" in text
    assert "status marker" in text
    assert "minacode-help" in text
