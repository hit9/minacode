"""tui runtime output (split from tests/test_tui_runtime.py)."""
import threading
from types import SimpleNamespace

from prompt_toolkit.formatted_text import fragment_list_to_text
from test_tui_runtime import TextRecordingOutput
from model_harness import async_create
from tui_harness import loop, run_interactive_tui, session, wait_until

from wizolt.base import (
    MalformedToolCallError,
    ToolCall,
    TurnBox,
)
from wizolt.cli import CommandLoop, TuiRuntime
from wizolt.config import ProviderConfig
from wizolt.engine import Agent
from wizolt.tools import CodeIndex
from wizolt.tui import TuiApp


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
    # A plain final answer is published by the engine through output_fn now, never by the
    # post-turn emit; the post-turn emit only prints errors the engine raised first.
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

    assert emitted == []  # the engine printed the answer; the runtime does not repeat it

def test_search_sources_footer_is_indented_like_the_answer_above_it(tmp_path, monkeypatch):
    """The footer belongs to the answer, and the engine publishes that answer through
    emit_agent_output at CONTENT_LEVEL. At column 0 the sources would hang off the left of the
    text they cite."""
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    command_loop.tui.set_running = lambda label: None
    command_loop.agent.run = lambda _text: "the final answer"
    command_loop.agent.turn_sources = [{"url": "https://a.example", "title": "A"}]
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)
    emitted: list[tuple[str, int]] = []
    command_loop.ui.emit_answer = lambda text, **kwargs: emitted.append((text, kwargs.get("indent", 0)))

    runtime = TuiRuntime(command_loop)
    runtime.run_agent_turn("do it")

    assert len(emitted) == 1  # the footer alone; the engine published the answer itself
    text, indent = emitted[0]
    assert "a.example" in text
    assert indent == TurnBox.CONTENT_LEVEL

def test_automatic_compaction_replaces_working_divider_status(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.config.providers["default"] = ProviderConfig(model="gpt-4", url="http://test", key="sk-test")
    command_loop.session.settings.max_context_tokens = 1
    command_loop.session.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        *({"role": "assistant", "content": f"recent {index}"} for index in range(8)),
        {"role": "user", "content": "latest request"},
    ]
    command_loop.tui = TuiApp()
    divider_during_compaction = []

    def compact(_messages, _tools, **_kwargs):
        divider_during_compaction.append(fragment_list_to_text(command_loop.view.queue_divider_fragments()))
        return "", "", '{"summary": "compact summary"}'

    command_loop.agent.model.api_request = compact

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
    command_loop.emit = lambda text="", indent=0: emitted.append((text, command_loop.view.model_stream_fragments()))
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
    app = TuiApp(activity_fragments_fn=command_loop.view.tui_activity_fragments)
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

    responses = SimpleNamespace(create=async_create(lambda **_params: events()))
    monkeypatch.setattr(command_loop.agent.model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    real_emit = command_loop.emit_agent_output

    def emit_promoted(text):
        real_emit(text)
        timeline.append("white response")

    monkeypatch.setattr(command_loop, "emit_agent_output", emit_promoted)

    def request():
        try:
            _, _, content = command_loop.agent.model.request([{"role": "user", "content": "make the change"}], [])
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
            assert command_loop.view.model_stream_fragments() == []
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
                "action": {"type": "search", "query": "wizolt"},
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
                "action": {"type": "search", "query": "wizolt"},
            },
        },
        {"type": "response.completed", "response": terminal},
    ]
    responses = SimpleNamespace(create=async_create(lambda **_params: iter(events)))
    monkeypatch.setattr(command_loop.agent.model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    emitted = []
    monkeypatch.setattr(command_loop, "emit_agent_output", emitted.append)

    _, _, content = command_loop.agent.model.request([{"role": "user", "content": "search"}], None)
    command_loop.agent_output(content)

    assert emitted == [answer]
    assert command_loop.model_stream_promoted_text == ""
    assert command_loop.view.model_stream_fragments() == []

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
    call = {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "wizolt"}}
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
    responses = SimpleNamespace(create=async_create(lambda **_params: iter(events)))
    monkeypatch.setattr(command_loop.agent.model, "client", lambda **kwargs: SimpleNamespace(responses=responses))
    emitted = []
    monkeypatch.setattr(command_loop, "emit_agent_output", emitted.append)

    _, _, content = command_loop.agent.model.request([{"role": "user", "content": "search"}], None)
    command_loop.agent_output(content)

    assert emitted == [lead, rest]
    assert command_loop.model_stream_promoted_text == ""

def test_turn_end_answer_drops_the_prefix_already_promoted_into_scrollback(tmp_path, monkeypatch):
    """The final answer is published once even when a mid-response promotion wrote its opening.

    The engine publishes through agent_output, which consumes the one-shot promotion marker
    and skips the already-promoted prefix; the runtime's post-turn emit no longer prints the
    answer (the engine did), so nothing repeats it."""
    command_loop = loop(tmp_path)
    command_loop.tui = TuiApp()
    runtime = TuiRuntime(command_loop)
    emitted = []
    command_loop.ui.emit_answer = lambda text, **_kwargs: emitted.append(text)
    monkeypatch.setattr(CodeIndex, "update_pending_async", lambda _index: None)

    def answer(_user_input):
        with command_loop.model_stream_lock:
            command_loop.model_stream_promoted_text = "Let me look that up."
        command_loop.agent_output("Let me look that up.\n\nThe searched answer.")
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
    monkeypatch.setattr(command_loop, "emit_agent_answer", lambda text: timeline.append(("assistant", text)))
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

    assert timeline == [
        ("user", ["also update the README"]),
        ("assistant", "Sure, editing both files."),
        ("assistant", "done"),  # the engine publishes the final answer through output_fn
    ]

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
