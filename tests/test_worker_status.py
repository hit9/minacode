"""worker status (split from tests/test_worker_handoff.py)."""
from test_worker_handoff import FakeModelClient, _delegate_call, _delegate_runner, _delegate_session

from wizolt.cli.worker import worker_command


def test_status_bar_shows_worker_segment(tmp_path):
    from wizolt.config import (
        Config,
        ProviderConfig,
    )
    from wizolt.render import StatusBar
    from wizolt.session import Session

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
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)
    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    def label():
        return "".join(text for _, text in loop.view.queue_divider_fragments())

    assert "[worker]" not in label()
    worker._active_turn_messages.append({"role": "user", "content": "order"})
    assert "[worker]" in label()

async def test_worker_model_stream_is_wired_from_the_runner(tmp_path, monkeypatch):
    parent = _delegate_session(tmp_path)
    model = FakeModelClient([({"role": "assistant", "content": "done"}, [], "done")])
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    runner = _delegate_runner(parent)
    calls = []
    runner.model_stream = lambda kind, text: calls.append((kind, text))
    await _delegate_call(parent, runner, action="send", order="o")
    on_stream = parent.worker._agent.model.on_stream
    assert on_stream is not runner.model_stream  # wrapped: `output_done` must not promote
    assert callable(on_stream)
    on_stream("output", "x")
    on_stream("output_done", "t")
    assert calls == [("output", "x"), ("", "")]

async def test_status_reports_worker_delegation_state(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    outputs: list = []
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=outputs.append)

    async def status_text():
        outputs.clear()
        await loop.command("/status")
        return "\n".join(str(text) for text in outputs)

    # No worker session: one `worker` row naming the configured [worker] provider. Everything is
    # one flat table — the session's own rows, the parent's, then the worker's under `worker*`.
    text = await status_text()
    assert text.lstrip().startswith("| field | value |")  # rendered in the content column
    assert "###" not in text
    assert "[worker] provider" in text and "default" in text

    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker

    # A fresh worker has no requests yet: the worker context row says so instead of inventing
    # tokens, and the model row mirrors the parent's (provider/model, api, reasoning).
    text = await status_text()
    assert "| worker | `default/" in text
    assert "| worker ctx | (no requests yet); `idle`, rounds `0` |" in text

    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    worker.usage.last_cached_prompt_tokens = 25
    worker.usage.prompt_tokens = 50
    worker.usage.cached_prompt_tokens = 25
    text = await status_text()
    # Scope to the worker rows: the parent's own cache row also says "(no requests yet)".
    worker_section = text.split("| worker |", 1)[1]
    assert "~50 / 100" in worker_section and "(no requests yet)" not in worker_section
    assert "| worker cache | " in worker_section and "last `50.0%`; session `50.0%`" in worker_section

    worker._active_turn_messages.append({"role": "user", "content": "order"})
    text = await status_text()
    assert "`delegating`, rounds `0`" in text

def test_worker_status_command_is_human_readable(tmp_path):
    from wizolt.cli import CommandLoop
    from wizolt.engine import Agent
    from wizolt.session import Session

    parent = _delegate_session(tmp_path)
    agent = Agent(parent, output_fn=lambda text: None)
    loop = CommandLoop(agent, input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert worker_command(loop, "") == chr(10).join(["worker: no active session", "worker provider: default"])
    assert worker_command(loop, "status") == worker_command(loop, "")

    worker = Session(cwd=str(tmp_path), config=parent.config, settings=parent.settings, uid=parent.uid + ".w", listed=False)
    parent.worker = worker
    worker.config.provider.model = "worker-model-x"
    worker.config.provider.reasoning = "high"
    worker.state.round_count = 3
    worker.usage.last_prompt_tokens = 50
    worker.usage.last_prompt_budget = 100
    text = worker_command(loop, "status")
    assert "worker: default/worker-model-x" in text
    assert "worker reasoning: high" in text
    assert "worker state: idle" in text
    assert "worker rounds: 3" in text
    assert "worker context: 50%" in text
    assert "<Delegate" not in text

    worker._active_turn_messages.append({"role": "user", "content": "order"})
    assert "worker state: delegating" in worker_command(loop, "status")

    # Without provider-reported usage the state estimate is the fallback, like the envelope.
    worker.usage.last_prompt_budget = 0
    worker.state.context_percent = 42
    assert "worker context: 42%" in worker_command(loop, "status")

async def test_worker_output_wraps_model_text_for_the_log_stream(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    tool_call = ToolCall(id="call1", name="Note", args={"action": "view"})
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "thinking out loud"}, [tool_call], "thinking out loud"),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    outputs = []
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=outputs.append)
    await _delegate_call(parent, runner, action="send", order="o")

    assert outputs, "the worker turn produced no output"
    rendered = [str(block) for block in outputs if isinstance(block, LogBlock)]  # str items raised before the fix
    assert any("thinking out loud" in text for text in rendered)

async def test_worker_interim_model_text_routes_to_worker_answer_when_wired(tmp_path, monkeypatch):
    from wizolt.base import LogBlock, ToolCall
    from wizolt.context import ContextManager
    from wizolt.runner import ToolRunner

    parent = _delegate_session(tmp_path)
    tool_call = ToolCall(id="call1", name="Note", args={"action": "view"})
    model = FakeModelClient(
        [
            ({"role": "assistant", "content": "**thinking out loud**"}, [tool_call], "**thinking out loud**"),
            ({"role": "assistant", "content": "done"}, [], "done"),
        ]
    )
    monkeypatch.setattr("wizolt.engine.ModelClient", lambda session: model)
    log_outputs = []
    answer_outputs = []
    append_answer = answer_outputs.append
    runner = ToolRunner(parent, ContextManager(parent), input_fn=lambda *a: "y", output_fn=log_outputs.append)
    runner.worker_answer = append_answer

    await _delegate_call(parent, runner, action="send", order="o")

    # Interim model text and final answer route to worker_answer for markdown rendering.
    assert answer_outputs == ["**thinking out loud**", "done"]
    # Worker tool output still flows through the ordinary log stream as LogBlock.
    assert any(isinstance(block, LogBlock) for block in log_outputs)
    # The bindings that make the split work: the worker agent's model text goes to the markdown
    # hook, and its tool runner is pinned to the plain log wrapper -- dropping the pin would let
    # tool output slip into the markdown channel.
    agent = parent.worker._agent
    assert agent.output_fn is append_answer
    assert agent.tools.output_fn is not append_answer

def test_worker_output_passes_memory_shaped_text_through_for_highlighting():
    from types import SimpleNamespace

    from wizolt.base import LogBlock, LogLine, LogRole
    from wizolt.tools.delegate import _worker_output

    outputs = []
    emit = _worker_output(SimpleNamespace(output_fn=outputs.append))
    note = "goal: do x\nplan:\n  - [x] done"
    emit(note)
    emit("plain model text")
    emit(LogLine("", "tool line", LogRole.TOOL))

    # The memory-shaped string passes through as a bare str so the parent's segments() ->
    # memory_segments() applies the per-line colors; other strings and LogLine items stay wrapped.
    assert outputs[0] is note
    assert isinstance(outputs[1], LogBlock)
    assert isinstance(outputs[2], LogBlock)
