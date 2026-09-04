"""loop display (split from tests/test_loop_commands.py)."""

import asyncio
from types import SimpleNamespace

import pytest
from agent_harness import call, queue, session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from test_loop_commands import queued_texts

import wizolt.cli.loop as loop_module
from wizolt.base import (
    LogBlock,
    TurnBox,
    __version__,
)
from wizolt.cli import CommandLoop
from wizolt.engine import Agent
from wizolt.render import UiPrinter
from wizolt.session import Session
from wizolt.tui import TuiApp


async def test_run_refuses_to_nest_the_cli_runtime(tmp_path):
    loop = CommandLoop(
        Agent(session(tmp_path), output_fn=lambda _text: None),
        input_fn=lambda _prompt: "",
        output_fn=lambda _text: None,
    )

    with pytest.raises(RuntimeError, match="cannot be called from a running event loop"):
        loop.run()


async def test_interactive_banner_precedes_tui_terminal_setup(tmp_path, monkeypatch):
    command_loop = CommandLoop(
        Agent(session(tmp_path), output_fn=lambda _text: None),
        input_fn=lambda _prompt: "",
        output_fn=lambda _text: None,
    )
    command_loop.interactive_input = True
    events = []
    tui_started = asyncio.Event()
    finish_tui = asyncio.Event()
    monkeypatch.setattr(command_loop, "emit_banner", lambda: events.append("banner"))

    class Runtime:
        def __init__(self, owner):
            assert owner is command_loop

        async def run(self, *, show_banner=True):
            tui_started.set()
            await finish_tui.wait()
            events.append(("tui", show_banner))
            return 7

    monkeypatch.setattr(loop_module, "TuiRuntime", Runtime)

    running = asyncio.create_task(command_loop._run_frontend())
    await tui_started.wait()
    # A terminal's first CPR can take a second to time out. The banner must already be visible
    # throughout that wait, and the eventual TUI startup must not print it again.
    assert events == ["banner"]
    finish_tui.set()
    assert await running == 7
    assert events == ["banner", ("tui", False)]


async def test_ps_command_uses_markdown_renderer(tmp_path):
    s = session(tmp_path)
    s.jobs["job.1"] = SimpleNamespace(id="job.1", status="running", command="pytest -q", elapsed=lambda: 13.7, update_status=lambda: None)
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    rendered = []
    plain = []
    loop.ui.emit_answer = lambda text, **kwargs: rendered.append(text)
    loop.emit = lambda text="", indent=0: plain.append(text)

    assert await loop.command("/ps") == (True, False)

    assert plain == []
    assert len(rendered) == 1
    assert rendered[0].startswith("### Active jobs")
    assert "| id | status | elapsed | command |" in rendered[0]


def test_emit_indents_plain_text_without_losing_its_style(tmp_path):
    """`emit(text, indent)` moves a plain line into a column. The margin is applied after
    styling, because `segments` dispatches on what the text starts with -- prefixing spaces
    first would strip every line of its color. A blank line gets no margin, or an empty emit
    would print a row of trailing spaces into the scrollback."""
    del tmp_path
    ui = UiPrinter(lambda _text: None)
    ui.color = True
    margin = LogBlock.margin(TurnBox.CONTENT_LEVEL)

    error = ui.indent_segments(ui.segments("Error: provider is down"), margin)
    assert error[0] == ("ansired", margin)  # the margin carries the style of the line it opens
    assert "".join(text for _, text in error) == f"{margin}Error: provider is down\n"

    two_lines = ui.indent_segments(ui.segments("first\nsecond"), margin)
    assert "".join(text for _, text in two_lines) == f"{margin}first\n{margin}second\n"

    assert "".join(text for _, text in ui.indent_segments(ui.segments(""), margin)) == "\n"


def test_turn_output_shares_one_column_and_session_chrome_does_not(tmp_path):
    """One left edge for the exchange: the user's line (its `• ` bullet hanging in the margin),
    the turn outcome, and a command's reply. The banner and the resume line frame it flush left."""
    s = session(tmp_path)
    output = []
    loop = CommandLoop(Agent(s, output_fn=output.append), input_fn=lambda prompt: "", output_fn=output.append)
    margin = LogBlock.margin(TurnBox.CONTENT_LEVEL)

    loop.emit_turn("Cancelled")
    loop.emit(f"wizolt {__version__}. /help for commands.")

    assert output == [f"{margin}Cancelled", f"wizolt {__version__}. /help for commands."]


def test_tui_completion_applies_single_match():
    class OneCompletion(Completer):
        def get_completions(self, document, _complete_event):
            yield Completion("hello", start_position=-len(document.text))

    buffer = Buffer(document=Document("he"), completer=OneCompletion())
    TuiApp.complete_input(buffer)
    assert buffer.text == "hello"


def test_tui_completion_starts_and_cycles_multiple_matches():
    class MultipleCompletions(Completer):
        def get_completions(self, document, _complete_event):
            yield Completion("alpha", start_position=-len(document.text))
            yield Completion("alpine", start_position=-len(document.text))

    completer = MultipleCompletions()
    buffer = Buffer(document=Document("al"), completer=completer)
    started = []
    buffer.start_completion = lambda **kwargs: started.append(kwargs)

    TuiApp.complete_input(buffer)
    assert started == [{"select_first": False}]

    completions = list(completer.get_completions(buffer.document, CompleteEvent()))
    buffer._set_completions(completions)
    TuiApp.complete_input(buffer)
    assert buffer.text == "alpha"
    TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "al"
    TuiApp.complete_input(buffer, reverse=True)
    assert buffer.text == "alpine"


def test_queue_acknowledges_only_claimed_duplicate_messages(tmp_path):
    s = session(tmp_path)
    queue(s, "same", "same")
    claimed = s.claim_user_inputs()
    s.enqueue_user_input("same")

    s.acknowledge_user_inputs(claimed)

    assert queued_texts(s) == ["same"]
    assert not s.pending_user_inputs[0].inflight


def test_queue_release_restores_interrupted_inputs(tmp_path):
    s = session(tmp_path)
    s.enqueue_user_input("ready")
    queued = s.pending_user_inputs[0]

    assert s.claim_user_inputs() == [queued]
    s.release_user_inputs()

    assert not queued.inflight


def test_recall_pending_input_can_revise_latest_inflight_message(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "second")
    s.claim_user_inputs()
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)
    retried = []

    text = loop.recall_pending_input(lambda: retried.append(True))

    assert text == "second"
    assert queued_texts(s) == ["first"]
    assert s.pending_user_inputs[0].inflight is False
    assert retried == [True]


async def test_clearing_recalled_message_leaves_it_deleted(tmp_path):
    s = session(tmp_path)
    queue(s, "first", "delete me")
    loop = CommandLoop(Agent(s, output_fn=lambda text: None), input_fn=lambda prompt: "", output_fn=lambda text: None)

    assert loop.recall_pending_input(lambda: None) == "delete me"
    # Recall mutates the queue; persisting it is the runtime's submission consumer, which is what
    # the await stands in for here.
    await s.save_snapshot()

    assert queued_texts(s) == ["first"]
    restored = Session.load_snapshot(s.uid, config=s.config)
    assert queued_texts(restored) == ["first"]


def test_pending_user_inputs_auto_submit_at_round_end(tmp_path):
    """Unconsumed pending_user_inputs are auto-submitted as next input."""
    s = session(tmp_path)
    queue(s, "leftover instruction")
    requested_tools = []

    class FakeModel:
        async def request(self, messages, tools=None):
            requested_tools.extend(tools or [])
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()

    def fake_read(prompt="", **kw):
        raise EOFError()

    loop = CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)

    loop.run()

    assert s.pending_user_inputs == []
    assert s.next_hints_available is False
    assert "NextHints" not in {tool["function"]["name"] for tool in requested_tools}
    assert any("leftover instruction" in msg.get("content", "") for msg in s.messages)


def test_simple_repl_schema_stays_next_hints_free_across_requests(tmp_path):
    """The simple REPL chooses its tool set before the first model request without NextHints,
    and that set stays stable on later requests: no tool is inserted or removed between
    requests, so the tool-schema prefix does not churn."""
    s = session(tmp_path)
    requested: list[set[str]] = []

    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def request(self, messages, tools=None):
            requested.append({tool["function"]["name"] for tool in (tools or [])})
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "content": ""}, [call("Read", [{"path": "missing"}])], ""
            return {"role": "assistant", "content": "done"}, [], "done"

    agent = Agent(s, output_fn=lambda text: None)
    agent.model = FakeModel()
    inputs = iter(["do it"])

    def fake_read(prompt="", **kw):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError()

    loop = CommandLoop(agent, input_fn=fake_read, output_fn=lambda text: None)
    loop.run()

    assert s.next_hints_available is False
    assert len(requested) == 2  # before the tool batch and before the final answer
    assert all("NextHints" not in names for names in requested)
    assert requested[0] == requested[1]
