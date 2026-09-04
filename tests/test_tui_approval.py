"""tui approval (split from tests/test_tui_app.py)."""
import asyncio

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys
from test_tui_app import ACTIONS, _active, _answered, _approval_app
from tui_harness import request_input_from_driver, run_interactive_tui, wait_for, wait_until

from wizolt.tui import TuiApp


async def test_tui_approval_form_fires_the_focused_action_on_enter():
    # Enter submits the focused action's answer -- the same whole line ("", "v", "c", "n") the
    # approval loop already understands, so the form is a renderer, not a second protocol.
    for steps, expected in ((0, ""), (1, "v"), (2, "c"), (3, "n"), (4, "")):  # 4 wraps back to Approve
        app = _approval_app()
        for _ in range(steps):
            _active(app, Keys.Tab)[0].handler(type("Event", (), {})())
        app._accept(app.input_buffer)
        assert _answered(app) == (expected, True)

    app = _approval_app()  # Shift-Tab from the default wraps backwards to the last action
    _active(app, Keys.BackTab)[0].handler(type("Event", (), {})())
    app._accept(app.input_buffer)
    assert _answered(app) == ("n", True)

async def test_tui_approval_form_yields_the_keyboard_once_a_reason_is_typed():
    # The whole point of selecting actions instead of binding letters: nothing a reason might
    # contain is spent on a shortcut, so navigation keys go back to editing the moment there is text.
    app = _approval_app()

    def live(key):
        return [binding for binding in app.make_bindings().bindings if binding.keys == (key,) and binding.filter()]

    assert len(live(Keys.Tab)) == 2  # the action row, layered over Tab's usual completion
    assert len(live(Keys.Escape)) == 1
    assert len(live(Keys.Right)) == 1

    app.input_buffer.reset(Document("cost too high"))
    assert len(live(Keys.Tab)) == 1  # Tab completes again
    assert live(Keys.Right) == []  # ... and the arrows move the cursor
    assert len(live(Keys.Escape)) == 1  # Escape stays bound, but now clears back to the row

    app._accept(app.input_buffer)  # Enter sends the reason rather than firing an action
    assert _answered(app) == ("cost too high", True)

async def test_tui_approval_form_escape_takes_back_a_reason_then_refuses():
    # Escape always undoes the current thing: with a reason typed it clears back to the action row,
    # and with nothing to take back it cancels -- which confirm() reads as a refusal with no reason.
    app = _approval_app()
    escape = _active(app, Keys.Escape)[0].handler
    app.input_buffer.reset(Document("cost too high"))

    escape(type("Event", (), {})())
    assert app.input_buffer.text == ""
    assert _answered(app) == (None, False)  # taken back, not submitted

    escape(type("Event", (), {})())
    assert _answered(app) == (None, True)

    # A prompt with no form binds none of it.
    plain = TuiApp()
    plain._input_loop = asyncio.get_running_loop()
    plain._input_pending = plain._input_loop.create_future()
    plain.input_mode = "approval"
    assert [binding for binding in plain.make_bindings().bindings if binding.keys == (Keys.Escape,) and binding.filter()] == []

async def test_tui_approval_form_row_shows_focus_and_dims_while_typing():
    app = _approval_app()

    def row():
        return "".join(text for _, text in app.approval_form_fragments())

    def styles():
        return [style for style, _ in app.approval_form_fragments()]

    assert all(label in row() for label, _ in ACTIONS)  # every action is visible, none memorized
    assert "class:approval.action.focused" in styles()
    assert "Tab to move" in row()

    _active(app, Keys.Tab)[0].handler(type("Event", (), {})())
    focused = [text for style, text in app.approval_form_fragments() if style == "class:approval.action.focused"]
    assert focused == [" View order "]

    # Typing disarms the row: Enter no longer fires the focused action, so it must stop looking armed.
    app.input_buffer.reset(Document("cost too high"))
    assert "class:approval.action.focused" not in styles()
    assert "Enter send · Esc back" in row()

async def test_tui_ctrl_d_submits_multiline_approval_input():
    app = TuiApp()
    app.input_mode = "approval"
    app._input_loop = asyncio.get_running_loop()
    app._input_pending = app._input_loop.create_future()
    app.input_buffer.reset(Document("first\nsecond"))
    binding = next(binding for binding in reversed(app.make_bindings().bindings) if binding.keys == (Keys.ControlD,) and binding.filter())
    event = type("Event", (), {"app": type("Application", (), {"exit": lambda self: None})()})()

    binding.handler(event)

    assert _answered(app) == ("first\nsecond", True)

def test_tui_ctrl_g_and_ctrl_x_ctrl_e_open_editor():
    opened = []
    app = TuiApp()
    app.edit_input_in_editor = lambda: opened.append(True)
    bindings = app.make_bindings()
    event = type("Event", (), {})()

    # A fresh TuiApp is in chat mode, where the editor bindings are active.
    for keys in ((Keys.ControlG,), (Keys.ControlX, Keys.ControlE)):
        binding = next(binding for binding in bindings.bindings if binding.keys == keys)
        assert binding.filter()
        binding.handler(event)

    assert opened == [True, True]

async def test_tui_app_approval_mode_resolves_the_pending_request():
    app = TuiApp()
    request = asyncio.ensure_future(app.request_input_async("[Y/n] "))
    await wait_for(lambda: app.input_mode == "approval")

    app.input_buffer.insert_text("y")
    app.input_buffer.validate_and_handle()

    assert await request == "y"
    assert app.input_mode == "chat"

async def test_tui_approval_restores_half_typed_draft():
    app = TuiApp()
    app.set_running("working")
    app.input_buffer.insert_text("unfinished draft")

    request = asyncio.ensure_future(app.request_input_async("Approve? "))
    await wait_for(lambda: app.input_mode == "approval")
    assert app.input_buffer.text == ""
    app.input_buffer.insert_text("y")
    app.input_buffer.validate_and_handle()

    assert await request == "y"
    assert app.input_mode == "running"
    assert app.input_buffer.text == "unfinished draft"

@pytest.mark.parametrize(
    ("prompt", "expected_above", "expected_prefix"),
    [
        ("\nPick?", [""], "Pick?"),  # Ask free-text: a blank line above the question
        ("Which one?\nA) foo\nB) bar", ["Which one?", "A) foo"], "B) bar"),  # a multi-line question
        ("\n\nStarts with its own newline", ["", ""], "Starts with its own newline"),
        ("[Y/n or reason] ", [], "[Y/n or reason] "),  # an ordinary tool approval is untouched
    ],
)
async def test_tui_approval_prompt_never_renders_a_newline_as_a_control_character(prompt, expected_above, expected_prefix):
    """The input row's prefix is a single-line BeforeInput processor, and BufferControl does not
    split processor output on "\n" the way FormattedTextControl does -- a literal newline reaches
    the screen as "^J". Every line but the last is rendered as its own row above the input."""
    app = TuiApp()
    request = asyncio.ensure_future(app.request_input_async(prompt))
    await wait_for(lambda: app.input_mode == "approval")

    assert app.input_prompt == expected_prefix
    assert app._input_prompt_above == expected_above
    assert all("\n" not in text for _, text in app.status_fragments())
    assert app.full_input_prompt() == prompt  # nothing is dropped, only relocated

    app.input_buffer.insert_text("typed")
    app.input_buffer.validate_and_handle()

    assert await request == "typed"
    assert app._input_prompt_above == []  # the rows belong to one prompt, not the restored mode

def test_interactive_tui_ctrl_c_cancels_approval_without_interrupting_turn(monkeypatch):
    interrupted = []
    result = []
    app = TuiApp(on_interrupt=lambda: interrupted.append(True))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        approval = request_input_from_driver(app)
        wait_until(lambda: app.input_mode == "approval")
        pipe_input.send_text("\x03")
        result.append(approval.result(timeout=2))
        wait_until(lambda: app.input_mode == "chat")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    # Ctrl-C cancels the approval: the request resolves to None, which is neither "" (confirm()
    # reads that as the default approve) nor text the model could mistake for a typed answer.
    # The turn itself is not interrupted.
    assert result == [None]
    assert interrupted == []

def test_interactive_tui_ctrl_d_on_an_empty_approval_cancels_instead_of_approving(monkeypatch):
    # EOF on an empty approval line used to submit "", which confirm() reads as the default
    # approve -- the same trap Ctrl-C fell into. It must cancel; with text typed it still submits.
    result = []
    app = TuiApp()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        for typed in ("", "too risky"):
            approval = request_input_from_driver(app)
            wait_until(lambda: app.input_mode == "approval")
            if typed:
                pipe_input.send_text(typed)
                wait_until(lambda text=typed: app.input_buffer.text == text)
            pipe_input.send_text("\x04")
            result.append(approval.result(timeout=2))
            wait_until(lambda: app.input_mode == "chat")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None, "too risky"]

def test_interactive_tui_exit_while_an_approval_is_pending_cancels_it(monkeypatch):
    # Shutting the app down resolves whatever is waiting on the user, but must not do so with ""
    # -- that would have confirm() grant the pending call on the way out.
    result = []
    app = TuiApp()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        approval = request_input_from_driver(app)
        wait_until(lambda: app.input_mode == "approval")
        app.app.loop.call_soon_threadsafe(app.app.exit)
        result.append(approval.result(timeout=2))

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None]
