"""tui modals (split from tests/test_tui_app.py)."""
import multiprocessing
import threading

import pytest
from prompt_toolkit.data_structures import Size
from test_tui_input import ctrl_c_queue_scenario
from tui_harness import ResizableOutput, loop, rendered_screen_text, run_interactive_tui, wait_until

from wizolt.cli.commands import select_choice
from wizolt.prompts import LIVE_FOLLOWUP_PREFIX
from wizolt.tui import TUI_MODAL_PENDING, TuiApp


def test_interactive_tui_modal_uses_real_j_and_enter_keys(monkeypatch):
    app = TuiApp()
    selected = {"index": 0}
    result = []

    def key(key, _data):
        if key == "j":
            selected["index"] = 1
            return TUI_MODAL_PENDING
        if key == "enter":
            return selected["index"]
        return TUI_MODAL_PENDING

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(target=lambda: result.append(app.show_modal(lambda: [("", "one\ntwo")], key)), daemon=True)
        waiter.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("j\r")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [1]

@pytest.mark.parametrize("exclusive", [False, True])
def test_interactive_tui_modal_survives_repeated_resize(monkeypatch, exclusive):
    app = TuiApp()
    output = ResizableOutput()
    result = []
    rendered = threading.Event()

    def fragments():
        return [("", "\n".join(f"choice {index}" for index in range(40)))]

    def key(key, _data):
        return None if key == "q" else TUI_MODAL_PENDING

    def after_render(_application):
        rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(target=lambda: result.append(app.show_modal(fragments, key, exclusive=exclusive)), daemon=True)
        waiter.start()
        wait_until(lambda: app.modal is not None)
        for rows, columns in ((10, 40), (35, 120), (8, 24), (24, 80)):
            rendered.clear()
            output.size = Size(rows=rows, columns=columns)
            app.app.loop.call_soon_threadsafe(app.app._on_resize)
            assert rendered.wait(timeout=1)
        pipe_input.send_text("q")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert result == [None]

def test_interactive_tui_renders_a_multi_line_question_as_rows_not_control_characters(monkeypatch):
    """The whole point of the split, on a real screen: every line of a multi-line Ask question is
    visible and no "^J" is drawn."""
    app = TuiApp()
    output = ResizableOutput(rows=12, columns=60)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        text = rendered_screen_text(application, output)
        if "B) bar" in text:
            frames.append(text)
            rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        asking = threading.Thread(target=lambda: app.request_input("\nWhich one?\nA) foo\nB) bar"), daemon=True)
        asking.start()
        wait_until(lambda: app.input_mode == "approval")
        assert rendered.wait(timeout=1)
        pipe_input.send_text("x\r")
        asking.join(timeout=1)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert "^J" not in frames[-1]
    for line in ("Which one?", "A) foo", "B) bar"):
        assert line in frames[-1]

@pytest.mark.parametrize("exclusive", [False, True])
def test_interactive_tui_modal_presentation_matches_legacy_scope(monkeypatch, exclusive):
    app = TuiApp(status_fragments_fn=lambda: [("", "status marker")])
    output = ResizableOutput(rows=12, columns=60)
    frames = []
    rendered = threading.Event()

    def after_render(application):
        text = rendered_screen_text(application, output)
        if "modal marker" in text:
            frames.append(text)
            rendered.set()

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(
            target=lambda: app.show_modal(lambda: [("", "modal marker")], lambda key, _data: None if key == "q" else TUI_MODAL_PENDING, exclusive=exclusive),
            daemon=True,
        )
        waiter.start()
        wait_until(lambda: app.modal is not None)
        assert rendered.wait(timeout=1)
        pipe_input.send_text("q")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive, output=output, after_render=after_render)

    assert "status marker" in frames[-1]
    if exclusive:
        assert frames[-1].splitlines()[-1] == "status marker"

def test_interactive_command_loop_ctrl_c_stops_llm_and_returns_to_input(tmp_path):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=ctrl_c_queue_scenario, args=(str(tmp_path), results))
    process.start()
    process.join(timeout=6)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        pytest.fail("Ctrl-C TUI scenario did not exit within 6 seconds")

    assert process.exitcode == 0
    outcome = results.get(timeout=1)
    assert "fatal" not in outcome, outcome
    assert outcome["driver_errors"] == []
    assert outcome["return_code"] == 0
    assert outcome["elapsed"] and outcome["elapsed"][0] < 1.0
    assert outcome["cancel_calls"] == 1
    assert "long request" in outcome["requests"][0]
    queued_request = outcome["requests"][1]
    assert "queued one" in queued_request
    marked_followup = LIVE_FOLLOWUP_PREFIX + "queued two"
    assert marked_followup in queued_request
    assert queued_request.index("queued one") < queued_request.index(marked_followup)
    assert outcome["draft_after_ctrl_c"] == [""]
    # The interrupted first turn produced no output, so it is retracted: "long request" leaves no
    # trace in the persisted conversation, while the queued follow-ups become the next turn.
    # The follow-up keeps the marker it was sent with; only the transcript hides it.
    assert outcome["persisted_user_inputs"] == ["queued one", marked_followup]
    assert outcome["restored_queue"] == []

def test_interactive_tui_ctrl_c_closes_modal_and_restores_input_focus(monkeypatch):
    received = []
    app = None

    def submit(text):
        received.append(text)
        app.app.exit()

    app = TuiApp(on_chat_submit=submit)

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        waiter = threading.Thread(
            target=lambda: app.show_modal(lambda: [("", "selector")], lambda _key, _data: TUI_MODAL_PENDING),
            daemon=True,
        )
        waiter.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\x03")
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        app.set_idle()
        wait_until(lambda: app.modal is None and app.app.layout.current_window is app.input_window)
        pipe_input.send_text("after cancel\r")

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert received == ["after cancel"]

def test_interactive_tui_resolved_modal_allows_followup_approval(monkeypatch):
    app = TuiApp()
    selected = []
    approved = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: selected.append(
                app.show_modal(
                    lambda: [("", "selector")],
                    lambda key, _data: "chosen" if key == "enter" else TUI_MODAL_PENDING,
                )
            ),
            daemon=True,
        )
        selector.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\r")
        selector.join(timeout=1)
        assert not selector.is_alive()
        wait_until(lambda: app.modal is None and app.app.layout.current_window is app.input_window)

        approval = threading.Thread(target=lambda: approved.append(app.request_input("Approve? ")), daemon=True)
        approval.start()
        wait_until(lambda: app.input_mode == "approval")
        pipe_input.send_text("y\r")
        approval.join(timeout=1)
        assert not approval.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert selected == ["chosen"]
    assert approved == ["y"]

def test_interactive_tui_choice_ctrl_c_reports_cancellation(monkeypatch, tmp_path):
    command_loop = loop(tmp_path)
    command_loop.interactive_input = True
    output = []
    command_loop.emit = lambda text="", indent=0: output.append(text)
    app = TuiApp()
    command_loop.tui = app
    result = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        selector = threading.Thread(
            target=lambda: result.append(select_choice(command_loop, "Pick", ("a", "b"))),
            daemon=True,
        )
        selector.start()
        wait_until(lambda: app.modal is not None)
        pipe_input.send_text("\x03")
        selector.join(timeout=1)
        assert not selector.is_alive()
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert result == [None]
    assert output == ["Cancelled"]
