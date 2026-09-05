"""TUI slash-command completion: a leading "/" opens the command list as it is typed."""

from tui_harness import run_interactive_tui, wait_until

from wizolt.cli import CommandCompleter, CommandLoop
from wizolt.tui import TuiApp


def _completions(app):
    state = app.input_buffer.complete_state
    return None if state is None else [c.text for c in state.completions]


def test_leading_slash_opens_command_completions_and_narrows_while_typing(monkeypatch):
    """A "/" at the start of the line names a command, so the list opens on the first character,
    without Tab, and narrows as more characters arrive -- matching how @/$ mentions behave."""
    app = TuiApp(completer=CommandCompleter())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)

        pipe_input.send_text("/")
        wait_until(lambda: _completions(app) == list(CommandLoop.COMMANDS))

        # Narrowing happens at a partial match; completing the full word adds nothing, so
        # prompt-toolkit then closes the menu on its own.
        pipe_input.send_text("reas")
        wait_until(lambda: _completions(app) == ["/reason"])

        # A word after the command stops completion: the line is no longer a bare command name.
        pipe_input.send_text(" on")
        wait_until(lambda: app.input_buffer.text == "/reas on")
        wait_until(lambda: _completions(app) is None)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_tab_still_browses_an_auto_opened_command_menu(monkeypatch):
    """Tab keeps its role on a menu the first "/" opened: it highlights a row to pick or commit."""
    app = TuiApp(completer=CommandCompleter())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)

        pipe_input.send_text("/")
        wait_until(lambda: _completions(app) is not None)
        pipe_input.send_text("\t")
        wait_until(lambda: app.input_buffer.complete_state is not None and app.input_buffer.complete_state.complete_index == 0)

        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)
