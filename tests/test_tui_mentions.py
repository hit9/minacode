"""tui mentions (split from tests/test_tui_app.py)."""
import time

import pytest
from tui_harness import run_interactive_tui, wait_until

from minacode.cli import CommandCompleter
from minacode.mentions import FilePick, active_mention
from minacode.tui import TuiApp


def test_mention_opens_completions_while_typing(monkeypatch):
    """`@`, `@kind:`, and `$` name something the completer knows, so the list opens as they are
    typed and narrows as more characters arrive - everything else in this prompt is prose and
    waits for Tab."""
    app = TuiApp(
        completer=CommandCompleter(
            mcp_servers=lambda: ("github", "gitlab", "playwright"),
            skills=lambda: ("release",),
            files=lambda: (("minacode/tui.py", "minacode/tui.py"), ("minacode/hints.py", "minacode/hints.py")),
        )
    )

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)

        pipe_input.send_text("use @gi")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab"])

        pipe_input.send_text("th")
        wait_until(lambda: completions() == ["@mcp:github"])  # the list narrows as typing continues

        pipe_input.send_text(" and @")
        wait_until(lambda: completions() == ["@file:", "@mcp:", "@skill:"])

        pipe_input.send_text("mcp:")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab", "@mcp:playwright"])

        pipe_input.send_text("gi")
        wait_until(lambda: completions() == ["@mcp:github", "@mcp:gitlab"])

        pipe_input.send_text(" and @file:tu")
        wait_until(lambda: completions() == ["@file:minacode/tui.py"])

        pipe_input.send_text(" and $")
        wait_until(lambda: completions() == ["@skill:release"])

        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

def test_selecting_mention_kind_opens_its_candidate_list(monkeypatch):
    app = TuiApp(completer=CommandCompleter(skills=lambda: ("release", "review")))

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@")
        wait_until(lambda: completions() == ["@file:", "@mcp:", "@skill:"])

        # Shift-Tab selects the last namespace row. Once that selection settles, its own candidates
        # replace the parent namespace menu without another key press.
        pipe_input.send_text("\x1b[Z")
        wait_until(lambda: app.input_buffer.text == "@skill:")
        wait_until(lambda: completions() == ["@skill:release", "@skill:review"])
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

@pytest.mark.parametrize(
    ("typed", "namespace", "expected"),
    [
        ("@m", "@mcp:", ["@mcp:github", "@mcp:gitlab"]),
        ("@sk", "@skill:", ["@skill:release", "@skill:review"]),
    ],
)
def test_selecting_partially_typed_name_kind_opens_its_candidate_list(monkeypatch, typed, namespace, expected):
    app = TuiApp(completer=CommandCompleter(mcp_servers=lambda: ("github", "gitlab"), skills=lambda: ("release", "review")))

    def completions():
        state = app.input_buffer.complete_state
        return None if state is None else [c.text for c in state.completions]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(typed)
        wait_until(lambda: completions() == [namespace])
        pipe_input.send_text("\t")
        wait_until(lambda: app.input_buffer.text == namespace)
        wait_until(lambda: completions() == expected)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

def test_prose_and_email_do_not_open_completions(monkeypatch):
    """A menu on every keystroke would be noise: only a mention at the cursor opens one, and an
    address is not a mention because the `@` follows a word character."""
    app = TuiApp(completer=CommandCompleter(mcp_servers=lambda: ("github",)))
    seen = []

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("mail me at hit9@icloud")
        wait_until(lambda: app.input_buffer.text == "mail me at hit9@icloud")
        seen.append(app.input_buffer.complete_state)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert seen == [None]

def test_mention_trigger_uses_canonical_scanner_spans():
    for text in (
        "use @file:tu",
        "use @",
        "use @skill:rel",
        "use @mcp:git",
    ):
        assert active_mention(text) is not None, text
    assert active_mention("mail me at hit9@icloud") is None
    assert active_mention("use file:notes here") is None
    assert active_mention("profile:x") is None

def test_file_picker_tab_replaces_only_active_span(monkeypatch):
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: FilePick("docs/中文 notes.txt") if query == "not" else FilePick())

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("inspect @file:not please")
        for _ in range(len(" please")):
            pipe_input.send_text("\x1b[D")
        pipe_input.send_text("\t")
        wait_until(lambda: app.input_buffer.text == 'inspect @file:"docs/中文 notes.txt" please')
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

def test_file_picker_opens_after_typing_without_tab(monkeypatch):
    queries = []
    app = TuiApp(
        file_picker_available_fn=lambda: True,
        file_picker_fn=lambda query: (queries.append(query), FilePick("minacode/tui.py"))[1],
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("inspect @file:")
        wait_until(lambda: app.input_buffer.text == "inspect @file:minacode/tui.py")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert queries == [""]

@pytest.mark.parametrize("typed", ["@f", "@fi"])
def test_selecting_partially_typed_file_kind_opens_picker(monkeypatch, typed):
    queries = []
    app = TuiApp(
        completer=CommandCompleter(),
        file_picker_available_fn=lambda: True,
        file_picker_fn=lambda query: (queries.append(query), FilePick())[1],
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text(typed)
        wait_until(lambda: app.input_buffer.complete_state is not None)
        pipe_input.send_text("\t")
        wait_until(lambda: queries == [""] and not app._file_picker_active)
        assert app.input_buffer.text == "@file:"
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

def test_pasting_file_namespace_does_not_open_picker(monkeypatch):
    queries = []
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: (queries.append(query), FilePick())[1])

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("\x1b[200~@file:\x1b[201~")
        wait_until(lambda: app.input_buffer.text == "@file:")
        time.sleep(app.MENTION_TRANSITION_DELAY * 2)
        assert queries == []
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

def test_file_picker_cancel_keeps_buffer(monkeypatch):
    queries = []
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=lambda query: (queries.append(query), FilePick())[1])

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@file:keep")
        wait_until(lambda: queries == ["keep"] and not app._file_picker_active)
        time.sleep(app.MENTION_TRANSITION_DELAY * 2)
        assert app.input_buffer.text == "@file:keep"
        assert queries == ["keep"]  # Cancel does not reopen until the input changes again.
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)
