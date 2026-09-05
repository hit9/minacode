"""tui mentions (split from tests/test_tui_app.py)."""

import time

import pytest
from tui_harness import run_interactive_tui, wait_until

from wizolt.cli import CommandCompleter
from wizolt.mentions import FilePick, active_mention
from wizolt.tui import TuiApp


def _recording_picker(queries):
    """A picker that records the query it was opened with and selects nothing."""

    async def pick(query):
        queries.append(query)
        return FilePick()

    return pick


def test_mention_opens_completions_while_typing(monkeypatch):
    """`@`, `@kind:`, and `$` name something the completer knows, so the list opens as they are
    typed and narrows as more characters arrive - everything else in this prompt is prose and
    waits for Tab."""
    app = TuiApp(
        completer=CommandCompleter(
            mcp_servers=lambda: ("github", "gitlab", "playwright"),
            skills=lambda: ("release",),
            files=lambda: (("wizolt/tui.py", "wizolt/tui.py"), ("wizolt/hints.py", "wizolt/hints.py")),
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
        wait_until(lambda: completions() == ["@file:wizolt/tui.py"])

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
    async def pick(query):
        return FilePick("docs/中文 notes.txt") if query == "not" else FilePick()

    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=pick)

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

    async def pick(query):
        queries.append(query)
        return FilePick("wizolt/tui.py")

    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=pick)

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("inspect @file:")
        wait_until(lambda: app.input_buffer.text == "inspect @file:wizolt/tui.py")
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)

    assert queries == [""]


@pytest.mark.parametrize("typed", ["@f", "@fi"])
def test_selecting_partially_typed_file_kind_opens_picker(monkeypatch, typed):
    queries = []
    app = TuiApp(
        completer=CommandCompleter(),
        file_picker_available_fn=lambda: True,
        file_picker_fn=_recording_picker(queries),
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


def test_browsing_bare_kind_menu_does_not_launch_file_picker(monkeypatch):
    """Highlighting @file: in the bare-@ menu is a preview, not a choice: arrow/Tab through the
    three kind rows without the file picker grabbing the terminal, and Enter on a later row
    commits it (the picker only opens on an explicit Enter on @file:)."""
    queries = []
    app = TuiApp(
        completer=CommandCompleter(mcp_servers=lambda: ("github",), skills=lambda: ("release", "review")),
        file_picker_available_fn=lambda: True,
        file_picker_fn=_recording_picker(queries),
    )

    def state():
        current = app.input_buffer.complete_state
        return None if current is None else (current.complete_index, [c.text for c in current.completions])

    kinds = ["@file:", "@mcp:", "@skill:"]

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@")
        wait_until(lambda: state() is not None and state()[1] == kinds)

        for expected_index, expected_text in ((0, "@file:"), (1, "@mcp:"), (2, "@skill:")):
            pipe_input.send_text("\x1b[B")
            wait_until(lambda text=expected_text, idx=expected_index: app.input_buffer.text == text and state() is not None and state()[0] == idx)
            assert state()[1] == kinds  # still browsing the same three kind rows
            assert queries == [] and not app._file_picker_active

        pipe_input.send_text("\r")
        wait_until(lambda: state() is None and app.input_buffer.text == "@skill:")
        assert queries == [] and not app._file_picker_active
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_enter_on_at_file_kind_row_opens_the_file_picker(monkeypatch):
    """Browsing to @file: is inert; an explicit Enter on the row commits the kind and opens the
    picker with an empty query, exactly as typing the namespace does."""
    queries = []
    app = TuiApp(
        completer=CommandCompleter(),
        file_picker_available_fn=lambda: True,
        file_picker_fn=_recording_picker(queries),
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@")
        wait_until(lambda: app.input_buffer.complete_state is not None)
        pipe_input.send_text("\x1b[B")
        wait_until(lambda: app.input_buffer.text == "@file:")
        assert queries == [] and not app._file_picker_active  # preview alone must not open it
        pipe_input.send_text("\r")
        wait_until(lambda: queries == [""] and not app._file_picker_active)
        assert app.input_buffer.text == "@file:"
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_pasting_file_namespace_does_not_open_picker(monkeypatch):
    queries = []
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=_recording_picker(queries))

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
    app = TuiApp(file_picker_available_fn=lambda: True, file_picker_fn=_recording_picker(queries))

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("@file:keep")
        wait_until(lambda: queries == ["keep"] and not app._file_picker_active)
        time.sleep(app.MENTION_TRANSITION_DELAY * 2)
        assert app.input_buffer.text == "@file:keep"
        assert queries == ["keep"]  # Cancel does not reopen until the input changes again.
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_enter_commits_highlighted_completion_without_sending(monkeypatch):
    """Tab previews a mention row; Enter commits it into the input instead of sending the
    message, so the prompt stays open and a second Enter sends."""
    submitted = []
    app = TuiApp(
        completer=CommandCompleter(skills=lambda: ("release", "review")),
        on_chat_submit=submitted.append,
    )

    def state():
        current = app.input_buffer.complete_state
        return None if current is None else (current.complete_index, [c.text for c in current.completions])

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("use @skill:")
        wait_until(lambda: (state() or (None, []))[1] == ["@skill:release", "@skill:review"])
        pipe_input.send_text("\t")
        wait_until(lambda: state() is not None and state()[0] is not None)
        pipe_input.send_text("\r")
        wait_until(lambda: state() is None and app.input_buffer.text == "use @skill:release")
        assert submitted == []
        pipe_input.send_text("\r")
        wait_until(lambda: submitted == ["use @skill:release"])
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)


def test_enter_sends_when_completion_menu_has_no_highlighted_row(monkeypatch):
    """The menu opens while typing with no row highlighted; Enter there still sends, so a fully
    typed mention goes out in one press (only Tab-highlighted rows are committed by Enter)."""
    submitted = []
    app = TuiApp(
        completer=CommandCompleter(skills=lambda: ("release", "review")),
        on_chat_submit=submitted.append,
    )

    def drive(pipe_input):
        wait_until(lambda: app.app is not None and app.app.is_running)
        pipe_input.send_text("use @skill:")
        wait_until(lambda: app.input_buffer.complete_state is not None and app.input_buffer.complete_state.current_completion is None)
        pipe_input.send_text("\r")
        wait_until(lambda: submitted == ["use @skill:"])
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(monkeypatch, app, drive=drive)
