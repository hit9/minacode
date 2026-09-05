"""editor and theme (split from tests/test_ui_render.py)."""

import asyncio
import os
import pathlib
import re
import signal
import sys
import threading

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.styles import Style
from rich.console import Console
from tui_harness import loop

import wizolt.render as render_module
import wizolt.tui.app as app_module
from wizolt.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
)
from wizolt.render import StatusBar, Theme, UiPrinter
from wizolt.tui import TuiApp


def test_both_appearances_define_every_role_in_a_shape_the_adapters_accept():
    """Light and dark are the same vocabulary, and every entry is a color both frameworks read."""
    for palette in (render_module.DARK_PALETTE, render_module.LIGHT_PALETTE):
        for role in render_module.THEME_ROLES:
            value = palette.color(role)
            assert value == render_module.TERMINAL_DEFAULT or re.fullmatch(r"#[0-9a-f]{6}", value), (role, value)


def test_unknown_roles_are_rejected_rather_than_silently_uncolored():
    with pytest.raises(KeyError):
        Theme.color("accent-ish")


def test_every_role_resolves_as_a_prompt_toolkit_class_and_a_rich_style(monkeypatch):
    for mode in ("dark", "light"):
        monkeypatch.setattr(Theme, "_mode", mode)
        style = Style.from_dict(Theme.tui_styles())
        console = Console(theme=Theme.rich_theme())
        for role in render_module.THEME_ROLES:
            attrs = style.get_attrs_for_style_str(Theme.tui_class(role))
            # "default" is the terminal's own foreground, which prompt-toolkit spells as None.
            assert attrs.color is not None or Theme.color(role) == render_module.TERMINAL_DEFAULT
            console.get_style(f"wizolt.{role.replace('_', '.')}")  # raises for a name Rich cannot resolve
        for name in ("wizolt.role.user", "wizolt.role.assistant"):
            assert console.get_style(name).bold


def test_status_roles_have_palette_entries():
    assert all(f"status_{role}" in render_module.THEME_ROLES for role in StatusBar.ROLE_KEYS)


def test_diff_colors_survive_the_palette_reorganization(monkeypatch):
    """Diff colors are pinned, not derived: reshuffling the palette must not move them."""
    assert Theme.DIFF_DARK == {
        "diff.added.bg": "bg:#003b00",
        "diff.added.fg": "fg:default",
        "diff.removed.bg": "bg:#520000",
        "diff.removed.fg": "fg:default",
    }
    assert Theme.DIFF_LIGHT == {
        "diff.added.bg": "bg:#d1f0d1",
        "diff.added.fg": "fg:#003b00",
        "diff.removed.bg": "bg:#f5c8c8",
        "diff.removed.fg": "fg:#520000",
    }
    for mode, expected in (("dark", Theme.DIFF_DARK), ("light", Theme.DIFF_LIGHT)):
        monkeypatch.setattr(Theme, "_mode", mode)
        assert {key: Theme.diff_style(key) for key in expected} == expected


def test_missing_pygments_degrades_to_plain_text_rather_than_failing(monkeypatch):
    monkeypatch.setattr(render_module, "pygments", None)
    monkeypatch.setattr(render_module, "get_style_by_name", None)
    monkeypatch.setattr(Theme, "_pygments_cache", {})
    assert Theme.pygments_style() is None
    assert UiPrinter.code_lines("x = 1\n", "python") is None
    assert UiPrinter.syntax_segments("x = 1", "python", "fg:default") == [("fg:default", "x = 1")]


def test_an_unloadable_pygments_style_degrades_to_plain_text(monkeypatch):
    monkeypatch.setattr(render_module, "get_style_by_name", lambda name: (_ for _ in ()).throw(ValueError(name)))
    monkeypatch.setattr(Theme, "_pygments_cache", {})
    assert Theme.pygments_style() is None
    assert UiPrinter.pygments_style(render_module.Token.Name) == "fg:default"


def test_editor_command_prefers_visual_then_editor_then_vim(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert TuiApp.editor_command() == ["vim"]

    monkeypatch.setenv("EDITOR", "code --wait")
    assert TuiApp.editor_command() == ["code", "--wait"]

    monkeypatch.setenv("VISUAL", "nvim")
    assert TuiApp.editor_command() == ["nvim"]


def fake_editor(tmp_path, monkeypatch, script: str) -> pathlib.Path:
    """Install a shell script as $EDITOR. `$1` is the temp file the editor is handed."""
    editor = tmp_path / "fake_editor.sh"
    editor.write_text("#!/bin/sh\n" + script)
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))
    monkeypatch.delenv("VISUAL", raising=False)
    return editor


async def test_edit_text_in_editor_roundtrips_edited_content(tmp_path, monkeypatch):
    # A fake $EDITOR that appends a marker to whatever file it is given.
    fake_editor(tmp_path, monkeypatch, 'printf " EDITED" >> "$1"\n')

    assert await TuiApp()._edit_text_in_editor("hello") == "hello EDITED"


async def test_edit_text_in_editor_leaves_input_untouched_when_editor_missing(monkeypatch):
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-binary")
    monkeypatch.delenv("VISUAL", raising=False)

    assert await TuiApp()._edit_text_in_editor("hello") is None


async def test_edit_text_in_editor_leaves_input_untouched_on_nonzero_exit(tmp_path, monkeypatch):
    """A non-zero exit (`:cq`, a crash) means "throw this away": the file is not even read."""
    fake_editor(tmp_path, monkeypatch, 'printf " EDITED" >> "$1"\nexit 3\n')

    assert await TuiApp()._edit_text_in_editor("hello") is None


async def test_edit_text_in_editor_removes_its_temp_file(tmp_path, monkeypatch):
    seen = tmp_path / "seen-path"
    fake_editor(tmp_path, monkeypatch, f'printf "%s" "$1" > {seen}\n')

    await TuiApp()._edit_text_in_editor("hello")

    assert not os.path.exists(seen.read_text())


async def test_a_missing_editor_still_removes_its_temp_file(tmp_path, monkeypatch):
    """The launch failed after the file existed; `finally` is what keeps /tmp from filling up."""
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-binary")
    monkeypatch.delenv("VISUAL", raising=False)
    created: list = []
    real_create = app_module._EditorTempFile.create
    monkeypatch.setattr(app_module._EditorTempFile, "create", classmethod(lambda cls, text: created.append(real_create(text)) or created[-1]))

    assert await TuiApp()._edit_text_in_editor("hello") is None

    assert created and not os.path.exists(created[0].path)


async def test_cancelling_temp_file_acquisition_removes_the_created_file(monkeypatch):
    """Cancellation can arrive after mkstemp succeeded but before its worker returns ownership."""
    entered = threading.Event()
    release = threading.Event()
    created = []
    real_create = app_module._EditorTempFile.create

    def slow_create(text):
        temp = real_create(text)
        created.append(temp)
        entered.set()
        release.wait(5)
        return temp

    monkeypatch.setattr(app_module._EditorTempFile, "create", classmethod(lambda _cls, text: slow_create(text)))
    edit = asyncio.create_task(TuiApp()._edit_text_in_editor("hello"))
    assert await asyncio.to_thread(entered.wait, 5)
    edit.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await edit
    assert created and not os.path.exists(created[0].path)


async def test_cancelling_the_editor_terminates_reaps_and_cleans_up(tmp_path, monkeypatch):
    """Cancellation ends the editor rather than leaving it holding the terminal.

    It is not enough for the awaiting task to go away: the child is the runtime's own, so it is
    signalled, waited for, and its scratch file removed before the cancellation is reported."""
    started = tmp_path / "started"
    fake_editor(tmp_path, monkeypatch, f"touch {started}\nsleep 30\n")
    created: list = []
    real_create = app_module._EditorTempFile.create
    monkeypatch.setattr(app_module._EditorTempFile, "create", classmethod(lambda cls, text: created.append(real_create(text)) or created[-1]))
    app = TuiApp()
    processes: list = []
    real_exec = asyncio.create_subprocess_exec

    async def record(*argv, **kwargs):
        process = await real_exec(*argv, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)

    edit = asyncio.ensure_future(app._edit_text_in_editor("hello"))
    while not started.exists():
        await asyncio.sleep(0.01)
    edit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await edit

    assert processes and processes[0].returncode is not None  # signalled and reaped, not orphaned
    assert created and not os.path.exists(created[0].path)


async def test_an_editor_that_ignores_term_is_killed_after_the_grace_period(tmp_path, monkeypatch):
    """TERM first, because a real editor may want to save; KILL because it may also never leave."""
    started = tmp_path / "started"
    fake_editor(tmp_path, monkeypatch, f"trap '' TERM\ntouch {started}\nsleep 30\n")
    app = TuiApp()
    monkeypatch.setattr(TuiApp, "EDITOR_TERM_GRACE", 0.2)
    processes: list = []
    real_exec = asyncio.create_subprocess_exec

    async def record(*argv, **kwargs):
        process = await real_exec(*argv, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)

    edit = asyncio.ensure_future(app._edit_text_in_editor("hello"))
    while not started.exists():
        await asyncio.sleep(0.01)
    edit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await edit

    assert processes[0].returncode == -signal.SIGKILL


async def test_the_loop_keeps_running_while_the_editor_is_open(tmp_path, monkeypatch):
    """The editor is awaited, not waited on: a heartbeat task still advances while it is up."""
    fake_editor(tmp_path, monkeypatch, "sleep 0.3\n")
    beats = 0

    async def heartbeat():
        nonlocal beats
        while True:
            beats += 1
            await asyncio.sleep(0.01)

    pulse = asyncio.ensure_future(heartbeat())
    await TuiApp()._edit_text_in_editor("hello")
    pulse.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pulse

    assert beats > 5


def test_editor_text_compose_and_strip_roundtrip():
    # The editor receives the draft plus the agent's reply below a scissors line; stripping
    # drops the reference context and returns exactly the (possibly edited) draft.
    composed, marker = TuiApp._compose_editor_text("my draft", "reply line one\nline two")
    assert "my draft" in composed
    assert TuiApp.EDITOR_CONTEXT_MARKER in composed
    assert marker and marker in composed
    assert "reply line one" in composed
    assert TuiApp._strip_editor_context(composed, marker) == "my draft"
    # Editing above the scissors line survives; everything below it is dropped.
    assert TuiApp._strip_editor_context(composed.replace("my draft", "edited draft"), marker) == "edited draft"


def test_editor_text_compose_without_context_is_identity():
    assert TuiApp._compose_editor_text("draft", "") == ("draft", "")
    assert TuiApp._compose_editor_text("draft", "   ") == ("draft", "")
    assert TuiApp._strip_editor_context("plain text\n", "") == "plain text"


def test_editor_strip_preserves_a_scissors_line_the_user_typed():
    # Only the marker this composition added is stripped; a scissors line already in the draft
    # (pasted Markdown or code) survives, whether or not reference context was appended.
    draft = f"before\n{TuiApp.EDITOR_CONTEXT_MARKER}\nafter"
    assert TuiApp._strip_editor_context(draft, "") == draft
    composed, marker = TuiApp._compose_editor_text(draft, "reply")
    assert TuiApp._strip_editor_context(composed, marker) == draft


def test_editor_context_returns_last_assistant_reply(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "only answer"},
        {"role": "assistant", "content": None},  # a tool-call turn carries no text
    ]
    assert command_loop.editor_context() == "only answer"

    command_loop.session.messages = [{"role": "user", "content": "only a question"}]
    assert command_loop.editor_context() == ""


def test_editor_context_combines_recent_replies(tmp_path):
    command_loop = loop(tmp_path)
    reply = "\n".join(f"long line {index}" for index in range(150))
    command_loop.session.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Done."},
    ]
    lines = command_loop.editor_context().splitlines()
    assert lines[0] == "Done."  # newest reply first
    assert lines[1] == "# --- (earlier reply) ---"
    assert lines[2] == "long line 0"
    assert lines[-1] == "long line 149"


def test_editor_context_caps_long_replies_to_recent_lines(tmp_path):
    command_loop = loop(tmp_path)
    total = command_loop.EDITOR_CONTEXT_MAX_LINES + 50
    reply = "\n".join(f"line {index}" for index in range(total))
    command_loop.session.messages = [{"role": "assistant", "content": reply}]
    lines = command_loop.editor_context().splitlines()
    # The cap covers the omission note too, so the reply never silently reads as complete.
    assert len(lines) == command_loop.EDITOR_CONTEXT_MAX_LINES
    assert lines[0] == command_loop.EDITOR_CONTEXT_ELLIPSIS
    assert lines[1] == "line 51"
    assert lines[-1] == f"line {total - 1}"


def test_editor_context_combined_budget_keeps_latest_without_note(tmp_path):
    command_loop = loop(tmp_path)
    max_lines = command_loop.EDITOR_CONTEXT_MAX_LINES
    earlier = "\n".join(f"line {index}" for index in range(max_lines))
    latest = "\n".join(f"latest {index}" for index in range(max_lines))
    command_loop.session.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": earlier},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": latest},
    ]
    lines = command_loop.editor_context().splitlines()
    assert len(lines) == max_lines
    assert not any(line.startswith("# [...") for line in lines)
    assert lines[-1] == f"latest {max_lines - 1}"


def test_desert_user_color_does_not_leak_into_default_ui_style(tmp_path, monkeypatch):
    command_loop = loop(tmp_path)
    for mode, expected in (("dark", "fg:#e0a96d"), ("light", "fg:#9a5b2e")):
        monkeypatch.setattr(Theme, "_mode", mode)
        assert UiPrinter.user_log_style() == expected
        assert command_loop.view.style().get_attrs_for_style_str("").color == ""


def test_tool_labels_take_the_palette_tool_color(monkeypatch):
    for mode in ("dark", "light"):
        monkeypatch.setattr(Theme, "_mode", mode)
        assert UiPrinter.log_styles(LogRole.TOOL) == (Theme.fg("tool"), Theme.fg("text"))


@pytest.mark.parametrize(("mode", "rgb"), [("dark", "224;169;109"), ("light", "154;91;46")])
def test_resumed_user_rendering_emits_desert_truecolor(mode, rgb, monkeypatch):
    monkeypatch.setattr(Theme, "_mode", mode)
    ui = UiPrinter(output_fn=lambda text: None)
    console = render_module.markdown_console(40)

    with console.capture() as capture:
        ui.render_message(console, "hello", "user", False, 0)

    assert f"\x1b[38;2;{rgb}m• hello\x1b[0m" in capture.get()


@pytest.mark.parametrize(
    ("configured", "colorfgbg", "expected"),
    [
        ("dark", "0;15", "dark"),
        ("light", "15;0", "light"),
        ("auto", "15;0", "dark"),
        ("auto", "0;7", "light"),
        ("auto", "7;8", "dark"),
        ("auto", "0;;15", "light"),
        ("auto", "invalid", "dark"),
    ],
)
def test_theme_resolution(configured, colorfgbg, expected, monkeypatch):
    monkeypatch.setenv("COLORFGBG", colorfgbg)
    assert Theme.resolve(configured) == expected


def test_tool_argument_rendering_tracks_theme_without_changing_text(monkeypatch):
    line = LogLine("Search", '"needle" path=src 0:20', LogRole.TOOL, syntax="tool-args")
    block = LogBlock([line])
    rendered = []

    for mode in ("dark", "light"):
        monkeypatch.setattr(Theme, "_mode", mode)
        segments = UiPrinter(output_fn=lambda text: None).log_segments(block)
        rendered.append(("".join(text for _, text in segments), {style for style, text in segments if text.strip()}))

    assert rendered[0][0] == rendered[1][0] == '  Search  "needle" path=src 0:20\n'
    assert rendered[0][1] != rendered[1][1]


def test_standalone_turn_rows_carry_no_edge_glyph(tmp_path, monkeypatch):
    """Standalone turn-level rows must not draw an edge. A BRANCH on a row with no parent line
    above it dangles (`├` joined to nothing) and shifts the label two columns past every sibling
    row. Cover the provider builtin-call row so it cannot reintroduce that defect."""
    loop_ = loop(tmp_path)
    captured = []
    monkeypatch.setattr(loop_.ui, "emit", lambda text="", indent=0: captured.append(text))
    loop_.builtin_call_output("search", "cache wiring")
    blocks = [item for item in captured if isinstance(item, LogBlock)]
    assert len(blocks) == 1
    for block in blocks:
        assert all(isinstance(line, LogLine) for line in block.items)
        assert all(line.edge is LogEdge.NONE for line in block.items)

    ui = UiPrinter(output_fn=lambda text: None)
    builtin = "".join(text for _, text in ui.log_segments(blocks[0])).splitlines()[0]
    tool_root = "".join(text for _, text in ui.log_segments(LogBlock([LogLine("Bash", "rg -n cache wizolt/", LogRole.TOOL)]))).splitlines()[0]
    user_echo = "\u2022 [Image #1 \u00b7 ef739e37-....png]"
    # All three rows start their content in the same column (two-cell indent, no edge).
    assert builtin.index("search") == tool_root.index("Bash") == user_echo.index("[Image") == 2


def test_interactive_renderer_keeps_theme_when_parent_exports_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(Theme, "_mode", "dark")
    emitted = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = UiPrinter()
    # Interactive TTY output stays colored regardless of NO_COLOR — wizolt owns its theming and
    # renders through prompt_toolkit's ANSI path, so the parent env var is not honored.
    assert ui.color
    ui.emit_answer("sent message", role="user", rule=False)

    desert_text = "".join(text for style, text in emitted if style == "#e0a96d")
    assert "• sent message" in desert_text
