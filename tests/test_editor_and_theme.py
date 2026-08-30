"""editor and theme (split from tests/test_ui_render.py)."""
import sys

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from rich.console import Console
from tui_harness import loop

import wizolt.render as render_module
from wizolt.base import (
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
)
from wizolt.render import StatusBar, Theme, UiPrinter
from wizolt.tui import TuiApp


def test_theme_palettes_have_identical_complete_keys():
    assert Theme.DARK.keys() == Theme.LIGHT.keys()
    assert all(Theme.DARK.values())
    assert all(Theme.LIGHT.values())

def test_status_roles_have_theme_entries():
    assert all(f"status.{role}" in Theme.DARK and f"status.{role}" in Theme.LIGHT for role in StatusBar.ROLE_KEYS)

def test_editor_command_prefers_visual_then_editor_then_vim(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert TuiApp.editor_command() == ["vim"]

    monkeypatch.setenv("EDITOR", "code --wait")
    assert TuiApp.editor_command() == ["code", "--wait"]

    monkeypatch.setenv("VISUAL", "nvim")
    assert TuiApp.editor_command() == ["nvim"]

def test_edit_text_in_editor_roundtrips_edited_content(tmp_path, monkeypatch):
    # A fake $EDITOR that appends a marker to whatever file it is given.
    editor = tmp_path / "fake_editor.sh"
    editor.write_text('#!/bin/sh\nprintf " EDITED" >> "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") == "hello EDITED"

def test_edit_text_in_editor_leaves_input_untouched_when_editor_missing(monkeypatch):
    monkeypatch.setenv("EDITOR", "definitely-not-an-editor-binary")
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") is None

def test_edit_text_in_editor_leaves_input_untouched_on_nonzero_exit(monkeypatch):
    monkeypatch.setenv("EDITOR", "false")
    monkeypatch.delenv("VISUAL", raising=False)

    assert TuiApp()._edit_text_in_editor("hello") is None

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
    for mode, expected in (("dark", "#e0a96d"), ("light", "#9a5b2e")):
        monkeypatch.setattr(Theme, "_mode", mode)
        assert UiPrinter.user_log_style() == expected
        assert command_loop.view.style().get_attrs_for_style_str("").color == ""

def test_tool_labels_keep_legacy_green_style():
    assert UiPrinter.LOG_STYLES[LogRole.TOOL][0] == "ansigreen"

@pytest.mark.parametrize(("mode", "rgb"), [("dark", "224;169;109"), ("light", "154;91;46")])
def test_resumed_user_rendering_emits_desert_truecolor(mode, rgb, monkeypatch):
    monkeypatch.setattr(Theme, "_mode", mode)
    ui = UiPrinter(output_fn=lambda text: None)
    console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=40)

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
