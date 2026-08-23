"""Rendering: themes, the external editor, the status bar, the Bash live preview, width
clipping, and choice-view state."""

import itertools
import os
import shutil
import sys
import time

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from tui_harness import loop, session

import minacode.render as render_module
from minacode.base import (
    SELECTION_BACK,
    SELECTION_FREE_TEXT,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    Text,
)
from minacode.config import (
    request_budget_for,
)
from minacode.render import BashLivePreview, StatusBar, Theme, UiPrinter
from minacode.tools import AskSpec
from minacode.tui import ASK_DONE, ASK_FREE_TEXT, TUI_MODAL_PENDING, AskViewState, ChoiceViewState, TuiApp






















































# One small, lexer-exercising sample per language an agent routinely edits. `.yaml` and `.pl` are
# the two that actually broke; the rest are here so the next style/lexer pairing that does the
# same is caught by this test rather than by an Edit dying in someone's session.
HIGHLIGHT_SAMPLES = {
    "a.py": "def f(x):\n    return {'k': x}\n",
    "a.js": "const a = {b: 1};\n",
    "a.ts": "let a: number = 1;\n",
    "a.tsx": "const A = () => <div id='x'/>;\n",
    "a.go": "package main\nfunc main() {}\n",
    "a.rs": "fn main() { let x = 1; }\n",
    "a.rb": "def f(x)\n  {k: x}\nend\n",
    "a.java": "class A { int x = 1; }\n",
    "a.c": "int main(void){return 0;}\n",
    "a.sh": 'set -e\necho "$HOME"\n',
    "a.yaml": "jobs:\n  t:\n    - run: pytest\n",
    "a.yml": "a: 1\nb:\n  - c\n",
    "a.toml": '[tool]\nname = "x"\n',
    "a.json": '{"a": [1, null]}\n',
    "a.md": "# T\n\n- a `b`\n",
    "a.html": "<div class='a'>x</div>\n",
    "a.css": "a { color: red; }\n",
    "a.sql": "SELECT * FROM t WHERE x = 1;\n",
    "a.pl": "my $x = 1;\n",
    "Dockerfile": "FROM x\nRUN y\n",
    "Makefile": "all:\n\techo hi\n",
}






































































# --- AskViewState: the Ask modal (options left, rich markdown preview right, batch keys) ---




























class TestCodeLogLines:
    """CODE-role lines: whole-block lexing plus a line-number gutter."""

    def block(self, code: str, lexer: str = "python") -> LogBlock:
        return LogBlock([LogLine("", line, LogRole.CODE, syntax=lexer) for line in code.splitlines()])

    def rendered(self, code: str, lexer: str = "python") -> str:
        return "".join(text for _style, text in UiPrinter().log_segments(self.block(code, lexer)))

    def test_lines_are_numbered_from_one(self):
        text = self.rendered("a = 1\nb = 2\n")
        assert text.splitlines() == ["  1  a = 1", "  2  b = 2"]

    def test_gutter_widens_with_the_line_count(self):
        rows = self.rendered("\n".join(f"x{index} = {index}" for index in range(10))).splitlines()
        assert rows[0].startswith("   1  ") and rows[-1].startswith("  10  ")

    def test_a_multiline_string_is_lexed_as_one_block(self):
        """The reason the whole block is lexed at once: a per-line lexer reads the closing line of a
        triple-quoted string as code and colors it as such."""
        segments = UiPrinter().log_segments(self.block('x = """a\nstill a string\n"""\n'))
        styles = {text.strip(): style for style, text in segments if text.strip()}
        assert styles["still a string"] == styles['"""a'] == styles['"""']

    def test_unknown_lexer_degrades_to_plain_text(self):
        assert self.rendered("a = 1\n", "no-such-lexer").splitlines() == ["  1  a = 1"]


def test_ui_batched_collects_into_one_print_formatted_text_call(monkeypatch):
    """The restored-transcript replay batches: every emit feeds one print call, not one per line."""

    calls: list[tuple[tuple, dict]] = []

    def recording(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(render_module, "print_formatted_text", recording)
    ui = UiPrinter(print)
    ui.color = True
    with ui.batched():
        ui.emit("first")
        with ui.batched():  # nested batches are a no-op, not a re-entry
            ui.emit("second")
        ui.emit_answer("**bold** answer", role="assistant", rule=False, indent=4)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["sep"] == "" and kwargs["flush"] is True
    text = "".join(fragment for part in args for _style, fragment in to_formatted_text(part))
    assert "first" in text and "second" in text and "bold" in text


def test_ui_batched_passthrough_when_plain():
    """Without color there is nothing to batch; each emit still calls output_fn directly."""

    calls: list[str] = []
    ui = UiPrinter(output_fn=calls.append)
    with ui.batched():
        ui.emit("one")
        ui.emit("two")
    assert calls == ["one", "two"]
