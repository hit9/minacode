"""emit and stream (split from tests/test_ui_render.py)."""
import os
import shutil
import sys
import time

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.utils import get_cwidth
from test_ui_render import HIGHLIGHT_SAMPLES
from tui_harness import loop

import minacode.render as render_module
from minacode.base import (
    Text,
)
from minacode.render import BashLivePreview, Theme, UiPrinter
from minacode.tui import TuiApp


def test_emit_turn_end_non_color_uses_elapsed_since_format():
    emitted = []
    ui = UiPrinter(output_fn=emitted.append)
    assert not ui.color

    ui.emit_turn_end(time.monotonic() - 5)
    ui.emit_turn_end(time.monotonic() - 65)

    # The footer reuses the divider's `elapsed_since` format: no leading `0m`, seconds zero-padded.
    assert emitted == ["done in 5s", "done in 1m05s"]

def test_emit_turn_end_renders_a_left_aligned_rule_under_a_blank_line(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    emitted = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = UiPrinter()
    assert ui.color
    ui.emit_turn_end(time.monotonic() - 65)

    text = "".join(fragment for _, fragment in emitted)
    # A blank line lifts the rule off the answer; the label sits just past a short lead (left-biased,
    # not centered, not flush) with a long trail of dashes running to the full width.
    assert "\n── done in 1m05s " in text
    assert "──────" in text

def test_editor_and_queued_user_text_use_desert_style(tmp_path, monkeypatch):
    monkeypatch.setattr(Theme, "_mode", "dark")
    expected = UiPrinter.user_log_style()
    app = TuiApp()
    app.build_layout()
    assert app.input_window.style == expected

    command_loop = loop(tmp_path)
    command_loop.session.enqueue_user_input("queued message")
    sent, waiting = command_loop.view.followup_fragments()
    assert any(style == expected and "queued message" in text for style, text in [*sent, *waiting])

def test_activity_blank_line_separates_flushed_followup_from_the_stream(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.enqueue_user_input("queued message")
    command_loop.session.claim_user_inputs()  # inflight: renders as the sent (echoed) follow-up
    command_loop.model_stream_kind = "output"
    command_loop.model_stream_text = "streamed reply line"

    text = "".join(fragment for _, fragment in command_loop.view.tui_activity_fragments())
    lines = text.splitlines()
    echo = next(index for index, line in enumerate(lines) if "queued message" in line)
    # Exactly one blank row separates the echoed follow-up from the stream's first row, so the
    # two never sit pressed together.
    assert lines[echo + 1] == ""
    assert lines[echo + 2]
    assert "streamed reply line" in text

def test_activity_leaves_no_hanging_blank_row_when_nothing_streams(tmp_path):
    command_loop = loop(tmp_path)
    command_loop.session.enqueue_user_input("queued message")
    command_loop.session.claim_user_inputs()

    text = "".join(fragment for _, fragment in command_loop.view.tui_activity_fragments())
    lines = text.splitlines()
    echo = next(index for index, line in enumerate(lines) if "queued message" in line)
    # The blank row between the echo and the divider is separation, not a hanging row: the
    # divider always follows it, so the activity region never ends on an empty line.
    assert lines[echo + 1] == ""
    assert lines[echo + 2]
    assert lines[-1]  # the divider closes the region; nothing streams below it

@pytest.mark.parametrize("width", [20, 40, 80])
def test_styled_wrapping_respects_terminal_width_for_unicode(width):
    prefix = [("", "  Read  ")]
    continuation = [("", "        ")]
    content_text = "路径/非常长/🙂/é/模块/filename.py:123"
    rows = Text.wrap_styled(prefix, continuation, [("fg:default", content_text)], width)

    assert "".join(text for _, text in rows[0]).startswith("  Read  ")
    assert all(sum(get_cwidth(text) for _, text in row) <= width for row in rows)
    assert "".join(text for row in rows for _, text in row).replace("  Read  ", "", 1).replace("        ", "") == content_text

@pytest.mark.parametrize("mode", ["dark", "light"])
def test_every_lexer_token_maps_to_a_style_in_both_themes(mode):
    """A pygments style only covers the tokens its authors thought about, and `style_for_token`
    raises KeyError for the rest instead of returning a default.

    The YAML lexer emits `Token.Literal.Scalar.Plain` and `Token.Punctuation.Indicator`, which
    neither theme's style names, so every Edit to a `.yaml` died rendering its own diff preview
    and reported the token name as the error -- CI configs, compose files, k8s manifests. Perl's
    `Token.Literal.String.Atom` is the same hole. Both themes, so neither was a way out.

    Token lookup has to be total for every lexer we can reach, which is what this sweeps."""
    lexers = pytest.importorskip("pygments.lexers")
    previous = Theme._mode
    try:
        Theme.set_mode(mode)
        for name, text in HIGHLIGHT_SAMPLES.items():
            lexer = lexers.get_lexer_for_filename(name, stripnl=False)
            for token_type, _ in lexer.get_tokens(text):
                style = UiPrinter.pygments_style(token_type)  # must not raise for any of them
                assert isinstance(style, str) and style
    finally:
        Theme.set_mode(previous)

def test_highlighting_inherits_from_the_token_hierarchy_rather_than_giving_up():
    """Not crashing is the floor. A token the style never named still has ancestors that carry a
    color, so a YAML plain scalar renders like the Literal it is instead of dropping the file to
    unstyled text -- and the Edit that previews it survives end to end."""
    pygments_token = pytest.importorskip("pygments.token")
    previous = Theme._mode
    try:
        Theme.set_mode("dark")
        ui = UiPrinter(lambda _text: None)
        diff = (
            "--- a/.github/workflows/ci.yaml\n"
            "+++ b/.github/workflows/ci.yaml\n"
            "@@ -40,3 +40,3 @@\n"
            "       - name: test\n"
            "-        run: uv run pytest\n"
            "+        run: uv run pytest -q\n"
        )
        assert "uv run pytest -q" in "".join(text for _, text in ui.diff_segments(diff))

        literal = UiPrinter.pygments_style(pygments_token.Token.Literal.Scalar.Plain)
        assert literal == UiPrinter.pygments_style(pygments_token.Token.Literal.String)  # inherited
        assert literal != "fg:default"  # and not quietly flattened
    finally:
        Theme.set_mode(previous)

def test_a_lexer_that_fails_mid_stream_costs_the_color_not_the_render():
    """get_tokens is a generator, so a broken lexer raises while the caller pulls from it, not
    when it is called. Highlighting is decoration; it must never take down the edit it previews."""

    class ExplodingLexer:
        def get_tokens(self, _text):
            yield ("Token.Text", "fine so far\n")
            raise RuntimeError("lexer blew up mid-stream")

    assert UiPrinter._tokenized_lines(ExplodingLexer(), "anything") is None

def test_bash_live_preview_clips_wide_output_to_terminal_width(monkeypatch):
    preview = BashLivePreview()
    preview.active = True
    preview.text = "界" * 20

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((20, 24)))
        assert all(get_cwidth("".join(text for _, text in row)) < 20 for row in preview.frame_rows())

def test_bash_live_preview_rewrites_previous_frame_without_appending(tmp_path, monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *args, **kwargs: None)
    preview = BashLivePreview()
    preview.output = recording_output
    preview.active = True
    preview.started_at = 100.0

    preview.render()
    first_rows = preview.rendered_lines
    recording_output.events.clear()
    preview.text = "line one\nline two"
    preview.render()

    assert recording_output.events[0] == ("write", f"\x1b[{first_rows}A")
    assert sum(event == "erase" for event, _ in recording_output.events) == preview.rendered_lines
    assert recording_output.events[-1] == ("flush", "")

def test_bash_live_preview_render_skips_identical_frames(monkeypatch, recording_output):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(render_module, "print_formatted_text", lambda *args, **kwargs: None)
    preview = BashLivePreview()
    preview.output = recording_output
    preview.active = True
    preview.started_at = 100.0

    preview.render()
    rows_before = preview.rendered_lines
    recording_output.events.clear()

    preview.render()
    assert len(recording_output.events) == 0
    assert preview.rendered_lines == rows_before

    preview.text = "new line"
    preview.render()
    assert len(recording_output.events) > 0
