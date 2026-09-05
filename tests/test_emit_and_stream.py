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

import wizolt.render as render_module
from wizolt.base import (
    Text,
)
from wizolt.render import BashLivePreview, Theme, UiPrinter
from wizolt.tui import TuiApp


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
    ui.trailing_blanks = 0  # an answer sits above it, with no gap of its own
    ui.emit_turn_end(time.monotonic() - 65)

    text = "".join(fragment for _, fragment in emitted)
    # A blank line lifts the rule off the answer; the label sits just past a short lead (left-biased,
    # not centered, not flush) with a long trail of dashes running to the full width.
    assert "\n── done in 1m05s " in text
    assert "──────" in text


def test_emit_turn_end_does_not_add_a_gap_to_one_that_is_already_there(monkeypatch):
    """Spacing is stated, not counted: a rule under a block that already ended in a blank row does
    not open a second one."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    emitted = []
    monkeypatch.setattr(render_module, "print_formatted_text", lambda value, **_kwargs: emitted.extend(to_formatted_text(value)))

    ui = UiPrinter()
    ui.emit("an answer")
    ui.emit("")
    ui.emit_turn_end(time.monotonic())

    text = "".join(fragment for _, fragment in emitted)
    assert "\n\n\n" not in text

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

        # The walk-up itself, on the style that first hit the hole: github-dark raises for a YAML
        # plain scalar, whose nearest named ancestor is the Literal it is a kind of. (A style that
        # names the root token, as the built-in ones now do, answers for every token itself, so it
        # cannot show whether the fallback works.)
        styles = pytest.importorskip("pygments.styles")
        github = styles.get_style_by_name("github-dark")
        plain = UiPrinter.token_definition(github, pygments_token.Token.Literal.Scalar.Plain)
        assert plain == github.style_for_token(pygments_token.Token.Literal)  # inherited
        assert plain is not None and plain["color"]  # and not quietly flattened
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


MARKDOWN_SAMPLE = (
    "# Title\n\nA paragraph with `code` and a [link](https://example.com).\n\n"
    "## Section\n\n- first\n- second\n\n> quoted\n\n"
    "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n```python\ndef f():\n    return 1\n```\n\nClosing.\n"
)


def rendered_answer(text, width, *, mode="dark", monkeypatch=None, **kwargs):
    """One emit_answer, as the plain rows it puts into scrollback."""
    printed = []
    ui = UiPrinter(output_fn=lambda _text: None)
    ui.color = True
    ui._scrollback_print = lambda fragment: printed.append(fragment)
    previous = Theme._mode
    size = os.terminal_size((width, 24))
    real_size = shutil.get_terminal_size
    shutil.get_terminal_size = lambda *args, **kw: size
    try:
        Theme.set_mode(mode)
        ui.emit_answer(text, **kwargs)
    finally:
        shutil.get_terminal_size = real_size
        Theme.set_mode(previous)
    plain = "".join(fragment for _, fragment in to_formatted_text(printed[0]))
    return render_module.UiPrinter.SGR_RE.sub("", plain).split("\n")


@pytest.mark.parametrize("width", [80, 120])
def test_markdown_blocks_are_parted_by_a_stable_gap(width):
    """A document has one rhythm: a blank row between blocks, two above a heading because a heading
    opens a section, and nothing else. No construct arrives glued to the one above it, and none
    carries an outer margin of its own -- the gap above a block belongs to the printer."""
    rows = rendered_answer(MARKDOWN_SAMPLE, width, role="assistant", rule=False)

    for opener in ("def f():", "▌ quoted", "• first", "Closing."):
        index = next(i for i, row in enumerate(rows) if opener in row)
        assert rows[index - 1].strip() == "" and rows[index - 2].strip(), (opener, rows[index - 3 : index + 1])
    for heading in ("Title", "Section"):
        index = next(i for i, row in enumerate(rows) if heading in row)
        assert index == 0 or (rows[index - 1].strip() == "" and rows[index - 2].strip() == ""), heading
    assert rows[0].strip() and rows[-2].strip()  # no leading or trailing gap of its own


@pytest.mark.parametrize("width", [80, 120])
def test_a_rendered_answer_leaves_no_trailing_whitespace_or_overlong_row(width):
    """Scrollback is permanent: a padded row becomes a wrap artifact the moment the terminal is
    narrowed, and an over-wide row wraps immediately."""
    rows = rendered_answer(MARKDOWN_SAMPLE, width, role="assistant", rule=False)

    assert all(row == row.rstrip() for row in rows), [row for row in rows if row != row.rstrip()]
    assert all(get_cwidth(row) <= width for row in rows)
    assert rows[-1] == "" and rows[-2].strip()  # ends with exactly one newline, not a blank run


def test_repeated_output_never_accumulates_blank_rows():
    """The gap between blocks is stated once through `separate`, so callers asking for room around
    the same seam -- or asking twice -- still leave one blank row."""
    printed = []
    ui = UiPrinter(output_fn=lambda _text: None)
    ui.color = True
    ui._scrollback_print = lambda fragment: printed.append(fragment)

    for _ in range(3):
        ui.separate()
        ui.separate()
        ui.emit("a line")
        ui.separate()

    text = "".join(fragment for part in printed for _, fragment in to_formatted_text(part))
    # Nothing above the first line to part it from, one blank row between the rest, one below.
    assert text == "a line\n\na line\n\na line\n\n"
    assert "\n\n\n" not in text


def test_blank_rows_at_the_end_of_a_block_are_not_doubled_by_the_next_one():
    ui = UiPrinter(output_fn=lambda _text: None)
    ui.color = True
    ui._scrollback_print = lambda _fragment: None

    ui.emit("a line")
    assert ui.trailing_blanks == 0
    ui.emit("")
    assert ui.trailing_blanks == 1
    ui.emit("")
    assert ui.trailing_blanks == 2
    ui.emit("another")
    assert ui.trailing_blanks == 0


def test_a_list_is_tight_until_an_item_wraps():
    """List spacing follows the list. Single-line items read as one object and stay together; once
    an item wraps, the bullets are no longer enough to show where an item ends, so the items are
    parted."""
    tight = rendered_answer("- one\n- two\n- three\n", 92, role="assistant", rule=False)
    assert not any(row.strip() == "" for row in tight[:-1])

    long_item = "an item long enough to wrap past the measure " * 3
    spaced = rendered_answer(f"- {long_item}\n- short\n", 92, role="assistant", rule=False)
    bullets = [index for index, row in enumerate(spaced) if row.lstrip().startswith("• ")]
    assert len(bullets) == 2
    assert spaced[bullets[1] - 1].strip() == ""

