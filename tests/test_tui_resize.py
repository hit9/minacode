"""Resize re-anchoring: a multiplexer reflow (tmux zoom/unzoom) must not make the prompt climb.

Ending a tmux zoom moves the pane content upward before prompt-toolkit's size poll notices the
resize. The stock handler then erases from the row it remembers and trusts the next cursor
position report, so the drifted answer inflates the renderer's available height: the
application creeps toward the top of the pane and the transcript scrolls out of view. These
tests run the real TuiApp against a terminal model that tracks its cursor, scrolls at the
bottom edge, and answers CPR from wherever the cursor physically is, then repeat the reflow
cycle and require the prompt to stay anchored at the bottom.
"""

from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout.screen import Char, Screen
from tui_harness import ResizableOutput, run_interactive_tui, wait_until

from wizolt.render import UiPrinter
from wizolt.tui.app import TuiApp

ROWS = 30
DRIFT = 3  # rows the pane content travels up on each unzoom
CYCLES = 6
TRANSCRIPT_ROWS = ROWS - 4  # the app's four bottom-anchored rows replace four transcript rows at startup


class ReflowingTerminal(ResizableOutput):
    """A terminal model that answers CPR from its tracked cursor row.

    Implements the output surface prompt-toolkit drives on the primary screen: cursor moves,
    newline writes that scroll once the cursor sits on the last row, erase operations, and a
    cursor position report delivered like a real terminal answers it — from the cursor's
    actual position, including wherever a reflow has just moved it.
    """

    def __init__(self, rows, columns):
        super().__init__(rows=rows, columns=columns)
        self.lines = [f"transcript {index:02d}" for index in range(1, rows + 1)]
        self.row = rows  # the session starts at the bottom of the scrollback
        self.column = 0
        self.report_cursor_row = None

    def get_rows_below_cursor_position(self):
        raise NotImplementedError

    @property
    def responds_to_cpr(self):
        return True

    def ask_for_cpr(self):
        if self.report_cursor_row is not None:
            self.report_cursor_row(self.row)

    def cursor_goto(self, row, column):
        self.row, self.column = row, column

    def cursor_up(self, amount):
        self.row -= amount

    def cursor_down(self, amount):
        self.row += amount

    def cursor_forward(self, amount):
        self.column += amount

    def cursor_backward(self, amount):
        self.column -= amount

    def write(self, data):
        for char in data:
            if char == "\r":
                self.column = 0
            elif char == "\n":
                if self.row >= self.size.rows:
                    del self.lines[0]
                    self.lines.append("")
                else:
                    self.row += 1
            elif char.isprintable():
                line = self.lines[self.row - 1]
                line = line[: self.column].ljust(self.column) + char + line[self.column + 1 :]
                self.lines[self.row - 1] = line
                self.column += 1

    def erase_down(self):
        self.lines[self.row - 1] = self.lines[self.row - 1][: self.column]
        for index in range(self.row, self.size.rows):
            self.lines[index] = ""

    def erase_end_of_line(self):
        self.lines[self.row - 1] = self.lines[self.row - 1][: self.column]

    def reflow_up(self, count):
        """The multiplexer moves the pane content up; the cursor travels with it."""
        del self.lines[:count]
        self.lines.extend([""] * count)
        self.row = max(1, self.row - count)


def test_resize_reflow_keeps_prompt_anchored_at_bottom(monkeypatch):
    output = ReflowingTerminal(ROWS, 80)
    app = TuiApp()
    prompt_rows = []
    transcript_rows = []

    def drive(_pipe_input):
        wait_until(lambda: any(line.startswith(UiPrinter.PROMPT_PREFIX) for line in output.lines))

        def run_resize_cycle():
            finished = []
            app.app.loop.call_soon_threadsafe(lambda: (app.app._on_resize(), finished.append(True)))
            wait_until(lambda: finished)

        for cycle in range(CYCLES):
            output.reflow_up(DRIFT)
            output.size = Size(rows=ROWS, columns=50 if cycle % 2 else 100)
            run_resize_cycle()
            prompt_rows.append(tuple(index + 1 for index, line in enumerate(output.lines) if line.startswith(UiPrinter.PROMPT_PREFIX)))
            transcript_rows.append(sum(1 for line in output.lines if line.startswith("transcript")))
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(
        monkeypatch,
        app,
        drive=drive,
        output=output,
        # Resolve the renderer method on every response: the resize handler replaces it with its
        # own witness for the reflowed cursor position, and a bound copy captured here would
        # bypass that and answer from the pre-replace method instead.
        on_application=lambda app: setattr(output, "report_cursor_row", lambda row: app.renderer.report_absolute_cursor_row(row)),
    )

    # The prompt never moves, and only one copy of it is ever visible: every cycle redraws at
    # the same bottom-anchored row and erases the reflowed copy instead of leaving ghosts.
    assert len(set(prompt_rows)) == 1
    assert all(len(rows) == 1 for rows in prompt_rows)
    assert prompt_rows[0][0] >= ROWS - 6
    # Each cycle costs exactly the drifted rows of transcript — the resize itself must not
    # erase any transcript above the app nor scroll it further.
    assert transcript_rows == [TRANSCRIPT_ROWS - DRIFT * (cycle + 1) for cycle in range(CYCLES)]


def _screen_with_rows(widths, styled_spaces=()):
    """A rendered screen whose rows each hold `width` columns of plain text."""
    screen = Screen()
    styled_row = max((row for row, _ in styled_spaces), default=-1)
    screen.height = max(len(widths), styled_row + 1)
    for row, width in enumerate(widths):
        for column in range(width):
            screen.data_buffer[row][column] = Char("x")
    for row, width in styled_spaces:
        for column in range(width):
            screen.data_buffer[row][column] = Char(" ", "class:content")
    return screen


def test_reflowed_height_counts_rows_after_a_rewrap():
    reflowed = TuiApp._reflowed_height
    # A blank drawn row still occupies one terminal row.
    assert reflowed(_screen_with_rows([0, 0]), 40) == 2
    # Rows split into ceil(width / columns) wrapped rows.
    assert reflowed(_screen_with_rows([40, 41, 79, 80, 81]), 40) == 1 + 2 + 2 + 2 + 3
    # Rows that already fit keep their count when the pane widens.
    assert reflowed(_screen_with_rows([40, 100]), 100) == 2
    # Trailing unstyled spaces are not content a reflow preserves; styled spaces are.
    assert reflowed(_screen_with_rows([], styled_spaces=((0, 50),)), 40) == 2


class DeferredCprTerminal(ReflowingTerminal):
    """Answers a cursor position report only when the test delivers it, the way a real terminal
    answers after the resize handler has already erased and repainted."""

    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self.queued_rows = []
        self.defer = False

    def ask_for_cpr(self):
        if self.report_cursor_row is None:
            return
        if self.defer:
            self.queued_rows.append(self.row)
        else:
            self.report_cursor_row(self.row)


def test_resize_reflow_delayed_cpr_answer_completes_the_erase(monkeypatch):
    """The resize repaint runs before the terminal answers CPR; the answer then erases the part
    of the reflowed copy the repaint left above the freshly anchored app. The end state must
    match the synchronous-terminal guarantees: one bottom-anchored prompt, no ghost copy, and
    exactly the drifted rows of transcript lost."""
    monkeypatch.setattr(TuiApp, "REANCHOR_CPR_TIMEOUT", 5.0)  # the test delivers the answer itself
    output = DeferredCprTerminal(ROWS, 80)
    app = TuiApp()
    prompt_rows = []
    transcript_rows = []

    def drive(_pipe_input):
        wait_until(lambda: any(line.startswith(UiPrinter.PROMPT_PREFIX) for line in output.lines))
        output.defer = True

        for cycle in range(CYCLES):
            output.reflow_up(DRIFT)
            output.size = Size(rows=ROWS, columns=50 if cycle % 2 else 100)
            finished = []
            app.app.loop.call_soon_threadsafe(lambda done=finished: (app.app._on_resize(), done.append(True)))
            wait_until(lambda done=finished: done)
            # The handler asked where the cursor sat before it erased; answer from that record.
            assert output.queued_rows, "resize must request a cursor position report"
            row = output.queued_rows.pop()
            answered = []
            app.app.loop.call_soon_threadsafe(lambda r=row, done=answered: (output.report_cursor_row(r), done.append(True)))
            wait_until(lambda done=answered: done)
            prompt_rows.append(tuple(index + 1 for index, line in enumerate(output.lines) if line.startswith(UiPrinter.PROMPT_PREFIX)))
            transcript_rows.append(sum(1 for line in output.lines if line.startswith("transcript")))
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(
        monkeypatch,
        app,
        drive=drive,
        output=output,
        on_application=lambda app: setattr(output, "report_cursor_row", lambda row: app.renderer.report_absolute_cursor_row(row)),
    )

    assert len(set(prompt_rows)) == 1
    assert all(len(rows) == 1 for rows in prompt_rows)
    assert prompt_rows[0][0] >= ROWS - 6
    assert transcript_rows == [TRANSCRIPT_ROWS - DRIFT * (cycle + 1) for cycle in range(CYCLES)]


class NoCprTerminal(ReflowingTerminal):
    """A terminal that never advertises cursor position reports."""

    @property
    def responds_to_cpr(self):
        return False


def test_resize_reflow_without_cpr_still_anchors_and_leaves_no_ghost(monkeypatch):
    output = NoCprTerminal(ROWS, 80)
    app = TuiApp()
    prompt_rows = []

    def drive(_pipe_input):
        wait_until(lambda: any(line.startswith(UiPrinter.PROMPT_PREFIX) for line in output.lines))
        for cycle in range(CYCLES):
            output.reflow_up(DRIFT)
            output.size = Size(rows=ROWS, columns=50 if cycle % 2 else 100)
            finished = []
            app.app.loop.call_soon_threadsafe(lambda done=finished: (app.app._on_resize(), done.append(True)))
            wait_until(lambda done=finished: done)
            prompt_rows.append(tuple(index + 1 for index, line in enumerate(output.lines) if line.startswith(UiPrinter.PROMPT_PREFIX)))
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(
        monkeypatch,
        app,
        drive=drive,
        output=output,
        on_application=lambda app: setattr(output, "report_cursor_row", lambda row: app.renderer.report_absolute_cursor_row(row)),
    )

    assert len(set(prompt_rows)) == 1
    assert all(len(rows) == 1 for rows in prompt_rows)
    assert prompt_rows[0][0] >= ROWS - 6


def test_write_to_scrollback_park_keeps_repaint_at_content_height(monkeypatch):
    """A promoted-output write parks the cursor at the pane bottom before the repaint, so the
    cursor position report that follows describes the bottom row and the app repaints at its
    content height instead of claiming every row below the transcript."""
    output = ReflowingTerminal(ROWS, 80)
    app = TuiApp()

    def drive(_pipe_input):
        wait_until(lambda: any(line.startswith(UiPrinter.PROMPT_PREFIX) for line in output.lines))
        renderer = app.app.renderer
        # A mid-pane cursor (e.g. a transcript print that did not park) would claim every row
        # below it; balloon the claim and verify the write's own repaint shrinks it back.
        renderer._min_available_height = ROWS
        app.write_to_scrollback(lambda: app.app.output.write("promoted line\n"))
        wait_until(lambda: renderer.last_rendered_screen is not None)
        # The parked cursor answered the CPR from the pane bottom: one row lies below it.
        assert renderer._min_available_height == 1
        preferred = app.app.layout.container.preferred_height(output.size.columns, output.size.rows).preferred
        assert renderer.last_rendered_screen.height <= max(preferred, 1)
        assert any(line.startswith("promoted line") for line in output.lines)
        app.app.loop.call_soon_threadsafe(app.app.exit)

    run_interactive_tui(
        monkeypatch,
        app,
        drive=drive,
        output=output,
        on_application=lambda app: setattr(output, "report_cursor_row", lambda row: app.renderer.report_absolute_cursor_row(row)),
    )
