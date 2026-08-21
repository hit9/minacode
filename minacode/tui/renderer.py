"""Primary-screen renderer that owns both the live layout and writes above it."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText, fragment_list_to_text, to_formatted_text
from prompt_toolkit.renderer import Renderer, print_formatted_text


class ScrollbackRenderer(Renderer):
    """Track the live layout's terminal origin and write scrollback above it.

    prompt-toolkit tracks its cursor relative to the layout. The absolute origin is learned from
    CPR and adjusted by the same renders that can scroll the layout upward. Keeping both values in
    this renderer prevents outside printers from guessing a terminal position from layout height.
    """

    _origin_row: int | None = None

    def reset(self, _scroll: bool = False, leave_alternate_screen: bool = True) -> None:
        self._origin_row = None
        super().reset(_scroll=_scroll, leave_alternate_screen=leave_alternate_screen)

    def report_absolute_cursor_row(self, row: int) -> None:
        super().report_absolute_cursor_row(row)
        self._origin_row = row - 1
        self._clamp_origin()

    def render(self, app: Application[Any], layout, is_done: bool = False) -> None:
        super().render(app, layout, is_done=is_done)
        self._clamp_origin()

    def _clamp_origin(self) -> None:
        """Account for a render extending below the terminal and scrolling its origin upward."""
        if self._origin_row is None or self._last_screen is None or self.full_screen:
            return
        self._origin_row = min(self._origin_row, max(0, self.output.get_size().rows - self._last_screen.height))

    @property
    def scrollback_ready(self) -> bool:
        return (
            not self.full_screen
            and self._origin_row is not None
            and self._origin_row > 0
            and self._last_screen is not None
            and self._last_size == self.output.get_size()
        )

    def print_scrollback(self, app: Application[Any], parts: list[Any]) -> bool:
        """Write parts above the layout, returning False when its position is not trustworthy."""
        fragments = [fragment for part in parts for fragment in to_formatted_text(part)]
        text = fragment_list_to_text(fragments)
        if not self.scrollback_ready or not text.endswith("\n"):
            return False
        last = fragments[-1]
        fragments[-1] = (last[0], last[1][:-1]) if len(last) == 2 else (last[0], last[1][:-1], last[2])
        origin = self._origin_row
        assert origin is not None
        restore = self._cursor_pos
        try:
            # `_origin_row` and `_cursor_pos` are zero-based renderer coordinates. DECSTBM and
            # Output.cursor_goto are both one-based terminal coordinates (Vt100_Output sends the
            # arguments verbatim), so the last row above the app is `origin`, while restoring a
            # point inside the app needs +1 on both axes. Do not "normalize" these asymmetrically.
            self.output.write_raw(f"\x1b[1;{origin}r")
            self.output.cursor_goto(origin, 1)
            self.output.write("\r\n")
            print_formatted_text(
                self.output,
                FormattedText(fragments),
                self.style,
                style_transformation=app.style_transformation,
                color_depth=app.color_depth,
            )
        finally:
            self.output.write_raw("\x1b[r")
            self.output.cursor_goto(origin + restore.y + 1, restore.x + 1)
            self.output.flush()
            app._redraw()
        return True


def install_scrollback_renderer(app: Application[Any]) -> ScrollbackRenderer:
    """Replace a not-yet-running application's stock renderer with the terminal owner."""
    renderer = ScrollbackRenderer(
        app.renderer.style,
        app.output,
        full_screen=False,
        mouse_support=False,
        cpr_not_supported_callback=lambda: None,
    )
    app.renderer = renderer
    return renderer
