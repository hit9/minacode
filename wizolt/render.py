"""wizolt terminal rendering, live output, and status display."""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from prompt_toolkit import print_formatted_text
from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.formatted_text import ANSI, FormattedText, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.output import create_output
from prompt_toolkit.utils import get_cwidth
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown, MarkdownElement, TableElement
from rich.padding import Padding
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text as RichText
from rich.theme import Theme as RichTheme

from wizolt.base import (
    MEMORY_PREFIXES,
    MODEL_REQUEST_RETRIES,
    Json,
    LogBlock,
    LogEdge,
    LogRole,
    Text,
)
from wizolt.session import Session
from wizolt.tools import CodeIndex

if TYPE_CHECKING:
    from pygments.style import Style as PygmentsStyle

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.styles import get_style_by_name
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = Token = None
    get_lexer_by_name = get_lexer_for_filename = get_style_by_name = None


def progress_bar(value: int, total: int, width: int = 14) -> str:
    """A fixed-width meter in eighth-block characters, clamped to [0, total]."""
    ratio = min(1.0, max(0.0, value / total)) if total else 0.0
    eighths = int(ratio * width * 8 + 0.5)
    full, partial = divmod(eighths, 8)
    partials = "▏▎▍▌▋▊▉"
    return "[" + "█" * full + (partials[partial - 1] if partial else "") + "░" * (width - full - bool(partial)) + "]"


def markdown_table(headers: list[str], rows: list[tuple]) -> str:
    def cell(value: object) -> str:
        return Text.clean(str(value)).replace("\n", " ").replace("|", "\\|")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
        ]
    )


MAX_RENDERED_SOURCES = 10


def search_sources_footer(sources: list[Json]) -> str:
    """A markdown source list for the provider-side searches a turn performed, or "" for none.

    This is presentation only. The sources stay on the messages that carry them, so the answer
    reaching history is exactly what the model wrote, and nothing new replays to the provider on
    the next turn."""
    seen = dict.fromkeys(url for source in sources if isinstance(source, dict) and (url := str(source.get("url") or "")))
    if not seen:
        return ""
    shown = list(seen)[:MAX_RENDERED_SOURCES]
    # Strip scheme and trailing slash for a compact one-line display.
    lines = [f"{index}. {url.split('://', 1)[-1].rstrip('/')}" for index, url in enumerate(shown, start=1)]
    if len(seen) > len(shown):
        lines.append(f"…and {len(seen) - len(shown)} more")
    return "\n".join(["", "**Sources**", "", *lines])


TERMINAL_DEFAULT = "default"


@dataclass(frozen=True)
class ThemePalette:
    """One appearance's colors, named by what they mean rather than by where they are drawn.

    This is the single owner of color in the UI. A component asks for the meaning it is expressing
    (`muted`, `error`, `status_provider`) and an adapter turns that into Rich, prompt-toolkit, or
    Pygments syntax; no component picks a color of its own, and no framework spelling appears here.

    Every value is a normalized `#rrggbb` string or the sentinel `default`, the terminal's own
    foreground. Body text takes `default` on purpose: the reader already chose a comfortable text
    color for their terminal, and the palette's job is to place everything else around it.

    Layout is deliberately absent. Spacing, indentation, and rules are decisions about structure,
    not about color, and they belong to the renderers.
    """

    appearance: Literal["light", "dark"]

    # Text hierarchy: body, supporting detail, and the near-invisible tones separators take.
    text: str
    muted: str
    subtle: str
    # One or two accents carry attention. Everything else is state (`success`/`warning`/`error`)
    # or identity (`user`, `tool`).
    accent: str
    accent_secondary: str
    user: str
    tool: str
    success: str
    warning: str
    error: str
    rule: str
    selection_fg: str
    selection_bg: str

    # Lightweight highlighting for tool arguments, which never reach a Pygments lexer.
    syntax_assign: str
    syntax_string: str
    syntax_number: str
    syntax_ident: str
    syntax_builtin: str
    # The Pygments style's own body color. A token painted in it is painted in no color at all, so
    # code inherits the terminal's foreground instead of a near-match the theme cannot control.
    syntax_default: str

    # The status row reads as a quiet footer, so its plain tone stays below full-strength text and
    # each field keeps just enough color to be told apart at a glance.
    status_base: str
    status_provider: str
    status_reason: str
    status_mcp: str
    status_context: str
    status_index: str
    status_yolo: str
    status_worker: str

    pygments_style: str

    def color(self, role: str) -> str:
        if role not in THEME_ROLES:
            raise KeyError(f"unknown theme role: {role}")
        return getattr(self, role)


# Every semantic role, in declaration order. `appearance` and `pygments_style` are not colors and
# are excluded, so an adapter can walk this list and know each entry renders as one.
THEME_ROLES: tuple[str, ...] = tuple(field.name for field in fields(ThemePalette) if field.name not in {"appearance", "pygments_style"})


# Two appearances, one shape. The tuning goals are the same in both: body text is the terminal's
# own, supporting detail is clearly weaker than body text without the interface going grey, one
# accent carries attention (with a second for the model's own voice), and state colors — success,
# warning, error — are told apart by hue rather than by shouting.
DARK_PALETTE = ThemePalette(
    appearance="dark",
    text=TERMINAL_DEFAULT,
    # The greys are warm. A cool grey beside warm text is what makes a terminal read as washed out,
    # and supporting detail is most of what is on screen.
    muted="#a49484",
    # Structure, not text: rails, gutters, bullets. Below muted, above the background.
    subtle="#6b6156",
    # Teal, not sky blue, and worn by few things: the prompt, field labels, inline code. One cool
    # note against warm text is contrast; a screen of blue is a scheme with nothing to say.
    accent="#5fb3a1",
    # The model's own voice: interim narration, thinking, the working divider.
    accent_secondary="#cf8fa6",
    user="#e0a96d",
    tool="#98c379",
    success="#98c379",
    warning="#e0a34a",
    error="#e8767c",
    # Visible enough to read as a boundary: a rule that fades into the background stops parting
    # anything, and the transcript loses its sense of blocks. Neutral and warm, never colored.
    rule="#7a6f63",
    # A warm band rather than the accent: a selected row is a large area of color, and an accent
    # at that size takes over the screen.
    selection_fg="#f2ece4",
    selection_bg="#3a332d",
    syntax_assign="#5fb3a1",
    syntax_string="#d9b48c",
    syntax_number="#cf8fa6",
    syntax_ident="#c8bfb3",
    syntax_builtin="#5fb3a1",
    # Must stay the Pygments style's own body color: a token painted in it is painted in no color
    # at all, so code inherits the terminal's foreground instead of a near-match.
    syntax_default="#dddddd",
    # The status row is a footer: its plain tone stays below full strength, and each field carries
    # just enough color to be picked out without competing with the transcript above it.
    status_base="#cdc3b6",
    status_provider="#5fb3a1",
    status_reason="#cf8fa6",
    status_mcp="#8fc9bb",
    status_context="#e0a34a",
    status_index="#a49484",
    status_yolo="#d987b4",
    status_worker="#e0b45f",
    # Warm highlighting, so a code block does not reintroduce the blues the rest of the scheme
    # spends its effort avoiding.
    pygments_style="gruvbox-dark",
)

LIGHT_PALETTE = ThemePalette(
    appearance="light",
    text=TERMINAL_DEFAULT,
    muted="#6f6257",
    subtle="#a89e93",
    accent="#116b60",
    accent_secondary="#9c4f6c",
    user="#9a5b2e",
    tool="#3f7a2e",
    success="#3f7a2e",
    warning="#a2620a",
    error="#b3261e",
    # Lighter than the text it parts, but still a line the eye lands on.
    rule="#a89e93",
    selection_fg="#2b2019",
    selection_bg="#ecdfcd",
    syntax_assign="#116b60",
    syntax_string="#7a5c2e",
    syntax_number="#9c4f6c",
    syntax_ident="#4a4038",
    syntax_builtin="#116b60",
    syntax_default="#3c3836",
    status_base="#54493f",
    status_provider="#116b60",
    status_reason="#9c4f6c",
    status_mcp="#1f6b5e",
    status_context="#a2620a",
    status_index="#635649",
    status_yolo="#8a3f6a",
    status_worker="#a2620a",
    pygments_style="gruvbox-light",
)


class Theme:
    """The active palette, and the adapters that speak it to Rich, prompt-toolkit, and Pygments.

    The palettes answer "what color is this meaning"; the adapters answer "how does this framework
    spell a color". Nothing else in the codebase does either.
    """

    # Diff colors are pinned, not derived. They were tuned against real diffs in both appearances
    # and a palette reshuffle must never move them, so they stay their own fixed mapping in the
    # frameworks' own spelling — the one place the palette deliberately does not own.
    DIFF_DARK: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#003b00",
        "diff.added.fg": "fg:default",
        "diff.removed.bg": "bg:#520000",
        "diff.removed.fg": "fg:default",
    }
    DIFF_LIGHT: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#d1f0d1",
        "diff.added.fg": "fg:#003b00",
        "diff.removed.bg": "bg:#f5c8c8",
        "diff.removed.fg": "fg:#520000",
    }

    _mode: ClassVar[str] = "dark"
    _pygments_cache: ClassVar[dict[str, type[PygmentsStyle] | None]] = {}

    @classmethod
    def set_mode(cls, mode: str) -> None:
        cls._mode = "light" if mode == "light" else "dark"

    @classmethod
    def palette(cls) -> ThemePalette:
        return LIGHT_PALETTE if cls._mode == "light" else DARK_PALETTE

    @classmethod
    def color(cls, role: str) -> str:
        """The active palette's color for one semantic role, as `#rrggbb` or `default`."""
        return cls.palette().color(role)

    @classmethod
    def diff_style(cls, key: str) -> str:
        return (cls.DIFF_LIGHT if cls._mode == "light" else cls.DIFF_DARK)[key]

    @classmethod
    def fg(cls, role: str, *attributes: str) -> str:
        """One role as a prompt-toolkit inline style, e.g. `fg:#8b949e bold`.

        For fragments built outside the style map — wrapped rows, ramps, anything assembled from a
        computed color. Fragments that can name a class should use one instead.
        """
        return " ".join((f"fg:{cls.color(role)}", *attributes))

    @classmethod
    def tui_class(cls, role: str) -> str:
        """The prompt-toolkit class name carrying one role, e.g. `class:status.provider`."""
        return "class:" + role.replace("_", ".")

    @classmethod
    def tui_styles(cls) -> dict[str, str]:
        """Every role as a prompt-toolkit class, for the view to compose its own map on top of.

        Roles are the base vocabulary, so a view class that is simply "this role" can point at it
        (`"choice.tool": "class:tool"`) and the light/dark difference stays here.
        """
        return {role.replace("_", "."): f"fg:{cls.color(role)}" for role in THEME_ROLES}

    @classmethod
    def rich_theme(cls) -> RichTheme:
        """Every role as a Rich style named `wizolt.<role>`, plus the Markdown element styles.

        Rich resolves an unknown style name by raising, so a console that renders our markup must
        be built with this theme; `markdown_console` is the only place that happens.
        """
        styles = {f"wizolt.{role.replace('_', '.')}": cls.color(role) for role in THEME_ROLES}
        # Rich resolves a style string either as a theme name or as attributes, never as both, so
        # anything wearing a weight on top of a role is named here rather than spelled at the call.
        styles["wizolt.role.user"] = f"bold {cls.color('accent')}"
        styles["wizolt.role.assistant"] = f"bold {cls.color('accent_secondary')}"
        return RichTheme({**styles, **cls.markdown_styles()}, inherit=True)

    @classmethod
    def markdown_styles(cls) -> dict[str, str]:
        """Rich's Markdown element styles, restated in the palette.

        Rich's own defaults are a second color scheme: cyan inline code on a black band, magenta
        headings and block quotes, blue links, a cyan table. Left alone they win every argument with
        the palette and turn an answer into a page of blue and cyan.

        The replacements say hierarchy with weight and tone instead of with color. Headings step
        down in emphasis rather than each taking a hue; body text, list text, and table cells stay
        the terminal's own foreground; inline code is the one accent prose carries; a link is
        underlined rather than colored, with its URL beside it in the supporting tone.
        """
        muted, subtle, accent = cls.color("muted"), cls.color("subtle"), cls.color("accent")
        return {
            "markdown.h1": "bold",
            "markdown.h2": f"bold {accent}",
            "markdown.h3": "bold",
            "markdown.h4": f"bold {muted}",
            "markdown.h5": f"italic {muted}",
            "markdown.h6": muted,
            "markdown.h7": f"italic {muted}",
            "markdown.paragraph": "none",
            "markdown.text": "none",
            "markdown.item": "none",
            # No background band: the block is read as code by its highlighting, and a filled band
            # is the single loudest thing a theme can put on a terminal.
            "markdown.code": accent,
            "markdown.code_block": "none",
            "markdown.block_quote": muted,
            "markdown.hr": cls.color("rule"),
            "markdown.item.bullet": f"bold {subtle}",
            "markdown.item.number": subtle,
            "markdown.list": "none",
            "markdown.link": "underline",
            "markdown.link_url": muted,
            "markdown.table.border": subtle,
            "markdown.table.header": "bold",
        }

    @classmethod
    def ramp(cls, start_role: str, end_role: str, steps: int) -> list[str]:
        """Interpolate `steps` hex colors from one role to another.

        Used for gradients that need more shades than the palette names, so a moving highlight can
        fade between two cells instead of snapping from one named color to the next.
        """
        start, end = cls.rgb(cls.color(start_role)), cls.rgb(cls.color(end_role))
        span = max(1, steps - 1)
        return [cls.mix(start, end, index / span) for index in range(steps)]

    @staticmethod
    def mix(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
        return "#" + "".join(f"{round(channel + (channel_end - channel) * ratio):02x}" for channel, channel_end in zip(start, end, strict=True))

    @staticmethod
    def rgb(color: str) -> tuple[int, int, int]:
        value = color.rpartition(":")[2].lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    @classmethod
    def detect(cls) -> str:
        # COLORFGBG is "fg;bg" (rxvt/urxvt/Konsole) or "fg;;bg" (iTerm2). Only the standard
        # white entries are reliably light; index 8 is bright black and must remain dark.
        fgbg = os.environ.get("COLORFGBG", "")
        if ";" in fgbg:
            with contextlib.suppress(ValueError):
                bg = int(fgbg.rsplit(";", 1)[1])
                return "light" if bg in {7, 15} else "dark"
        return "dark"

    @classmethod
    def resolve(cls, configured: str) -> str:
        configured = (configured or "auto").strip().lower()
        return configured if configured in ("light", "dark") else cls.detect()

    @classmethod
    def pygments_style(cls) -> type[PygmentsStyle] | None:
        if pygments is None or get_style_by_name is None:
            return None
        name = cls.palette().pygments_style
        if name not in cls._pygments_cache:
            try:
                cls._pygments_cache[name] = get_style_by_name(name)
            except Exception:  # noqa: BLE001 - optional Pygments styles must degrade to plain rendering.
                cls._pygments_cache[name] = None
        return cls._pygments_cache[name]


class _Heading(Heading):
    """Headings that read as headings, not as banners.

    Rich centers `h1`, which in a transcript reads as a title page rather than as the top of a
    section: the text drifts away from the left edge every other line is on. Every level is left
    aligned here, and the levels differ by weight and tone (see `Theme.markdown_styles`).
    """

    LEVEL_ALIGN: ClassVar[dict[str, Any]] = dict.fromkeys(Heading.LEVEL_ALIGN, "left")


class _CodeBlock(CodeBlock):
    """A fenced block highlighted on the terminal's own background.

    Rich pads the block and fills it with the Pygments theme's background, which on a terminal that
    is not exactly that color reads as a misplaced rectangle. Dropping the fill and the padding
    leaves the highlighting, which is what the fence was for, and matches how the transcript
    renders the code it prints itself.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console, options
        yield Syntax(str(self.text).rstrip(), self.lexer_name, theme=self.theme, word_wrap=True, padding=0, background_color="default")


class _Table(TableElement):
    """A table with no outer edge, so it does not arrive wrapped in blank rows.

    Rich's `show_edge` draws an empty row above and below the table; with the blank line Markdown
    already puts between blocks, a table ends up floating two rows away from its own paragraph.
    The header underline is enough to bound it.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console, options
        table = Table(box=box.SIMPLE, pad_edge=False, style="markdown.table.border", show_edge=False, collapse_padding=True)
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading)
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*[element.content for element in row.cells])
        yield table


class WizoltMarkdown(Markdown):
    """The one Markdown renderable in the app, so every surface lays a document out the same way.

    Hyperlinks stay off: prompt-toolkit cannot parse the OSC 8 escapes Rich emits for them, and
    `UiPrinter` strips those before they reach the terminal, so a hyperlink would only cost the URL.
    """

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _Heading,
        "fence": _CodeBlock,
        "code_block": _CodeBlock,
        "table_open": _Table,
    }

    def __init__(self, markup: str) -> None:
        # The theme's own Pygments style, so a fenced block is colored like the code the transcript
        # prints. `Theme.pygments_style` returning None means the name did not load; Rich falls back
        # to plain ANSI rather than raising on it.
        style = Theme.palette().pygments_style if Theme.pygments_style() is not None else "ansi_dark"
        super().__init__(markup, code_theme=style, hyperlinks=False)


def markdown_console(width: int) -> Console:
    """A Rich console for the capture-then-emit path, carrying the palette's named styles.

    Every Rich render in the app goes through one of these. The console is always a capture target,
    never a direct writer: printing Rich output while the prompt-toolkit application is live
    interleaves raw escapes with its renderer, so callers capture and emit the result as ANSI.

    Colors come from the theme, so `wizolt.*` style names resolve here and nowhere else.
    """
    return Console(force_terminal=True, color_system="truecolor", no_color=False, width=max(10, width), theme=Theme.rich_theme())


class UiPrinter:
    """Render completed output into native terminal scrollback.

    The durable half of the terminal boundary: what it prints survives the session and stays
    searchable with the terminal's own tools, so nothing here clears the screen. Live previews and
    status belong to the prompt-toolkit application instead.

    Because the output is permanent it is sanitized rather than passed through. Rich pads every line
    to the console width, which bakes trailing whitespace into scrollback and becomes wrap artifacts
    when the terminal is later narrowed, so padding is stripped unless it carries a background color
    and is part of a visible band. Terminal control strings prompt-toolkit cannot parse are stripped
    up front, since it drops their framing but leaks the payload as visible garbage.

    Color is decided once, from whether output is a real terminal.
    """

    MESSAGE_ROLE_STYLES: ClassVar[dict[str, str]] = {"user": "wizolt.role.user", "assistant": "wizolt.role.assistant"}
    PROMPT_PREFIX: ClassVar[str] = "> "
    USER_LOG_PREFIX: ClassVar[str] = "• "
    # How long scrollback emits wait before printing as one batch. Each print suspends the live
    # application (erase + repaint), so batching a burst of tool-result lines into one suspend
    # keeps the animated divider on screen; 30ms is well below human perception.
    SCROLLBACK_BATCH_WINDOW: ClassVar[float] = 0.03
    MCP_STATUS_RE: ClassVar[re.Pattern[str]] = re.compile(r"● (connected|connecting|disconnected|disconnecting|error|skipped)")
    MCP_STATUS_ANSI: ClassVar[dict[str, str]] = {
        "connected": "\x1b[32m",
        "connecting": "\x1b[32m",
        "disconnected": "\x1b[33m",
        "disconnecting": "\x1b[33m",
        "error": "\x1b[31m",
        "skipped": "\x1b[90m",
    }

    @classmethod
    def user_log_style(cls) -> str:
        return Theme.fg("user")

    TOOL_ARG_TOKEN: ClassVar[re.Pattern] = re.compile(
        r"""\s+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_][\w.-]*=|(?:tr|job)\.\d+|\d+(?::\d+)?|[;,]|[^\s;,]+"""
    )

    def __init__(self, output_fn=print):
        self.output_fn = output_fn
        self.color = output_fn is print and sys.stdout.isatty()
        # Batch mode: while active, every emit appends its styled fragments instead of printing,
        # so a burst of output (the restored-transcript replay) is printed by a single
        # print_formatted_text call and flushes once. Under the TUI each call coordinates with the
        # renderer, so batching turns a hundred of those into one. Only the colored path batches.
        self._batch_parts: list[FormattedText | ANSI] | None = None
        # Scrollback window: while a prompt_toolkit application is live, each print_formatted_text
        # suspends it (erase the whole rendered output, print above it, repaint), which makes the
        # animated divider visibly blink on every emit. A short window batches a burst of emits
        # (a tool result's lines) into one suspend; outside a live application nothing changes.
        self._scrollback_parts: list[FormattedText | ANSI] = []
        self._scrollback_scheduled = False
        self._scrollback_generation = 0
        # Parts/scheduling state are touched from the app loop and late synchronous fallback
        # callers during shutdown, so all access goes through this lock.
        self._scrollback_lock = threading.Lock()
        # Rendered rows since the last full-width rule was drawn. Read by the loop to decide
        # whether a new rule would land too close to the one above it to be worth drawing.
        self.rows_since_rule = 0
        # Blank rows currently sitting at the end of what has been printed. This is what makes the
        # gaps between blocks stable: a caller says "part this from what came before" through
        # `separate`, and saying it twice, or saying it under a rule that already left a gap, still
        # leaves one blank row. Starts at one, so the first block of a session does not open with a
        # blank row it has nothing to be parted from.
        self.trailing_blanks = 1

    def track_layout(self, text: str) -> None:
        """Record what one emit left on screen: rows drawn, and blank rows left at the end.

        Everything printed goes through here, so the two spacing questions -- how far the last rule
        is, and whether a gap is already there -- are answered from what actually reached the
        terminal rather than from each caller's guess about it.
        """
        self.rows_since_rule += text.count("\n")
        rows = text.split("\n")
        if rows and rows[-1] == "":
            rows.pop()  # the newline that ended the last row, not a row of its own
        blanks = 0
        for row in reversed(rows):
            if self.SGR_RE.sub("", row).strip():
                break
            blanks += 1
        # A wholly blank emit extends the run above it; anything with content restarts the count.
        self.trailing_blanks = self.trailing_blanks + blanks if blanks == len(rows) else blanks

    def separate(self, rows: int = 1) -> None:
        """Ensure `rows` blank rows part what comes next from what is already on screen.

        The one way a gap is opened. Callers state the separation they want rather than counting
        newlines, so a block cannot arrive glued to the one above it, and two callers each asking
        for room cannot stack two blank rows between the same pair of blocks.

        Uncolored output is left alone: it is a transcript for another program to read, and its
        spacing is the caller's exact text.
        """
        if not self.color:
            return
        for _ in range(max(0, rows - self.trailing_blanks)):
            self.emit()

    def write_direct(self, callback: Callable[[], None]) -> None:
        """Run one write with no application to print above, draining anything queued first.

        The fallback for a write the ordered queue can no longer accept -- the runtime is unwinding
        and the application may already have stopped. Ordering still holds: whatever was batched
        goes out before this does."""

        self.drain_scrollback()
        callback()

    def drain_scrollback(self) -> None:
        """Synchronously print anything still queued in the batching window.

        Called before any direct print (so a later line never lands ahead of queued ones) and
        at application shutdown (so a turn's last lines are not lost to an un-fired timer).
        """
        with self._scrollback_lock:
            parts = self._scrollback_parts
            self._scrollback_parts = []
            self._scrollback_scheduled = False
            self._scrollback_generation += 1
        if parts:
            print_formatted_text(*parts, sep="", end="", flush=True)

    def _scrollback_print(self, fragment: FormattedText | ANSI) -> None:
        """Print above a live application, batching a burst of emits into one suspend.

        Outside a live application this drains any queued batch first and then prints
        immediately, exactly as before, so headless and pre-TUI output is unchanged and
        nothing queued is reordered or lost.

        The batching path depends on the live application being reachable from this thread:
        TuiApp.run never wraps create_app_session, so the app registers on the shared default
        AppSession (prompt_toolkit's `_current_app_session` ContextVar default) and every
        thread's get_app_or_none() sees it. Should a future caller wrap app.run in a private
        create_app_session instead, this degrades gracefully to per-emit direct prints -- the
        pre-batch behavior, correct but with the divider blink back.
        """
        app = get_app_or_none()
        # `_running_in_terminal` has no public accessor; it is the only way to see "a suspend is
        # already in progress" so a nested emit prints inside it instead of rejoining the window
        # (which would delay it past the suspend's wait-then-return contract).
        if app is None or not app.is_running or app._running_in_terminal:
            self.drain_scrollback()
            print_formatted_text(fragment, end="", flush=True)
            return
        loop = app.loop
        assert loop is not None  # a running application always has one; the checker cannot see it
        self._queue_scrollback(app, fragment)

    def _queue_scrollback(self, app: Any, fragment: FormattedText | ANSI) -> None:
        with self._scrollback_lock:
            self._scrollback_parts.append(fragment)
            if self._scrollback_scheduled:
                return
            self._scrollback_scheduled = True
            generation = self._scrollback_generation
        app.loop.call_soon_threadsafe(self._schedule_scrollback, app, generation)

    def _schedule_scrollback(self, app: Any, generation: int) -> None:
        with self._scrollback_lock:
            if not self._scrollback_scheduled or generation != self._scrollback_generation:
                return
        app.loop.call_later(self.SCROLLBACK_BATCH_WINDOW, self._flush_scrollback, generation)

    def _flush_scrollback(self, generation: int | None = None) -> None:
        with self._scrollback_lock:
            if generation is not None and generation != self._scrollback_generation:
                return
            self._scrollback_scheduled = False
            parts = self._scrollback_parts
            self._scrollback_parts = []
        if parts:
            print_formatted_text(*parts, sep="", end="", flush=True)

    @contextlib.contextmanager
    def batched(self):
        """Collect every emit into one scrollback flush instead of one suspend per emit."""
        if not self.color or self._batch_parts is not None:
            yield
            return
        self._batch_parts = []
        try:
            yield
        finally:
            parts = self._batch_parts
            self._batch_parts = None
            if parts:
                self._flush_batch(parts)

    def _flush_batch(self, parts: list[FormattedText | ANSI]) -> None:
        """Flush a collected batch through the scrollback path, keeping the batch a single print.

        Inside a live application the parts join the same queue as ordinary emits -- landing after
        anything already queued and flushing as one print; outside one they print directly, in
        order, as one call."""
        app = get_app_or_none()
        if app is None or not app.is_running or app._running_in_terminal:
            self.drain_scrollback()
            print_formatted_text(*parts, sep="", end="", flush=True)
            return
        loop = app.loop
        assert loop is not None  # a running application always has one; the checker cannot see it
        self._queue_scrollback(app, FormattedText([segment for part in parts for segment in to_formatted_text(part)]))

    def emit(self, text: str | LogBlock = "", indent: int = 0) -> None:
        """Print one line or log block. `indent` moves plain text into a column; a LogBlock is
        left alone, since it already carries its own margins and rails.

        The margin is applied after styling, not before: `segments` dispatches on what the text
        starts with (`• `, `Error:`, `+ `), so prefixing spaces first would strip every line of
        its color."""
        if isinstance(text, LogBlock):
            indent = 0
        if not self.color:
            self.output_fn(self.indent_message(str(text), "", indent) if indent else str(text))
            return
        segments = self.log_segments(text) if isinstance(text, LogBlock) else self.segments(text)
        if indent:
            segments = self.indent_segments(segments, LogBlock.margin(indent))
        # Counted in rendered rows, not in blocks or calls: what decides whether two rules sit too
        # close is how far apart they are on screen, and one Bash call with its output goes further
        # than four Reads. A block that wrapped counts the rows it actually took.
        self.track_layout("".join(fragment for _, fragment in segments))
        if self._batch_parts is not None:
            self._batch_parts.append(FormattedText(segments))
            return
        self._scrollback_print(FormattedText(segments))

    @staticmethod
    def indent_segments(segments: list[tuple[str, str]], margin: str) -> list[tuple[str, str]]:
        """Open every rendered line with `margin`, carrying the style of the fragment it opens.

        A trailing newline ends the last line rather than starting another, so the margin that
        would follow it is dropped; otherwise an empty emit would print a row of spaces."""
        indented: list[tuple[str, str]] = []
        at_line_start = True
        for style, value in segments:
            for index, piece in enumerate(value.split("\n")):
                if index:
                    indented.append((style, "\n"))
                    at_line_start = True
                if at_line_start and piece:
                    indented.append((style, margin))
                    at_line_start = False
                if piece:
                    indented.append((style, piece))
        return indented

    # Rich right-pads every rendered line with spaces up to the console width so backgrounds and
    # padding can fill the row. Uncolored padding gets baked into scrollback and turns into wrap
    # zigzags on a narrower terminal, so we strip it — but padding that carries a background color
    # (syntax-highlighted code blocks, /diff previews) must be preserved so the block still reads
    # as a solid band. We track the SGR bg state per token and only strip whitespace rendered with
    # bg off.
    SGR_RE: ClassVar[re.Pattern[str]] = re.compile(r"\x1b\[([0-9;]*)m")
    RECORD_TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"(?:tr|job)\.\d+|\d+(?::\d+)?")
    # OSC / APC / DCS / SOS / PM sequences are terminal control strings that prompt_toolkit's ANSI
    # parser doesn't recognize. When they slip through Rich's output (OSC 8 hyperlinks were the
    # historical culprit, iTerm image escapes / Kitty graphics / shell-integration marks are
    # potential future ones), pt eats the ESC framing but leaks the payload as visible garbage
    # (e.g. `8;id=…;https://…;;` for OSC 8). Strip these up front so pt only ever sees CSI escapes.
    # The trade is that any legitimate uses of these (clickable hyperlinks, inline images) never
    # reach the terminal — but they weren't working through pt anyway; better clean than garbled.
    NON_CSI_ESCAPE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\x1b[\]_PX^][^\x07\x1b]*(?:\x07|\x1b\\)")

    @classmethod
    def strip_unknown_escapes(cls, text: str) -> str:
        return cls.NON_CSI_ESCAPE_RE.sub("", text)

    @classmethod
    def strip_trailing_pad(cls, text: str) -> str:
        return "\n".join(cls._strip_line_pad(line) for line in text.split("\n"))

    @classmethod
    def _strip_line_pad(cls, line: str) -> str:
        tokens: list[tuple[str, str]] = []  # ("sgr"|"text", payload)
        bg_states: list[bool] = []  # bg active while each token renders
        bg, idx = False, 0
        for m in cls.SGR_RE.finditer(line):
            if m.start() > idx:
                tokens.append(("text", line[idx : m.start()]))
                bg_states.append(bg)
            tokens.append(("sgr", m.group(0)))
            bg_states.append(bg)
            for param in (m.group(1) or "0").split(";"):
                n = int(param) if param else 0
                if n == 0 or n == 49:
                    bg = False
                elif 40 <= n <= 47 or 100 <= n <= 107 or n == 48:
                    bg = True
            idx = m.end()
        if idx < len(line):
            tokens.append(("text", line[idx:]))
            bg_states.append(bg)
        seen_content = False
        for i in range(len(tokens) - 1, -1, -1):
            kind, payload = tokens[i]
            if kind == "sgr" or seen_content:
                continue
            if bg_states[i]:
                if payload.strip():
                    seen_content = True
                continue
            stripped = payload.rstrip()
            if stripped != payload:
                tokens[i] = ("text", stripped)
            if stripped:
                seen_content = True
        return "".join(payload for _, payload in tokens)

    def emit_answer(self, text: str, *, role: str = "", rule: bool = True, indent: int = 0, compact: bool = False) -> None:
        if not self.color:
            if role == "user":
                text, role = "\n" + self.USER_LOG_PREFIX + text, ""
            elif role == "assistant":
                role = ""
            self.output_fn(self.indent_message(text, role, indent))
            return
        if role == "user":
            # The user's message opens a turn, so it is parted from whatever the last one left
            # behind -- through the printer, which knows whether a gap is already there, rather
            # than by a blank row Rich prints whether one is needed or not.
            self.separate()
        console = markdown_console(shutil.get_terminal_size().columns)
        with console.capture() as capture:
            self.render_message(console, text, role, rule, indent)
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        if compact:
            # Rich markdown pads a blank line after every heading plus a whitespace row above and
            # below each table box; /status wants the heading tight against its table, so drop
            # internal blank lines -- but keep one blank row at each boundary so the command's
            # output does not butt straight against the transcript above it or the prompt below.
            lines = [line for line in cleaned.split("\n") if self.SGR_RE.sub("", line).strip()]
            cleaned = "\n" + "\n".join(lines) + "\n"
        self.track_layout(cleaned)
        if self._batch_parts is not None:
            self._batch_parts.append(ANSI(cleaned))
            return
        self._scrollback_print(ANSI(cleaned))

    # The label sits just past a short lead rather than flush at column 0 (Rich's `align="left"`
    # pushes it to the very edge, which reads as a stray label, not text on a rule) and not
    # centered: a long trail of dashes runs to the full width, so the rule still closes the turn
    # edge to edge.
    TURN_END_LEAD: ClassVar[int] = 2

    def emit_phase_rule(self) -> None:
        """Close a stretch of the turn with the same quiet full-width rule the turn ends with,
        minus the label: the agent's own words -- or, in a long run of silent calls, their
        absence -- are the label, so the rule carries none. It is drawn below an interim
        narration the way the turn-end rule is drawn below the answer, and again at a batch
        boundary when the agent has been working in silence long enough.

        The caller decides whether it would land too close to the rule above it (rule_due);
        this method only draws what it is told to.
        """
        if not self.color:
            return
        # The rule owns both of its seams: a blank row above parts it from the block it closes, and
        # one below keeps whatever follows off it. Neither is the caller's to draw, and neither
        # doubles up when the block above already ended in a gap.
        self.separate()
        width = shutil.get_terminal_size((80, 20)).columns
        fragments = FormattedText([(Theme.fg("rule"), "─" * width + "\n"), ("", "\n")])
        if self._batch_parts is not None:
            self._batch_parts.append(fragments)
        else:
            self._scrollback_print(fragments)
        self.track_layout("─" * width + "\n\n")
        # Distance to the next rule is measured from here, so the rule's own rows do not count.
        self.rows_since_rule = 0

    def rule_due(self, min_rows: int) -> bool:
        """Whether a phase rule would land at least `min_rows` rendered rows below the last one
        drawn, so it is far enough to be worth drawing. Color is part of the answer: without it
        there are no rules to be close to."""
        return self.color and self.rows_since_rule >= min_rows

    def emit_turn_end(self, started_at: float) -> None:
        """Close the turn with a quiet full-width gray rule carrying its total duration.

        The durable counterpart to the animated working divider: the divider counts up while the
        turn runs and is torn down when it ends, so the final elapsed value is frozen here. It
        reuses `elapsed_since` so the rule reads like the divider's last frame (`5s`, `1m05s`)
        instead of the old `0m5s` / `1m5s`. The label is left-biased (a short lead of dashes, then
        the label, then a long trail to the full width) and a blank line lifts the rule off the
        answer above it.
        """
        label = f"done in {Text.elapsed_since(started_at)}"
        if not self.color:
            self.output_fn(label)
            return
        self.separate()
        width = shutil.get_terminal_size((80, 20)).columns
        lead = "─" * self.TURN_END_LEAD + " "
        trail = max(0, width - get_cwidth(lead) - get_cwidth(label) - 1)
        fragments = [
            (Theme.fg("rule"), lead),
            (Theme.fg("text"), label),
            (Theme.fg("rule"), " " + "─" * trail + "\n"),
        ]
        if self._batch_parts is not None:
            self._batch_parts.append(FormattedText(fragments))
        else:
            self._scrollback_print(FormattedText(fragments))
        self.track_layout("".join(fragment for _, fragment in fragments))
        # Distance to the next rule is measured from here, so the rule's own row does not count.
        self.rows_since_rule = 0

    def emit_worker_rule(self, label: str) -> None:
        """Open or close a delegation with a full-width rule whose yellow label names the worker.

        The durable counterpart to the start marker's live divider and a sibling of the turn-end
        rule: gray dashes run edge to edge, the label sits just past a short lead, and the trail
        fills the terminal width. The label is yellow (the worker's identity color) instead of the
        turn-end label's default tone, so the bracket reads at a glance. Blank lines on both
        sides lift the rule off the content above and below it.
        """
        if not self.color:
            self.output_fn(label)
            return
        self.separate()
        width = shutil.get_terminal_size((80, 20)).columns
        limit = max(1, width - 6)
        if get_cwidth(label) > limit:
            available = max(1, limit - get_cwidth("…"))
            clipped = []
            used = 0
            for char in label:
                char_width = max(0, get_cwidth(char))
                if used + char_width > available:
                    break
                clipped.append(char)
                used += char_width
            label = "".join(clipped) + "…"
        lead = "─" * self.TURN_END_LEAD + " "
        trail = max(0, width - get_cwidth(lead) - get_cwidth(label) - 1)
        fragments = [
            (Theme.fg("rule"), lead),
            (Theme.fg("status_worker"), label),
            (Theme.fg("rule"), " " + "─" * trail + "\n"),
        ]
        if self._batch_parts is not None:
            self._batch_parts.append(FormattedText(fragments))
        else:
            self._scrollback_print(FormattedText(fragments))
        self.track_layout("".join(fragment for _, fragment in fragments))
        self.separate()

    @staticmethod
    def indent_message(text: str, role: str = "", indent: int = 0) -> str:
        body = "\n".join(LogBlock.margin(indent) + line for line in text.splitlines() or [""])
        return f"{LogBlock.margin(indent)}{role}:\n{body}" if role else body

    @classmethod
    def colorize_mcp_status(cls, text: str) -> str:
        return cls.MCP_STATUS_RE.sub(lambda match: cls.MCP_STATUS_ANSI[match.group(1)] + "●\x1b[39m " + match.group(1), text)

    def render_message(self, console: Console, text: str, role: str, rule: bool, indent: int) -> None:
        error = text.startswith(("Error:", "ConfigError:", "Unknown command:"))
        styled_text = self.colorize_mcp_status(text) if role != "user" else text
        if rule and not error:
            console.print(Rule(style="wizolt.rule", characters="─"))
        margin = LogBlock.margin(indent)
        if role == "user":
            console.print(Padding(RichText(UiPrinter.USER_LOG_PREFIX + text, style="wizolt.user"), (0, 0, 0, len(margin))))
        elif role == "assistant":
            content = RichText(styled_text, style="wizolt.error") if error else WizoltMarkdown(styled_text)
            console.print(Padding(content, (0, 0, 0, len(margin))))
        else:
            if role:
                label = RichText(role + ":", style=self.MESSAGE_ROLE_STYLES.get(role, "wizolt.muted"))
                console.print(Padding(label, (0, 0, 0, len(margin))))
            content = RichText(styled_text, style="wizolt.error") if error else WizoltMarkdown(styled_text)
            console.print(Padding(content, (0, 0, 0, len(margin))))

    def emit_markdown(self, text: str) -> None:
        # Render markdown to an ANSI string and emit via prompt_toolkit. Printing Rich output directly
        # while the TUI is running can interleave raw escapes with its renderer; capturing first and
        # emitting as ANSI keeps all terminal output inside the shared application.
        if not self.color:
            self.emit(text)
            return
        console = markdown_console(shutil.get_terminal_size().columns)
        with console.capture() as capture:
            console.print(WizoltMarkdown(text))
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        if self._batch_parts is not None:
            self._batch_parts.append(ANSI(cleaned))
            return
        self._scrollback_print(ANSI(cleaned))

    @staticmethod
    def tab_segments(titles: tuple[str, ...], active: int) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        for index, title in enumerate(titles):
            parts.append(("class:tab.active" if index == active else "class:tab.inactive", f" {title} "))
            if index < len(titles) - 1:
                parts.append(("class:choice.disabled", " │ "))
        return parts

    def segments(self, text: str) -> list[tuple[str, str]]:
        if text.startswith(MEMORY_PREFIXES):
            return self.memory_segments(text)
        if text.startswith(self.USER_LOG_PREFIX):
            prefix, content = self.USER_LOG_PREFIX, text[len(self.USER_LOG_PREFIX) :]
            return [(self.user_log_style(), prefix + content + "\n")]
        if text.startswith("+ "):
            return [(Theme.fg("muted"), "+ "), (Theme.fg("text"), text[2:] + "\n")]
        if text.startswith("done in "):
            return [(Theme.fg("muted"), text + "\n")]
        if text.startswith("wizolt "):
            return [(Theme.fg("accent"), text + "\n")]
        if text.startswith(("Error:", "ConfigError:", "Unknown command:")):
            return [(Theme.fg("error"), text + "\n")]
        return [(Theme.fg("text"), line + "\n") for line in text.splitlines() or [""]]

    # A log line's label and its text, as palette roles. The label carries the identity (which tool,
    # which worker, an error) and the text is body copy, so most rows pair a colored label with
    # ordinary text; the rows that are themselves supporting detail go muted on both.
    LOG_ROLES: ClassVar[dict[LogRole, tuple[str, str]]] = {
        LogRole.TOOL: ("tool", "text"),
        LogRole.AUTO: ("accent_secondary", "text"),
        LogRole.META: ("muted", "muted"),
        LogRole.WORKER: ("status_worker", "text"),
        LogRole.FIELD: ("accent", "text"),
        LogRole.OUTPUT: ("muted", "muted"),
        LogRole.ERROR: ("error", "text"),
        LogRole.MUTED: ("muted", "muted"),
        LogRole.DIFF: ("text", "text"),
        LogRole.CODE: ("text", "text"),
    }

    @classmethod
    def log_styles(cls, role: LogRole) -> tuple[str, str]:
        """One log role's (label, text) styles under the active theme."""
        label, text = cls.LOG_ROLES[role]
        return Theme.fg(label), Theme.fg(text)

    def log_segments(self, block: LogBlock) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        width = max(1, shutil.get_terminal_size((120, 20)).columns - 1)
        entries = [(line, level, self.margin_segments(level, rails)) for line, level, rails in block.walk_rows()]
        index = 0
        while index < len(entries):
            line, level, margin = entries[index]
            if line.role is LogRole.DIFF:
                end = index + 1
                while end < len(entries) and entries[end][0].role is LogRole.DIFF and entries[end][1] == level:
                    end += 1
                diff_lines = [entry[0] for entry in entries[index:end]]
                sample_prefix = [*margin, *self.edge_segments(diff_lines[0].edge)]
                sample_prefix_width = sum(get_cwidth(fragment[1]) for fragment in sample_prefix)
                diff_row_width = max(1, width - sample_prefix_width)
                diff_text = "\n".join(item.text for item in diff_lines)
                highlighted = self.segment_lines(self.diff_segments(diff_text, diff_row_width))
                for item, rendered in zip(diff_lines, highlighted):
                    prefix = [*margin, *self.edge_segments(item.edge)]
                    rendered = self.remove_line_ending(rendered)
                    for row in Text.wrap_styled(prefix, prefix, rendered, width):
                        if item.text.startswith("+") and not item.text.startswith("+++"):
                            background = Theme.diff_style("diff.added.bg")
                        elif item.text.startswith("-") and not item.text.startswith("---"):
                            background = Theme.diff_style("diff.removed.bg")
                        else:
                            background = ""
                        if background:
                            used = sum(get_cwidth(fragment[1]) for fragment in row)
                            row.append((background, " " * max(0, width - used)))
                        segments.extend([*row, ("", "\n")])
                index = end
                continue
            if line.role is LogRole.CODE:
                end = index + 1
                while end < len(entries) and entries[end][0].role is LogRole.CODE and entries[end][1] == level:
                    end += 1
                code = [entry[0] for entry in entries[index:end]]
                # Lexed as one block, for the same reason a diff is: a per-line lexer loses every
                # construct that spans lines. The numbers are the excerpt's own 1..N, which is what
                # a ToolScript traceback (`File "<toolscript>", line N`) counts in.
                highlighted = self.code_lines("\n".join(item.text for item in code), code[0].syntax)
                number_width = len(str(len(code)))
                for number, item in enumerate(code, 1):
                    rendered = highlighted[number - 1] if highlighted is not None and number - 1 < len(highlighted) else [(Theme.fg("text"), item.text)]
                    prefix = [*margin, *self.edge_segments(item.edge), (Theme.fg("subtle"), f"{number:>{number_width}}  ")]
                    # A wrapped code row keeps the margin itself (rails included) and blanks only
                    # the edge and gutter, so a long line cannot punch a hole in the rail.
                    continuation = [*margin, ("", " " * sum(get_cwidth(fragment[1]) for fragment in prefix[len(margin) :]))]
                    for row in Text.wrap_styled(prefix, continuation, rendered, width):
                        segments.extend([*row, ("", "\n")])
                index = end
                continue
            label_style, text_style = self.log_styles(line.role)
            prefix = [*margin, *self.edge_segments(line.edge)]
            if line.label:
                prefix.append((label_style, line.label))
            content: list[tuple[str, str]] = []
            if line.text:
                separator = "  " if line.edge is LogEdge.NONE and line.label else " " if line.label else ""
                prefix.append((text_style, separator))
                content.extend(self.syntax_segments(line.text, line.syntax, text_style))
            if line.meta:
                content.append((Theme.fg("error" if line.role is LogRole.ERROR else "muted"), line.meta))
            continuation = [*margin, ("", " " * get_cwidth(line.text_prefix()))]
            for row in Text.wrap_styled(prefix, continuation, content, width):
                segments.extend([*row, ("", "\n")])
            index += 1
        return segments

    @staticmethod
    def margin_segments(level: int, rails: tuple[int, ...]) -> list[tuple[str, str]]:
        """A line's indent as styled segments: rail units in the gray the tree edges use, plain
        spacing everywhere else. Runs of one style are merged, so an indent with no rails in it
        stays the single blank segment it has always been."""
        segments: list[tuple[str, str]] = []
        for rail, text in LogBlock.margin_units(level, rails):
            style = Theme.fg("subtle") if rail else ""
            if segments and segments[-1][0] == style:
                segments[-1] = (style, segments[-1][1] + text)
            else:
                segments.append((style, text))
        return segments

    @staticmethod
    def edge_segments(edge: LogEdge) -> list[tuple[str, str]]:
        return [] if edge is LogEdge.NONE else [(Theme.fg("subtle"), edge.value + " ")]

    @staticmethod
    def remove_line_ending(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
        result = list(segments)
        if result and result[-1][1].endswith("\n"):
            style, text = result[-1]
            result[-1] = (style, text[:-1])
            if not result[-1][1]:
                result.pop()
        return result

    @classmethod
    def syntax_segments(cls, text: str, lexer_name: str, fallback_style: str) -> list[tuple[str, str]]:
        if lexer_name == "tool-args":
            return cls.tool_arg_segments(text, fallback_style)
        if pygments is None or get_lexer_by_name is None or not lexer_name:
            return [(fallback_style, text)]
        try:
            lexer = get_lexer_by_name(lexer_name, stripnl=False, ensurenl=False)
            return [(cls.pygments_style(token_type), value) for token_type, value in lexer.get_tokens(text) if value]
        except Exception:  # noqa: BLE001 - third-party lexers must degrade to plain rendering.
            return [(fallback_style, text)]

    @classmethod
    def tool_arg_segments(cls, text: str, fallback_style: str) -> list[tuple[str, str]]:
        segments = []
        for match in cls.TOOL_ARG_TOKEN.finditer(text):
            token = match.group(0)
            if token.isspace():
                style = fallback_style
            elif token.endswith("="):
                style = Theme.fg("syntax_assign")
            elif token.startswith(('"', "'")):
                style = Theme.fg("syntax_string")
            elif UiPrinter.RECORD_TOKEN_RE.fullmatch(token):
                style = Theme.fg("syntax_number")
            elif token in {";", ","}:
                style = Theme.fg("muted")
            else:
                style = Theme.fg("syntax_ident")
            segments.append((style, token))
        return segments or [(fallback_style, text)]

    def memory_segments(self, text: str) -> list[tuple[str, str]]:
        segments = []
        for line in text.splitlines() or [""]:
            # Deliberately narrower than MEMORY_PREFIXES: these two carry their value on the same
            # line, so the whole line is the headline; `plan:`/`known:` head a list below them and
            # are matched exactly, one case down.
            if line.startswith(("goal:", "check:")):
                segments.append((Theme.fg("accent_secondary"), line))
            elif line in {"summary:", "plan:", "known:"}:
                segments.append((Theme.fg("accent"), line))
            elif line.lstrip().startswith("- [x]"):
                segments.append((Theme.fg("success"), line))
            elif line.lstrip().startswith("- [~]"):
                segments.append((Theme.fg("warning"), line))
            elif line.lstrip().startswith("- [-]"):
                segments.append((Theme.fg("error"), line))
            elif line.lstrip().startswith("+ "):
                segments.append((Theme.fg("success"), line))
            else:
                segments.append((Theme.fg("text"), line))
            segments.append(("", "\n"))
        return segments

    @staticmethod
    def token_definition(style: Any, token_type: Any) -> dict | None:
        """The style entry for a token, falling back to its ancestors, or None if nothing matches.

        `style_for_token` raises KeyError when a style defines nothing anywhere in a token's
        subtree, and a style only has to cover the tokens its own authors thought about: the YAML
        lexer emits `Token.Literal.Scalar.Plain` and `Token.Punctuation.Indicator`, which
        github-dark never mentions, so highlighting any `.yaml` used to abort mid-render and take
        the whole Edit down with it.

        Walking up to the parent is what Pygments' own token hierarchy is for: a plain scalar
        renders like the Literal it is, rather than dropping the whole file to unstyled text.
        """
        while token_type is not None:
            try:
                return style.style_for_token(token_type)
            except KeyError:
                token_type = getattr(token_type, "parent", None)
        return None

    @classmethod
    def pygments_style(cls, token_type: Any) -> str:
        style = Theme.pygments_style()
        if style is None or Token is None:
            return "fg:default"
        if token_type in Token.Text.Whitespace:
            return "fg:default"
        if token_type in Token.Name.Builtin:
            return Theme.fg("syntax_builtin")
        definition = cls.token_definition(style, token_type)
        if definition is None:
            return "fg:default"
        color = definition.get("color")
        default_hex = Theme.color("syntax_default").lstrip("#")
        parts = ["fg:default" if not color or color.lower() == default_hex else f"fg:#{color}"]
        parts.extend(attribute for attribute in ("bold", "italic", "underline") if definition.get(attribute))
        return " ".join(parts)

    def _diff_tokenize_lines(self, code_text: str, path: str | None) -> list[list[tuple[str, str]]] | None:
        """Tokenize a whole block of code and return highlighted segments per line.

        Pygments lexers are designed to work on whole files; splitting by diff
        lines and lexing each one independently breaks multiline strings and
        indentation-sensitive languages.  We therefore lex the assembled code
        block once and split the resulting token stream back into lines.
        """
        if pygments is None or get_lexer_for_filename is None or not path:
            return None
        try:
            lexer = get_lexer_for_filename(path, stripnl=False)
        except Exception:  # noqa: BLE001 - third-party lexer lookup must degrade to plain rendering.
            return None
        return self._tokenized_lines(lexer, code_text)

    @classmethod
    def code_lines(cls, code_text: str, lexer_name: str) -> list[list[tuple[str, str]]] | None:
        """Highlighted segments per source line for a standalone block of code, by lexer name.

        The same whole-block lexing _diff_tokenize_lines relies on, for code that arrives without a
        filename to infer a lexer from: a ToolScript body, anything a viewer shows. Returns None
        when pygments is missing or the name is unknown, leaving the caller to render plain text."""
        if pygments is None or get_lexer_by_name is None or not lexer_name:
            return None
        try:
            lexer = get_lexer_by_name(lexer_name, stripnl=False)
        except Exception:  # noqa: BLE001 - third-party lexer lookup must degrade to plain rendering.
            return None
        return cls._tokenized_lines(lexer, code_text)

    @classmethod
    def _tokenized_lines(cls, lexer: Any, code_text: str) -> list[list[tuple[str, str]]] | None:
        # The whole walk is guarded, not just the call that starts it: get_tokens is a generator,
        # so a lexer that fails does it while this loop pulls from it, not here. Highlighting is
        # decoration -- a lexer that cannot cope must cost the color, never the edit it was
        # previewing.
        try:
            lines: list[list[tuple[str, str]]] = [[]]
            for token_type, value in lexer.get_tokens(code_text):
                style = cls.pygments_style(token_type)
                parts = value.split("\n")
                for i, part in enumerate(parts):
                    if i > 0:
                        lines.append([])
                    if part:
                        lines[-1].append((style, part))
            return lines
        except Exception:  # noqa: BLE001 - third-party lexer execution must degrade to plain rendering.
            return None

    # Width taken by the line-number gutter emitted inside diff_segments (`NNNN NNNN | `).
    DIFF_GUTTER_WIDTH: ClassVar[int] = 12

    def diff_segments(self, text: str, row_width: int | None = None) -> list[tuple[str, str]]:
        return self._diff_segments(text, row_width=row_width, live=False)

    def diff_segments_live(self, text: str, row_width: int | None = None) -> list[tuple[str, str]]:
        """Same as diff_segments, but pads the bg band to the current pane width. Only for live
        live renderers that repaint on resize (the `/diff` viewer). Scrollback callers must
        NOT use this — baked-in wide padding wraps on a later pane shrink and drops the bg color on
        the wrapped continuation, which looks broken."""
        return self._diff_segments(text, row_width=row_width, live=True)

    def _diff_segments(self, text: str, *, row_width: int | None, live: bool) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        old_line: int | None = None
        new_line: int | None = None
        lines = text.splitlines()
        # The live viewer repaints on resize, so it can pad directly to the current pane width.
        # Scrollback is padded only after wrapping in log_segments; padding a logical line here
        # would either be discarded at a word boundary or create an extra visual row.
        changed_width: int | None = None
        if live:
            if row_width is None:
                row_width = shutil.get_terminal_size((120, 20)).columns - 3
            changed_width = max(1, row_width - self.DIFF_GUTTER_WIDTH)

        # Determine the target file path from the diff header.  The `+++` line
        # names the resulting file; for created files `---` is /dev/null.
        file_path: str | None = None
        for header in lines:
            if header.startswith("+++"):
                candidate = header[4:].strip()
                if candidate != "/dev/null":
                    file_path = candidate
                break

        # Collect lines that belong to the new file version: context lines and
        # added lines.  These are lexed together so the highlighted diff is
        # syntactically coherent. Removed lines stay neutral on a red background so
        # the "before" state does not interfere with lexing the "after" state.
        new_code_lines: list[str] = []
        new_code_indices: list[int] = []
        for i, line in enumerate(lines):
            # Skip the unified-diff file headers / hunk markers (the trailing space avoids matching a
            # real added line whose content starts with "+++"); feed only actual code to the lexer.
            if line.startswith(("+++ ", "--- ", "@@ ")):
                continue
            if line.startswith(("+", " ")):
                new_code_lines.append(line[1:])
                new_code_indices.append(i)

        highlighted: list[list[tuple[str, str]]] | None = None
        if new_code_lines:
            highlighted = self._diff_tokenize_lines("\n".join(new_code_lines), file_path)

        hl_by_index: dict[int, list[tuple[str, str]]] = {}
        if highlighted is not None:
            for hl_index, line_index in enumerate(new_code_indices):
                if hl_index < len(highlighted):
                    hl_by_index[line_index] = highlighted[hl_index]

        def hunk_start(part: str, prefix: str) -> int | None:
            if not part.startswith(prefix):
                return None
            try:
                return int(part[1:].split(",", 1)[0])
            except ValueError:
                return None

        # The styles below stay ANSI names on purpose. A diff's colors are pinned (see Theme's diff
        # mapping): the signs, gutter, and hunk headers were tuned against these bands in both
        # appearances, so they are not migrated to palette roles with the rest of the UI.
        def number(old: int | None, new: int | None, background: str = "") -> None:
            old_text = "" if old is None else str(old)
            new_text = "" if new is None else str(new)
            segments.append((("ansibrightblack " + background).strip(), f"{old_text:>4} {new_text:>4} | "))

        def append_hl(prefix: str, prefix_style: str, content_hl: list[tuple[str, str]], suffix: str, background: str = "") -> None:
            def styled(style: str) -> str:
                return (style + " " + background).strip()

            segments.append((styled(prefix_style), prefix))
            for style, piece in content_hl:
                segments.append((styled(style), piece))
            width = get_cwidth(prefix) + sum(get_cwidth(fragment[1]) for fragment in content_hl)
            padding = " " * max(0, changed_width - width) if background and changed_width is not None else ""
            segments.append((background if padding else "", padding + suffix))

        for index, line in enumerate(lines):
            suffix = "\n" if index < len(lines) - 1 else ""
            if line.startswith("@@"):
                parts = line.split()
                if len(parts) >= 3:
                    old_line = hunk_start(parts[1], "-")
                    new_line = hunk_start(parts[2], "+")
                number(None, None)
                segments.append(("ansicyan", line + suffix))
            elif line.startswith(("---", "+++")):
                number(None, None)
                segments.append(("ansibrightblack", line + suffix))
            elif line.startswith("+"):
                background = Theme.diff_style("diff.added.bg")
                number(None, new_line, background)
                content_hl = hl_by_index.get(index) or [(Theme.diff_style("diff.added.fg"), line[1:])]
                append_hl("+", "ansigreen", content_hl, suffix, background)
                new_line = None if new_line is None else new_line + 1
            elif line.startswith("-"):
                background = Theme.diff_style("diff.removed.bg")
                number(old_line, None, background)
                append_hl("-", "ansired", [(Theme.diff_style("diff.removed.fg"), line[1:])], suffix, background)
                old_line = None if old_line is None else old_line + 1
            elif line.startswith(" "):
                number(old_line, new_line)
                content_hl = hl_by_index.get(index) or [("fg:default", line[1:])]
                append_hl(" ", "fg:default", content_hl, suffix)
                old_line = None if old_line is None else old_line + 1
                new_line = None if new_line is None else new_line + 1
            else:
                number(None, None)
                segments.append(("fg:default", line + suffix))
        return segments

    @staticmethod
    def segment_lines(segments: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
        lines: list[list[tuple[str, str]]] = [[]]
        for style, text in segments:
            parts = text.split("\n")
            for index, part in enumerate(parts):
                if index > 0:
                    lines[-1].append((style, "\n"))
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        if lines and not lines[-1]:
            lines.pop()
        return lines


class LiveSpark:
    """The mark that says a live region is still alive, for regions with nothing else moving.

    A streaming response and a running command both spend long stretches with only a clock
    changing, and a still block does not say it is still running. Both cap their rail with this
    instead of another `│`, and share one definition so the two regions never drift apart.

    Deliberately wordless: whatever the region is doing is already named on the divider under it,
    and saying it twice on one screen is worse than not saying it here at all.
    """

    # Exactly the width of a rail, so it sits in the rail's column and the rows keep their own.
    # A text glyph rather than an emoji: terminals draw emoji in their own colors, so a breath put
    # on one lands on whatever is beside it instead, and emoji are two cells before the space.
    GLYPH: ClassVar[str] = "✦ "
    # Two weights of the star, one per breath, swapped at the crest so every star is seen at full
    # brightness: the opening breath is the fine four-point star, the next breath the heavy
    # six-point one, then back. GLYPH is the phase-zero entry.
    GLYPHS: ClassVar[tuple[str, ...]] = ("✦ ", "✶ ")
    # Twice the divider pulse's period: that dot marks a request in flight and should read as a
    # heartbeat, while this one sits over a wall of text and would nag at that rate.
    PERIOD: ClassVar[float] = 3.2
    STEPS: ClassVar[int] = 12
    # The divider's own accent, so the live rule and the live region it caps read as one thing.
    ROLE: ClassVar[str] = "accent"
    # How far the breath reaches past that color, as a fraction of the way to black at the trough
    # and to white at the crest. Wide on purpose: a shallow fade reads as the terminal mis-drawing
    # a cell rather than as a breath, and the crest has to clear the gray rows beside it. The
    # crest stays shy of pure white so a light terminal does not lose the mark against its own
    # background, but is close enough to read as white on a dark one.
    # The star is thin in its own shape and the terminal has no font size, so it is bold for the
    # whole ramp: that is the only way the mark reads heavier than the rows beside it. The breath
    # survives as the color ramp alone.
    FLOOR: ClassVar[float] = 0.78
    CEILING: ClassVar[float] = 0.92

    @classmethod
    def ramp(cls) -> list[str]:
        """The spark's shades, darkest to brightest, around the divider's accent.

        Derived from the palette rather than hard-coded like the divider's pulse: that dot is green
        in both themes and answers to nothing, while this spark caps the divider's own rule and has
        to keep sharing its color when a theme changes it.
        """
        hue = Theme.rgb(Theme.color(cls.ROLE))
        low = Theme.rgb(Theme.mix(hue, (0, 0, 0), cls.FLOOR))
        high = Theme.rgb(Theme.mix(hue, (255, 255, 255), cls.CEILING))
        span = max(1, cls.STEPS - 1)
        return ["fg:" + Theme.mix(low, high, step / span) + " bold" for step in range(cls.STEPS)]

    @classmethod
    def style(cls, started_at: float = 0.0) -> str:
        """Where on the ramp the spark sits now: a triangular breath measured from `started_at`,
        opening at the crest and falling from there.

        Both halves of that matter. Timed off the wall clock instead, a region catches the cycle
        wherever it happens to be, so one appearing near the trough opens near-black and stays
        unreadable for over a second — precisely the moment it exists to announce. And starting
        bright is what makes the arrival the loudest frame; the breath afterwards is what says it
        is still going.

        A falsy `started_at` falls back to the wall clock, which is the old arbitrary phase but
        never a crash: a caller with no anchor still gets a breathing spark.
        """
        elapsed = (time.monotonic() - started_at) if started_at else time.monotonic()
        phase = (elapsed % cls.PERIOD) / cls.PERIOD
        intensity = abs(2.0 * phase - 1.0)  # 1 at the start, 0 at the half-period, 1 again
        ramp = cls.ramp()
        return ramp[min(len(ramp) - 1, int(intensity * len(ramp)))]

    @classmethod
    def glyph(cls, started_at: float = 0.0) -> str:
        """The spark's mark at this phase: one star per breath, the fine four-point star on the
        opening breath and the heavy six-point star on the next, so each is seen at full
        brightness; `GLYPH` is the opening entry. Shares the phase clock with `style`."""
        elapsed = (time.monotonic() - started_at) if started_at else time.monotonic()
        breath = int(elapsed // cls.PERIOD)
        return cls.GLYPHS[breath % 2]


class BashLivePreview:
    HEIGHT: ClassVar[int] = 6
    MAX_CHARS: ClassVar[int] = 8000
    # Heartbeat tick so the elapsed timer advances even while a command produces no output
    # (e.g. quiet long-runners or `... | tail` that buffers until EOF), so the terminal never
    # looks frozen during a blocking command.
    TICK: ClassVar[float] = 0.3

    def __init__(self):
        self.output = create_output(sys.stderr)
        self.active = False
        self.rendered_lines = 0
        self.rendered_rows: list[list[tuple[str, str]]] = []
        self.text = ""
        self.started_at = 0.0
        # Monotonic absolute end of the current wait budget, for the `· Ns left` countdown; None
        # for Bash, which has no budget to count down.
        self.deadline: float | None = None
        self.lock = threading.Lock()
        self.timer: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        with self.lock:
            self.active, self.rendered_lines, self.rendered_rows, self.text = True, 0, [], ""
            self.started_at = time.monotonic()
            self.deadline = None
            self.render()
        self.timer = threading.Thread(target=self.tick, daemon=True)
        self.timer.start()

    def tick(self) -> None:
        while True:
            time.sleep(self.TICK)
            with self.lock:
                if not self.active:
                    return
                self.render()

    def update(self, text: str) -> None:
        with self.lock:
            if not self.active:
                return
            self.text = (self.text + text)[-self.MAX_CHARS :]
            self.render()

    def finish(self) -> None:
        with self.lock:
            if not self.active:
                return
            self.active = False
        timer = self.timer
        if timer is not None:
            timer.join()
        with self.lock:
            self.rendered_lines, self.rendered_rows, self.text = 0, [], ""
            self.deadline = None

    def render(self) -> None:
        if not self.active:
            return
        rows = self.frame_rows()
        if rows == self.rendered_rows:
            return
        previous = self.rendered_lines
        if self.rendered_lines:
            self.output.write_raw(f"\x1b[{self.rendered_lines}A")
        for row in rows:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(row), output=self.output, end="", flush=True)
            self.output.write_raw("\n")
        for _ in range(max(0, previous - len(rows))):
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.write_raw("\n")
        if previous > len(rows):
            self.output.write_raw(f"\x1b[{previous - len(rows)}A")
        self.output.flush()
        self.rendered_lines = len(rows)
        self.rendered_rows = rows

    def frame_rows(self) -> list[list[tuple[str, str]]]:
        """The frame as styled rows: a breathing spark capping the rail, then the output under it.

        The status row used to carry a BRANCH edge under a `hierarchy(None, ...)` — a `├` with no
        root above it at all, claiming a line joining from a place there was never anything. The
        spark takes that cell instead and is the same width, so the rows keep their column. It is
        also the only thing that moves: a command that goes quiet for minutes leaves nothing else
        on screen changing but the clock, which is what the tick loop was already there for.
        """
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        body = [line.expandtabs(4) for line in self.text.replace("\r", "\n").splitlines()[-self.HEIGHT :]]
        label = Text.elapsed_since(self.started_at, precise=True)
        remaining = f" · {math.ceil(max(0.0, self.deadline - time.monotonic()))}s left" if self.deadline is not None else ""
        # `limit` leaves a column of slack so a full-width line cannot auto-wrap and desync the
        # cursor-up math in render().
        rail = LogBlock.prefix(2, LogEdge.CONTINUE)
        limit = max(1, width - get_cwidth(rail) - 1)
        # Always emit a status row so the frame is visible even before any output arrives.
        status = f"output · {label}{remaining}" if body else f"running… {label}{remaining}"
        rows = [[(LiveSpark.style(self.started_at), LogBlock.margin(2) + LiveSpark.glyph(self.started_at)), (Theme.fg("muted"), status)]]
        if body:
            # A blank row keeps the spark off the rail: the star caps the region, it does not sit on it.
            rows.append([("", "")])
            rows.extend([(Theme.fg("muted"), rail + Text.clip_width(line, limit))] for line in body)
        return rows


class StatusBar:
    """A quiet, colored summary of the session, owning none of the state it displays.

    Every value displayed is read from session state the engine already maintains. It is a view: it
    never blocks a turn, and must never become the reason a piece of state exists.

    The TUI reads the fragments whenever its ordinary events redraw the screen. The simple frontend
    writes the same static row to stderr once at turn start and erases it on stop, keeping it out of
    piped transcripts without a timer or repaint thread.
    """

    RETRY_NOTICE_DURATION: ClassVar[float] = 2.0
    ROLE_KEYS: ClassVar[tuple[str, ...]] = ("provider", "reason", "mcp", "context", "index", "yolo", "worker")

    @classmethod
    def role_style(cls, role: str) -> str:
        return Theme.fg("status_" + role) if role in cls.ROLE_KEYS else Theme.fg("status_base")

    def __init__(self, session: Session):
        self.session = session
        self.started_at = 0.0
        self.running = False
        self.rendered = False
        self.output = create_output(sys.stderr)
        self.seen_retry_count = session.state.model_retry_count
        self.retry_notice_until = 0.0

    def start(self, *, reset: bool = True) -> None:
        if self.running or not sys.stderr.isatty():
            return
        self.begin(reset=reset)
        self.output.write_raw("\r")
        self.output.erase_end_of_line()
        print_formatted_text(FormattedText(self.fragments()), output=self.output, end="", flush=True)
        self.rendered = True
        self.running = True

    def begin(self, *, reset: bool = True) -> None:
        if reset or not self.started_at:
            self.started_at = time.monotonic()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.clear()

    def is_running(self) -> bool:
        return self.running

    def clear(self) -> None:
        if self.rendered:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.flush()
            self.rendered = False

    def retry_notice_active(self) -> bool:
        now = time.monotonic()
        count = self.session.state.model_retry_count
        if count != self.seen_retry_count:
            self.seen_retry_count = count
            self.retry_notice_until = now + self.RETRY_NOTICE_DURATION
        return self.retry_notice_until > now

    def output_rate(self) -> str:
        """The model's live output speed while a response streams, as `↓ 48 tok/s`, or "" when
        nothing is streaming.

        Estimated from streamed characters at four per token, because token deltas are not on the
        wire: providers report usage once, when the request is over, which is exactly too late for
        the line the reader is watching. The `↓` marks the incoming stream, not a measured token
        rate. Suppressed for the first second, where a couple of chunks over a near-zero elapsed
        reads as a wild number.

        Read off the in-flight worker when there is one, like every other value on this row.
        """
        state = self.active_session().state
        if not state.stream_started_at or not state.stream_chars:
            return ""
        elapsed = time.monotonic() - state.stream_started_at
        if elapsed < 1.0:
            return ""
        return f"↓ {round(state.stream_chars / 4 / elapsed)} tok/s"

    def active_session(self) -> Session:
        """The session whose stream the working divider describes."""
        worker = self.session.worker
        return worker if worker is not None and bool(worker._active_turn_messages) else self.session

    def model_attempt_status(self) -> str:
        attempt = self.session.state.current_model_attempt
        return f"attempt {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else ""

    def retry_status(self) -> str:
        # The two-second notice window covers the brief aftermath after a wait ends. While the wait
        # itself is in progress (its monotonic deadline is still in the future) the full text
        # (attempt, reason, countdown) must keep showing for its whole duration, which can far
        # outlast that window.
        waiting = self.session.state.model_retry_until > time.monotonic()
        if not waiting and not self.retry_notice_active():
            return ""
        attempt = self.session.state.current_model_attempt
        text = f"retrying {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else "retrying"
        state = self.session.state
        reason = state.model_retry_reason
        if reason:
            text += " · " + reason
        # The model publishes the wait deadline as a fact; the renderer formats the countdown.
        remaining = max(0, math.ceil(state.model_retry_until - time.monotonic()))
        if remaining:
            text += f" · {remaining}s"
        return text

    def fragments(self) -> StyleAndTextTuples:
        """Render the stable status row in its fixed group order and semantic colors."""
        config = self.session.config
        provider = config.provider
        model = provider.model.rsplit("/", 1)[-1] or "(no model)"
        usage = self.session.usage
        if usage.last_prompt_tokens and usage.last_prompt_budget:
            ctx_percent = min(100, usage.last_prompt_tokens * 100 // usage.last_prompt_budget)
        else:
            ctx_percent = self.session.state.context_percent
        cache_percent = usage.last_cached_prompt_tokens * 100 // usage.last_prompt_tokens if usage.last_prompt_tokens else 0
        mcp_count = sum(self.session.mcp.connected(item.name) for item in self.session.mcp.parse_configs()) if self.session.mcp is not None else 0
        skill_count = len(self.session.skills.skills) if self.session.skills else 0

        identity: list[tuple[str, str]] = []
        if self.session.settings.yolo:
            identity.append(("[yolo] ", "yolo"))
        identity.extend([(config.active_provider + "/" + model, "provider"), (" · ", "sep"), (provider.reasoning, "reason")])
        groups: list[list[tuple[str, str]]] = [
            identity,
            [(f"mcp {mcp_count}", "mcp"), (" · ", "sep"), (f"skills {skill_count}", "mcp")],
            [(f"ctx {ctx_percent}%", "context"), (" · ", "sep"), (f"cache {cache_percent}%", "context")],
            [("index" + self.index_status(), "index")],
        ]
        fragments: StyleAndTextTuples = []
        for group in groups:
            if fragments:
                fragments.append((Theme.fg("subtle"), " | "))
            fragments.extend((self.role_style(role), text) if role != "sep" else (Theme.fg("subtle"), text) for text, role in group)

        text = "".join(fragment[1] for fragment in fragments)
        columns = shutil.get_terminal_size((120, 20)).columns
        if get_cwidth(text) >= columns:
            return self.clip_fragments(fragments, columns - 1)
        return fragments

    @staticmethod
    def clip_fragments(fragments: StyleAndTextTuples, width: int) -> StyleAndTextTuples:
        """Clip styled fragments to a display width while keeping each segment's style, mirroring
        Text.clip_width's trailing ellipsis. Lets the idle status bar keep its role colors when the
        line is wider than the terminal instead of collapsing to a single tone."""
        width = max(0, width)
        if width == 0:
            return [("", "")]
        ellipsis = "." * min(3, width)
        available = width - get_cwidth(ellipsis)
        clipped: StyleAndTextTuples = []
        used = 0
        for style, text, *_ in fragments:
            for char in text:
                char_width = max(0, get_cwidth(char))
                if used + char_width > available:
                    clipped.append((style, ellipsis))
                    return clipped
                clipped.append((style, char))
                used += char_width
        return clipped or [("", "")]

    def index_status(self) -> str:
        if self.session.state.code_index_error:
            return CodeIndex.label("error")
        if self.session.state.code_index_refreshing:
            notice = self.session.state.code_index_notice or "syncing"
            return CodeIndex.label(notice)
        return CodeIndex.label(self.session.state.code_index_status)
