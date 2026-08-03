"""minacode terminal rendering, live output, and status display."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI, FormattedText, StyleAndTextTuples
from prompt_toolkit.output import create_output
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text as RichText

from minacode.base import (
    MODEL_REQUEST_RETRIES,
    Json,
    LogBlock,
    LogEdge,
    LogLine,
    LogRole,
    Text,
    __version__,
)
from minacode.context import ContextManager
from minacode.session import Session
from minacode.tools import CodeIndex

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


class Theme:
    DARK: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#003b00",
        "diff.added.fg": "fg:default",
        "diff.removed.bg": "bg:#520000",
        "diff.removed.fg": "fg:default",
        "syntax.assign": "fg:#79c0ff",
        "syntax.string": "fg:#a5d6ff",
        "syntax.number": "fg:#d2a8ff",
        "syntax.ident": "fg:#a5d6ff",
        "syntax.builtin": "fg:#79c0ff",
        "syntax.default_hex": "e6edf3",
        # The status line sits under the conversation and should read as a quiet footer, not compete
        # with it, so its plain tone stays below full white.
        "status.base": "#cbd5e1",
        "status.sep": "#4b5563",
        "status.provider": "#cbd5e1",
        "status.sweep.start": "#4f9fc4",
        "status.sweep.end": "#9b82c9",
        "status.sweep.crest": "#cfe6f2",
        "status.reason": "#a5b4fc",
        "status.mcp": "#93c5fd",
        "status.ctx": "#facc15",
        "status.update": "#fb923c",
        "status.index": "#94a3b8",
        "status.warn": "#fb7185",
        "divider.glow": "#67e8f9",
        "divider.rule": "#4b5563",
        "user.log": "#e0a96d",
        "pygments": "github-dark",
    }

    LIGHT: ClassVar[dict[str, str]] = {
        "diff.added.bg": "bg:#d1f0d1",
        "diff.added.fg": "fg:#003b00",
        "diff.removed.bg": "bg:#f5c8c8",
        "diff.removed.fg": "fg:#520000",
        "syntax.assign": "fg:#005cc5",
        "syntax.string": "fg:#032f62",
        "syntax.number": "fg:#6f42c1",
        "syntax.ident": "fg:#032f62",
        "syntax.builtin": "fg:#005cc5",
        "syntax.default_hex": "24292e",
        "status.base": "#4b5563",
        "status.sep": "#9ca3af",
        "status.provider": "#4b5563",
        # On a light terminal the crest is the darkest point: contrast, not brightness, is what
        # makes the travelling band read as a highlight.
        "status.sweep.start": "#3b7ea3",
        "status.sweep.end": "#6b52a3",
        "status.sweep.crest": "#1f2937",
        "status.reason": "#5b21b6",
        "status.mcp": "#1e40af",
        "status.ctx": "#a16207",
        "status.update": "#9a3412",
        "status.index": "#475569",
        "status.warn": "#b91c1c",
        "divider.glow": "#0e7490",
        "divider.rule": "#9ca3af",
        "user.log": "#9a5b2e",
        "pygments": "default",
    }

    _mode: ClassVar[str] = "dark"
    _pygments_cache: ClassVar[dict[str, type[PygmentsStyle] | None]] = {}

    @classmethod
    def set_mode(cls, mode: str) -> None:
        cls._mode = "light" if mode == "light" else "dark"

    @classmethod
    def style(cls, key: str) -> str:
        return (cls.LIGHT if cls._mode == "light" else cls.DARK)[key]

    @classmethod
    def ramp(cls, start_key: str, end_key: str, steps: int) -> list[str]:
        """Interpolate `steps` hex colors from one palette entry to another.

        Used for gradients that need more shades than the palette names, so a moving highlight can
        fade between two cells instead of snapping from one named color to the next.
        """
        start, end = cls.rgb(cls.style(start_key)), cls.rgb(cls.style(end_key))
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
        name = cls.style("pygments")
        if name not in cls._pygments_cache:
            try:
                cls._pygments_cache[name] = get_style_by_name(name)
            except Exception:  # noqa: BLE001 - optional Pygments styles must degrade to plain rendering.
                cls._pygments_cache[name] = None
        return cls._pygments_cache[name]


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

    MESSAGE_ROLE_STYLES: ClassVar[dict[str, str]] = {"user": "cyan bold", "assistant": "magenta bold"}
    PROMPT_PREFIX: ClassVar[str] = "> "
    USER_LOG_PREFIX: ClassVar[str] = "• "
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
        return Theme.style("user.log")

    TOOL_ARG_TOKEN: ClassVar[re.Pattern] = re.compile(
        r"""\s+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_][\w.-]*=|(?:tr|job)\.\d+|\d+(?::\d+)?|[;,]|[^\s;,]+"""
    )

    def __init__(self, output_fn=print):
        self.output_fn = output_fn
        self.color = output_fn is print and sys.stdout.isatty()

    def emit(self, text: str | LogBlock = "") -> None:
        if not self.color:
            self.output_fn(str(text))
            return
        segments = self.log_segments(text) if isinstance(text, LogBlock) else self.segments(text)
        print_formatted_text(FormattedText(segments), end="", flush=True)

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

    def emit_answer(self, text: str, *, role: str = "", rule: bool = True, indent: int = 0) -> None:
        if not self.color:
            if role == "user":
                text, role = "\n" + self.USER_LOG_PREFIX + text, ""
            elif role == "assistant":
                role = ""
            self.output_fn(self.indent_message(text, role, indent))
            return
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=shutil.get_terminal_size().columns)
        with console.capture() as capture:
            self.render_message(console, text, role, rule, indent)
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        print_formatted_text(ANSI(cleaned), end="", flush=True)

    # The label sits just past a short lead rather than flush at column 0 (Rich's `align="left"`
    # pushes it to the very edge, which reads as a stray label, not text on a rule) and not
    # centered: a long trail of dashes runs to the full width, so the rule still closes the turn
    # edge to edge.
    TURN_END_LEAD: ClassVar[int] = 2

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
        self.emit()
        width = shutil.get_terminal_size((80, 20)).columns
        lead = "─" * self.TURN_END_LEAD + " "
        trail = max(0, width - get_cwidth(lead) - get_cwidth(label) - 1)
        fragments = [
            ("ansibrightblack", lead),
            ("fg:default", label),
            ("ansibrightblack", " " + "─" * trail + "\n"),
        ]
        print_formatted_text(FormattedText(fragments), end="", flush=True)

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
            console.print(Rule(style="bright_black", characters="─"))
        margin = LogBlock.margin(indent)
        if role == "user":
            console.print("")
            console.print(Padding(RichText(UiPrinter.USER_LOG_PREFIX + text, style=self.user_log_style()), (0, 0, 0, len(margin))))
        elif role == "assistant":
            content = RichText(styled_text, style="red") if error else Markdown(styled_text, hyperlinks=False)
            console.print(Padding(content, (0, 0, 0, len(margin))))
        else:
            if role:
                label = RichText(role + ":", style=self.MESSAGE_ROLE_STYLES.get(role, "bright_black"))
                console.print(Padding(label, (0, 0, 0, len(margin))))
            content = RichText(styled_text, style="red") if error else Markdown(styled_text, hyperlinks=False)
            console.print(Padding(content, (0, 0, 0, len(margin))))

    def emit_markdown(self, text: str) -> None:
        # Render markdown to an ANSI string and emit via prompt_toolkit. Printing Rich output directly
        # while the TUI is running can interleave raw escapes with its renderer; capturing first and
        # emitting as ANSI keeps all terminal output inside the shared application.
        if not self.color:
            self.emit(text)
            return
        console = Console(force_terminal=True, color_system="truecolor", no_color=False, width=shutil.get_terminal_size().columns)
        with console.capture() as capture:
            console.print(Markdown(text, hyperlinks=False))
        cleaned = self.strip_unknown_escapes(self.strip_trailing_pad(capture.get()))
        print_formatted_text(ANSI(cleaned), end="", flush=True)

    @staticmethod
    def tab_segments(titles: tuple[str, ...], active: int) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        for index, title in enumerate(titles):
            parts.append(("class:tab.active" if index == active else "class:tab.inactive", f" {title} "))
            if index < len(titles) - 1:
                parts.append(("class:choice.disabled", " │ "))
        return parts

    def segments(self, text: str) -> list[tuple[str, str]]:
        if text.startswith(("goal:", "check:", "plan:", "known:")):
            return self.memory_segments(text)
        if text.startswith(self.USER_LOG_PREFIX):
            prefix, content = self.USER_LOG_PREFIX, text[len(self.USER_LOG_PREFIX) :]
            return [(self.user_log_style(), prefix + content + "\n")]
        if text.startswith("+ "):
            return [("ansibrightblack", "+ "), ("fg:default", text[2:] + "\n")]
        if text.startswith("done in "):
            return [("ansibrightblack", text + "\n")]
        if text.startswith("minacode "):
            return [("ansicyan", text + "\n")]
        if text.startswith(("Error:", "ConfigError:", "Unknown command:")):
            return [("ansired", text + "\n")]
        return [("fg:default", line + "\n") for line in text.splitlines() or [""]]

    LOG_STYLES: ClassVar[dict[LogRole, tuple[str, str]]] = {
        LogRole.TOOL: ("ansigreen", "fg:default"),
        LogRole.AUTO: ("ansiblue", "fg:default"),
        LogRole.META: ("ansibrightblack", "ansibrightblack"),
        LogRole.OUTPUT: ("ansibrightblack", "ansibrightblack"),
        LogRole.ERROR: ("ansired", "fg:default"),
        LogRole.MUTED: ("ansibrightblack", "ansibrightblack"),
        LogRole.DIFF: ("fg:default", "fg:default"),
    }

    def log_segments(self, block: LogBlock) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        width = max(1, shutil.get_terminal_size((120, 20)).columns - 1)
        entries = list(block.walk())
        index = 0
        while index < len(entries):
            line, level = entries[index]
            if line.role is LogRole.DIFF:
                end = index + 1
                while end < len(entries) and entries[end][0].role is LogRole.DIFF and entries[end][1] == level:
                    end += 1
                diff_lines = [entry[0] for entry in entries[index:end]]
                sample_prefix = [("", block.margin(level)), *self.edge_segments(diff_lines[0].edge)]
                sample_prefix_width = sum(get_cwidth(fragment[1]) for fragment in sample_prefix)
                diff_row_width = max(1, width - sample_prefix_width)
                diff_text = "\n".join(item.text for item in diff_lines)
                highlighted = self.segment_lines(self.diff_segments(diff_text, diff_row_width))
                for item, rendered in zip(diff_lines, highlighted):
                    prefix = [("", block.margin(level)), *self.edge_segments(item.edge)]
                    rendered = self.remove_line_ending(rendered)
                    for row in Text.wrap_styled(prefix, prefix, rendered, width):
                        if item.text.startswith("+") and not item.text.startswith("+++"):
                            background = Theme.style("diff.added.bg")
                        elif item.text.startswith("-") and not item.text.startswith("---"):
                            background = Theme.style("diff.removed.bg")
                        else:
                            background = ""
                        if background:
                            used = sum(get_cwidth(fragment[1]) for fragment in row)
                            row.append((background, " " * max(0, width - used)))
                        segments.extend([*row, ("", "\n")])
                index = end
                continue
            label_style, text_style = self.LOG_STYLES[line.role]
            prefix = [("", block.margin(level)), *self.edge_segments(line.edge)]
            if line.label:
                prefix.append((label_style, line.label))
            content: list[tuple[str, str]] = []
            if line.text:
                separator = "  " if line.edge is LogEdge.NONE and line.label else " " if line.label else ""
                prefix.append((text_style, separator))
                content.extend(self.syntax_segments(line.text, line.syntax, text_style))
            if line.meta:
                content.append(("ansired" if line.role is LogRole.ERROR else "ansibrightblack", line.meta))
            continuation = [("", block.margin(level) + " " * get_cwidth(line.text_prefix()))]
            for row in Text.wrap_styled(prefix, continuation, content, width):
                segments.extend([*row, ("", "\n")])
            index += 1
        return segments

    @staticmethod
    def edge_segments(edge: LogEdge) -> list[tuple[str, str]]:
        return [] if edge is LogEdge.NONE else [("ansibrightblack", edge.value + " ")]

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
                style = Theme.style("syntax.assign")
            elif token.startswith(('"', "'")):
                style = Theme.style("syntax.string")
            elif UiPrinter.RECORD_TOKEN_RE.fullmatch(token):
                style = Theme.style("syntax.number")
            elif token in {";", ","}:
                style = "ansibrightblack"
            else:
                style = Theme.style("syntax.ident")
            segments.append((style, token))
        return segments or [(fallback_style, text)]

    def memory_segments(self, text: str) -> list[tuple[str, str]]:
        segments = []
        for line in text.splitlines() or [""]:
            if line.startswith(("goal:", "check:")):
                segments.append(("ansimagenta", line))
            elif line in {"summary:", "plan:", "known:"}:
                segments.append(("ansicyan", line))
            elif line.lstrip().startswith("- [x]"):
                segments.append(("ansigreen", line))
            elif line.lstrip().startswith("- [~]"):
                segments.append(("ansiyellow", line))
            elif line.lstrip().startswith("- [-]"):
                segments.append(("ansired", line))
            elif line.lstrip().startswith("+ "):
                segments.append(("ansigreen", line))
            else:
                segments.append(("fg:default", line))
            segments.append(("", "\n"))
        return segments

    @classmethod
    def pygments_style(cls, token_type: Any) -> str:
        style = Theme.pygments_style()
        if style is None or Token is None:
            return "fg:default"
        if token_type in Token.Text.Whitespace:
            return "fg:default"
        if token_type in Token.Name.Builtin:
            return Theme.style("syntax.builtin")
        definition = style.style_for_token(token_type)
        color = definition.get("color")
        default_hex = Theme.style("syntax.default_hex")
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
        try:
            tokens = lexer.get_tokens(code_text)
        except Exception:  # noqa: BLE001 - third-party lexer execution must degrade to plain rendering.
            return None

        lines: list[list[tuple[str, str]]] = [[]]
        for token_type, value in tokens:
            style = self.pygments_style(token_type)
            parts = value.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append((style, part))
        return lines

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
                background = Theme.style("diff.added.bg")
                number(None, new_line, background)
                content_hl = hl_by_index.get(index) or [(Theme.style("diff.added.fg"), line[1:])]
                append_hl("+", "ansigreen", content_hl, suffix, background)
                new_line = None if new_line is None else new_line + 1
            elif line.startswith("-"):
                background = Theme.style("diff.removed.bg")
                number(old_line, None, background)
                append_hl("-", "ansired", [(Theme.style("diff.removed.fg"), line[1:])], suffix, background)
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
        self.lock = threading.Lock()
        self.timer: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        with self.lock:
            self.active, self.rendered_lines, self.rendered_rows, self.text = True, 0, [], ""
            self.started_at = time.monotonic()
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

    def render(self) -> None:
        if not self.active:
            return
        rows: list[list[tuple[str, str]]] = [[("ansibrightblack", line)] for line in self.frame_lines()]
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

    def frame_lines(self) -> list[str]:
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        body = [line.expandtabs(4) for line in self.text.replace("\r", "\n").splitlines()[-self.HEIGHT :]]
        label = Text.elapsed_since(self.started_at, precise=True)
        # `limit` leaves a column of slack so a full-width line cannot auto-wrap and desync the
        # cursor-up math in render().
        limit = max(1, width - get_cwidth(LogBlock.prefix(2, LogEdge.CONTINUE)) - 1)

        def clip(line: str) -> str:
            return Text.clip_width(line, limit)

        # Always emit a status row so the frame is visible even before any output arrives.
        status = f"output · {label}" if body else f"running… {label}"
        lines = [LogLine(status, role=LogRole.META, edge=LogEdge.BRANCH)]
        lines.extend(LogLine("", clip(line), LogRole.OUTPUT, LogEdge.CONTINUE) for line in body)
        return str(LogBlock.hierarchy(None, lines)).splitlines()


class StatusBar:
    """Show what the agent is doing now, on a timer thread, owning none of it.

    Every value displayed is read from session state the engine already maintains. It is a view: it
    never blocks a turn, and must never become the reason a piece of state exists.

    It writes to stderr, and only when stderr is a terminal, so the repainting line stays out of piped
    transcripts and clear of the completed output on stdout. It redraws in place and erases on stop,
    leaving nothing in scrollback.
    """

    INTERVAL: ClassVar[float] = 0.2
    RETRY_NOTICE_DURATION: ClassVar[float] = 2.0
    INDEX_SPINNER: ClassVar[tuple[str, ...]] = ("~", "/", "-", "\\", "|")
    ROLE_KEYS: ClassVar[tuple[str, ...]] = ("provider", "reason", "mcp", "ctx", "update", "index", "warn")
    # The working sweep: one crest crossing the line every couple of seconds. `SWEEP_FALLOFF` sets
    # its half width as a fraction of the line (5.0 → a fifth), wide enough that it drifts rather
    # than blinks. Bands and levels quantize the gradient and the crest; see `sweep_fragments`.
    SWEEP_CYCLES_PER_SEC: ClassVar[float] = 0.55
    SWEEP_FALLOFF: ClassVar[float] = 5.0
    SWEEP_BANDS: ClassVar[int] = 10
    SWEEP_LEVELS: ClassVar[int] = 10

    @classmethod
    def role_style(cls, role: str) -> str:
        return Theme.style("status." + role) if role in cls.ROLE_KEYS else Theme.style("status.base")

    def __init__(self, session: Session):
        self.session = session
        self.started_at = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rendered = False
        self.output = create_output(sys.stderr)
        self.seen_retry_count = session.state.model_retry_count
        self.retry_notice_until = 0.0

    def start(self, *, reset: bool = True) -> None:
        if self.thread is not None or not sys.stderr.isatty():
            return
        self.begin(reset=reset)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def begin(self, *, reset: bool = True) -> None:
        if reset or not self.started_at:
            self.started_at = time.monotonic()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        self.clear()

    def is_running(self) -> bool:
        return self.thread is not None

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(self.display_fragments(active=True)), output=self.output, end="", flush=True)
            self.rendered = True
            self.stop_event.wait(self.INTERVAL)

    def clear(self) -> None:
        if self.rendered:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.flush()
            self.rendered = False

    def display_fragments(self, *, active: bool) -> StyleAndTextTuples:
        if not active:
            return self.fragments(sweep=False, show_elapsed=False)
        return self.fragments(sweep=True, show_elapsed=True)

    def retry_notice_active(self) -> bool:
        now = time.monotonic()
        count = self.session.state.model_retry_count
        if count != self.seen_retry_count:
            self.seen_retry_count = count
            self.retry_notice_until = now + self.RETRY_NOTICE_DURATION
        return self.retry_notice_until > now

    def model_attempt_status(self) -> str:
        attempt = self.session.state.current_model_attempt
        return f"attempt {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else ""

    def retry_status(self) -> str:
        if not self.retry_notice_active():
            return ""
        attempt = self.session.state.current_model_attempt
        text = f"retrying {attempt}/{MODEL_REQUEST_RETRIES + 1}" if attempt > 1 else "retrying"
        reason = self.session.state.model_retry_reason
        return text + (" · " + reason if reason else "")

    def fragments(self, *, sweep: bool, show_elapsed: bool) -> StyleAndTextTuples:
        entries = self.entries(show_elapsed=show_elapsed)
        text = " | ".join(text for text, _ in entries)
        columns = shutil.get_terminal_size((120, 20)).columns
        if get_cwidth(text) >= columns:
            if sweep:
                return self.sweep_fragments(Text.clip_width(text, columns - 1))
            # Idle: clip per segment so the role colors survive instead of the whole line
            # collapsing to one status.base tone (a colorless white bar in a narrow pane).
            return self.clip_fragments(self.styled_fragments(entries), columns - 1)
        return self.sweep_fragments(text) if sweep else self.styled_fragments(entries)

    def entries(self, *, show_elapsed: bool) -> list[tuple[str, str]]:
        provider = self.session.config.provider
        model = provider.model.rsplit("/", 1)[-1] or "(no model)"
        reason = provider.reasoning
        parts = [(self.session.config.active_provider + "/" + model, "provider"), (reason, "reason")]

        mcp_status = self.mcp_status()
        if mcp_status:
            parts.append((mcp_status, "mcp"))
        skill_count = len(self.session.skills.skills) if self.session.skills else 0
        if skill_count:
            parts.append((f"skills {skill_count}", "mcp"))
        running_jobs = len(self.session.running_jobs())
        if running_jobs:
            parts.append((f"jobs {running_jobs}", "warn"))
        usage = self.session.usage
        if usage.last_prompt_tokens:
            # The provider-reported tokens of the last request are the display truth; the estimate
            # (state.context_percent) stays as the fallback before any request exists. This only
            # renders; compaction keeps triggering on the estimate (see DESIGN.md, context.py).
            ctx_percent = min(100, usage.last_prompt_tokens * 100 // ContextManager(self.session).request_token_budget())
        else:
            ctx_percent = self.session.state.context_percent
        ctx_text = "ctx " + str(ctx_percent) + "%"
        if usage.last_prompt_tokens:
            ctx_text += " · cache " + str(usage.last_cached_prompt_tokens * 100 // usage.last_prompt_tokens) + "%"
        parts.append((ctx_text, "ctx"))
        update_status = self.update_status()
        if update_status:
            parts.append((update_status, "update"))
        index_status = self.index_status()
        if index_status:
            parts.append(("index" + index_status, "index"))
        if self.session.settings.yolo:
            parts.append(("yolo", "warn"))
        if show_elapsed:
            turn_step = self.session.state.turn_step
            max_steps = self.session.settings.max_steps
            # Only meaningful near the cap, where the turn is about to be cut off; hidden while far from it.
            if turn_step * 5 >= max_steps * 4:
                parts.append((f"step {turn_step}/{max_steps}", "warn"))
            if retry_status := self.retry_status():
                parts.append((retry_status, "warn"))
            elif attempt_status := self.model_attempt_status():
                parts.append((attempt_status, "warn"))
        return parts

    def styled_fragments(self, entries: list[tuple[str, str]]) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for index, (text, role) in enumerate(entries):
            if index:
                fragments.append((Theme.style("status.sep"), " | "))
            fragments.append((self.role_style(role), text))
        return fragments or [("", "")]

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

    def sweep_fragments(self, text: str) -> StyleAndTextTuples:
        """Paint the working status line as a crest travelling over a quiet gradient.

        Both the gradient and the crest are quantized, so neighbouring cells share one style string.
        A continuous per-cell color costs an escape sequence per column every frame and mints a
        style string prompt-toolkit caches forever; in bands the renderer emits one escape per run
        and the set of strings stays small. The steps are far finer than the eye resolves over a
        crest this wide, so the motion still reads as continuous.
        """
        if not text:
            return [("", "")]
        width = max(1, len(text) - 1)
        sweep = (time.monotonic() * self.SWEEP_CYCLES_PER_SEC) % 1.0
        bases = Theme.ramp("status.sweep.start", "status.sweep.end", self.SWEEP_BANDS)
        crest = Theme.rgb(Theme.style("status.sweep.crest"))
        fragments: StyleAndTextTuples = []
        for index, char in enumerate(text):
            ratio = index / width
            base = bases[round(ratio * (self.SWEEP_BANDS - 1))]
            level = round(max(0.0, 1.0 - abs(ratio - sweep) * self.SWEEP_FALLOFF) ** 2 * (self.SWEEP_LEVELS - 1))
            fragments.append((base if not level else Theme.mix(Theme.rgb(base), crest, level / (self.SWEEP_LEVELS - 1)), char))
        return fragments

    def index_status(self) -> str:
        if self.session.state.code_index_error:
            return CodeIndex.label("error")
        if self.session.state.code_index_refreshing:
            notice = self.session.state.code_index_notice or "syncing"
            return self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)] if notice in {"syncing", "updating"} else notice
        return CodeIndex.label(self.session.state.code_index_status)

    def update_status(self) -> str:
        update = self.session.update
        if update.checking:
            return "update..."
        return "update " + update.latest if update.newer_than(__version__) else ""

    def mcp_status(self) -> str:
        if self.session.mcp is None:
            return ""
        configs = self.session.mcp.parse_configs()
        if not configs:
            return ""
        status = self.session.mcp.discovery_status
        if status == "discovering":
            spinner = self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)]
            loaded, total = self.session.mcp.discovery_progress()
            return f"mcp {loaded}/{total}{spinner}"
        if status == "error":
            return "mcp err"
        if status != "ready":
            return ""
        # "!" flags that the tools index overflowed the cap and some tools are hidden.
        return f"mcp {len(self.session.mcp.tools)}{'!' if self.session.mcp.index_truncated else ''}"
