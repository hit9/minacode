"""Interactive view state machines and fragments for the prompt-toolkit application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, TypeVar

from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text

from wizolt.base import SELECTION_BACK, SELECTION_FREE_TEXT, Text
from wizolt.render import UiPrinter, WizoltMarkdown, markdown_console
from wizolt.tools.ask import AskSpec

TUI_MODAL_PENDING = object()
ViewLine = TypeVar("ViewLine")


@dataclass
class TabbedViewState:
    titles: tuple[str, ...]
    tab: int = 0
    scroll: int = 0

    def switch(self, delta: int) -> None:
        self.tab = (self.tab + delta) % len(self.titles)
        self.scroll = 0

    def scroll_by(self, delta: int) -> None:
        self.scroll = max(0, self.scroll + delta)

    def visible(self, lines: list[ViewLine], height: int) -> list[ViewLine]:
        self.scroll = min(self.scroll, max(0, len(lines) - height))
        return lines[self.scroll : self.scroll + height]


@dataclass
class DiffViewState:
    REFRESH: ClassVar[object] = object()

    class Mode(Enum):
        LIST = auto()
        FILE = auto()

    view: TabbedViewState
    mode: Mode = Mode.LIST
    file: int = 0

    def reset(self) -> None:
        self.mode = self.Mode.LIST
        self.file = 0
        self.view.scroll = 0

    def switch_tab(self, delta: int) -> None:
        self.view.switch(delta)
        self.reset()

    def move_file(self, delta: int, count: int) -> None:
        if count:
            self.file = (self.file + delta) % count

    def clamp_file(self, count: int) -> None:
        self.file = self.file % count if count else 0

    def open_file(self, count: int) -> None:
        if self.mode is self.Mode.LIST and count:
            self.mode = self.Mode.FILE
            self.view.scroll = 0

    def close_file(self) -> None:
        if self.mode is self.Mode.FILE:
            self.mode = self.Mode.LIST
            self.view.scroll = 0

    def handle_key(self, key: str, file_count: int, viewport: int) -> Any:
        if key in {"q", "c-c"}:
            return None
        if key == "escape":
            if self.mode is self.Mode.LIST:
                return None
            self.close_file()
        elif key in {"down", "j", "up", "k"}:
            delta = 1 if key in {"down", "j"} else -1
            if self.mode is self.Mode.LIST and file_count:
                self.move_file(delta, file_count)
            elif self.mode is self.Mode.FILE:
                self.view.scroll_by(delta)
        elif key in {"h", "l", "tab"}:
            self.switch_tab(1 if key in {"l", "tab"} else -1)
        elif key == "right" and self.mode is self.Mode.LIST:
            self.switch_tab(1)
        elif key == "left":
            if self.mode is self.Mode.FILE:
                self.close_file()
            else:
                self.switch_tab(-1)
        elif key == "enter" and self.mode is self.Mode.LIST and file_count:
            self.open_file(file_count)
        elif self.mode is self.Mode.FILE and key in {"pagedown", "pageup", "c-d", "c-u"}:
            distance = max(1, viewport if key in {"pagedown", "pageup"} else viewport // 2)
            self.view.scroll_by(distance if key in {"pagedown", "c-d"} else -distance)
        elif key in {"g", "G"}:  # less-style: g→top, G→bottom
            if self.mode is self.Mode.LIST and file_count:
                self.file = 0 if key == "g" else file_count - 1
            elif self.mode is self.Mode.FILE:
                self.view.scroll = 0 if key == "g" else 10**9  # clamped to the last page on render
        elif key == "r":
            self.reset()
            return self.REFRESH
        return TUI_MODAL_PENDING


@dataclass
class SegmentLogViewState:
    """List/detail state for the compaction log (`/compact log`). Same shape as DiffViewState
    without tabs or refresh: the stored segments are a closed set while the viewer is open, since
    only a compaction appends one and none can run during a modal."""

    class Mode(Enum):
        LIST = auto()
        DETAIL = auto()

    mode: Mode = Mode.LIST
    selected: int = 0
    scroll: int = 0

    def move(self, delta: int, count: int) -> None:
        if count:
            self.selected = (self.selected + delta) % count

    def open(self, count: int) -> None:
        if self.mode is self.Mode.LIST and count:
            self.mode = self.Mode.DETAIL
            self.scroll = 0

    def close(self) -> None:
        if self.mode is self.Mode.DETAIL:
            self.mode = self.Mode.LIST
            self.scroll = 0

    def visible(self, lines: list[ViewLine], height: int) -> list[ViewLine]:
        self.scroll = min(max(0, self.scroll), max(0, len(lines) - height))
        return lines[self.scroll : self.scroll + height]

    def handle_key(self, key: str, count: int, viewport: int) -> Any:
        if key in {"q", "c-c"}:
            return None
        if key == "escape":
            if self.mode is self.Mode.LIST:
                return None
            self.close()
        elif key in {"down", "j", "up", "k"}:
            delta = 1 if key in {"down", "j"} else -1
            if self.mode is self.Mode.LIST:
                self.move(delta, count)
            else:
                self.scroll = max(0, self.scroll + delta)
        elif key in {"enter", "right", "l"} and self.mode is self.Mode.LIST:
            self.open(count)
        elif key in {"left", "h"}:
            self.close()
        elif self.mode is self.Mode.DETAIL and key in {"pagedown", "pageup", "c-d", "c-u"}:
            distance = max(1, viewport if key in {"pagedown", "pageup"} else viewport // 2)
            self.scroll = max(0, self.scroll + (distance if key in {"pagedown", "c-d"} else -distance))
        elif key in {"g", "G"}:  # less-style: g→top, G→bottom
            if self.mode is self.Mode.LIST:
                self.selected = 0 if key == "g" else max(0, count - 1)
            else:
                self.scroll = 0 if key == "g" else 10**9  # clamped to the last page on render
        return TUI_MODAL_PENDING


@dataclass
class ChoiceViewState:
    FREE_TEXT: ClassVar[str] = "\x00free_text"

    choices: tuple[str, ...]
    labels: dict[str, str]
    disabled: set[str]
    query: str = ""
    selected: int = 0
    searching: bool = False
    # Rows drawn at once, 0 for all of them. A list long enough to need this is drawn through a
    # viewport that follows the selection, because the inline modal grows to fit its content and a
    # forty-row list would push the rest of the screen out of the way to show rows nobody asked for.
    max_rows: int = 0

    def visible(self) -> tuple[str, ...]:
        if not self.query:
            return self.choices
        needle = self.query.lower()
        visible: list[str] = []
        header = ""
        section: list[str] = []
        for choice in self.choices:
            if choice in self.disabled:
                if section:
                    visible.extend(([header] if header else []) + section)
                header, section = choice, []
            elif needle in (choice + " " + self.labels.get(choice, choice)).lower():
                section.append(choice)
        if section:
            visible.extend(([header] if header else []) + section)
        return tuple(visible)

    def enabled(self) -> tuple[str, ...]:
        return tuple(choice for choice in self.visible() if choice not in self.disabled)

    def clamp(self, options: tuple[str, ...] | None = None) -> tuple[str, ...]:
        options = options if options is not None else self.enabled()
        self.selected = min(max(self.selected, 0), len(options) - 1) if options else 0
        return options

    def move(self, delta: int) -> None:
        options = self.enabled()
        if options:
            self.selected = min(max(self.selected + delta, 0), len(options) - 1)

    def set_query(self, query: str) -> None:
        self.query = query
        self.selected = 0

    def window(self, visible: tuple[str, ...], options: tuple[str, ...]) -> tuple[int, int]:
        """The half-open row range of `visible` to draw, centred on the selection.

        The whole list when it fits or when no cap is set. The selection is clamped to the middle of
        the viewport rather than to its edges, so moving through a long list scrolls it instead of
        walking the cursor to the bottom and stopping."""
        if self.max_rows <= 0 or len(visible) <= self.max_rows:
            return 0, len(visible)
        target = visible.index(options[self.selected]) if options else 0
        start = min(max(0, target - self.max_rows // 2), len(visible) - self.max_rows)
        return start, start + self.max_rows

    def selected_choice(self) -> str | None:
        options = self.clamp()
        return options[self.selected] if options else None

    def fragments(
        self,
        title: str,
        preview_fn: Callable[[str], StyleAndTextTuples | str] | None = None,
        label_fn: Callable[[str], StyleAndTextTuples] | None = None,
    ) -> StyleAndTextTuples:
        """The list as fragments. `label_fn` styles one row's label in pieces instead of printing it
        flat, for a list whose rows carry more than one kind of thing (the Ctrl-O browser's key,
        tool name, and arguments). The row's own style wins where it has one, so a selected row
        stays a solid bar instead of being repainted part by part."""
        visible = self.visible()
        options = self.clamp()
        suffix = (" /" + self.query) if self.query else ""
        if self.query and not self.searching:
            suffix += " (filtered)"
        # The break after the header separates it from the rows, so the title reads as a title
        # instead of as the first item of the list under it. The title takes the same two-column
        # indent as everything else in the modal and everything a command prints; it was the one
        # line at column zero.
        #
        # The gap between the modal and whatever was printed above it is the container's job
        # (TuiApp's modal_region draws it for every non-exclusive modal), not this view's, so no
        # view hard-codes its own leading break.
        parts: StyleAndTextTuples = []
        parts += [
            ("class:choice.title", ("  " + title if title else "") + suffix + "\n"),
            ("class:choice.disabled", "  j/k move, / search, Esc/q back/cancel\n"),
            ("", "\n"),
        ]
        if self.query and not options:
            return [*parts, ("class:choice.disabled", "  no matches\n")]
        start, end = self.window(visible, options)
        number = 0
        for index, choice in enumerate(visible):
            label = self.labels.get(choice, choice)
            if choice in self.disabled:
                if start <= index < end:
                    parts.append(("class:choice.disabled", "  " + label + "\n"))
                continue
            number += 1
            # Numbering runs over the whole list, not the window: a row keeps the same number
            # whether or not the viewport currently shows it.
            if not (start <= index < end):
                continue
            selected = number - 1 == self.selected
            if selected:
                parts.append(("[SetCursorPosition]", ""))
            style = "class:choice.selected" if selected else ""
            prefix = ("> " if selected else "  ") + f"{number:2d}. "
            if label_fn is not None:
                parts.append((style, prefix))
                # The selected row stays a solid reverse bar: composing the part colours into it
                # would repaint the bar in each part's colour rather than highlight the row.
                # Indexed rather than unpacked: a fragment may carry a third mouse-handler element,
                # and this row only ever wants the style and the text.
                parts.extend((style or fragment[0], fragment[1]) for fragment in (label_fn(choice) or [("", label)]))
                parts.append((style, "\n"))
            elif match := UiPrinter.MCP_STATUS_RE.search(label):
                parts.append((style, prefix + label[: match.start()]))
                marker_style = (style + " class:choice.status." + match.group(1)).strip()
                parts.append((marker_style, "●"))
                parts.append((style, label[match.start() + 1 :] + "\n"))
            else:
                parts.append((style, prefix + label + "\n"))
        if end - start < len(visible):
            parts.append(("class:choice.disabled", f"  showing {start + 1}-{end} of {len(visible)}\n"))
        if preview_fn and options:
            preview = preview_fn(options[self.selected])
            if isinstance(preview, str):
                # A plain-text preview: every line takes the rail and the preview style.
                preview = [("class:choice.preview", "  │ " + line + "\n") for line in preview.replace("\\n", "\n").splitlines()]
            if preview:
                parts.append(("class:choice.disabled", "  ──────────────────────────────────\n"))
                parts.extend(preview)
        if self.searching:
            parts.append(("", "/" + self.query))
        return parts

    def handle_key(self, key: str, data: str = "") -> Any:
        if self.searching and key not in {"enter", "escape", "backspace", "c-h"}:
            text = data if key == "any" else key
            if len(text) == 1 and text not in "\r\n":
                self.set_query(self.query + text)
        elif key in {"j", "down"} and not self.searching:
            self.move(1)
        elif key in {"k", "up"} and not self.searching:
            self.move(-1)
        elif key in {"g", "G"} and not self.searching:  # less-style: g→first, G→last
            self.move(-len(self.enabled()) if key == "g" else len(self.enabled()))
        elif key == "/":
            self.searching = True
            self.set_query("")
        elif key in {"backspace", "c-h"} and self.searching:
            self.set_query(self.query[:-1])
        elif key == "escape":
            if self.searching:
                self.searching = False
            elif self.query:
                self.set_query("")
            else:
                return SELECTION_BACK
        elif key == "q" and not self.searching:
            return SELECTION_BACK
        elif key == "enter":
            if self.searching:
                self.searching = False
            elif (choice := self.selected_choice()) is not None:
                return SELECTION_FREE_TEXT if choice == self.FREE_TEXT else choice
        elif key == "c-c":
            return KeyboardInterrupt()
        elif key.isdigit() and not self.searching:
            number = int(key)
            options = self.enabled()
            if 1 <= number <= len(options):
                self.selected = number - 1
        return TUI_MODAL_PENDING


# Batch-level Ask modal results on top of ChoiceViewState's own SELECTION_* returns: ASK_DONE means
# every question was answered; (ASK_FREE_TEXT, index) means the question at `index` dropped to the
# shared input row and the modal should reopen for the rest.
ASK_DONE = object()
ASK_FREE_TEXT = object()


@dataclass
class AskViewState:
    """The Ask tool's multi-question modal: one ChoiceViewState page per question, followed by
    the selected option's rich markdown preview. Enter advances to the next question and submits
    on the last; Tab cycles pages; `n` edits a note appended to the answer; Esc cancels the whole
    batch. A question without choices is a single "Type freely..." page that reports
    ASK_FREE_TEXT so the caller drops to the shared input row."""

    specs: list[AskSpec]
    pages: list[ChoiceViewState]
    active: int = 0
    picked: list[str | None] = field(default_factory=list)
    notes: dict[int, str] = field(default_factory=dict)
    notes_mode: bool = False
    note_buffer: str = ""

    def __post_init__(self) -> None:
        if not self.picked:
            self.picked = [None] * len(self.specs)

    @classmethod
    def build(cls, specs: list[AskSpec]) -> AskViewState:
        """One page per question; the free-text escape hatch is always offered and a recommended
        choice is pre-selected and marked."""
        pages: list[ChoiceViewState] = []
        for spec in specs:
            choices = list(spec.choices) if spec.choices else []
            labels: dict[str, str] = {}
            current = ""
            if spec.recommended is not None and spec.choices and 0 <= spec.recommended < len(spec.choices):
                current = spec.choices[spec.recommended]
                labels[current] = current + " (recommended)"
            choices.append(ChoiceViewState.FREE_TEXT)
            labels[ChoiceViewState.FREE_TEXT] = "Type freely..."
            pages.append(
                ChoiceViewState(
                    tuple(choices),
                    labels,
                    set(),
                    selected=choices.index(current) if current else 0,
                )
            )
        return cls(specs, pages)

    def preview_text(self) -> str:
        """The selected option's preview markdown, or '' when it has none."""
        spec = self.specs[self.active]
        if not spec.previews or not spec.choices:
            return ""
        choice = self.pages[self.active].selected_choice()
        if choice is None or choice not in spec.choices:
            return ""
        index = spec.choices.index(choice)
        return spec.previews[index] if index < len(spec.previews) else ""

    def fragments(self, width: int, max_height: int) -> StyleAndTextTuples:
        """Render one readable column bounded for wide terminals and wrapped by display width."""
        page = self.pages[self.active]
        content_width = max(10, min(width - 2, 120))
        title_prefix = f"({self.active + 1}/{len(self.specs)}) "
        title_rows = Text.wrap_styled(
            [("class:choice.title", title_prefix)],
            [("class:choice.title", " " * len(title_prefix))],
            [("class:choice.title", self.specs[self.active].question)],
            content_width,
        )
        # The gap between the modal and the activity region above it is the container's job
        # (TuiApp's modal_region), matching every other non-exclusive modal.
        parts: StyleAndTextTuples = []
        for row in title_rows:
            parts.extend((*row, ("", "\n")))
        parts.append(("", "\n"))
        note_rows: list[list[tuple[str, str]]] = []
        if self.notes_mode:
            note_rows = Text.wrap_styled(
                [("class:choice.disabled", "notes: ")],
                [("class:choice.disabled", "       ")],
                [("class:choice.disabled", self.note_buffer)],
                content_width,
            )
        elif note := self.notes.get(self.active):
            note_rows = Text.wrap_styled(
                [("class:choice.disabled", "notes: ")],
                [("class:choice.disabled", "       ")],
                [("class:choice.disabled", note)],
                content_width,
            )
        for row in note_rows:
            parts.extend((*row, ("", "\n")))
        if not page.enabled():
            body: list[list[tuple[str, str]]] = [[("class:choice.disabled", "  no matches")]]
        else:
            body = self._option_rows(page, content_width)
            preview = self.preview_text()
            if preview:
                preview_rows = self._preview_rows(preview, max(10, content_width - 4))
                body.append([("class:choice.disabled", "  " + "─" * min(34, content_width - 2))])
                body.append([("class:choice.preview", "  │")])
                body.extend([("class:choice.preview", "  │ "), *row] for row in preview_rows)
        footer = self._footer_rows(content_width)
        query_rows = Text.wrap_styled([("", "/")], [("", " ")], [("", page.query)], content_width) if page.searching else [[]]
        # Title rows, its blank separator, the search/blank row above the footer, the footer, and
        # any note row are fixed chrome. A page that cannot fit them leaves no room for body rows.
        fixed = len(title_rows) + 1 + len(note_rows) + len(query_rows) + len(footer)
        budget = max(0, max_height - fixed)
        if len(body) > budget:
            if budget <= 0:
                body = []
            else:
                overflow = len(body) - budget + 1
                body = body[: budget - 1] + [[("class:choice.disabled", f"  … {overflow} more lines")]]
        for row in body:
            parts.extend((*row, ("", "\n")))
        for row in query_rows:
            parts.extend((*row, ("", "\n")))
        for row in footer:
            parts.extend((*row, ("", "\n")))
        return parts

    @staticmethod
    def _option_rows(page: ChoiceViewState, width: int) -> list[list[tuple[str, str]]]:
        """The page's option rows (one fragment list per row, no trailing newline), styled like
        ChoiceViewState.fragments' option block, wrapped by terminal-cell width."""
        rows: list[list[tuple[str, str]]] = []
        number = 0
        for choice in page.visible():
            label = page.labels.get(choice, choice)
            if choice in page.disabled:
                rows.extend(
                    Text.wrap_styled(
                        [("class:choice.disabled", "  ")],
                        [("class:choice.disabled", "  ")],
                        [("class:choice.disabled", label)],
                        width,
                    )
                )
                continue
            number += 1
            selected = number - 1 == page.selected
            prefix = ("> " if selected else "  ") + f"{number:2d}. "
            style = "class:choice.selected" if selected else ""
            wrapped = Text.wrap_styled(
                [(style, prefix)],
                [(style, " " * len(prefix))],
                [(style, label)],
                width,
            )
            if selected:
                wrapped[0].insert(0, ("[SetCursorPosition]", ""))
            rows.extend(wrapped)
        return rows

    @staticmethod
    def _preview_rows(markdown_text: str, panel_width: int) -> list[list[tuple[str, str]]]:
        """Render markdown to ANSI with Rich, split into physical rows, and carry styles across them.

        Preview snippets are ASCII layouts, diffs, and tables whose newlines are structural, so
        each source line gets a hard line break (Markdown folds in-paragraph newlines to spaces)."""
        hard_breaks = "\n".join(line.rstrip() + "  " for line in markdown_text.split("\n"))
        console = markdown_console(panel_width)
        with console.capture() as capture:
            console.print(WizoltMarkdown(hard_breaks))
        cleaned = UiPrinter.strip_unknown_escapes(UiPrinter.strip_trailing_pad(capture.get()))
        rows = AskViewState._ansi_lines(cleaned)
        while rows and not any(fragment[1] for fragment in rows[-1]):
            rows.pop()
        return rows

    @staticmethod
    def _ansi_lines(text: str) -> list[list[tuple[str, str]]]:
        """Split an ANSI string into one (style, text) tuple list per line. prompt_toolkit's ANSI
        parser emits a fragment per character (it walks SGR state per char), so adjacent fragments
        with the same style are merged back into runs; SGR state across a newline carries exactly
        as Rich emitted it."""
        lines: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        for fragment in to_formatted_text(ANSI(text)):
            style = fragment[0]
            chunk = fragment[1]
            assert isinstance(chunk, str)  # ANSI fragments are always (style, str)
            pieces = chunk.split("\n")
            for index, piece in enumerate(pieces):
                if index:
                    lines.append(current)
                    current = []
                if current and current[-1][0] == style:
                    text = current[-1][1]
                    assert isinstance(text, str)  # ANSI fragments are always (style, str)
                    current[-1] = (style, text + piece)
                else:
                    current.append((style, piece))
        lines.append(current)
        return lines

    def _footer_rows(self, width: int) -> list[list[tuple[str, str]]]:
        text = "↑↓/jk move · Enter select · Tab page · n note · / search · Esc cancel" if width >= 70 else "↑↓/jk · Enter · Tab · n · / · Esc"
        return Text.wrap_styled([], [], [("class:choice.disabled", text)], width)

    def handle_key(self, key: str, data: str = "") -> Any:
        """The active page's keys plus the batch-level ones: Tab cycles pages, `n` edits a note,
        Enter advances (submitting on the last page), Esc cancels the whole batch."""
        if self.notes_mode:
            if key == "escape":
                self.notes_mode = False
                self.note_buffer = ""
                return TUI_MODAL_PENDING
            if key == "enter":
                self.notes[self.active] = self.note_buffer
                self.notes_mode = False
                self.note_buffer = ""
                return TUI_MODAL_PENDING
            if key in {"backspace", "c-h"}:
                self.note_buffer = self.note_buffer[:-1]
                return TUI_MODAL_PENDING
            text = data if key == "any" else key
            if len(text) == 1 and text not in "\r\n":
                self.note_buffer += text
            return TUI_MODAL_PENDING
        page = self.pages[self.active]
        if key in {"tab", "s-tab"}:
            self.active = (self.active + (1 if key == "tab" else -1)) % len(self.specs)
            return TUI_MODAL_PENDING
        if (key == "n" or (key == "any" and data == "n")) and not page.searching:
            self.notes_mode = True
            self.note_buffer = self.notes.get(self.active, "")
            return TUI_MODAL_PENDING
        result = page.handle_key(key, data)
        if result is TUI_MODAL_PENDING:
            return TUI_MODAL_PENDING
        if result is SELECTION_BACK:
            return SELECTION_BACK  # the page unwound its own search/query layers; batch cancelled
        if result is SELECTION_FREE_TEXT:
            return (ASK_FREE_TEXT, self.active)
        if isinstance(result, str):
            self.picked[self.active] = result
            if all(p is not None for p in self.picked):
                return ASK_DONE
            self.active = next(i for i, p in enumerate(self.picked) if p is None)
            return TUI_MODAL_PENDING
        return result  # KeyboardInterrupt and anything else pass through
