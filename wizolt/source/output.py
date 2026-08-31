"""Structured tool output: source blocks, their model-facing text, and budget projection.

A tool's result cannot be a plain string any more: it has to carry both the full text kept under
`tr.N` and the exact source blocks that survived output bounding, because only the surviving lines
may become a view. ToolOutput is that pair, and rendering and projection are things it does to
itself -- the runner supplies the assigned keys and the budget, and nothing else needs to know how
a block is laid out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from wizolt.source.view import EDIT, READ, SEARCH, SourceSpan, SourceViewDraft


@dataclass(frozen=True)
class TextBlock:
    """A non-source output part whose omitted middle is retained under the result's tr.N key."""

    head: str
    tail: str
    estimated_tokens: int
    omitted_tokens: int
    budget_tokens: int
    note_recall: str = ""
    note_file: str = ""
    note_hint: str = ""

    def render(self) -> str:
        attrs = f'omitted="middle" max_tokens="{self.budget_tokens}" estimated_tokens="{self.estimated_tokens}" omitted_tokens="{self.omitted_tokens}"'
        for name, value in (("recall", self.note_recall), ("file", self.note_file), ("hint", self.note_hint)):
            attrs += f" {name}={_quote(value)}" if value else ""
        note = f"<bounded_output {attrs}/>"
        return "\n".join(part for part in (self.head.rstrip(), note, self.tail.lstrip()) if part)


@dataclass(frozen=True)
class SourceBlock:
    """A draft plus per-line markers (e.g. Search's `>` match / ` ` context prefix).

    `bounded` records a projection clip: the visible spans are head spans followed by tail spans,
    and `split_span` is the index of the first tail span. The omitted middle is not part of any
    registered span and cannot be targeted.
    """

    draft: SourceViewDraft
    markers: tuple[str, ...]  # one marker per line across all spans, in order
    bounded: bool = False
    estimated_tokens: int = 0
    omitted_tokens: int = 0
    budget_tokens: int = 0  # the projection budget the block was clipped to
    split_span: int = 0  # index of the first tail span when bounded
    note_recall: str = ""  # tr.N key the full retained output lives under, filled by the runner
    note_file: str = ""  # materialized asset path for the full retained output, filled by the runner
    note_hint: str = ""  # what the model should do with the omitted middle

    @classmethod
    def plain(cls, draft: SourceViewDraft) -> SourceBlock:
        """A block with no markers: everything Read, InspectCode, and Edit produce."""
        return cls(draft, ("",) * draft.line_count)

    @classmethod
    def around(cls, path: str, display_path: str, lines: Sequence[str], center: int, producer: str = EDIT) -> SourceBlock:
        """A plain block of the current lines around 0-based `center`, for a failed Edit."""
        return cls.plain(SourceViewDraft.around(path, display_path, lines, center, producer))

    def render(self, key: str = "") -> str:
        """Render this block with ordinary numbered lines.

        `key` is the assigned `view.N`; pass "" to omit the `source=` attribute, which is what the
        retained `tr.N` copy gets -- retained text is a record, not editable evidence.
        """
        draft = self.draft
        if not draft.spans:
            # An empty file is the one view that legitimately shows nothing. A block with no spans
            # over a file that has content means its producer selected nothing; render it as the
            # empty block it is rather than failing while building an error message.
            return "\n".join([self._open_tag(key), "(empty file)" if draft.total_lines == 0 else "(no lines selected)", self._close_tag()])
        width = len(str(max(span.end for span in draft.spans)))
        rows: list[str] = []
        for index, (span, offset, line) in enumerate(self._rows()):
            marker = self.markers[index] if index < len(self.markers) else ""
            rows.append(f"{marker}{span.start + offset:>{width}} | {line.rstrip(chr(10))}")
        if not self.bounded:
            return "\n".join([self._open_tag(key), *rows, self._close_tag()])
        head = sum(len(span.lines) for span in draft.spans[: self.split_span])
        return "\n".join([self._open_tag(key), *rows[:head], self._note(), *rows[head:], self._close_tag()])

    def _rows(self) -> list[tuple[SourceSpan, int, str]]:
        return [(span, offset, line) for span in self.draft.spans for offset, line in enumerate(span.lines)]

    def _note(self) -> str:
        attrs = f'omitted="middle" max_tokens="{self.budget_tokens}" estimated_tokens="{self.estimated_tokens}" omitted_tokens="{self.omitted_tokens}"'
        for name, value in (("recall", self.note_recall), ("file", self.note_file), ("hint", self.note_hint)):
            attrs += f' {name}="{value}"' if value else ""
        return f"<bounded_output {attrs}/>"

    def _open_tag(self, key: str) -> str:
        draft = self.draft
        source = f" source={_quote(key)}" if key else ""
        ranges = draft.ranges_label()
        if draft.producer == READ:
            return f"<Read path={_quote(draft.display_path)}{source} lines={_quote(ranges)} total_lines={draft.total_lines}>"
        if draft.producer == SEARCH:
            return f"<file path={_quote(draft.display_path)}{source} lines={_quote(ranges)}>"
        return f"<source path={_quote(draft.display_path)}{source} lines={_quote(ranges)}>"

    def _close_tag(self) -> str:
        return {READ: "</Read>", SEARCH: "</file>"}.get(self.draft.producer, "</source>")

    def overhead(self, estimate: Callable[[str], int]) -> int:
        """What this block costs before a single line of source: its tags and its omission note.

        Clipping has to pay for these out of the block's budget. Counting only the numbered rows
        would let every clipped block quietly exceed its share by its own wrapper.
        """
        return estimate("\n".join((self._open_tag("view.000"), self._note(), self._close_tag())))


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class ToolOutput:
    """Structured tool output: the full plain retained text plus model-facing parts.

    A part is either literal text or a source block. Source blocks are rendered with their
    assigned `view.N` key on the main thread; the retained text carries ordinary line numbers and
    no `source=` attribute.
    """

    retained_text: str
    parts: tuple[str | TextBlock | SourceBlock, ...]

    @classmethod
    def of(cls, value: str | ToolOutput) -> ToolOutput:
        """Normalize an ordinary tool result to the same structured abstraction."""
        return value if isinstance(value, ToolOutput) else cls(value, (value,))

    @classmethod
    def rendered(cls, parts: Sequence[str | TextBlock | SourceBlock]) -> ToolOutput:
        """An output whose retained text is its own unkeyed rendering: what tools return."""
        output = cls("", tuple(parts))
        return cls(output.render([None] * len(output.drafts)), tuple(parts))

    @property
    def has_source(self) -> bool:
        return any(isinstance(part, SourceBlock) for part in self.parts)

    @property
    def drafts(self) -> tuple[SourceViewDraft, ...]:
        return tuple(part.draft for part in self.parts if isinstance(part, SourceBlock))

    def render(self, keys: Sequence[str | None]) -> str:
        """Model-facing text, taking the assigned `view.N` for each source part in order.

        A None entry renders the unkeyed form. This output's own `retained_text` is not used.
        """
        key_iter = iter(keys)
        return "\n".join(
            part.render(next(key_iter, None) or "") if isinstance(part, SourceBlock) else part.render() if isinstance(part, TextBlock) else part
            for part in self.parts
        )

    def project(self, *, max_tokens: int, estimate: Callable[[str], int]) -> ToolOutput:
        """Clip source blocks and literal text so the model text fits `max_tokens`.

        The budget covers the whole result rather than each block, because source-bearing output
        skips the generic character bounding downstream: a batched Read would otherwise return one
        full budget per file. Each block takes an equal share of what is left and returns whatever
        it does not use. A block that does not fit its share is split into a visible head and a
        visible tail; the omitted middle is not part of the returned block, so it cannot be
        targeted by guessing its line numbers. Projection is pure and key-independent, so it is
        safe to run before view ids are allocated.

        The budget is a target, not a hard cap: `estimate` is approximate, and the runner adds the
        recall key, asset path, and hint to each omission note afterwards, which it cannot know
        before finding out that anything was clipped. The overshoot is bounded by one note per
        clipped block.
        """
        if not self.has_source or estimate(self.render([None] * len(self.drafts))) <= max_tokens:
            return self
        remaining = max_tokens
        pending = len(self.parts)
        parts: list[str | TextBlock | SourceBlock] = []
        for part in self.parts:
            budget = max(1, remaining // pending)
            pending -= 1
            size = estimate(part.render() if isinstance(part, TextBlock) else part.render() if isinstance(part, SourceBlock) else part)
            if size <= budget:
                parts.append(part)
                remaining = max(0, remaining - size)
                continue
            clipped = (
                _clip(part, budget, estimate)
                if isinstance(part, SourceBlock)
                else _clip_text(part.render() if isinstance(part, TextBlock) else part, budget, max_tokens, size, estimate)
            )
            clipped_size = estimate(clipped.render())
            remaining = max(0, remaining - clipped_size)
            if isinstance(clipped, SourceBlock):
                clipped = replace(clipped, estimated_tokens=size, omitted_tokens=max(0, size - clipped_size), budget_tokens=max_tokens)
            parts.append(clipped)
        return ToolOutput(self.retained_text, tuple(parts))


def _clip_text(text: str, budget: int, max_tokens: int, size: int, estimate: Callable[[str], int]) -> TextBlock:
    """Keep a line-friendly head and tail of ordinary text inside one part's budget."""
    placeholder = TextBlock("", "", size, size, max_tokens)
    available = max(1, budget - estimate(placeholder.render()))
    # Token estimates are not necessarily character counts. Start from their observed ratio and
    # shrink until the rendered block fits; the loop is logarithmic even for a multi-megabyte diff.
    keep = min(len(text), max(1, len(text) * available // max(1, size)))
    while keep > 1:
        head_size = max(1, keep * 2 // 5)
        tail_size = max(0, keep - head_size)
        head = _head_excerpt(text, head_size)
        tail = _tail_excerpt(text, tail_size)
        clipped = TextBlock(head, tail, size, max(0, size - estimate(head) - estimate(tail)), max_tokens)
        if estimate(clipped.render()) <= budget:
            return clipped
        keep //= 2
    return TextBlock(text[:1], "", size, max(0, size - estimate(text[:1])), max_tokens)


def _head_excerpt(text: str, limit: int) -> str:
    window = text[:limit]
    snapped = window.rsplit("\n", 1)[0]
    return snapped if len(snapped) >= limit // 2 else window


def _tail_excerpt(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    window = text[-limit:]
    snapped = window.split("\n", 1)[-1]
    return snapped if len(snapped) >= limit // 2 else window


# (span index, offset in span, marker, line): one rendered row, kept addressable while clipping.
_Row = tuple[int, int, str, str]


def _clip(block: SourceBlock, budget: int, estimate: Callable[[str], int]) -> SourceBlock:
    """Split one block into head spans then tail spans that together fit `budget`.

    Lines are consumed from the front until the budget is met, then from the back. A single
    oversized line cannot be split and stays in the head. The middle is dropped entirely, so the
    returned spans are only the visible head and tail groups.
    """
    draft = block.draft
    budget = max(1, budget - block.overhead(estimate))
    width = max(1, len(str(max(span.end for span in draft.spans))))
    rows: list[_Row] = []
    costs: list[int] = []  # a row's cost is asked for repeatedly; a large file has many rows
    for span_index, span in enumerate(draft.spans):
        for offset, line in enumerate(span.lines):
            marker = block.markers[len(rows)] if len(rows) < len(block.markers) else ""
            rows.append((span_index, offset, marker, line))
            costs.append(estimate(f"{marker}{span.start + offset:>{width}} | {line.rstrip(chr(10))}"))

    head = 0  # rows taken from the front
    used = 0
    while head < len(rows) and (used + costs[head] <= budget or not head):
        used += costs[head]
        head += 1
    tail = len(rows)  # first row taken from the back
    tail_used = 0
    while tail > head and (tail_used + costs[tail - 1] <= budget - used or tail == len(rows)):
        tail_used += costs[tail - 1]
        tail -= 1
    chosen = [*rows[:head], *rows[tail:]] or rows[:1]  # never an empty view
    if len(chosen) == len(rows):
        return block  # everything fit after all (estimates differ slightly); nothing is omitted

    spans: list[SourceSpan] = []
    markers: list[str] = []
    head_spans = 0
    position = 0
    for span_index, group in _group(chosen):
        if position + len(group) <= head:
            head_spans += 1
        position += len(group)
        spans.append(SourceSpan(draft.spans[span_index].start + group[0][1], tuple(row[3] for row in group)))
        markers.extend(row[2] for row in group)
    clipped = SourceBlock(SourceViewDraft(draft.path, draft.display_path, draft.total_lines, tuple(spans), draft.producer), tuple(markers))
    return replace(clipped, bounded=True, split_span=head_spans)


def _group(rows: Sequence[_Row]) -> list[tuple[int, list[_Row]]]:
    """Group consecutive rows into per-span runs, splitting where the clip dropped lines."""
    groups: list[tuple[int, list[_Row]]] = []
    for row in rows:
        if groups and groups[-1][0] == row[0] and groups[-1][1][-1][1] + 1 == row[1]:
            groups[-1][1].append(row)
        else:
            groups.append((row[0], [row]))
    return groups
