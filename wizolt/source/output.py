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
        if draft.total_lines == 0 and not draft.spans:
            return "\n".join([self._open_tag(key), "(empty file)", self._close_tag()])
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
    parts: tuple[str | SourceBlock, ...]

    @classmethod
    def of(cls, value: str | ToolOutput) -> ToolOutput:
        """Normalize an ordinary tool result to the same structured abstraction."""
        return value if isinstance(value, ToolOutput) else cls(value, (value,))

    @classmethod
    def rendered(cls, parts: Sequence[str | SourceBlock]) -> ToolOutput:
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
        return "\n".join(part.render(next(key_iter, None) or "") if isinstance(part, SourceBlock) else part for part in self.parts)

    def project(self, *, max_tokens: int, estimate: Callable[[str], int]) -> ToolOutput:
        """Clip source blocks so the model text fits `max_tokens`; literal parts are kept whole.

        The budget covers the whole result rather than each block, because source-bearing output
        skips the generic character bounding downstream: a batched Read would otherwise return one
        full budget per file. Each block takes an equal share of what is left and returns whatever
        it does not use. A block that does not fit its share is split into a visible head and a
        visible tail; the omitted middle is not part of the returned block, so it cannot be
        targeted by guessing its line numbers. Projection is pure and key-independent, so it is
        safe to run before view ids are allocated.
        """
        if not self.has_source or estimate(self.render([None] * len(self.drafts))) <= max_tokens:
            return self
        literal = sum(estimate(part) if isinstance(part, str) else 0 for part in self.parts)
        remaining = max_tokens - literal
        if remaining <= 0:
            # Literal parts alone exhaust the budget; keep the head of each block as evidence.
            remaining = max(1, max_tokens // 2)
        pending = sum(1 for part in self.parts if isinstance(part, SourceBlock))
        parts: list[str | SourceBlock] = []
        for part in self.parts:
            if not isinstance(part, SourceBlock):
                parts.append(part)
                continue
            budget = max(1, remaining // pending)
            pending -= 1
            size = estimate(part.render())
            if size <= budget:
                parts.append(part)
                remaining = max(0, remaining - size)
                continue
            clipped = _clip(part, budget, estimate)
            clipped_size = estimate(clipped.render())
            remaining = max(0, remaining - clipped_size)
            parts.append(replace(clipped, estimated_tokens=size, omitted_tokens=max(0, size - clipped_size), budget_tokens=max_tokens))
        return ToolOutput(self.retained_text, tuple(parts))


# (span index, offset in span, marker, line): one rendered row, kept addressable while clipping.
_Row = tuple[int, int, str, str]


def _clip(block: SourceBlock, budget: int, estimate: Callable[[str], int]) -> SourceBlock:
    """Split one block into head spans then tail spans that together fit `budget`.

    Lines are consumed from the front until the budget is met, then from the back. A single
    oversized line cannot be split and stays in the head. The middle is dropped entirely, so the
    returned spans are only the visible head and tail groups.
    """
    draft = block.draft
    rows: list[_Row] = []
    for span_index, span in enumerate(draft.spans):
        for offset, line in enumerate(span.lines):
            marker = block.markers[len(rows)] if len(rows) < len(block.markers) else ""
            rows.append((span_index, offset, marker, line))
    width = max(1, len(str(max(span.end for span in draft.spans))))

    def cost(row: _Row) -> int:
        span_index, offset, marker, line = row
        return estimate(f"{marker}{draft.spans[span_index].start + offset:>{width}} | {line.rstrip(chr(10))}")

    head: list[_Row] = []
    used = 0
    for row in rows:
        if used + cost(row) > budget and head:
            break
        used += cost(row)
        head.append(row)
    tail: list[_Row] = []
    tail_used = 0
    for row in reversed(rows[len(head) :]):
        if tail_used + cost(row) > budget - used and tail:
            break
        tail_used += cost(row)
        tail.append(row)
    tail.reverse()
    chosen = [*head, *tail] or rows[:1]  # never an empty view
    if len(chosen) == len(rows):
        return block  # everything fit after all (estimates differ slightly); nothing is omitted
    spans: list[SourceSpan] = []
    markers: list[str] = []
    head_spans = 0
    position = 0
    for span_index, group in _group(chosen):
        head_spans += position + len(group) <= len(head)
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
