"""Projection: clip source blocks to a token budget, keeping head then tail spans."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from wizolt.source.render import render_source_block, render_tool_output
from wizolt.source.values import SourceBlock, SourceSpan, SourceViewDraft, ToolOutput


def project_output(output: ToolOutput, *, max_tokens: int, estimate: Callable[[str], int]) -> ToolOutput:
    """Clip source blocks so the model text fits `max_tokens`; literal parts are kept whole.

    A block that does not fit is split into a visible head span group and a visible tail span
    group; the omitted middle is not part of the returned block. Blocks that already fit are kept
    unchanged. Projection is pure: the returned output renders to the same text with or without
    keys, so it is safe before view ids are allocated.
    """
    if not output.has_source:
        return output
    if estimate(render_tool_output(output, [None] * len(output.drafts))) <= max_tokens:
        return output
    literal = sum(estimate(part) if isinstance(part, str) else 0 for part in output.parts)
    remaining = max_tokens - literal
    if remaining <= 0:
        # Literal parts alone exhaust the budget; keep the head of each block as evidence.
        remaining = max(1, max_tokens // 2)
    # The budget covers the whole result, not each block: a batched Read or a Search spanning
    # several files would otherwise emit one full budget per file. Each block takes an equal share
    # of what is left, and a block that comes in under its share returns the rest to the pool.
    pending = sum(1 for part in output.parts if isinstance(part, SourceBlock))
    blocks: list[str | SourceBlock] = []
    for part in output.parts:
        if not isinstance(part, SourceBlock):
            blocks.append(part)
            continue
        block_budget = max(1, remaining // pending)
        pending -= 1
        size = estimate(render_source_block(part))
        if size <= block_budget:
            blocks.append(part)
            remaining = max(0, remaining - size)
            continue
        clipped = _clip_block(part, block_budget, estimate)
        clipped_size = estimate(render_source_block(clipped))
        remaining = max(0, remaining - clipped_size)
        blocks.append(
            _replace_block(
                clipped,
                estimated_tokens=size,
                omitted_tokens=max(0, size - clipped_size),
                budget_tokens=max_tokens,
            )
        )
    return ToolOutput(output.retained_text, tuple(blocks))


def _replace_block(block: SourceBlock, **changes: object) -> SourceBlock:
    from dataclasses import replace

    return replace(block, **changes)


def _clip_block(block: SourceBlock, budget_tokens: int, estimate: Callable[[str], int]) -> SourceBlock:
    """Split one block into head spans then tail spans that together fit `budget_tokens`.

    Lines are consumed from the front until the budget is met, then from the back. A single
    oversized line cannot be split and stays in the head. The middle is dropped entirely, so the
    returned spans are only the visible head and tail groups.
    """
    draft = block.draft
    markers = block.markers
    rows: list[tuple[int, int, str, str]] = []  # (span index, offset in span, marker, line)
    marker_index = 0
    for span_index, span in enumerate(draft.spans):
        for offset, line in enumerate(span.lines):
            marker = markers[marker_index] if marker_index < len(markers) else ""
            rows.append((span_index, offset, marker, line))
            marker_index += 1

    width = max(1, len(str(max(span.end for span in draft.spans))))

    def cost(row: tuple[int, int, str, str]) -> int:
        span_index, offset, marker, line = row
        number = draft.spans[span_index].start + offset
        return estimate(f"{marker}{number:>{width}} | {line.rstrip(chr(10))}")

    head: list[tuple[int, int, str, str]] = []
    used = 0
    for row in rows:
        if used + cost(row) > budget_tokens and head:
            break
        used += cost(row)
        head.append(row)
    tail: list[tuple[int, int, str, str]] = []
    tail_used = 0
    for row in reversed(rows[len(head) :]):
        if tail_used + cost(row) > budget_tokens - used and tail:
            break
        tail_used += cost(row)
        tail.append(row)
    tail.reverse()
    chosen = [*head, *tail]
    if not chosen:
        chosen = rows[:1]  # never an empty view
    if len(chosen) == len(rows):
        # Everything fit after all (estimates differ slightly); nothing is omitted.
        return block
    groups = _group_rows(chosen)
    spans: list[SourceSpan] = []
    out_markers: list[str] = []
    head_span_count = 0
    row_pos = 0
    for span_index, group in groups:
        if row_pos + len(group) <= len(head):
            head_span_count += 1
        row_pos += len(group)
        first_offset = group[0][1]
        spans.append(SourceSpan(draft.spans[span_index].start + first_offset, tuple(row[3] for row in group)))
        out_markers.extend(row[2] for row in group)
    clipped = SourceBlock(
        SourceViewDraft(draft.path, draft.display_path, draft.total_lines, tuple(spans), draft.producer),
        tuple(out_markers),
    )
    return _replace_block(clipped, bounded=True, split_span=head_span_count)


def _group_rows(rows: Sequence[tuple[int, int, str, str]]) -> list[tuple[int, list[tuple[int, int, str, str]]]]:
    """Group consecutive (span_index, offset, marker, line) rows into per-span runs."""
    groups: list[tuple[int, list[tuple[int, int, str, str]]]] = []
    for row in rows:
        span_index = row[0]
        if groups and groups[-1][0] == span_index and groups[-1][1][-1][1] + 1 == row[1]:
            groups[-1][1].append(row)
        else:
            groups.append((span_index, [row]))
    return groups
