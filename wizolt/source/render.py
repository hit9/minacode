"""Rendering source views and structured tool output into model-facing text."""

from __future__ import annotations

from collections.abc import Sequence

from wizolt.source.values import EDIT, READ, SEARCH, SourceBlock, SourceViewDraft, ToolOutput, _ranges_label, fresh_context_block


def _width(block: SourceBlock) -> int:
    """The number column width for a block's largest visible line number."""
    spans = block.draft.spans
    if not spans:
        return 1
    return len(str(max(span.end for span in spans)))


def render_source_block(block: SourceBlock, key: str = "") -> str:
    """Render one source block with ordinary numbered lines.

    `key` is the assigned `view.N`; pass "" to omit the `source=` attribute (retained text).
    """
    draft = block.draft
    if draft.total_lines == 0 and not draft.spans:
        return "\n".join([_open_tag(draft, key), "(empty file)", _close_tag(draft.producer)])
    width = _width(block)
    rows: list[str] = []
    marker_index = 0
    for span in draft.spans:
        for offset, line in enumerate(span.lines):
            marker = block.markers[marker_index] if marker_index < len(block.markers) else ""
            rows.append(f"{marker}{span.start + offset:>{width}} | {line.rstrip(chr(10))}")
            marker_index += 1
    open_tag = _open_tag(draft, key)
    if block.bounded:
        head = rows[: _bounded_head_rows(block)]
        tail = rows[_bounded_head_rows(block) :]
        note = (
            f'<bounded_output omitted="middle" max_tokens="{block.budget_tokens}" estimated_tokens="{block.estimated_tokens}" omitted_tokens="{block.omitted_tokens}"'
            + (f' recall="{block.note_recall}"' if block.note_recall else "")
            + (f' file="{block.note_file}"' if block.note_file else "")
            + (f' hint="{block.note_hint}"' if block.note_hint else "")
            + "/>"
        )
        return "\n".join([open_tag, *head, note, *tail, _close_tag(draft.producer)])
    return "\n".join([open_tag, *rows, _close_tag(draft.producer)])


def _bounded_head_rows(block: SourceBlock) -> int:
    """Number of rendered rows that belong to the head spans of a bounded block."""
    total = 0
    for index, span in enumerate(block.draft.spans):
        if index >= block.split_span:
            break
        total += len(span.lines)
    return total


def _open_tag(draft: SourceViewDraft, key: str) -> str:
    source_attr = f" source={_quote(key)}" if key else ""
    ranges = _ranges_label(draft.spans)
    if draft.producer == READ:
        return f"<Read path={_quote(draft.display_path)}{source_attr} lines={_quote(ranges)} total_lines={draft.total_lines}>"
    if draft.producer == SEARCH:
        return f"<file path={_quote(draft.display_path)}{source_attr} lines={_quote(ranges)}>"
    return f"<source path={_quote(draft.display_path)}{source_attr} lines={_quote(ranges)}>"


def _close_tag(producer: str) -> str:
    if producer == READ:
        return "</Read>"
    if producer == SEARCH:
        return "</file>"
    return "</source>"


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_tool_output(output: ToolOutput, keys: Sequence[str | None]) -> str:
    """Render model-facing text for a projected output.

    `keys` supplies the assigned `view.N` for each source part in order; a None entry renders the
    retained (unkeyed) form. The output's own `retained_text` is not used here.
    """
    parts: list[str] = []
    key_iter = iter(keys)
    for part in output.parts:
        if isinstance(part, SourceBlock):
            parts.append(render_source_block(part, next(key_iter, None) or ""))
        else:
            parts.append(part)
    return "\n".join(parts)


def rendered_fresh_block(path: str, display_path: str, lines: Sequence[str], center: int, producer: str = EDIT, key: str = "") -> str:
    """Render a fresh context block with a view id, for structured Edit failures."""
    return render_source_block(fresh_context_block(path, display_path, lines, center, producer), key)
