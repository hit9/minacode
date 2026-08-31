"""Pure source-view values, rendering, range validation, and exact relocation.

A source view is an immutable, session-scoped record of source lines actually projected to the
model. Tools build drafts; the runner commits them on the main thread. This package owns the pure
parts only: it imports no runner, tool, context, or Session code, so source-producing tools stay
parallel-safe and no feature module becomes a second writer of active turn state.

The whole public surface is re-exported here, so callers keep importing from ``wizolt.source``
unchanged. The implementation is split by concern: ``values`` (value types, ids, error
categories, and view construction from lines), ``render`` (model-facing text), ``project``
(budget clipping), and ``relocate`` (edit target extraction and relocation).
"""

from wizolt.source.project import project_output
from wizolt.source.relocate import (
    insertion_witness,
    range_lines,
    relocate_target,
    relocate_witness,
    same_position,
)
from wizolt.source.render import render_source_block, render_tool_output, rendered_fresh_block
from wizolt.source.values import (
    EDIT,
    INSPECT,
    MAX_VIEW_DRIFT,
    PLANNED_EDIT_STALE,
    READ,
    SEARCH,
    SOURCE_MISSING,
    SOURCE_PATH_MISMATCH,
    SOURCE_RANGE_UNSEEN,
    SOURCE_TARGET_AMBIGUOUS,
    SOURCE_TARGET_CHANGED,
    SOURCE_TARGET_CONSUMED,
    SourceBlock,
    SourceSpan,
    SourceView,
    SourceViewDraft,
    ToolOutput,
    as_tool_output,
    fresh_context_block,
    fresh_context_draft,
    merge_ranges,
    parse_view_key,
    source_error,
    spans_from_lines,
    view_key,
    view_line,
)

__all__ = [
    "EDIT",
    "INSPECT",
    "MAX_VIEW_DRIFT",
    "PLANNED_EDIT_STALE",
    "READ",
    "SEARCH",
    "SOURCE_MISSING",
    "SOURCE_PATH_MISMATCH",
    "SOURCE_RANGE_UNSEEN",
    "SOURCE_TARGET_AMBIGUOUS",
    "SOURCE_TARGET_CHANGED",
    "SOURCE_TARGET_CONSUMED",
    "SourceBlock",
    "SourceSpan",
    "SourceView",
    "SourceViewDraft",
    "ToolOutput",
    "as_tool_output",
    "fresh_context_block",
    "fresh_context_draft",
    "insertion_witness",
    "merge_ranges",
    "parse_view_key",
    "project_output",
    "range_lines",
    "relocate_target",
    "relocate_witness",
    "render_source_block",
    "render_tool_output",
    "rendered_fresh_block",
    "same_position",
    "source_error",
    "spans_from_lines",
    "view_key",
    "view_line",
]
