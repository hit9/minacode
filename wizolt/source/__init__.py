"""Pure source-view values, rendering, range validation, and exact relocation.

A source view is an immutable, session-scoped record of source lines actually projected to the
model. Tools build drafts; the runner commits them on the main thread. This package owns the pure
parts only: it imports no runner, tool, context, or Session code, so source-producing tools stay
parallel-safe and no feature module becomes a second writer of active turn state.

Behavior lives on the values it belongs to -- `view.range_lines(...)`, `block.render(key)`,
`output.project(...)` -- so the surface here is four types plus the exact-matching functions,
which take current file lines rather than a view and therefore have no receiver to live on.
"""

from wizolt.source.output import SourceBlock, TextBlock, ToolOutput
from wizolt.source.relocate import context_matches, relocate_target, same_position
from wizolt.source.view import (
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
    SourceSpan,
    SourceView,
    SourceViewDraft,
    source_error,
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
    "TextBlock",
    "ToolOutput",
    "context_matches",
    "relocate_target",
    "same_position",
    "source_error",
]
