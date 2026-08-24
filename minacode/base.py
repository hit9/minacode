"""minacode base: errors, text helpers, configuration, and shared data types."""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Generic, TypeVar

from prompt_toolkit.utils import get_cwidth

__version__ = "0.32.0"

_ResourceT = TypeVar("_ResourceT")

Json = dict[str, Any]
ToolArgs = list[Any]


HTTP_USER_AGENT = "minacode/" + __version__


def configure_logging() -> None:
    """Quiet third-party loggers whose expected failures minacode already surfaces itself.

    Refresh failures / re-auth fall back to minacode's own handling, which surfaces an
    actionable "authentication required" message; suppress this logger's ERROR-level
    traceback spam (incl. the RuntimeError minacode raises as control flow).
    """
    logging.getLogger("fastmcp.client.auth.oauth").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.auth.oauth2").setLevel(logging.CRITICAL)
    # MCP client transports log expected-and-already-surfaced failures (httpx ReadTimeout on a
    # slow server, dropped SSE/stdio frames, JSON-RPC parse errors) at ERROR with full
    # tracebacks via logging.lastResort, which dumps them onto the TUI mid-render.
    # MCPManager captures these same failures into server_errors and the status bar, so the
    # library's own transport traceback is pure noise. Raise it out of the ERROR band.
    for _transport_logger in ("mcp.client.streamable_http", "mcp.client.sse", "mcp.client.stdio"):
        logging.getLogger(_transport_logger).setLevel(logging.CRITICAL)


MAX_TOOL_OUTPUT_TOKENS = 6_000
# Extension of the file a truncated tool result is materialized to, named after the result's key.
# ContextManager.materialize_output writes it; the session store's asset collector retains it.
TOOL_OUTPUT_ASSET_SUFFIX = ".txt"
# Cap on the AGENTS.md (or CLAUDE.md fallback) content injected into every request's fixed
# prefix, per DESIGN.md's "bound the fixed prefix" rule; truncation happens in
# ContextManager.environment, so SystemInfo.detect returns the file verbatim.
MAX_AGENTS_MD_TOKENS = 8_000
MODEL_REQUEST_RETRIES = 5
# Retry pacing: exponential backoff with jitter; RETRY_MAX_DELAY also clamps provider Retry-After
# values so a single aberrant header cannot stall the CLI for minutes. The wider budget costs
# wall-clock time only, which is visible and interruptible (see model.request()); retransmitted
# request prefixes are cache hits, so tokens are nearly free.
RETRY_BASE_DELAY = 1.0  # seconds; delay = RETRY_BASE_DELAY * 2 ** attempt, then jittered 0.5x-1.5x
RETRY_MAX_DELAY = 30.0  # seconds; single-wait ceiling, also clamps Retry-After
# Assistant turns carry the provider's own reply verbatim under these keys — Responses output
# items and Anthropic content blocks — so tool loops can replay opaque reasoning the protocol
# requires back unmodified. They are minacode's bookkeeping and never reach a request body.
RESPONSES_OUTPUT_KEY = "_responses_output"
ANTHROPIC_CONTENT_KEY = "_anthropic_content"
# Sources a provider-side search attached to one assistant message. Stored for rendering and resume,
# never replayed: the provider already carries its own search state in the echo keys above.
SEARCH_SOURCES_KEY = "_search_sources"
# Set when the provider ended a response without ending the turn, having paused a long server-side
# tool run. The message must be sent back unchanged to resume, so this travels with it as metadata.
PAUSED_TURN_KEY = "_paused_turn"
# Which endpoint issued the echo keys above: base URL, model, and a key fingerprint. The opaque
# halves of an echo — Responses `encrypted_content`, Anthropic thinking `signature` — are verified
# by their issuer, so replaying one anywhere else is rejected rather than ignored. Protocol alone
# cannot tell issuers apart: `/provider` can move between two Responses hosts, or between two
# entries on one host holding keys for different organizations. See ModelClient.provider_origin.
PROVIDER_ORIGIN_KEY = "_provider_origin"
PROVIDER_ECHO_KEYS = (RESPONSES_OUTPUT_KEY, ANTHROPIC_CONTENT_KEY, SEARCH_SOURCES_KEY, PAUSED_TURN_KEY, PROVIDER_ORIGIN_KEY)


def builtin_function_names(entries: Iterable[Json]) -> tuple[str, ...]:
    """Names of the builtin tools the provider calls back for instead of running entirely alone.

    Kimi's builtin functions are declared like any other builtin tool, but the model emits a real
    tool call for them and expects the client to answer it, so both the runner (to recognize the
    call) and the no-tools guard (to keep it) need the declared names."""
    names: list[str] = []
    for entry in entries:
        if entry.get("type") != "builtin_function":
            continue
        function = entry.get("function")
        name = function.get("name") if isinstance(function, dict) else ""
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def builtin_tool_label(name: str) -> str:
    """A display label for a tool the provider runs for itself.

    One tool carries a different name in each protocol — `web_search_call` as a Responses output
    item, `web_search` as a Messages server tool, `$web_search` as a Kimi builtin function — and
    all of them should read as the same phase in the transcript."""
    return (name.lstrip("$").removesuffix("_call").replace("_", " ").strip() or "provider tool").title()


# Protocol-neutral metadata for lifecycle/context checkpoint messages. Provider adapters remove
# this key while preserving the canonical role/content pair in the conversation log.
SESSION_EVENT_KEY = "_session_event"


# Image-delivery states of the active main route (REQUIREMENT-3 main-first image fallback).
# UNKNOWN routes optimistically receive raw images first; TEXT_ONLY_STATIC comes from the
# provider compatibility catalog; TEXT_ONLY_LEARNED is session-local evidence created when an
# eligible main request returns HTTP 400 for a request carrying a current-turn raw image.
IMAGE_ROUTE_UNKNOWN = "unknown"
IMAGE_ROUTE_TEXT_ONLY_STATIC = "text_only_static"
IMAGE_ROUTE_TEXT_ONLY_LEARNED = "text_only_learned"


@dataclass(frozen=True)
class ImageRouteNotice:
    """One gray routing notice for a text-only image delivery decision.

    `reason` says why the raw image was not delivered to the main model; `images` names
    the observed inputs; `described_by` optionally names the [vision] entry. Presentation
    only; never enters model context.
    """

    reason: str
    described_by: str = ""
    images: tuple[str, ...] = ()


SELECTION_BACK = object()
SELECTION_FREE_TEXT = object()
DISMISSED = "(The user dismissed the question without answering.)"

# The line prefixes a rendered Note body can open with (memory.py writes them). A fact about the
# Note format rather than about drawing, so it lives here: the renderer uses it to pick the
# per-line memory colors, and the Delegate worker wrapper uses it to recognize a worker's Note
# output and pass it through to that same renderer.
MEMORY_PREFIXES = ("goal:", "check:", "plan:", "known:")


class MinacodeError(Exception): ...


class ConfigError(MinacodeError): ...


class ModelError(MinacodeError): ...


class ModelResponseTimeout(ModelError): ...


class ModelOutputTruncated(ModelError): ...


class MalformedToolCallError(ModelError): ...


class ModelStreamIncomplete(ModelError): ...


class ModelRequestRetry(MinacodeError): ...


class ToolError(MinacodeError): ...


def split_lines(text: str) -> list[str]:
    """Canonical line model shared by Read, Edit, and persisted diffs: split on "\n" only, keeping
    the newline (like file.readlines()). str.splitlines(True) also breaks on \r, \v, \f, \x1c-\x1e,
    \x85, \u2028, \u2029, which would number lines differently than Read and desync anchors."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def oneline(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to `limit` characters with an ellipsis."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def drop_nulls(value: object) -> object:
    """Recursively drop null values. Strict schemas express optional params as nullable, so callers
    may send explicit null for an omitted argument; in minacode null means "absent"."""
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


class Text:
    BASE36: ClassVar[str] = "0123456789abcdefghijklmnopqrstuvwxyz"

    @staticmethod
    def clean(text: str) -> str:
        return text.encode("utf-8", errors="replace").decode("utf-8")

    @classmethod
    def base36(cls, value: int) -> str:
        out = ""
        while value:
            value, digit = divmod(value, 36)
            out = cls.BASE36[digit] + out
        return out or "0"

    @classmethod
    def value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls.clean(value)
        if isinstance(value, dict):
            return {cls.clean(str(key)): cls.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.value(item) for item in value]
        return value

    @staticmethod
    def elapsed_since(started_at: float, *, precise: bool = False) -> str:
        raw = max(0.0, time.monotonic() - started_at) if started_at else 0.0
        if raw < 60:
            return f"{raw:.1f}s" if precise else f"{int(raw)}s"
        minutes, seconds = divmod(int(raw), 60)
        return f"{minutes}m{seconds:02d}s"

    @staticmethod
    def age(seconds: float) -> str:
        """Wall-clock age in the coarsest unit that still says something. `elapsed_since` measures a
        running turn from a monotonic clock; this reads a stored timestamp, where minutes rarely matter."""
        for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
            if seconds >= size:
                return f"{int(seconds // size)}{unit} ago"
        return "just now"

    @staticmethod
    def abbreviate_count(value: int) -> str:
        """A count in the coarsest unit that still says something: 1.0M, 1.5K, or the bare number.
        Shared so `/status`, the status bar, and a worker's usage line all abbreviate alike."""
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

    @staticmethod
    def clip_width(text: str, width: int) -> str:
        width = max(0, width)
        if get_cwidth(text) <= width:
            return text
        ellipsis = "." * min(3, width)
        available = width - get_cwidth(ellipsis)
        clipped = []
        used = 0
        for char in text:
            char_width = max(0, get_cwidth(char))
            if used + char_width > available:
                break
            clipped.append(char)
            used += char_width
        return "".join(clipped).rstrip() + ellipsis

    @staticmethod
    def wrap_styled(
        prefix: list[tuple[str, str]],
        continuation: list[tuple[str, str]],
        content: list[tuple[str, str]],
        width: int | None = None,
    ) -> list[list[tuple[str, str]]]:
        logical_lines: list[list[tuple[str, str, int]]] = [[]]
        for style, text in content:
            for char in text:
                if char == "\n":
                    logical_lines.append([])
                else:
                    logical_lines[-1].append((style, char, get_cwidth(char)))

        def row_segments(row_prefix: list[tuple[str, str]], cells: list[tuple[str, str, int]]) -> list[tuple[str, str]]:
            row = list(row_prefix)
            for style, char, _ in cells:
                if row and row[-1][0] == style:
                    row[-1] = (style, row[-1][1] + char)
                else:
                    row.append((style, char))
            return row

        rows: list[list[tuple[str, str]]] = []
        row_prefix = prefix
        for logical in logical_lines:
            remaining = logical
            while True:
                prefix_width = sum(get_cwidth(text) for _, text in row_prefix)
                available = max(1, width - prefix_width) if width else None
                if available is None or sum(cell_width for _, _, cell_width in remaining) <= available:
                    rows.append(row_segments(row_prefix, remaining))
                    break
                used = 0
                fit = 0
                while fit < len(remaining) and used + remaining[fit][2] <= available:
                    used += remaining[fit][2]
                    fit += 1
                fit = max(1, fit)
                whitespace = max((index for index in range(fit) if remaining[index][1].isspace()), default=-1)
                cut = whitespace if whitespace > 0 else fit
                rows.append(row_segments(row_prefix, remaining[:cut]))
                remaining = remaining[cut + 1 :] if whitespace > 0 else remaining[cut:]
                row_prefix = continuation
            row_prefix = continuation
        return rows


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_write_prompt_tokens: int = 0
    last_prompt_tokens: int = 0
    last_prompt_budget: int = 0
    last_cached_prompt_tokens: int = 0
    last_cache_write_prompt_tokens: int = 0

    def context_percent(self, fallback: int = 0) -> int:
        """How full the last request's prompt was, as a percentage of that request's budget.

        The provider-reported pair describes one real request, so a changed max_context_tokens only
        shows up after the next request rather than re-scoring the last one. `fallback` (the
        session's own estimate, state.context_percent) stands in before any request exists.
        """
        if not self.last_prompt_tokens or not self.last_prompt_budget:
            return fallback
        return min(100, self.last_prompt_tokens * 100 // self.last_prompt_budget)

    @staticmethod
    def field(usage: Any, *paths: str) -> int:
        """First present dotted path in `usage` (dict keys or attributes) as an int, else 0."""
        for path in paths:
            raw = usage
            for key in path.split("."):
                raw = raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)
                if raw is None:
                    break
            else:
                return int(raw or 0)
        return 0

    def add(self, usage: Any, budget: int | None = None, *, touch_last: bool = True) -> None:
        """Add one completed request to the totals, and unless `touch_last` is False make it the
        new last-request snapshot (the status bar's ctx/cache reading). Non-main-model requests
        (vision observations) stay in the totals but must not move that snapshot."""
        self.calls += 1
        prompt_tokens = self.field(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self.field(usage, "completion_tokens", "output_tokens")
        # fmt: off
        cached_tokens = self.field(usage, "prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens")
        cache_write_tokens = self.field(
            usage,
            "cache_creation_input_tokens",
            "prompt_tokens_details.cache_write_tokens",
            "input_tokens_details.cache_write_tokens",
        )
        # fmt: on
        # OpenAI-shaped usage counts cache hits inside `prompt_tokens`, but Anthropic's
        # `input_tokens` is only what was neither read from nor written to the cache. Fold the cache
        # legs back in so the prompt total means the same thing for every provider; otherwise a
        # cached Anthropic request reports a hit ratio far above 100% and a tiny token total.
        if not self.field(usage, "prompt_tokens"):
            prompt_tokens += self.field(usage, "cache_read_input_tokens") + self.field(usage, "cache_creation_input_tokens")
        total_tokens = self.field(usage, "total_tokens") or prompt_tokens + completion_tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cached_prompt_tokens += cached_tokens
        self.cache_write_prompt_tokens += cache_write_tokens
        if touch_last:
            self.last_prompt_tokens = prompt_tokens
            if budget is not None:
                self.last_prompt_budget = budget
            self.last_cached_prompt_tokens = cached_tokens
            self.last_cache_write_prompt_tokens = cache_write_tokens


@dataclass
class UpdateStatus:
    _VERSION_RE: ClassVar[re.Pattern] = re.compile(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?")
    latest: str = ""
    checking: bool = False
    error: str = ""

    def newer_than(self, current: str) -> bool:
        current_version = self.version_tuple(current)
        latest_version = self.version_tuple(self.latest)
        return bool(current_version and latest_version and latest_version > current_version)

    @staticmethod
    def version_tuple(value: str) -> tuple[int, ...]:
        match = UpdateStatus._VERSION_RE.match(value)
        return tuple(int(part or 0) for part in match.groups()) if match else ()


@dataclass
class ToolCall:
    id: str
    name: str
    args: ToolArgs
    # A malformed-argument error captured while parsing the call. Deferred so it surfaces as a
    # tool result the model can correct from, instead of aborting the whole turn at parse time.
    error: str = ""


class LogEdge(Enum):
    NONE = ""
    BRANCH = "├"
    CONTINUE = "│"
    END = "└"


class LogRole(Enum):
    TOOL = auto()
    AUTO = auto()
    META = auto()
    OUTPUT = auto()
    ERROR = auto()
    MUTED = auto()
    DIFF = auto()
    WORKER = auto()
    FIELD = auto()
    CODE = auto()


@dataclass(frozen=True)
class ApprovalView:
    """The full text behind a call, for the read-only viewer its confirmation offers.

    A tool that commits to something longer than one log line returns one of these from
    `approval_view()`, and the runner uses it twice: to render the clipped, syntax-highlighted
    excerpt inside the approval block, and to feed the viewer opened by the confirm-time `v` key.
    The same view is what the post-hoc Ctrl-O browser shows, which is the only way to read the
    text under yolo -- there is no confirmation prompt there to press `v` at.

    `label` names it in the viewer title and the action row ("order", "script"). `lexer` is a
    pygments lexer name; empty means the text is prose and renders as markdown. `rows` are the
    header fields shown above the text. `result` is what the call returned, shown below the text
    when the view is opened after the fact; it is empty at a confirmation prompt, where the call
    has not run yet.
    """

    label: str
    text: str
    lexer: str = ""
    rows: list[tuple[str, str]] = field(default_factory=list)
    result: str = ""


@dataclass(frozen=True)
class LogLine:
    label: str
    text: str = ""
    role: LogRole = LogRole.OUTPUT
    edge: LogEdge = LogEdge.NONE
    meta: str = ""
    syntax: str = ""

    def text_prefix(self) -> str:
        edge = "" if self.edge is LogEdge.NONE else self.edge.value + " "
        separator = "  " if self.edge is LogEdge.NONE else " "
        return edge + self.label + (separator if self.label and self.text else "")


@dataclass
class LogBlock:
    INDENT: ClassVar[str] = "  "
    # The indent unit of a nested region, drawn instead of blank spacing so the call that opened
    # the region stays connected to everything logged inside it. Exactly as wide as INDENT: a rail
    # replaces spacing, never adds a column, so turning one on cannot shift the tree.
    RAIL: ClassVar[str] = "│ "
    items: list[LogLine | LogBlock]
    # True on a wrapper whose whole subtree was logged by an enclosing tool call (a ToolScript's
    # nested calls). Every line below the region's own root lines then carries the rail; see
    # margin_units and ToolRunner.emit.
    gutter: bool = False

    @classmethod
    def hierarchy(cls, root: LogLine | None, children: list[LogLine]) -> LogBlock:
        items: list[LogLine | LogBlock] = [root] if root else []
        if children:
            items.append(cls(list(children)))
        return cls(items)

    @property
    def has_children(self) -> bool:
        return any(isinstance(item, LogBlock) for item in self.items)

    @classmethod
    def margin(cls, level: int) -> str:
        return cls.INDENT * level

    @classmethod
    def prefix(cls, level: int, edge: LogEdge = LogEdge.NONE) -> str:
        return cls.margin(level) + ((edge.value + " ") if edge is not LogEdge.NONE else "")

    @classmethod
    def margin_units(cls, level: int, rails: tuple[int, ...] = ()) -> list[tuple[bool, str]]:
        """A line's indent, unit by unit, each flagged as a rail or as plain spacing.

        A nested region's rail sits at the unit its own root lines draw an edge in, so the root's
        `│` and the rail below it land in one column and the region reads as a single bracket. The
        root lines themselves are shorter than that unit and are unaffected."""
        return [(index in rails, cls.RAIL if index in rails else cls.INDENT) for index in range(level)]

    def walk(self, parent_level: int = 0):
        for line, level, _ in self.walk_rows(parent_level):
            yield line, level

    def walk_rows(self, parent_level: int = 0, rails: tuple[int, ...] = ()):
        """Every line with its depth and the rail units its margin carries.

        A gutter block claims the unit one past its own items' level: that is where the lines it
        contains draw their edges, so deeper lines rail in the same column instead of floating free
        under them."""
        level = parent_level + 1
        if self.gutter:
            rails = (*rails, level + 1)
        for item in self.items:
            if isinstance(item, LogLine):
                yield item, level, rails
            else:
                yield from item.walk_rows(level, rails)

    def __str__(self) -> str:
        rows = []
        for line, level, rails in self.walk_rows():
            margin = "".join(text for _, text in self.margin_units(level, rails))
            prefix = margin + line.text_prefix()
            continuation = margin + " " * get_cwidth(line.text_prefix())
            rows.extend(Text.wrap_styled([("", prefix)], [("", continuation)], [("", line.text + line.meta)]))
        return "\n".join("".join(text for _, text in row) for row in rows)


@dataclass
class TurnBox:
    ROOT_LEVEL: ClassVar[int] = 0
    CONTENT_LEVEL: ClassVar[int] = 1
    SEPARATOR: ClassVar[str] = ""
    messages: list[Json]

    @classmethod
    def group(cls, messages: list[Json]) -> list[TurnBox]:
        boxes: list[TurnBox] = []
        current: list[Json] = []
        for message in messages:
            current.append(message)
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                boxes.append(cls(current))
                current = []
        if current:
            boxes.append(cls(current))
        return boxes


class ActiveResource(Generic[_ResourceT]):
    """Thread-safe lifecycle for a resource that another thread may need to cancel."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: _ResourceT | None = None

    @contextlib.contextmanager
    def track(self, value: _ResourceT) -> Generator[None, None, None]:
        with self.lock:
            self.value = value
        try:
            yield
        finally:
            with self.lock:
                if self.value is value:
                    self.value = None

    def apply(self, action: Callable[[_ResourceT], None]) -> None:
        with self.lock:
            value = self.value
        if value is not None:
            action(value)
