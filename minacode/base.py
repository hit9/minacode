"""minacode base: errors, text helpers, configuration, and shared data types."""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Generic, TypeVar

from prompt_toolkit.utils import get_cwidth

try:
    import pygments
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = None
    Token = None  # keep the name defined so class-body/token lookups don't NameError

__version__ = "0.22.1"

_ResourceT = TypeVar("_ResourceT")

Json = dict[str, Any]
ToolArgs = list[Any]


HTTP_USER_AGENT = "minacode/" + __version__
logging.getLogger("fastmcp.client.auth.oauth").setLevel(logging.WARNING)
# Refresh failures / re-auth fall back to minacode's own handling, which surfaces an
# actionable "authentication required" message; suppress this logger's ERROR-level
# traceback spam (incl. the RuntimeError minacode raises as control flow).
logging.getLogger("mcp.client.auth.oauth2").setLevel(logging.CRITICAL)
# MCP client transports log expected-and-already-surfaced failures (httpx ReadTimeout on a
# slow server, dropped SSE/stdio frames, JSON-RPC parse errors) at ERROR with full
# tracebacks via logging.lastResort, which dumps them onto the TUI mid-render.
# MCPManager captures these same failures into server_errors and the status bar, so the
# library's own transport traceback is pure noise. Raise it out of the ERROR band.
for _transport_logger in ("mcp.client.streamable_http", "mcp.client.sse", "mcp.client.stdio"):
    logging.getLogger(_transport_logger).setLevel(logging.CRITICAL)
MAX_TOOL_OUTPUT_TOKENS = 6_000
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
PROVIDER_ECHO_KEYS = (RESPONSES_OUTPUT_KEY, ANTHROPIC_CONTENT_KEY, SEARCH_SOURCES_KEY, PAUSED_TURN_KEY)


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


SELECTION_BACK = object()
SELECTION_FREE_TEXT = object()
DISMISSED = "(The user dismissed the question without answering.)"


class MinacodeError(Exception): ...


class ConfigError(MinacodeError): ...


class ModelError(MinacodeError): ...


class ModelResponseTimeout(ModelError): ...


class ModelOutputTruncated(ModelError): ...


class MalformedToolCallError(ModelError): ...


class ModelRequestRetry(MinacodeError): ...


class ToolError(MinacodeError): ...


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
            for style, char, _char_width in cells:
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
                prefix_width = sum(get_cwidth(text) for _style, text in row_prefix)
                available = max(1, width - prefix_width) if width else None
                if available is None or sum(cell_width for _style, _char, cell_width in remaining) <= available:
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

    def add(self, usage: Any, budget: int | None = None) -> None:
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
    items: list[LogLine | LogBlock]

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

    def walk(self, parent_level: int = 0):
        level = parent_level + 1
        for item in self.items:
            if isinstance(item, LogLine):
                yield item, level
            else:
                yield from item.walk(level)

    def __str__(self) -> str:
        rows = []
        for line, level in self.walk():
            prefix = self.margin(level) + line.text_prefix()
            continuation = self.margin(level) + " " * get_cwidth(line.text_prefix())
            rows.extend(Text.wrap_styled([("", prefix)], [("", continuation)], [("", line.text + line.meta)]))
        return "\n".join("".join(text for _style, text in row) for row in rows)


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
    def track(self, value: _ResourceT) -> Iterator[None]:
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
