"""minacode base: errors, text helpers, configuration, and shared data types."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import re
import shutil
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Generic, TypeVar
from urllib.parse import urlparse

from prompt_toolkit.utils import get_cwidth

from minacode.model_catalog import REASONING_LEVELS
from minacode.provider_compat import COMPATIBILITY_PROFILES, CompatibilityProfile, ResolvedProvider, compatibility_for_host

try:
    import pygments
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = None
    Token = None  # keep the name defined so class-body/token lookups don't NameError

__version__ = "0.19.1"

_ResourceT = TypeVar("_ResourceT")

Json = dict[str, Any]
ToolArgs = list[Any]


HTTP_USER_AGENT = "minacode/" + __version__
logging.getLogger("fastmcp.client.auth.oauth").setLevel(logging.WARNING)
# Refresh failures / re-auth fall back to minacode's own handling, which surfaces an
# actionable "authentication required" message; suppress this logger's ERROR-level
# traceback spam (incl. the RuntimeError minacode raises as control flow).
logging.getLogger("mcp.client.auth.oauth2").setLevel(logging.CRITICAL)
DEFAULT_MAX_CONTEXT_TOKENS = 256 * 1024
MAX_TOOL_OUTPUT_TOKENS = 6_000
MODEL_REQUEST_RETRIES = 5
PROVIDER_API_CHOICES = ("auto", "chat", "responses", "anthropic")
IMAGE_INPUT_CHOICES = ("auto", "on", "off")
REASONING_CHOICES = ("off", *REASONING_LEVELS)
CHAT_REASONING_CHOICES = (
    "auto",
    "off",
    "reasoning",
    "reasoning_effort",
    "thinking",
    "thinking_toggle",
    "thinking_effort",
    "enable_thinking",
    "mandatory_thinking",
)
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
ANTHROPIC_DEFAULT_MAX_TOKENS = 16_384
DEFAULT_OUTPUT_RESERVE_TOKENS = ANTHROPIC_DEFAULT_MAX_TOKENS
# The configured cap and the reserve subtracted from the input budget describe the same output, so
# they are one number. Reasoning counts against this cap on the Responses and Anthropic wires, where
# a smaller value truncates a high-effort step before it emits any text or tool call.
DEFAULT_MAX_TOKENS = DEFAULT_OUTPUT_RESERVE_TOKENS
MIN_CONTEXT_SAFETY_TOKENS = 4_096
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
class ProviderConfig:
    COMPATIBILITY: ClassVar[dict[str, CompatibilityProfile]] = COMPATIBILITY_PROFILES

    url: str = ""
    key: str = ""
    model: str = ""
    api: str = "auto"
    stream: bool = True
    image_input: str = "auto"
    prompt_cache_key: str = "auto"
    available_models: tuple[str, ...] = ()
    temperature: float | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    strict_tools: bool = False
    reasoning: str = "medium"
    chat_reasoning: str = "auto"
    timeout: int = 120
    response_timeout: int = 600
    extra_body: Json = field(default_factory=dict)
    builtin_tools: tuple[Json, ...] = ()

    @classmethod
    def from_dict(cls, data: Json) -> ProviderConfig:
        api = Config.str(data, "api", "auto")
        image_input = Config.str(data, "image_input", "auto")
        prompt_cache_key = cls.clean_prompt_cache_key(Config.str(data, "prompt_cache_key", "auto"))
        reasoning = Config.str(data, "reasoning", "medium")
        chat_reasoning = Config.str(data, "chat_reasoning", "auto")
        for key, value, choices in (
            ("api", api, PROVIDER_API_CHOICES),
            ("image_input", image_input, IMAGE_INPUT_CHOICES),
            ("reasoning", reasoning, REASONING_CHOICES),
            ("chat_reasoning", chat_reasoning, CHAT_REASONING_CHOICES),
        ):
            if value not in choices:
                raise ConfigError("provider." + key + " must be one of " + ", ".join(choices))
        return cls(
            url=Config.str(data, "url"),
            key=Config.str(data, "key"),
            model=Config.str(data, "model"),
            api=api,
            stream=Config.bool(data, "stream", True),
            image_input=image_input,
            prompt_cache_key=prompt_cache_key,
            available_models=Config.str_tuple(data, "available_models"),
            temperature=Config.float(data, "temperature", None),
            max_tokens=max(0, Config.int(data, "max_tokens", DEFAULT_MAX_TOKENS)),
            strict_tools=Config.bool(data, "strict_tools", False),
            reasoning=reasoning,
            chat_reasoning=chat_reasoning,
            timeout=Config.int(data, "timeout", 120),
            response_timeout=max(0, Config.int(data, "response_timeout", 600)),
            extra_body=Config.table(data, "extra_body"),
            builtin_tools=Config.table_tuple(data, "builtin_tools"),
        )

    def builtin_function_names(self) -> tuple[str, ...]:
        """Declared builtin functions, which the runner answers instead of rejecting as unknown.
        Evidence: https://platform.kimi.ai/docs/guide/use-web-search"""
        return builtin_function_names(self.builtin_tools)

    def resolve(self) -> ResolvedProvider:
        """Fold explicit configuration and documented compatibility into one request policy."""

        url = self.url.rstrip("/").removesuffix("/chat/completions").removesuffix("/responses").removesuffix("/messages")
        host = (urlparse(url).hostname or "").lower()
        profile = compatibility_for_host(host, self.COMPATIBILITY)
        model = self.model.lower()

        api = self.api
        if api == "auto":
            path = urlparse(self.url.rstrip("/")).path
            suffix_api = next(
                (value for suffix, value in (("/responses", "responses"), ("/messages", "anthropic"), ("/chat/completions", "chat")) if path.endswith(suffix)),
                None,
            )
            api = suffix_api or profile.rule_value(profile.api_rules, model) or "chat"

        chat_reasoning = self.chat_reasoning
        if chat_reasoning == "auto":
            chat_reasoning = profile.rule_value(profile.chat_reasoning_rules, model) or profile.chat_reasoning or "off"

        if self.reasoning == "off":
            reasoning_effort = profile.rule_value(profile.reasoning_effort_off_rules, model)
            if api == "responses":
                reasoning_effort = profile.rule_value(profile.responses_reasoning_effort_off_rules, model) or reasoning_effort
        else:
            effort = self.reasoning_effort()
            reasoning_effort = profile.reasoning_effort_value(model, effort)

        suppress_temperature = profile.suppress_temperature or any(model.startswith(prefix) for prefix in profile.suppress_temperature_models)
        if not suppress_temperature:
            reasoning_enabled = self.reasoning != "off"
            suppress_temperature = reasoning_enabled and chat_reasoning in ("thinking", "enable_thinking")

        strict_tools_active = self.strict_tools and profile.strict_tools and api in ("chat", "responses")
        if strict_tools_active and profile.strict_beta and not url.endswith("/beta"):
            url += "/beta"

        return ResolvedProvider(
            api=api,
            base_url=url,
            host=host,
            chat_reasoning=chat_reasoning,
            chat_reasoning_history=profile.rule_value(profile.chat_reasoning_history_rules, model) or profile.chat_reasoning_history,
            reasoning_effort=reasoning_effort,
            responses_reasoning=profile.responses_reasoning_models is None or any(model.startswith(prefix) for prefix in profile.responses_reasoning_models),
            suppress_temperature=suppress_temperature,
            prompt_cache_key=profile.prompt_cache_key,
            strict_tools_active=strict_tools_active,
            builtin_tools_by_wire=profile.builtin_tools_by_wire,
        )

    def reasoning_effort(self) -> str:
        return self.reasoning if self.reasoning in REASONING_LEVELS else "medium"

    def output_token_budget(self) -> int:
        return self.max_tokens or DEFAULT_OUTPUT_RESERVE_TOKENS

    @staticmethod
    def clean_prompt_cache_key(value: str) -> str:
        value = value.strip()
        if not value:
            return "auto"
        lower = value.lower()
        if lower in {"auto", "off"}:
            return lower
        if len(value) > 64 or any(char.isspace() for char in value):
            raise ConfigError("provider.prompt_cache_key must be auto, off, or a stable key up to 64 chars without whitespace")
        return value


@dataclass
class RuntimeSettings:
    shell_timeout: int = 60
    # Bash foreground wait budget: if the command hasn't exited within this many seconds the running
    # process is promoted to a background job (see BashTool.stream_process) and control returns to
    # the model with a partial-output payload. Set to 0 to disable promotion (fall back to killing
    # on shell_timeout).
    bash_wait_timeout: int = 10
    max_steps: int = 200
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    session_retention_days: int = 7
    # Max read-only tool calls from one model batch to execute concurrently; 1 disables parallelism.
    max_parallel_tools: int = 4
    yolo: bool = False
    quick_hints: bool = True
    theme: str = "auto"

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False, theme: str = "") -> RuntimeSettings:
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            bash_wait_timeout=max(0, Config.int(runtime, "bash_wait_timeout", 10)),
            max_steps=max(1, Config.int(runtime, "max_agent_steps", 200)),
            max_context_tokens=max(1, Config.int(runtime, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_parallel_tools=max(1, Config.int(runtime, "max_parallel_tools", 4)),
            session_retention_days=max(0, Config.int(runtime, "session_retention_days", 7)),
            yolo=yolo or Config.bool(runtime, "yolo", False),
            quick_hints=Config.bool(runtime, "quick_hints", True),
            theme=theme or Config.str(runtime, "theme", "auto"),
        )


@dataclass
class Config:
    active_provider: str = "default"
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {"default": ProviderConfig()})
    data_dir: str = "~/.minacode"
    mcp: Json = field(default_factory=dict)

    # Backward compatibility: the data dir moved from ~/.nanocode to ~/.minacode.
    LEGACY_DATA_DIR: ClassVar[str] = "~/.nanocode"

    def __post_init__(self) -> None:
        # When the data dir is still the new default but does not exist yet and the legacy
        # ~/.nanocode dir does, keep using the legacy dir so existing sessions, skills, and
        # cache are found without a migration step.
        if (
            self.data_dir == "~/.minacode"
            and not os.path.exists(os.path.expanduser(self.data_dir))
            and os.path.exists(os.path.expanduser(self.LEGACY_DATA_DIR))
        ):
            self.data_dir = self.LEGACY_DATA_DIR

    @property
    def provider(self) -> ProviderConfig:
        return self.providers[self.active_provider]

    @classmethod
    def from_dict(cls, data: Json) -> Config:
        provider_root = cls.table(data, "provider")
        active = cls.str(provider_root, "active", "default")
        providers = {name: ProviderConfig.from_dict(value) for name, value in provider_root.items() if name != "active" and isinstance(value, dict)}
        if not providers:
            providers = {active: ProviderConfig.from_dict(provider_root)}
        if active not in providers:
            raise ConfigError(f"provider.active `{active}` does not exist")
        paths = cls.table(data, "paths")
        return cls(active_provider=active, providers=providers, data_dir=cls.str(paths, "data_dir", "~/.minacode"), mcp=cls.table(data, "mcp"))

    @staticmethod
    def table(data: Json, key: str) -> Json:
        return value if isinstance((value := data.get(key)), dict) else {}

    @staticmethod
    def table_tuple(data: Json, key: str) -> tuple[Json, ...]:
        """A list of tables passed through verbatim, checked only for the shape every host shares.

        Entries reach the wire unmodified, so validating their contents would mean tracking each
        host's tool catalog. `type` is the one field every documented builtin tool carries, and
        requiring it turns a typo into a config error instead of a provider 400."""
        value = data.get(key)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"config value `{key}` must be a list of tables")
        entries: list[Json] = []
        for item in value:
            if not isinstance(item, dict):
                raise ConfigError(f"config value `{key}` must be a list of tables")
            if not (isinstance(item.get("type"), str) and item["type"]):
                raise ConfigError(f"config value `{key}` entries must each set a non-empty `type`")
            entries.append(dict(item))
        return tuple(entries)

    @staticmethod
    def str(data: Json, key: str, default: str = "") -> str:
        return default if (value := data.get(key)) is None else str(value)

    @staticmethod
    def str_tuple(data: Json, key: str) -> tuple[str, ...]:
        value = data.get(key)
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise ConfigError(f"config value `{key}` must be a string list")

    @staticmethod
    def bool(data: Json, key: str, default: bool = False) -> bool:
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        lower = value.lower() if isinstance(value, str) else ""
        if lower in {"on", "true", "yes", "1", "off", "false", "no", "0"}:
            return lower in {"on", "true", "yes", "1"}
        raise ConfigError(f"config value `{key}` must be boolean")

    @staticmethod
    def int(data: Json, key: str, default: int) -> int:
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config value `{key}` must be integer")
        return value

    @staticmethod
    def float(data: Json, key: str, default: float | None) -> float | None:
        value = data.get(key)
        if value is None:
            return default
        if value is False or (isinstance(value, str) and value.lower() == "off"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config value `{key}` must be number or off")
        return float(value)


class ConfigFile:
    DEFAULT_PATH: ClassVar[str] = os.path.join(os.path.expanduser("~"), ".minacode", "config.toml")
    LEGACY_PATH: ClassVar[str] = os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")
    # Only the provider block is required; every other key falls back to its built-in default, so the
    # commented lines below just document the common knobs and their defaults.
    DEFAULT_TEXT: ClassVar[str] = """# minacode configuration — unset keys use built-in defaults.

[provider]
active = "default"

[provider.default]
url = ""
key = ""
model = ""
# api = "auto"                 # auto | chat | responses | anthropic
# stream = true
# image_input = "auto"         # auto | on | off
# reasoning = "medium"
# max_tokens = 16384           # output cap per request, reasoning included; 0 uses provider default
                               # also reserved from the input budget, so it trades against
                               # runtime.max_context_tokens one for one
# timeout = 120                # transport inactivity
# response_timeout = 600       # total generation time; 0 disables
# available_models = ["gpt-5", "gpt-5-mini"]

# builtin_tools = [{ type = "web_search" }]   # provider-side tools, passed through verbatim
                                              # OpenAI/Qwen: { type = "web_search" }
                                              # Anthropic:   { type = "web_search_20250305", name = "web_search" }
                                              # Z.AI:        { type = "web_search", web_search = { enable = "True" } }

# [runtime]                    # optional overrides (defaults shown)
# yolo = false
# quick_hints = true           # model-suggested next-step chips; toggle with /hints
# max_context_tokens = 262144      # 256K; how much of the model's window to use, not its size.
                               # Raise it for a 1M-window model; lower it for a smaller one.
# max_agent_steps = 200
# shell_timeout = 60

# [mcp.example]                # url (+ auth = "oauth") for remote, or command/args for stdio
# url = "https://example.com/mcp"
# auto_connect = false
"""

    @classmethod
    def resolve_path(cls, path: str | None) -> str:
        if path:
            return os.path.expanduser(path)
        # Backward compatibility: read the legacy ~/.nanocode/config.toml when the new
        # ~/.minacode/config.toml does not exist yet.
        if not os.path.exists(cls.DEFAULT_PATH) and os.path.exists(cls.LEGACY_PATH):
            return cls.LEGACY_PATH
        return cls.DEFAULT_PATH

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = cls.resolve_path(path)
        if os.path.exists(config_path):
            return config_path, False
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(cls.DEFAULT_TEXT)
        return config_path, True

    @classmethod
    def load(cls, path: str | None = None) -> Json:
        config_path = cls.resolve_path(path)
        try:
            with open(config_path, "rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError as error:
            raise ConfigError(f"config not found: {config_path}; run --init-config") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid config {config_path}: {error}") from error
        return data if isinstance(data, dict) else {}


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_write_prompt_tokens: int = 0
    last_prompt_tokens: int = 0
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

    def add(self, usage: Any) -> None:
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
class SystemInfo:
    # fmt: off
    COMMANDS: ClassVar[tuple[str, ...]] = (
        "bash", "git", "rg", "sed", "grep", "find", "awk", "python3", "jq", "xargs", "cat", "head", "tail", "wc",
        "sort", "uniq", "make", "cmake", "gcc", "g++", "clang", "clang++", "node", "npm", "uv", "pytest",
    )
    # fmt: on

    cwd: str
    os: str
    arch: str
    commands: tuple[str, ...]

    @classmethod
    def detect(cls, cwd: str) -> SystemInfo:
        return cls(
            cwd=cwd,
            os=platform.system() or sys.platform,
            arch=platform.machine() or "unknown",
            commands=tuple(name for name in cls.COMMANDS if shutil.which(name)),
        )


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
