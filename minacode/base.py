"""minacode base: errors, text helpers, configuration, and shared data types."""

from __future__ import annotations

import argparse
import codecs
import contextlib
import copy
import difflib
import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import platform
import queue
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import code_symbol_index as csi
import anthropic
import openai
from anthropic import Anthropic
from json_repair import repair_json
from openai import OpenAI
from prompt_toolkit import print_formatted_text, search as pt_search
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, HighlightIncrementalSearchProcessor, Processor, Transformation
from prompt_toolkit.output import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import SearchToolbar
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text as RichText

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.styles import get_style_by_name
    from pygments.token import Token
except ImportError:  # pragma: no cover - optional highlighting dependency
    pygments = None
    Token = None  # keep the name defined so class-body/token lookups don't NameError

__version__ = "0.11.0"

Json = dict[str, Any]


HTTP_USER_AGENT = "minacode/" + __version__
logging.getLogger("fastmcp.client.auth.oauth").setLevel(logging.WARNING)
# Refresh failures / re-auth fall back to minacode's own handling, which surfaces an
# actionable "authentication required" message; suppress this logger's ERROR-level
# traceback spam (incl. the RuntimeError minacode raises as control flow).
logging.getLogger("mcp.client.auth.oauth2").setLevel(logging.CRITICAL)
DEFAULT_MAX_CONTEXT_TOKENS = 240 * 1024
MAX_TOOL_OUTPUT_TOKENS = 6_000
MODEL_REQUEST_RETRIES = 2
PROVIDER_API_CHOICES = ("auto", "chat", "anthropic")
REASONING_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
REASONING_CHOICES = ("off", *REASONING_LEVELS)
CHAT_REASONING_CHOICES = ("auto", "off", "reasoning", "reasoning_effort", "thinking", "enable_thinking")
ANTHROPIC_DEFAULT_MAX_TOKENS = 16_384
DEEPSEEK_DEFAULT_MAX_TOKENS = 32_768
DEFAULT_OUTPUT_RESERVE_TOKENS = ANTHROPIC_DEFAULT_MAX_TOKENS
MIN_CONTEXT_SAFETY_TOKENS = 4_096
CHAT_REASONING_EFFORT_VALUES: dict[str, dict[str, str | int]] = {
    "thinking": {"minimal": "high", "low": "high", "medium": "high", "high": "max", "xhigh": "max"},
    "enable_thinking": {"minimal": 256, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384},
}
SELECTION_BACK = object()
SELECTION_FREE_TEXT = object()
DISMISSED = "(The user dismissed the question without answering.)"


class MinacodeError(Exception): ...


class ConfigError(MinacodeError): ...


class ModelError(MinacodeError): ...


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
    # fmt: off
    PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        "api.openai.com": {"chat_reasoning_rules": (("reasoning_effort", ("o1", "o3", "o4", "gpt-5")),), "strict_tools": True},
        "openrouter.ai": {"chat_reasoning": "reasoning"},
        "opencode.ai": {"api_rules": (("anthropic", ("claude-", "qwen3.")),), "chat_reasoning_rules": (("reasoning", ("deepseek-v4",)),)},
        "api.deepseek.com": {"chat_reasoning": "thinking", "max_tokens": DEEPSEEK_DEFAULT_MAX_TOKENS, "prompt_cache_key": False, "strict_tools": True, "strict_beta": True},
    }
    # fmt: on

    url: str = ""
    key: str = ""
    model: str = ""
    api: str = "auto"
    prompt_cache_key: str = "auto"
    available_models: tuple[str, ...] = ()
    temperature: float | None = None
    max_tokens: int = 0
    strict_tools: bool = False
    reasoning: str = "medium"
    chat_reasoning: str = "auto"
    timeout: int = 180
    extra_body: Json = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Json) -> "ProviderConfig":
        api = Config.str(data, "api", "auto")
        prompt_cache_key = cls.clean_prompt_cache_key(Config.str(data, "prompt_cache_key", "auto"))
        reasoning = Config.str(data, "reasoning", "medium")
        chat_reasoning = Config.str(data, "chat_reasoning", "auto")
        for key, value, choices in (
            ("api", api, PROVIDER_API_CHOICES),
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
            prompt_cache_key=prompt_cache_key,
            available_models=Config.str_tuple(data, "available_models"),
            temperature=Config.float(data, "temperature", None),
            max_tokens=max(0, Config.int(data, "max_tokens", 0)),
            strict_tools=Config.bool(data, "strict_tools", False),
            reasoning=reasoning,
            chat_reasoning=chat_reasoning,
            timeout=Config.int(data, "timeout", 180),
            extra_body=Config.table(data, "extra_body"),
        )

    def _stripped_url(self) -> str:
        url = self.url.rstrip("/")
        return url.removesuffix("/chat/completions").removesuffix("/responses").removesuffix("/messages")

    def base_url(self) -> str:
        # Strict tool calling is a beta feature on some hosts (DeepSeek); route to /beta only when active.
        url = self._stripped_url()
        return url + "/beta" if self.resolved_strict_tools() and self._profile().get("strict_beta") and not url.endswith("/beta") else url

    def host(self) -> str:
        return (urlparse(self._stripped_url()).hostname or "").lower()

    def _profile(self) -> Json:
        return self.PROFILES.get(self.host()) or {}

    def resolved_chat_reasoning(self) -> str:
        return self.profile_value(self.chat_reasoning, "off", "chat_reasoning", "chat_reasoning_rules")

    def resolved_api(self) -> str:
        return self.profile_value(self.api, "chat", "api", "api_rules")

    def profile_value(self, configured: str, default: str, profile_attr: str, rules_attr: str) -> str:
        if configured != "auto":
            return configured
        if not (profile := self._profile()):
            return default
        model = self.model.lower()
        for value, prefixes in profile.get(rules_attr, ()):
            if any(model.startswith(prefix) for prefix in prefixes):
                return str(value)
        return str(profile.get(profile_attr, default))

    def reasoning_effort(self) -> str:
        return self.reasoning if self.reasoning in REASONING_LEVELS else "medium"

    def resolved_max_tokens(self) -> int:
        # Generic OpenAI-compatible providers keep their own server-side cap; only opted-in profiles get a ceiling.
        return self.max_tokens or int(self._profile().get("max_tokens", 0))

    def output_token_budget(self) -> int:
        return self.resolved_max_tokens() or DEFAULT_OUTPUT_RESERVE_TOKENS

    def supports_prompt_cache_key(self) -> bool:
        # Default on for unknown OpenAI-compatible hosts (status quo); profiles opt out
        # (e.g. DeepSeek caches automatically by prefix and ignores the key).
        return bool(self._profile().get("prompt_cache_key", True))

    def supports_strict_tools(self) -> bool:
        return bool(self._profile().get("strict_tools"))

    def resolved_strict_tools(self) -> bool:
        # Only emit strict schemas on the chat path of a host known to support strict mode.
        return self.strict_tools and self.supports_strict_tools() and self.resolved_api() == "chat"

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
    theme: str = "auto"

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False, theme: str = "") -> "RuntimeSettings":
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            bash_wait_timeout=max(0, Config.int(runtime, "bash_wait_timeout", 10)),
            max_steps=max(1, Config.int(runtime, "max_agent_steps", 200)),
            max_context_tokens=max(1, Config.int(runtime, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            max_parallel_tools=max(1, Config.int(runtime, "max_parallel_tools", 4)),
            session_retention_days=max(0, Config.int(runtime, "session_retention_days", 7)),
            yolo=yolo or Config.bool(runtime, "yolo", False),
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
    def from_dict(cls, data: Json) -> "Config":
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
# api = "auto"                 # auto | anthropic | openai | ...
# reasoning = "medium"
# timeout = 180
# available_models = ["gpt-5", "gpt-5-mini"]

# [runtime]                    # optional overrides (defaults shown)
# yolo = false
# max_context_tokens = 245760      # 240K
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
    last_prompt_tokens: int = 0
    last_cached_prompt_tokens: int = 0

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
        total_tokens = self.field(usage, "total_tokens") or prompt_tokens + completion_tokens
        # fmt: off
        cached_tokens = self.field(usage, "prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens")
        # fmt: on
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cached_prompt_tokens += cached_tokens
        self.last_prompt_tokens = prompt_tokens
        self.last_cached_prompt_tokens = cached_tokens


@dataclass
class UpdateStatus:
    latest: str = ""
    checking: bool = False
    error: str = ""

    def newer_than(self, current: str) -> bool:
        current_version = self.version_tuple(current)
        latest_version = self.version_tuple(self.latest)
        return bool(current_version and latest_version and latest_version > current_version)

    @staticmethod
    def version_tuple(value: str) -> tuple[int, ...]:
        match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
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
    def detect(cls, cwd: str) -> "SystemInfo":
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
    args: list[Any]
    # A malformed-argument error captured while parsing the call. Deferred so it surfaces as a
    # tool result the model can correct from, instead of aborting the whole turn at parse time.
    error: str = ""
