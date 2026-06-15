"""nanocode: A small terminal coding agent written in Python."""

from __future__ import annotations

import argparse
import difflib
import asyncio
import fnmatch
import hashlib
import json
import os
import platform
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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import code_symbol_index as csi
from anthropic import Anthropic
from json_repair import repair_json
from openai import OpenAI
from prompt_toolkit import print_formatted_text, search as pt_search
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_done
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, HighlightIncrementalSearchProcessor
from prompt_toolkit.output import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar
from rich.console import Console
from rich.markdown import Markdown

__version__ = "0.5.12"

Json = dict[str, Any]
HTTP_USER_AGENT = "nanocode/" + __version__
DEFAULT_MAX_CONTEXT_TOKENS = 128_000
MAX_TOOL_OUTPUT_TOKENS = 6_000
MODEL_REQUEST_RETRIES = 2
PROVIDER_API_CHOICES = ("auto", "chat", "anthropic")
REASONING_LEVELS = ("minimal", "low", "medium", "high", "xhigh")
REASONING_CHOICES = ("off", *REASONING_LEVELS)
CHAT_REASONING_CHOICES = ("auto", "off", "reasoning", "reasoning_effort", "thinking", "enable_thinking")
ANTHROPIC_DEFAULT_MAX_TOKENS = 16_384
CHAT_REASONING_EFFORT_VALUES: dict[str, dict[str, str | int]] = {
    "thinking": {"minimal": "high", "low": "high", "medium": "high", "high": "high", "xhigh": "max"},
    "enable_thinking": {"minimal": 256, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384},
}
SELECTION_BACK = object()


class NanocodeError(Exception):
    pass


class ConfigError(NanocodeError):
    pass


class ModelError(NanocodeError):
    pass


class ModelRequestRetry(NanocodeError):
    pass


class ToolError(NanocodeError):
    pass


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


@dataclass
class ProviderConfig:
    ALIYUN_CHAT_REASONING_RULES: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("enable_thinking", ("qwen", "qwq", "qvq")),
        ("thinking", ("deepseek-v4",)),
    )
    PROFILES: ClassVar[dict[str, dict[str, Any]]] = {
        "api.openai.com": {"chat_reasoning_rules": (("reasoning_effort", ("o1", "o3", "o4", "gpt-5")),)},
        "openrouter.ai": {"chat_reasoning": "reasoning"},
        "opencode.ai": {"api_rules": (("anthropic", ("claude-", "qwen3.")),), "chat_reasoning_rules": (("reasoning", ("deepseek-v4",)),)},
        "api.deepseek.com": {"chat_reasoning": "thinking"},
        "dashscope.aliyuncs.com": {"chat_reasoning_rules": ALIYUN_CHAT_REASONING_RULES},
        "dashscope-intl.aliyuncs.com": {"chat_reasoning_rules": ALIYUN_CHAT_REASONING_RULES},
        "dashscope-us.aliyuncs.com": {"chat_reasoning_rules": ALIYUN_CHAT_REASONING_RULES},
    }

    url: str = ""
    key: str = ""
    model: str = ""
    api: str = "auto"
    prompt_cache_key: str = "auto"
    available_models: tuple[str, ...] = ()
    temperature: float | None = None
    reasoning: str = "medium"
    chat_reasoning: str = "auto"
    timeout: int = 180

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
            reasoning=reasoning,
            chat_reasoning=chat_reasoning,
            timeout=Config.int(data, "timeout", 180),
        )

    def base_url(self) -> str:
        url = self.url.rstrip("/")
        for suffix in ("/chat/completions", "/responses", "/messages"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url

    def host(self) -> str:
        return (urlparse(self.base_url()).hostname or "").lower()

    def resolved_chat_reasoning(self) -> str:
        return self.profile_value(self.chat_reasoning, "off", "chat_reasoning", "chat_reasoning_rules")

    def resolved_api(self) -> str:
        return self.profile_value(self.api, "chat", "api", "api_rules")

    def profile_value(self, configured: str, default: str, profile_attr: str, rules_attr: str) -> str:
        if configured != "auto":
            return configured
        profile = self.PROFILES.get(self.host())
        if not profile:
            return default
        model = self.model.lower()
        for value, prefixes in profile.get(rules_attr, ()):
            if any(model.startswith(prefix) for prefix in prefixes):
                return str(value)
        return str(profile.get(profile_attr, default))

    def reasoning_effort(self) -> str:
        return self.reasoning if self.reasoning in REASONING_LEVELS else "medium"

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
    max_steps: int = 200
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    check_updates: bool = True
    update_check_interval_hours: int = 24
    yolo: bool = False
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False) -> "RuntimeSettings":
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            max_steps=max(1, Config.int(runtime, "max_agent_steps", Config.int(runtime, "max_steps", 200))),
            max_context_tokens=max(1, Config.int(runtime, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            check_updates=Config.bool(runtime, "check_updates", True),
            update_check_interval_hours=max(1, Config.int(runtime, "update_check_interval_hours", 24)),
            yolo=yolo or Config.bool(runtime, "yolo", False),
        )


@dataclass
class Config:
    active_provider: str = "default"
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {"default": ProviderConfig()})
    data_dir: str = "~/.nanocode"
    mcp: Json = field(default_factory=dict)

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
        return cls(active_provider=active, providers=providers, data_dir=cls.str(paths, "data_dir", "~/.nanocode"), mcp=cls.table(data, "mcp"))

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
    DEFAULT_PATH: ClassVar[str] = os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")
    DEFAULT_TEXT: ClassVar[str] = """# nanocode configuration

[provider]
active = "default"

[provider.default]
url = ""
key = ""
model = ""
api = "auto"
prompt_cache_key = "auto"
# available_models = ["gpt-5", "gpt-5-mini"]
# temperature = 0.2
reasoning = "medium"
# chat_reasoning = "auto"
timeout = 180

[paths]
data_dir = "~/.nanocode"

[runtime]
shell_timeout = 60
max_agent_steps = 200
max_context_tokens = 128000
check_updates = true
update_check_interval_hours = 24
yolo = false

# [mcp.example]
# url = "https://example.com/mcp"
# bearer_token_env_var = "EXAMPLE_MCP_TOKEN"
# enabled = true
"""

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = os.path.expanduser(path or cls.DEFAULT_PATH)
        if os.path.exists(config_path):
            return config_path, False
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(cls.DEFAULT_TEXT)
        return config_path, True

    @classmethod
    def load(cls, path: str | None = None) -> Json:
        config_path = os.path.expanduser(path or cls.DEFAULT_PATH)
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
    last_total_tokens: int = 0
    last_cached_prompt_tokens: int = 0

    def add(self, usage: Any) -> None:
        def value(*paths: str) -> int:
            for path in paths:
                raw = usage
                for key in path.split("."):
                    raw = raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)
                    if raw is None:
                        break
                else:
                    return int(raw or 0)
            return 0

        self.calls += 1
        prompt_tokens = value("prompt_tokens", "input_tokens")
        completion_tokens = value("completion_tokens", "output_tokens")
        total_tokens = value("total_tokens") or prompt_tokens + completion_tokens
        cached_tokens = value(
            "prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens"
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cached_prompt_tokens += cached_tokens
        self.last_total_tokens = total_tokens
        self.last_cached_prompt_tokens = cached_tokens


@dataclass
class AgentState:
    goal: str = ""
    plan: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    check: str = ""
    summary: str = ""
    code_index_status: str = ""
    code_index_error: str = ""
    code_index_notice: str = ""
    code_index_refreshing: bool = False
    context_percent: int = 0
    turn_step: int = 0
    turn_tool_calls: int = 0
    turn_messages: int = 0
    current_model_call_started_at: float = 0.0
    manual_model_retry_requested: bool = False
    model_retry_count: int = 0

    @staticmethod
    def plan_text(item: str) -> str:
        return re.sub(r"^\[(?: |x|X|~|-)\]\s+", "", item.strip()).strip()

    @classmethod
    def plan_rows_for(cls, items: list[str], *, status: bool = False) -> list[str]:
        rows = [
            f"- [{'~' if index == 0 else ' '}] {item}" if status else "- " + item
            for index, item in enumerate(filter(None, (cls.plan_text(str(item)) for item in items)))
        ]
        return rows or ["- (empty)"]

    def plan_rows(self, *, status: bool = False) -> list[str]:
        return self.plan_rows_for(self.plan, status=status)

    def current_focus(self) -> str:
        return next((text for item in self.plan if (text := self.plan_text(item))), "")

    def apply(self, data: Json) -> None:
        for attr in ("goal", "summary", "check"):
            if isinstance(data.get(attr), str):
                setattr(self, attr, str(data[attr]).strip())
        for attr in ("plan", "known"):
            value = data.get(attr)
            if isinstance(value, list):
                items = list(filter(None, (str(item).strip() for item in value)))
                setattr(self, attr, [self.plan_text(item) for item in items] if attr == "plan" else items)

    def format(self) -> str:
        known = ["- " + item for item in self.known] or ["- (empty)"]
        return "\n".join(["Goal: " + (self.goal or "(empty)"), "Plan:", *self.plan_rows(), "Known:", *known, "Check: " + (self.check or "(empty)")])


@dataclass
class UpdateStatus:
    latest: str = ""
    checked_at: float = 0.0
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
class ToolResultRecord:
    key: str
    name: str
    args: list[Any]
    output: str
    note: str = ""


@dataclass
class ToolErrorRecord:
    key: str
    name: str
    args: list[Any]
    error: str


@dataclass
class MCPServerConfig:
    name: str
    url: str = ""
    bearer_token_env_var: str = ""
    env_http_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    error: str = ""


@dataclass
class MCPToolInfo:
    server: str
    name: str
    description: str
    input_schema: Json
    annotations: Json = field(default_factory=dict)


@dataclass
class SystemInfo:
    COMMANDS: ClassVar[tuple[str, ...]] = (
        "bash",
        "git",
        "rg",
        "sed",
        "grep",
        "find",
        "awk",
        "python3",
        "jq",
        "xargs",
        "cat",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "make",
        "cmake",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "node",
        "npm",
        "uv",
        "pytest",
    )

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
class Session:
    cwd: str = field(default_factory=os.getcwd)
    system_info: SystemInfo | None = None
    config: Config = field(default_factory=Config)
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    messages: list[Json] = field(default_factory=list)
    state: AgentState = field(default_factory=AgentState)
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_records: list[ToolResultRecord] = field(default_factory=list)
    tool_errors: list[ToolErrorRecord] = field(default_factory=list)
    pending_user_inputs: list[str] = field(default_factory=list)
    tool_counter: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    update: UpdateStatus = field(default_factory=UpdateStatus)
    mcp: MCPManager | None = None
    _gitignore_cache: dict[str, tuple[int, list[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.system_info is None:
            self.system_info = SystemInfo.detect(self.cwd)
        if self.mcp is None:
            self.mcp = MCPManager(self)

    @classmethod
    def from_config_file(cls, *, path: str | None = None, yolo: bool = False) -> "Session":
        data = ConfigFile.load(path)
        return cls(config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo))

    def resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        return os.path.abspath(path if os.path.isabs(path) else os.path.join(self.cwd, path))

    def relpath(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.cwd)
        except ValueError:
            return path

    def in_cwd(self, path: str) -> bool:
        try:
            return os.path.commonpath([os.path.realpath(self.cwd), os.path.realpath(path)]) == os.path.realpath(self.cwd)
        except ValueError:
            return False

    def data_path(self, *parts: str) -> str:
        root = os.path.expanduser(self.config.data_dir)
        return os.path.abspath(os.path.join(root if os.path.isabs(root) else os.path.join(self.cwd, root), *parts))

    def debug_dir(self) -> str:
        return self.data_path("debug")

    def missing_config(self) -> list[str]:
        provider = self.config.provider
        return [key for key, value in (("provider.url", provider.url), ("provider.key", provider.key), ("provider.model", provider.model)) if not value]

    def store_tool_result(self, name: str, args: list[Any], output: str, note: str = "") -> str:
        self.tool_counter += 1
        key = f"tr.{self.tool_counter}"
        args, output = Text.value(list(args)), Text.clean(output)
        self.tool_results[key] = output
        self.tool_records.append(ToolResultRecord(key, name, args, output, note))
        if len(self.tool_results) > 400:
            old = self.tool_records.pop(0)
            self.tool_results.pop(old.key, None)
        return key

    def record_tool_error(self, key: str, name: str, args: list[Any], error: str) -> None:
        self.tool_errors.append(ToolErrorRecord(key, name, Text.value(list(args)), " ".join(Text.clean(error).split())))
        self.tool_errors = self.tool_errors[-5:]


class UpdateChecker:
    PYPI_URL = "https://pypi.org/pypi/nanocode-cli/json"
    CACHE_FILE = "update.json"
    TIMEOUT = 5

    def __init__(self, session: Session):
        self.session = session

    def start(self) -> None:
        self.load_cache()
        if (
            not self.session.settings.check_updates
            or self.session.update.checking
            or time.time() - self.session.update.checked_at < self.session.settings.update_check_interval_hours * 3600
        ):
            return
        self.session.update.checking = True
        threading.Thread(target=self.check, daemon=True).start()

    def check(self) -> None:
        try:
            self.session.update.latest = self.fetch_latest()
            self.session.update.error = ""
        except Exception as error:
            self.session.update.error = Text.clean(str(error))
        finally:
            self.session.update.checked_at = time.time()
            self.session.update.checking = False
            self.save_cache()

    def fetch_latest(self) -> str:
        request = Request(self.PYPI_URL, headers={"Accept": "application/json", "User-Agent": HTTP_USER_AGENT})
        with urlopen(request, timeout=self.TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        version = data.get("info", {}).get("version") if isinstance(data, dict) else ""
        if not isinstance(version, str) or not UpdateStatus.version_tuple(version):
            raise NanocodeError("invalid PyPI version response")
        return version

    def load_cache(self) -> None:
        try:
            with open(self.session.data_path(self.CACHE_FILE), encoding="utf-8") as file:
                data = json.load(file)
            latest = str(data.get("latest") or "")
            self.session.update.latest = latest if UpdateStatus.version_tuple(latest) else ""
            self.session.update.checked_at = float(data.get("checked_at") or 0)
        except Exception:
            pass

    def save_cache(self) -> None:
        path = self.session.data_path(self.CACHE_FILE)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"checked_at": self.session.update.checked_at, "latest": self.session.update.latest}, file)
        except Exception:
            pass

    def status_line(self) -> str:
        update = self.session.update
        if not self.session.settings.check_updates:
            return "update: off"
        if update.checking:
            return "update: checking"
        return (
            f"update: {__version__} -> {update.latest} (uv tool upgrade nanocode-cli)"
            if update.newer_than(__version__)
            else "update: error"
            if update.error
            else "update: current"
            if update.latest
            else "update: unknown"
        )


class Tool:
    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    SIGNATURE: ClassVar[str] = ""
    EXAMPLE: ClassVar[tuple[str, ...]] = ()
    RANGE_SCHEMA: ClassVar[Json] = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
    SKIP_DIRS: ClassVar[set[str]] = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}
    MUTATES: ClassVar[bool] = False
    STORES_RESULT: ClassVar[bool] = True

    def __init__(self, session: Session, args: list[Any]):
        self.session = session
        self.args = args

    @classmethod
    def schema(cls) -> Json:
        description = "\n".join([cls.DESCRIPTION, "Signature: " + cls.SIGNATURE, *(("- " + item) for item in cls.EXAMPLE if item)])
        return {"type": "function", "function": {"name": cls.NAME, "description": description, "parameters": cls.params_schema()}}

    @classmethod
    def params_schema(cls) -> Json:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [payload]

    def needs_confirmation(self) -> bool:
        return self.MUTATES

    def preview(self) -> str:
        return f"{self.NAME}({', '.join(self.short_args())})"

    def short_args(self) -> list[str]:
        return [self.compact(arg) for arg in self.args]

    def call(self) -> str:
        raise NotImplementedError

    def strings(self, *, min_count: int = 0, max_count: int | None = None) -> list[str]:
        if len(self.args) < min_count or (max_count is not None and len(self.args) > max_count):
            limit = f"{min_count}" if max_count == min_count else f"{min_count}-{max_count or 'many'}"
            raise ToolError(f"{self.NAME} requires {limit} string args")
        if not all(isinstance(arg, str) for arg in self.args):
            raise ToolError(f"{self.NAME} args must be strings")
        return [str(arg) for arg in self.args]

    @staticmethod
    def line_range(value: Any, label: str = "range") -> tuple[int, int]:
        if not isinstance(value, list) or len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ToolError(f"{label} must be [start,end] integers")
        start, end = value
        if start < 0 or end < 0:
            raise ToolError(f"{label} values must be >= 0")
        return int(start), int(end)

    @staticmethod
    def compact(value: Any, limit: int = 120) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def process_result(tag: str, code: int, stdout: str, stderr: str) -> str:
        lines = [f"<{tag}>", f"* exit_code: {code}"]
        for name, text in (("stdout", stdout), ("stderr", stderr)):
            if text:
                lines.extend([f"<{name}>", text.rstrip(), f"</{name}>"])
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    @staticmethod
    def file_stat(path: str) -> str:
        stat = os.stat(path)
        return f'<file_stat mtime_ns="{stat.st_mtime_ns}" size="{stat.st_size}"/>'

    def gitignore_patterns(self, root: str) -> list[str]:
        patterns = []
        cache = self.session._gitignore_cache
        paths = [os.path.join(self.session.cwd, ".gitignore")]
        if os.path.isdir(root):
            paths.append(os.path.join(root, ".gitignore"))
        for path in dict.fromkeys(paths):
            try:
                mtime = os.stat(path).st_mtime_ns
                cached = cache.get(path)
                if cached is not None and cached[0] == mtime:
                    patterns.extend(cached[1])
                    continue
                with open(path, encoding="utf-8") as file:
                    pats = [line.strip() for line in file if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!")]
                cache[path] = (mtime, pats)
                patterns.extend(pats)
            except OSError:
                cache.pop(path, None)
        return patterns

    def ignored(self, path: str, patterns: list[str]) -> bool:
        rel = self.session.relpath(path).replace(os.sep, "/")
        name = os.path.basename(path)
        parts = [part for part in rel.split("/") if part and part != "."]
        for raw in patterns:
            directory = raw.endswith("/")
            pattern = raw.rstrip("/")
            if not pattern:
                continue
            if "/" in pattern:
                matched = fnmatch.fnmatch(rel, pattern) or (directory and (rel == pattern or rel.startswith(pattern + "/")))
            else:
                matched = any(fnmatch.fnmatch(part, pattern) for part in parts) or fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern)
            if matched:
                return True
        return False

    def default_ignored(self, path: str, patterns: list[str]) -> bool:
        rel = self.session.relpath(path).replace(os.sep, "/")
        hidden = rel not in {"", "."} and any(part.startswith(".") for part in rel.split("/") if part and part != ".")
        return hidden or self.ignored(path, patterns)


class ReadTool(Tool):
    NAME = "Read"
    DESCRIPTION = "Read UTF-8 file line ranges; returns file stat, total lines, line:hash text, and updates FILE STATE."
    SIGNATURE = "Read(path,ranges=[[start,end],...]) or Read(files=[{path,ranges}]); lines are 0-based, end-exclusive"
    EXAMPLE = (
        'Read ranges. Example: {"path":"src/app.py","ranges":[[0,80],[120,180]]}',
        'Read several files. Example: {"files":[{"path":"src/app.py","ranges":[[0,80]]},{"path":"README.md","ranges":[[0,40]]}]}',
    )

    @classmethod
    def arg_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}, "ranges": {"type": "array", "minItems": 1, "items": cls.RANGE_SCHEMA}},
            "required": ["path"],
            "additionalProperties": False,
        }

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1},
                "files": {"type": "array", "items": cls.arg_schema(), "minItems": 1},
            },
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return (
            payload["files"]
            if isinstance(payload.get("files"), list)
            else [{"path": payload.get("path", ""), "ranges": cls.ranges_arg(payload.get("ranges") or [[0, 0]])}]
        )

    @classmethod
    def ranges_arg(cls, value: Any) -> Any:
        return [value] if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and not isinstance(item, bool) for item in value) else value

    @staticmethod
    def line_hash(line: str) -> str:
        return Text.base36(int(hashlib.sha1(line.encode("utf-8")).hexdigest()[:6], 16)).rjust(5, "0")

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(path) for path, _ranges in self.targets())

    def call(self) -> str:
        return "\n\n".join(self.read_one(path, ranges) for path, ranges in self.targets())

    def short_args(self) -> list[str]:
        return [self.session.relpath(path) + " " + ",".join(f"{start}:{end}" for start, end in ranges) for path, ranges in self.targets()]

    def targets(self) -> list[tuple[str, list[tuple[int, int]]]]:
        if not self.args:
            raise ToolError("Read requires at least one {path,ranges} object")
        targets = []
        for index, spec in enumerate(self.args):
            if not isinstance(spec, dict):
                raise ToolError("Read args must be {path,ranges} objects")
            if unexpected := sorted(set(spec) - {"path", "ranges"}):
                raise ToolError("Read unexpected field: " + ", ".join(unexpected))
            path = str(spec.get("path") or "").strip()
            raw_ranges = self.ranges_arg(spec.get("ranges") if "ranges" in spec else [[0, 0]])
            if not path:
                raise ToolError("Read requires non-empty path")
            if not isinstance(raw_ranges, list) or not raw_ranges:
                raise ToolError("Read requires non-empty ranges")
            ranges = [self.line_range(value, f"args[{index}].ranges") for value in raw_ranges]
            targets.append((self.session.resolve_path(path), ranges))
        return targets

    def read_one(self, path: str, ranges: list[tuple[int, int]]) -> str:
        with open(path, encoding="utf-8") as file:
            lines = file.readlines()
        out = [f"<Read path={json.dumps(self.session.relpath(path))}>", self.file_stat(path), f"<total_lines>{len(lines)}</total_lines>"]
        for start, requested_end in ranges:
            start = min(start, len(lines))
            end = max(start, len(lines) if requested_end == 0 else min(len(lines), requested_end))
            out.append(f"<range>{start}:{end}</range>")
            out.append("<content hashline-numbered>")
            out.append("".join(f"{i}:{self.line_hash(lines[i])}|{lines[i]}" for i in range(start, end)).rstrip("\n"))
            out.append("</content>")
        out.append("</Read>")
        return "\n".join(out)


class LineCountTool(Tool):
    NAME = "LineCount"
    DESCRIPTION = "Count UTF-8 file lines; returns per-file counts, missing/not_file rows, and total."
    SIGNATURE = "LineCount(paths=[path,...])"
    EXAMPLE = ('Count several files. Example: {"paths":["src/app.py","pyproject.toml"]}',)

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
            "required": ["paths"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return list(payload.get("paths") or [])

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(path) for path in self.paths())

    def call(self) -> str:
        rows = []
        total = 0
        for path in self.paths():
            relpath = self.session.relpath(path)
            if not os.path.exists(path):
                rows.append(f"* missing: {relpath}")
                continue
            if not os.path.isfile(path):
                rows.append(f"* not_file: {relpath}")
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as file:
                    count = sum(1 for _ in file)
            except OSError as error:
                rows.append(f"* error: {relpath}: {error}")
                continue
            total += count
            rows.append(f"* {relpath}: {count}")
        return "\n".join(["<LineCountToolResult>", *rows, f"<total>{total}</total>", "</LineCountToolResult>"])

    def paths(self) -> list[str]:
        return [self.session.resolve_path(path) for path in self.strings(min_count=1)]


class ListTool(Tool):
    NAME = "List"
    DESCRIPTION = "List one directory, including hidden entries; returns child dirs/files/symlinks with file text/binary labels."
    SIGNATURE = "List(path, glob?); glob filters child names, not recursively"
    EXAMPLE = ('List child entries. Example: {"path":"."}', 'Filter child names. Example: {"path":"tests","glob":"test_*.py"}')

    @classmethod
    def params_schema(cls) -> Json:
        return {"type": "object", "properties": {"path": {"type": "string"}, "glob": {"type": "string"}}, "required": ["path"], "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [str(payload.get("path") or "."), *([str(payload["glob"])] if payload.get("glob") else [])]

    def needs_confirmation(self) -> bool:
        return not self.session.in_cwd(self.path())

    def call(self) -> str:
        path = self.path()
        args = self.strings(max_count=2)
        pattern = args[1] if len(args) > 1 else ""
        if not os.path.isdir(path):
            raise ToolError("not a directory")
        rows = []
        with os.scandir(path) as scan:
            for entry in scan:
                if pattern and not fnmatch.fnmatch(entry.name, pattern):
                    continue
                kind = (
                    "symlink"
                    if entry.is_symlink()
                    else "dir"
                    if entry.is_dir(follow_symlinks=False)
                    else "file"
                    if entry.is_file(follow_symlinks=False)
                    else "other"
                )
                label = kind + ((" " + self.file_type(entry.path)) if kind == "file" else "")
                rows.append((kind, label, self.session.relpath(entry.path) + ("/" if kind == "dir" else "")))
        order = {"dir": 0, "file": 1, "symlink": 2, "other": 3}
        rows.sort(key=lambda item: (order[item[0]], item[2]))
        return "\n".join(["<ListToolResult>"] + [f"* {label}: {name}" for _kind, label, name in rows] + ["</ListToolResult>"])

    def path(self) -> str:
        args = self.strings(max_count=2)
        return self.session.resolve_path(args[0] if args else ".")

    @staticmethod
    def file_type(path: str) -> str:
        try:
            chunk = open(path, "rb").read(4096)
            chunk.decode("utf-8")
            return "text" if b"\0" not in chunk else "binary"
        except Exception:
            return "binary"


class FindTool(Tool):
    NAME = "Find"
    DESCRIPTION = "Find file/dir paths by name or relative glob; skips hidden/gitignored entries."
    SIGNATURE = "Find(name,path?,type?,limit?) or Find(queries=[...]); type=file|dir|any"
    EXAMPLE = (
        'Find files. Example: {"name":"*.py"}',
        'Find dirs. Example: {"name":"migrations","type":"dir"}',
        'Batch. Example: {"queries":[{"name":"*.py","path":"src"},{"name":"test_*","path":"tests"}]}',
    )
    MAX_LIMIT = 500

    @classmethod
    def arg_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"},
                "type": {"type": "string", "enum": ["file", "dir", "any"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_LIMIT},
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    @classmethod
    def params_schema(cls) -> Json:
        props = dict(cls.arg_schema()["properties"])
        props["queries"] = {"type": "array", "items": cls.arg_schema(), "minItems": 1}
        return {"type": "object", "properties": props, "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return payload.get("queries") or [payload]

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(request["path"]) for request in self.requests())

    def call(self) -> str:
        return "\n\n".join(self.find(request) for request in self.requests())

    def short_args(self) -> list[str]:
        rows = []
        for request in self.requests():
            rel = self.session.relpath(str(request["path"]))
            rows.append(
                " ".join(
                    [
                        str(request["name"]),
                        *(["path=" + rel] if rel != "." else []),
                        *(["type=" + str(request["type"])] if request["type"] != "file" else []),
                        *(["limit=" + str(request["limit"])] if request["limit"] != 100 else []),
                    ]
                )
            )
        return ["; ".join(rows)]

    def requests(self) -> list[Json]:
        if not self.args:
            raise ToolError("Find requires at least one query object")
        requests = []
        for item in self.args:
            if not isinstance(item, dict):
                raise ToolError("Find args must be query objects")
            if unexpected := sorted(set(item) - {"name", "path", "type", "limit"}):
                raise ToolError("Find unexpected field: " + ", ".join(unexpected))
            name = str(item.get("name") or "").strip()
            kind = str(item.get("type") or "file")
            limit = item.get("limit", 100)
            if not name:
                raise ToolError("Find requires name")
            if kind not in {"file", "dir", "any"}:
                raise ToolError("Find type must be file, dir, or any")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > self.MAX_LIMIT:
                raise ToolError(f"Find limit must be 1..{self.MAX_LIMIT}")
            requests.append({"name": name, "path": self.session.resolve_path(str(item.get("path") or ".")), "type": kind, "limit": limit})
        return requests

    def find(self, request: Json) -> str:
        rows = self.matches(str(request["path"]), str(request["name"]), str(request["type"]))
        limit = int(request["limit"])
        shown = rows[:limit] + ([f"* omitted: {len(rows) - limit}"] if len(rows) > limit else [])
        header = f"<FindToolResult pattern={json.dumps(request['name'])} matches={len(rows)}>"
        return "\n".join([header, *shown, "</FindToolResult>"])

    def matches(self, root: str, pattern: str, kind: str) -> list[str]:
        patterns = self.gitignore_patterns(root)
        if self.default_ignored(root, patterns):
            return []
        rows: list[str] = []
        if os.path.isfile(root):
            return [self.row("file", root)] if kind in {"file", "any"} and self.match_path(root, pattern) else []
        if not os.path.isdir(root):
            raise ToolError("Find path is not a file or directory")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames if name not in self.SKIP_DIRS and not name.startswith(".") and not self.ignored(os.path.join(dirpath, name), patterns)
            ]
            if kind in {"dir", "any"}:
                rows.extend(self.row("dir", os.path.join(dirpath, name)) for name in dirnames if self.match_path(os.path.join(dirpath, name), pattern))
            if kind in {"file", "any"}:
                rows.extend(
                    self.row("file", os.path.join(dirpath, name))
                    for name in filenames
                    if not name.startswith(".")
                    and not self.ignored(os.path.join(dirpath, name), patterns)
                    and self.match_path(os.path.join(dirpath, name), pattern)
                )
        return sorted(rows)

    def match_path(self, path: str, pattern: str) -> bool:
        return fnmatch.fnmatch(os.path.basename(path), pattern) or fnmatch.fnmatch(self.session.relpath(path).replace(os.sep, "/"), pattern)

    def row(self, kind: str, path: str) -> str:
        return f"* {kind}: {self.session.relpath(path)}" + ("/" if kind == "dir" else "")


class SearchTool(Tool):
    NAME = "Search"
    DESCRIPTION = "Search UTF-8 text files with case-insensitive regex; skips binary/hidden/gitignored files and returns path:line:hash matches."
    SIGNATURE = "Search(pattern,path?,glob?,context?) or Search(queries=[...]); pattern is regex, A|B|C is ok"
    EXAMPLE = (
        'Search source with context. Example: {"pattern":"class .*Tool","path":"src","glob":"*.py","context":2}',
        'Search multiple queries. Example: {"queries":[{"pattern":"TODO","glob":"*.py"},{"pattern":"FIXME","path":"tests","glob":"*.py"}]}',
        'Batch regex terms. Example: {"queries":[{"pattern":"done in|elapsed|duration","glob":"*.py","context":2}]}',
    )
    MAX_FILE_BYTES = 2_000_000
    MAX_CONTEXT = 30

    @classmethod
    def arg_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "context": {"type": "integer", "minimum": 0, "maximum": cls.MAX_CONTEXT},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    @classmethod
    def params_schema(cls) -> Json:
        props = dict(cls.arg_schema()["properties"])
        props["queries"] = {"type": "array", "items": cls.arg_schema(), "minItems": 1}
        return {"type": "object", "properties": props, "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return payload.get("queries") or [payload]

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(request["path"]) for request in self.requests())

    def call(self) -> str:
        return "\n\n".join(self.search(request) for request in self.requests())

    def short_args(self) -> list[str]:
        rows = []
        for request in self.requests():
            rel = self.session.relpath(str(request["path"]))
            rows.append(
                " ".join(
                    [
                        json.dumps(request["pattern"], ensure_ascii=False),
                        *(["path=" + rel] if rel != "." else []),
                        *(["glob=" + str(request["glob"])] if request["glob"] else []),
                        *(["C=" + str(request["context"])] if request["context"] else []),
                    ]
                )
            )
        return ["; ".join(rows)]

    def requests(self) -> list[Json]:
        if not self.args:
            raise ToolError("Search requires at least one query object")
        requests = []
        for item in self.args:
            if not isinstance(item, dict):
                raise ToolError("Search args must be query objects")
            if unexpected := sorted(set(item) - {"pattern", "path", "glob", "context"}):
                raise ToolError("Search unexpected field: " + ", ".join(unexpected))
            pattern = str(item.get("pattern") or "").replace("\\n", "\n")
            if not pattern:
                raise ToolError("Search requires pattern")
            context = item.get("context", 0)
            if isinstance(context, bool) or not isinstance(context, int) or context < 0 or context > self.MAX_CONTEXT:
                raise ToolError(f"Search context must be 0..{self.MAX_CONTEXT}")
            requests.append(
                {"pattern": pattern, "path": self.session.resolve_path(str(item.get("path") or ".")), "glob": str(item.get("glob") or ""), "context": context}
            )
        return requests

    def search(self, request: Json) -> str:
        patterns = self.gitignore_patterns(str(request["path"]))
        rows = [] if self.default_ignored(str(request["path"]), patterns) else None
        rows = rows if rows is not None or "\n" in str(request["pattern"]) else self.rg_matches(request)
        rows = rows if rows is not None else self.python_matches(request)
        header = f"<SearchToolResult pattern={json.dumps(request['pattern'])} matches={len(rows)}>"
        return "\n".join([header, *rows, "</SearchToolResult>"])

    def rg_matches(self, request: Json) -> list[str] | None:
        rg = shutil.which("rg")
        if not rg:
            return None
        cmd = [rg, "--json", "--line-number", "--with-filename", "--color=never", "--ignore-case", "--max-filesize", "2M"]
        if request["context"]:
            cmd.extend(["-C", str(request["context"])])
        if request["glob"]:
            cmd.extend(["--glob", str(request["glob"])])
        cmd.extend([str(request["pattern"]), str(request["path"])])
        proc = subprocess.run(cmd, cwd=self.session.cwd, text=True, capture_output=True, timeout=self.session.settings.shell_timeout)
        if proc.returncode == 2:
            proc = subprocess.run(
                [*cmd[:1], "--pcre2", *cmd[1:]], cwd=self.session.cwd, text=True, capture_output=True, timeout=self.session.settings.shell_timeout
            )
        if proc.returncode not in (0, 1):
            return None
        rows = []
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in {"match", "context"}:
                continue
            data = event.get("data") or {}
            path = data.get("path", {}).get("text")
            number = data.get("line_number")
            text = data.get("lines", {}).get("text", "")
            if not path or not isinstance(number, int):
                continue
            prefix = ">" if event["type"] == "match" else " "
            rows.append(self.match_line(prefix, path, number - 1, text))
        return rows

    def files(self, root: str, glob_pattern: str) -> list[str]:
        gitignore = self.gitignore_patterns(root)
        if self.default_ignored(root, gitignore):
            return []
        if os.path.isfile(root):
            return [root]
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in self.SKIP_DIRS and not name.startswith(".") and not self.ignored(os.path.join(dirpath, name), gitignore)
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = os.path.join(dirpath, filename)
                rel = self.session.relpath(path)
                if self.ignored(path, gitignore):
                    continue
                if glob_pattern and not (fnmatch.fnmatch(filename, glob_pattern) or fnmatch.fnmatch(rel, glob_pattern)):
                    continue
                found.append(path)
        return found

    def python_matches(self, request: Json) -> list[str]:
        try:
            regex = re.compile(str(request["pattern"]), re.IGNORECASE | re.MULTILINE)
        except re.error as error:
            raise ToolError(f"invalid regex: {error}") from error
        rows = []
        for path in self.files(str(request["path"]), str(request["glob"])):
            for row, _matched in self.file_matches(path, regex, int(request["context"])):
                rows.append(row)
        return rows

    def file_matches(self, path: str, regex: re.Pattern[str], context: int) -> list[tuple[str, bool]]:
        try:
            if os.path.getsize(path) > self.MAX_FILE_BYTES:
                return []
            with open(path, encoding="utf-8") as file:
                lines = file.readlines()
        except (OSError, UnicodeDecodeError):
            return []
        rows: list[tuple[str, bool]] = []
        content = "".join(lines)
        starts = (
            [content.count("\n", 0, match.start()) for match in regex.finditer(content)]
            if "\n" in regex.pattern
            else [index for index, line in enumerate(lines) if regex.search(line)]
        )
        for index in starts:
            seen = set()
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            for line_index in range(start, end):
                if line_index in seen:
                    continue
                seen.add(line_index)
                prefix = ">" if line_index == index else " "
                rows.append((self.match_line(prefix, path, line_index, lines[line_index]), line_index == index))
        return rows

    def match_line(self, prefix: str, path: str, line_index: int, line: str) -> str:
        return f"{prefix} {self.session.relpath(path)}:{line_index}:{ReadTool.line_hash(line)}|{line.rstrip()}"


class CodeIndex:
    AUTO_UPDATE_LIMIT: ClassVar[int] = 20
    SYMBOLS: ClassVar[dict[str, str]] = {
        "ready": "✓",
        "synced": "✓",
        "stale": "*",
        "syncing": "~",
        "updating": "~",
        "missing": "?",
        "unavailable": "!",
        "error": "!",
    }

    def __init__(self, session: Session):
        self.session = session

    def available(self) -> bool:
        status, message = self.status()
        self.session.state.code_index_error = message if status == "error" else ""
        return status in {"ready", "stale"}

    def set_status(self, status: str, message: str = "") -> None:
        self.session.state.code_index_status = "synced" if status == "ready" else status
        self.session.state.code_index_error = message if status == "error" else ""

    @classmethod
    def label(cls, status: str) -> str:
        return cls.SYMBOLS.get(status, status)

    @classmethod
    def status_line(cls, status: str, message: str = "") -> str:
        status = "synced" if status == "ready" else status
        return f"index{cls.label(status)} {status}" + ((": " + message) if message else "")

    def notice(self, text: str = "", *, refreshing: bool = False) -> None:
        self.session.state.code_index_notice = text
        self.session.state.code_index_refreshing = refreshing
        if text:
            self.session.state.code_index_status = "syncing" if text in {"syncing", "updating"} else text

    def fail(self, error: Any) -> str:
        self.session.state.code_index_error = str(error).strip()
        self.notice("error")
        return self.session.state.code_index_error

    def finish(self, status: str = "synced") -> None:
        self.notice("")
        self.session.state.code_index_error, self.session.state.code_index_status = "", status

    def status(self, *, check: bool = False, max_pending_files: int = 20) -> tuple[str, str]:
        try:
            data = csi.status(self.session.cwd, check=check, max_pending_files=max_pending_files)
        except Exception as error:
            self.set_status("error", str(error))
            return "error", str(error)
        status = str(getattr(data, "status", "") or "error")
        message = str(getattr(data, "message", None) or getattr(data, "reason", None) or "")
        pending = getattr(data, "pending_changes", None)
        files = getattr(data, "pending_files", ()) or ()
        if pending and pending != "unknown":
            sample = ", ".join(str(path) for path in (files or [])[:3])
            message = (message + "; " if message else "") + "pending " + str(pending) + ((" (" + sample + ")") if sample else "")
        preserves_stale = status == "ready" and pending == "unknown" and self.session.state.code_index_status == "stale"
        if not self.session.state.code_index_refreshing and not preserves_stale:
            self.set_status(status, message)
        return status, message

    def sync(self, *, force: bool = False) -> str:
        if self.session.state.code_index_refreshing:
            return "code_index: syncing"
        if force:
            csi.clean(self.session.cwd)
        self.notice("syncing", refreshing=True)
        try:
            csi.index(self.session.cwd)
        except Exception as error:
            return "code_index: error\n" + self.fail(error)
        self.finish()
        status, message = self.status(check=True)
        index_path = os.path.join(self.session.cwd, ".code-symbol-index", "index.sqlite")
        lines = ["code_index: " + ("rebuilt" if force else "synced"), "status: " + status, "path: " + index_path]
        if message:
            lines.append("note: " + message)
        return "\n".join(lines)

    def update(self, paths: list[str]) -> str:
        paths = self.update_paths(paths)
        if not paths or self.session.state.code_index_refreshing or not self.available():
            return ""
        self.notice("updating", refreshing=True)
        try:
            csi.update(paths, root=self.session.cwd)
        except Exception as error:
            return self.fail(error)
        self.finish()
        return "updated " + str(len(paths)) + " file(s)"

    def update_pending(self) -> str:
        if self.session.state.code_index_refreshing:
            return ""
        try:
            data = csi.status(self.session.cwd, check=True, max_pending_files=self.AUTO_UPDATE_LIMIT + 1)
        except Exception:
            return ""
        self.set_status(str(getattr(data, "status", "") or "error"), str(getattr(data, "message", None) or getattr(data, "reason", None) or ""))
        if getattr(data, "status", "") != "stale":
            return ""
        pending = getattr(data, "pending_changes", None)
        files = [str(path) for path in getattr(data, "pending_files", ()) or () if path]
        if not files or len(files) > self.AUTO_UPDATE_LIMIT or (isinstance(pending, int) and pending > self.AUTO_UPDATE_LIMIT):
            return ""
        return self.update([self.session.resolve_path(path) for path in files])

    def refresh_existing_async(self) -> bool:
        if self.session.state.code_index_refreshing:
            return False
        status, _message = self.status()
        if status not in {"ready", "stale"}:
            return False
        self.notice("syncing", refreshing=True)
        try:
            worker = csi.refresh_async(self.session.cwd)
        except Exception as error:
            self.fail(error)
            return False

        def finish() -> None:
            worker.join()
            try:
                self.session.state.code_index_refreshing = False
                self.session.state.code_index_notice = ""
                self.status(check=True)
            except Exception as error:
                self.fail(error)

        threading.Thread(target=finish, daemon=True).start()
        return True

    def update_paths(self, paths: list[str]) -> list[str]:
        paths = [self.session.resolve_path(path) for path in paths]
        return list(dict.fromkeys(path for path in paths if self.session.in_cwd(path) and os.path.isfile(path)))


class InspectCodeTool(Tool):
    NAME = "InspectCode"
    MAX_LIMIT: ClassVar[int] = 80
    MAX_OUTLINE_LIMIT: ClassVar[int] = 1000
    MAX_DEPTH: ClassVar[int] = 5
    MODES: ClassVar[tuple[str, ...]] = ("find", "inspect", "outline", "refs", "impls", "callers", "callees")
    SYMBOL_MODES: ClassVar[frozenset[str]] = frozenset({"find", "inspect", "refs", "impls", "callers", "callees"})
    RESOLVE_MODES: ClassVar[frozenset[str]] = frozenset({"inspect", "refs", "impls", "callers", "callees"})
    CHAIN_MODES: ClassVar[frozenset[str]] = frozenset({"callers", "callees"})
    OPTION_KEYS: ClassVar[tuple[str, ...]] = ("limit", "kind", "path", "symbol", "exact_only", "depth", "offset", "all_kinds", "ref_kind", "loose")
    DESCRIPTION = "Use the code index: find returns symbols; inspect returns anchors/members/references; outline returns a file symbol tree; refs lists classified references; impls lists implementors; callers/callees walk the call chain."
    SIGNATURE = "InspectCode(mode,target,kind?,path?,symbol?,limit?,exact_only?,depth?,offset?,all_kinds?,ref_kind?,loose?)"
    EXAMPLE = (
        'Find symbols; kind can be class|function|method|variable|constant|enum|struct|interface|module|type|trait|field|property|impl|namespace|dict_key, comma-ok. Example: {"mode":"find","target":"Tool","kind":"class,function","limit":20}',
        'Inspect one symbol; path narrows candidates. Example: {"mode":"inspect","target":"Tool","path":"src/app.py"}',
        'Outline one file; symbol narrows subtree. Example: {"mode":"outline","target":"src/app.py","symbol":"App","limit":300}',
        'List references; default hides import/attribute noise, ref_kind filters to call|read|write|inherit|type|import|attribute|usage (comma-ok), all_kinds shows everything, offset pages. Example: {"mode":"refs","target":"Tool","ref_kind":"call,write","offset":0}',
        'List implementors of an interface/base. Example: {"mode":"impls","target":"Tool","kind":"class"}',
        'Walk transitive callers/callees up to depth; callees loose includes ambiguous cross-module matches. Example: {"mode":"callers","target":"handle_job","depth":3}',
    )

    @classmethod
    def params_schema(cls) -> Json:
        props = {
            "mode": {"type": "string", "enum": list(cls.MODES)},
            "target": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": cls.MAX_OUTLINE_LIMIT},
            "kind": {"type": "string"},
            "path": {"type": "string"},
            "symbol": {"type": "string"},
            "exact_only": {"type": "boolean"},
            "depth": {"type": "integer", "minimum": 1, "maximum": cls.MAX_DEPTH},
            "offset": {"type": "integer", "minimum": 0},
            "all_kinds": {"type": "boolean"},
            "ref_kind": {"type": "string"},
            "loose": {"type": "boolean"},
        }
        return {"type": "object", "properties": props, "required": ["mode", "target"], "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        options = {key: payload[key] for key in cls.OPTION_KEYS if key in payload}
        return [str(payload.get("mode") or ""), str(payload.get("target") or ""), *([options] if options else [])]

    def call(self) -> str:
        if len(self.args) not in (2, 3):
            raise ToolError("InspectCode requires mode, target[, options]")
        if not isinstance(self.args[0], str) or not isinstance(self.args[1], str):
            raise ToolError("InspectCode mode and target must be strings")
        mode, target = self.args[0].lower(), self.args[1].strip()
        if len(self.args) == 3 and not isinstance(self.args[2], dict):
            raise ToolError("InspectCode options must be an object")
        options = self.args[2] if len(self.args) == 3 else {}
        if unexpected := sorted(set(options) - set(self.OPTION_KEYS)):
            raise ToolError("InspectCode unexpected option: " + ", ".join(unexpected))
        if mode not in self.MODES:
            raise ToolError("InspectCode mode must be one of: " + ", ".join(self.MODES))
        if not target:
            raise ToolError("InspectCode target is required")
        if mode in self.SYMBOL_MODES and re.search(r"\s", target):
            raise ToolError("InspectCode symbol target must not contain whitespace")
        if mode in self.RESOLVE_MODES and (target.endswith(".py") or os.path.exists(self.session.resolve_path(target))):
            raise ToolError(f"InspectCode {mode} target must be a symbol, not a file")
        if mode == "outline" and not os.path.isfile(self.session.resolve_path(target)):
            raise ToolError("InspectCode outline target must be an existing file")
        limit = options.get("limit")
        max_limit = self.MAX_OUTLINE_LIMIT if mode == "outline" else self.MAX_LIMIT
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > max_limit):
            raise ToolError(f"InspectCode {mode} limit must be 1..{max_limit}")
        depth = options.get("depth")
        if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1 or depth > self.MAX_DEPTH):
            raise ToolError(f"InspectCode depth must be 1..{self.MAX_DEPTH}")
        offset = options.get("offset")
        if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
            raise ToolError("InspectCode offset must be >= 0")
        ref_kind = options.get("ref_kind")
        if ref_kind is not None:
            if not isinstance(ref_kind, str):
                raise ToolError("InspectCode ref_kind must be a string")
            if options.get("all_kinds"):
                raise ToolError("InspectCode ref_kind and all_kinds are mutually exclusive")
            tokens = [token.strip() for token in ref_kind.split(",") if token.strip()]
            if unknown := sorted(set(tokens) - csi.REFERENCE_KINDS):
                raise ToolError("InspectCode unknown ref_kind: " + ", ".join(unknown) + "; valid: " + ", ".join(sorted(csi.REFERENCE_KINDS)))
        index = CodeIndex(self.session)
        if not index.available():
            raise ToolError("code index is not available; run /index")
        try:
            output = self.inspect_text(mode, target, options, limit)
        except csi.CodeSymbolIndexError as error:
            return self.process_result("InspectCodeToolResult", 1, "", str(error))
        return self.process_result("InspectCodeToolResult", 0, str(output), "")

    def inspect_text(self, mode: str, target: str, options: Json, limit: int | None) -> str:
        common = {
            "root": self.session.cwd,
            "kind": options.get("kind") or None,
            "path": options.get("path") or None,
            "exact_only": bool(options.get("exact_only")),
            "format": "text",
        }
        if mode == "find":
            return csi.search(target, limit=limit or csi.DEFAULT_SEARCH_LIMIT, **common)
        if mode == "inspect":
            return csi.inspect(target, limit=limit or csi.DEFAULT_PAGE_LIMIT, anchors=True, **common)
        if mode == "refs":
            ref_kinds = options.get("ref_kind") or ("all" if options.get("all_kinds") else "behavioral")
            return csi.refs(target, limit=limit or csi.DEFAULT_MAX_REFERENCES, offset=int(options.get("offset") or 0), ref_kinds=ref_kinds, **common)
        if mode == "impls":
            return csi.impls(target, limit=limit or csi.DEFAULT_MAX_IMPLEMENTORS, offset=int(options.get("offset") or 0), **common)
        if mode in self.CHAIN_MODES:
            depth = int(options.get("depth") or 3)
            if mode == "callees":
                return csi.callees(target, limit=limit or csi.DEFAULT_MAX_CALLEES, depth=depth, loose=bool(options.get("loose")), **common)
            return csi.callers(target, limit=limit or csi.DEFAULT_MAX_CALLERS, depth=depth, **common)
        symbol = options.get("symbol") or None
        return csi.outline(
            target, root=self.session.cwd, symbol=str(symbol) if symbol else None, max_symbols=limit or csi.DEFAULT_MAX_OUTLINE_SYMBOLS, format="text"
        )


@dataclass
class Edit:
    op: str
    start: str = ""
    end: str = ""
    content: str = ""
    old: str = ""
    new: str = ""


class EditTool(Tool):
    NAME = "Edit"
    DESCRIPTION = "Create or patch one UTF-8 file; op=create makes a new file; Edit start/end anchors are inclusive."
    SIGNATURE = "Edit(path, edits=[{op,start?,end?,content?,old?,new?}]); ops=create|replace|delete|insert_before|insert_after|replace_all"
    EXAMPLE = (
        'create file. Example: {"path":"src/app.py","edits":[{"op":"create","content":"print(1)\\n"}]}',
        'replace range. Example: {"path":"src/app.py","edits":[{"op":"replace","start":"10:1ab2c","end":"12:3de4f","content":"new_value = 1\\n"}]}',
        'delete range. Example: {"path":"src/app.py","edits":[{"op":"delete","start":"20:0aa11","end":"22:0bb22"}]}',
        'insert_before line. Example: {"path":"src/app.py","edits":[{"op":"insert_before","start":"30:0cc33","content":"setup()\\n"}]}',
        'insert_after line. Example: {"path":"src/app.py","edits":[{"op":"insert_after","start":"40:0dd44","content":"cleanup()\\n"}]}',
        'replace_all exact text; do not mix with anchored ops. Example: {"path":"src/app.py","edits":[{"op":"replace_all","old":"OldName","new":"NewName"}]}',
    )
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        edit = {
            "type": "object",
            "properties": {key: {"type": "string"} for key in ("op", "start", "end", "content", "old", "new")},
            "required": ["op"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": edit, "minItems": 1}},
            "required": ["path", "edits"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [payload.get("path", ""), payload.get("edits", [])]

    def call(self) -> str:
        path, original, created, new_content, changes = self.build()
        if new_content == original and not created:
            raise ToolError("edit produced no changes")
        if created:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(new_content)
        return "\n".join(
            [
                f"<Edit path={json.dumps(self.session.relpath(path))}>",
                self.file_stat(path),
                self.diff(path, original, new_content).rstrip(),
                self.edit_context(new_content, changes),
                "</Edit>",
            ]
        )

    def preview(self) -> str:
        path, original, _created, new_content, _changes = self.build()
        if new_content == original and os.path.exists(path):
            raise ToolError("edit produced no changes")
        return self.diff(path, original, new_content) or f"Edit({path})"

    def short_args(self) -> list[str]:
        path, _edits = self.parse()
        return [self.session.relpath(path)]

    def diff(self, path: str, original: str, new_content: str) -> str:
        relpath = self.session.relpath(path)
        return "".join(
            difflib.unified_diff(
                original.splitlines(True),
                new_content.splitlines(True),
                fromfile="/dev/null" if not original and not os.path.exists(path) else relpath,
                tofile=relpath,
            )
        )

    def parse(self) -> tuple[str, list[Edit]]:
        if len(self.args) != 2:
            raise ToolError("Edit requires path and edits")
        if not isinstance(self.args[0], str):
            raise ToolError("Edit path must be a string")
        path = self.session.resolve_path(str(self.args[0]))
        raw_edits = self.args[1]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ToolError("Edit edits must be a non-empty array")
        edits = []
        for item in raw_edits:
            if not isinstance(item, dict):
                raise ToolError("each edit must be an object")
            if unexpected := sorted(set(item) - {"op", "start", "end", "content", "old", "new"}):
                raise ToolError("Edit unexpected field: " + ", ".join(unexpected))
            op = str(item.get("op") or "")
            if op not in {"create", "replace", "delete", "insert_before", "insert_after", "replace_all"}:
                raise ToolError("unknown edit op")
            if op == "create" and len(raw_edits) != 1:
                raise ToolError("create cannot be mixed with other edits")
            if op in {"replace", "delete"} and (not item.get("start") or not item.get("end")):
                raise ToolError(f"{op} requires start and end anchors")
            if op in {"insert_before", "insert_after"} and not item.get("start"):
                raise ToolError(f"{op} requires start anchor")
            edits.append(
                Edit(
                    op=op,
                    start=str(item.get("start") or ""),
                    end=str(item.get("end") or ""),
                    content=self.content_text(str(item.get("content") or "")),
                    old=self.normalize_text(str(item.get("old") or "")),
                    new=self.content_text(str(item.get("new") or "")),
                )
            )
        return path, edits

    def build(self) -> tuple[str, str, bool, str, list[tuple[int, int, int, int]]]:
        path, edits = self.parse()
        creating = edits[0].op == "create"
        if os.path.exists(path):
            if creating:
                raise ToolError("file already exists")
            with open(path, encoding="utf-8") as file:
                original = file.read()
            created = False
        elif creating:
            parent = os.path.dirname(path) or "."
            if not self.session.in_cwd(parent):
                raise ToolError("refusing to create parent directories outside workspace")
            original, created = "", True
        else:
            raise ToolError("file does not exist; use op=create to create it")
        new_content, changes = self.apply(original, edits)
        return path, original, created, new_content, changes

    def apply(self, original: str, edits: list[Edit]) -> tuple[str, list[tuple[int, int, int, int]]]:
        if edits[0].op == "create":
            lines = self.content_lines(edits[0].content, False)
            return "".join(lines), [(0, 0, 0, len(lines))]
        if any(edit.op == "replace_all" for edit in edits):
            if any(edit.op != "replace_all" for edit in edits):
                raise ToolError("replace_all cannot be mixed with anchored edits")
            content = original
            for edit in edits:
                if not edit.old and content:
                    raise ToolError("replace_all requires old")
                if edit.old and edit.old not in content:
                    raise ToolError("replace_all old text not found")
                content = content.replace(edit.old, edit.new)
            return content, [(0, 0, 0, len(content.splitlines(True)))]
        lines = original.splitlines(True)
        replacements = []
        for edit in edits:
            start = self.resolve_anchor(lines, edit.start)
            if edit.op in {"replace", "delete"}:
                end = self.resolve_anchor(lines, edit.end)
                if end < start:
                    raise ToolError("end anchor is before start anchor")
                replacement = [] if edit.op == "delete" else self.content_lines(edit.content, end + 1 < len(lines))
                replacements.append((start, end + 1, replacement))
            elif edit.op in {"insert_before", "insert_after"}:
                index = start if edit.op == "insert_before" else start + 1
                replacements.append((index, index, self.content_lines(edit.content, index < len(lines))))
            else:
                raise ToolError("unknown edit op")
        previous = None
        for start, end, _ in sorted(replacements):
            if previous and (start < previous[1] or (start == previous[0] and end == previous[1])):
                raise ToolError(f"edits overlap or share an insertion point: {previous[0]}:{previous[1]} and {start}:{end}")
            previous = (start, end)
        new_lines = list(lines)
        for start, end, replacement in sorted(replacements, reverse=True):
            new_lines[start:end] = replacement
        changes = []
        delta = 0
        for start, end, replacement in sorted(replacements):
            new_start = start + delta
            new_end = new_start + len(replacement)
            clear_end = 0 if len(replacement) != end - start else new_start + (end - start)
            changes.append((new_start, clear_end, new_start, new_end))
            delta += len(replacement) - (end - start)
        return "".join(new_lines), changes

    def edit_context(self, content: str, changes: list[tuple[int, int, int, int]]) -> str:
        lines = content.splitlines(True)
        out = []
        for clear_start, clear_end, start, end in changes:
            out.append(f"<invalidate>{clear_start}:{clear_end}</invalidate>")
            shown = lines[start:end]
            if shown:
                out.append("<content hashline-numbered>")
                out.extend(f"{start + index}:{ReadTool.line_hash(line)}|{line.rstrip(chr(10))}" for index, line in enumerate(shown))
                out.append("</content>")
        return "\n".join(out)

    def content_lines(self, content: str, followed_by_more: bool) -> list[str]:
        content = self.normalize_text(content)
        if content == "":
            return []
        lines = content.splitlines(True)
        if followed_by_more and lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def content_text(cls, value: str) -> str:
        value = cls.normalize_text(value)
        return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t") if "\n" not in value and "\\n" in value else value

    def resolve_anchor(self, lines: list[str], anchor: str) -> int:
        match = re.fullmatch(r"(\d+):([0-9a-z]{5})", anchor.split("|", 1)[0].strip())
        if not match:
            raise ToolError('invalid anchor; use "line:hash" from Read or Search')
        index = int(match.group(1))
        if index >= len(lines):
            raise ToolError("anchor line out of range")
        expected = match.group(2).lower()
        actual = ReadTool.line_hash(lines[index])
        if actual != expected:
            current = f"{index}:{actual}|{lines[index].rstrip()}"
            raise ToolError(f"stale anchor {anchor}; current is {current}")
        return index


class BashTool(Tool):
    NAME = "Bash"
    DESCRIPTION = "Run one bash shell invocation in the workspace; returns exit_code/stdout/stderr and shows live output. Avoid unbounded output; limit noisy commands with head/tail/sed/rg filters or command-specific limits, and inspect large outputs in chunks."
    SIGNATURE = "Bash(command)"
    EXAMPLE = (
        'Check environment. Example: {"command":"python3 --version"}',
        'Run a project command. Example: {"command":"python3 -m py_compile nanocode.py"}',
    )
    MUTATES = True
    live_output: Callable[[str, str], None] | None = None

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {"command": {"type": "string", "minLength": 1, "pattern": "^.*\\S.*$"}},
            "required": ["command"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [payload.get("command", "")]

    def command(self) -> str:
        command = self.strings(min_count=1, max_count=1)[0]
        if not command.strip():
            raise ToolError("Bash command must be non-empty")
        return command

    def short_args(self) -> list[str]:
        return [self.command()]

    def call(self) -> str:
        command = self.command()
        bash = shutil.which("bash") or "bash"
        proc = None
        try:
            proc = subprocess.Popen(
                [bash, "-lc", command], cwd=self.session.cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
            )
            assert proc.stdout is not None and proc.stderr is not None
            return self.stream_process(proc)
        except KeyboardInterrupt:
            stdout, stderr = self.kill_and_collect(proc)
            return self.process_result("BashToolResult", -1, stdout, stderr + ("\n" if stderr else "") + "interrupted")
        finally:
            if self.live_output is not None:
                self.live_output("", "")

    def stream_process(self, proc: subprocess.Popen[bytes]) -> str:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        timed_out = False
        deadline = time.monotonic() + self.session.settings.shell_timeout
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    self.kill_process_group(proc)
                    proc.wait()
                    self.drain_selector(selector, stdout_parts, stderr_parts)
                    break
                for key, _ in selector.select(min(0.2, remaining)):
                    self.read_stream_chunk(selector, key, stdout_parts, stderr_parts)
            if proc.returncode is None:
                proc.wait()
        finally:
            selector.close()
        stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
        if timed_out:
            stderr += ("\n" if stderr else "") + "timeout"
            return self.process_result("BashToolResult", -1, stdout, stderr)
        return self.process_result("BashToolResult", proc.returncode or 0, stdout, stderr)

    def drain_selector(self, selector: selectors.BaseSelector, stdout_parts: list[str], stderr_parts: list[str]) -> None:
        for key in list(selector.get_map().values()):
            while self.read_stream_chunk(selector, key, stdout_parts, stderr_parts):
                pass

    def read_stream_chunk(
        self,
        selector: selectors.BaseSelector,
        key: selectors.SelectorKey,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> bool:
        try:
            data = os.read(key.fileobj.fileno(), 4096)
        except OSError:
            data = b""
        if not data:
            try:
                selector.unregister(key.fileobj)
            except Exception:
                pass
            try:
                key.fileobj.close()
            except Exception:
                pass
            return False
        text = data.decode("utf-8", errors="replace")
        (stdout_parts if key.data == "stdout" else stderr_parts).append(text)
        if self.live_output is not None:
            self.live_output(str(key.data), text)
        return True

    @staticmethod
    def kill_process_group(proc: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()

    @classmethod
    def kill_and_collect(cls, proc: subprocess.Popen[Any] | None) -> tuple[str, str]:
        if proc is None:
            return "", ""
        cls.kill_process_group(proc)
        stdout, stderr = proc.communicate()
        return tuple(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or "" for value in (stdout, stderr))


class GitTool(Tool):
    NAME = "Git"
    DESCRIPTION = 'Run git with explicit argv (default: cwd from Environment; use cwd= for other directories). For add, pass explicit file paths; broad add is rejected.'
    SIGNATURE = "Git(argv=[command,...], cwd?)"
    EXAMPLE = (
        'Status. Example: {"argv":["status","--short"]}',
        'Diff. Example: {"cwd":"src","argv":["diff","--","app.py"]}',
        'Stage explicit files. Example: {"argv":["add","--","nanocode.py","README.md"]}',
        'Show commit/file. Example: {"argv":["show","--stat","HEAD"]}',
    )
    READONLY = {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame"}
    BROAD_ADD = {"-A", "--all", ".", "./", ":/", "*"}

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {"cwd": {"type": "string"}, "argv": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
            "required": ["argv"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        argv = list(payload.get("argv") or [])
        return [("cwd=" + str(payload["cwd"])), *argv] if payload.get("cwd") else argv

    def needs_confirmation(self) -> bool:
        args, _ = self.git_args()
        return not args or args[0] not in self.READONLY

    def call(self) -> str:
        args, cwd = self.git_args()
        git = shutil.which("git")
        if not git:
            raise ToolError("git not found")
        try:
            proc = subprocess.run([git, *args], cwd=cwd, text=True, capture_output=True, timeout=self.session.settings.shell_timeout)
            return self.process_result("GitToolResult", proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as error:
            return self.process_result("GitToolResult", -1, error.stdout or "", (error.stderr or "") + "\ntimeout")

    def git_args(self) -> tuple[list[str], str]:
        if not self.args:
            raise ToolError("Git requires arguments")
        args = [str(arg) for arg in self.args]
        cwd = self.session.cwd
        if args[0].startswith("cwd="):
            raw_cwd = args.pop(0)[4:]
            if not raw_cwd:
                raise ToolError("cwd= requires a path")
            cwd = self.session.resolve_path(raw_cwd)
            if not self.session.in_cwd(cwd):
                raise ToolError("git cwd outside workspace")
            if not os.path.isdir(cwd):
                raise ToolError("git cwd is not a directory")
        if not args:
            raise ToolError("Git requires arguments")
        self.validate_add(args, cwd)
        return args, cwd

    def validate_add(self, args: list[str], cwd: str) -> None:
        if args[0] != "add":
            return
        paths, explicit = [], False
        for arg in args[1:]:
            if arg == "--":
                explicit = True
            elif arg in self.BROAD_ADD or arg.startswith("--pathspec-from-file"):
                raise ToolError("Git add requires explicit file paths")
            elif explicit or not arg.startswith("-"):
                paths.append(arg)
        if not paths:
            raise ToolError("Git add requires explicit file paths")
        for path in paths:
            if path.startswith(":") or any(char in path for char in "*?[]") or not self.session.in_cwd(os.path.abspath(os.path.join(cwd, path))):
                raise ToolError("Git add requires explicit file paths inside workspace")


class RecallTool(Tool):
    NAME = "Recall"
    DESCRIPTION = "Recall stored non-Recall tool results by tr.N key; ranges slice output lines to control context."
    SIGNATURE = "Recall(keys=[tr.N,...], ranges?); ranges are 0-based [start,end] output lines"
    EXAMPLE = (
        'Recall full result. Example: {"keys":["tr.1"]}',
        'Recall output line ranges. Example: {"keys":["tr.1","tr.2"],"ranges":[[0,80]]}',
    )
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string", "pattern": "^tr\\.\\d+$"}, "minItems": 1},
                "ranges": {"type": "array", "items": cls.RANGE_SCHEMA, "minItems": 1},
            },
            "required": ["keys"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [{"keys": payload.get("keys", []), **({"ranges": payload["ranges"]} if "ranges" in payload else {})}]

    def call(self) -> str:
        requests = self.requests()
        if not requests:
            raise ToolError("Recall requires at least one key")
        chunks = ["<RecallToolResult>"]
        for key, ranges in requests:
            value = self.session.tool_results.get(key)
            if value is None:
                chunks.append(f"* {key}: missing")
                continue
            chunks.append(f"<Result key={json.dumps(key)}>")
            chunks.append(self.slice(value, ranges).rstrip())
            chunks.append("</Result>")
        chunks.append("</RecallToolResult>")
        return "\n".join(chunks)

    def short_args(self) -> list[str]:
        rows = []
        for key, ranges in self.requests():
            row = key
            if ranges:
                row += " " + ",".join(f"{start}:{end}" for start, end in ranges)
            rows.append(row)
        return ["; ".join(rows)]

    def requests(self) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
        if len(self.args) != 1 or not isinstance(self.args[0], dict):
            raise ToolError("Recall requires keys")
        payload = self.args[0]
        if unexpected := sorted(set(payload) - {"keys", "ranges"}):
            raise ToolError("Recall unexpected field: " + ", ".join(unexpected))
        raw_keys = payload.get("keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ToolError("Recall requires keys")
        ranges = self.parse_ranges(payload)
        keys = []
        for item in raw_keys:
            key = str(item).strip()
            if not re.fullmatch(r"tr\.\d+", key):
                raise ToolError("Recall key must look like tr.N")
            keys.append(key)
        return list(dict.fromkeys((key, ranges) for key in keys))

    def parse_ranges(self, payload: Json) -> tuple[tuple[int, int], ...]:
        raw = payload.get("ranges")
        if raw is None:
            return ()
        if not isinstance(raw, list) or not raw:
            raise ToolError("Recall ranges must be a non-empty array")
        return tuple(self.line_range(item, "Recall range") for item in raw)

    @staticmethod
    def slice(value: str, ranges: tuple[tuple[int, int], ...]) -> str:
        if not ranges:
            return value
        lines = value.splitlines()
        return "\n".join("\n".join(lines[start:end]) for start, end in ranges if end > start)


class NoteTool(Tool):
    NAME = "Note"
    DESCRIPTION = "Maintain durable working notes; goal, plan, and check replace current values, known appends unique facts."
    SIGNATURE = "Note(goal?, plan?, known?, check?)"
    EXAMPLE = ('Set memory. Example: {"goal":"ship parser fix","plan":["inspect parser","patch bug"],"known":["tests use pytest"],"check":"pytest passed"}',)
    STORES_RESULT = False

    @classmethod
    def params_schema(cls) -> Json:
        strings = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        return {"type": "object", "properties": {"goal": {"type": "string"}, "plan": strings, "known": strings, "check": {"type": "string"}}, "additionalProperties": False}

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [payload]

    def call(self) -> str:
        if len(self.args) != 1 or not isinstance(self.args[0], dict):
            raise ToolError("Note requires named fields")
        data = self.args[0]
        if unexpected := sorted(set(data) - {"goal", "plan", "known", "check"}):
            raise ToolError("Note unexpected field: " + ", ".join(unexpected))
        changed = []
        if "goal" in data:
            self.session.state.goal = str(data["goal"]).strip()
            changed.append("goal")
        if "plan" in data:
            if not isinstance(data["plan"], list):
                raise ToolError("Note plan must be an array")
            self.session.state.plan = [AgentState.plan_text(str(item)) for item in data["plan"] if AgentState.plan_text(str(item))]
            changed.append("plan")
        if "known" in data:
            if not isinstance(data["known"], list):
                raise ToolError("Note known must be an array")
            self.session.state.known = list(dict.fromkeys([*self.session.state.known, *(str(item).strip() for item in data["known"] if str(item).strip())]))
            changed.append("known")
        if "check" in data:
            self.session.state.check = str(data["check"]).strip()
            changed.append("check")
        if not changed:
            raise ToolError("Note requires goal, plan, known, or check")
        return "Updated memory: " + ", ".join(changed)

    def short_args(self) -> list[str]:
        data = self.args[0] if self.args and isinstance(self.args[0], dict) else {}
        lines = []
        if goal := str(data.get("goal") or "").strip():
            lines.append("goal -> " + Tool.compact(goal, 120))
        if isinstance(data.get("plan"), list):
            lines.extend(["plan:", *(f"  {row}" for row in AgentState.plan_rows_for(data["plan"], status=True) if row != "- (empty)")])
        if isinstance(data.get("known"), list):
            known = [Tool.compact(item, 120) for item in data["known"] if str(item).strip() and str(item).strip() not in self.session.state.known]
            if known:
                lines.extend(["known:", *(f"  + {item}" for item in known)])
        if check := str(data.get("check") or "").strip():
            lines.append("check -> " + Tool.compact(check, 120))
        return ["\n".join(lines) or "{}"]


class MCPTool(Tool):
    NAME = "MCP"
    DESCRIPTION = "Call or describe external MCP server tools"
    SIGNATURE = 'MCP(action="call"|"describe", server, tool, arguments={})'
    MUTATES = True

    @classmethod
    def params_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["call", "describe"],
                    "description": '"call" invokes the tool; "describe" returns metadata',
                },
                "server": {
                    "type": "string",
                    "description": "MCP server name from config",
                },
                "tool": {
                    "type": "string",
                    "description": "Remote MCP tool name",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the remote tool (required for call)",
                },
            },
            "required": ["action", "server", "tool"],
            "additionalProperties": False,
        }

    @classmethod
    def payload_args(cls, payload: Json) -> list[Any]:
        return [payload]

    def payload(self) -> Json:
        if len(self.args) != 1 or not isinstance(self.args[0], dict):
            raise ToolError("MCP requires named fields")
        return self.args[0]

    def needs_confirmation(self) -> bool:
        payload = self.payload()
        action = payload.get("action", "")
        if action == "describe":
            return False
        if action != "call" or self.session.mcp is None:
            return False
        return self.session.mcp.tool_needs_confirmation(str(payload.get("server") or ""), str(payload.get("tool") or ""))

    def short_args(self) -> list[str]:
        payload = self.payload()
        action = str(payload.get("action") or "")
        server = str(payload.get("server") or "")
        tool_name = str(payload.get("tool") or "")
        target = (server + "." + tool_name).strip(".")
        return [part for part in (action, target) if part]

    def call(self) -> str:
        payload = self.payload()
        action = payload.get("action", "")
        server = payload.get("server", "")
        tool_name = payload.get("tool", "")
        arguments = payload.get("arguments", {})
        if action == "call" and not isinstance(arguments, dict):
            raise ToolError("MCP arguments must be an object")

        mcp = self.session.mcp
        if mcp is None:
            raise ToolError("MCP not configured")

        if action == "describe":
            return mcp.describe_tool(server, tool_name)
        elif action == "call":
            return mcp.call_tool(server, tool_name, arguments)
        else:
            raise ToolError(f"unknown MCP action: {action}")

TOOLS: tuple[type[Tool], ...] = (
    MCPTool,
    ReadTool,
    LineCountTool,
    ListTool,
    FindTool,
    InspectCodeTool,
    SearchTool,
    EditTool,
    BashTool,
    GitTool,
    RecallTool,
    NoteTool,
)
TOOL_REGISTRY: dict[str, type[Tool]] = {tool.NAME: tool for tool in TOOLS}


@dataclass
class ToolCall:
    id: str
    name: str
    args: list[Any]


class ContextManager:
    COMPACT_TITLE: ClassVar[str] = "--- Prior Conversation Summary (compacted) ---"
    COMPACT_RECENT_MESSAGES: ClassVar[int] = 8
    MCP_DETAILS_BUDGET: ClassVar[int] = 8_000
    MCP_DETAILS_MAX_TOOLS: ClassVar[int] = 20
    MCP_DETAILS_MAX_TOOL_CHARS: ClassVar[int] = 2_000
    CODE_EXTENSIONS: ClassVar[set[str]] = set(
        ".c .cc .cpp .cxx .css .go .h .hpp .html .java .js .json .jsx .kt .lua .php .py .rb .rs .scss .sh .sql "
        ".swift .toml .ts .tsx .vue .yaml .yml".split()
    )
    CODE_FILENAMES: ClassVar[set[str]] = {"CMakeLists.txt", "Dockerfile", "Makefile", "go.mod", "package.json", "pyproject.toml"}

    @dataclass
    class FileContextItem:
        order: int
        phase: int
        kind: str
        source: str
        tool: str
        path: str
        start: int
        end: int
        line: str
        mtime_ns: int
        size: int

    def __init__(self, session: Session):
        self.session = session

    def model_messages(self, base_system: str, turn_messages: list[Json] | None = None) -> list[Json]:
        file_context = self.file_context() or "(empty)"
        mcp_tools = self.mcp_tools_context()
        mcp_details = self.mcp_tool_details()

        messages: list[Json] = [
            {"role": "system", "content": base_system.strip()},
            {"role": "user", "content": "--- Environment ---\n" + (self.environment() or "(empty)")},
        ]

        if mcp_tools:
            messages.append({"role": "user", "content": mcp_tools})

        messages.extend(self.session.messages)
        messages.extend(turn_messages or [])
        messages.append({"role": "user", "content": "--- Memory ---\n" + (self.memory_context(with_date=True) or "(empty)")})

        if mcp_details:
            messages.append({"role": "user", "content": mcp_details})

        messages.append({"role": "user", "content": "--- FILE STATE ---\n" + file_context})
        return Text.value(messages)


    def mcp_tools_context(self) -> str:
        if self.session.mcp is None:
            return ""
        return self.session.mcp.render_tools_index()

    def mcp_tool_details(self) -> str:
        items = self.active_mcp_tool_details()
        if not items:
            return ""

        chunks: list[str] = []
        chunks.append("--- MCP TOOL DETAILS ---")
        chunks.append('Details previously requested with MCP(action="describe").')
        chunks.append('Use MCP(action="call", server, tool, arguments) to call them.')
        chunks.append("")

        for _source_order, server, tool, key, body in items:
            detail_block = f"{server}.{tool} source={key}\n{body}"
            chunks.append(detail_block)
            chunks.append("")

        return "\n".join(chunks).strip()

    def active_mcp_tool_details(self) -> list[tuple[int, str, str, str, str]]:
        details: dict[tuple[str, str], tuple[int, str, str]] = {}
        for order, record in enumerate(self.session.tool_records):
            if record.name != "MCP":
                continue
            for match in re.finditer(r'<MCPDescribe server=(".*?") tool=(".*?")>(.*?)</MCPDescribe>', record.output, re.DOTALL):
                try:
                    server = str(json.loads(match.group(1)))
                    tool = str(json.loads(match.group(2)))
                    body = match.group(3).strip()
                except (json.JSONDecodeError, ValueError):
                    continue
                details[(server, tool)] = (order, record.key, body)

        items = sorted((order, server, tool, key, body) for (server, tool), (order, key, body) in details.items())
        items = items[-self.MCP_DETAILS_MAX_TOOLS :]
        active: list[tuple[int, str, str, str, str]] = []
        total_chars = len("--- MCP TOOL DETAILS ---\n") + 160
        truncated = False
        for order, server, tool, key, body in items:
            header = f"{server}.{tool} source={key}"
            detail_block = f"{header}\n{body}"
            if len(detail_block) > self.MCP_DETAILS_MAX_TOOL_CHARS:
                body = body[: max(0, self.MCP_DETAILS_MAX_TOOL_CHARS - len(header) - 24)] + "\n... detail truncated"
                detail_block = f"{header}\n{body}"
                truncated = True
            if total_chars + len(detail_block) + 1 > self.MCP_DETAILS_BUDGET:
                truncated = True
                break
            active.append((order, server, tool, key, body))
            total_chars += len(detail_block) + 1
        if truncated and active:
            order, server, tool, key, body = active[-1]
            active[-1] = (order, server, tool, key, body + '\n... MCP tool details truncated; use MCP(action="describe", server, tool) again if needed.')
        return active

    def update_percent(self, messages: list[Json]) -> int:
        tokens = self.estimated_tokens(messages)
        self.session.state.context_percent = min(100, tokens * 100 // self.session.settings.max_context_tokens)
        return self.session.state.context_percent

    def maybe_compact(self, model: "ModelClient", base_system: str, turn_messages: list[Json] | None = None) -> None:
        if not self.over_budget(base_system, turn_messages):
            return
        compacted, keep = self.compaction_parts()
        if compacted:
            try:
                self.apply_compaction(model.compact(self.compaction_input(compacted)), keep, turn_messages)
            except Exception:
                self.apply_compaction_fallback(keep, turn_messages)
        if turn_messages is not None and self.over_budget(base_system, turn_messages):
            compacted, keep = self.turn_compaction_parts(turn_messages)
            if compacted:
                try:
                    self.apply_turn_compaction(model.compact(self.compaction_input(compacted)), keep, turn_messages)
                except Exception:
                    self.apply_turn_compaction_fallback(keep, turn_messages)

    def over_budget(self, base_system: str, turn_messages: list[Json] | None = None) -> bool:
        return self.estimated_tokens(self.model_messages(base_system, turn_messages)) >= self.session.settings.max_context_tokens

    def memory_context(self, *, with_date: bool = False) -> str:
        rows = [
            "Goal: " + (self.session.state.goal or "(empty; use Note for multi-step work)"),
            "Plan:\n" + "\n".join(self.session.state.plan_rows() or ["- (empty; use Note for a short plan)"]),
            "Known:\n" + "\n".join("- " + item for item in self.session.state.known or ["(empty)"]),
            "Check: " + (self.session.state.check or "(empty)"),
        ]
        if with_date:
            rows.append("Date: " + datetime.now().astimezone().strftime("%Y-%m-%d"))
        return "\n\n".join(rows)

    def environment(self) -> str:
        info = self.session.system_info
        return "\n".join(
            [
                f"- cwd: {info.cwd}",
                f"- os: {info.os}",
                f"- arch: {info.arch}",
                f"- shell_timeout: {self.session.settings.shell_timeout}s",
                "- detected_commands: " + (", ".join(info.commands) or "(none)"),
            ]
        )

    def file_context(self) -> str:
        lines_by_path, omitted = self.active_file_lines()
        return self.render_file_lines(lines_by_path, omitted)

    def active_file_lines(self) -> tuple[dict[str, dict[int, tuple[str, str, str]]], dict[str, dict[str, int]]]:
        lines_by_path: dict[str, dict[int, tuple[str, str, str]]] = {}
        omitted: dict[str, dict[str, int]] = {}
        items = sorted(self.file_items(), key=lambda item: (item.order, item.phase, item.path, item.start))
        wanted: dict[str, set[int]] = {}
        for item in items:
            if item.kind == "line":
                wanted.setdefault(item.path, set()).add(item.start)
        current_lines: dict[str, dict[int, str] | None] = {}
        current_stats: dict[str, tuple[int, int] | None] = {}
        for item in items:
            file_lines = lines_by_path.setdefault(item.path, {})
            if item.kind == "clear":
                for number in list(file_lines):
                    if number >= item.start and (item.end == 0 or number < item.end):
                        del file_lines[number]
                continue
            if self.item_current(item, wanted, current_stats, current_lines):
                file_lines[item.start] = (item.source, item.tool, item.line)
            else:
                omitted.setdefault(item.path, {}).setdefault(item.source, 0)
                omitted[item.path][item.source] += 1
        return lines_by_path, omitted

    def file_items(self) -> list[ContextManager.FileContextItem]:
        items: list[ContextManager.FileContextItem] = []
        for order, record in enumerate(self.session.tool_records, start=1):
            if record.name not in {"Read", "Edit"}:
                continue
            for block in re.finditer(r"(?s)<(Read|Edit)\s+path=(\".*?\").*?>(.*?)</\1>", record.output):
                try:
                    path = str(json.loads(block.group(2)))
                except json.JSONDecodeError:
                    continue
                body = block.group(3)
                stat = self.output_stat(body)
                for match in re.finditer(r"<invalidate>(\d+):(\d+)</invalidate>", body):
                    items.append(self.FileContextItem(order, 0, "clear", record.key, record.name, path, int(match.group(1)), int(match.group(2)), "", *stat))
                for match in re.finditer(r"(?s)<content hashline-numbered>\n(.*?)\n</content>", body):
                    for line in match.group(1).splitlines():
                        line_match = re.match(r"(\d+):[0-9a-z]{5}\|", line)
                        if line_match:
                            items.append(self.FileContextItem(order, 1, "line", record.key, record.name, path, int(line_match.group(1)), 0, line, *stat))
        return items

    def file_count(self) -> int:
        return len({item.path for item in self.file_items() if item.kind == "line"})

    def item_current(
        self,
        item: ContextManager.FileContextItem,
        wanted: dict[str, set[int]],
        current_stats: dict[str, tuple[int, int] | None],
        current_lines: dict[str, dict[int, str] | None],
    ) -> bool:
        if item.path not in current_stats:
            current_stats[item.path] = self.current_stat(item.path)
        stat = current_stats[item.path]
        if stat is None:
            return False
        if item.mtime_ns > 0 and item.size >= 0 and stat == (item.mtime_ns, item.size):
            return True
        if item.path not in current_lines:
            current_lines[item.path] = self.read_lines(item.path, wanted.get(item.path, set()))
        lines = current_lines[item.path]
        hash_match = re.match(r"\d+:([0-9a-z]{5})\|", item.line)
        return bool(lines is not None and hash_match and item.start in lines and ReadTool.line_hash(lines[item.start]) == hash_match.group(1))

    def render_file_lines(self, lines_by_path: dict[str, dict[int, tuple[str, str, str]]], omitted: dict[str, dict[str, int]]) -> str:
        def recent(path: str) -> int:
            return max(
                (int(source[3:]) for source, _tool, _line in lines_by_path[path].values() if source.startswith("tr.") and source[3:].isdigit()),
                default=-1,
            )

        paths = sorted((path for path in lines_by_path if lines_by_path[path]), key=lambda path: (-recent(path), path))
        code_edits = self.recent_code_edits()
        check_status = self.check_status(code_edits)
        focus, actions, errors = self.session.state.current_focus(), self.recent_file_actions(), self.recent_tool_errors()
        if not paths and not omitted and not focus and not actions and not code_edits and not check_status and not errors:
            return ""
        chunks = ["Read/Edit outputs update this section. Treat listed ranges as current file state."] if paths else []
        if focus:
            chunks.extend(["", "Current focus: " + focus])
        if paths:
            chunks.extend(["", "Files:"])
            for path in paths:
                chunks.extend(f"- {path} {start}:{end} current" for start, end in self.coverage(lines_by_path[path]))
        if actions:
            chunks.extend(["", "Recent file events:", *actions])
        if code_edits:
            chunks.extend(["", "Recent code edits:", *code_edits])
        if check_status:
            chunks.extend(["", "Check status:", *check_status])
        if errors:
            chunks.extend(["", "Recent tool errors:", *errors])
        if paths:
            chunks.extend(["", "Content:", "Format: line:hash|text. Use line:hash as edit anchors."])
            for path in paths:
                for start, end, source, tool, segment_lines in self.segments(lines_by_path[path]):
                    chunks.append(f"@@ {path} {start}:{end} current source={source} tool={tool}")
                    chunks.extend(segment_lines)
                chunks.append("")
        if omitted:
            chunks.append("Omitted content:")
            for path in sorted(omitted):
                chunks.extend(f"- {path} source={source} lines={count}" for source, count in sorted(omitted[path].items()))
        return "\n".join(chunks).strip() if len(chunks) > 4 else ""

    def recent_file_actions(self) -> list[str]:
        actions = [
            f"- {record.key} {record.name} {record.note or ' '.join(Tool.compact(arg, 80) for arg in record.args)}".strip()
            for record in self.session.tool_records[-20:]
            if record.name in {"Read", "Edit"}
        ]
        return actions[-10:]

    def recent_tool_errors(self) -> list[str]:
        return [
            f"- {' '.join(part for part in (record.key, record.name, ' '.join(Tool.compact(arg, 80) for arg in record.args)) if part)}: {Tool.compact(record.error, 160)}"
            for record in self.session.tool_errors[-5:]
        ]

    def recent_code_edits(self) -> list[str]:
        rows: dict[str, str] = {}
        for record in self.session.tool_records[-20:]:
            if record.name == "Edit":
                for match in re.finditer(r'<Edit\s+path=(".*?")', record.output):
                    try:
                        path = str(json.loads(match.group(1)))
                    except json.JSONDecodeError:
                        continue
                    if self.code_like_path(path):
                        rows[path] = f"- {record.key} Edit {path}"
        return list(rows.values())[-8:]

    def check_status(self, code_edits: list[str]) -> list[str]:
        if not code_edits:
            return []
        check = self.session.state.check.strip()
        return ["- " + check] if check else ["- Code changed recently. Use Note(check=...) after checks, or final must say checks not run."]

    @classmethod
    def code_like_path(cls, path: str) -> bool:
        name = os.path.basename(path)
        return name in cls.CODE_FILENAMES or os.path.splitext(name)[1].lower() in cls.CODE_EXTENSIONS

    def coverage(self, numbered: dict[int, tuple[str, str, str]]) -> list[tuple[int, int]]:
        numbers = sorted(numbered)
        if not numbers:
            return []
        ranges = []
        start = previous = numbers[0]
        for number in numbers[1:]:
            if number == previous + 1:
                previous = number
                continue
            ranges.append((start, previous + 1))
            start = previous = number
        ranges.append((start, previous + 1))
        return ranges

    def segments(self, numbered: dict[int, tuple[str, str, str]]) -> list[tuple[int, int, str, str, list[str]]]:
        items = sorted(numbered.items())
        if not items:
            return []
        segments = []
        start = previous = items[0][0]
        source, tool, first = items[0][1]
        lines = [first]
        for number, (line_source, line_tool, line) in items[1:]:
            if number == previous + 1 and line_source == source and line_tool == tool:
                previous = number
                lines.append(line)
                continue
            segments.append((start, previous + 1, source, tool, lines))
            start = previous = number
            source, tool = line_source, line_tool
            lines = [line]
        segments.append((start, previous + 1, source, tool, lines))
        return segments

    def compaction_input(self, messages: list[Json]) -> str:
        older, recent = self.compaction_parts_for(messages)
        return "\n\n".join(
            [
                "State:\n" + self.session.state.format(),
                "Previous Summary:\n" + (self.session.state.summary or "(empty)"),
                "Older Messages:\n" + self.messages_text(older),
                "Recent Messages (rewrite briefly inside summary):\n" + self.messages_text(recent),
            ]
        )

    def compaction_parts(self) -> tuple[list[Json], list[Json]]:
        index = self.latest_user_index(self.session.messages)
        return (self.session.messages, []) if index is None else (self.session.messages[:index], self.session.messages[index:])

    def turn_compaction_parts(self, messages: list[Json]) -> tuple[list[Json], list[Json]]:
        index = self.latest_user_index(messages)
        if index is None:
            return self.compaction_parts_for(messages)
        compacted, keep = self.compaction_parts_for(messages[index + 1 :])
        return compacted, messages[: index + 1] + keep

    def compaction_parts_for(self, messages: list[Json]) -> tuple[list[Json], list[Json]]:
        cut = max(0, len(messages) - self.COMPACT_RECENT_MESSAGES)
        return messages[:cut], messages[cut:]

    def messages_text(self, messages: list[Json]) -> str:
        return "\n\n".join(f"{message.get('role', 'message')}:\n{message.get('content') or ''}" for message in messages) or "(empty)"

    def apply_compaction(self, data: Json, keep: list[Json], tool_messages: list[Json] | None = None) -> None:
        self.session.state.apply(data)
        summary = self.session.state.summary
        self.session.messages = ([{"role": "user", "content": self.COMPACT_TITLE + "\n" + summary}] if summary else []) + keep
        self.prune_tool_records([*self.session.messages, *(tool_messages or [])])

    def apply_compaction_fallback(self, keep: list[Json], tool_messages: list[Json] | None = None) -> None:
        self.session.state.summary = (self.session.state.summary + "\nPrevious context was deterministically trimmed.").strip()
        summary = self.session.state.summary
        self.session.messages = ([{"role": "user", "content": self.COMPACT_TITLE + "\n" + summary}] if summary else []) + keep
        self.prune_tool_records([*keep, *(tool_messages or [])])

    def apply_turn_compaction(self, data: Json, keep: list[Json], turn_messages: list[Json]) -> None:
        self.session.state.apply(data)
        summary = self.session.state.summary
        index = self.latest_user_index(keep)
        insert = len(keep) if index is None else index + 1
        turn_messages[:] = keep[:insert] + ([{"role": "user", "content": self.COMPACT_TITLE + "\n" + summary}] if summary else []) + keep[insert:]
        self.prune_tool_records([*self.session.messages, *turn_messages])

    def apply_turn_compaction_fallback(self, keep: list[Json], turn_messages: list[Json]) -> None:
        self.session.state.summary = (self.session.state.summary + "\nCurrent turn context was deterministically trimmed.").strip()
        self.apply_turn_compaction({"summary": self.session.state.summary}, keep, turn_messages)

    def prune_tool_records(self, keep_messages: list[Json]) -> None:
        records = self.session.tool_records
        keep = set(re.findall(r"\btr\.\d+\b", self.messages_text(keep_messages)))
        index = {record.key: offset for offset, record in enumerate(records)}
        paths: dict[str, list[str]] = {}
        for record in records:
            paths[record.key] = []
            for match in re.findall(r"<(?:Read|Edit)\s+path=(\".*?\")", record.output):
                try:
                    paths[record.key].append(str(json.loads(match)))
                except json.JSONDecodeError:
                    pass
        mins: dict[str, int] = {}
        for path, lines in self.active_file_lines()[0].items():
            for source, _tool, _line in lines.values():
                if source in index:
                    mins[path] = min(mins.get(path, index[source]), index[source])
                    keep.add(source)
        for offset, record in enumerate(records):
            if record.name == "Edit" and any(path in mins and offset >= mins[path] for path in paths.get(record.key, [])):
                keep.add(record.key)
        for _order, _server, _tool, key, _body in self.active_mcp_tool_details():
            keep.add(key)

        self.session.tool_records = [record for record in records if record.key in keep][-400:]
        self.session.tool_results = {record.key: record.output for record in self.session.tool_records}

    def latest_user_index(self, messages: list[Json]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user" and not str(messages[index].get("content") or "").startswith(self.COMPACT_TITLE):
                return index
        return None

    def bound_output(self, text: str, key: str = "") -> str:
        estimated = self.estimated_text_tokens(text)
        if estimated <= MAX_TOOL_OUTPUT_TOKENS:
            return text
        limit = MAX_TOOL_OUTPUT_TOKENS * 4
        head_limit = max(1, limit * 2 // 5)
        tail_limit = max(1, limit - head_limit)
        head = self.head_excerpt(text, head_limit)
        tail = self.tail_excerpt(text, tail_limit)
        omitted_tokens = max(0, estimated - self.estimated_text_tokens(head) - self.estimated_text_tokens(tail))
        note = f'<bounded_output omitted="middle" max_tokens="{MAX_TOOL_OUTPUT_TOKENS}" estimated_tokens="{estimated}" omitted_tokens="{omitted_tokens}"'
        note += f' recall="{key}"' if key else ""
        note += "/>"
        return "\n".join(part for part in (head.rstrip(), note, tail.lstrip()) if part)

    @staticmethod
    def head_excerpt(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit].rsplit("\n", 1)[0] or text[:limit]

    @staticmethod
    def tail_excerpt(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[-limit:].split("\n", 1)[-1] or text[-limit:]

    def output_stat(self, output: str) -> tuple[int, int]:
        match = re.search(r'<file_stat mtime_ns="(\d+)" size="(\d+)"\s*/>', output)
        return (int(match.group(1)), int(match.group(2))) if match else (0, -1)

    def current_stat(self, path: str) -> tuple[int, int] | None:
        try:
            stat = os.stat(self.session.resolve_path(path))
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def read_lines(self, path: str, numbers: set[int]) -> dict[int, str] | None:
        if not numbers:
            return {}
        found = {}
        try:
            with open(self.session.resolve_path(path), encoding="utf-8") as file:
                for index, line in enumerate(file):
                    if index in numbers:
                        found[index] = line
                    if index >= max(numbers):
                        break
        except OSError:
            return None
        return found

    @staticmethod
    def estimated_tokens(messages: list[Json]) -> int:
        chars = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        return (chars + 3) // 4

    @staticmethod
    def estimated_text_tokens(text: str) -> int:
        return (len(text) + 3) // 4


class EditBatchPlan:
    @dataclass
    class Line:
        text: str
        origin: int | None

    @dataclass
    class FileState:
        path: str
        lines: list["EditBatchPlan.Line"]
        original: list[str]
        exists: bool

        def text(self) -> str:
            return "".join(line.text for line in self.lines)

        def current_origin(self, origin: int) -> int | None:
            for index, line in enumerate(self.lines):
                if line.origin == origin:
                    return index
            return None

    @dataclass
    class PlannedEdit:
        path: str
        before: str
        after: str
        created: bool
        changes: list[tuple[int, int, int, int]]

        def preview(self, tool: EditTool) -> str:
            return tool.diff(self.path, self.before, self.after) or f"Edit({self.path})"

        def call(self, tool: EditTool) -> str:
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as file:
                    current = file.read()
            elif self.created and not self.before:
                current = ""
            else:
                raise ToolError("planned edit is stale; file state changed")
            if current != self.before:
                raise ToolError("planned edit is stale; file state changed")
            if self.created:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(self.after)
            return "\n".join(
                [
                    f"<Edit path={json.dumps(tool.session.relpath(self.path))}>",
                    tool.file_stat(self.path),
                    tool.diff(self.path, self.before, self.after).rstrip(),
                    tool.edit_context(self.after, self.changes),
                    "</Edit>",
                ]
            )

    def __init__(self, session: Session):
        self.session = session
        self.files: dict[str, EditBatchPlan.FileState] = {}
        self.planned: dict[str, EditBatchPlan.PlannedEdit] = {}
        self.errors: dict[str, str] = {}

    def build(self, calls: list[ToolCall]) -> "EditBatchPlan":
        for call in calls:
            if call.name != "Edit":
                continue
            try:
                self.plan_call(call, EditTool(self.session, call.args))
            except ToolError as error:
                self.errors[call.id] = str(error)
        return self

    def plan_call(self, call: ToolCall, tool: EditTool) -> None:
        path, edits = tool.parse()
        state = self.file_state(path, edits[0].op == "create")
        before, created = state.text(), not state.exists
        lines, changes = self.apply(tool, state, edits)
        after = "".join(line.text for line in lines)
        if after == before and not created:
            raise ToolError("edit produced no changes")
        self.planned[call.id] = self.PlannedEdit(path, before, after, created, changes)
        state.lines, state.exists = lines, True

    def file_state(self, path: str, creating: bool) -> FileState:
        if path in self.files:
            state = self.files[path]
            if not state.exists and not creating:
                raise ToolError("file does not exist; use op=create to create it")
            if state.exists and creating:
                raise ToolError("file already exists")
            return state
        if os.path.exists(path):
            if creating:
                raise ToolError("file already exists")
            with open(path, encoding="utf-8") as file:
                original = file.readlines()
            state = self.FileState(path, [self.Line(line, index) for index, line in enumerate(original)], original, True)
        elif creating:
            parent = os.path.dirname(path) or "."
            if not self.session.in_cwd(parent):
                raise ToolError("refusing to create parent directories outside workspace")
            state = self.FileState(path, [], [], False)
        else:
            raise ToolError("file does not exist; use op=create to create it")
        self.files[path] = state
        return state

    def apply(self, tool: EditTool, state: FileState, edits: list[Edit]) -> tuple[list[Line], list[tuple[int, int, int, int]]]:
        if edits[0].op == "create":
            lines = self.new_lines(tool.content_lines(edits[0].content, False))
            return lines, [(0, 0, 0, len(lines))]
        if any(edit.op == "replace_all" for edit in edits):
            if any(edit.op != "replace_all" for edit in edits):
                raise ToolError("replace_all cannot be mixed with anchored edits")
            content = state.text()
            for edit in edits:
                if not edit.old and content:
                    raise ToolError("replace_all requires old")
                if edit.old and edit.old not in content:
                    raise ToolError("replace_all old text not found")
                content = content.replace(edit.old, edit.new)
            lines = [self.Line(line, None) for line in content.splitlines(True)]
            return lines, [(0, 0, 0, len(lines))]

        replacements = []
        for edit in edits:
            start = self.resolve_anchor(state, edit.start)
            if edit.op in {"replace", "delete"}:
                end = self.resolve_anchor(state, edit.end)
                if end < start:
                    raise ToolError("end anchor is before start anchor")
                replacement = [] if edit.op == "delete" else self.new_lines(tool.content_lines(edit.content, end + 1 < len(state.lines)))
                replacements.append((start, end + 1, replacement))
            elif edit.op in {"insert_before", "insert_after"}:
                index = start if edit.op == "insert_before" else start + 1
                replacements.append((index, index, self.new_lines(tool.content_lines(edit.content, index < len(state.lines)))))
            else:
                raise ToolError("unknown edit op")

        previous = None
        for start, end, _replacement in sorted(replacements):
            if previous and (start < previous[1] or (start == previous[0] and end == previous[1])):
                raise ToolError(f"edits overlap or share an insertion point: {previous[0]}:{previous[1]} and {start}:{end}")
            previous = (start, end)

        lines = list(state.lines)
        for start, end, replacement in sorted(replacements, reverse=True):
            lines[start:end] = replacement
        return lines, self.changes(replacements)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]

    @staticmethod
    def changes(replacements: list[tuple[int, int, list[Line]]]) -> list[tuple[int, int, int, int]]:
        changes, delta = [], 0
        for start, end, replacement in sorted(replacements):
            new_start = start + delta
            new_end = new_start + len(replacement)
            clear_end = 0 if len(replacement) != end - start else new_start + (end - start)
            changes.append((new_start, clear_end, new_start, new_end))
            delta += len(replacement) - (end - start)
        return changes

    def resolve_anchor(self, state: FileState, anchor: str) -> int:
        match = re.fullmatch(r"(\d+):([0-9a-z]{5})", anchor.split("|", 1)[0].strip())
        if not match:
            raise ToolError('invalid anchor; use "line:hash" from Read or Search')
        index, expected = int(match.group(1)), match.group(2).lower()
        if index < len(state.lines) and ReadTool.line_hash(state.lines[index].text) == expected:
            return index
        if index < len(state.original) and ReadTool.line_hash(state.original[index]) == expected:
            current = state.current_origin(index)
            if current is not None:
                return current
            raise ToolError(f"stale anchor {anchor}; original line was changed in this batch")
        current = f"{index}:{ReadTool.line_hash(state.lines[index].text)}|{state.lines[index].text.rstrip()}" if index < len(state.lines) else "out of range"
        raise ToolError(f"stale anchor {anchor}; current is {current}")


class MCPManager:
    RAW_OUTPUT_LIMIT: ClassVar[int] = 200_000
    DESCRIBE_DESCRIPTION_LIMIT: ClassVar[int] = 1_000
    DESCRIBE_ARGUMENT_LIMIT: ClassVar[int] = 50
    DESCRIBE_ARGUMENT_DESCRIPTION_LIMIT: ClassVar[int] = 160

    def __init__(self, session: Session):
        self.session = session
        self.tools: dict[str, list[MCPToolInfo]] = {}
        self.server_errors: dict[str, str] = {}
        self._discovered = False
        self.discovery_status: str = "stale"  # stale | discovering | ready | error

    def parse_configs(self) -> list[MCPServerConfig]:
        configs: list[MCPServerConfig] = []
        mcp_config = self.session.config.mcp
        if isinstance(mcp_config, dict):
            for name, raw in mcp_config.items():
                if isinstance(raw, dict):
                    config = MCPServerConfig(
                        name=str(name),
                        url=Config.str(raw, "url"),
                        bearer_token_env_var=Config.str(raw, "bearer_token_env_var"),
                        enabled=Config.bool(raw, "enabled", True),
                    )
                    headers = raw.get("env_http_headers")
                    if headers is None:
                        pass
                    elif isinstance(headers, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
                        config.env_http_headers = dict(headers)
                    else:
                        config.error = "env_http_headers must be a string map"
                    configs.append(config)
        return configs

    def discover_enabled(self) -> None:
        self.discovery_status = "discovering"
        try:
            configs = self.parse_configs()
            configured = {config.name for config in configs}
            for name in list(self.tools):
                if name not in configured:
                    self.tools.pop(name, None)
                    self.server_errors.pop(name, None)
            for config in configs:
                if not config.enabled:
                    self.tools.pop(config.name, None)
                    self.server_errors.pop(config.name, None)
                    continue
                self._discover_one(config)
            self._discovered = True
            self.discovery_status = "ready"
        except Exception as error:
            self.server_errors["-"] = str(error)
            self.discovery_status = "error"

    def discover_server(self, name: str) -> None:
        configs = self.parse_configs()
        config = next((c for c in configs if c.name == name and c.enabled), None)
        if config is None:
            self.tools.pop(name, None)
            self.server_errors[name] = "server not found or disabled"
            return
        self._discover_one(config)

    def _discover_one(self, config: MCPServerConfig) -> None:
        if config.error:
            self.set_server_error(config.name, config.error)
            return
        headers = self._build_mcp_headers(config)
        if isinstance(headers, str):
            self.set_server_error(config.name, headers)
            return

        if not config.url:
            self.set_server_error(config.name, "url is required")
            return

        try:
            tools = asyncio.run(self._list_tools(config.url, headers))
            self.tools[config.name] = [
                MCPToolInfo(
                    server=config.name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema,
                    annotations=self.tool_annotations(t),
                )
                for t in tools
            ]
            self.server_errors.pop(config.name, None)
        except Exception as e:
            self.set_server_error(config.name, str(e))

    def set_server_error(self, name: str, error: str) -> None:
        self.tools.pop(name, None)
        self.server_errors[name] = error

    @staticmethod
    def tool_annotations(tool: Any) -> Json:
        annotations = getattr(tool, "annotations", None)
        if annotations is None:
            return {}
        if isinstance(annotations, dict):
            return annotations
        if hasattr(annotations, "model_dump"):
            data = annotations.model_dump(mode="json", exclude_none=True)
            return data if isinstance(data, dict) else {}
        return {}

    def tool_needs_confirmation(self, server: str, tool_name: str) -> bool:
        info = self.tool_info(server, tool_name)
        if info is None:
            return False
        annotations = info.annotations
        if annotations.get("readOnlyHint") is True:
            return False
        return annotations.get("destructiveHint") is True

    def tool_info(self, server: str, tool_name: str) -> MCPToolInfo | None:
        return next((tool for tool in self.tools.get(server, []) if tool.name == tool_name), None)

    async def _list_tools(self, url: str, headers: dict[str, str]) -> list[Any]:
        from fastmcp.client import Client
        from fastmcp.client.transports import StreamableHttpTransport

        async with Client(StreamableHttpTransport(url, headers=headers)) as client:
            return await client.list_tools()

    def _build_mcp_headers(self, config: MCPServerConfig) -> dict[str, str] | str:
        headers: dict[str, str] = {}
        if config.bearer_token_env_var:
            token = os.environ.get(config.bearer_token_env_var)
            if not token:
                return f"missing environment variable {config.bearer_token_env_var}"
            headers["Authorization"] = f"Bearer {token}"
        if config.env_http_headers:
            for header_name, env_var in config.env_http_headers.items():
                value = os.environ.get(env_var)
                if not value:
                    return f"missing environment variable {env_var}"
                if header_name.lower() == "authorization":
                    return "conflicting Authorization header; use bearer_token_env_var instead"
                headers[header_name] = value
        return headers

    def call_tool(self, server: str, tool_name: str, arguments: Json) -> str:
        configs = self.parse_configs()
        config = next((c for c in configs if c.name == server and c.enabled), None)
        if config is None:
            raise ToolError(f"MCP server '{server}' not found")
        if config.error:
            raise ToolError(config.error)
        if not config.url:
            raise ToolError("url is required")

        headers = self._build_mcp_headers(config)
        if isinstance(headers, str):
            raise ToolError(headers)

        if server not in self.tools:
            self.discover_server(server)
        if server in self.server_errors:
            raise ToolError(f"MCP server '{server}' error: {self.server_errors[server]}")

        try:
            result = asyncio.run(self._call_tool(config.url, headers, tool_name, arguments))
        except Exception as e:
            raise ToolError(f"MCP call failed: {e}")

        text = self.normalize_result(result)
        return f"<MCPCall server={json.dumps(server)} tool={json.dumps(tool_name)}>\n{text}\n</MCPCall>"

    def normalize_result(self, result: Any) -> str:
        parts: list[str] = []
        content = getattr(result, "content", result)
        items = content if isinstance(content, list) else [content]
        for item in items:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text") or ""))
                elif item_type == "resource":
                    parts.append(json.dumps(item.get("resource"), ensure_ascii=False, indent=2))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, indent=2))
                continue
            item_type = getattr(item, "type", "")
            if item_type == "text":
                parts.append(str(getattr(item, "text", "") or ""))
            elif item_type == "resource":
                parts.append(str(getattr(item, "resource", "") or ""))
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2))
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part).strip()
        if len(text) > self.RAW_OUTPUT_LIMIT:
            text = text[: self.RAW_OUTPUT_LIMIT] + f"\n<MCPOutputTruncated chars={json.dumps(len(text))}/>"
        return text

    async def _call_tool(self, url: str, headers: dict[str, str], name: str, arguments: Json) -> Any:
        from fastmcp.client import Client
        from fastmcp.client.transports import StreamableHttpTransport

        async with Client(StreamableHttpTransport(url, headers=headers)) as client:
            return await client.call_tool(name, arguments)

    def describe_tool(self, server: str, tool_name: str) -> str:
        tools = self.tools.get(server)
        if tools is None:
            self.discover_server(server)
            tools = self.tools.get(server)

        if tools is None:
            if server in self.server_errors:
                raise ToolError(f"MCP server '{server}' error: {self.server_errors[server]}")
            raise ToolError(f"MCP server '{server}' not found")

        info = self.tool_info(server, tool_name)
        if info is None:
            raise ToolError(f"MCP tool '{tool_name}' not found on server '{server}'")

        return self._render_describe(server, info)

    def _render_describe(self, server: str, info: MCPToolInfo) -> str:
        schema = info.input_schema or {}
        lines = [f"<MCPDescribe server={json.dumps(server)} tool={json.dumps(info.name)}>"]
        if info.description:
            lines.append("<description>")
            lines.append(self.compact_text(info.description, self.DESCRIBE_DESCRIPTION_LIMIT))
            lines.append("</description>")
        lines.append("<arguments>")
        props = schema.get("properties", {})
        props = props if isinstance(props, dict) else {}
        required = schema.get("required", [])
        required = required if isinstance(required, list) else []
        for index, (name, prop) in enumerate(props.items()):
            if index >= self.DESCRIBE_ARGUMENT_LIMIT:
                lines.append(f"... {len(props) - self.DESCRIBE_ARGUMENT_LIMIT} more arguments omitted")
                break
            req = "required" if name in required else "optional"
            prop = prop if isinstance(prop, dict) else {}
            typ = prop.get("type", "any")
            desc = self.compact_text(str(prop.get("description", "") or ""), self.DESCRIBE_ARGUMENT_DESCRIPTION_LIMIT)
            lines.append(f"- {name} {req} {typ}: {desc}")
        lines.append("</arguments>")
        lines.append("<schema_summary>")
        if props:
            summary = f"Top-level object with {', '.join(props.keys())}."
        else:
            summary = "No arguments."
        if len(summary) > 200:
            summary = summary[:197] + "..."
        lines.append(summary)
        lines.append("</schema_summary>")
        lines.append("</MCPDescribe>")
        return "\n".join(lines)

    @staticmethod
    def compact_text(text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."

    def render_tools_index(self) -> str:
        if not self.tools:
            return ""

        lines: list[str] = []
        lines.append("--- MCP TOOLS ---")
        lines.append('Use MCP(action="call", server, tool, arguments) for external MCP server tools.')
        lines.append('Use MCP(action="describe", server, tool) for full details when needed.')
        lines.append("Format: server.tool(req: type; opt: type) - description")
        lines.append("")

        configs = self.parse_configs()
        for config in configs:
            if not config.enabled or config.name not in self.tools:
                continue
            tools = self.tools.get(config.name, [])
            if not tools:
                continue
            name_display = config.name.capitalize()
            lines.append(f"[{config.name}] {name_display}")
            for tool in tools:
                line = self._format_tool_line(config.name, tool)
                if line:
                    lines.append(line)
            lines.append("")

        text = "\n".join(lines)
        if len(text) > 4000:
            text = text[:3990] + "\n... MCP tools truncated; use /mcp tools for full list."
        return text

    def _format_tool_line(self, server: str, info: MCPToolInfo) -> str:
        schema = info.input_schema or {}
        props = schema.get("properties", {})
        props = props if isinstance(props, dict) else {}
        required = schema.get("required", [])
        required = required if isinstance(required, list) else []

        def _fmt(name: str) -> str:
            t = props.get(name, {}).get("type", "")
            return f"{name}: {t}" if t else name

        req_args = [_fmt(k) for k in required if k in props]
        opt_args = [_fmt(k) for k in props if k not in required]

        if len(req_args) > 8:
            req_args = req_args[:8] + ["..."]
        if len(opt_args) > 8:
            opt_args = opt_args[:8] + ["..."]

        parts = []
        if req_args:
            parts.append("(" + ", ".join(req_args))
        else:
            parts.append("(")
        if opt_args:
            parts.append("; " + ", ".join(opt_args))
        parts.append(")")
        args_str = "".join(parts)

        desc = (info.description or "").split("\n")[0].strip()
        desc = " ".join(desc.split())
        if len(desc) > 80:
            desc = desc[:77] + "..."

        line = f"{server}.{info.name}{args_str} - {desc}"
        if len(line) > 200:
            line = line[:197] + "..."
        return line

    def render_tool_listing(self, server: str | None = None) -> str:
        lines: list[str] = []
        configs = self.parse_configs()
        for config in configs:
            if not config.enabled:
                continue
            if server and config.name != server:
                continue
            name_display = config.name.capitalize()
            lines.append(f"[{config.name}] {name_display}")
            if config.name in self.server_errors:
                lines.append(f"  error: {self.server_errors[config.name]}")
                continue
            tools = self.tools.get(config.name, [])
            if not tools:
                lines.append("  (no tools discovered)")
                continue
            for tool in tools:
                line = self._format_tool_line(config.name, tool)
                if line:
                    lines.append(f"  {line}")
            lines.append("")
        return "\n".join(lines) if lines else "(no MCP servers configured)"

    def render_server_status(self) -> str:
        lines: list[str] = []
        configs = self.parse_configs()
        for config in configs:
            parts = [config.name]
            if not config.enabled:
                parts.append("disabled")
            elif config.name in self.server_errors:
                parts.append(f"error: {self.server_errors[config.name]}")
            else:
                parts.append("enabled")
                if config.name in self.tools:
                    parts.append("connected")
                    parts.append(f"tools={len(self.tools[config.name])}")
                else:
                    parts.append("not connected")
            if config.bearer_token_env_var:
                parts.append(f"auth=bearer_token_env_var({config.bearer_token_env_var})")
            if config.env_http_headers:
                for header_name in config.env_http_headers:
                    parts.append(f"auth=env_header({header_name})")
            lines.append("  ".join(parts))
        return "\n".join(lines) if lines else "(no MCP servers configured)"

class ToolRunner:
    def __init__(self, session: Session, context: ContextManager, input_fn=input, output_fn=print):
        self.session = session
        self.context = context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.preview_fn: Callable[[str], bool] | None = None
        self.preview_full_fn: Callable[[str], None] | None = None
        self.live_output: Callable[[str, str], None] | None = None
        self.live_start: Callable[[], None] | None = None

    def run(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        messages = []
        index = 0
        first, refused = True, False
        while index < len(calls):
            end = index + 1 if self.edit_barrier(calls[index]) else self.edit_segment_end(calls, index)
            segment = calls[index:end]
            plan = EditBatchPlan(self.session).build(segment) if not refused and any(call.name == "Edit" for call in segment) else EditBatchPlan(self.session)
            for call in segment:
                self.session.state.turn_tool_calls += 1
                suffix = batch_suffix if first else ""
                first = False
                status, content = (
                    ("skipped", self.tool_message(call, "", "Skipped: previous tool call was refused", failed=True))
                    if refused
                    else self.run_one(call, batch_suffix=suffix, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, ""))
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                if status == "refused":
                    refused = True
            index = end
        return messages

    def edit_segment_end(self, calls: list[ToolCall], start: int) -> int:
        end = start
        while end < len(calls) and not self.edit_barrier(calls[end]):
            end += 1
        return end

    def edit_barrier(self, call: ToolCall) -> bool:
        if call.name == "Edit":
            return False
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return True
        if tool_class.MUTATES:
            return True
        if call.name == "Git":
            try:
                return tool_class(self.session, call.args).needs_confirmation()
            except Exception:
                return True
        return False

    def run_one(
        self,
        call: ToolCall,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
        plan_error: str = "",
    ) -> tuple[str, str]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "failed", self.reject(call, f"ToolError: unknown tool {call.name}", batch_suffix=batch_suffix)
        tool = tool_class(self.session, call.args)
        if isinstance(tool, BashTool):
            tool.live_output = self.live_output
        started, approved, display = time.monotonic(), False, None
        try:
            display = self.short_call(call, tool.short_args())
            if plan_error:
                raise ToolError(plan_error)
            needs_confirmation = tool.needs_confirmation()
            if needs_confirmation and self.session.settings.yolo:
                self.output_fn(self.approval_display(call, tool, "auto", batch_suffix=batch_suffix, planned_edit=planned_edit))
            elif needs_confirmation:
                confirmed, reason = self.confirm(call, tool, batch_suffix=batch_suffix, planned_edit=planned_edit)
                if not confirmed:
                    output = "Cancelled: user refused tool call" + ((": " + reason) if reason else "")
                    return "refused", self.finish(call, output, failed=True, elapsed=time.monotonic() - started, display=display, batch_suffix=batch_suffix)
                approved = True
            if isinstance(tool, BashTool) and self.live_start is not None:
                self.live_start()
            output = planned_edit.call(tool) if planned_edit and isinstance(tool, EditTool) else tool.call()
        except ToolError as error:
            return "failed", self.reject(call, f"ToolError: {error}", elapsed=time.monotonic() - started, display=display, batch_suffix=batch_suffix)
        except Exception as error:
            output = f"ToolError: {error}"
            return "failed", self.finish(call, output, failed=True, elapsed=time.monotonic() - started, display=display, batch_suffix=batch_suffix)
        return "ok", self.finish(call, output, elapsed=time.monotonic() - started, approved=approved, display=display, batch_suffix=batch_suffix)

    def reject(self, call: ToolCall, output: str, *, elapsed: float | None = None, display: str | None = None, batch_suffix: str = "") -> str:
        if self.session.settings.debug:
            return self.finish(call, output, failed=True, elapsed=elapsed, display=display, batch_suffix=batch_suffix)
        self.session.record_tool_error("-", call.name, call.args, output)
        self.output_fn(self.finish_display(call, "", output, failed=True, display=display, batch_suffix=batch_suffix))
        return self.tool_message(call, "", output, failed=True, display=display)

    def finish(
        self,
        call: ToolCall,
        output: str,
        *,
        failed: bool = False,
        elapsed: float | None = None,
        approved: bool = False,
        display: str | None = None,
        store: bool = True,
        batch_suffix: str = "",
    ) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        key = (
            self.session.store_tool_result(call.name, call.args, output, self.tool_note(call, output))
            if not failed and store and (tool_class is None or tool_class.STORES_RESULT)
            else ""
        )
        if failed:
            self.session.record_tool_error(key or "-", call.name, call.args, output)
        elif key:
            self.update_code_index(call, output)
        self.output_fn(self.finish_display(call, key, output, failed=failed, approved=approved, display=display, batch_suffix=batch_suffix))
        return self.tool_message(call, key, output, failed=failed, display=display)

    def tool_message(self, call: ToolCall, key: str, output: str, *, failed: bool = False, display: str | None = None) -> str:
        head = "tool " + ((key + " ") if key else ("- " if failed else "")) + (display or self.short_call(call))
        if not failed and call.name in {"Read", "Edit"}:
            return head + " -> FILE STATE"
        if not failed and call.name == "MCP" and "<MCPDescribe " in output:
            return head + " -> MCP TOOL DETAILS"
        rows = [head]
        if failed:
            rows.append("status: failed")
        rows.extend(["output:", self.context.bound_output(output, key).rstrip()])
        return "\n".join(rows).strip()

    def tool_note(self, call: ToolCall, output: str) -> str:
        if call.name != "Read":
            return ""
        notes = []
        for block in re.finditer(r"(?s)<Read\s+path=(\".*?\").*?<total_lines>(\d+)</total_lines>(.*?)</Read>", output):
            try:
                path = str(json.loads(block.group(1)))
            except json.JSONDecodeError:
                continue
            total = int(block.group(2))
            for start, end in re.findall(r"<range>(\d+):(\d+)</range>", block.group(3)):
                start_i, end_i = int(start), int(end)
                suffix = " FULL FILE" if start_i == 0 and end_i == total else ""
                notes.append(f"{path} {start_i}:{end_i}{suffix}")
        return "; ".join(notes)

    def update_code_index(self, call: ToolCall, output: str) -> None:
        if call.name != "Edit":
            return
        paths = [str(call.args[0])] if call.args and isinstance(call.args[0], str) else []
        for match in re.finditer(r'<Edit\s+path=(".*?")', output):
            try:
                paths.append(str(json.loads(match.group(1))))
            except json.JSONDecodeError:
                pass
        CodeIndex(self.session).update(list(dict.fromkeys(paths)))

    def confirm(self, call: ToolCall, tool: Tool, batch_suffix: str = "", planned_edit: EditBatchPlan.PlannedEdit | None = None) -> tuple[bool, str]:
        display = self.approval_display(call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit, preview_lines=40)
        if self.preview_full_fn and tool.NAME == "Edit":
            self.preview_full_fn(self.approval_display(call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit, preview_lines=None))
        if not (self.preview_fn and self.preview_fn(display)):
            self.output_fn(display)
        answer = self.input_fn("[Y/n or reason] ").strip()
        lower = answer.lower()
        if lower in {"", "y", "yes"}:
            return True, ""
        return False, "" if lower in {"n", "no"} else answer

    def approval_display(
        self,
        call: ToolCall,
        tool: Tool,
        status: str,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
        preview_lines: int | None = 40,
    ) -> str:
        header = self.with_batch_suffix(("approve " if status == "confirm" else "auto ") + self.short_call(call), batch_suffix)
        if tool.NAME != "Edit":
            return header
        preview = planned_edit.preview(tool) if planned_edit and isinstance(tool, EditTool) else tool.preview()
        return header + (("\n" + block) if (block := self.preview_block(preview, max_lines=preview_lines)) else "")

    def preview_block(self, preview: str, *, max_lines: int | None = 40) -> str:
        lines = preview.rstrip().splitlines()
        if not lines:
            return ""
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... preview truncated: {len(lines) - max_lines} more lines (Ctrl-A: full preview) ..."]
        return "\n".join(["  preview", *("  " + line for line in lines)])

    def finish_display(
        self, call: ToolCall, key: str, output: str, *, failed: bool, approved: bool = False, display: str | None = None, batch_suffix: str = ""
    ) -> str:
        if call.name == "Note" and not failed and display:
            return self.with_batch_suffix(display.removeprefix("Note ").strip(), batch_suffix)
        tag = " [refused]" if failed and "user refused" in output else " [failed]" if failed else " [approved]" if approved else ""
        line = self.with_batch_suffix("tool " + (display or self.short_call(call)) + ((" -> " + key) if key else "") + tag, batch_suffix)
        lines = [line]
        if failed:
            lines.append("  error " + self.oneline(output, 220))
        return "\n".join(lines)

    @staticmethod
    def with_batch_suffix(text: str, suffix: str) -> str:
        return text + (("  " + suffix) if suffix else "")

    def short_call(self, call: ToolCall, args: list[str] | None = None) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        if args is None:
            try:
                args = tool_class(self.session, call.args).short_args() if tool_class is not None else [Tool.compact(arg) for arg in call.args]
            except Exception:
                args = [Tool.compact(arg) for arg in call.args]
        text = " ".join([call.name, *args]).strip()
        return text if "\n" in text else self.oneline(text, 200)

    @staticmethod
    def oneline(text: str, limit: int) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class DebugTrace:
    STRING_LIMIT: ClassVar[int] = 20_000

    @classmethod
    def write(cls, session: Session, *, activity: str, label: str, payload: Any) -> str:
        if not session.settings.debug:
            return ""
        directory = session.debug_dir()
        os.makedirs(directory, exist_ok=True)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label or "event")
        path = os.path.join(directory, f"last-{safe_label}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"activity": Text.clean(activity or "debug"), "label": safe_label, "payload": cls.value(payload)}, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return path

    @classmethod
    def value(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(mode="json")
            except TypeError:
                value = value.model_dump()
        if isinstance(value, dict):
            return {str(key): cls.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.value(item) for item in value]
        if isinstance(value, str):
            value = Text.clean(value)
            return value if len(value) <= cls.STRING_LIMIT else value[: cls.STRING_LIMIT] + "...<truncated>"
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return Text.clean(str(value))

    @classmethod
    def prompt(cls, session: Session, *, activity: str, messages: list[Json]) -> None:
        cls.write(session, activity=activity, label="prompt", payload={"messages": messages})

    @classmethod
    def model_request(cls, session: Session, *, activity: str, api: str, model: str, params: Json, tools: list[Json] | None) -> None:
        payload = {"api": api, "model": model, "tool_names": cls.tool_names(tools), "param_keys": sorted(params), "params": cls.filtered_params(params)}
        cls.write(session, activity=activity, label="model-request", payload=payload)

    @classmethod
    def model_response(cls, session: Session, *, activity: str, api: str, model: str, raw: Any, text: str, tool_names: list[str]) -> None:
        payload = {"api": api, "model": model, "assistant_text_len": len(text), "tool_names": tool_names, "raw": raw}
        cls.write(session, activity=activity, label="model-response", payload=payload)

    @classmethod
    def model_error(cls, session: Session, *, activity: str, api: str, model: str, params: Json, error: Exception | str) -> None:
        payload = {"api": api, "model": model, "error": str(error), "param_keys": sorted(params), "params": cls.filtered_params(params)}
        cls.write(session, activity=activity, label="model-error", payload=payload)

    @staticmethod
    def filtered_params(params: Json) -> Json:
        return {key: value for key, value in params.items() if key not in {"messages", "tools"}}

    @staticmethod
    def tool_names(tools: list[Json] | None) -> list[str]:
        return [
            str(((schema.get("function") if isinstance(schema.get("function"), dict) else {}).get("name") or schema.get("name") or "(unknown)"))
            for schema in tools or []
        ]


class ModelClient:
    def __init__(self, session: Session):
        self.session = session

    def request(self, messages: list[Json]) -> tuple[Json, list[ToolCall], str]:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        tools = [tool.schema() for tool in TOOL_REGISTRY.values()]
        for attempt in range(MODEL_REQUEST_RETRIES + 1):
            self.session.state.current_model_call_started_at = time.monotonic()
            try:
                return self.anthropic_request(messages, tools) if provider.resolved_api() == "anthropic" else self.chat_request(messages, tools)
            except KeyboardInterrupt:
                if self.session.state.manual_model_retry_requested:
                    self.session.state.manual_model_retry_requested = False
                    raise ModelRequestRetry() from None
                raise
            except ModelError as error:
                retryable = self.retryable_error(error)
                if attempt >= MODEL_REQUEST_RETRIES or not retryable:
                    if attempt:
                        raise ModelError(f"{error} (after {attempt + 1} attempts)") from error
                    raise
                self.session.state.model_retry_count += 1
                time.sleep(0.5 * (attempt + 1))
            finally:
                self.session.state.current_model_call_started_at = 0.0
        raise ModelError("model request retry exhausted")

    @staticmethod
    def retryable_error(error: Exception) -> bool:
        status = getattr(error.__cause__, "status_code", None) or getattr(error.__cause__, "code", None)
        text = str(error).lower()
        try:
            if int(status) in {408, 409, 425, 429, 500, 502, 503, 504}:
                return True
        except Exception:
            pass
        if re.search(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(408|409|425|429|5\d\d)\b", text):
            return True
        return any(
            part in text for part in ("internal server error", "timeout", "timed out", "connection reset", "connection aborted", "temporarily unavailable")
        )

    def chat_request(self, messages: list[Json], tools: list[Json] | None = None, *, activity: str = "agent") -> tuple[Json, list[ToolCall], str]:
        messages = Text.value(messages)
        provider = self.session.config.provider
        params: Json = {"model": provider.model, "messages": messages, "stream": False}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = True
        prompt_cache_key = self.prompt_cache_key(provider, tools)
        if prompt_cache_key:
            params["prompt_cache_key"] = prompt_cache_key
        self.apply_provider_params(params, provider)
        DebugTrace.prompt(self.session, activity=activity, messages=messages)
        DebugTrace.model_request(self.session, activity=activity, api="chat", model=provider.model, params=params, tools=tools)
        try:
            response = self.client().chat.completions.create(**params)
        except Exception as error:
            DebugTrace.model_error(self.session, activity=activity, api="chat", model=provider.model, params=params, error=error)
            raise ModelError(str(error)) from error
        self.session.usage.add(getattr(response, "usage", None))
        message = response.choices[0].message
        assistant = self.assistant_message(message)
        calls = self.tool_calls(message)
        content = str(getattr(message, "content", None) or "")
        DebugTrace.model_response(
            self.session, activity=activity, api="chat", model=provider.model, raw=response, text=content, tool_names=[call.name for call in calls]
        )
        return assistant, calls, content

    def compact(self, context: str) -> Json:
        prompt = """
Compact the nanocode working context.
Return one JSON object only. No markdown, prose, code fences, or comments.
Use keys: summary, goal, plan, known.
Rewrite recent conversation briefly inside summary.
Keep only durable facts needed to continue; preserve file paths, symbols, constraints, and tr.N keys.
""".strip()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": Text.clean(context)}]
        _, _, content = (
            self.anthropic_request(messages, None, activity="compact")
            if self.session.config.provider.resolved_api() == "anthropic"
            else self.chat_request(messages, None, activity="compact")
        )
        data = self.parse_json_object(content)
        if not isinstance(data, dict):
            raise ModelError("compactor returned non-object JSON")
        return data

    @classmethod
    def parse_json_object(cls, text: str) -> Json:
        text = cls.strip_json_fence(Text.clean(text).strip())
        if not text:
            raise ModelError("compactor returned empty output")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = repair_json(text, return_objects=True)
        if isinstance(data, dict):
            return data
        raise ModelError("compactor returned invalid JSON: " + Tool.compact(text, 200))

    @staticmethod
    def strip_json_fence(text: str) -> str:
        match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
        return (match.group(1) if match else text).strip()

    def client(self) -> OpenAI:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        return OpenAI(
            api_key=provider.key, base_url=provider.base_url(), timeout=provider.timeout, max_retries=0, default_headers={"User-Agent": HTTP_USER_AGENT}
        )

    def anthropic_client(self) -> Anthropic:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        url = provider.base_url().rstrip("/")
        return Anthropic(
            api_key=provider.key,
            base_url=url[: -len("/v1")] if url.endswith("/v1") else url,
            timeout=provider.timeout,
            max_retries=0,
            default_headers={"User-Agent": HTTP_USER_AGENT},
        )

    def prompt_cache_key(self, provider: ProviderConfig, tools: list[Json] | None) -> str:
        configured = provider.prompt_cache_key
        if configured == "off":
            return ""
        if configured != "auto":
            return configured
        payload = {
            "api": provider.resolved_api(),
            "cwd": self.session.cwd,
            "host": provider.host(),
            "model": provider.model,
            "tools": ",".join(sorted(DebugTrace.tool_names(tools))) or "(none)",
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return "nanocode-" + digest[:24]

    def anthropic_request(self, messages: list[Json], tools: list[Json] | None, *, activity: str = "agent") -> tuple[Json, list[ToolCall], str]:
        messages = Text.value(messages)
        provider = self.session.config.provider
        params = self.anthropic_params(messages, tools)
        DebugTrace.prompt(self.session, activity=activity, messages=messages)
        DebugTrace.model_request(self.session, activity=activity, api="anthropic", model=provider.model, params=params, tools=tools)
        try:
            result = self.anthropic_client().messages.create(**params)
        except Exception as error:
            DebugTrace.model_error(self.session, activity=activity, api="anthropic", model=provider.model, params=params, error=error)
            raise ModelError(str(error)) from error
        self.session.usage.add(self.message_field(result, "usage"))
        assistant, calls, content = self.anthropic_result(result)
        DebugTrace.model_response(
            self.session, activity=activity, api="anthropic", model=provider.model, raw=result, text=content, tool_names=[call.name for call in calls]
        )
        return assistant, calls, content

    def anthropic_params(self, messages: list[Json], tools: list[Json] | None) -> Json:
        provider = self.session.config.provider
        params: Json = {
            "model": provider.model,
            "system": "\n\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system").strip(),
            "messages": self.anthropic_messages(messages),
            "max_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if provider.temperature is not None:
            params["temperature"] = provider.temperature
        if tools:
            params["tools"] = self.anthropic_tool_schemas(tools)
            params["tool_choice"] = {"type": "auto"}
        if provider.reasoning != "off":
            budget = CHAT_REASONING_EFFORT_VALUES["enable_thinking"].get(provider.reasoning_effort(), 4096)
            params["thinking"] = {"type": "enabled", "budget_tokens": min(ANTHROPIC_DEFAULT_MAX_TOKENS - 1024, int(budget))}
        return params

    def anthropic_messages(self, messages: list[Json]) -> list[Json]:
        converted: list[Json] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                self.append_anthropic_message(converted, "user", str(message.get("content") or ""))
            elif role == "assistant":
                blocks = self.anthropic_assistant_blocks(message)
                if blocks:
                    self.append_anthropic_message(converted, "assistant", blocks)
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": str(message.get("tool_call_id") or ""), "content": str(message.get("content") or "")}
                self.append_anthropic_message(converted, "user", [block])
        return converted or [{"role": "user", "content": ""}]

    @staticmethod
    def append_anthropic_message(messages: list[Json], role: str, content: str | list[Json]) -> None:
        if messages and messages[-1].get("role") == role:
            previous = messages[-1].get("content")
            if isinstance(previous, list) and isinstance(content, list):
                previous.extend(content)
                return
            if isinstance(previous, str) and isinstance(content, str):
                messages[-1]["content"] = (previous + "\n\n" + content).strip()
                return
        messages.append({"role": role, "content": content})

    def anthropic_assistant_blocks(self, message: Json) -> list[Json]:
        blocks: list[Json] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
            try:
                payload = json.loads(str(function.get("arguments") or "{}"))
            except json.JSONDecodeError:
                payload = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(raw.get("id") or uuid.uuid4().hex),
                    "name": str(function.get("name") or ""),
                    "input": payload if isinstance(payload, dict) else {"args": [payload]},
                }
            )
        return blocks

    @staticmethod
    def anthropic_tool_schemas(tools: list[Json]) -> list[Json]:
        def convert(schema: Json) -> Json:
            function = schema.get("function") if isinstance(schema.get("function"), dict) else {}
            return {
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "input_schema": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
            }

        return [convert(schema) for schema in tools]

    def anthropic_result(self, result: Any) -> tuple[Json, list[ToolCall], str]:
        text_parts: list[str] = []
        tool_calls: list[Json] = []
        calls: list[ToolCall] = []
        for block in self.message_field(result, "content") or []:
            block_type = self.message_field(block, "type")
            if block_type == "text":
                text_parts.append(str(self.message_field(block, "text") or ""))
            elif block_type == "tool_use":
                raw_input = self.message_field(block, "input")
                payload = raw_input if isinstance(raw_input, dict) else {}
                name = str(self.message_field(block, "name") or "")
                call_id = str(self.message_field(block, "id") or uuid.uuid4().hex)
                arguments = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
                calls.append(ToolCall(id=call_id, name=name, args=self.tool_payload(name, payload)))
        text = "".join(text_parts)
        assistant: Json = {"role": "assistant", "content": text or None}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, calls, text

    def apply_provider_params(self, params: Json, provider: ProviderConfig) -> None:
        if provider.temperature is not None:
            params["temperature"] = provider.temperature
        chat_reasoning = provider.resolved_chat_reasoning()
        reasoning_enabled = provider.reasoning != "off"
        effort = provider.reasoning_effort()
        extra: Json = {}
        if reasoning_enabled and chat_reasoning == "reasoning":
            extra["reasoning"] = {"effort": effort}
        elif reasoning_enabled and chat_reasoning == "reasoning_effort":
            params["reasoning_effort"] = effort
        elif chat_reasoning == "thinking":
            extra["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
            if reasoning_enabled:
                params["reasoning_effort"] = CHAT_REASONING_EFFORT_VALUES["thinking"].get(effort, "high")
        elif chat_reasoning == "enable_thinking":
            extra["enable_thinking"] = reasoning_enabled
            if reasoning_enabled:
                values = CHAT_REASONING_EFFORT_VALUES["enable_thinking"]
                extra["thinking_budget"] = values.get(effort, values["medium"])
        if extra:
            params["extra_body"] = extra

    def assistant_message(self, message: Any) -> Json:
        data: Json = {"role": "assistant", "content": self.message_field(message, "content")}
        reasoning_content = self.message_field(message, "reasoning_content")
        if reasoning_content:
            data["reasoning_content"] = reasoning_content
        tool_calls = [
            {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"}}
            for call in getattr(message, "tool_calls", None) or []
        ]
        if tool_calls:
            data["tool_calls"] = tool_calls
        return data

    @staticmethod
    def message_field(message: Any, key: str) -> Any:
        if isinstance(message, dict):
            return message.get(key)
        value = getattr(message, key, None)
        if value is not None:
            return value
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict) and key in extra:
            return extra[key]
        if hasattr(message, "model_dump"):
            dumped = message.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped.get(key)
        return None

    def tool_calls(self, message: Any) -> list[ToolCall]:
        calls = []
        for raw in getattr(message, "tool_calls", None) or []:
            try:
                payload = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                calls.append(ToolCall(id=raw.id, name=raw.function.name, args=[]))
                continue
            calls.append(ToolCall(id=raw.id, name=raw.function.name, args=self.tool_payload(raw.function.name, payload)))
        return calls

    @staticmethod
    def tool_payload(name: str, payload: Any) -> list[Any]:
        if isinstance(payload, dict) and (tool := TOOL_REGISTRY.get(name)):
            return tool.payload_args(payload)
        return [payload]


class Agent:
    SYSTEM_PROMPT = """\
You are nanocode, a concise terminal coding agent.

TOOLS: Read LineCount List Find InspectCode Search Edit Bash Git Recall Note.
Use EXACT tool names and named parameters. Obey each tool DESCRIPTION/SIGNATURE.

FLOW:
- ACT when clear; keep using tools until done.
- Every turn must call tools or return final; never emit empty content.
- Prefer built-ins over Bash. Batch independent read-only calls.
- All assistant text is user-visible, including interim narration before tool calls and final answers.
- Write every assistant text message as markdown in the latest user's language; do not switch languages unless asked.

CONTEXT:
- FILE STATE is the current snapshot for listed ranges; Read and Edit refresh it automatically.
- FILE STATE is not Memory/Recall; do not call it "full memory" or treat it as old output.
- Do not re-Read a file/range already present in FILE STATE when it has the needed lines and anchors; proceed to Edit.
- After a successful Edit, trust FILE STATE and do not re-Read just to verify the edited range.
- Recall bounded tr.N only when needed; prefer FILE STATE over old outputs.
- For multi-step work, call Note early with goal and a short plan; update plan/known/check when they change.

EDITS:
- Inspect/read before edits.
- Patch with Edit line:hash anchors from the newest FILE STATE.
- After stale-anchor errors or successful edits, discard old anchors for that file/range.
- If a stale-anchor error includes `current is line:hash|text`, retry Edit with that current anchor; do not Read first.
- Read after a stale-anchor error only when there is no usable `current is` anchor or surrounding lines are needed.
- Do not batch multiple Edit calls for the same file; sequence them.
- Keep edits small/local/reversible; never overwrite unrelated user work.
- Edit op=create creates files, including empty files.

FINAL:
- Concise markdown in the user's language.
- Include changed files and checks when relevant.
- Mention checks run, or say checks not run.\
"""

    def __init__(self, session: Session, input_fn=input, output_fn=print):
        self.session = session
        self.context = ContextManager(session)
        self.model = ModelClient(session)
        self.tools = ToolRunner(session, self.context, input_fn=input_fn, output_fn=output_fn)
        self.output_fn = output_fn

    def run(self, user_input: str) -> str:
        self.session.state.turn_step = 0
        self.session.state.turn_tool_calls = 0
        tool_batches = 0
        turn_messages: list[Json] = [{"role": "user", "content": user_input}]
        for step in range(self.session.settings.max_steps):
            self.session.state.turn_step = step + 1
            while True:
                try:
                    messages, pending = self.messages(turn_messages)
                    assistant, tool_calls, content = self.model.request(messages)
                    self.accept_pending_inputs(turn_messages, pending)
                    break
                except ModelRequestRetry:
                    continue
            if not tool_calls:
                if not content.strip():
                    raise ModelError("empty final response")
                answer = content.strip()
                self.session.messages.extend([*turn_messages, {"role": "assistant", "content": answer}])
                self.session.state.turn_messages = 0
                return answer
            assistant = self.assistant_turn_message(assistant, tool_calls, content)
            turn_messages.append(assistant)
            if content.strip():
                self.output_fn(content.strip())
            tool_batches += 1
            turn_messages.extend(self.tools.run(tool_calls, batch_suffix=f"·{tool_batches}" if tool_batches > 1 else ""))
        stopped = f"Stopped after max_agent_steps={self.session.settings.max_steps}"
        self.session.messages.extend([*turn_messages, {"role": "assistant", "content": stopped}])
        self.session.state.turn_messages = 0
        return stopped

    def messages(self, turn_messages: list[Json]) -> tuple[list[Json], list[str]]:
        pending = [text for text in self.session.pending_user_inputs if text.strip()]
        request_turn = [*turn_messages, *({"role": "user", "content": text} for text in pending)]
        self.session.state.turn_messages = len(request_turn)
        self.context.maybe_compact(self.model, self.SYSTEM_PROMPT, request_turn)
        messages = self.context.model_messages(self.SYSTEM_PROMPT, request_turn)
        self.context.update_percent(messages)
        return messages, pending

    def accept_pending_inputs(self, turn_messages: list[Json], pending: list[str]) -> None:
        if not pending:
            return
        turn_messages.extend({"role": "user", "content": text} for text in pending)
        remaining = list(self.session.pending_user_inputs)
        for text in pending:
            for index, value in enumerate(remaining):
                if value.strip() == text:
                    del remaining[index]
                    break
        self.session.pending_user_inputs = remaining

    @staticmethod
    def assistant_turn_message(assistant: Json, tool_calls: list[ToolCall], content: str) -> Json:
        message = dict(assistant or {})
        message["role"] = "assistant"
        message["content"] = message.get("content") if message.get("content") is not None else (content.strip() or None)
        if tool_calls and not message.get("tool_calls"):
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps({"args": call.args}, ensure_ascii=False)}}
                for call in tool_calls
            ]
        return Text.value(message)


class CommandCompleter(Completer):
    COMMANDS = (
        "/help",
        "/status",
        "/memory",
        "/config",
        "/api",
        "/debug",
        "/compact",
        "/index",
        "/model",
        "/provider",
        "/reason",
        "/set",
        "/yolo",
        "/exit",
        "/quit",
    )
    SET_KEYS = (
        "provider.model",
        "provider.url",
        "provider.key",
        "provider.api",
        "provider.prompt_cache_key",
        "provider.reasoning",
        "provider.chat_reasoning",
        "provider.available_models",
        "provider.temperature",
        "provider.timeout",
        "runtime.yolo",
        "runtime.max_agent_steps",
        "runtime.max_context_tokens",
        "runtime.shell_timeout",
        "runtime.check_updates",
        "runtime.update_check_interval_hours",
    )
    SET_VALUES = {
        "provider.api": PROVIDER_API_CHOICES,
        "provider.prompt_cache_key": ("auto", "off"),
        "provider.reasoning": REASONING_CHOICES,
        "provider.chat_reasoning": CHAT_REASONING_CHOICES,
        "provider.temperature": ("off",),
        "runtime.yolo": ("on", "off", "true", "false"),
        "runtime.check_updates": ("on", "off", "true", "false"),
    }

    def __init__(self, providers: Callable[[], tuple[str, ...]] = tuple, models: Callable[[], tuple[str, ...]] = tuple):
        self.providers = providers
        self.models = models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/set "):
            tail = text[len("/set ") :]
            if " " not in tail:
                yield from self.matches(self.SET_KEYS, tail)
                return
            key, _, value = tail.partition(" ")
            yield from self.matches(self.SET_VALUES.get(key, ()), value)
            return
        for command, values in (
            ("/api ", lambda: PROVIDER_API_CHOICES),
            ("/debug ", lambda: ("on", "off")),
            ("/model ", self.models),
            ("/provider ", self.providers),
            ("/reason ", lambda: REASONING_CHOICES),
        ):
            if text.startswith(command):
                yield from self.matches(values(), text[len(command) :])
                return
        if text.startswith("/mcp "):
            tail = text[len("/mcp ") :]
            if " " not in tail:
                yield from self.matches(("tools", "refresh"), tail)
                return

        if text.startswith("/") and " " not in text:
            yield from self.matches(self.COMMANDS, text)

    @staticmethod
    def matches(values, prefix: str):
        return (Completion(value, start_position=-len(prefix)) for value in values if value.startswith(prefix))


class UiPrinter:
    def __init__(self, output_fn=print):
        self.output_fn = output_fn
        self.color = output_fn is print and sys.stdout.isatty()
        self.console = Console() if self.color else None

    def emit(self, text: str = "") -> None:
        if not self.color:
            self.output_fn(text)
            return
        print_formatted_text(FormattedText(self.segments(str(text))), end="", flush=True)

    def emit_answer(self, text: str) -> None:
        if not self.color or text.startswith(("Error:", "ConfigError:", "Unknown command:")):
            self.emit(text)
            return
        assert self.console is not None
        self.console.print(Markdown(text))

    def segments(self, text: str) -> list[tuple[str, str]]:
        if text.startswith("tool "):
            return self.tool_segments(text)
        if text.startswith("approve ") or text.startswith("auto "):
            return self.approval_segments(text)
        if text.startswith(("goal ->", "goal:", "plan:", "known:")):
            return self.memory_segments(text)
        if text.startswith("+ "):
            return [("ansibrightblack", "+ "), ("ansiwhite", text[2:] + "\n")]
        if text.startswith("[done in "):
            return [("ansibrightblack", text + "\n")]
        if text.startswith("nanocode "):
            return [("ansicyan", text + "\n")]
        if text.startswith("Error:") or text.startswith("ConfigError:") or text.startswith("Unknown command:"):
            return [("ansired", text + "\n")]
        return [("ansiwhite", line + "\n") for line in text.splitlines() or [""]]

    def tool_segments(self, text: str) -> list[tuple[str, str]]:
        segments = []
        for line in text.splitlines() or [""]:
            if line.startswith("tool "):
                body = line[5:]
                call, sep, tail = body.partition(" -> ")
                failed = body.endswith(" [failed]") or body.endswith(" [refused]")
                call_style = "ansired" if failed else "ansigreen"
                tail_style = "ansired" if failed else "ansibrightblack"
                segments.extend([("ansibrightblack", "tool "), (call_style, call)])
                if sep:
                    segments.append((tail_style, sep + tail))
            elif line.startswith("  error "):
                segments.extend([("ansibrightblack", "  error "), ("ansired", line[8:])])
            elif line.startswith("  "):
                label, value = line[:8], line[8:]
                segments.extend([("ansibrightblack", label), ("ansiwhite", value)])
            else:
                segments.append(("ansiwhite", line))
            segments.append(("", "\n"))
        return segments

    def memory_segments(self, text: str) -> list[tuple[str, str]]:
        segments = []
        for line in text.splitlines() or [""]:
            if line.startswith(("goal ->", "goal:")):
                segments.append(("ansimagenta", line))
            elif line in {"summary:", "plan:", "known:"}:
                segments.append(("ansicyan", line))
            elif line.lstrip().startswith("+ "):
                segments.append(("ansigreen", line))
            else:
                segments.append(("ansiwhite", line))
            segments.append(("", "\n"))
        return segments

    def approval_segments(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines() or [text]
        head, _, rest = lines[0].partition(" ")
        style = "ansiyellow" if head == "approve" else "ansiblue"
        segments = [(style, head + ((" " + rest) if rest else "") + "\n")]
        if len(lines) <= 1:
            return segments
        if lines[1].strip() == "preview":
            segments.append(("ansibrightblack", "  preview\n"))
            preview = "\n".join(line[2:] if line.startswith("  ") else line for line in lines[2:])
            preview_segments = self.indent_segments(self.diff_segments(preview), "  ")
            segments.extend(preview_segments)
            if preview_segments and not preview_segments[-1][1].endswith("\n"):
                segments.append(("", "\n"))
            return segments
        segments.extend(("ansibrightblack", line + "\n") for line in lines[1:])
        return segments

    def diff_segments(self, text: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        old_line: int | None = None
        new_line: int | None = None
        lines = text.splitlines()

        def hunk_start(part: str, prefix: str) -> int | None:
            if not part.startswith(prefix):
                return None
            try:
                return int(part[1:].split(",", 1)[0])
            except ValueError:
                return None

        def number(old: int | None, new: int | None) -> None:
            old_text = "" if old is None else str(old)
            new_text = "" if new is None else str(new)
            segments.append(("ansibrightblack", f"{old_text:>4} {new_text:>4} | "))

        for index, line in enumerate(lines):
            suffix = "\n" if index < len(lines) - 1 else ""
            if line.startswith("@@"):
                parts = line.split()
                if len(parts) >= 3:
                    old_line = hunk_start(parts[1], "-")
                    new_line = hunk_start(parts[2], "+")
                number(None, None)
                segments.append(("ansicyan", line + suffix))
            elif line.startswith(("---", "+++")):
                number(None, None)
                segments.append(("ansibrightblack", line + suffix))
            elif line.startswith("+"):
                number(None, new_line)
                segments.append(("ansigreen", line + suffix))
                new_line = None if new_line is None else new_line + 1
            elif line.startswith("-"):
                number(old_line, None)
                segments.append(("ansired", line + suffix))
                old_line = None if old_line is None else old_line + 1
            elif line.startswith(" "):
                number(old_line, new_line)
                segments.append(("ansiwhite", line + suffix))
                old_line = None if old_line is None else old_line + 1
                new_line = None if new_line is None else new_line + 1
            else:
                number(None, None)
                segments.append(("ansiwhite", line + suffix))
        return segments

    @staticmethod
    def indent_segments(segments: list[tuple[str, str]], indent: str) -> list[tuple[str, str]]:
        indented: list[tuple[str, str]] = []
        at_start = True
        for style, text in segments:
            for part in text.splitlines(keepends=True):
                if at_start:
                    indented.append(("ansibrightblack", indent))
                indented.append((style, part))
                at_start = part.endswith("\n")
        return indented

class BashLivePreview:
    HEIGHT: ClassVar[int] = 6
    MAX_CHARS: ClassVar[int] = 8000

    def __init__(self):
        self.output = create_output(sys.stderr)
        self.active = False
        self.rendered_lines = 0
        self.text = ""

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        self.active, self.rendered_lines, self.text = True, 0, ""

    def update(self, text: str) -> None:
        if not self.active:
            return
        self.text = (self.text + text)[-self.MAX_CHARS :]
        self.render()

    def finish(self) -> None:
        if not self.active:
            return
        self.active, self.rendered_lines = False, 0

    def render(self) -> None:
        if not self.active:
            return
        lines = self.frame_lines()
        previous = self.rendered_lines
        if self.rendered_lines:
            self.output.write_raw(f"\x1b[{self.rendered_lines}A")
        for line in lines:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText([("ansibrightblack", line)]), output=self.output, end="", flush=True)
            self.output.write_raw("\n")
        for _ in range(max(0, previous - len(lines))):
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.write_raw("\n")
        if previous > len(lines):
            self.output.write_raw(f"\x1b[{previous - len(lines)}A")
        self.output.flush()
        self.rendered_lines = len(lines)

    def frame_lines(self) -> list[str]:
        width = max(20, shutil.get_terminal_size((120, 20)).columns)
        body = [line.expandtabs(4) for line in self.text.replace("\r", "\n").splitlines()[-self.HEIGHT :]]
        limit = width - 2
        return ["  output", *("  " + (line if len(line) <= limit else line[: max(0, limit - 3)] + "...") for line in body)] if body else []


class ModelRetryShortcut:
    CTRL_G = 0x07

    def __init__(self, session: Session):
        self.session = session
        self.fd: int | None = None
        self.original_attrs = None
        self.previous_handler = None

    def __enter__(self):
        if not sys.stdin.isatty() or not hasattr(signal, "SIGQUIT"):
            return self
        try:
            import termios

            self.fd = sys.stdin.fileno()
            self.original_attrs = termios.tcgetattr(self.fd)
            attrs = list(self.original_attrs)
            attrs[6] = list(attrs[6])
            attrs[6][termios.VQUIT] = self.control_char(attrs[6], self.CTRL_G)
            if hasattr(termios, "VREPRINT"):
                attrs[6][termios.VREPRINT] = self.control_char(attrs[6], os.fpathconf(self.fd, "PC_VDISABLE"))
            termios.tcsetattr(self.fd, termios.TCSADRAIN, attrs)
            self.previous_handler = signal.getsignal(signal.SIGQUIT)
            signal.signal(signal.SIGQUIT, self.handle_signal)
        except Exception:
            self.fd = None
            self.original_attrs = None
        return self

    def __exit__(self, *args) -> None:
        try:
            import termios

            if self.previous_handler is not None:
                signal.signal(signal.SIGQUIT, self.previous_handler)
            if self.fd is not None and self.original_attrs is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attrs)
        except Exception:
            pass
        self.fd = None
        self.original_attrs = None
        self.previous_handler = None

    @staticmethod
    def control_char(chars: list[Any], value: int) -> int | bytes:
        return bytes([value]) if chars and isinstance(chars[0], bytes) else value

    def handle_signal(self, _signum: int, _frame: Any) -> None:
        if self.session.state.current_model_call_started_at > 0:
            self.session.state.manual_model_retry_requested = True
            self.session.state.model_retry_count += 1
            raise KeyboardInterrupt


class StatusBar:
    INTERVAL: ClassVar[float] = 0.2
    INDEX_SPINNER: ClassVar[tuple[str, ...]] = ("~", "/", "-", "\\", "|")
    BASE_STYLE: ClassVar[str] = "#e6edf3"
    SEP_STYLE: ClassVar[str] = "#4b5563"
    STYLES: ClassVar[dict[str, str]] = {
        "provider": "#e6edf3",
        "reason": "#a5b4fc",
        "debug": "#64748b",
        "mcp": "#93c5fd",
        "ctx": "#facc15",
        "update": "#fb923c",
        "index": "#94a3b8",
        "warn": "#fb7185",
        "runtime": "#c084fc",
    }

    def __init__(self, session: Session):
        self.session = session
        self.started_at = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rendered = False
        self.output = create_output(sys.stderr)
        self.seen_retry_count = session.state.model_retry_count
        self.retry_notice_until = 0.0

    def start(self, *, reset: bool = True) -> None:
        if self.thread is not None or not sys.stderr.isatty():
            return
        self.begin(reset=reset)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def begin(self, *, reset: bool = True) -> None:
        if reset or not self.started_at:
            self.started_at = time.monotonic()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        self.clear()

    def is_running(self) -> bool:
        return self.thread is not None

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(self.active_fragments()), output=self.output, end="", flush=True)
            self.rendered = True
            self.stop_event.wait(self.INTERVAL)

    def refresh_retry_state(self) -> None:
        count = self.session.state.model_retry_count
        if count == self.seen_retry_count:
            return
        self.seen_retry_count = count
        self.retry_notice_until = time.monotonic() + 2.0

    def model_elapsed(self) -> float:
        return max(0.0, time.monotonic() - started) if (started := self.session.state.current_model_call_started_at) > 0 else 0.0

    def clear(self) -> None:
        if self.rendered:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.flush()
            self.rendered = False

    def idle_fragments(self) -> list[tuple[str, str]]:
        return self.fragments(0.0, sweep=False, show_elapsed=False)

    def active_fragments(self) -> list[tuple[str, str]]:
        self.refresh_retry_state()
        elapsed = max(0.0, time.monotonic() - self.started_at) if self.started_at else 0.0
        return self.fragments(elapsed, sweep=True, show_elapsed=True)

    def fragments(self, elapsed: float, *, sweep: bool, show_elapsed: bool) -> list[tuple[str, str]]:
        entries = self.entries(elapsed, show_elapsed=show_elapsed)
        text = " | ".join(text for text, _ in entries)
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(text) >= columns:
            text = text[: max(0, columns - 4)] + "..."
            return self.sweep_fragments(text, elapsed) if sweep else [(self.BASE_STYLE, text)]
        return self.sweep_fragments(text, elapsed) if sweep else self.styled_fragments(entries)

    def text(self, elapsed: float, *, show_elapsed: bool) -> str:
        return " | ".join(text for text, _ in self.entries(elapsed, show_elapsed=show_elapsed))

    def entries(self, elapsed: float, *, show_elapsed: bool) -> list[tuple[str, str]]:
        provider = self.session.config.provider
        model = provider.model.rsplit("/", 1)[-1] or "(no model)"
        reason = provider.reasoning
        if self.session.settings.debug:
            reason += "/" + provider.resolved_chat_reasoning()
        parts = [(self.session.config.active_provider + "/" + model, "provider"), (reason, "reason")]
        if self.session.settings.debug:
            parts.append(("api " + provider.resolved_api(), "debug"))

        mcp_status = self.mcp_status()
        if mcp_status:
            parts.append((mcp_status, "mcp"))
        parts.append(("ctx " + str(self.session.state.context_percent) + "%", "ctx"))
        if self.session.settings.debug and self.session.usage.cached_prompt_tokens:
            parts.append(("cache " + str(self.session.usage.cached_prompt_tokens), "debug"))
        update_status = self.update_status()
        if update_status:
            parts.append((update_status, "update"))
        index_status = self.index_status()
        if index_status:
            parts.append(("index" + index_status, "index"))
        if self.session.settings.yolo:
            parts.append(("yolo", "warn"))
        if show_elapsed:
            parts.extend(
                [
                    ("step " + str(self.session.state.turn_step) + "/" + str(self.session.settings.max_steps), "runtime"),
                    ("tools " + str(self.session.state.turn_tool_calls), "runtime"),
                ]
            )
        if show_elapsed and self.session.pending_user_inputs:
            parts.append(("+" + str(len(self.session.pending_user_inputs)), "warn"))
        if show_elapsed:
            parts.append((self.duration(elapsed), "runtime"))
            if self.retry_notice_until > time.monotonic():
                parts.append(("retrying", "warn"))
            elif self.model_elapsed() >= self.stress_after():
                parts.append(("ctrl-g retry", "warn"))
        return parts

    def styled_fragments(self, entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        for index, (text, role) in enumerate(entries):
            if index:
                fragments.append((self.SEP_STYLE, " | "))
            fragments.append((self.STYLES.get(role, self.BASE_STYLE), text))
        return fragments or [("", "")]

    def sweep_fragments(self, text: str, elapsed: float) -> list[tuple[str, str]]:
        if not text:
            return [("", "")]
        width = max(1, len(text) - 1)
        sweep = (time.monotonic() * 0.55) % 1.0
        model_elapsed = self.model_elapsed()
        heat = min(1.0, max(0.0, model_elapsed - self.stress_after()) / max(30.0, self.session.config.provider.timeout - self.stress_after()))
        fragments = []
        for index, char in enumerate(text):
            ratio = index / width
            red = round(75 + (180 - 75) * ratio)
            green = round(180 + (130 - 180) * ratio)
            blue = 235
            red = round(red + (240 - red) * heat)
            green = round(green * (1 - 0.65 * heat))
            blue = round(blue * (1 - 0.75 * heat))
            intensity = max(0.0, 1.0 - abs(ratio - sweep) * 5.0) ** 2
            red = round(red + (230 - red) * intensity)
            green = round(green + (245 - green) * intensity)
            blue = round(blue + (255 - blue) * intensity)
            fragments.append((f"#{red:02x}{green:02x}{blue:02x}", char))
        return fragments

    def index_status(self) -> str:
        if self.session.state.code_index_error:
            return CodeIndex.label("error")
        if self.session.state.code_index_refreshing:
            notice = self.session.state.code_index_notice or "syncing"
            return self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)] if notice in {"syncing", "updating"} else notice
        return CodeIndex.label(self.session.state.code_index_status)

    def update_status(self) -> str:
        if not self.session.settings.check_updates:
            return ""
        update = self.session.update
        if update.checking:
            return "update..."
        return "update " + update.latest if update.newer_than(__version__) else ""

    def mcp_status(self) -> str:
        status = self.session.mcp.discovery_status
        if status == "discovering":
            return "mcp" + self.INDEX_SPINNER[int(time.monotonic() / self.INTERVAL) % len(self.INDEX_SPINNER)]
        if status == "error":
            return "mcp err"
        return f"mcp {len(self.session.mcp.tools)}" if status == "ready" else ""

    def stress_after(self) -> float:
        return max(30.0, self.session.config.provider.timeout * 0.5)

    @staticmethod
    def duration(seconds: float) -> str:
        minutes, rest = divmod(int(seconds), 60)
        return f"{seconds:.1f}s" if seconds < 60 else f"{minutes}m{rest:02d}s"


class CommandLoop:
    MODEL_CONFIGURED_LABEL = "---- Configured models ----"
    MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
    MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))

    HELP = """Commands:
  /help              Show this help.
  /status            Show runtime status.
  /memory            Show durable agent memory.
  /config            Show active config.
  /api [NAME]        Show or set provider API format: auto, chat, anthropic.
  /debug [on|off]    Toggle model I/O debug traces.
  /compact           Compact context now.
  /index [force]      Sync or rebuild code symbol index.
  /provider [NAME]   Select or show the active provider.
  /model [MODEL]     Select or set the active model.
  /reason            Select reasoning effort.
  /set KEY VALUE     Set provider.* and runtime.*.
  /yolo              Toggle tool confirmations.
  /mcp               Show MCP server status.
  /mcp tools [NAME]   List MCP tools.
  /mcp refresh [NAME] Refresh MCP servers.
  /exit, /quit       Exit.
Tools:
  Read, LineCount, List, Find, InspectCode, Search, Edit, Bash, Git, Recall, Note, MCP.
"""

    def __init__(self, agent: Agent, input_fn=input, output_fn=print):
        self.agent = agent
        self.session = agent.session
        self.input_fn = input_fn
        self.ui = UiPrinter(output_fn)
        self.status_bar = StatusBar(self.session)
        self.live_preview = BashLivePreview()
        self.live_status_paused = False
        self.live_queue_paused = False
        self.transient_tool_lines = 0
        self.approval_full_preview = ""
        self.interactive_input = input_fn is input and sys.stdin.isatty()
        self.queue_input_paused = threading.Event()
        self.queue_input_active = threading.Event()
        self.queue_input_app: Application | None = None
        self.queue_input_text = ""
        if self.interactive_input:
            history_path = self.session.data_path("history.txt")
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            self.input_history = FileHistory(history_path)
        else:
            self.input_history = None
        self.input_completer = CommandCompleter(
            providers=lambda: tuple(sorted(self.session.config.providers)), models=lambda: self.session.config.provider.available_models
        )
        self.agent.output_fn = self.agent_output
        self.agent.tools.output_fn = self.tool_output
        self.agent.tools.input_fn = self.tool_input
        self.agent.tools.preview_fn = self.tool_preview
        self.agent.tools.preview_full_fn = lambda text: setattr(self, "approval_full_preview", text)
        self.agent.tools.live_start = self.tool_live_start
        self.agent.tools.live_output = self.tool_live_output

    @staticmethod
    def exit_app(app: Application) -> None:
        def close() -> None:
            try:
                app.exit(result=None)
            except Exception:
                pass

        loop = getattr(app, "loop", None)
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(close)
        else:
            close()

    def queue_input_until(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if self.queue_input_paused.is_set():
                stop_event.wait(0.05)
                continue
            self.run_queue_input_app(stop_event)

    def run_queue_input_app(self, stop_event: threading.Event) -> None:
        prompt = FormattedText([("class:prompt", "+> ")])

        def changed(buffer: Buffer) -> None:
            self.queue_input_text = buffer.text

        buffer = Buffer(document=Document(self.queue_input_text), multiline=False, on_text_changed=changed)
        control = BufferControl(buffer=buffer, input_processors=[BeforeInput(prompt)])
        input_window = Window(control, height=1, dont_extend_height=True, wrap_lines=False)
        bindings = KeyBindings()

        def record(event, texts: list[str]) -> None:
            texts = [Text.clean(text.strip()) for text in texts if text.strip()]
            if not texts:
                return
            self.session.pending_user_inputs.extend(texts)

            def show() -> None:
                for text in texts:
                    self.emit("+ " + text)

            run_in_terminal(show)

        @bindings.add("enter", eager=True)
        def _enter(event):
            record(event, [buffer.text])
            self.queue_input_text = ""
            buffer.reset(Document(""))

        @bindings.add("c-c", eager=True)
        def _ctrl_c(event):
            os.kill(os.getpid(), signal.SIGINT)

        @bindings.add("c-g", eager=True)
        def _ctrl_g(event):
            if self.session.state.current_model_call_started_at > 0:
                self.session.state.manual_model_retry_requested = True
                self.session.state.model_retry_count += 1
                os.kill(os.getpid(), signal.SIGINT)

        @bindings.add(Keys.BracketedPaste)
        def _paste(event):
            parts = event.data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if len(parts) == 1:
                buffer.insert_text(parts[0])
                return
            record(event, [buffer.text + parts[0], *parts[1:-1]])
            self.queue_input_text = parts[-1]
            buffer.reset(Document(self.queue_input_text))

        app = Application(
            layout=Layout(HSplit([input_window, self.status_window(active=True)]), focused_element=input_window),
            key_bindings=bindings,
            full_screen=False,
            style=self.style(),
            refresh_interval=StatusBar.INTERVAL,
            erase_when_done=True,
        )
        self.queue_input_app = app
        self.queue_input_active.set()

        def stop_when_needed() -> None:
            while not stop_event.is_set() and not self.queue_input_paused.is_set():
                stop_event.wait(0.05)
            self.exit_app(app)

        threading.Thread(target=stop_when_needed, daemon=True).start()
        try:
            with patch_stdout():
                app.run()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self.queue_input_text = buffer.text
            self.queue_input_active.clear()
            if self.queue_input_app is app:
                self.queue_input_app = None

    def pause_queue_input(self) -> None:
        self.queue_input_paused.set()
        if self.queue_input_app is not None:
            self.exit_app(self.queue_input_app)
        deadline = time.monotonic() + 1.0
        while self.queue_input_active.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)

    def run(self) -> int:
        self.emit(f"nanocode {__version__}. /help for commands.")
        CodeIndex(self.session).refresh_existing_async()
        # Async MCP discovery — show nano> immediately, discover in background
        threading.Thread(target=self.session.mcp.discover_enabled, daemon=True).start()
        UpdateChecker(self.session).start()
        while True:
            try:
                user_input = self.read_input(initial_text=self.queue_input_text)
                self.queue_input_text = ""
            except EOFError:
                self.emit("")
                return 0
            except KeyboardInterrupt:
                self.emit("Cancelled")
                continue
            if not user_input.strip():
                continue
            handled, exit_now = self.command(user_input.strip())
            if exit_now:
                return 0
            if handled:
                continue
            started = time.monotonic()
            stop_input = threading.Event()
            watcher = threading.Thread(target=self.queue_input_until, args=(stop_input,), daemon=True) if self.interactive_input else None
            try:
                if watcher:
                    self.status_bar.begin()
                    watcher.start()
                else:
                    self.status_bar.start()
                try:
                    with ModelRetryShortcut(self.session):
                        answer = self.agent.run(user_input)
                except KeyboardInterrupt:
                    self.emit("Cancelled")
                    continue
                except NanocodeError as error:
                    answer = f"Error: {error}"
            finally:
                stop_input.set()
                self.pause_queue_input()
                if watcher:
                    watcher.join(timeout=1.0)
                self.queue_input_paused.clear()
                self.session.state.manual_model_retry_requested = False
                CodeIndex(self.session).update_pending()
                self.status_bar.stop()
            elapsed = time.monotonic() - started
            self.ui.emit_answer(answer)
            self.emit(f"[done in {int(elapsed // 60)}m{elapsed % 60:.0f}s]")

    def style(self) -> Style:
        return Style.from_dict(
            {
                "prompt": "ansicyan bold",
                "approval": "ansiyellow",
                "approval.wait": "ansimagenta",
                "choice.title": "ansicyan bold",
                "choice.selected": "reverse",
                "choice.disabled": "ansibrightblack",
                "completion-menu": "noreverse bg:default",
                "completion-menu.completion": "noreverse bg:default fg:ansiwhite",
                "completion-menu.completion.current": "noreverse bg:default fg:ansicyan bold",
                "completion-menu.meta.completion": "noreverse bg:default fg:ansibrightblack",
                "completion-menu.meta.completion.current": "noreverse bg:default fg:ansicyan",
                "bottom-toolbar": "noreverse bg:default fg:default",
                "bottom-toolbar.text": "noreverse bg:default fg:default",
                "search-toolbar": "noreverse bg:default fg:default",
                "search-toolbar.prompt": "ansicyan",
                "search-toolbar.text": "ansiwhite",
            }
        )

    def status_window(self, *, active: bool = False) -> Window:
        return Window(
            FormattedTextControl(self.status_bar.active_fragments if active else self.status_bar.idle_fragments, style="class:bottom-toolbar.text"),
            style="class:bottom-toolbar",
            height=1,
            dont_extend_height=True,
        )

    def run_input_app(self, app: Application) -> Any:
        self.pause_queue_input()
        try:
            with patch_stdout():
                return app.run()
        finally:
            self.queue_input_paused.clear()

    def input_prompt_fragments(self, prompt_text: str, prompt_style: str) -> list[tuple[str, str]]:
        if prompt_style != "class:approval" or not prompt_text:
            return [(prompt_style, prompt_text)]
        frame = "|/-\\"[int(time.monotonic() / 0.2) % 4]
        return [("class:approval", prompt_text), ("class:approval.wait", frame + " ")]

    def read_input(
        self,
        prompt_text: str = "nano> ",
        *,
        multiline: bool = False,
        submit_on_enter: bool = False,
        prompt_style: str = "class:prompt",
        initial_text: str = "",
    ) -> str:
        if self.input_history is None:
            return self.input_fn(prompt_text)

        def accept(buffer: Buffer) -> bool:
            app.exit(result=buffer.text)
            return True

        buffer = Buffer(
            history=self.input_history,
            completer=self.input_completer,
            complete_while_typing=False,
            enable_history_search=True,
            multiline=multiline,
            accept_handler=accept,
            document=Document(initial_text, cursor_position=len(initial_text)),
        )
        search_toolbar = SearchToolbar()
        control = BufferControl(
            buffer=buffer,
            input_processors=[HighlightIncrementalSearchProcessor(), BeforeInput(lambda: self.input_prompt_fragments(prompt_text, prompt_style))],
            search_buffer_control=search_toolbar.control,
            preview_search=True,
        )
        input_window = Window(control, height=Dimension(min=1, max=6), dont_extend_height=True, wrap_lines=True)
        bindings = KeyBindings()

        @bindings.add("c-c", eager=True)
        @bindings.add("<sigint>", eager=True)
        def _ctrl_c(event):
            event.app.exit(exception=KeyboardInterrupt())

        @bindings.add("c-d", eager=True)
        def _ctrl_d(event):
            if multiline:
                event.app.exit(result=buffer.text)
            elif buffer.text:
                buffer.delete()
            else:
                event.app.exit(exception=EOFError())

        @bindings.add("escape", "enter", filter=Condition(lambda: multiline), eager=True)
        def _escape_enter(event):
            event.app.exit(result=buffer.text)

        @bindings.add("enter", filter=Condition(lambda: submit_on_enter), eager=True)
        def _enter(event):
            event.app.exit(result=buffer.text)

        @bindings.add("c-r", eager=True)
        def _ctrl_r(event):
            direction = pt_search.SearchDirection.BACKWARD
            if event.app.layout.current_control is search_toolbar.control:
                pt_search.do_incremental_search(direction, count=event.arg)
            else:
                pt_search.start_search(direction=direction)

        @bindings.add("c-a", filter=Condition(lambda: bool(self.approval_full_preview)), eager=True)
        def _ctrl_a(event):
            run_in_terminal(self.open_approval_preview)

        @bindings.add("tab")
        def _tab(event):
            if buffer.complete_state:
                buffer.complete_next()
                return
            completions = list(self.input_completer.get_completions(buffer.document, CompleteEvent(completion_requested=True)))
            if len(completions) == 1:
                buffer.apply_completion(completions[0])
            else:
                buffer.start_completion(select_first=False)

        @bindings.add("s-tab")
        def _shift_tab(event):
            buffer.complete_previous() if buffer.complete_state else buffer.start_completion(select_last=True)

        @bindings.add(Keys.BracketedPaste)
        def _paste(event):
            buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n"))
            event.app.invalidate()

        completion_space = ConditionalContainer(Window(height=12, dont_extend_height=True), filter=has_completions & ~is_done)
        root = FloatContainer(
            HSplit([input_window, completion_space, search_toolbar, self.status_window()]),
            [Float(CompletionsMenu(max_height=12, scroll_offset=1), xcursor=True, ycursor=True, attach_to_window=input_window, transparent=True)],
        )
        app = Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=bindings,
            full_screen=False,
            style=self.style(),
            refresh_interval=StatusBar.INTERVAL,
            erase_when_done=True,
        )
        text = self.run_input_app(app)
        print_formatted_text(FormattedText([(prompt_style, prompt_text), ("", text)]), style=self.style())
        return text

    def emit(self, text: str = "") -> None:
        self.ui.emit(str(text))

    def with_status_paused(self, action):
        had_queue = self.queue_input_active.is_set()
        if had_queue:
            self.pause_queue_input()
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.stop()
        try:
            return action()
        finally:
            if was_running:
                self.status_bar.start(reset=False)
            if had_queue:
                self.queue_input_paused.clear()

    def tool_output(self, text: str = "") -> None:
        if text.startswith("approve ") and self.interactive_input and sys.stdout.isatty():
            self.with_status_paused(lambda: self.show_transient_tool_output(text))
            return

        def emit() -> None:
            self.clear_transient_tool_output()
            self.emit(text)

        self.with_status_paused(emit)

    def agent_output(self, text: str = "") -> None:
        self.with_status_paused(lambda: self.emit_agent_output(text))

    def tool_input(self, prompt: str = "") -> str:
        def read() -> str:
            try:
                return (
                    self.read_input(prompt, multiline=True, submit_on_enter=True, prompt_style="class:approval")
                    if self.interactive_input
                    else self.input_fn(prompt)
                )
            finally:
                if self.interactive_input and sys.stdout.isatty():
                    self.clear_transient_tool_output()
                    self.approval_full_preview = ""

        return self.with_status_paused(read)

    def show_transient_tool_output(self, text: str) -> None:
        self.clear_transient_tool_output()
        self.emit(text)
        self.transient_tool_lines = len(text.splitlines() or [""])

    def tool_preview(self, text: str) -> bool:
        if not text.startswith("approve Edit ") or not self.interactive_input or not sys.stdout.isatty():
            return False
        self.with_status_paused(lambda: self.show_transient_tool_preview(text))
        return True

    def open_approval_preview(self) -> None:
        if not self.approval_full_preview:
            return
        fd, path = tempfile.mkstemp(prefix="nanocode-preview-", suffix=".diff")
        try:
            pager_env = os.environ.get("PAGER", "")
            pager = shlex.split(pager_env) if pager_env else ([less, "-R"] if (less := shutil.which("less")) else [])
            text = self.ansi_diff_preview(self.approval_full_preview) if pager and os.path.basename(pager[0]) == "less" else self.approval_full_preview
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(text.rstrip() + "\n")
            if pager:
                subprocess.run([*pager, path])
            else:
                print(self.approval_full_preview)
                input("Press Enter to return...")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def ansi_diff_preview(text: str) -> str:
        colors = [("---", "\033[90m"), ("+++", "\033[90m"), ("@@", "\033[36m"), ("+", "\033[32m"), ("-", "\033[31m")]
        lines = []
        for line in text.splitlines():
            style = next((color for prefix, color in colors if line.lstrip().startswith(prefix)), "")
            lines.append(style + line + ("\033[0m" if style else ""))
        return "\n".join(lines)

    def show_transient_tool_preview(self, text: str) -> None:
        self.clear_transient_tool_output()
        lines = text.rstrip().splitlines()
        if not lines:
            return
        height, width = 12, max(20, shutil.get_terminal_size((120, 20)).columns)
        shown = lines[:height] + ([f"... preview truncated: {len(lines) - height} more lines (Ctrl-A: full preview) ..."] if len(lines) > height else [])
        self.emit("\n".join(line[: max(0, width - 1)] for line in shown))
        self.transient_tool_lines = len(shown)

    def emit_agent_output(self, text: str) -> None:
        self.clear_transient_tool_output()
        if self.ui.color and text.strip():
            self.emit()
            self.ui.emit_answer(text)
            self.emit()
            return
        self.emit(text)

    def clear_transient_tool_output(self) -> None:
        if not self.transient_tool_lines:
            return
        for _ in range(self.transient_tool_lines):
            sys.stdout.write("\x1b[1A\r\x1b[2K")
        sys.stdout.flush()
        self.transient_tool_lines = 0

    def tool_live_start(self) -> None:
        if not self.ui.color:
            return
        self.live_queue_paused = self.interactive_input and not self.queue_input_paused.is_set()
        if self.live_queue_paused or self.queue_input_active.is_set():
            self.pause_queue_input()
        self.live_status_paused = self.status_bar.is_running()
        if self.live_status_paused:
            self.status_bar.stop()
        self.live_preview.start()

    def tool_live_output(self, _stream: str, text: str) -> None:
        if not self.ui.color:
            return
        if text:
            if not self.live_preview.active:
                self.live_queue_paused = self.interactive_input and not self.queue_input_paused.is_set()
                if self.live_queue_paused or self.queue_input_active.is_set():
                    self.pause_queue_input()
                self.live_status_paused = self.status_bar.is_running()
                if self.live_status_paused:
                    self.status_bar.stop()
                self.live_preview.start()
            self.live_preview.update(text)
            return
        if self.live_preview.active:
            self.live_preview.finish()
        if self.live_status_paused:
            self.status_bar.start(reset=False)
            self.live_status_paused = False
        if self.live_queue_paused:
            self.queue_input_paused.clear()
            self.live_queue_paused = False

    def command(self, text: str) -> tuple[bool, bool]:
        if text in {"/exit", "/quit", "exit", "quit"}:
            return True, True
        if not text.startswith("/"):
            return False, False
        name, _, args = text.partition(" ")
        handlers = {
            "/help": self.help,
            "/status": self.status,
            "/memory": self.memory,
            "/config": self.config,
            "/api": self.api,
            "/debug": self.debug,
            "/compact": self.compact,
            "/index": self.index,
            "/provider": self.provider,
            "/model": self.model,
            "/reason": self.reason,
            "/set": self.set_value,
            "/yolo": self.yolo,
            "/mcp": self.mcp_command,
        }
        handler = handlers.get(name)
        output = handler(args.strip()) if handler else f"Unknown command: {name}"
        (self.ui.emit_answer if name == "/status" else self.emit)(output)
        return True, False

    def mcp_command(self, args: str) -> str:
        mcp = self.session.mcp
        if mcp is None:
            return "MCP not configured"

        parts = args.split()
        if not parts:
            return mcp.render_server_status()

        sub = parts[0]
        rest = parts[1:]

        if sub == "tools":
            server = rest[0] if rest else None
            return mcp.render_tool_listing(server)
        elif sub == "refresh":
            name = rest[0] if rest else ""
            if name:
                mcp.discover_server(name)
            else:
                mcp.discover_enabled()
            return mcp.render_server_status()
        else:
            return f"Unknown /mcp subcommand: {sub}. Try /mcp, /mcp tools [server], /mcp refresh [server]"


    def visible_choices(self, choices: tuple[str, ...], labels: dict[str, str], disabled: set[str], query: str) -> tuple[str, ...]:
        if not query:
            return choices
        needle = query.lower()
        visible: list[str] = []
        header = ""
        section: list[str] = []

        def flush() -> None:
            if section:
                if header:
                    visible.append(header)
                visible.extend(section)
            section.clear()

        for choice in choices:
            if choice in disabled:
                flush()
                header = choice
                continue
            text = (choice + " " + labels.get(choice, choice)).lower()
            if needle in text:
                section.append(choice)
        flush()
        return tuple(visible)

    def select_choice(
        self,
        title: str,
        choices: tuple[str, ...],
        *,
        labels: dict[str, str] | None = None,
        current: str = "",
        disabled: set[str] | frozenset[str] = frozenset(),
    ) -> str | object | None:
        labels = labels or {}
        if not choices or not self.interactive_input:
            return None
        try:
            return self.choice_application(title, choices, labels, current, set(disabled))
        except (EOFError, KeyboardInterrupt):
            self.emit("Cancelled")
            return None

    def choice_application(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str],
        current: str,
        disabled: set[str],
    ) -> str | object | None:
        state = {"query": "", "selected": 0, "search": False}
        searching = Condition(lambda: bool(state["search"]))

        def enabled() -> tuple[str, ...]:
            return tuple(choice for choice in self.visible_choices(choices, labels, disabled, str(state["query"])) if choice not in disabled)

        def clamp() -> None:
            options = enabled()
            state["selected"] = min(max(int(state["selected"]), 0), len(options) - 1) if options else 0

        def move(event, delta: int) -> None:
            options = enabled()
            if options:
                state["selected"] = min(max(int(state["selected"]) + delta, 0), len(options) - 1)
            event.app.invalidate()

        def fragments():
            query = str(state["query"])
            visible = self.visible_choices(choices, labels, disabled, query)
            options = enabled()
            clamp()
            suffix = (" /" + query) if query else ""
            if query and not state["search"]:
                suffix += " (filtered)"
            parts: list[tuple[str, str]] = [
                ("class:choice.title", title + suffix + "\n"),
                ("class:choice.disabled", "  j/k move, / search, Esc back/cancel\n"),
            ]
            if query and not options:
                parts.append(("class:choice.disabled", "  no matches\n"))
                return parts
            number = 0
            for choice in visible:
                label = labels.get(choice, choice)
                if choice in disabled:
                    parts.append(("class:choice.disabled", "  " + label + "\n"))
                    continue
                number += 1
                selected = number - 1 == int(state["selected"])
                style = "class:choice.selected" if selected else ""
                if selected:
                    parts.append(("[SetCursorPosition]", ""))
                parts.append((style, ("> " if selected else "  ") + f"{number:2d}. {label}\n"))
            if state["search"]:
                parts.append(("", "/" + query))
            return parts

        bindings = KeyBindings()

        @bindings.add("j", filter=~searching, eager=True)
        @bindings.add("down", eager=True)
        def _j(event):
            move(event, 1)

        @bindings.add("k", filter=~searching, eager=True)
        @bindings.add("up", eager=True)
        def _k(event):
            move(event, -1)

        @bindings.add("/", eager=True)
        def _search(event):
            state["search"] = True
            state["query"] = ""
            state["selected"] = 0
            event.app.invalidate()

        @bindings.add("backspace", filter=searching, eager=True)
        @bindings.add("c-h", filter=searching, eager=True)
        def _backspace(event):
            state["query"] = str(state["query"])[:-1]
            state["selected"] = 0
            event.app.invalidate()

        @bindings.add("escape", eager=True)
        def _escape(event):
            if state["search"]:
                state["search"] = False
                event.app.invalidate()
                return
            if state["query"]:
                state["query"] = ""
                state["selected"] = 0
                event.app.invalidate()
                return
            event.app.exit(result=SELECTION_BACK)

        @bindings.add("enter", eager=True)
        def _enter(event):
            if state["search"]:
                state["search"] = False
                event.app.invalidate()
                return
            options = enabled()
            if options:
                event.app.exit(result=options[int(state["selected"])])

        @bindings.add("c-c", eager=True)
        @bindings.add("<sigint>", eager=True)
        def _ctrl_c(event):
            event.app.exit(exception=KeyboardInterrupt())

        for number in range(1, 10):

            @bindings.add(str(number), eager=True)
            def _digit(event, number=number):
                if state["search"]:
                    state["query"] = str(state["query"]) + event.data
                    state["selected"] = 0
                    event.app.invalidate()
                    return
                options = enabled()
                if number <= len(options):
                    state["selected"] = number - 1
                    event.app.invalidate()

        @bindings.add(Keys.Any, filter=searching)
        def _typed(event):
            if event.data and event.data not in "\r\n":
                state["query"] = str(state["query"]) + event.data
                state["selected"] = 0
                event.app.invalidate()

        options = enabled()
        state["selected"] = options.index(current) if current in options else 0
        content = FormattedTextControl(fragments, focusable=True)
        choice_window = Window(content, dont_extend_height=True, wrap_lines=False)
        app = Application(
            layout=Layout(HSplit([choice_window, self.status_window()]), focused_element=choice_window),
            key_bindings=bindings,
            full_screen=False,
            style=self.style(),
            refresh_interval=StatusBar.INTERVAL,
            erase_when_done=True,
        )
        return self.run_input_app(app)

    def select_model(self, choices: tuple[str, ...]) -> str | object | None:
        current = self.session.config.provider.model
        labels = {label: label for label in self.MODEL_LABELS if label in choices}
        labels.update({current: current + " (current)"} if current in choices else {})
        return self.select_choice("Model", choices, labels=labels, current=current, disabled=self.MODEL_LABELS)

    def select_provider(self, choices: tuple[str, ...]) -> str | object | None:
        current = self.session.config.active_provider
        return self.select_choice("Provider", choices, labels={current: current + " (current)"}, current=current)

    def select_reasoning(self) -> str | object | None:
        current = self.session.config.provider.reasoning
        labels = {"off": "off - disable reasoning"}
        labels[current] = labels.get(current, current) + " (current)"
        return self.select_choice("Reasoning effort", REASONING_CHOICES, labels=labels, current=current)

    def help(self, args: str) -> str:
        return self.HELP.rstrip()

    def status(self, args: str) -> str:
        usage = self.session.usage
        provider = self.session.config.provider
        self.agent.context.update_percent(self.agent.context.model_messages(self.agent.SYSTEM_PROMPT))
        index_status, index_message = CodeIndex(self.session).status(check=True)
        if self.session.state.code_index_refreshing:
            index_status, index_message = self.session.state.code_index_notice or "syncing", ""
        elif self.session.state.code_index_error:
            index_status, index_message = "error", self.session.state.code_index_error
        if index_status in {"missing", "unavailable", "error"} and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index"
        elif index_status == "stale" and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index or wait for auto update"
        cache_ratio = (usage.cached_prompt_tokens * 100 / usage.total_tokens) if usage.total_tokens else 0
        last_cache_ratio = (usage.last_cached_prompt_tokens * 100 / usage.last_total_tokens) if usage.last_total_tokens else 0
        rows = [
            ("workspace", "`" + self.session.cwd + "`"),
            (
                "model",
                f"`{self.session.config.active_provider}/{provider.model or '(empty)'}`; api `{provider.resolved_api()} ({provider.api})`; reasoning `{provider.reasoning} ({provider.resolved_chat_reasoning()})`",
            ),
            (
                "context",
                f"ctx `{self.session.state.context_percent}%`; history `{len(self.session.messages)}`; turn `{self.session.state.turn_messages}`; tools `{len(self.session.tool_results)}`; files `{self.agent.context.file_count()}`; known `{len(self.session.state.known)}`",
            ),
            ("goal", self.session.state.goal or "(empty)"),
            (
                "usage",
                f"calls `{usage.calls}`; total `{usage.total_tokens}`; cached `{usage.cached_prompt_tokens}` (`{cache_ratio:.1f}%`); last `{usage.last_cached_prompt_tokens}/{usage.last_total_tokens}` (`{last_cache_ratio:.1f}%`)",
            ),
            (
                "runtime",
                f"yolo `{'on' if self.session.settings.yolo else 'off'}`; debug `{'on' if self.session.settings.debug else 'off'}`; max steps `{self.session.settings.max_steps}`",
            ),
            ("index", CodeIndex.status_line(index_status, index_message)),
            ("update", UpdateChecker(self.session).status_line().removeprefix("update: ")),
        ]
        return "\n".join(
            [
                "| status | value |",
                "| --- | --- |",
                *(f"| {name} | {Text.clean(str(value)).replace(chr(10), ' ').replace('|', chr(92) + '|')} |" for name, value in rows),
            ]
        )

    def memory(self, args: str) -> str:
        state = self.session.state
        known = ["- " + item for item in state.known] or ["- (empty)"]
        return "\n".join(
            ["goal: " + (state.goal or "(empty)"), "summary:", state.summary or "(empty)", "plan:", *state.plan_rows(status=True), "known:", *known]
        )

    def config(self, args: str) -> str:
        provider = self.session.config.provider
        return "\n".join(
            [
                f"provider.active: {self.session.config.active_provider}",
                f"provider.available: {', '.join(sorted(self.session.config.providers))}",
                f"provider.url: {provider.url or '(empty)'}",
                f"provider.key: {'(set)' if provider.key else '(empty)'}",
                f"provider.model: {provider.model or '(empty)'}",
                f"provider.api: {provider.api}",
                f"provider.resolved_api: {provider.resolved_api()}",
                f"provider.prompt_cache_key: {provider.prompt_cache_key}",
                f"provider.available_models: {', '.join(provider.available_models) or '(empty)'}",
                f"provider.reasoning: {provider.reasoning}",
                f"provider.resolved_chat_reasoning: {provider.resolved_chat_reasoning()}",
                f"provider.chat_reasoning: {provider.chat_reasoning}",
                f"provider.temperature: {provider.temperature if provider.temperature is not None else '(off)'}",
                f"provider.timeout: {provider.timeout}",
                f"paths.data_dir: {self.session.data_path()}",
                f"runtime.shell_timeout: {self.session.settings.shell_timeout}",
                f"runtime.max_agent_steps: {self.session.settings.max_steps}",
                f"runtime.max_context_tokens: {self.session.settings.max_context_tokens}",
                f"runtime.check_updates: {'on' if self.session.settings.check_updates else 'off'}",
                f"runtime.update_check_interval_hours: {self.session.settings.update_check_interval_hours}",
                f"runtime.yolo: {'on' if self.session.settings.yolo else 'off'}",
                f"runtime.debug: {'on' if self.session.settings.debug else 'off'}",
            ]
        )

    def api(self, args: str) -> str:
        value = args.strip()
        provider = self.session.config.provider
        if not value:
            return "provider.api: " + provider.api + "\nprovider.resolved_api: " + provider.resolved_api()
        if value not in PROVIDER_API_CHOICES:
            return "Usage: /api auto|chat|anthropic"
        provider.api = value
        return "Set provider.api = " + value + "\nprovider.resolved_api: " + provider.resolved_api()

    def debug(self, args: str) -> str:
        value = args.strip().lower()
        if not value:
            self.session.settings.debug = not self.session.settings.debug
        elif value in {"on", "true", "yes", "1"}:
            self.session.settings.debug = True
        elif value in {"off", "false", "no", "0"}:
            self.session.settings.debug = False
        else:
            return "Usage: /debug [on|off]"
        status = "on" if self.session.settings.debug else "off"
        lines = ["debug: " + status]
        if self.session.settings.debug:
            lines.append("debug_dir: " + self.session.debug_dir())
        return "\n".join(lines)

    def compact(self, args: str) -> str:
        if args.strip():
            return "Usage: /compact"
        before = len(self.session.messages)
        compacted, keep = self.agent.context.compaction_parts()
        if not compacted:
            return "No prior conversation to compact"
        fallback = False
        try:
            self.status_bar.start()
            data = self.agent.model.compact(self.agent.context.compaction_input(compacted))
        except KeyboardInterrupt:
            return "Cancelled"
        except Exception:
            self.agent.context.apply_compaction_fallback(keep)
            fallback = True
            data = None
        finally:
            self.status_bar.stop()
        if data is not None:
            self.agent.context.apply_compaction(data, keep)
        self.agent.context.update_percent(self.agent.context.model_messages(self.agent.SYSTEM_PROMPT))
        return (
            "Compacted context: messages "
            + str(before)
            + " -> "
            + str(len(self.session.messages))
            + ", prior summary inserted, ctx "
            + str(self.session.state.context_percent)
            + "%"
            + (" (fallback)" if fallback else "")
        )

    def index(self, args: str) -> str:
        value = args.strip()
        if value not in {"", "force"}:
            return "Usage: /index [force]"
        try:
            self.status_bar.start()
            return CodeIndex(self.session).sync(force=value == "force")
        finally:
            self.status_bar.stop()

    def provider(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /provider [NAME]"
        if parts:
            return self.set_provider(parts[0])
        choices = tuple(sorted(self.session.config.providers))
        if len(choices) <= 1:
            return self.provider_summary()
        choice = self.select_provider(choices)
        return self.set_provider(choice) if isinstance(choice, str) else ("No change" if choice is SELECTION_BACK else self.provider_summary())

    def provider_summary(self) -> str:
        return "provider: " + self.session.config.active_provider + "\nproviders: " + ", ".join(sorted(self.session.config.providers))

    def set_provider(self, name: str) -> str:
        if name not in self.session.config.providers:
            return "Unknown provider: " + name
        self.session.config.active_provider = name
        return "Set provider = " + name

    def model(self, args: str) -> str:
        parts = args.split()
        if len(parts) > 1:
            return "Usage: /model [MODEL]"
        if parts:
            result = self.set_model(parts[0])
            return "No change" if result is SELECTION_BACK else str(result)
        choices = self.model_choices()
        if not choices:
            return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
        while True:
            choice = self.select_model(choices)
            if choice is SELECTION_BACK:
                return "No change"
            if not isinstance(choice, str):
                return "Current provider.model is " + (self.session.config.provider.model or "(empty)")
            if choice in self.MODEL_LABELS:
                continue
            result = self.set_model(choice, back_to_model=True)
            if result is SELECTION_BACK:
                continue
            return str(result)

    def model_choices(self) -> tuple[str, ...]:
        provider = self.session.config.provider
        configured = tuple(dict.fromkeys(provider.available_models))
        remote = tuple(model for model in self.remote_models(provider) if model not in configured)
        choices: list[str] = []
        if configured:
            choices.extend((self.MODEL_CONFIGURED_LABEL, *configured))
        if remote:
            choices.extend((self.MODEL_DISCOVERED_LABEL, *remote))
        return tuple(choices)

    def remote_models(self, provider: ProviderConfig) -> tuple[str, ...]:
        if not provider.url or not provider.key:
            return ()
        try:
            page = OpenAI(
                api_key=provider.key,
                base_url=provider.base_url(),
                timeout=min(provider.timeout, 10),
                max_retries=0,
                default_headers={"User-Agent": HTTP_USER_AGENT},
            ).models.list()
        except Exception:
            return ()
        names = []
        for item in getattr(page, "data", page) or []:
            name = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            if isinstance(name, str) and name:
                names.append(name)
        return tuple(sorted(dict.fromkeys(names)))

    def set_model(self, model: str, *, back_to_model: bool = False) -> str | object:
        reasoning = self.select_reasoning()
        if reasoning is SELECTION_BACK:
            return SELECTION_BACK if back_to_model else "No change"
        provider = self.session.config.provider
        provider.model = model
        lines = ["Set provider.model = " + model]
        if isinstance(reasoning, str):
            provider.reasoning = reasoning
            lines.append("Set provider.reasoning = " + reasoning)
        return "\n".join(lines)

    def reason(self, args: str) -> str:
        if args:
            return "Usage: /reason"
        choice = self.select_reasoning()
        if isinstance(choice, str):
            self.session.config.provider.reasoning = choice
            return "Set provider.reasoning = " + choice
        return "No change"

    def yolo(self, args: str) -> str:
        self.session.settings.yolo = not self.session.settings.yolo
        return "yolo: " + ("on" if self.session.settings.yolo else "off")

    def set_value(self, args: str) -> str:
        key, _, value = args.partition(" ")
        if not key or not value:
            return "Usage: /set KEY VALUE"
        provider = self.session.config.provider
        runtime = self.session.settings
        choice_fields = {"provider.api": PROVIDER_API_CHOICES, "provider.reasoning": REASONING_CHOICES, "provider.chat_reasoning": CHAT_REASONING_CHOICES}
        try:
            if key in {"provider.model", "provider.url", "provider.key"}:
                setattr(provider, key.split(".", 1)[1], value)
            elif key in choice_fields:
                if value not in choice_fields[key]:
                    return "Invalid value for " + key
                setattr(provider, key.split(".", 1)[1], value)
            elif key == "provider.prompt_cache_key":
                provider.prompt_cache_key = ProviderConfig.clean_prompt_cache_key(value)
            elif key == "provider.available_models":
                provider.available_models = tuple(item.strip() for item in value.split(",") if item.strip())
            elif key == "provider.temperature":
                provider.temperature = None if value == "off" else float(value)
            elif key == "provider.timeout":
                provider.timeout = max(1, int(value))
            elif key == "runtime.yolo":
                runtime.yolo = value.lower() in {"on", "true", "yes", "1"}
            elif key == "runtime.check_updates":
                runtime.check_updates = Config.bool({key: value}, key)
                if runtime.check_updates:
                    UpdateChecker(self.session).start()
            elif key == "runtime.update_check_interval_hours":
                runtime.update_check_interval_hours = max(1, int(value))
            elif key == "runtime.max_agent_steps":
                runtime.max_steps = max(1, int(value))
            elif key == "runtime.max_context_tokens":
                runtime.max_context_tokens = max(1, int(value))
            elif key == "runtime.shell_timeout":
                runtime.shell_timeout = max(1, int(value))
            else:
                return "Unknown config key: " + key
        except (ConfigError, ValueError):
            return "Invalid value for " + key
        return "Set " + key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nanocode")
    parser.add_argument("--config", default=None, help="Path to config TOML")
    parser.add_argument("--init-config", action="store_true", help="Create a default config file")
    parser.add_argument("--yolo", action="store_true", help="Skip confirmations for mutating tools")
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    try:
        if args.init_config:
            path, created = ConfigFile.init(args.config)
            print(("Created" if created else "Exists") + " config: " + path)
            return 0
        session = Session.from_config_file(path=args.config, yolo=args.yolo)
        return CommandLoop(Agent(session)).run()
    except ConfigError as error:
        print("ConfigError: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
