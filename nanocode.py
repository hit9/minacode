"""
nanocode
~~~~~~~~
A small terminal coding agent written in Python.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

from anthropic import Anthropic
from openai import OpenAI
from prompt_toolkit import print_formatted_text, search as pt_search
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
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

__version__ = "0.5.0"

Json = dict[str, Any]
HTTP_USER_AGENT = "nanocode/" + __version__
DEFAULT_MAX_CONTEXT_TOKENS = 128_000
MAX_TOOL_OUTPUT_TOKENS = 6_000
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
    yolo: bool = False
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False) -> "RuntimeSettings":
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            max_steps=max(1, Config.int(runtime, "max_agent_steps", Config.int(runtime, "max_steps", 200))),
            max_context_tokens=max(1, Config.int(runtime, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)),
            yolo=yolo or Config.bool(runtime, "yolo", False),
        )


@dataclass
class Config:
    active_provider: str = "default"
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {"default": ProviderConfig()})
    data_dir: str = "~/.nanocode"

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
        return cls(active_provider=active, providers=providers, data_dir=cls.str(paths, "data_dir", "~/.nanocode"))

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
        if isinstance(value, str) and value.lower() in {"on", "true", "yes", "1", "off", "false", "no", "0"}:
            return value.lower() in {"on", "true", "yes", "1"}
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
yolo = false
"""

    @classmethod
    def path(cls) -> str:
        return os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = os.path.expanduser(path or cls.path())
        if os.path.exists(config_path):
            return config_path, False
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(cls.DEFAULT_TEXT)
        return config_path, True

    @classmethod
    def load(cls, path: str | None = None) -> Json:
        config_path = os.path.expanduser(path or cls.path())
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
        cached_prompt_tokens = value(
            "prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens"
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cached_prompt_tokens += cached_prompt_tokens


@dataclass
class AgentState:
    goal: str = ""
    plan: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    summary: str = ""
    code_index_error: str = ""
    code_index_notice: str = ""
    code_index_refreshing: bool = False
    context_percent: int = 0
    turn_step: int = 0
    turn_tool_calls: int = 0
    debug_count: int = 0
    current_model_call_started_at: float = 0.0
    manual_model_retry_requested: bool = False
    model_retry_count: int = 0

    def apply(self, data: Json) -> None:
        for attr in ("goal", "summary"):
            if isinstance(data.get(attr), str):
                setattr(self, attr, str(data[attr]).strip())
        for attr in ("plan", "known"):
            value = data.get(attr)
            if isinstance(value, list):
                setattr(self, attr, [str(item).strip() for item in value if str(item).strip()])

    def format(self) -> str:
        plan = ["- " + item for item in self.plan] or ["- (empty)"]
        known = ["- " + item for item in self.known] or ["- (empty)"]
        return "\n".join(["Goal: " + (self.goal or "(empty)"), "Plan:", *plan, "Known:", *known])


@dataclass
class ToolResultRecord:
    key: str
    name: str
    args: list[Any]
    intention: str
    output: str

    def summary(self) -> str:
        text = f"- tool={self.name} key={self.key} args=[{', '.join(Tool.compact(arg, 80) for arg in self.args)}]"
        return text + (" why=" + self.intention if self.intention else "")


@dataclass
class ToolErrorRecord:
    key: str
    name: str
    args: list[Any]
    intention: str
    error: str

    def summary(self) -> str:
        text = f"- tool={self.name} key={self.key} args=[{', '.join(Tool.compact(arg, 80) for arg in self.args)}] error={Tool.compact(self.error, 200)}"
        return text + (" why=" + self.intention if self.intention else "")


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
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    messages: list[Json] = field(default_factory=list)
    state: AgentState = field(default_factory=AgentState)
    tool_results: dict[str, str] = field(default_factory=dict)
    tool_records: list[ToolResultRecord] = field(default_factory=list)
    tool_errors: list[ToolErrorRecord] = field(default_factory=list)
    tool_counter: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)

    def __post_init__(self) -> None:
        if self.system_info is None:
            self.system_info = SystemInfo.detect(self.cwd)

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
        root = root if os.path.isabs(root) else os.path.join(self.cwd, root)
        return os.path.abspath(os.path.join(root, *parts))

    def debug_dir(self) -> str:
        return self.data_path("sessions", self.session_id, "debug")

    def missing_config(self) -> list[str]:
        provider = self.config.provider
        return [key for key, value in (("provider.url", provider.url), ("provider.key", provider.key), ("provider.model", provider.model)) if not value]

    def store_tool_result(self, name: str, args: list[Any], intention: str, output: str) -> str:
        self.tool_counter += 1
        key = f"tr.{self.tool_counter}"
        self.tool_results[key] = output
        self.tool_records.append(ToolResultRecord(key, name, list(args), intention, output))
        if len(self.tool_results) > 400:
            old = self.tool_records.pop(0)
            self.tool_results.pop(old.key, None)
        return key

    def forget_tool_results(self, keys: list[str]) -> int:
        wanted = set(keys)
        count = sum(1 for key in wanted if key in self.tool_results)
        for key in wanted:
            self.tool_results.pop(key, None)
        self.tool_records = [record for record in self.tool_records if record.key not in wanted]
        return count

    def record_tool_error(self, key: str, name: str, args: list[Any], intention: str, error: str) -> None:
        self.tool_errors.append(ToolErrorRecord(key, name, list(args), intention, " ".join(error.split())))
        self.tool_errors = self.tool_errors[-5:]


class Tool:
    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    SIGNATURE: ClassVar[str] = ""
    EXAMPLE: ClassVar[tuple[str, ...]] = ()
    MUTATES: ClassVar[bool] = False
    STORES_RESULT: ClassVar[bool] = True

    def __init__(self, session: Session, args: list[Any]):
        self.session = session
        self.args = args

    @classmethod
    def schema(cls) -> Json:
        description = " ".join(part for part in (cls.DESCRIPTION, cls.SIGNATURE, *cls.EXAMPLE) if part)
        return {
            "type": "function",
            "function": {
                "name": cls.NAME,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intention": {"type": "string", "description": "Question being answered or concrete outcome needed."},
                        "args": cls.args_schema(),
                    },
                    "required": ["intention", "args"],
                    "additionalProperties": False,
                },
            },
        }

    @classmethod
    def arg_schema(cls) -> Json:
        return {"anyOf": [{"type": "object"}, {"type": "array"}, {"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]}

    @classmethod
    def args_schema(cls) -> Json:
        return {"type": "array", "items": cls.arg_schema()}

    def needs_confirmation(self) -> bool:
        return self.MUTATES

    def preview(self) -> str:
        return f"{self.NAME}({', '.join(self.display_args())})"

    def display_args(self) -> list[str]:
        return [self.compact(arg) for arg in self.args]

    def short_args(self) -> list[str]:
        return self.display_args()

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
        if stdout:
            lines.extend(["<stdout>", stdout.rstrip(), "</stdout>"])
        if stderr:
            lines.extend(["<stderr>", stderr.rstrip(), "</stderr>"])
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    @staticmethod
    def file_stat(path: str) -> str:
        stat = os.stat(path)
        return f'<file_stat mtime_ns="{stat.st_mtime_ns}" size="{stat.st_size}"/>'


class ReadTool(Tool):
    NAME = "Read"
    DESCRIPTION = "Read exact UTF-8 file ranges. Output includes line:hash anchors for safe Edit."
    SIGNATURE = "Read({path, ranges:[[start,end], ...]}[, ...]); end=0 means EOF"
    EXAMPLE = ('Example args: [{"path":"nanocode.py","ranges":[[0,80],[120,0]]}]',)

    @classmethod
    def arg_schema(cls) -> Json:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "ranges": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2},
                },
            },
            "required": ["path", "ranges"],
            "additionalProperties": False,
        }

    @staticmethod
    def line_hash(line: str) -> str:
        return hashlib.sha1(line.encode("utf-8")).hexdigest()[:6]

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
            unexpected = sorted(set(spec) - {"path", "ranges"})
            if unexpected:
                raise ToolError("Read unexpected field: " + ", ".join(unexpected))
            path = str(spec.get("path") or "").strip()
            raw_ranges = spec.get("ranges")
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
    DESCRIPTION = "Count lines in UTF-8 files; missing paths are reported."
    SIGNATURE = "LineCount(path[, path...])"
    EXAMPLE = ('Example args: ["nanocode.py", "pyproject.toml"]',)

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string"}

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
    DESCRIPTION = "List one directory, optionally filtered by a glob. Use for navigation, not source truth."
    SIGNATURE = "List([path][, glob])"
    EXAMPLE = ('Example args: ["."]', 'Example args: ["tests", "test_*.py"]')

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string"}

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
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "dir"
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    kind = "other"
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


class SearchTool(Tool):
    NAME = "Search"
    DESCRIPTION = "Search files with case-insensitive regex and optional context lines."
    SIGNATURE = "Search({pattern, path?, glob?, context?}[, ...])"
    EXAMPLE = (
        'Example args: [{"pattern":"class .*Tool","path":"nanocode.py"},{"pattern":"TODO","glob":"*.py","context":2}]',
        'Regex alternation. Args: [{"pattern":"done in|elapsed|duration","glob":"*.py","context":2}]',
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

    def needs_confirmation(self) -> bool:
        return any(not self.session.in_cwd(request["path"]) for request in self.requests())

    def call(self) -> str:
        sections = [self.search(request) for request in self.requests()]
        return "\n\n".join(sections)

    def short_args(self) -> list[str]:
        rows = []
        for request in self.requests():
            parts = [json.dumps(request["pattern"], ensure_ascii=False)]
            if self.session.relpath(str(request["path"])) != ".":
                parts.append("path=" + self.session.relpath(str(request["path"])))
            if request["glob"]:
                parts.append("glob=" + str(request["glob"]))
            if request["context"]:
                parts.append("C=" + str(request["context"]))
            rows.append(" ".join(parts))
        return ["; ".join(rows)]

    def requests(self) -> list[Json]:
        if not self.args:
            raise ToolError("Search requires at least one query object")
        requests = []
        for item in self.args:
            if not isinstance(item, dict):
                raise ToolError("Search args must be query objects")
            unexpected = sorted(set(item) - {"pattern", "path", "glob", "context"})
            if unexpected:
                raise ToolError("Search unexpected field: " + ", ".join(unexpected))
            pattern = str(item.get("pattern") or "").replace("\\n", "\n")
            if not pattern:
                raise ToolError("Search requires pattern")
            context = item.get("context", 0)
            if isinstance(context, bool) or not isinstance(context, int) or context < 0 or context > self.MAX_CONTEXT:
                raise ToolError(f"Search context must be 0..{self.MAX_CONTEXT}")
            requests.append(
                {
                    "pattern": pattern,
                    "path": self.session.resolve_path(str(item.get("path") or ".")),
                    "glob": str(item.get("glob") or ""),
                    "context": context,
                }
            )
        return requests

    def search(self, request: Json) -> str:
        rows = None if "\n" in str(request["pattern"]) else self.rg_matches(request)
        rows = rows if rows is not None else self.python_matches(request)
        header = f"<SearchToolResult pattern={json.dumps(request['pattern'])} matches={len(rows)}>"
        return "\n".join([header, *rows, "</SearchToolResult>"])

    def rg_matches(self, request: Json) -> list[str] | None:
        rg = shutil.which("rg")
        if not rg:
            return None
        cmd = [
            rg,
            "--json",
            "--line-number",
            "--with-filename",
            "--color=never",
            "--ignore-case",
            "--max-filesize",
            "2M",
        ]
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
        if os.path.isfile(root):
            return [root]
        found = []
        gitignore = self.gitignore_patterns(root)
        skip_dirs = {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skip_dirs
                and not name.startswith(".")
                and not name.startswith(".venv")
                and not self.ignored(os.path.join(dirpath, name), gitignore)
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
        if "\n" in regex.pattern:
            starts = [content.count("\n", 0, match.start()) for match in regex.finditer(content)]
        else:
            starts = [index for index, line in enumerate(lines) if regex.search(line)]
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

    def gitignore_patterns(self, root: str) -> list[str]:
        paths = [os.path.join(self.session.cwd, ".gitignore")]
        if os.path.isdir(root):
            paths.append(os.path.join(root, ".gitignore"))
        patterns = []
        for path in dict.fromkeys(paths):
            try:
                with open(path, encoding="utf-8") as file:
                    patterns.extend(line.strip() for line in file if line.strip() and not line.lstrip().startswith("#") and not line.startswith("!"))
            except OSError:
                pass
        return patterns

    def ignored(self, path: str, patterns: list[str]) -> bool:
        rel = self.session.relpath(path).replace(os.sep, "/")
        name = os.path.basename(path)
        for pattern in patterns:
            pattern = pattern.rstrip("/")
            if not pattern:
                continue
            if "/" in pattern and fnmatch.fnmatch(rel, pattern):
                return True
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                return True
        return False


class CodeIndex:
    AUTO_UPDATE_LIMIT: ClassVar[int] = 20

    def __init__(self, session: Session):
        self.session = session

    def db_dir(self) -> str:
        return os.path.join(self.session.cwd, ".code-symbol-index")

    def db_path(self) -> str:
        return os.path.join(self.db_dir(), "index.sqlite")

    def available(self) -> bool:
        status, message = self.status()
        self.session.state.code_index_error = message if status == "error" else ""
        return status in {"ready", "stale"}

    def notice(self, text: str = "", *, refreshing: bool = False) -> None:
        self.session.state.code_index_notice = text
        self.session.state.code_index_refreshing = refreshing

    def fail(self, error: Any) -> str:
        self.session.state.code_index_error = str(error).strip()
        self.notice("error")
        return self.session.state.code_index_error

    def finish(self, proc: subprocess.CompletedProcess[str]) -> bool:
        if proc.returncode != 0:
            self.fail((proc.stderr or proc.stdout).strip())
            return False
        self.session.state.code_index_error = ""
        self.notice("")
        return True

    def status(self, *, check: bool = False, max_pending_files: int = 20) -> tuple[str, str]:
        if shutil.which("code-symbol-index") is None:
            return "unavailable", "code-symbol-index not found"
        try:
            proc = self.run(["status", "--check" if check else "", "--max-pending-files", str(max_pending_files), "--json"], timeout=20)
        except Exception as error:
            return "error", str(error)
        if proc.returncode != 0:
            return "error", (proc.stderr or proc.stdout).strip()
        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as error:
            return "error", str(error)
        status = str(data.get("status") or "error")
        message = str(data.get("message") or data.get("reason") or "")
        pending = data.get("pending_changes")
        files = data.get("pending_files")
        if pending:
            sample = ", ".join(str(path) for path in (files or [])[:3])
            message = (message + "; " if message else "") + "pending " + str(pending) + ((" (" + sample + ")") if sample else "")
        return status, message

    def sync(self, *, force: bool = False) -> str:
        if shutil.which("code-symbol-index") is None:
            return "code_index: error\ncode-symbol-index not found"
        if self.session.state.code_index_refreshing:
            return "code_index: syncing"
        if force:
            shutil.rmtree(self.db_dir(), ignore_errors=True)
        self.notice("syncing", refreshing=True)
        try:
            proc = self.run(["index"], timeout=max(60, self.session.settings.shell_timeout))
        except Exception as error:
            return "code_index: error\n" + self.fail(error)
        if not self.finish(proc):
            return "code_index: error\n" + self.session.state.code_index_error
        status, message = self.status(check=True)
        lines = ["code_index: " + ("rebuilt" if force else "synced"), "status: " + status, "path: " + self.db_path()]
        if message:
            lines.append("note: " + message)
        return "\n".join(lines)

    def update(self, paths: list[str]) -> str:
        paths = self.update_paths(paths)
        if not paths or self.session.state.code_index_refreshing or not self.available():
            return ""
        self.notice("updating", refreshing=True)
        try:
            proc = self.run(["update", *paths], timeout=max(30, self.session.settings.shell_timeout))
        except Exception as error:
            return self.fail(error)
        if not self.finish(proc):
            return self.session.state.code_index_error
        return proc.stdout.strip()

    def update_pending(self) -> str:
        if self.session.state.code_index_refreshing:
            return ""
        status, _message = self.status(check=True, max_pending_files=self.AUTO_UPDATE_LIMIT + 1)
        if status != "stale":
            return ""
        try:
            proc = self.run(["status", "--check", "--max-pending-files", str(self.AUTO_UPDATE_LIMIT + 1), "--json"], timeout=20)
            data = json.loads(proc.stdout or "{}") if proc.returncode == 0 else {}
        except Exception:
            return ""
        pending = data.get("pending_changes")
        files = [str(path) for path in data.get("pending_files") or [] if path]
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

        def refresh() -> None:
            try:
                proc = self.run(["index"], timeout=max(60, self.session.settings.shell_timeout))
                self.finish(proc)
            except Exception as error:
                self.fail(error)

        threading.Thread(target=refresh, daemon=True).start()
        return True

    def update_paths(self, paths: list[str]) -> list[str]:
        paths = [self.session.resolve_path(path) for path in paths]
        return list(dict.fromkeys(path for path in paths if self.session.in_cwd(path) and os.path.isfile(path)))

    def run(self, args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        command, rest = args[0], [arg for arg in args[1:] if arg]
        cmd = ["code-symbol-index", command, "--root", self.session.cwd, *rest]
        return subprocess.run(cmd, cwd=self.session.cwd, text=True, capture_output=True, timeout=timeout)


class InspectCodeTool(Tool):
    NAME = "InspectCode"
    DESCRIPTION = "Find symbols, inspect one symbol, or outline one file using the code index."
    SIGNATURE = "InspectCode(mode, target[, options])"
    EXAMPLE = (
        'Find symbols, returns kind/file/range/signature. kind: class|function|variable|constant|enum|struct|dict_key; comma-ok. Args: ["find","Tool",{"kind":"class","limit":20}]',
        'Inspect one symbol, returns source anchors/members/references. Args: ["inspect","Tool",{"path":"nanocode.py"}]',
        'Outline one file, returns symbol tree and ranges. Args: ["outline","nanocode.py"]',
    )

    @classmethod
    def arg_schema(cls) -> Json:
        return {"description": "mode, target, optional options object"}

    @classmethod
    def args_schema(cls) -> Json:
        return {"type": "array", "items": cls.arg_schema(), "minItems": 2, "maxItems": 3}

    def call(self) -> str:
        if len(self.args) not in (2, 3):
            raise ToolError("InspectCode requires mode, target[, options]")
        if not isinstance(self.args[0], str) or not isinstance(self.args[1], str):
            raise ToolError("InspectCode mode and target must be strings")
        mode, target = self.args[0].lower(), self.args[1].strip()
        if len(self.args) == 3 and not isinstance(self.args[2], dict):
            raise ToolError("InspectCode options must be an object")
        options = self.args[2] if len(self.args) == 3 else {}
        unexpected = sorted(set(options) - {"limit", "kind", "path", "symbol", "exact_only"})
        if unexpected:
            raise ToolError("InspectCode unexpected option: " + ", ".join(unexpected))
        if mode not in {"find", "inspect", "outline"}:
            raise ToolError("InspectCode mode must be find, inspect, or outline")
        if not target:
            raise ToolError("InspectCode target is required")
        if mode in {"find", "inspect"} and re.search(r"\s", target):
            raise ToolError("InspectCode symbol target must not contain whitespace")
        if mode == "inspect" and (target.endswith(".py") or os.path.exists(self.session.resolve_path(target))):
            raise ToolError("InspectCode inspect target must be a symbol, not a file")
        if mode == "outline" and not os.path.isfile(self.session.resolve_path(target)):
            raise ToolError("InspectCode outline target must be an existing file")
        limit = options.get("limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 80):
            raise ToolError("InspectCode limit must be 1..80")
        index = CodeIndex(self.session)
        if not index.available():
            raise ToolError("code index is not available; run /index")
        cmd = ["search" if mode == "find" else mode, target]
        if mode == "inspect":
            cmd.append("--anchors")
        for key, flag in (("limit", "--limit"), ("kind", "--kind"), ("path", "--path"), ("symbol", "--symbol")):
            value = options.get(key)
            if value not in (None, "", False):
                cmd.extend([flag, str(value)])
        if options.get("exact_only"):
            cmd.append("--exact-only")
        proc = index.run(cmd, timeout=self.session.settings.shell_timeout)
        return self.process_result("InspectCodeToolResult", proc.returncode, proc.stdout, proc.stderr)


class CreateFileTool(Tool):
    NAME = "CreateFile"
    DESCRIPTION = "Create one new UTF-8 file; fails if it already exists."
    SIGNATURE = "CreateFile(path, content)"
    EXAMPLE = (
        'Create one file only; returns path/created/chars. Args: ["notes.txt","hello\\n"]',
        'Call separately for each file. Args: ["demo/main.cpp","int main() {}\\n"]',
    )
    MUTATES = True

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string"}

    @classmethod
    def args_schema(cls) -> Json:
        return {
            "type": "array",
            "description": 'Exactly ["path","content"].',
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        }

    def preview(self) -> str:
        path, content = self.payload()
        lines = content.splitlines(True)
        return "".join(difflib.unified_diff([], lines, fromfile="/dev/null", tofile=self.session.relpath(path))) or f"CreateFile({self.session.relpath(path)})"

    def call(self) -> str:
        path, content = self.payload()
        parent = os.path.dirname(path) or "."
        if os.path.exists(path):
            raise ToolError("file already exists")
        if not os.path.isdir(parent):
            if not self.session.in_cwd(parent):
                raise ToolError("refusing to create parent directories outside workspace")
            os.makedirs(parent, exist_ok=True)
        with open(path, "x", encoding="utf-8") as file:
            file.write(content)
        return f"<CreateFileToolResult path={json.dumps(self.session.relpath(path))} created=true chars={len(content)} />"

    def short_args(self) -> list[str]:
        path, _content = self.payload()
        return [self.session.relpath(path)]

    def payload(self) -> tuple[str, str]:
        if self.args and all(isinstance(arg, list) and len(arg) == 2 for arg in self.args):
            raise ToolError('CreateFile creates one file per call; call it separately for each file. Args must be ["path","content"].')
        if len(self.args) != 2:
            raise ToolError('CreateFile requires exactly ["path","content"]')
        if not isinstance(self.args[0], str):
            raise ToolError('CreateFile path must be a string; args must be ["path","content"]')
        path = self.session.resolve_path(str(self.args[0]))
        content = str(self.args[1])
        return path, content


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
    DESCRIPTION = "Patch an existing UTF-8 file with anchored edits or exact text replacement."
    SIGNATURE = "Edit(path, edits)"
    EXAMPLE = (
        'replace: ["code.py",[{"op":"replace","start":"10:abc123","end":"12:def456","content":"new text\\n"}]]',
        'delete: ["code.py",[{"op":"delete","start":"10:abc123","end":"12:def456"}]]',
        'insert_before: ["code.py",[{"op":"insert_before","start":"10:abc123","content":"new line\\n"}]]',
        'insert_after: ["code.py",[{"op":"insert_after","start":"10:abc123","content":"new line\\n"}]]',
        'replace_all: ["code.py",[{"op":"replace_all","old":"OldName","new":"NewName"}]]',
    )
    MUTATES = True

    @classmethod
    def arg_schema(cls) -> Json:
        return {"description": 'Exactly ["path", edits].'}

    @classmethod
    def args_schema(cls) -> Json:
        return {"type": "array", "items": cls.arg_schema(), "minItems": 2, "maxItems": 2}

    def call(self) -> str:
        path, edits = self.parse()
        with open(path, encoding="utf-8") as file:
            original = file.read()
        new_content, changes = self.apply(original, edits)
        if new_content == original:
            raise ToolError("edit produced no changes")
        with open(path, "w", encoding="utf-8") as file:
            file.write(new_content)
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(True), new_content.splitlines(True), fromfile=self.session.relpath(path), tofile=self.session.relpath(path)
            )
        )
        return "\n".join(
            [f"<Edit path={json.dumps(self.session.relpath(path))}>", self.file_stat(path), diff.rstrip(), self.edit_context(new_content, changes), "</Edit>"]
        )

    def preview(self) -> str:
        path, edits = self.parse()
        with open(path, encoding="utf-8") as file:
            original = file.read()
        new_content, _changes = self.apply(original, edits)
        if new_content == original:
            raise ToolError("edit produced no changes")
        return (
            "".join(
                difflib.unified_diff(
                    original.splitlines(True), new_content.splitlines(True), fromfile=self.session.relpath(path), tofile=self.session.relpath(path)
                )
            )
            or f"Edit({path})"
        )

    def short_args(self) -> list[str]:
        path, _edits = self.parse()
        return [self.session.relpath(path)]

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
            unexpected = sorted(set(item) - {"op", "start", "end", "content", "old", "new"})
            if unexpected:
                raise ToolError("Edit unexpected field: " + ", ".join(unexpected))
            op = str(item.get("op") or "")
            if op not in {"replace", "delete", "insert_before", "insert_after", "replace_all"}:
                raise ToolError("unknown edit op")
            if op in {"replace", "delete"} and (not item.get("start") or not item.get("end")):
                raise ToolError(f"{op} requires start and end anchors")
            if op in {"insert_before", "insert_after"} and not item.get("start"):
                raise ToolError(f"{op} requires start anchor")
            edits.append(
                Edit(
                    op=op,
                    start=str(item.get("start") or ""),
                    end=str(item.get("end") or ""),
                    content=self.normalize_text(str(item.get("content") or "")),
                    old=self.normalize_text(str(item.get("old") or "")),
                    new=self.normalize_text(str(item.get("new") or "")),
                )
            )
        return path, edits

    def apply(self, original: str, edits: list[Edit]) -> tuple[str, list[tuple[int, int, int, int]]]:
        if any(edit.op == "replace_all" for edit in edits):
            if any(edit.op != "replace_all" for edit in edits):
                raise ToolError("replace_all cannot be mixed with anchored edits")
            content = original
            for edit in edits:
                if not edit.old:
                    raise ToolError("replace_all requires old")
                if edit.old not in content:
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

    def resolve_anchor(self, lines: list[str], anchor: str) -> int:
        match = re.fullmatch(r"(\d+):([0-9a-fA-F]{6})", anchor.split("|", 1)[0].strip())
        if not match:
            raise ToolError('invalid anchor; use "line:hash" from Read or Search')
        index = int(match.group(1))
        if index >= len(lines):
            raise ToolError("anchor line out of range")
        expected = match.group(2).lower()
        actual = ReadTool.line_hash(lines[index])
        if actual != expected:
            raise ToolError(f"stale anchor {anchor}; current hash is {actual}")
        return index


class BashTool(Tool):
    NAME = "Bash"
    DESCRIPTION = "Run one bash command in the workspace."
    SIGNATURE = "Bash(command)"
    EXAMPLE = (
        'Check environment, returns exit/stdout/stderr. Args: ["python3 --version"]',
        'Run a project command. Args: ["python3 -m py_compile nanocode.py"]',
    )
    MUTATES = True
    live_output: Callable[[str, str], None] | None = None

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string"}

    @classmethod
    def args_schema(cls) -> Json:
        return {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "minItems": 1,
            "maxItems": 1,
        }

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
                [bash, "-lc", command],
                cwd=self.session.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
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
        return cls.pipe_text(stdout), cls.pipe_text(stderr)

    @staticmethod
    def pipe_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""


class GitTool(Tool):
    NAME = "Git"
    DESCRIPTION = "Run git with argv args; cwd=path may be the first arg."
    SIGNATURE = "Git([cwd=path,] git_arg...)"
    EXAMPLE = (
        'Read repo status, returns exit/stdout/stderr. Args: ["status","--short"]',
        'Diff inside a subdir. Args: ["cwd=src","diff","--","app.py"]',
    )
    READONLY = {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame"}

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string"}

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
        return args, cwd


class RecallTool(Tool):
    NAME = "Recall"
    DESCRIPTION = "Recall stored tool results by tr.N key, optionally sliced by output lines."
    SIGNATURE = "Recall(key...) or Recall({key|keys, ranges?})"
    EXAMPLE = (
        'Recall full result. Args: ["tr.1"]',
        'Recall output line ranges, 0-based end-exclusive. Args: [{"keys":["tr.1","tr.2"],"ranges":[[0,80]]}]',
    )
    STORES_RESULT = False

    @classmethod
    def arg_schema(cls) -> Json:
        range_schema = {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}
        return {
            "type": "object",
            "description": "Use key or keys, optionally with ranges.",
            "properties": {
                "key": {"type": "string", "pattern": "^tr\\.\\d+$"},
                "keys": {"type": "array", "items": {"type": "string", "pattern": "^tr\\.\\d+$"}, "minItems": 1},
                "ranges": {"type": "array", "items": range_schema, "minItems": 1},
            },
            "additionalProperties": False,
        }

    @classmethod
    def args_schema(cls) -> Json:
        return {"type": "array", "items": cls.arg_schema(), "minItems": 1}

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
        requests: list[tuple[str, tuple[tuple[int, int], ...]]] = []
        common_ranges: list[tuple[int, int]] = []
        for arg in self.args:
            if isinstance(arg, dict):
                unexpected = sorted(set(arg) - {"key", "result_key", "keys", "range", "ranges"})
                if unexpected:
                    raise ToolError("Recall unexpected field: " + ", ".join(unexpected))
                keys = [str(arg.get(name) or "").strip() for name in ("key", "result_key")]
                keys.extend(str(item).strip() for item in arg.get("keys", []) if str(item).strip())
                keys = [key for key in keys if key]
                if not keys:
                    raise ToolError("Recall object requires key, result_key, or keys")
                ranges = self.parse_ranges(arg)
                requests.extend((key, ranges) for key in keys)
            elif isinstance(arg, str) and re.fullmatch(r"\s*\d+\s*[-:,]\s*\d+\s*", arg):
                common_ranges.append(self.range_token(arg))
            else:
                key = str(arg).strip()
                if key:
                    requests.append((key, ()))
        if common_ranges:
            common = tuple(common_ranges)
            requests = [(key, ranges or common) for key, ranges in requests]
        return list(dict.fromkeys(requests))

    def parse_ranges(self, payload: Json) -> tuple[tuple[int, int], ...]:
        ranges = []
        if "range" in payload:
            ranges.append(self.parse_range(payload["range"]))
        if "ranges" in payload:
            raw = payload["ranges"]
            if not isinstance(raw, list) or not raw:
                raise ToolError("Recall ranges must be a non-empty array")
            ranges.extend(self.parse_range(item) for item in raw)
        return tuple(ranges)

    def parse_range(self, value: Any) -> tuple[int, int]:
        return self.range_token(value) if isinstance(value, str) else self.line_range(value, "Recall range")

    @staticmethod
    def range_token(value: str) -> tuple[int, int]:
        match = re.fullmatch(r"\s*(\d+)\s*[-:,]\s*(\d+)\s*", value)
        if not match:
            raise ToolError("range token must look like 0,120")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def slice(value: str, ranges: tuple[tuple[int, int], ...]) -> str:
        if not ranges:
            return value
        lines = value.splitlines()
        return "\n".join("\n".join(lines[start:end]) for start, end in ranges if end > start)


class ForgetTool(Tool):
    NAME = "Forget"
    DESCRIPTION = "Forget stored tool result keys that are no longer useful."
    SIGNATURE = "Forget(key...)"
    EXAMPLE = ('Args: ["tr.1","tr.2"]',)
    STORES_RESULT = False

    @classmethod
    def arg_schema(cls) -> Json:
        return {"type": "string", "pattern": "^tr\\.\\d+$"}

    @classmethod
    def args_schema(cls) -> Json:
        return {"type": "array", "items": cls.arg_schema(), "minItems": 1}

    def call(self) -> str:
        keys = list(dict.fromkeys(self.strings(min_count=1)))
        count = self.session.forget_tool_results(keys)
        return f"Forgot {count}/{len(keys)} tool results"


TOOLS: tuple[type[Tool], ...] = (
    ReadTool,
    LineCountTool,
    ListTool,
    InspectCodeTool,
    SearchTool,
    CreateFileTool,
    EditTool,
    BashTool,
    GitTool,
    RecallTool,
    ForgetTool,
)
TOOL_REGISTRY: dict[str, type[Tool]] = {tool.NAME: tool for tool in TOOLS}


@dataclass
class ToolCall:
    id: str
    name: str
    args: list[Any]
    intention: str = ""


class ContextManager:
    @dataclass
    class FileContextItem:
        order: int
        phase: int
        kind: str
        source: str
        path: str
        start: int
        end: int
        line: str
        mtime_ns: int
        size: int

    def __init__(self, session: Session):
        self.session = session
        self.latest_keys: list[str] = []

    def start_tool_batch(self) -> None:
        self.latest_keys = []

    def store_tool_result(self, call: ToolCall, output: str) -> str:
        key = self.session.store_tool_result(call.name, call.args, call.intention, output)
        self.latest_keys.append(key)
        return key

    def model_messages(self, base_system: str, user_input: str = "", extra_messages: list[Json] | None = None) -> list[Json]:
        return [{"role": "system", "content": base_system.strip()}, {"role": "user", "content": self.render(user_input, extra_messages)}]

    def maybe_compact(self, model: "ModelClient", base_system: str, user_input: str = "", extra_messages: list[Json] | None = None) -> None:
        if self.estimated_tokens(self.model_messages(base_system, user_input, extra_messages)) < self.session.settings.max_context_tokens:
            return
        try:
            self.session.state.apply(model.compact(self.compaction_input(extra_messages)))
            self.session.messages = self.session.messages[-6:]
            self.latest_keys = []
        except Exception:
            self.session.state.summary = (self.session.state.summary + "\nPrevious context was deterministically trimmed.").strip()
            self.session.messages = self.session.messages[-6:]
            self.latest_keys = []

    def render(self, user_input: str = "", extra_messages: list[Json] | None = None) -> str:
        sections = [
            ("Environment", self.environment()),
            ("State", self.session.state.format()),
            ("Summary", self.session.state.summary or "(empty)"),
            ("Recent Conversation", self.recent_conversation(extra_messages)),
            ("Tool Result Index", self.tool_index()),
            ("File Context", self.file_context()),
            ("Discovery Context", self.discovery_context()),
            ("Error Feedback", self.error_feedback()),
            ("Latest Tool Results", self.latest_results()),
            ("Current User Request", user_input.strip() or "(empty)"),
        ]
        return "\n\n".join(f"--- {name} ---\n{body or '(empty)'}" for name, body in sections)

    def environment(self) -> str:
        info = self.session.system_info
        return "\n".join(
            [
                "- cwd: " + info.cwd,
                "- os: " + info.os,
                "- arch: " + info.arch,
                "- shell_timeout: " + str(self.session.settings.shell_timeout) + "s",
                "- detected_commands: " + (", ".join(info.commands) or "(none)"),
            ]
        )

    def tool_index(self) -> str:
        return "\n".join(record.summary() for record in self.session.tool_records) or "(empty)"

    def error_feedback(self) -> str:
        if not self.session.tool_errors:
            return ""
        return "\n".join(["Recent failed tool calls:"] + [record.summary() for record in self.session.tool_errors])

    def latest_results(self) -> str:
        records = [record for record in self.session.tool_records if record.key in set(self.latest_keys)]
        return "\n\n".join(self.block(record) for record in records)

    def discovery_context(self) -> str:
        records = [record for record in self.session.tool_records if record.name in {"Search", "InspectCode"}]
        if not records:
            return ""
        lines = [
            "Source Policy:",
            "- Search and InspectCode are discovery leads, not current source truth.",
            "- Use Read before editing exact code.",
            "",
        ]
        for record in records:
            lines.extend(["Source: " + record.key + " tool=" + record.name, self.bound_output(record.output, record.key).rstrip(), ""])
        return "\n".join(lines).rstrip()

    def file_context(self) -> str:
        lines_by_path: dict[str, dict[int, tuple[str, str]]] = {}
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
                file_lines[item.start] = (item.source, item.line)
            else:
                omitted.setdefault(item.path, {}).setdefault(item.source, 0)
                omitted[item.path][item.source] += 1
        return self.bound_output(self.render_file_lines(lines_by_path, omitted))

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
                    items.append(self.FileContextItem(order, 0, "clear", record.key, path, int(match.group(1)), int(match.group(2)), "", *stat))
                for match in re.finditer(r"(?s)<content hashline-numbered>\n(.*?)\n</content>", body):
                    for line in match.group(1).splitlines():
                        line_match = re.match(r"(\d+):[0-9a-f]{6}\|", line)
                        if line_match:
                            items.append(self.FileContextItem(order, 1, "line", record.key, path, int(line_match.group(1)), 0, line, *stat))
        return items

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
        hash_match = re.match(r"\d+:([0-9a-f]{6})\|", item.line)
        return bool(lines is not None and hash_match and item.start in lines and ReadTool.line_hash(lines[item.start]) == hash_match.group(1))

    def render_file_lines(self, lines_by_path: dict[str, dict[int, tuple[str, str]]], omitted: dict[str, dict[str, int]]) -> str:
        chunks = [
            "Source Policy:",
            "- Built dynamically from active Read and Edit results.",
            "- Newer lines overwrite older lines; Edit invalidations clear stale ranges.",
            "- Lines are checked against the current file before being shown.",
            "",
        ]
        for path in sorted(lines_by_path):
            segments = self.segments(lines_by_path[path])
            if not segments:
                continue
            chunks.extend(["File: " + path, "Ranges:"])
            chunks.extend(f"- {start}:{end} source={source}" for start, end, source, _ in segments)
            chunks.append("Content:")
            for start, end, source, segment_lines in segments:
                chunks.append(f"@@ {start}:{end} source={source}")
                chunks.extend(segment_lines)
            chunks.append("")
        if omitted:
            chunks.append("Omitted stale content:")
            for path in sorted(omitted):
                chunks.extend(f"- {path} source={source} stale_lines={count}" for source, count in sorted(omitted[path].items()))
        return "\n".join(chunks).strip() if len(chunks) > 5 else ""

    def segments(self, numbered: dict[int, tuple[str, str]]) -> list[tuple[int, int, str, list[str]]]:
        items = sorted(numbered.items())
        if not items:
            return []
        segments = []
        start = previous = items[0][0]
        source, first = items[0][1]
        lines = [first]
        for number, (line_source, line) in items[1:]:
            if number == previous + 1 and line_source == source:
                previous = number
                lines.append(line)
                continue
            segments.append((start, previous + 1, source, lines))
            start = previous = number
            source = line_source
            lines = [line]
        segments.append((start, previous + 1, source, lines))
        return segments

    def recent_conversation(self, extra_messages: list[Json] | None = None) -> str:
        messages = [*self.session.messages, *(extra_messages or [])]
        return "\n\n".join(f"{message['role']}:\n{message.get('content') or ''}" for message in messages) or "(empty)"

    def compaction_input(self, extra_messages: list[Json] | None = None) -> str:
        return "\n\n".join(
            [
                "State:\n" + self.session.state.format(),
                "Summary:\n" + (self.session.state.summary or "(empty)"),
                "Conversation:\n" + self.recent_conversation(extra_messages),
                "Tool Result Index:\n" + self.tool_index(),
                "Latest Tool Results:\n" + self.latest_results(),
            ]
        )

    def block(self, record: ToolResultRecord) -> str:
        return "\n".join([record.summary(), "output:", self.bound_output(record.output, record.key).rstrip()])

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

    def output_path(self, output: str, tool_name: str) -> str:
        match = re.search(r"<" + re.escape(tool_name) + r'\s+path=(".*?")', output)
        if not match:
            return ""
        try:
            return str(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return ""

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


class ToolRunner:
    def __init__(self, session: Session, context: ContextManager, input_fn=input, output_fn=print):
        self.session = session
        self.context = context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.live_output: Callable[[str, str], None] | None = None

    def run(self, calls: list[ToolCall]) -> list[Json]:
        self.session.state.turn_tool_calls += len(calls)
        for call in calls:
            if self.run_one(call)[0] == "refused":
                break
        return []

    def run_one(self, call: ToolCall) -> tuple[str, str]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "failed", self.reject(call, f"ToolError: unknown tool {call.name}")
        tool = tool_class(self.session, call.args)
        if isinstance(tool, BashTool):
            tool.live_output = self.live_output
        started = time.monotonic()
        approved = False
        try:
            tool.short_args()
            needs_confirmation = tool.needs_confirmation()
            if needs_confirmation and self.session.settings.yolo:
                self.output_fn(self.approval_display(call, tool, "auto"))
            elif needs_confirmation:
                confirmed, reason = self.confirm(call, tool)
                if not confirmed:
                    output = "Cancelled: user refused tool call" + ((": " + reason) if reason else "")
                    self.finish(call, output, failed=True, elapsed=time.monotonic() - started)
                    return "refused", output
                approved = True
            output = tool.call()
        except ToolError as error:
            return "failed", self.reject(call, f"ToolError: {error}", elapsed=time.monotonic() - started)
        except Exception as error:
            output = f"ToolError: {error}"
            return "failed", self.finish(call, output, failed=True, elapsed=time.monotonic() - started)
        return "ok", self.finish(call, output, elapsed=time.monotonic() - started, approved=approved)

    def reject(self, call: ToolCall, output: str, *, elapsed: float | None = None) -> str:
        if self.session.settings.debug:
            return self.finish(call, output, failed=True, elapsed=elapsed)
        self.session.record_tool_error("-", call.name, call.args, call.intention, output)
        return output

    def finish(self, call: ToolCall, output: str, *, failed: bool = False, elapsed: float | None = None, approved: bool = False) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        key = self.context.store_tool_result(call, output) if tool_class is None or tool_class.STORES_RESULT else ""
        if failed:
            self.session.record_tool_error(key or "-", call.name, call.args, call.intention, output)
        elif key:
            self.update_code_index(call, output)
        self.output_fn(self.finish_display(call, key, output, failed=failed, approved=approved))
        return f"result_key: {key}\n{self.context.bound_output(output, key)}" if key else output

    def update_code_index(self, call: ToolCall, output: str) -> None:
        if call.name not in {"CreateFile", "Edit"}:
            return
        paths = []
        for match in re.finditer(r'<(?:CreateFileToolResult|Edit)\s+path=(".*?")', output):
            try:
                paths.append(str(json.loads(match.group(1))))
            except json.JSONDecodeError:
                pass
        CodeIndex(self.session).update(paths)

    def confirm(self, call: ToolCall, tool: Tool) -> tuple[bool, str]:
        self.output_fn(self.approval_display(call, tool, "confirm"))
        while True:
            answer = self.input_fn("Approve " + tool.NAME + "? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                return True, ""
            if answer in {"n", "no"}:
                return False, self.input_fn("Reason? [optional] ").strip()
            self.output_fn("Please answer y or n.")

    def approval_display(self, call: ToolCall, tool: Tool, status: str) -> str:
        header = ("approve " if status == "confirm" else "auto ") + self.short_call(call)
        if tool.NAME not in {"Edit", "CreateFile"}:
            return header
        preview = self.preview_block(tool.preview())
        return header + (("\n" + preview) if preview else "")

    def preview_block(self, preview: str, *, max_lines: int = 80) -> str:
        lines = preview.rstrip().splitlines()
        if not lines:
            return ""
        lines = lines[:max_lines] + (["... preview truncated ..."] if len(lines) > max_lines else [])
        return "\n".join(["  preview"] + ["  " + line for line in lines])

    def finish_display(self, call: ToolCall, key: str, output: str, *, failed: bool, approved: bool = False) -> str:
        tag = " [refused]" if failed and "user refused" in output else " [failed]" if failed else " [approved]" if approved else ""
        line = "tool " + self.short_call(call) + ((" -> " + key) if key else "") + tag
        lines = [line]
        if failed:
            lines.append("  error " + self.oneline(output, 220))
        return "\n".join(lines)

    def short_call(self, call: ToolCall) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        try:
            args = tool_class(self.session, call.args).short_args() if tool_class is not None else [Tool.compact(arg) for arg in call.args]
        except Exception:
            args = [Tool.compact(arg) for arg in call.args]
        text = " ".join([call.name, *args]).strip()
        return self.oneline(text, 200)

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
        session.state.debug_count += 1
        directory = session.debug_dir()
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_activity = re.sub(r"[^A-Za-z0-9_.-]+", "-", activity or "debug")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label or "event")
        path = os.path.join(directory, f"{timestamp}-{session.state.debug_count:04d}-{safe_activity}-{safe_label}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(cls.value(payload), file, ensure_ascii=False, indent=2)
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
            return value if len(value) <= cls.STRING_LIMIT else value[: cls.STRING_LIMIT] + "...<truncated>"
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return str(value)

    @classmethod
    def prompt(cls, session: Session, *, activity: str, messages: list[Json]) -> None:
        cls.write(session, activity=activity, label="prompt", payload={"messages": messages})

    @classmethod
    def model_request(cls, session: Session, *, activity: str, api: str, model: str, params: Json, tools: list[Json] | None) -> None:
        payload = {"api": api, "model": model, "tool_names": cls.tool_names(tools), "param_keys": sorted(params), "params": cls.filtered_params(params)}
        cls.write(session, activity=activity, label="model-request", payload=payload)

    @classmethod
    def model_response(cls, session: Session, *, activity: str, api: str, model: str, raw: Any, text: str, tool_names: list[str]) -> None:
        cls.write(
            session,
            activity=activity,
            label="model-response",
            payload={"api": api, "model": model, "assistant_text_len": len(text), "tool_names": tool_names, "raw": raw},
        )

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
        self.session.state.current_model_call_started_at = time.monotonic()
        try:
            if provider.resolved_api() == "anthropic":
                return self.anthropic_request(messages, tools)
            return self.chat_request(messages, tools)
        except KeyboardInterrupt:
            if self.session.state.manual_model_retry_requested:
                self.session.state.manual_model_retry_requested = False
                raise ModelRequestRetry() from None
            raise
        finally:
            self.session.state.current_model_call_started_at = 0.0

    def chat_request(self, messages: list[Json], tools: list[Json] | None = None, *, activity: str = "agent") -> tuple[Json, list[ToolCall], str]:
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
Return exactly one JSON object with keys: summary, goal, plan, known.
Keep only durable facts needed to continue; preserve file paths, symbols, constraints, and tr.N keys.
""".strip()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": context}]
        _, _, content = (
            self.anthropic_request(messages, None, activity="compact")
            if self.session.config.provider.resolved_api() == "anthropic"
            else self.chat_request(messages, None, activity="compact")
        )
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ModelError("compactor returned non-object JSON")
        return data

    def client(self) -> OpenAI:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        return OpenAI(
            api_key=provider.key,
            base_url=provider.base_url(),
            timeout=provider.timeout,
            max_retries=0,
            default_headers={"User-Agent": HTTP_USER_AGENT},
        )

    def anthropic_client(self) -> Anthropic:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        return Anthropic(
            api_key=provider.key,
            base_url=self.anthropic_base_url(provider),
            timeout=provider.timeout,
            max_retries=0,
            default_headers={"User-Agent": HTTP_USER_AGENT},
        )

    @staticmethod
    def anthropic_base_url(provider: ProviderConfig) -> str:
        url = provider.base_url().rstrip("/")
        return url[: -len("/v1")] if url.endswith("/v1") else url

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
            "tools": self.tool_schema_names(tools),
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return "nanocode-" + digest[:24]

    @staticmethod
    def tool_schema_names(tools: list[Json] | None) -> str:
        names = [
            name
            for schema in tools or []
            if (
                name := str(
                    (schema.get("function") if isinstance(schema.get("function"), dict) else {}).get("name") or schema.get("name") or schema.get("type") or ""
                )
            )
        ]
        return ",".join(sorted(names)) or "(none)"

    def anthropic_request(self, messages: list[Json], tools: list[Json] | None, *, activity: str = "agent") -> tuple[Json, list[ToolCall], str]:
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
            "system": self.anthropic_system(messages),
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

    @staticmethod
    def anthropic_system(messages: list[Json]) -> str:
        return "\n\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system").strip()

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
                args, intention = self.tool_payload(payload)
                calls.append(ToolCall(id=call_id, name=name, args=args, intention=intention))
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
            except json.JSONDecodeError as error:
                calls.append(ToolCall(id=raw.id, name=raw.function.name, args=[], intention=f"invalid JSON arguments: {error}"))
                continue
            args, intention = self.tool_payload(payload)
            calls.append(ToolCall(id=raw.id, name=raw.function.name, args=args, intention=intention))
        return calls

    @staticmethod
    def tool_payload(payload: Any) -> tuple[list[Any], str]:
        if isinstance(payload, dict):
            args = payload.get("args")
            return (args if isinstance(args, list) else [payload], str(payload.get("intention") or ""))
        return [payload], ""


class Agent:
    SYSTEM_PROMPT = """You are nanocode, a concise terminal coding agent.
Tools: Read LineCount List InspectCode Search CreateFile Edit Bash Git Recall Forget. Call as {"intention":"why","args":[...]}.
Trust File Context; Discovery is leads. Recall tr.N when needed. Forget stale tr.N results. Inspect/read before edits. Keep changes small; never overwrite user work.
For multi-step tasks, use concise plan/known as working memory.
Output: concise markdown, USER'S LANGUAGE.
"""

    def __init__(self, session: Session, input_fn=input, output_fn=print):
        self.session = session
        self.context = ContextManager(session)
        self.model = ModelClient(session)
        self.tools = ToolRunner(session, self.context, input_fn=input_fn, output_fn=output_fn)
        self.output_fn = output_fn

    def run(self, user_input: str) -> str:
        self.session.state.goal = user_input.strip()
        self.session.state.turn_step = 0
        self.session.state.turn_tool_calls = 0
        self.context.start_tool_batch()
        user_message = {"role": "user", "content": user_input}
        turn_messages: list[Json] = []
        for step in range(self.session.settings.max_steps):
            self.session.state.turn_step = step + 1
            while True:
                try:
                    _assistant, tool_calls, content = self.model.request(self.messages(user_input, turn_messages))
                    break
                except ModelRequestRetry:
                    continue
            if not tool_calls:
                answer = content.strip() or "(empty response)"
                self.session.messages.extend([user_message, *turn_messages, {"role": "assistant", "content": answer}])
                return answer
            if content.strip():
                message = {"role": "assistant", "content": content.strip()}
                turn_messages.append(message)
                self.output_fn(message["content"])
            self.tools.run(tool_calls)
        self.session.messages.extend(
            [user_message, *turn_messages, {"role": "assistant", "content": f"Stopped after max_agent_steps={self.session.settings.max_steps}"}]
        )
        return f"Stopped after max_agent_steps={self.session.settings.max_steps}"

    def messages(self, user_input: str, turn_messages: list[Json] | None = None) -> list[Json]:
        self.context.maybe_compact(self.model, self.SYSTEM_PROMPT, user_input, turn_messages)
        messages = self.context.model_messages(self.SYSTEM_PROMPT, user_input, turn_messages)
        tokens = ContextManager.estimated_tokens(messages)
        self.session.state.context_percent = min(100, round(tokens * 100 / self.session.settings.max_context_tokens))
        return messages


class CommandCompleter(Completer):
    COMMANDS = ("/help", "/status", "/config", "/api", "/debug", "/compact", "/index", "/model", "/provider", "/reason", "/set", "/yolo", "/exit", "/quit")
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
    )
    SET_VALUES = {
        "provider.api": PROVIDER_API_CHOICES,
        "provider.prompt_cache_key": ("auto", "off"),
        "provider.reasoning": REASONING_CHOICES,
        "provider.chat_reasoning": CHAT_REASONING_CHOICES,
        "provider.temperature": ("off",),
        "runtime.yolo": ("on", "off", "true", "false"),
    }

    def __init__(
        self,
        providers: Callable[[], tuple[str, ...]] = tuple,
        models: Callable[[], tuple[str, ...]] = tuple,
    ):
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
        if text.startswith("/") and " " not in text:
            yield from self.matches(self.COMMANDS, text)

    @staticmethod
    def matches(values, prefix: str):
        yield from (Completion(value, start_position=-len(prefix)) for value in values if value.startswith(prefix))


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
        if text.startswith("[done in "):
            return [("ansibrightblack", text + "\n")]
        if text.startswith("nanocode "):
            return [("ansicyan", text + "\n")]
        if text.startswith("Error:") or text.startswith("ConfigError:") or text.startswith("Unknown command:"):
            return [("ansired", text + "\n")]
        return self.text_segments(text)

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

    def text_segments(self, text: str) -> list[tuple[str, str]]:
        return [("ansiwhite", line + "\n") for line in text.splitlines() or [""]]


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
        body = self.text.replace("\r", "\n").splitlines()[-self.HEIGHT :]
        return ["  output"] + ["  " + self.fit(line, width - 2) for line in body] if body else []

    @staticmethod
    def fit(text: str, width: int) -> str:
        text = text.expandtabs(4)
        return text if len(text) <= width else text[: max(0, width - 3)] + "..."


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
        if reset or not self.started_at:
            self.started_at = time.monotonic()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

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
            self.refresh_retry_state()
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(self.fragments(self.elapsed(), sweep=True, show_elapsed=True)), output=self.output, end="", flush=True)
            self.rendered = True
            self.stop_event.wait(self.INTERVAL)

    def refresh_retry_state(self) -> None:
        count = self.session.state.model_retry_count
        if count == self.seen_retry_count:
            return
        self.seen_retry_count = count
        now = time.monotonic()
        self.retry_notice_until = now + 2.0

    def model_elapsed(self) -> float:
        started = self.session.state.current_model_call_started_at
        return max(0.0, time.monotonic() - started) if started > 0 else 0.0

    def clear(self) -> None:
        if self.rendered:
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            self.output.flush()
            self.rendered = False

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started_at) if self.started_at else 0.0

    def idle_fragments(self) -> list[tuple[str, str]]:
        return self.fragments(0.0, sweep=False, show_elapsed=False)

    def fragments(self, elapsed: float, *, sweep: bool, show_elapsed: bool) -> list[tuple[str, str]]:
        text = self.text(elapsed, show_elapsed=show_elapsed)
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(text) >= columns:
            text = text[: max(0, columns - 4)] + "..."
        return self.sweep_fragments(text, elapsed) if sweep else [("ansicyan", text)]

    def text(self, elapsed: float, *, show_elapsed: bool) -> str:
        provider = self.session.config.provider
        model = provider.model.rsplit("/", 1)[-1] or "(no model)"
        reason = provider.reasoning
        if self.session.settings.debug:
            reason += "/" + provider.resolved_chat_reasoning()
        parts = [self.session.config.active_provider + "/" + model, reason]
        if self.session.settings.debug:
            parts.append("api " + provider.resolved_api())
        parts.append("ctx " + str(self.session.state.context_percent) + "%")
        if self.session.settings.debug and self.session.usage.cached_prompt_tokens:
            parts.append("cache " + str(self.session.usage.cached_prompt_tokens))
        if self.session.settings.debug:
            if self.session.state.code_index_error:
                parts.append("idx error")
            elif self.session.state.code_index_refreshing:
                parts.append("idx " + (self.session.state.code_index_notice or "syncing"))
        if self.session.settings.yolo:
            parts.append("yolo")
        if show_elapsed:
            parts.extend(
                ["step " + str(self.session.state.turn_step) + "/" + str(self.session.settings.max_steps), "tools " + str(self.session.state.turn_tool_calls)]
            )
        if self.session.settings.debug:
            parts.append("dbg " + str(self.session.state.debug_count))
        if show_elapsed:
            parts.append(self.duration(elapsed))
            if self.retry_notice_until > time.monotonic():
                parts.append("retrying")
            elif self.model_elapsed() >= self.stress_after():
                parts.append("ctrl-g retry")
        return " | ".join(parts)

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

    def stress_after(self) -> float:
        return max(30.0, self.session.config.provider.timeout * 0.5)

    @staticmethod
    def duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rest = divmod(int(seconds), 60)
        return f"{minutes}m{rest:02d}s"


class CommandLoop:
    MODEL_CONFIGURED_LABEL = "---- Configured models ----"
    MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
    MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))

    HELP = """Commands:
  /help              Show this help.
  /status            Show runtime status.
  /config            Show active config.
  /api [NAME]        Show or set provider API format: auto, chat, anthropic.
  /debug [on|off]    Toggle model I/O debug traces.
  /compact           Compact context now.
  /index [force]      Sync or rebuild code symbol index.
  /provider [NAME]   Select or show the active provider.
  /model [MODEL]     Select or set the active model.
  /reason            Select reasoning effort.
  /set KEY VALUE     Set provider.*, runtime.yolo, runtime.max_agent_steps, runtime.max_context_tokens, runtime.shell_timeout.
  /yolo              Toggle tool confirmations.
  /exit, /quit       Exit.
Tools:
  Read, LineCount, List, InspectCode, Search, CreateFile, Edit, Bash, Git, Recall, Forget.
"""

    def __init__(self, agent: Agent, input_fn=input, output_fn=print):
        self.agent = agent
        self.session = agent.session
        self.input_fn = input_fn
        self.ui = UiPrinter(output_fn)
        self.status_bar = StatusBar(self.session)
        self.live_preview = BashLivePreview()
        self.live_status_paused = False
        self.transient_tool_lines = 0
        self.interactive_input = input_fn is input and sys.stdin.isatty()
        self.input_history = self.make_input_history() if self.interactive_input else None
        self.input_completer = self.make_completer()
        self.agent.output_fn = self.agent_output
        self.agent.tools.output_fn = self.tool_output
        self.agent.tools.input_fn = self.tool_input
        self.agent.tools.live_output = self.tool_live_output

    def run(self) -> int:
        self.emit(f"nanocode {__version__}. /help for commands.")
        CodeIndex(self.session).refresh_existing_async()
        while True:
            try:
                user_input = self.read_input()
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
            try:
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
                self.session.state.manual_model_retry_requested = False
                CodeIndex(self.session).update_pending()
                self.status_bar.stop()
            elapsed = time.monotonic() - started
            self.ui.emit_answer(answer)
            m, s = divmod(elapsed, 60)
            self.emit(f"[done in {int(m)}m{s:.0f}s]")

    def style(self) -> Style:
        return Style.from_dict(
            {
                "prompt": "ansicyan bold",
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

    def make_input_history(self):
        history_path = self.session.data_path("history.txt")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        return FileHistory(history_path)

    def make_completer(self) -> CommandCompleter:
        return CommandCompleter(providers=lambda: tuple(sorted(self.session.config.providers)), models=lambda: self.session.config.provider.available_models)

    def status_window(self) -> Window:
        return Window(
            FormattedTextControl(self.status_bar.idle_fragments, style="class:bottom-toolbar.text"),
            style="class:bottom-toolbar",
            height=1,
            dont_extend_height=True,
        )

    def read_input(self, prompt_text: str = "nano> ") -> str:
        if self.input_history is None:
            return self.input_fn(prompt_text)
        prompt = FormattedText([("class:prompt", prompt_text)])

        def accept(buffer: Buffer) -> bool:
            app.exit(result=buffer.text)
            return True

        buffer = Buffer(
            history=self.input_history,
            completer=self.input_completer,
            complete_while_typing=False,
            enable_history_search=True,
            multiline=False,
            accept_handler=accept,
        )
        search_toolbar = SearchToolbar()
        control = BufferControl(
            buffer=buffer,
            input_processors=[HighlightIncrementalSearchProcessor(), BeforeInput(prompt)],
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
            if buffer.text:
                buffer.delete()
            else:
                event.app.exit(exception=EOFError())

        @bindings.add("c-r", eager=True)
        def _ctrl_r(event):
            direction = pt_search.SearchDirection.BACKWARD
            if event.app.layout.current_control is search_toolbar.control:
                pt_search.do_incremental_search(direction, count=event.arg)
            else:
                pt_search.start_search(direction=direction)

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
        with patch_stdout():
            text = app.run()
        print_formatted_text(FormattedText([("class:prompt", prompt_text), ("", text)]), style=self.style())
        return text

    def emit(self, text: str = "") -> None:
        self.ui.emit(str(text))

    def with_status_paused(self, action):
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.stop()
        try:
            return action()
        finally:
            if was_running:
                self.status_bar.start(reset=False)

    def tool_output(self, text: str = "") -> None:
        if text.startswith("approve ") and self.interactive_input and sys.stdout.isatty():
            self.with_status_paused(lambda: self.show_transient_tool_output(text))
            return
        self.with_status_paused(lambda: self.emit_tool_output(text))

    def agent_output(self, text: str = "") -> None:
        self.with_status_paused(lambda: self.emit(text))

    def tool_input(self, prompt: str = "") -> str:
        def read() -> str:
            try:
                return self.input_fn(prompt)
            finally:
                if self.interactive_input and sys.stdout.isatty():
                    sys.stdout.write("\x1b[1A\r\x1b[2K")
                    sys.stdout.flush()
                    self.clear_transient_tool_output()

        return self.with_status_paused(read)

    def show_transient_tool_output(self, text: str) -> None:
        self.clear_transient_tool_output()
        self.emit(text)
        self.transient_tool_lines = len(text.splitlines() or [""])

    def emit_tool_output(self, text: str) -> None:
        self.clear_transient_tool_output()
        self.emit(text)

    def clear_transient_tool_output(self) -> None:
        if not self.transient_tool_lines:
            return
        for _ in range(self.transient_tool_lines):
            sys.stdout.write("\x1b[1A\r\x1b[2K")
        sys.stdout.flush()
        self.transient_tool_lines = 0

    def tool_live_output(self, _stream: str, text: str) -> None:
        if not self.ui.color:
            return
        if text:
            if not self.live_preview.active:
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

    def command(self, text: str) -> tuple[bool, bool]:
        if text in {"/exit", "/quit", "exit", "quit"}:
            return True, True
        if not text.startswith("/"):
            return False, False
        name, _, args = text.partition(" ")
        handlers = {
            "/help": self.help,
            "/status": self.status,
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
        }
        handler = handlers.get(name)
        self.emit(handler(args.strip()) if handler else f"Unknown command: {name}")
        return True, False

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
        if not choices:
            return None
        if self.interactive_input:
            try:
                return self.choice_application(title, choices, labels, current, set(disabled))
            except (EOFError, KeyboardInterrupt):
                self.emit("Cancelled")
                return None
        return self.choice_prompt(title, choices, labels, current, set(disabled))

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
            if not options:
                state["selected"] = 0
            else:
                state["selected"] = min(max(int(state["selected"]), 0), len(options) - 1)

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
            layout=Layout(
                HSplit(
                    [
                        choice_window,
                        self.status_window(),
                    ]
                ),
                focused_element=choice_window,
            ),
            key_bindings=bindings,
            full_screen=False,
            style=self.style(),
            refresh_interval=StatusBar.INTERVAL,
            erase_when_done=True,
        )
        return app.run()

    def choice_prompt(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str],
        current: str,
        disabled: set[str],
    ) -> str | None:
        query = ""
        while True:
            visible = self.visible_choices(choices, labels, disabled, query)
            lines = [title + (" /" + query if query else "")]
            enabled: list[str] = []
            for choice in visible:
                label = labels.get(choice, choice)
                if choice in disabled:
                    lines.append("  " + label)
                    continue
                enabled.append(choice)
                mark = " *" if choice == current else ""
                lines.append(f"  {len(enabled)}. {label}{mark}")
            if not enabled:
                lines.append("  no matches")
            self.emit("\n".join(lines))
            try:
                answer = self.read_input("select> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not answer:
                return None
            if answer.startswith("/"):
                query = answer[1:].strip()
                continue
            if answer.isdigit() and 1 <= int(answer) <= len(enabled):
                return enabled[int(answer) - 1]
            for choice in enabled:
                if answer == choice or answer.lower() == labels.get(choice, choice).lower():
                    return choice
            self.emit("No match: " + answer)

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
        index_status, index_message = CodeIndex(self.session).status(check=True)
        if self.session.state.code_index_refreshing:
            index_status, index_message = "syncing", self.session.state.code_index_notice
        elif self.session.state.code_index_error:
            index_status, index_message = "error", self.session.state.code_index_error
        if index_status in {"missing", "unavailable", "error"} and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index"
        elif index_status == "stale" and "run /index" not in index_message:
            index_message = (index_message + "; " if index_message else "") + "run /index or wait for auto update"
        return "\n".join(
            [
                f"cwd: {self.session.cwd}",
                f"provider: {self.session.config.active_provider}",
                f"model: {provider.model or '(empty)'}",
                f"api: {provider.resolved_api()} ({provider.api})",
                f"reasoning: {provider.reasoning} ({provider.resolved_chat_reasoning()})",
                f"messages: {len(self.session.messages)}",
                f"tool_results: {len(self.session.tool_results)}",
                f"goal: {self.session.state.goal or '(empty)'}",
                f"known: {len(self.session.state.known)}",
                f"tokens: calls={usage.calls} total={usage.total_tokens} cached={usage.cached_prompt_tokens}",
                f"runtime: yolo={'on' if self.session.settings.yolo else 'off'} debug={'on' if self.session.settings.debug else 'off'} max_steps={self.session.settings.max_steps}",
                f"code_index: {index_status}" + ((": " + index_message) if index_message else ""),
            ]
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
        try:
            self.status_bar.start()
            data = self.agent.model.compact(self.agent.context.compaction_input())
        except KeyboardInterrupt:
            return "Cancelled"
        except Exception as error:
            return "Error: " + str(error)
        finally:
            self.status_bar.stop()
        self.session.state.apply(data)
        self.session.messages = self.session.messages[-6:]
        self.agent.context.latest_keys = []
        messages = self.agent.context.model_messages(self.agent.SYSTEM_PROMPT, "")
        tokens = ContextManager.estimated_tokens(messages)
        self.session.state.context_percent = min(100, round(tokens * 100 / self.session.settings.max_context_tokens))
        return (
            "Compacted context: messages " + str(before) + " -> " + str(len(self.session.messages)) + ", ctx " + str(self.session.state.context_percent) + "%"
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
