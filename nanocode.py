"""
nanocode
~~~~~~~~
A lightweight terminal-based AI coding assistant
https://github.com/hit9/nanocode
Install: uv tool install nanocode-cli
"""

import argparse
import difflib
import fcntl
import fnmatch
import hashlib
import itertools
import json
import os
import platform
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, ClassVar, Iterator, Iterable, Self, Type, TypeAlias

import json_repair
from prompt_toolkit.application import Application
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

__version__ = "0.3.29"
HTTP_USER_AGENT = "nanocode/" + __version__


JsonValue: TypeAlias = Any
Json: TypeAlias = dict[str, JsonValue]
############################
# Errors
############################


class Error(Exception): ...


class ToolCallError(Error): ...


class ToolCallArgError(ToolCallError): ...


class LLMError(Error): ...


class ConfigError(Error): ...


class ModelRequestTimeout(Error): ...


class Cancellation(Error): ...


############################
# Conversation (dataclasses)
############################


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationItem:
    role: Role
    time: datetime = field(default_factory=datetime.now)

    def format_ts(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M:%S")

    def format_transcript(self, title: str, content: str, indent: str = "") -> str:
        quoted = ["> " + line if line else ">" for line in content.splitlines()]
        if not quoted:
            quoted = [">"]
        return _format_lines([f"#### {title} {self.format_ts()}", *quoted], indent)


@dataclass
class UserMessage(ConversationItem):
    role: Role = Role.USER
    content: str = ""

    def format(self, indent: str = "") -> str:
        return self.format_transcript("User", self.content, indent)


@dataclass
class AssistantMessage(ConversationItem):
    role: Role = Role.ASSISTANT
    content: str = ""

    def format(self, indent: str = "") -> str:
        return self.format_transcript("Assistant", self.content, indent)


############################
# State Dataclasses
############################


class PlanStatus(StrEnum):
    TODO = "todo"
    DOING = "doing"
    DONE = "done"
    BLOCKED = "blocked"

    def __str__(self) -> str:
        symbols = {
            PlanStatus.TODO: "○",
            PlanStatus.DOING: "◔",
            PlanStatus.DONE: "✓",
            PlanStatus.BLOCKED: "☒",
        }
        return f"{symbols.get(self, '')} {self.value}".strip()


ALL_PLAN_STATUSES = frozenset(PlanStatus)


class TaskCode(StrEnum):
    NEW = "new"
    WORKING = "working"
    VERIFYING = "verifying"
    DONE = "done"


class WorkMode(StrEnum):
    NORMAL = "normal"
    INVESTIGATE = "investigate"


ALL_WORK_MODES = frozenset(WorkMode)


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    RULED_OUT = "ruled_out"
    DROPPED = "dropped"
    CONFIRMED = "confirmed"


ALL_HYPOTHESIS_STATUSES = frozenset(HypothesisStatus)
HYPOTHESIS_STATUS_SCHEMA = "|".join(status.value for status in HypothesisStatus)
HYPOTHESIS_STATUS_TEXT = ", ".join(status.value for status in HypothesisStatus)


@dataclass
class PlanItem:
    text: str
    status: PlanStatus = PlanStatus.TODO
    id: str = ""
    context: str = ""

    def format(self, indent: str = "") -> str:
        text = "- [" + str(self.status) + "] " + self.text
        if self.id:
            text += " (id=" + self.id + ")"
        lines = [text]
        if self.context:
            lines.append("  context: " + self.context)
        return _format_lines(lines, indent)


@dataclass(eq=False)
class KnownItem:
    text: str
    source: tuple[str, ...] = ()

    def format(self, indent: str = "") -> str:
        source = "[" + ", ".join(self.source) + "] " if self.source else ""
        return indent + source + self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KnownItem):
            return self.text == other.text and self.source == other.source
        if isinstance(other, str):
            return self.text == other
        return False

    @staticmethod
    def text_of(item: "KnownItem | str") -> str:
        return item.text if isinstance(item, KnownItem) else str(item)

    @staticmethod
    def source_of(item: "KnownItem | str") -> tuple[str, ...]:
        return item.source if isinstance(item, KnownItem) else ()

    @staticmethod
    def format_item(item: "KnownItem | str") -> str:
        return item.format() if isinstance(item, KnownItem) else str(item)

    @classmethod
    def from_json(cls, value: JsonValue) -> "KnownItem | None":
        fact = _memory_fact_from_json(value)
        if fact is None:
            return None
        item = _json_dict(value)
        return cls(text=fact, source=_source_from_json(item) if item else ())


@dataclass
class Hypothesis:
    text: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    id: str = ""
    source: tuple[str, ...] = ()
    context: str = ""

    def format(self, indent: str = "") -> str:
        prefix = "[" + self.status + "] "
        if self.id:
            prefix += self.id + ": "
        suffix = (" [" + ", ".join(self.source) + "]") if self.source else ""
        lines = [prefix + self.text + suffix]
        if self.context:
            lines.append("  context: " + self.context)
        return _format_lines(lines, indent)

    @classmethod
    def from_json(cls, value: JsonValue) -> "Hypothesis | None":
        item = _json_dict(value)
        text = _json_str(item.get("text")) or ""
        if not text:
            return None
        status = _json_str(item.get("status")) or HypothesisStatus.ACTIVE
        if status not in ALL_HYPOTHESIS_STATUSES:
            status = HypothesisStatus.ACTIVE
        return cls(
            text=text,
            status=HypothesisStatus(status),
            id=_json_str(item.get("id")) or "",
            source=_source_from_json(item),
            context=_json_str(item.get("context")) or "",
        )


class VerificationStatus(StrEnum):
    IDLE = "idle"
    REQUIRED = "required"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class VerificationBlocker(StrEnum):
    NONE = ""
    USER = "user"
    ENVIRONMENT = "environment"
    TOOL = "tool"
    UNKNOWN = "unknown"


ALL_VERIFICATION_BLOCKERS = frozenset(VerificationBlocker)


@dataclass
class Verification:
    goal: str = ""
    status: VerificationStatus = VerificationStatus.IDLE
    kind: str = ""
    method: str = ""
    criteria: list[str] = field(default_factory=list)
    context: str = ""
    blocker: VerificationBlocker = VerificationBlocker.NONE

    def format(self, indent: str = "") -> str:
        lines = ["status: " + self.status]
        if self.goal:
            lines.append("goal: " + self.goal)
        if self.kind:
            lines.append("kind: " + self.kind)
        if self.method:
            lines.append("method: " + self.method)
        if self.criteria:
            lines.append("criteria:")
            lines.extend("- " + item for item in self.criteria)
        if self.context:
            lines.append("context: " + self.context)
        if self.blocker:
            lines.append("blocker: " + self.blocker)
        return _format_lines(lines, indent)

    def reset(self) -> None:
        self.goal = ""
        self.status = VerificationStatus.IDLE
        self.kind = ""
        self.method = ""
        self.criteria = []
        self.context = ""
        self.blocker = VerificationBlocker.NONE

    def has_context(self) -> bool:
        return bool(self.goal or self.kind or self.method or self.criteria or self.context or self.blocker or self.status != VerificationStatus.IDLE)


@dataclass
class ToolResultItem:
    description: str
    value: str
    log_path: str = ""
    original_lines: int = 0
    original_chars: int = 0
    excerpted: bool = False

    def format(self, indent: str = "", *, result_key: str = "", include_content: bool = False, content: str | None = None) -> str:
        lines = ["- result_key: " + result_key] if result_key else ["- result"]
        lines.append("  description: " + self.description)
        if self.original_lines or self.original_chars:
            lines.append("  size: " + str(self.original_lines) + " lines, " + str(self.original_chars) + " chars")
        if self.excerpted:
            lines.append("  excerpted: true")
        if include_content:
            lines.append("  content:")
            lines.append("  <content>")
            lines.append(self.value if content is None else content)
            lines.append("  </content>")
        return _format_lines(lines, indent)


@dataclass
class UserRules:
    content: str = ""

    @classmethod
    def load(cls, path: str) -> "UserRules":
        try:
            with open(path, "r", encoding="utf-8") as file:
                return cls(file.read().strip())
        except FileNotFoundError:
            return cls()

    def add(self, rule: str) -> bool:
        rule = self._clean_rule(rule)
        if not rule or rule in self._rules():
            return False
        prefix = "# User Rules\n\n" if not self.content.strip() else self.content.rstrip() + "\n"
        self.content = prefix + "- " + rule
        return True

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write((self.content.strip() or "# User Rules").rstrip() + "\n")

    def format(self, indent: str = "") -> str:
        return _format_lines((self.content.strip() or "(empty)").splitlines(), indent)

    def _rules(self) -> set[str]:
        return {rule for line in self.content.splitlines() if (rule := self._clean_rule(line)) and not rule.startswith("#")}

    @staticmethod
    def _clean_rule(rule: str) -> str:
        rule = " ".join(rule.strip().split())
        return rule[2:].strip() if rule.startswith("- ") else rule


@dataclass
class Blackboard:
    user_input: str = ""
    task_code: TaskCode = TaskCode.DONE
    work_mode: WorkMode = WorkMode.NORMAL
    goal: str = ""
    goal_reached: bool = False
    plan: list[PlanItem] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    known: list[KnownItem] = field(default_factory=list)
    memory_checkpoint_tool_result_counter: int = 0
    stable_knowledge: dict[str, list[str]] = field(default_factory=dict)
    verification_required: bool = False
    verification: Verification = field(default_factory=Verification)

    def source_result_keys(self) -> set[str]:
        keys = {key for item in self.known for key in KnownItem.source_of(item) if key.startswith("tr.")}
        keys.update(key for item in self.hypotheses for key in item.source if key.startswith("tr."))
        return keys


@dataclass
class ProviderConfig:
    url: str = ""
    key: str = ""
    model: str = ""
    available_models: tuple[str, ...] = ()
    temperature: float | None = None
    reasoning: bool | None = True
    reasoning_effort: str = "medium"
    reasoning_payload: str = ""
    stream: bool | None = True
    timeout: int | None = 180
    first_token_timeout: int | None = 90

    @classmethod
    def from_dict(cls, data: Json) -> "ProviderConfig":
        defaults = cls()
        return cls(
            url=Config.str(data, "url", defaults.url),
            key=Config.str(data, "key", defaults.key),
            model=Config.str(data, "model", defaults.model),
            available_models=Config.str_tuple(data, "available_models"),
            temperature=Config.float(data, "temperature", defaults.temperature),
            reasoning=Config.bool(data, "reasoning", defaults.reasoning),
            reasoning_effort=Config.str(data, "reasoning_effort", defaults.reasoning_effort),
            reasoning_payload=cls._reasoning_payload(data, defaults.reasoning_payload),
            stream=Config.bool(data, "stream", defaults.stream),
            timeout=Config.int(data, "timeout", defaults.timeout),
            first_token_timeout=Config.int(data, "first_token_timeout", defaults.first_token_timeout),
        )

    @classmethod
    def _reasoning_payload(cls, data: Json, default: str) -> str:
        value = Config.str(data, "reasoning_payload", default)
        if value not in ("", "reasoning", "reasoning_effort"):
            raise ConfigError("config provider.reasoning_payload must be one of: reasoning, reasoning_effort, empty")
        return value


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, *, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens


############################
# Config
############################


@dataclass
class RuntimeSettings:
    shell_timeout: int = 60
    compact_at: int = 50
    max_agent_steps: int = 100
    plan_timeout: int = 360
    plan_first_token_timeout: int = 180
    auto_clean_recent: str = "3d"
    yolo: bool = False
    plan_mode: bool = False
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Json, *, yolo: bool = False, plan_mode: bool = False, debug: bool = False) -> "RuntimeSettings":
        runtime = Config.table(data, "runtime")
        return cls(
            shell_timeout=Config.int(runtime, "shell_timeout", 60),
            compact_at=Config.int(runtime, "compact_at", 50),
            max_agent_steps=max(1, Config.int(runtime, "max_agent_steps", 100) or 0),
            plan_timeout=max(1, Config.int(runtime, "plan_timeout", 360) or 0),
            plan_first_token_timeout=max(1, Config.int(runtime, "plan_first_token_timeout", 180) or 0),
            auto_clean_recent=cls.clean_retention(Config.str(runtime, "auto_clean_recent", "3d")),
            yolo=yolo or bool(Config.bool(runtime, "yolo", False)),
            plan_mode=plan_mode or bool(Config.bool(runtime, "plan_mode", False)),
            debug=debug,
        )

    @staticmethod
    def clean_retention(value: str) -> str:
        value = value.strip().lower()
        if value in {"", "off", "0", "0m", "0h", "0d"}:
            return "off"
        if not re.fullmatch(r"[1-9]\d*[mhd]", value):
            raise ConfigError("runtime.auto_clean_recent must be off or a duration like 30m, 12h, 3d")
        return value

    @staticmethod
    def clean_retention_seconds(value: str) -> int:
        value = RuntimeSettings.clean_retention(value)
        if value == "off":
            return 0
        units = {"m": 60, "h": 3600, "d": 86400}
        return int(value[:-1]) * units[value[-1]]


@dataclass
class Config:
    active_provider: str = "default"
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: {"default": ProviderConfig()})
    data_dir: str = ".nanocode"

    @classmethod
    def from_dict(cls, data: Json) -> "Config":
        provider_table = cls.table(data, "provider")
        paths = cls.table(data, "paths")
        active = cls.str(provider_table, "active", "default")
        providers = {str(name): ProviderConfig.from_dict(value) for name, value in provider_table.items() if name != "active" and isinstance(value, dict)}
        if not providers:
            providers[active] = ProviderConfig()
        if active not in providers:
            raise ConfigError("config provider.active must match a [provider.<name>] section")
        return cls(
            active_provider=active,
            providers=providers,
            data_dir=cls.str(paths, "data_dir", "~/.nanocode"),
        )

    @property
    def provider(self) -> ProviderConfig:
        return self.providers[self.active_provider]

    @classmethod
    def table(cls, config: Json, name: str) -> Json:
        value = config.get(name)
        return value if isinstance(value, dict) else {}

    @classmethod
    def str(cls, config: Json, key: str, default: str = "") -> str:
        value = config.get(key)
        return default if value is None else str(value)

    @classmethod
    def bool(cls, config: Json, key: str, default: bool | None = None) -> bool | None:
        value = config.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        raise ConfigError(f"config value `{key}` must be a boolean")

    @classmethod
    def float(cls, config: Json, key: str, default: float | None = None) -> float | None:
        value = config.get(key)
        if value is None:
            return default
        if value is False or (isinstance(value, str) and value.lower() == "off"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config value `{key}` must be a number or off")
        return float(value)

    @classmethod
    def int(cls, config: Json, key: str, default: int | None = None) -> int | None:
        value = config.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config value `{key}` must be an integer")
        return value

    @classmethod
    def str_tuple(cls, config: Json, key: str) -> tuple[str, ...]:
        value = config.get(key)
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ConfigError(f"config value `{key}` must be a string array")
        models = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"config value `{key}` must be a string array")
            if item := item.strip():
                models.append(item)
        return tuple(models)


class ConfigFile:
    DEFAULT_TEXT: ClassVar[str] = """# nanocode configuration
# Location: ~/.nanocode/config.toml

[provider]
active = "default"

[provider.default]
# OpenAI-compatible chat completions base URL, for example "https://api.openai.com/v1".
url = ""
# API key for the configured provider.
key = ""
# Default model used by nanocode.
model = ""
# Optional: add available_models = ["model-a", "model-b"] manually to pin preferred
# /model choices above automatically discovered provider models.
# Optional. Uncomment only for models/providers that support temperature.
# temperature = 0.7
reasoning = true
reasoning_effort = "medium"
# Optional reasoning payload shape. Leave unset for broad OpenAI-compatible
# compatibility. Set only for providers that require it, for example OpenRouter:
# reasoning_payload = "reasoning" sends {"reasoning":{"effort":...}}
# reasoning_payload = "reasoning_effort" sends a top-level effort.
stream = true
timeout = 180
# Stream mode only: retry if no first content token arrives within this many seconds.
first_token_timeout = 90

[paths]
# Global nanocode data directory. Project/session data is stored below this directory.
data_dir = "~/.nanocode"

[runtime]
shell_timeout = 60
compact_at = 50
max_agent_steps = 100
plan_timeout = 360
plan_first_token_timeout = 180
# Automatically delete tool-result logs older than this from inactive sessions. Use "off" to disable.
auto_clean_recent = "3d"
yolo = false
plan_mode = false
"""

    @classmethod
    def path(cls) -> str:
        return os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = os.path.expanduser(path) if path else cls.path()
        if os.path.exists(config_path):
            return config_path, False
        parent = os.path.dirname(config_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(cls.DEFAULT_TEXT)
        return config_path, True

    @classmethod
    def load(cls, path: str | None = None) -> Json:
        config_path = os.path.expanduser(path) if path else cls.path()
        try:
            with open(config_path, "rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError as error:
            raise ConfigError(f"Config file not found: {config_path}. Run `nanocode --init-config` to create one.") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"Invalid config file {config_path}: {error}") from error
        return data if isinstance(data, dict) else {}


############################
# Agent Runtime (dataclasses)
############################


class AgentMode(StrEnum):
    ACT = "act"
    OBSERVE = "observe"


@dataclass
class AgentRuntime:
    recent_edits: list[str] = field(default_factory=list)
    consecutive_tool_turns: int = 0


@dataclass
class AgentRunResult:
    done: bool = False
    value: JsonValue = None


class RangeFingerprintStore:
    MAX_ENTRIES: ClassVar[int] = 200

    @dataclass
    class Entry:
        fingerprint: str
        filepath: str
        start: int
        end: int
        content: str

    @dataclass
    class Resolved:
        start: int
        end: int
        fingerprint: str
        relocated_from: tuple[int, int] | None = None

    def __init__(self):
        self._entries: list[RangeFingerprintStore.Entry] = []

    def remember(self, *, filepath: str, start: int, end: int, content: str) -> str:
        fingerprint = _range_fingerprint(content)
        entry = self.Entry(fingerprint=fingerprint, filepath=os.path.realpath(filepath), start=start, end=end, content=content)
        if entry not in self._entries:
            self._entries.append(entry)
            del self._entries[: max(0, len(self._entries) - self.MAX_ENTRIES)]
        return fingerprint

    def clear(self) -> None:
        self._entries = []

    def __len__(self) -> int:
        return len(self._entries)

    def resolve(self, lines: list[str], *, filepath: str, start: int, end: int, fingerprint: str) -> Resolved:
        resolved_start = min(start, len(lines))
        resolved_end = len(lines) if end == 0 else min(end, len(lines))
        resolved_end = max(resolved_end, resolved_start)
        current = "".join(lines[resolved_start:resolved_end])
        current_fingerprint = _range_fingerprint(current)
        if current_fingerprint == fingerprint:
            return self.Resolved(start=resolved_start, end=resolved_end, fingerprint=current_fingerprint)

        for content in self._candidate_contents(
            filepath=filepath,
            start=resolved_start,
            end=resolved_end,
            fingerprint=fingerprint,
        ):
            if _range_fingerprint(content) == current_fingerprint:
                return self.Resolved(start=resolved_start, end=resolved_end, fingerprint=current_fingerprint)

        matches = self._find_matches(lines, filepath=filepath, start=resolved_start, end=resolved_end, fingerprint=fingerprint)
        message = (
            f"fingerprint mismatch for range {start}:{end}: expected {fingerprint}, current {current_fingerprint}; "
            f"call Read(filepath, {start}, {end}) and reuse that range fingerprint"
        )
        other_ranges = self._ranges_for_fingerprint(filepath=filepath, fingerprint=fingerprint)
        if other_ranges:
            message += "; this fingerprint was cached for range(s): " + ", ".join(f"{range_start}:{range_end}" for range_start, range_end in other_ranges)
        if not matches:
            raise ToolCallError(message)
        if len(matches) > 1:
            raise ToolCallError(message + "; cached range matched multiple locations")
        relocated_start, relocated_end = matches[0]
        return self.Resolved(
            start=relocated_start,
            end=relocated_end,
            fingerprint=_range_fingerprint("".join(lines[relocated_start:relocated_end])),
            relocated_from=(resolved_start, resolved_end),
        )

    def _find_matches(self, lines: list[str], *, filepath: str, start: int, end: int, fingerprint: str) -> list[tuple[int, int]]:
        contents = [content for content in self._candidate_contents(filepath=filepath, start=start, end=end, fingerprint=fingerprint) if content]

        matches = []
        for content in contents:
            expected = content.splitlines(keepends=True)
            if not expected:
                continue
            last_start = len(lines) - len(expected)
            for position in range(max(0, last_start + 1)):
                if lines[position : position + len(expected)] == expected:
                    matches.append((position, position + len(expected)))
                    if len(matches) > 1:
                        return matches
        return matches

    def _candidate_contents(self, *, filepath: str, start: int, end: int, fingerprint: str) -> list[str]:
        filepath = os.path.realpath(filepath)
        contents: list[str] = []
        for entry in self._entries:
            if entry.fingerprint != fingerprint or entry.filepath != filepath:
                continue
            if start == end:
                if entry.start == start and entry.end == end and entry.content == "":
                    contents.append("")
                continue
            entry_lines = entry.content.splitlines(keepends=True)
            cached_end = entry.start + len(entry_lines)
            if start < entry.start or end > cached_end:
                continue
            candidate = "".join(entry_lines[start - entry.start : end - entry.start])
            if candidate not in contents:
                contents.append(candidate)
        return contents

    def _ranges_for_fingerprint(self, *, filepath: str, fingerprint: str) -> list[tuple[int, int]]:
        filepath = os.path.realpath(filepath)
        ranges = []
        for entry in self._entries:
            if entry.fingerprint != fingerprint or entry.filepath != filepath:
                continue
            item = (entry.start, entry.end)
            if item not in ranges:
                ranges.append(item)
        return ranges


@dataclass
class RuntimeState:
    debug_prompt_count: int = 0
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    current_model_call_started_at: float = 0.0
    current_model_call_label: str = ""
    current_model_call_reasoning_label: str = ""
    current_model_call_activity: str = ""
    current_model_call_has_content: bool = False
    status_notice: str = ""
    status_notice_until: float = 0.0
    conversation: list[ConversationItem] = field(default_factory=list)
    user_rules: UserRules = field(default_factory=UserRules)
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)
    tool_result_store: dict[str, ToolResultItem] = field(default_factory=dict)
    tool_result_counter: int = 0
    turn_tool_calls: int = 0
    session_tool_calls: int = 0
    turn_model_calls: int = 0


@dataclass
class Session:
    # ---- system ----
    system: str = field(default_factory=platform.system)
    arch: str = field(default_factory=platform.machine)
    cwd: str = field(default_factory=os.getcwd)
    bash: str = field(default_factory=lambda: shutil.which("bash") or "")
    config: Config = field(default_factory=Config)
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    state: RuntimeState = field(default_factory=RuntimeState)
    session_id: str = field(default_factory=lambda: Session._new_session_id())

    @classmethod
    def from_config_file(cls, *, path: str | None = None, yolo: bool = False, plan_mode: bool = False, debug: bool = False) -> "Session":
        return cls.from_config_data(ConfigFile.load(path), yolo=yolo, plan_mode=plan_mode, debug=debug)

    @classmethod
    def from_config_data(cls, data: Json, *, yolo: bool = False, plan_mode: bool = False, debug: bool = False) -> "Session":
        session = cls(config=Config.from_dict(data), settings=RuntimeSettings.from_dict(data, yolo=yolo, plan_mode=plan_mode, debug=debug))
        session.load_user_rules()
        return session

    def resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return os.path.abspath(path)

    def data_path(self, *parts: str) -> str:
        base = os.path.expanduser(self.config.data_dir)
        if not os.path.isabs(base):
            base = os.path.join(self.cwd, base)
        return os.path.abspath(os.path.join(base, *parts))

    def is_path_in_cwd(self, path: str) -> bool:
        cwd = os.path.realpath(self.cwd)
        path = os.path.realpath(path)
        try:
            return os.path.commonpath([cwd, path]) == cwd
        except ValueError:
            return False

    def append_conversation(self, item: ConversationItem) -> None:
        self.state.conversation.append(item)

    def project_key(self) -> str:
        cwd = os.path.realpath(self.cwd)
        basename = self._safe_path_name(os.path.basename(cwd.rstrip(os.sep)) or "root")
        digest = hashlib.sha1(cwd.encode("utf-8")).hexdigest()[:10]
        return basename + "-" + digest

    @staticmethod
    def _safe_path_name(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
        return value or "project"

    @staticmethod
    def _new_session_id() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()) + "-" + uuid.uuid4().hex[:8]

    def project_dir(self) -> str:
        return self.data_path("projects", self.project_key())

    def session_dir(self) -> str:
        return self.data_path("sessions", self.session_id)

    def history_path(self) -> str:
        return self.data_path("history")

    def debug_dir(self) -> str:
        return os.path.join(self.session_dir(), "debug")

    def tool_results_dir(self) -> str:
        return os.path.join(self.session_dir(), "tool_results")

    def lock_path(self) -> str:
        return os.path.join(self.session_dir(), "session.lock")

    def user_rules_path(self) -> str:
        return os.path.join(self.project_dir(), "user_rules.md")

    def load_user_rules(self) -> None:
        self.state.user_rules = UserRules.load(self.user_rules_path())

    def save_user_rules(self) -> None:
        self.state.user_rules.save(self.user_rules_path())

    def missing_required_config(self) -> list[str]:
        provider = self.config.provider
        return [key for key, value in (("provider.url", provider.url), ("provider.key", provider.key), ("provider.model", provider.model)) if not value]


############################
# Tools
############################


class ToolEffect(StrEnum):
    READONLY = "readonly"
    EDIT = "edit"
    OTHER = "other"


MAX_TOOL_OUTPUT_CHARS = 12_000


class Tool:
    NAME: ClassVar[str] = ""
    DESCRIPTION: ClassVar[tuple[str, ...]] = ()
    SIGNATURE: ClassVar[str]
    EXAMPLE: ClassVar[tuple[str, ...]] = ()
    EFFECT: ClassVar[ToolEffect] = ToolEffect.OTHER
    REQUIRES_CONFIRMATION: ClassVar[bool | None] = None

    @classmethod
    def name(cls) -> str:
        return cls.NAME or cls.__name__.removesuffix("Tool")

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        return [cls.cli_token(arg) for arg in args]

    @staticmethod
    def cli_content_summary(value: str) -> str:
        line_count = _tool_output_line_count(value)
        if line_count > 1:
            return "<" + str(line_count) + " lines>"
        return "<" + str(len(value)) + " chars>"

    @staticmethod
    def cli_token(value: str) -> str:
        text = str(value)
        if "\n" in text:
            return Tool.cli_content_summary(text)
        text = _shorten(text, 100)
        if not text:
            return '""'
        if re.fullmatch(r"[A-Za-z0-9_./:@=,+%~*{}-]+", text):
            return text
        return json.dumps(text, ensure_ascii=False)

    @classmethod
    def effect(cls) -> ToolEffect:
        return cls.EFFECT

    def requires_confirmation(self, session: Session) -> bool:
        return self.REQUIRES_CONFIRMATION if self.REQUIRES_CONFIRMATION is not None else self.effect() == ToolEffect.EDIT

    def call_live(self, sink: Callable[[str], None] | None = None) -> str:
        return self.call()


ToolClass: TypeAlias = Type[Tool]


@dataclass
class ParsedToolCall:
    name: str
    intention: str
    args: list[str]

    @property
    def executed(self) -> str:
        return self.name + "(" + ", ".join(json.dumps(arg, ensure_ascii=False) for arg in self.args) + ")"


@dataclass
class ToolCallExecution:
    call: ParsedToolCall
    outcome: str
    output: str
    error_type: Type[Exception] | None = None
    result_key: str = ""
    result_excerpted: bool = False
    requires_verification: bool = False


@dataclass
class PreparedToolCall:
    call: ParsedToolCall
    tool: Tool


@dataclass
class BoundedToolOutput:
    value: str
    excerpted: bool
    original_lines: int
    original_chars: int


def _tool_output_line_count(output: str) -> int:
    if not output:
        return 0
    return output.count("\n") + (0 if output.endswith("\n") else 1)


def _bound_tool_output(output: str, *, log_path: str = "", max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> BoundedToolOutput:
    original_chars = len(output)
    original_lines = _tool_output_line_count(output)
    if original_chars <= max_chars:
        return BoundedToolOutput(output, False, original_lines, original_chars)

    header = (
        "[tool result excerpt]\n"
        "excerpted: true\n"
        "original_lines: " + str(original_lines) + "\noriginal_chars: " + str(original_chars) + "\n"
    )
    labels = ("\n--- head ---\n", "\n--- middle ---\n", "\n--- tail ---\n")
    body_budget = max_chars - len(header) - sum(len(label) for label in labels)
    if body_budget <= 0:
        return BoundedToolOutput(header[:max_chars], True, original_lines, original_chars)

    head_size = body_budget // 3
    middle_size = body_budget // 3
    tail_size = body_budget - head_size - middle_size
    middle_start = max(0, original_chars // 2 - middle_size // 2)
    value = header + labels[0] + output[:head_size] + labels[1] + output[middle_start : middle_start + middle_size] + labels[2] + output[-tail_size:]
    return BoundedToolOutput(value[:max_chars], True, original_lines, original_chars)


RESULT_KEY_PATTERN: re.Pattern[str] = re.compile(r"\b(?:(?:result_)?key|recall)[:=]\s*(tr\.\d+)\b")


def _format_tool_call_summary(call: ParsedToolCall) -> str:
    return "tool=" + call.name + " args=" + json.dumps(call.args, ensure_ascii=False, separators=(",", ":"))


@dataclass
class ToolResultContext:
    latest: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    pending_observe: list[str] = field(default_factory=list)
    kept_results: list[str] = field(default_factory=list)

    def forget_results(self, keys: list[str]) -> list[str]:
        wanted = set(keys)
        if not wanted:
            return []
        removed = []

        def remove_blocks(blocks: list[str]) -> list[str]:
            kept = []
            for block in blocks:
                key = self.result_key(block)
                if key in wanted:
                    removed.append(key)
                else:
                    kept.append(block)
            return kept

        def compact_blocks(blocks: list[str]) -> list[str]:
            compacted = []
            for block in blocks:
                key = self.result_key(block)
                if key in wanted:
                    removed.append(key)
                    compacted.append(self.compact_block(block))
                else:
                    compacted.append(block)
            return compacted

        self.kept_results = remove_blocks(self.kept_results)
        self.pending_observe = remove_blocks(self.pending_observe)
        self.latest = compact_blocks(self.latest)
        self.recent = compact_blocks(self.recent)
        return list(dict.fromkeys(removed))

    def keep_results(self, actions: list[Json], observed_blocks: list[str], *, max_chars: int) -> list[str]:
        wanted = []
        for action in actions:
            if _json_str(action.get("type")) == "keep":
                wanted.extend(key for key in _source_from_json(action) if key.startswith("tr."))
        wanted = list(dict.fromkeys(wanted))
        if not wanted:
            return []
        by_key = self.blocks_by_key(observed_blocks)
        selected = {key: by_key[key] for key in wanted if key in by_key}
        if not selected:
            return []
        existing = self.blocks_by_key(self.kept_results)
        self.kept_results = [block for key, block in existing.items() if key not in selected] + [selected[key] for key in wanted if key in selected]
        while self.kept_results and len("\n\n".join(self.kept_results)) > max_chars:
            del self.kept_results[0]
        retained = self.blocks_by_key(self.kept_results)
        return [key for key in wanted if key in selected and key in retained]

    def append_latest(self, executions: list[ToolCallExecution], *, max_summaries: int, max_chars: int) -> None:
        if not executions:
            return
        self.append_recent(self.latest, max_summaries=max_summaries, max_chars=max_chars)
        self.latest = [self.format_execution(execution) for execution in executions]
        self.prune_recent(max_summaries=max_summaries, max_chars=max_chars)

    def append_recent(self, blocks: list[str], *, max_summaries: int, max_chars: int) -> None:
        if not blocks:
            return
        self.recent.extend(blocks)
        self.prune_recent(max_summaries=max_summaries, max_chars=max_chars)

    def prune_recent(self, *, max_summaries: int, max_chars: int) -> None:
        self.recent = [self.compact_block(block) for block in self.recent]
        del self.recent[: max(0, len(self.recent) - max_summaries)]
        latest = [self.compact_block(block) for block in self.latest]
        while self.recent and len("\n\n".join(self.recent + latest)) > max_chars:
            del self.recent[0]

    def queue_observation(self, blocks: list[str], *, checkpoint: int) -> bool:
        queued = {self.result_counter(block) for block in self.pending_observe}
        for block in blocks:
            counter = self.result_counter(block)
            if counter > checkpoint and counter not in queued:
                self.pending_observe.append(block)
                queued.add(counter)
        return bool(self.pending_observe)

    def mark_checkpoint(self, checkpoint: int) -> None:
        self.pending_observe = [block for block in self.pending_observe if self.result_counter(block) > checkpoint]

    def compact_observed(self, observed_blocks: list[str]) -> None:
        observed = {self.result_counter(block) for block in observed_blocks}
        if not observed:
            return

        def compact(block: str) -> str:
            if self.is_full_block(block) and self.result_counter(block) in observed:
                return self.compact_block(block)
            return block

        self.recent = [compact(block) for block in self.recent]
        self.latest = [compact(block) for block in self.latest]

    def visible_counter(self, mode: AgentMode) -> int:
        if mode == AgentMode.OBSERVE and self.pending_observe:
            return self.max_counter(self.pending_observe)
        return self.max_counter(self.recent + self.latest)

    @classmethod
    def blocks_by_key(cls, blocks: list[str]) -> dict[str, str]:
        return {key: block for block in blocks for key in [cls.result_key(block)] if key}

    @staticmethod
    def format_execution(execution: ToolCallExecution) -> str:
        status = "ok" if execution.outcome == "success" else "fail"
        fields = [status, _format_tool_call_summary(execution.call)]
        if execution.result_key:
            fields.append("key=" + execution.result_key)
        lines = ["- " + " ".join(fields)]
        if execution.call.intention:
            lines.append("  why: " + execution.call.intention)
        lines.extend(["  output:", execution.output])
        return "\n".join(lines)

    @staticmethod
    def is_full_block(block: str) -> bool:
        return "\n  output:\n" in block

    @classmethod
    def compact_block(cls, block: str) -> str:
        if not cls.is_full_block(block):
            return block
        header, output = block.split("\n  output:\n", 1)
        match = RESULT_KEY_PATTERN.search(header)
        parts = [str(_tool_output_line_count(output)) + " lines, " + str(len(output)) + " chars"] if output else []
        if "[tool result excerpt]" in output or "excerpted: true" in output:
            parts.append("excerpt")
        if match:
            parts.append("recall=" + match.group(1))
        elif output:
            parts.append(_shorten(" ".join(output.split()), 220))
        return header + "\n  out: " + ("; ".join(parts) if parts else "ok")

    @classmethod
    def result_key(cls, block: str) -> str:
        match = RESULT_KEY_PATTERN.search(block)
        return match.group(1) if match else ""

    @classmethod
    def result_counter(cls, block: str) -> int:
        key = cls.result_key(block)
        return int(key.split(".", 1)[1]) if key else 0

    @classmethod
    def max_counter(cls, blocks: list[str]) -> int:
        return max((cls.result_counter(block) for block in blocks), default=0)

    @staticmethod
    def forget_result_keys_from_actions(actions: list[Json]) -> list[str]:
        keys: list[str] = []
        for action in actions:
            if _json_str(action.get("type")) == "forget":
                keys.extend(key for key in _source_from_json(action) if key.startswith("tr."))
        return list(dict.fromkeys(keys))


ConfirmationResult: TypeAlias = bool | str
ConfirmCallback: TypeAlias = Callable[[ParsedToolCall, Tool], ConfirmationResult]
ToolDisplayCallback: TypeAlias = Callable[[ParsedToolCall, Tool], None]
ToolLiveOutputCallback: TypeAlias = Callable[[ParsedToolCall, str], None]
ToolLiveDoneCallback: TypeAlias = Callable[[ParsedToolCall], None]
MessageCallback: TypeAlias = Callable[[str], None]
StatusAction: TypeAlias = Callable[[], str]
StatusRunner: TypeAlias = Callable[[StatusAction], str]


class SelectionBack:
    pass


SELECTION_BACK = SelectionBack()
SelectionResult: TypeAlias = str | None | SelectionBack
ReasoningSelector: TypeAlias = Callable[[], SelectionResult]
ModelSelector: TypeAlias = Callable[[tuple[str, ...], str], SelectionResult]
ProviderSelector: TypeAlias = Callable[[tuple[str, ...], str], SelectionResult]


class SessionLock:
    def __init__(self, path: str):
        self.path = path
        self.file = None

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise
        self.file.seek(0)
        self.file.truncate()
        self.file.write(json.dumps({"pid": os.getpid(), "locked_at": datetime.now().isoformat()}, ensure_ascii=False))
        self.file.flush()

    def release(self) -> None:
        if self.file is None:
            return
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()
        self.file = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    @staticmethod
    def is_locked(path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r+", encoding="utf-8") as file:
                try:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
        return False


@dataclass
class CleanResult:
    cleaned: int = 0
    failed: int = 0
    skipped: int = 0


class SessionLogCleaner:
    def __init__(self, session: Session):
        self.session = session

    def clean(self, *, older_than_seconds: int = 0) -> CleanResult:
        result = CleanResult()
        sessions_dir = self.session.data_path("sessions")
        if not os.path.isdir(sessions_dir):
            return result
        cutoff = time.time() - older_than_seconds if older_than_seconds > 0 else 0.0
        for session_name in os.listdir(sessions_dir):
            session_dir = os.path.join(sessions_dir, session_name)
            if not os.path.isdir(session_dir):
                continue
            if SessionLock.is_locked(os.path.join(session_dir, "session.lock")):
                result.skipped += 1
                continue
            tool_results_dir = os.path.join(session_dir, "tool_results")
            if not os.path.isdir(tool_results_dir):
                continue
            for name in os.listdir(tool_results_dir):
                path = os.path.join(tool_results_dir, name)
                if not name.endswith(".log") or not os.path.isfile(path):
                    continue
                if cutoff and os.path.getmtime(path) >= cutoff:
                    continue
                try:
                    os.remove(path)
                    result.cleaned += 1
                except OSError:
                    result.failed += 1
        return result


############################
# Tool Helpers
############################


def _parse_line_range(start_arg: str, end_arg: str) -> tuple[int, int]:
    try:
        start = max(0, int(start_arg))
    except (ValueError, TypeError):
        raise ToolCallArgError("invalid start: should be an integer")
    try:
        end = max(0, int(end_arg))
    except (ValueError, TypeError):
        raise ToolCallArgError("invalid end: should be an integer")
    if end:
        end = max(end, start)
    return start, end


def _range_fingerprint(content: str) -> str:
    return hashlib.blake2s(content.encode("utf-8"), digest_size=3).hexdigest()


############################
# Tool Implementations
############################


@dataclass
class ReadTool(Tool):
    MAX_LINES: ClassVar[int] = 600
    EFFECT: ClassVar[ToolEffect] = ToolEffect.READONLY
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Read a single known UTF-8 file; pass multiple 0-based start,end ranges for it.",
        "Each range returns at most 600 lines.",
    )
    SIGNATURE: ClassVar[str] = "Read(filepath[, range_token...]) -> ReadToolResult<fingerprint, content>"
    EXAMPLE: ClassVar[tuple[str, ...]] = (
        'Example args: ["code.py", "0,80", "160,220"]',
        'Example args: ["code.py"]',
    )

    filepath: str = ""
    start: int = 0
    end: int = 0
    ranges: list[tuple[int, int]] = field(default_factory=list)
    filepaths: list[str] = field(default_factory=list)
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        if not args:
            return []
        tokens = [cls.cli_token(args[0])]
        if len(args) == 3 and args[1].isdigit() and args[2].isdigit():
            return tokens + [args[1] + ":" + args[2]]
        return tokens + [str(arg) for arg in args[1:]]

    @staticmethod
    def _parse_line_range_token(value: str) -> tuple[int, int]:
        match = re.fullmatch(r"\s*(\d+)\s*[-:,]\s*(\d+)\s*", value)
        if match is None:
            raise ToolCallArgError("invalid range: use a comma token like 0,120")
        return _parse_line_range(match.group(1), match.group(2))

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) == 0:
            raise ToolCallArgError(
                'Read args error: got 0 args; expected ["filepath"] or ["filepath", "start,end"]. Example: Read("nanocode.py", "2065,2095"). Do not call Read().'
            )
        filepath = session.resolve_path(args[0])
        if len(args) == 1:
            ranges = [(0, 0)]
        elif all(re.fullmatch(r"\s*\d+\s*[-:,]\s*\d+\s*", arg) for arg in args[1:]):
            ranges = [cls._parse_line_range_token(arg) for arg in args[1:]]
        elif len(args) == 3 and cls._is_integer_token(args[1]) and cls._is_integer_token(args[2]):
            ranges = [_parse_line_range(args[1], args[2])]
        elif cls._all_args_are_existing_files(session, args):
            filepaths = [session.resolve_path(arg) for arg in args]
            return cls(
                filepath=filepaths[0],
                start=0,
                end=0,
                ranges=[(0, 0)],
                filepaths=filepaths,
                cwd=session.cwd,
                range_fingerprints=session.state.range_fingerprints,
            )
        elif len(args) == 3:
            ranges = [_parse_line_range(args[1], args[2])]
        elif len(args) == 2:
            raise ToolCallArgError('Read args error: invalid range token; expected ["filepath", "start,end"]. Example: Read("nanocode.py", "2065,2095").')
        else:
            raise ToolCallArgError('Read args error: for multiple ranges use comma tokens. Example: Read("nanocode.py", "0,40", "200,260").')
        start, end = ranges[0]
        return cls(filepath=filepath, start=start, end=end, ranges=ranges, cwd=session.cwd, range_fingerprints=session.state.range_fingerprints)

    @staticmethod
    def _all_args_are_existing_files(session: Session, args: list[str]) -> bool:
        if len(args) < 2:
            return False
        return all(os.path.isfile(session.resolve_path(arg)) for arg in args)

    @staticmethod
    def _is_integer_token(value: str) -> bool:
        return re.fullmatch(r"\s*-?\d+\s*", str(value)) is not None

    def requires_confirmation(self, session: Session) -> bool:
        return any(not session.is_path_in_cwd(filepath) for filepath in self._target_filepaths())

    def preview(self) -> str:
        if self.filepaths:
            return "Read(" + ", ".join(self.filepaths) + ")"
        if len(self.ranges) > 1:
            ranges = ", ".join(str(start) + ":" + str(end) for start, end in self.ranges)
            return f"Read({self.filepath}, {ranges})"
        return f"Read({self.filepath}, {self.start}, {self.end})"

    def call(self) -> str:
        if self.filepaths:
            lines = ["<ReadToolResult>", "  <file_count>" + str(len(self.filepaths)) + "</file_count>"]
            for filepath in self.filepaths:
                content, returned_end, fingerprint_end, fingerprint, truncated, total_lines = self._read_range(0, 0, filepath=filepath)
                lines.append("  <ReadFile>")
                lines.append("    <path>" + filepath + "</path>")
                lines.extend(self._format_range_result(0, returned_end, fingerprint_end, fingerprint, truncated, total_lines, content, indent="    "))
                lines.append("  </ReadFile>")
            lines.append("</ReadToolResult>")
            return "\n".join(lines)

        if len(self.ranges) > 1:
            lines = ["<ReadToolResult>", "  <range_count>" + str(len(self.ranges)) + "</range_count>"]
            for start, end in self.ranges:
                content, returned_end, fingerprint_end, fingerprint, truncated, total_lines = self._read_range(start, end)
                lines.append("  <ReadRange>")
                lines.extend(self._format_range_result(start, returned_end, fingerprint_end, fingerprint, truncated, total_lines, content, indent="    "))
                lines.append("  </ReadRange>")
            lines.append("</ReadToolResult>")
            return "\n".join(lines)

        content, returned_end, fingerprint_end, fingerprint, truncated, total_lines = self._read_range(self.start, self.end)
        lines = ["<ReadToolResult>"]
        lines.extend(self._format_range_result(self.start, returned_end, fingerprint_end, fingerprint, truncated, total_lines, content, indent="  "))
        lines.append("</ReadToolResult>")
        return "\n".join(lines)

    def _target_filepaths(self) -> list[str]:
        return self.filepaths or [self.filepath]

    def _read_range(self, start: int, end: int, *, filepath: str | None = None) -> tuple[str, int, int, str, bool, int]:
        target_filepath = filepath or self.filepath
        total_lines = 0
        selected_lines = []
        truncated = False
        bounded_read_lines = end - start if end else 0
        if end and bounded_read_lines <= self.MAX_LINES:
            with open(target_filepath, "r", encoding="utf-8") as f:
                selected_lines = list(itertools.islice(f, start, end))
        else:
            with open(target_filepath, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    total_lines = index + 1
                    if index < start:
                        continue
                    if end and index >= end:
                        continue
                    if len(selected_lines) < self.MAX_LINES:
                        selected_lines.append(line)
                        continue
                    truncated = True
        content = "".join(selected_lines)
        returned_end = start + len(selected_lines)
        fingerprint_end = returned_end if truncated else end
        fingerprint = self.range_fingerprints.remember(
            filepath=target_filepath,
            start=start,
            end=fingerprint_end,
            content=content,
        )
        return content, returned_end, fingerprint_end, fingerprint, truncated, total_lines

    def _format_range_result(
        self,
        start: int,
        returned_end: int,
        fingerprint_end: int,
        fingerprint: str,
        truncated: bool,
        total_lines: int,
        content: str,
        *,
        indent: str,
    ) -> list[str]:
        lines = [
            indent + "<range>" + str(start) + ":" + str(fingerprint_end) + "</range>",
            indent + "<fingerprint>" + fingerprint + "</fingerprint>",
        ]
        if truncated:
            note = (
                f"Read returned {returned_end - start} lines from {start}:{returned_end} of {total_lines} total lines. "
                "Use Search to locate relevant text or Read smaller ranges in batches."
            )
            lines.extend(
                [
                    indent + "<truncated>true</truncated>",
                    indent + "<total_lines>" + str(total_lines) + "</total_lines>",
                    indent + "<note>" + note + "</note>",
                ]
            )
        lines.extend([indent + "<content no-indention>", content, indent + "</content>"])
        return lines


@dataclass
class LineCountTool(Tool):
    EFFECT: ClassVar[ToolEffect] = ToolEffect.READONLY
    DESCRIPTION: ClassVar[tuple[str, ...]] = ("Count lines for one or more files. Useful before reading large files or deciding Read ranges.",)
    SIGNATURE: ClassVar[str] = "LineCount(*filepaths) -> LineCountToolResult<total_lines>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["code.py", "other.py"]',)

    filepaths: list[str] = field(default_factory=list)

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if not args:
            raise ToolCallArgError("requires at least one arg: filepath")
        return cls(filepaths=[session.resolve_path(p) for p in args])

    def requires_confirmation(self, session: Session) -> bool:
        return any(not session.is_path_in_cwd(fp) for fp in self.filepaths)

    def preview(self) -> str:
        n = len(self.filepaths)
        sample = ", ".join(self.filepaths[:2]) if n > 2 else ", ".join(self.filepaths)
        suffix = f"+{n - 2} more" if n > 2 else ""
        return f"LineCount([{sample}{suffix}])"

    def call(self) -> str:
        wc_path = shutil.which("wc") or ""
        if wc_path:
            result = subprocess.run([wc_path, "-l", *self.filepaths], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise ToolCallError((result.stderr or "wc failed").strip())
            lines = result.stdout.strip().splitlines()
            total = int(lines[-1].split()[0]) if lines else 0
            return "<LineCountToolResult>" + str(total) + "</LineCountToolResult>"
        total = 0
        for filepath in self.filepaths:
            with open(filepath, "r", encoding="utf-8", errors="replace") as file:
                total += sum(1 for _ in file)
        return "<LineCountToolResult>" + str(total) + "</LineCountToolResult>"


@dataclass
class ListDirTool(Tool):
    EFFECT: ClassVar[ToolEffect] = ToolEffect.READONLY
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "List one directory non-recursively; optional glob filters immediate entry names.",
        "Batch multiple ListDir actions in one turn when checking several known directories.",
    )
    SIGNATURE: ClassVar[str] = "ListDir([dirpath][, glob]) -> ListDirToolResult<entries>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["src"]', 'Example args: ["src", "*.py"]', "Current dir args: []")

    dirpath: str = ""
    glob_pattern: str = ""
    cwd: str = ""

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) not in (0, 1, 2):
            raise ToolCallArgError("requires 0 to 2 args: [dirpath][, glob]")
        dir_path = str(args[0]) if args else "."
        glob_pattern = str(args[1]) if len(args) == 2 else ""
        return cls(dirpath=session.resolve_path(dir_path), glob_pattern=glob_pattern, cwd=session.cwd)

    def preview(self) -> str:
        if self.glob_pattern:
            return f'ListDir({self.dirpath}, "{self.glob_pattern}")'
        return f"ListDir({self.dirpath})"

    def requires_confirmation(self, session: Session) -> bool:
        return not session.is_path_in_cwd(self.dirpath)

    def _dir_entry_type(self, entry: os.DirEntry[str]) -> str:
        if entry.is_symlink():
            return "symlink"
        if entry.is_dir(follow_symlinks=False):
            return "dir"
        if entry.is_file(follow_symlinks=False):
            return "file"
        return "other"

    def _entry_type_sort_key(self, entry_type: str) -> int:
        return {"dir": 0, "file": 1, "symlink": 2, "other": 3}.get(entry_type, 4)

    def call(self) -> str:
        if not os.path.isdir(self.dirpath):
            raise ToolCallError("not a directory")
        entries = []
        with os.scandir(self.dirpath) as scan:
            for entry in scan:
                if self.glob_pattern and not fnmatch.fnmatch(entry.name, self.glob_pattern):
                    continue
                entries.append(
                    {
                        "name": entry.name,
                        "path": entry.path,
                        "type": self._dir_entry_type(entry),
                    }
                )
        entries.sort(key=lambda item: (self._entry_type_sort_key(str(item["type"])), str(item["name"])))
        lines = ["<ListDirToolResult>"]
        for e in entries:
            lines.append(f"* ({e['type']}): {os.path.relpath(str(e['path']), self.cwd)}")
        lines.append("</ListDirToolResult>")
        return "\n".join(lines)


@dataclass
class SearchTool(Tool):
    MAX_MATCHES: ClassVar[int] = 100
    MAX_FILE_BYTES: ClassVar[int] = 2_000_000
    RG_MAX_FILESIZE: ClassVar[str] = "2M"
    CONTEXT_LINES: ClassVar[int] = 4
    MAX_CONTEXT_LINES: ClassVar[int] = 30
    EFFECT: ClassVar[ToolEffect] = ToolEffect.READONLY
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Case-insensitive regex search before Read; use A|B|C for alternatives and \\n for multiline matches.",
        "For exact text, escape regex metacharacters like braces, parens, dots, stars, and brackets.",
        "Scope with path=FILE_OR_DIR, optionally filter with one glob=*.py, set context=N for 0..30 lines; omitted path defaults to current directory.",
        "Second positional arg is always path, third positional arg is always glob; with path=, extra leading positional args are joined as regex alternatives.",
        "Use at most one glob= per Search. For multiple extensions, run multiple Search actions or search path=. without glob.",
        "Batch multiple Search actions in one turn when checking independent patterns or multiple globs.",
        "Only options are path=, glob=, context=; escape regex symbols for literal text.",
    )
    SIGNATURE: ClassVar[str] = "Search(pattern[, path=path][, glob=pattern][, context=N]) -> SearchToolResult<matches>"
    EXAMPLE: ClassVar[tuple[str, ...]] = (
        'Example args: ["class .*Tool", "path=nanocode.py", "context=0"]',
        'Example args: ["TODO|FIXME", "path=.", "glob=*.py", "context=2"]',
        'Multiple globs: use separate actions like ["pytest", "path=.", "glob=*.toml"] and ["pytest", "path=.", "glob=*.ini"].',
        'Literal paren args: ["def __init__\\(", "path=.", "glob=*.py"]',
    )

    @dataclass(frozen=True)
    class Match:
        path: str
        line_number: int
        text: str
        context: list[tuple[int, str]]

    pattern: str = ""
    target_path: str = ""
    glob_pattern: str = ""
    context_lines: int = CONTEXT_LINES
    cwd: str = ""
    gitignore_patterns: list[str] = field(default_factory=list)

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        args = cls._join_pattern_args_with_explicit_path(args)
        if len(args) < 1 or len(args) > 4:
            raise ToolCallArgError("requires 1 to 4 args: pattern[, path=path][, glob=pattern][, context=N]")
        if any(str(arg).startswith("ignore_case") or str(arg).startswith("case_sensitive") for arg in args[1:]):
            raise ToolCallArgError("Search supports only path=, glob=, and context= options; ignore_case is not supported")
        raw_pattern = str(args[0])
        if not raw_pattern:
            raise ToolCallArgError("pattern cannot be empty")
        pattern = raw_pattern[3:] if raw_pattern.startswith("re:") else raw_pattern
        if not pattern:
            raise ToolCallArgError("pattern cannot be empty")
        pattern = pattern.replace("\\n", "\n").replace("\\r", "\r")
        target_path_arg = "."
        glob_pattern = ""
        context_lines = cls.CONTEXT_LINES
        path_set = False
        for raw_option in args[1:]:
            option = str(raw_option)
            if option.startswith("ignore_case") or option.startswith("case_sensitive"):
                raise ToolCallArgError("Search supports only path=, glob=, and context= options; ignore_case is not supported")
            if option.startswith("path="):
                if path_set:
                    raise ToolCallArgError("path option cannot be combined with positional path")
                target_path_arg = option.split("=", 1)[1] or "."
                path_set = True
                continue
            if option.startswith("context=") or option.isdigit():
                try:
                    context_lines = cls._parse_context_arg(option)
                except ValueError:
                    raise ToolCallArgError("context must be an integer between 0 and " + str(cls.MAX_CONTEXT_LINES))
                continue
            if option.startswith("glob=") or option.startswith("glob_pattern="):
                if glob_pattern:
                    raise ToolCallArgError("unexpected search option: " + option)
                option = option.split("=", 1)[1]
                if not option:
                    raise ToolCallArgError("glob option cannot be empty")
                glob_pattern = option
                continue
            if not option:
                if path_set:
                    raise ToolCallArgError("unexpected search option: " + option)
                target_path_arg = "."
                path_set = True
                continue
            if path_set and not glob_pattern:
                glob_pattern = option
                continue
            if path_set:
                raise ToolCallArgError("unexpected search option: " + option)
            target_path_arg = option
            path_set = True
        try:
            re.compile(pattern)
        except re.error as error:
            raise ToolCallArgError("invalid regex: " + str(error))
        return cls(
            pattern=pattern,
            target_path=session.resolve_path(target_path_arg),
            glob_pattern=glob_pattern,
            context_lines=context_lines,
            cwd=session.cwd,
            gitignore_patterns=cls._load_gitignore_patterns(session.cwd),
        )

    @classmethod
    def _join_pattern_args_with_explicit_path(cls, args: list[str]) -> list[str]:
        values = [str(arg) for arg in args]
        path_index = next((index for index, value in enumerate(values[1:], start=1) if value.startswith("path=")), None)
        if path_index is None or path_index <= 1:
            return values
        return ["|".join(values[:path_index]), *values[path_index:]]

    @classmethod
    def _parse_context_arg(cls, value: str) -> int:
        raw_context = value[len("context=") :] if value.startswith("context=") else value
        context = int(raw_context)
        if context < 0 or context > cls.MAX_CONTEXT_LINES:
            raise ValueError
        return context

    def requires_confirmation(self, session: Session) -> bool:
        return not session.is_path_in_cwd(self.target_path)

    def preview(self) -> str:
        if self.glob_pattern:
            return f'Search("{self.pattern}", {self.target_path}, "{self.glob_pattern}")'
        return f'Search("{self.pattern}", {self.target_path})'

    def _relpath(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.cwd)
        except ValueError:
            return path

    def _matches_glob(self, path: str) -> bool:
        if not self.glob_pattern:
            return True
        return fnmatch.fnmatch(os.path.basename(path), self.glob_pattern) or fnmatch.fnmatch(self._relpath(path), self.glob_pattern)

    @staticmethod
    def _load_gitignore_patterns(cwd: str) -> list[str]:
        path = os.path.join(cwd, ".gitignore")
        patterns = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    pattern = line.strip()
                    if not pattern or pattern.startswith("#") or pattern.startswith("!"):
                        continue
                    patterns.append(pattern.lstrip("/"))
        except OSError:
            pass
        return patterns

    def _is_hidden_path(self, path: str) -> bool:
        return any(part.startswith(".") for part in self._relpath(path).split(os.sep) if part and part != ".")

    def _is_gitignored(self, path: str, is_dir: bool = False) -> bool:
        relpath = self._relpath(path).replace(os.sep, "/")
        name = os.path.basename(path)
        parts = relpath.split("/")
        for pattern in self.gitignore_patterns:
            directory_only = pattern.endswith("/")
            pattern = pattern.rstrip("/")
            if not pattern:
                continue
            if directory_only:
                if "/" in pattern:
                    matched = relpath == pattern or relpath.startswith(pattern + "/")
                else:
                    matched = pattern in parts
                if matched:
                    return True
                continue
            if "/" in pattern:
                if fnmatch.fnmatch(relpath, pattern):
                    return True
            elif fnmatch.fnmatch(name, pattern) or any(fnmatch.fnmatch(part, pattern) for part in parts):
                return True
        return False

    def _is_skipped_path(self, path: str, is_dir: bool = False) -> bool:
        return self._is_hidden_path(path) or self._is_gitignored(path, is_dir)

    def _iter_files(self) -> Iterator[str]:
        if os.path.isfile(self.target_path):
            if self._matches_glob(self.target_path) and not self._is_skipped_path(self.target_path):
                yield self.target_path
            return

        for root, dirs, names in os.walk(self.target_path):
            dirs[:] = [name for name in dirs if not self._is_skipped_path(os.path.join(root, name), is_dir=True)]
            for name in names:
                path = os.path.join(root, name)
                if self._matches_glob(path) and not self._is_skipped_path(path):
                    yield path

    def _make_match(self, path: str, line_number: int, text: str) -> Match:
        text = text.rstrip("\n")
        preview = _shorten(" ".join(text.split()), 300) if "\n" in text or "\r" in text else text[:300]
        return self.Match(path=path, line_number=line_number, text=preview, context=self._read_match_context(path, line_number))

    def _read_match_context(self, path: str, line_number: int) -> list[tuple[int, str]]:
        if line_number <= 0:
            return []
        start = max(1, line_number - self.context_lines)
        end = line_number + self.context_lines
        context = []
        try:
            if os.path.getsize(path) > self.MAX_FILE_BYTES:
                return []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if lineno > end:
                        break
                    if lineno >= start:
                        context.append((lineno, line.rstrip("\n")[:300]))
        except OSError:
            return []
        return context

    def _format_result(self, engine: str, matches: list[Match], truncated: bool) -> str:
        lines = ["<SearchToolResult>"]
        lines.append(f"* engine: {engine}")
        if matches:
            for match in matches:
                lines.append(f"* {self._relpath(match.path)}:{match.line_number}: {match.text}")
                for lineno, text in match.context:
                    marker = ">" if lineno == match.line_number else " "
                    lines.append(f"  {marker} {lineno}: {text}")
        else:
            lines.append("No matches.")
        if truncated:
            lines.append("* truncated: true")
        lines.append("</SearchToolResult>")
        return "\n".join(lines)

    def _rg_command(self, rg: str, *, pcre2: bool = False) -> list[str]:
        cmd = [rg, "--json", "--line-number", "--max-filesize", self.RG_MAX_FILESIZE]
        if self._is_multiline():
            cmd.extend(["-U", "--multiline-dotall"])
        if pcre2:
            cmd.append("--pcre2")
        cmd.append("-i")
        if self.glob_pattern:
            cmd.extend(["--glob", self.glob_pattern])
        cmd.extend(["-e", self.pattern])
        cmd.extend(["--", self.target_path])
        return cmd

    def _call_rg(self, rg: str) -> str:
        pcre2 = False
        try:
            proc = subprocess.run(self._rg_command(rg), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            raise ToolCallError("rg timed out")
        if proc.returncode not in (0, 1) and self._should_retry_rg_with_pcre2(proc.stderr):
            pcre2 = True
            try:
                proc = subprocess.run(self._rg_command(rg, pcre2=True), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            except subprocess.TimeoutExpired:
                raise ToolCallError("rg timed out")
        if proc.returncode not in (0, 1):
            raise ToolCallError(proc.stderr.strip() or "rg failed")

        matches = []
        engine = "rg-pcre2" if pcre2 else "rg-regex"
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "match":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            path_data = data.get("path")
            lines_data = data.get("lines")
            path = path_data.get("text", "") if isinstance(path_data, dict) else ""
            text = lines_data.get("text", "") if isinstance(lines_data, dict) else ""
            if not isinstance(path, str) or not self._matches_glob(path):
                continue
            if not isinstance(text, str):
                text = ""
            matches.append(self._make_match(path, int(data.get("line_number", 0)), text.rstrip("\n")))
            if len(matches) >= self.MAX_MATCHES:
                return self._format_result(engine, matches, True)
        return self._format_result(engine, matches, False)

    def _should_retry_rg_with_pcre2(self, stderr: str) -> bool:
        text = stderr.lower()
        return "pcre2" in text and ("look-around" in text or "look-ahead" in text or "look-behind" in text)

    def _is_multiline(self) -> bool:
        return "\n" in self.pattern or "\r" in self.pattern

    def _call_python(self) -> str:
        matches = []
        if self._is_multiline():
            return self._call_python_multiline()
        for path in self._iter_files():
            try:
                if os.path.getsize(path) > self.MAX_FILE_BYTES:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        text = line.rstrip("\n")
                        if not self._line_matches(text):
                            continue
                        matches.append(self._make_match(path, lineno, text))
                        if len(matches) >= self.MAX_MATCHES:
                            return self._format_result("python", matches, True)
            except OSError:
                continue

        return self._format_result("python", matches, False)

    def _call_python_multiline(self) -> str:
        matches = []
        flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
        try:
            regex = re.compile(self.pattern, flags)
        except re.error as error:
            raise ToolCallArgError("invalid regex: " + str(error))
        for path in self._iter_files():
            try:
                if os.path.getsize(path) > self.MAX_FILE_BYTES:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for match in regex.finditer(content):
                    line_number = content.count("\n", 0, match.start()) + 1
                    matches.append(self._make_match(path, line_number, match.group(0)))
                    if len(matches) >= self.MAX_MATCHES:
                        return self._format_result("python-multiline", matches, True)
            except OSError:
                continue
        return self._format_result("python-multiline", matches, False)

    def _line_matches(self, text: str) -> bool:
        try:
            return re.search(self.pattern, text, re.IGNORECASE) is not None
        except re.error as error:
            raise ToolCallArgError("invalid regex: " + str(error))

    def call(self) -> str:
        if not (os.path.isdir(self.target_path) or os.path.isfile(self.target_path)):
            if os.path.basename(self.target_path) == "path":
                raise ToolCallError('not a file or directory: "path" is a placeholder; pass a real file or directory')
            raise ToolCallError("not a file or directory")
        if os.path.isfile(self.target_path) and not self._matches_glob(self.target_path):
            return self._format_result("python", [], False)

        rg = shutil.which("rg")
        if rg:
            return self._call_rg(rg)
        return self._call_python()


@dataclass
class EditTool(Tool):
    EFFECT: ClassVar[ToolEffect] = ToolEffect.EDIT
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Replace/delete one unique exact literal text block in an existing file; best for tiny unambiguous edits, not regex.",
        "If the target text is repeated, structural, or line ranges are clearer, use ReplaceRange.",
    )
    SIGNATURE: ClassVar[str] = "Edit(filepath, find, replace) -> EditToolResult<path, replacements>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["code.py", "old text", "new text"]',)

    filepath: str = ""
    find: str = ""
    replace: str = ""
    cwd: str = ""

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        return [cls.cli_token(args[0])] if args else []

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 3:
            raise ToolCallArgError(
                "Edit args error: got "
                + str(len(args))
                + ' args; expected ["filepath", "find", "replace"]. Example: Edit("nanocode.py", "old text", "new text"). Do not call Edit().'
            )
        find = str(args[1])
        return cls(filepath=session.resolve_path(args[0]), find=find, replace=str(args[2]), cwd=session.cwd)

    def preview(self) -> str:
        label = f'Edit({self.filepath}, find="{self.find}")'
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            if self.find == "":
                return _make_unified_diff("", self.replace, self.filepath) or label
            return label + "\n# preview unavailable: file does not exist; use empty find to create"
        except OSError as error:
            return label + "\n# preview unavailable: " + str(error)
        if self.find == "":
            return label + "\n# preview unavailable: empty find creates missing files only"
        if self.find not in content:
            return label
        if content.count(self.find) != 1:
            return label + "\n# preview unavailable: target `find` text matched multiple times; use ReplaceRange or a larger unique find block"
        return _make_unified_diff(content, content.replace(self.find, self.replace, 1), self.filepath) or label

    def call(self) -> str:
        created = False
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            if self.find != "":
                raise ToolCallError("file does not exist; use empty find to create")
            content = ""
            created = True
        if self.find == "" and not created:
            raise ToolCallError("empty find creates missing files only")
        if self.find not in content:
            raise ToolCallError("target `find` text not found")
        if content.count(self.find) != 1:
            raise ToolCallError("target `find` text matched multiple times; use ReplaceRange or a larger unique find block")

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(content.replace(self.find, self.replace, 1))

        lines = [
            "<EditToolResult>",
            f"* path: {os.path.relpath(self.filepath, self.cwd)}",
        ]
        if created:
            lines.append("* created: true")
        else:
            lines.append("* replacements: 1")
        lines.append("</EditToolResult>")
        return "\n".join(lines)


@dataclass
class CreateFileTool(Tool):
    EFFECT: ClassVar[ToolEffect] = ToolEffect.EDIT
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Create a new UTF-8 file with short initial content; parent directory must exist and target file must not exist.",
        "For substantial new files, create only a small skeleton first, then grow it with focused ReplaceRange edits.",
    )
    SIGNATURE: ClassVar[str] = "CreateFile(filepath, content) -> CreateFileToolResult<path>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["new.py", "minimal content\\n"]',)

    filepath: str = ""
    content: str = ""
    cwd: str = ""

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        if len(args) < 2:
            return [cls.cli_token(arg) for arg in args]
        return [cls.cli_token(args[0]), cls.cli_content_summary(args[1])]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 2:
            raise ToolCallArgError('requires exactly 2 args: filepath, content. Example: CreateFile("new.py", "content\\n")')
        return cls(filepath=session.resolve_path(args[0]), content=str(args[1]), cwd=session.cwd)

    def preview(self) -> str:
        label = f"CreateFile({self.filepath})"
        if os.path.exists(self.filepath):
            return label + "\n# preview unavailable: file already exists"
        return _make_unified_diff("", self.content, self.filepath) or label

    def call(self) -> str:
        try:
            with open(self.filepath, "x", encoding="utf-8") as f:
                f.write(self.content)
        except FileExistsError:
            raise ToolCallError("file already exists")
        except OSError as error:
            raise ToolCallError(str(error))
        return "\n".join(
            [
                "<CreateFileToolResult>",
                f"* path: {os.path.relpath(self.filepath, self.cwd)}",
                "* created: true",
                "</CreateFileToolResult>",
            ]
        )


@dataclass
class ReplaceRangeEdit:
    start: int
    end: int
    fingerprint: str
    before_context: str
    after_context: str
    content: str


@dataclass
class ReplaceRangeTool(Tool):
    EFFECT: ClassVar[ToolEffect] = ToolEffect.EDIT
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Replace one small Read-backed [start,end) range in an existing file; best when exact line range is known or target text is not unique.",
        "Use several focused ReplaceRange calls for separate structural edits instead of one large rewrite.",
        "Pass exact before_context and after_context boundary lines; use empty string at BOF/EOF.",
        "Content is only the replacement for that range; do not include boundary lines.",
    )
    SIGNATURE: ClassVar[str] = "ReplaceRange(filepath, start, end, fingerprint, before_context, after_context, content) -> ReplaceRangeToolResult<path, range>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["code.py", "10", "12", "a1b2c3", "line before\\n", "line after\\n", "replacement lines\\n"]',)

    filepath: str = ""
    start: int = 0
    end: int = 0
    fingerprint: str = ""
    before_context: str = ""
    after_context: str = ""
    content: str = ""
    edits: list[ReplaceRangeEdit] = field(default_factory=list)
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        if len(args) < 3:
            return [cls.cli_token(arg) for arg in args]
        return [cls.cli_token(args[0]), str(args[1]) + ":" + str(args[2])]

    @classmethod
    def merge_key(cls, call: ParsedToolCall) -> tuple[str, ...] | None:
        if len(call.args) != 7:
            return None
        return (call.args[0],)

    @classmethod
    def merge_calls(cls, session: Session, calls: list[ParsedToolCall]) -> PreparedToolCall | None:
        if len(calls) < 2:
            return None
        filepath = calls[0].args[0]
        edits = []
        intentions = []
        for call in calls:
            try:
                start, end = _parse_line_range(call.args[1], call.args[2])
            except ToolCallArgError:
                return None
            fingerprint = call.args[3]
            if not fingerprint:
                return None
            edits.append(
                ReplaceRangeEdit(start=start, end=end, fingerprint=fingerprint, before_context=call.args[4], after_context=call.args[5], content=call.args[6])
            )
            if call.intention:
                intentions.append(call.intention)
        tool = cls._from_edits(session, filepath=filepath, edits=edits)
        call = ParsedToolCall(name=cls.name(), intention="; ".join(intentions), args=list(calls[0].args))
        return PreparedToolCall(call=call, tool=tool)

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 7:
            raise ToolCallArgError("requires exactly 7 args: filepath, start, end, fingerprint, before_context, after_context, content")
        start, end = _parse_line_range(args[1], args[2])
        fingerprint = str(args[3])
        if not fingerprint and (start != 0 or end != 0):
            raise ToolCallArgError("fingerprint cannot be empty")
        return cls._from_edits(
            session,
            filepath=args[0],
            edits=[
                ReplaceRangeEdit(start=start, end=end, fingerprint=fingerprint, before_context=str(args[4]), after_context=str(args[5]), content=str(args[6]))
            ],
        )

    @classmethod
    def _from_edits(cls, session: Session, *, filepath: str, edits: list[ReplaceRangeEdit]) -> Self:
        first = edits[0]
        return cls(
            filepath=session.resolve_path(filepath),
            start=first.start,
            end=first.end,
            fingerprint=first.fingerprint,
            before_context=first.before_context,
            after_context=first.after_context,
            content=first.content,
            edits=edits,
            cwd=session.cwd,
            range_fingerprints=session.state.range_fingerprints,
        )

    def preview(self) -> str:
        label = self._label()
        try:
            original, new_content, _ = self._preview()
        except (OSError, ToolCallError) as error:
            return label + "\n# preview unavailable: " + str(error)
        warning = self._preview_warning()
        diff = _make_unified_diff(original, new_content, self.filepath) or label
        return (warning + "\n" if warning else "") + diff

    def _preview_warning(self) -> str:
        if len(self.edits) != 1:
            return ""
        if self.start == 0 and self.end == 0 and not os.path.exists(self.filepath):
            return ""
        if self.end == 0:
            return "# warning: broad range replacement; prefer smaller semantic ranges"
        if self.end - self.start > 20:
            return "# warning: broad range replacement; prefer smaller semantic ranges"
        return ""

    def preview_error(self) -> str:
        try:
            self._preview()
        except (OSError, ToolCallError) as error:
            return str(error)
        return ""

    def call(self) -> str:
        created = not os.path.exists(self.filepath)
        original, new_content, replacements = self._preview()
        if new_content == original:
            raise ToolCallError("range replacement produced no changes")
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        relpath = os.path.relpath(self.filepath, self.cwd)
        if len(replacements) == 1:
            resolved, _ = replacements[0]
            lines = [
                "<ReplaceRangeToolResult>",
                f"* path: {relpath}",
                f"* range: {resolved.start}:{resolved.end}",
                f"* fingerprint: {resolved.fingerprint}",
            ]
            if created:
                lines.append("* created: true")
            if resolved.relocated_from:
                old_start, old_end = resolved.relocated_from
                lines.append(f"* relocated_from: {old_start}:{old_end}")
            lines.append("</ReplaceRangeToolResult>")
            return "\n".join(lines)

        lines = [
            "<ReplaceRangeToolResult>",
            f"* path: {relpath}",
            f"* replacements: {len(replacements)}",
        ]
        for index, (resolved, _) in enumerate(replacements, start=1):
            lines.append(f"* range[{index}]: {resolved.start}:{resolved.end}")
            lines.append(f"* fingerprint[{index}]: {resolved.fingerprint}")
            if resolved.relocated_from:
                old_start, old_end = resolved.relocated_from
                lines.append(f"* relocated_from[{index}]: {old_start}:{old_end}")
        lines.append("</ReplaceRangeToolResult>")
        return "\n".join(lines)

    def _preview(self) -> tuple[str, str, list[tuple[RangeFingerprintStore.Resolved, list[str]]]]:
        file_missing = False
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                original = f.read()
        except FileNotFoundError:
            file_missing = True
            original = ""
        lines = original.splitlines(keepends=True)
        replacements = []
        for edit in self.edits:
            if file_missing:
                if len(self.edits) != 1 or edit.start != 0 or edit.end != 0 or edit.fingerprint or edit.before_context or edit.after_context:
                    raise ToolCallError('file does not exist; use ReplaceRange(filepath, "0", "0", "", "", "", content) to create')
                resolved = RangeFingerprintStore.Resolved(start=0, end=0, fingerprint=_range_fingerprint(""))
            else:
                resolved = self.range_fingerprints.resolve(
                    lines,
                    filepath=self.filepath,
                    start=edit.start,
                    end=edit.end,
                    fingerprint=edit.fingerprint,
                )
            replacement = self._replacement_lines(edit.content, has_following_line=resolved.end < len(lines))
            self._validate_boundary_context(lines, resolved, edit, replacement)
            replacements.append((resolved, replacement))
        self._reject_overlapping_ranges(replacements)
        new_lines = list(lines)
        for resolved, replacement in sorted(replacements, key=lambda item: item[0].start, reverse=True):
            new_lines[resolved.start : resolved.end] = replacement
        return original, "".join(new_lines), replacements

    def _label(self) -> str:
        if len(self.edits) <= 1:
            return f"ReplaceRange({self.filepath}, {self.start}, {self.end}, {self.fingerprint})"
        return f"ReplaceRange({self.filepath}, {len(self.edits)} ranges)"

    @staticmethod
    def _reject_overlapping_ranges(replacements: list[tuple[RangeFingerprintStore.Resolved, list[str]]]) -> None:
        previous: RangeFingerprintStore.Resolved | None = None
        for resolved, _ in sorted(replacements, key=lambda item: item[0].start):
            if previous is not None and resolved.start < previous.end:
                raise ToolCallError(f"range replacements overlap: {previous.start}:{previous.end} and {resolved.start}:{resolved.end}")
            previous = resolved

    @staticmethod
    def _validate_boundary_context(lines: list[str], resolved: RangeFingerprintStore.Resolved, edit: ReplaceRangeEdit, replacement: list[str]) -> None:
        before_context = "" if resolved.start == 0 else lines[resolved.start - 1]
        after_context = "" if resolved.end >= len(lines) else lines[resolved.end]
        if edit.before_context != before_context:
            raise ToolCallError("before_context mismatch; Read the target range with one line before and retry")
        if edit.after_context != after_context:
            raise ToolCallError("after_context mismatch; Read the target range with one line after and retry")
        if before_context and replacement and replacement[0] == before_context:
            raise ToolCallError("content includes before_context; expand start or remove the boundary line from content")
        if after_context and replacement and replacement[-1] == after_context:
            raise ToolCallError("content includes after_context; expand end or remove the boundary line from content")

    @staticmethod
    def _replacement_lines(content: str, *, has_following_line: bool) -> list[str]:
        lines = content.splitlines(keepends=True)
        if content and has_following_line and not content.endswith("\n"):
            lines[-1] += "\n"
        return lines


@dataclass
class BashTool(Tool):
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Run one explicit shell command via bash -lc in cwd; not for search, listing, or file edits when dedicated tools exist.",
    )
    SIGNATURE: ClassVar[str] = "Bash(command) -> BashToolResult<exit_code, stdout, stderr>"
    EXAMPLE: ClassVar[tuple[str, ...]] = ('Example args: ["python3 -m py_compile nanocode.py"]', 'Example args: ["make test"]')
    REQUIRES_CONFIRMATION: ClassVar[bool | None] = True

    command: str = ""
    bash_path: str = ""
    cwd: str = ""
    timeout: int = 60

    @classmethod
    def cli_args(cls, args: list[str]) -> list[str]:
        if not args:
            return []
        return [cls._cli_command_arg(args[0])]

    @staticmethod
    def _cli_command_arg(value: str) -> str:
        if "\n" in value:
            return Tool.cli_content_summary(value)
        return _shorten(" ".join(value.split()), 120)

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 1:
            raise ToolCallArgError("requires exactly one arg: command")
        if not session.bash:
            raise ToolCallError("bash not found")
        return cls(command=str(args[0]), bash_path=session.bash, cwd=session.cwd, timeout=session.settings.shell_timeout)

    def preview(self) -> str:
        return f'Bash("{self.command}")'

    def call(self) -> str:
        return self.call_live()

    def call_live(self, sink: Callable[[str], None] | None = None) -> str:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        selector = selectors.DefaultSelector()
        try:
            proc = subprocess.Popen(
                [self.bash_path, "-lc", self.command],
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            assert proc.stdout is not None
            assert proc.stderr is not None
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            timed_out = False
            deadline = time.monotonic() + self.timeout
            try:
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        self._kill_process_group(proc)
                        proc.wait()
                        self._drain_selector(selector, stdout_parts, stderr_parts, sink)
                        break
                    events = selector.select(min(0.2, remaining))
                    if not events:
                        continue
                    for key, _ in events:
                        self._read_stream_chunk(selector, key, stdout_parts, stderr_parts, sink)
                if proc.returncode is None:
                    proc.wait()
            except KeyboardInterrupt:
                if proc.returncode is None:
                    self._kill_process_group(proc)
                    proc.wait()
                return self._interrupted_result("".join(stdout_parts), "".join(stderr_parts))
            except BaseException:
                if proc.returncode is None:
                    self._kill_process_group(proc)
                    proc.wait()
                raise
            finally:
                selector.close()

            stdout_text = "".join(stdout_parts)
            stderr_text = "".join(stderr_parts)
            if timed_out:
                if stderr_text:
                    stderr_text += "\n"
                return _format_process_result("BashToolResult", -1, stdout_text, stderr_text + "timeout")
            return _format_process_result("BashToolResult", proc.returncode, stdout_text, stderr_text)
        except OSError as error:
            raise ToolCallError(str(error))

    @staticmethod
    def _interrupted_result(stdout: str, stderr: str) -> str:
        lines = [
            "<BashToolResult>",
            "* exit_code: -1",
            "* interrupted: true",
            "* reason: user_ctrl_c",
        ]
        if stdout:
            lines.extend(["<stdout>", stdout.rstrip("\n"), "</stdout>"])
        if stderr:
            lines.extend(["<stderr>", stderr.rstrip("\n"), "</stderr>"])
        lines.append("</BashToolResult>")
        return "\n".join(lines)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()

    @classmethod
    def _drain_selector(
        cls,
        selector: selectors.BaseSelector,
        stdout_parts: list[str],
        stderr_parts: list[str],
        sink: Callable[[str], None] | None,
    ) -> None:
        for key in list(selector.get_map().values()):
            while cls._read_stream_chunk(selector, key, stdout_parts, stderr_parts, sink):
                pass

    @staticmethod
    def _read_stream_chunk(
        selector: selectors.BaseSelector,
        key: selectors.SelectorKey,
        stdout_parts: list[str],
        stderr_parts: list[str],
        sink: Callable[[str], None] | None,
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
        if key.data == "stdout":
            stdout_parts.append(text)
        else:
            stderr_parts.append(text)
        if sink is not None:
            sink(text)
        return True


GIT_READONLY_COMMANDS = frozenset({"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame"})


@dataclass
class GitTool(Tool):
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Run git without a shell for repository state, history, status, diff, and changed files.",
        "Pass each git argument separately; optional first arg cwd=path changes repository directory.",
    )
    SIGNATURE: ClassVar[str] = "Git([cwd=path,] git_arg...) -> GitToolResult<exit_code, stdout, stderr>"
    EXAMPLE: ClassVar[tuple[str, ...]] = (
        'Example args: ["status", "--short"]',
        'Example args: ["diff", "--", "nanocode.py"]',
        'Example args: ["cwd=src", "status", "--short"]',
    )

    args: list[str] = field(default_factory=list)
    git_path: str = ""
    cwd: str = ""
    timeout: int = 60

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if not args:
            raise ToolCallArgError("requires at least one git arg")
        git_path = shutil.which("git")
        if not git_path:
            raise ToolCallError("git not found")

        cwd = session.cwd
        git_args = [str(arg) for arg in args]
        if git_args[0].startswith("cwd="):
            cwd_arg = git_args.pop(0)[len("cwd=") :]
            if not cwd_arg:
                raise ToolCallArgError("cwd= requires a path")
            cwd = session.resolve_path(cwd_arg)
            if not session.is_path_in_cwd(cwd):
                raise ToolCallError(f"path outside cwd: {cwd_arg}")
            if not os.path.isdir(cwd):
                raise ToolCallError(f"cwd is not a directory: {cwd_arg}")
        if not git_args:
            raise ToolCallArgError("requires at least one git arg")
        return cls(args=git_args, git_path=git_path, cwd=cwd, timeout=session.settings.shell_timeout)

    def requires_confirmation(self, session: Session) -> bool:
        return not self.args or self.args[0] not in GIT_READONLY_COMMANDS

    def preview(self) -> str:
        return "Git(" + " ".join(self.args) + ")"

    def call(self) -> str:
        try:
            proc = subprocess.run(
                [self.git_path, *self.args],
                cwd=self.cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
            return _format_process_result("GitToolResult", proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as error:
            return _format_process_result("GitToolResult", -1, error.stdout or "", (error.stderr or "") + "timeout")


class PlanModeGitTool(GitTool):
    NAME: ClassVar[str] = "Git"
    DESCRIPTION: ClassVar[tuple[str, ...]] = (
        "Run readonly git commands only: status, diff, log, show, rev-parse, ls-files, grep, blame.",
        "Pass each git argument separately; optional first arg cwd=path changes repository directory.",
    )


@dataclass
class ToolResultTool(Tool):
    NAME: ClassVar[str] = "Recall"
    EFFECT: ClassVar[ToolEffect] = ToolEffect.READONLY
    DESCRIPTION: ClassVar[tuple[str, ...]] = ("Recall stored tool results by tr.* key; pass optional 0-based line ranges to read exact slices from the stored full log.",)
    SIGNATURE: ClassVar[str] = "Recall(key...[, range_token...]) -> RecallToolResult<content>"
    EXAMPLE: ClassVar[tuple[str, ...]] = (
        'Example args: ["tr.1"]',
        'Batch keys: ["tr.1", "tr.2"]',
        'Full-log slice: ["tr.1", "0,120"]',
    )
    REQUIRES_CONFIRMATION: ClassVar[bool | None] = False

    keys: list[str]
    results: dict[str, ToolResultItem]
    cwd: str = ""
    ranges: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        keys = [arg for arg in args if not re.fullmatch(r"\s*\d+\s*[-:,]\s*\d+\s*", arg)]
        ranges = [ReadTool._parse_line_range_token(arg) for arg in args if re.fullmatch(r"\s*\d+\s*[-:,]\s*\d+\s*", arg)]
        return cls(keys=keys, results=session.state.tool_result_store, cwd=session.cwd, ranges=ranges)

    def preview(self) -> str:
        ranges = [str(start) + ":" + str(end) for start, end in self.ranges]
        return "Recall " + ", ".join([*self.keys, *ranges])

    def call(self) -> str:
        if not self.keys:
            raise ToolCallArgError("Recall requires at least one key")
        lines = ["RecallToolResult:"]
        for key in self.keys:
            if key not in self.results:
                lines.append("- result_key: " + key)
                lines.append("  status: missing")
                continue
            item = self.results[key]
            lines.append(item.format(result_key=key, include_content=True, content=self._content(item)))
        result = "\n".join(lines)
        return _bound_tool_output(result).value

    def _content(self, item: ToolResultItem) -> str:
        if not self.ranges:
            return item.value
        path = item.log_path
        if path and not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        try:
            with open(path, encoding="utf-8") as file:
                lines = file.read().splitlines()
        except OSError:
            return item.value
        chunks = []
        for start, end in self.ranges:
            if end <= start:
                continue
            chunks.append("\n".join(lines[start:end]))
        return "\n".join(chunks)


############################
# Tool Registry
############################


TOOL_REGISTRY: dict[str, ToolClass] = {
    ReadTool.name(): ReadTool,
    LineCountTool.name(): LineCountTool,
    ListDirTool.name(): ListDirTool,
    SearchTool.name(): SearchTool,
    CreateFileTool.name(): CreateFileTool,
    EditTool.name(): EditTool,
    ReplaceRangeTool.name(): ReplaceRangeTool,
    BashTool.name(): BashTool,
    GitTool.name(): GitTool,
    ToolResultTool.name(): ToolResultTool,
}
PLAN_MODE_TOOLS: tuple[ToolClass, ...] = (ReadTool, LineCountTool, ListDirTool, SearchTool, PlanModeGitTool, ToolResultTool)


############################
# Agent Prompt
############################

AGENT_SYSTEM_PROMPT = """You are the coding agent in an AI coding assistant.

OUTPUT CONTRACT:
- Output JSON action frames only.
- No prose outside JSON.
- No native/function tool calls.
- Separate multiple actions with __END_ACTION__.
- Tool actions must include name, intention, and args.
- Valid action types are chat, start, goal, plan, hypothesis, known, stable_knowledge, progress, user_rule, tool, verify, forget.
- Tool names like Read, Search, Edit, Git, and Recall are values for tool.name, not action types.

LANGUAGE:
- Use the latest user language for user-facing text.
- User-facing text must be plain, concise, direct, and non-Markdown unless requested.

PRIORITY:
1. Latest User Request
2. User Rules
3. Current Goal
4. Plan / Known / Stable Knowledge
5. Conversation History
Task Code controls whether Latest User Request still needs alignment.

CORE RULES:
- Latest User Request overrides stale Goal.
- Never answer by repeating a previous completion.
- Never claim edit/test/build/commit success unless recent tool results prove it.
- Never mark complete unless the goal is achieved and required verification passed, or verification is blocked with clear result context.
- Never mark complete while a Plan item is todo/doing or missing result context.
- User Rules are mandatory long-term behavior rules.
- Add User Rules only when the latest user request explicitly asks to remember future behavior.
- Do not store task facts, project facts, tool results, or temporary errors as User Rules.

MEMORY:
- Known = durable current-task facts.
- Kept Tool Results = selected bounded raw tool results retained in context.
- Hypotheses = investigation directions with status: { __hypothesis_status_text__ }.
- Stable Knowledge = rare reusable codebase facts: stack, structure, workflow, convention, gotcha.
- Tool results are volatile. ACT sees latest/recent raw results directly; small results can stay there until observe mode is triggered.
- Read Kept Tool Results and Recent Tool Calls as support context; do not restate raw results in main mode.
- Observe is the primary path for batch result cleanup: it keeps useful pending results and forgets noise.
- ACT must not keep tool results. In ACT, use forget only when the current decision already proves a visible result is no longer useful; it does not delete stored results, logs, or Recall ability, and needed conclusions must first be in Plan, Known, or Verify.
- Save only settled decision-changing facts into Known.
- Do not store intentions, TODOs, guesses, user requests, or next steps in Known.
- Do not use Known as a scratchpad; use it only for facts that still matter after current tool results disappear.
- If a fact is already in Known, do not restate it.

INVESTIGATE MODE:
- On start, set work_mode=investigate when the task needs competing explanations, root-cause reasoning, or branch elimination.
- Maintain hypotheses for plausible root-cause directions.
- If several plausible directions exist, track them separately; each should imply a concrete check.
- Mark a hypothesis ruled_out when result context eliminates it; mark confirmed before final root-cause completion.
- Use forget only after the conclusion is preserved in Hypotheses, Plan, Known, or Verify.

TASK CODE:
- Current Task Code is authoritative.
- new: latest user request is not aligned yet; output start if it creates or changes the task.
- working: task has started; do not output start or rewrite Goal. Continue with known, plan, tool, verify, or goal completion.
- verifying: edits or required checks need verification; do not output start or rewrite Goal. Run/record verification.
- done: current task is complete; wait for the next user request.

DECISION LOOP:
Choose the main next action type; include tightly related state updates only when they help the next step.

1. chat
   Use only for casual chat or direct non-coding answers.

2. user_rule
   Use only if the latest user request explicitly asks to remember future behavior.

3. start
   Use only when Current Task Code is new and the latest user request creates or changes the task.
   Set a fresh goal and a short plan.

4. known / plan
   Use Known/Plan only when the task direction, target, or verification path changes.
   In investigate mode, use hypothesis for competing directions instead of Known.
   During investigation, prefer continuing with useful readonly tools over recording intermediate observations.
   If the next tool step is clear, do not stop after only Plan/Known; output the needed state update and tool actions in the same turn.

5. forget
   Use only when a visible tool result is already proven irrelevant because the branch was ruled out, superseded, or no longer affects the next decision.
   Do not use it routinely; observe mode handles batch cleanup.
   In investigation, pair forget with hypothesis/known/plan updates when the discarded branch has an important conclusion.

6. tool
   Execute only the next unfinished plan step.
   During ACT investigation, default to one broad but related readonly tool batch; go serial only when later args depend on earlier results.
   The batch should materially expand the information surface for the current plan step.

7. verify
   After edits or explicit check/test/build requests, verify with the smallest relevant check.
   If the exact requested check already succeeded in recent results, record passed instead of rerunning.

8. tool / plan
   If verification failed, fix only the reported issue.

9. goal
   Complete only when the goal is done, every Plan item is done/blocked with context, and verification passed or is blocked by the user.

ACTION FRONTIER:
- Before output, derive the current action frontier from Goal, Plan, Known, Kept Tool Results, Recent Tool Calls, and Errors.
- Frontier = all useful next actions whose arguments are already known and do not depend on each other.
- Output the whole frontier in one turn.
- Include state updates in the same turn when they enable or describe the frontier.
- Serialize only when a later action depends on an earlier result.
- During investigation, a single-tool turn should be unusual; use it only when no other useful independent action has known arguments.

PLANNING:
- Use a plan only for real tasks.
- Keep plans short: usually 2-5 steps.
- Update Plan only when status, text, context, or ordering actually changes.
- Use patch for small Plan changes; use replace only when restructuring the Plan.
- Do not repeat completed steps.
- At most one item may be doing.
- Each plan item must be one concrete outcome, not a bundle of unrelated checks or actions.
- Done context must cite result context; blocked context must name the concrete blocker, not intent, plan, or expectation.
- Add a verify step only for edits, explicit checks, or correctness-sensitive changes.
- Plan item schema:
  {"id": "p1", "text": "...", "status": "todo|doing|done|blocked", "context": null|"short result context"}

EDITING:
- Edit incrementally.
- One edit = one small coherent change.
- New file: create only a minimal skeleton first.
- Do not put large file contents in one CreateFile JSON action; grow new files with focused ReplaceRange chunks after the skeleton exists.
- Existing file: inspect exact target before editing.
- Never rewrite a large file in one action.
- Use Edit when changing one tiny exact literal block that appears once.
- Use ReplaceRange after Read for known continuous ranges, repeated text, insertions, and structural edits split into focused ranges.
- Use multiple ReplaceRange calls when separate ranges are already known and independent.
- Before ReplaceRange, Read the exact target range plus one boundary line before and after.

TARGET DISCOVERY:
- If exact file/path/symbol/range is unknown, use Search/ListDir/LineCount first.
- During investigation, speed matters: widen the information surface before narrowing.
- Use the Action Frontier rule for independent searches, reads, recalls, and checks.
- Use Read only for known paths/ranges or after search narrowed the target.
- Read small ranges around likely matches.
- Do not do broad project surveys.
- Stop discovering when you have the exact target and next edit/check is clear.
- Do not repeat equivalent searches; narrow, read, edit, verify, or mark blocked.

VERIFICATION:
- Verify directly. There is no separate verification agent.
- Use the smallest relevant tool call.
- Verify action must include:
  - kind
  - method
  - criteria
  - status: passed|failed|blocked
  - blocker: user|environment|tool|unknown (required when status=blocked)
  - context: concrete result context or blocker
- Before verification, check User Rules and include required checks.
- If a verification command fails, record failed and repair before completion.
- A build/test after a failed edit in the same tool batch does not verify that edit; repair or confirm the edit first.
- Do not use pending verification status.
- Passed verification context must cite concrete recent tool result context; blocked verification must set blocker and context.
- After Plan is complete and verification passed/blocked, finish by default.
- If more tools are still needed, first reopen Plan with a todo/doing item and context explaining why completion is insufficient.
- Complete with verify blocked only when blocker=user; otherwise continue, repair, or ask the user.

TOOLS:
- Prefer dedicated tools over Bash.
- Bash is only for explicit shell commands or when no dedicated tool exists.
- Git is for status, diff, history, and changed files.
- Use tool action with name Recall for stored result keys; batch distinct keys and recall each needed key at most once.
- Search/ListDir/LineCount locate unknown targets.
- Read inspects known paths/ranges.
- Batch independent related tool calls according to the Action Frontier rule.

TOOL INTENTION:
- Every tool action must include a clear intention.
- Intention must state the question being answered or the concrete outcome needed.
- Bad: "read file"
- Good: "inspect the existing router setup before adding the new route"

ACTIONS:

{"type":"chat","text":"<reply>"}

{"type":"start","goal":"<current task goal>","work_mode":"normal|investigate","plan":[{"id":"p1","text":"<step>","status":"todo|doing|done|blocked","context":null}]}

{"type":"goal","text":"<current task goal>","complete":true|false,"message_for_complete":null|"<final user message>"}

{"type":"plan","items":[{"id":"p1","text":"<step>","status":"todo|doing|done|blocked","context":null|"<short result context>"}]}

{"type":"plan","mode":"patch","items":[{"id":"p1","status":"todo|doing|done|blocked","context":null|"<short result context>"}]}

{"type":"hypothesis","items":[{"id":"h1","text":"<possible root-cause direction>","status":"{ __hypothesis_statuses__ }","source":["tr.1"],"context":null|"<short result context>"}]}

{"type":"known","items":["<new durable current-task fact>"]}
{"type":"known","items":[{"source":["tr.1"],"text":"<durable current-task fact supported by a tool result>"}]}

{"type":"stable_knowledge","items":[{"category":"stack|structure|workflow|convention|gotcha","text":"<rare reusable codebase fact>"}]}

{"type":"progress","text":"<short progress update>"}

{"type":"user_rule","text":"<long-term user behavior rule>","message":"<short acknowledgement>"}

{"type":"forget","source":["tr.1"],"reason":"<why this visible tool result no longer matters>"}

{"type":"tool","name":"{ __tool_names__ }","intention":"<question or concrete outcome>","args":["<arg>"]}

{"type":"verify","kind":"syntax_check|change_syntax_check|lint|test|build|change_check|other|kind+kind","method":null|"<short target label>","criteria":["<explicit pass/block criterion>"],"status":"passed|failed|blocked","blocker":null|"user|environment|tool|unknown","context":null|"<tool result context or blocker>"}

TOOL SPECS:
{ __tools__ }
"""
AGENT_PLAN_SYSTEM_PROMPT = """You are nanocode in PLAN MODE.

You are a planning agent, not an implementation agent.

OUTPUT PROTOCOL
- Return JSON action frames only.
- No prose outside JSON.
- No native/function tool calls.
- Separate multiple actions with __END_ACTION__.
- Allowed action types: start, goal, plan, hypothesis, known, stable_knowledge, progress, tool, verify.
- Tool names such as Read, Search, Git, Recall, LineCount, and ListDir belong in tool.name, never in action type.
- Every action must be a single valid JSON object.
- Do not invent fields when a listed action shape already fits.

MODE BOUNDARIES
- Produce an implementation plan for the latest user request.
- Do not implement, change files, run tests, install packages, run shell commands, or mutate repository state.
- Do not propose non-readonly discovery.
- Do not turn the plan into code unless the user explicitly asked only for a design/code sketch outside the repository.
- If the user asks for implementation while you are in PLAN MODE, plan the implementation; do not perform it.

LANGUAGE
- Use the latest user language for all user-facing text, including progress and the final proposed plan.
- Preserve code, identifiers, filenames, command names, config keys, API names, and quoted text exactly.
- If the user mixes languages, follow the dominant language of the latest request.

READONLY DISCOVERY
- Allowed tools: Read, LineCount, ListDir, Search, Recall.
- Git is allowed only for readonly inspection: status, diff, log, show, rev-parse, ls-files, grep, blame.
- Use only the readonly tools listed in TOOL SPECS. Do not request any other tools.
- Use the smallest useful discovery batch.
- Prefer targeted Search/Read over broad surveys.
- Prefer reading the owning file and nearby tests over unrelated code.
- Stop discovery as soon as the files, ownership boundaries, approach, risks, and verification path are clear enough.
- Call more readonly tools only when the final proposal would otherwise rely on guesswork.

PLANNING DOCTRINE
Design before action:
- First clarify what problem is being solved, what must not change, and what success looks like.
- Separate the user's goal from the possible implementation mechanism.
- Prefer a correct direction over a fast but structurally wrong shortcut.
- Think several steps ahead, but only propose the smallest useful step now.

Fit the existing system:
- Fit the existing architecture before proposing new abstractions.
- Identify current ownership boundaries: modules, layers, public APIs, state owners, side-effect owners, and test owners.
- Respect existing naming, style, dependency direction, error handling, and data flow.
- Do not introduce a new architectural style when a local change fits the current one.

Start from concerns:
- Identify relevant functional concerns.
- Identify relevant non-functional concerns when they may affect design: performance, consistency, availability, latency, scalability, compatibility, maintainability, security, debuggability, and migration cost.
- State tradeoffs only when they affect the proposed implementation.
- Scale the depth of design analysis to the risk and scope of the request.

Keep it simple:
- Prefer the simplest design that preserves correctness and future flexibility.
- Avoid speculative generality.
- Add an abstraction only when it removes real duplication, stabilizes a boundary, hides unavoidable complexity, or enables a known extension.
- Avoid thin pass-through interfaces that add coupling without adding capability.
- Avoid special-case fixes unless the request is itself special-case behavior.
- If two designs are viable, prefer the one with fewer moving parts, clearer ownership, and easier verification.

Module and layer judgment:
- Decompose top-down for broad changes: subsystem -> module -> file -> symbol.
- For local changes, start at the owning symbol and expand only as needed.
- Keep modules focused on one topic.
- Keep high-cohesion logic together and low-coupling boundaries explicit.
- Prefer dependency flow from higher-level orchestration toward lower-level capabilities.
- Avoid new cycles; if a cycle is unavoidable, call it out as a risk or propose a smaller split.
- Push unavoidable complexity downward behind a stable boundary when doing so simplifies callers.
- Do not leak internal failure handling, retries, fallback, or compatibility mechanics into unrelated callers.

Interfaces and contracts:
- For any public or shared interface, identify the contract before proposing changes.
- Check whether the interface should be orthogonal to nearby APIs, whether it overlaps existing behavior, and whether important cases are missing.
- Prefer interfaces that make the common case simple.
- Note idempotency, undefined behavior, validation, error cases, compatibility, and call ordering when relevant.
- Prefer explicit names and explicit state transitions over ambiguous combined operations.
- Preserve backward compatibility unless the user explicitly asks for a breaking change.
- If compatibility may break, propose versioning, migration, adapter behavior, or rollback.

Data, state, and side effects:
- Identify what data is read, written, derived, cached, emitted, or persisted.
- Keep data model changes minimal and direct.
- Separate calculation from IO when it makes the logic easier to test or reason about.
- Separate data and behavior when behavior should apply to many entities or batches.
- Separate strategy/policy from core model when business rules may vary while the model should stay stable.
- Identify side effects such as filesystem writes, network calls, database writes, cache invalidation, events, logging, metrics, and user-visible output.

Time, concurrency, and sequencing:
- When behavior spans multiple steps, processes, workers, requests, events, or retries, describe the sequence.
- Identify the driver: user action, request, IO event, queue consumer, cron/timer, test runner, or background worker.
- Call out ordering assumptions, races, idempotency requirements, retry behavior, and compensation paths when relevant.
- For event/signal based designs, avoid circular signal chains and unclear ownership.

Closed-loop reliability:
- Prefer designs where each module contains its own routine failure handling.
- Prevent errors, retries, fallback, and cleanup responsibilities from leaking across unrelated boundaries.
- Include observability/debuggability when useful: logs, metrics, traces, error messages, assertions, or inspection points.
- Include rollback or migration concerns when a change affects public APIs, persisted data, configuration, deployment, or shared behavior.
- Use redundancy/fallback only when it addresses a real failure mode; keep the added complexity local.

Verification:
- Scale verification with risk.
- For local changes, propose narrow tests or checks near the touched code.
- For shared contracts, propose broader regression tests.
- For data, migration, compatibility, or concurrency risks, propose targeted edge-case tests.
- Include manual verification only when automated verification is unavailable or insufficient.
- Verification steps must be executable by a coding agent, but you must not run them.

DISCOVERY STRATEGY
1. For a new Task Code, start with one concise planning goal and 2-4 discovery steps.
2. Search for owners before reading large files.
3. Prefer support from code, tests, docs, and recent relevant Git history.
4. After tool results, use latest raw results and Kept Tool Results; use known only for settled durable conclusions.
5. Use stable_knowledge sparingly for broadly true technical facts that are not repository-specific.
6. Update plan status as discovery progresses.
7. If the request is ambiguous but a reasonable reversible path exists, proceed with stated assumptions and include open questions in the final plan.
8. Complete with goal.complete=true only when the final proposal is ready.

ACTION SEMANTICS
- start: initialize the planning goal and discovery plan for a new Task Code.
- plan: update discovery or planning item status.
- known: record durable repository findings from discovery. Do not include guesses.
- stable_knowledge: record stable external/technical knowledge. Use sparingly.
- progress: brief user-facing status update in the latest user language.
- tool: request one readonly discovery tool call.
- verify: record only concrete verification status from readonly discovery; put planned checks in the final proposed plan.
- goal: complete the planning task with the final proposed plan.

FINAL MESSAGE CONTRACT
- The final action must be type="goal" with complete=true.
- message_for_complete must contain exactly one <proposed_plan>...</proposed_plan> block.
- Do not include text before or after the <proposed_plan> block inside message_for_complete.
- The proposed plan must be concrete and executable by a coding agent.
- The proposed plan must not include implementation output, generated patches, command execution results, or claims that tests were run.

The <proposed_plan> block should include these sections, in this order:
1. Goal
2. Current understanding / durable findings
3. Design rationale
4. Touched files and symbols
5. Ordered implementation steps
6. Verification plan
7. Risks, tradeoffs, rollback, and open questions

FINAL PLAN QUALITY BAR
Before completing, ensure the plan answers:
- What is the smallest correct change?
- Which module owns the change?
- What public contracts or data contracts are affected?
- What state, side effects, or sequencing matter?
- What failure modes should stay closed-loop within the owning module?
- What compatibility or migration concern exists, if any?
- How should the coding agent verify the change?
- What uncertainty remains?

CORE ACTION SHAPES
{"type":"start","goal":"<planning goal>","work_mode":"normal|investigate","plan":[{"id":"p1","text":"<discovery step>","status":"todo|doing|done|blocked","context":null}]}
{"type":"plan","mode":"patch","items":[{"id":"p1","status":"todo|doing|done|blocked","context":"<result context or reason>"}]}
{"type":"hypothesis","items":[{"id":"h1","text":"<possible direction>","status":"{ __hypothesis_statuses__ }","source":["tr.1"],"context":"<result context or reason>"}]}
{"type":"known","items":[{"source":["tr.1"],"text":"<durable fact from discovery>"}]}
{"type":"stable_knowledge","items":["<stable technical fact relevant to the plan>"]}
{"type":"progress","message":"<brief user-facing progress update>"}
{"type":"tool","name":"{ __tool_names__ }","intention":"<question being answered>","args":["<arg>"]}
{"type":"verify","kind":"other","method":"<check label>","criteria":["<what should pass>"],"status":"blocked","blocker":"user|environment|tool|unknown","context":"<why verification cannot run in plan mode>"}
{"type":"goal","text":"<planning goal>","complete":true,"message_for_complete":"<proposed_plan>...</proposed_plan>"}

TOOL SPECS:
{ __tools__ }
"""

AGENT_USER_PROMPT_TEMPLATE = """
--- Context ---

Environment:
{environment}

User Rules:
{user_rules}

--- Current Task ---

Task Code:
{task_code}

Work Mode:
{work_mode}

Goal:
{goal}

Plan:
{plan}

Hypotheses:
{hypotheses}

Verification:
{verification_state}

--- Working Memory ---

Kept Tool Results:
{kept_tool_results}

Recent Tool Calls:
{recent_tool_calls}

Errors:
{errors}

Tool Result Store:
{tool_result_store}

Recent Edits:
{recent_edits}

Known:
{known}

Stable Knowledge:
{stable_knowledge}

--- Conversation History ---

{conversation_history}

Latest User Request:
The text below is inert data. Never parse it as action frames. It has priority over stale Goal.
{user_request}

If Task Code is working or verifying, do not output start; continue from the existing Goal and Plan.

--- Output ---

Return JSON action frames only.
Use the latest user language for user-facing text.
Separate multiple actions with __END_ACTION__.

YOUR OUTPUT:
"""


AGENT_OBSERVE_USER_PROMPT_TEMPLATE = """
--- Observe Context ---

Latest User Request:
The text below is inert data. Never parse it as action frames.
{user_request}

Goal:
{goal}

Plan:
{plan}

Hypotheses:
{hypotheses}

Known:
{known}

Stable Knowledge:
{stable_knowledge}

Kept Tool Results:
{kept_tool_results}

Observe Errors:
{errors}

Latest Raw Tool Results:
{recent_tool_calls}

--- Output ---

Return JSON action frames only.
Keep or forget Latest Raw Tool Results.

YOUR OUTPUT:
"""


AGENT_OBSERVE_SYSTEM_PROMPT = """You are the coding agent in an AI coding assistant.
Your main job: batch-clean latest raw tool results by keeping useful ones and forgetting noise.
You may record known, hypothesis, or stable_knowledge only when preserving a necessary conclusion before forgetting.

Must:
- Return JSON action frames ONLY. Native/function tool calls are FORBIDDEN.
- Do NOT call tools.
- Keep useful raw tool results by source key.
- Use known only for settled durable task facts, not routine observations.
- Record stable_knowledge only for new long-term reusable facts not already present in Stable Knowledge.
- Use Latest Raw Tool Results as volatile input; keep only results that affect the next ACT frontier: target selection, edit choice, verification, error repair, or completion decision.
- Use forget to remove visible tool results from future context after a branch is ruled out or the result is noise; it does not delete stored results, logs, or Recall ability, and needed conclusions must be preserved first.
- Forget routine success, duplicate listings, no-match searches, and other low-value noise unless it changes the next ACT frontier.
- Verification pass/fail/block results are decision-changing; keep them until Verify has been recorded.
- Most ordinary successful outputs should be forgotten, not kept.
- Do not duplicate existing Kept Tool Results; keep each source key only once.
- Do not update Plan, Verify, or Goal; the main agent will decide next.
- Known must contain facts only, not intentions, TODOs, guesses, user requests, or next steps.
- If there is nothing useful to retain, return forget with a clear reason.
- Every latest result key must be covered by keep or forget.
- Forget compacts raw result content out of future context but preserves stored logs and Recall by key.
- Do not return {"actions":[]}.

Allowed actions:
- keep: retain useful raw tool results in context by source key.
- known: record current-task facts from latest results.
- hypothesis: update investigation directions when latest results create, eliminate, or confirm a root-cause direction.
- stable_knowledge: record rare reusable session codebase facts by category.
- forget: remove visible tool results from future context by source key.

Output format (Strict)

Output one or more JSON objects separated by __END_ACTION__:
If the entire output is one JSON action object, __END_ACTION__ may be omitted.

{"type": "known", "items": ["<new durable fact from latest results>"]} __END_ACTION__
{"type": "known", "items": [{"source": ["tr.1"], "text": "<new durable fact from latest results>"}]} __END_ACTION__
{"type": "hypothesis", "items": [{"id": "h1", "text": "<possible direction>", "status": "{ __hypothesis_statuses__ }", "source": ["tr.1"], "context": "<result context or reason>"}]} __END_ACTION__
{"type": "keep", "source": ["tr.1"], "reason": "<why this raw result should remain in context>"} __END_ACTION__
{"type": "stable_knowledge", "items": [{"category": "stack|structure|workflow|convention|gotcha", "text": "<stable reusable session codebase fact>"}]} __END_ACTION__
{"type": "forget", "source": ["tr.2"], "reason": "<why this visible raw result no longer matters>"} __END_ACTION__
"""


############################
# Compactor Prompt
############################


COMPACTOR_PROMPT = """You are nanocode's conversation-history compactor.

Compress conversation history and Known facts so the coding agent can continue later.
Do not solve the task or add unsupported facts.

Preserve continuity-critical facts:
- user requests and changes
- decisions made
- current goal and commitments
- plan/status
- files, paths, symbols, and APIs touched
- commands run and outcomes
- known facts and context keys needed later
- unresolved blockers and open questions
- verification context

Omit noise:
- raw logs
- repeated output
- full stack traces
- chatter
- context values unless needed for continuity

Write the shortest complete continuation summary.
Compress Known to concise durable facts.

Output strict JSON only: {"summary": "<summary>", "known": [{"text": "<stable fact>", "source": ["tr.1"]}]}
Known may use strings only when no source exists.
"""


COMPACT_USER_PROMPT_TEMPLATE = """
----------- Known_To_Compact Begin ------------
{known}
--------- Known_To_Compact End ----------------

----------- Conversation_To_Compact Begin ------
{conversation}
-------- Conversation_To_Compact End -----------
"""


class PromptBuilder:
    def __init__(
        self,
        session: Session,
        *,
        system_prompt_template: str = AGENT_SYSTEM_PROMPT,
        user_prompt_template: str = AGENT_USER_PROMPT_TEMPLATE,
        blackboard: Blackboard | None = None,
        runtime: AgentRuntime | None = None,
        tool_context: ToolResultContext | None = None,
    ):
        self.session = session
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.blackboard = blackboard or Blackboard()
        self.runtime = runtime or AgentRuntime()
        self.tool_context = tool_context or ToolResultContext()

    def system_prompt(self, template: str | None = None, *, tools: Iterable[ToolClass] | None = None) -> str:
        tool_classes = tuple(TOOL_REGISTRY.values() if tools is None else tools)
        return (
            (template or self.system_prompt_template)
            .replace("{ __tools__ }", self._format_tools(tool_classes))
            .replace("{ __tool_names__ }", "|".join(tool.name() for tool in tool_classes))
            .replace("{ __hypothesis_statuses__ }", HYPOTHESIS_STATUS_SCHEMA)
            .replace("{ __hypothesis_status_text__ }", HYPOTHESIS_STATUS_TEXT)
            .strip()
        )

    def user_prompt(self, recent_tool_calls: str, errors: str) -> str:
        current = self.blackboard
        conversation = self.session.state.conversation
        return self.user_prompt_template.format(
            environment="\n".join(["- system: " + self.session.system, "- arch: " + self.session.arch, "- cwd: " + self.session.cwd]),
            conversation_history="\n\n".join(item.format() for item in conversation) if conversation else "(empty)",
            user_rules=self.session.state.user_rules.format(),
            known="\n".join(KnownItem.format_item(item) for item in current.known) if current.known else "(empty)",
            kept_tool_results="\n\n".join(self.tool_context.kept_results) or "(empty)",
            stable_knowledge=self._format_stable_knowledge(),
            tool_result_store=self._format_tool_result_store(
                set(RESULT_KEY_PATTERN.findall(recent_tool_calls)) | set(ToolResultContext.blocks_by_key(self.tool_context.kept_results))
            ),
            task_code=self.blackboard.task_code,
            work_mode=self.blackboard.work_mode,
            goal=current.goal or "(empty)",
            plan="\n".join(item.format() for item in current.plan) if current.plan else "(empty)",
            hypotheses="\n".join(item.format() for item in current.hypotheses) if current.hypotheses else "(empty)",
            verification_state=current.verification.format(),
            errors=errors or "(empty)",
            recent_tool_calls=recent_tool_calls or "(empty)",
            recent_edits="\n".join(self.runtime.recent_edits) if self.runtime.recent_edits else "(empty)",
            user_request=self._format_user_request(),
        ).strip()

    def observe_user_prompt(self, recent_tool_calls: str, errors: str) -> str:
        current = self.blackboard
        return AGENT_OBSERVE_USER_PROMPT_TEMPLATE.format(
            user_rules=self.session.state.user_rules.format(),
            goal=current.goal or "(empty)",
            plan="\n".join(item.format() for item in current.plan) if current.plan else "(empty)",
            hypotheses="\n".join(item.format() for item in current.hypotheses) if current.hypotheses else "(empty)",
            known="\n".join(KnownItem.format_item(item) for item in current.known) if current.known else "(empty)",
            stable_knowledge=self._format_stable_knowledge(),
            kept_tool_results="\n\n".join(self.tool_context.kept_results) or "(empty)",
            errors=errors or "(empty)",
            recent_tool_calls=recent_tool_calls or "(empty)",
            user_request=self._format_user_request(),
        ).strip()

    def _format_user_request(self) -> str:
        user_request = self.blackboard.user_input or "(empty)"
        fence = "`" * max(3, max((len(match.group(0)) for match in re.finditer(r"`{3,}", user_request)), default=0) + 1)
        return fence + "text\n" + user_request + "\n" + fence

    def _format_tools(self, tools: Iterable[ToolClass]) -> str:
        lines = []
        for tool in tools:
            lines.append("- " + tool.SIGNATURE)
            for item in tool.DESCRIPTION:
                lines.append("  - " + item)
            for item in tool.EXAMPLE:
                lines.append("  - " + item)
        return "\n".join(lines)

    def _format_stable_knowledge(self) -> str:
        knowledge = self.blackboard.stable_knowledge
        if not any(knowledge.values()):
            return "(empty)"
        lines = []
        for category in STABLE_KNOWLEDGE_CATEGORIES:
            items = [item for item in knowledge.get(category, []) if item]
            if not items:
                continue
            lines.append(category + ":")
            lines.extend("- " + item for item in items)
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_tool_result_store(self, visible_result_keys: set[str] | None = None) -> str:
        if not self.session.state.tool_result_store:
            return "(empty)"
        hidden_keys = visible_result_keys or set()
        lines = []
        for key, item in self.session.state.tool_result_store.items():
            if key in hidden_keys:
                continue
            lines.append(item.format(result_key=key))
        if not lines:
            return "(empty; current result keys are already shown in Recent Tool Calls)"
        return "\n".join(lines)


############################
# LLM Request (ModelClient)
############################


class ModelClient:
    ACTION_FRAME_END: ClassVar[str] = "__END_ACTION__"
    ACTION_FRAME_END_SPLIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\**_*\s*END[\s_-]*ACTION\s*_*\**", re.IGNORECASE)

    def __init__(self, session: Session):
        self.session = session
        self._timeout_reason = "request model timeout"

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        raise ModelRequestTimeout(self._timeout_reason)

    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        activity: str = "agent",
        parse_actions: bool = True,
    ) -> Json:
        config = self.session.config.provider
        if not config.url:
            raise LLMError("config provider.url is required")
        if not config.key:
            raise LLMError("config provider.key is required")
        model = config.model
        if not model:
            raise LLMError("config provider.model is required")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: Json = {
            "model": model,
            "messages": messages,
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        stream = config.stream is not False
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        timeout, first_token_timeout = self._request_timeouts(config, activity=activity)
        if config.reasoning is not False and config.reasoning_payload == "reasoning":
            payload["reasoning"] = {"effort": config.reasoning_effort or "medium"}
        if config.reasoning is not False and config.reasoning_payload == "reasoning_effort":
            payload["reasoning_effort"] = config.reasoning_effort or "medium"
        self._write_debug_prompt(activity=activity, messages=messages)
        url = config.url.rstrip("/")

        request = urllib.request.Request(
            url=url if url.endswith("/chat/completions") else url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + config.key,
                "Content-Type": "application/json",
                "User-Agent": HTTP_USER_AGENT,
            },
        )
        try:
            self.session.state.current_model_call_started_at = time.monotonic()
            self.session.state.current_model_call_label = model
            self.session.state.current_model_call_reasoning_label = config.reasoning_effort if config.reasoning else "off"
            self.session.state.current_model_call_activity = activity
            self.session.state.current_model_call_has_content = False
            request_deadline = self.session.state.current_model_call_started_at + max(0, timeout)
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            self._timeout_reason = "request model timeout"
            signal.setitimer(signal.ITIMER_REAL, max(0, timeout))
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if stream:
                        content, usage = self._read_streaming_content(
                            response,
                            request_deadline=request_deadline,
                            first_token_timeout=first_token_timeout,
                        )
                        result: Json = {"usage": usage}
                    else:
                        body = response.read().decode("utf-8")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                self.session.state.current_model_call_started_at = 0.0
                self.session.state.current_model_call_label = ""
                self.session.state.current_model_call_reasoning_label = ""
                self.session.state.current_model_call_activity = ""
                self.session.state.current_model_call_has_content = False
        except ModelRequestTimeout as error:
            raise LLMError(str(error) or "request model timeout")
        except (socket.timeout, TimeoutError):
            raise LLMError("request model timeout")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise LLMError("API request failed: HTTP " + str(error.code) + ": " + _shorten(body))
        except urllib.error.URLError as error:
            if isinstance(error.reason, (socket.timeout, TimeoutError)):
                raise LLMError("request model timeout")
            raise LLMError(str(error))
        except Exception as error:
            raise LLMError(str(error))

        if not stream:
            try:
                result = json.loads(body)
            except json.JSONDecodeError:
                raise LLMError("API response is not JSON: " + _shorten(body))

        self._record_usage(_json_dict(result.get("usage") if isinstance(result, dict) else None), config)
        if not stream:
            content = self._message_content(result)
        if content is None:
            return self._invalid_model_response(self._format_missing_message_content(result))
        if not parse_actions:
            return self._parse_json_content(content)
        return self._parse_model_content(content)

    def _request_timeouts(self, config: ProviderConfig, *, activity: str) -> tuple[int, int | None]:
        timeout = config.timeout if config.timeout is not None else 180
        first_token_timeout = config.first_token_timeout if config.first_token_timeout is not None else timeout
        if activity == "agent" and self.session.settings.plan_mode:
            return self.session.settings.plan_timeout, self.session.settings.plan_first_token_timeout
        return timeout, first_token_timeout

    def _read_streaming_content(self, response: Any, *, request_deadline: float, first_token_timeout: int | None) -> tuple[str, Json]:
        parts: list[str] = []
        usage: Json = {}
        first_content_seen = False
        self._arm_stream_timeout(request_deadline=request_deadline, first_content_seen=False, first_token_timeout=first_token_timeout)
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_data = _json_dict(event)
            event_usage = _json_dict(event_data.get("usage"))
            if event_usage:
                usage = event_usage
            choices = _json_list(event_data.get("choices"))
            if not choices:
                continue
            delta = _json_dict(_json_dict(choices[0]).get("delta"))
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                continue
            if not first_content_seen:
                first_content_seen = True
                self.session.state.current_model_call_has_content = True
                self._arm_stream_timeout(request_deadline=request_deadline, first_content_seen=True, first_token_timeout=first_token_timeout)
            parts.append(content)
        return "".join(parts), usage

    def _arm_stream_timeout(self, *, request_deadline: float, first_content_seen: bool, first_token_timeout: int | None) -> None:
        remaining = request_deadline - time.monotonic()
        if remaining <= 0:
            raise ModelRequestTimeout("request model timeout")
        self._timeout_reason = "request model timeout"
        if not first_content_seen and first_token_timeout is not None and first_token_timeout > 0:
            if first_token_timeout < remaining:
                remaining = first_token_timeout
                self._timeout_reason = "request first token timeout"
        signal.setitimer(signal.ITIMER_REAL, remaining)

    def _write_debug_prompt(self, *, activity: str, messages: list[Json]) -> str:
        if not self.session.settings.debug:
            return ""
        self.session.state.debug_prompt_count += 1
        directory = self.session.debug_dir()
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filepath = os.path.join(directory, f"{timestamp}-{self.session.state.debug_prompt_count:04d}-{activity or 'request'}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._format_debug_prompt(messages=messages))
        return filepath

    def _format_debug_prompt(self, *, messages: list[Json]) -> str:
        lines = []
        for index, message in enumerate(messages, start=1):
            role = _json_str(message.get("role")) or "(unknown)"
            content = message.get("content")
            lines.append(f"--- {role} message {index} ---")
            if isinstance(content, str):
                lines.append(content)
            else:
                lines.append(json.dumps(content, ensure_ascii=False, indent=2))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _parse_model_content(self, content: str) -> Json:
        text = content.strip()
        text = self._strip_leaked_think_tags(text)
        text = self._strip_json_fence(text)
        text = self._strip_leaked_think_tags(text)
        if not self._has_action_frame_end(text):
            actions, error = self._parse_unmarked_actions(text)
            if actions:
                return {"actions": actions}
            if error == "":
                return {"actions": []}
            return self._invalid_model_response(content, "expected one JSON action object or action frames ending with " + self.ACTION_FRAME_END + "; " + error)
        actions: list[Json] = []
        frame_errors: list[str] = []
        for frame_number, frame in enumerate(self._action_frames(text), start=1):
            parsed_actions, error = self._parse_action_frame(frame, frame_number)
            if parsed_actions:
                actions.extend(parsed_actions)
                continue
            if error:
                frame_errors.append(error)
        if not actions:
            if not frame_errors:
                return {"actions": []}
            reason = "expected at least one valid action frame ending with " + self.ACTION_FRAME_END
            if frame_errors:
                reason += "; " + "; ".join(frame_errors[:3])
            return self._invalid_model_response(content, reason)
        response: Json = {"actions": actions}
        if frame_errors:
            response["_format_frame_errors"] = frame_errors
        return response

    def _parse_json_content(self, content: str) -> Json:
        text = content.strip()
        text = self._strip_leaked_think_tags(text)
        text = self._strip_json_fence(text)
        text = self._strip_leaked_think_tags(text)
        try:
            value = json_repair.loads(text)
        except Exception as error:
            raise LLMError("model returned invalid JSON: " + str(error))
        if not isinstance(value, dict):
            raise LLMError("model returned JSON that is not an object")
        return value

    def _action_frames(self, text: str) -> list[str]:
        frames: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if not self._has_action_frame_end(line):
                current.append(line)
                continue
            parts = self.ACTION_FRAME_END_SPLIT_PATTERN.split(line)
            for index, part in enumerate(parts):
                if part:
                    current.append(part)
                if index < len(parts) - 1:
                    frames.append("\n".join(current).strip())
                    current = []
        trailing = "\n".join(current).strip()
        if trailing:
            frames.append(trailing)
        return frames

    def _parse_action_frame(self, frame: str, frame_number: int) -> tuple[list[Json], str]:
        frame = frame.strip()
        if not frame:
            return [], ""
        try:
            value = json_repair.loads(frame)
        except Exception as error:
            return [], "frame " + str(frame_number) + ": " + str(error)
        actions, error = self._actions_from_json_value(value)
        if error:
            return [], "frame " + str(frame_number) + ": " + error
        return actions, ""

    def _actions_from_json_value(self, value: JsonValue) -> tuple[list[Json], str]:
        if isinstance(value, dict):
            if "actions" in value:
                return self._actions_from_json_value(value.get("actions"))
            self._normalize_tool_type(value)
            if not _json_str(value.get("type")):
                return [], "action missing type"
            return [value], ""
        if isinstance(value, list):
            actions = []
            for index, raw in enumerate(value, start=1):
                action = _json_dict(raw)
                if not action:
                    return [], "array item " + str(index) + ": expected JSON object action"
                self._normalize_tool_type(action)
                if not _json_str(action.get("type")):
                    return [], "array item " + str(index) + ": action missing type"
                actions.append(action)
            return actions, ""
        return [], "expected JSON object action"

    def _normalize_tool_type(self, action: Json) -> None:
        action_type = _json_str(action.get("type"))
        tool_name = next((name for name in TOOL_REGISTRY if name.lower() == action_type.lower()), "") if action_type else ""
        if tool_name:
            action["type"] = "tool"
            action.setdefault("name", tool_name)

    def _parse_unmarked_actions(self, text: str) -> tuple[list[Json], str]:
        actions: list[Json] = []
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text) and text[index].isspace():
            index += 1
        prefix = ""
        if index < len(text) and text[index] != "{":
            if text[index] == "[":
                try:
                    value, index = self._decode_json_array_text(text, index)
                except (json.JSONDecodeError, ValueError) as error:
                    return [], str(error)
                parsed, error = self._actions_from_json_value(value)
                if error:
                    return [], error
                while index < len(text) and text[index].isspace():
                    index += 1
                if index < len(text):
                    return [], "unexpected text after JSON action array"
                return parsed, ""
            action_start = text.find("{", index)
            if action_start < 0:
                progress = self._plain_progress_text(text[index:])
                if progress:
                    return [{"type": "progress", "text": progress}], ""
                try:
                    decoder.raw_decode(text, index)
                except json.JSONDecodeError as error:
                    return [], str(error)
                return [], "expected JSON object action"
            prefix = self._progress_text(text[:action_start])
            index = action_start
        while True:
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                if prefix and actions:
                    actions.insert(0, {"type": "progress", "text": prefix})
                return actions, ""
            try:
                value, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError as error:
                if actions:
                    return [], str(error)
                if self._should_repair_json_decode_error(str(error), text):
                    repaired, repair_error = self._repair_single_json_action(text)
                    if not repair_error:
                        if prefix:
                            repaired.insert(0, {"type": "progress", "text": prefix})
                        return repaired, ""
                return [], str(error)
            parsed, error = self._actions_from_json_value(value)
            if error:
                return [], error
            actions.extend(parsed)
            while index < len(text) and text[index].isspace():
                index += 1
            if index < len(text) and text[index] == ",":
                index += 1
                continue
            if index < len(text) and text[index] != "{":
                next_action = text.find("{", index)
                if next_action < 0:
                    if self._should_repair_trailing_json_text(text[index:]):
                        repaired, error = self._repair_single_json_action(text)
                        if not error:
                            return repaired, ""
                    return [], "unexpected text after JSON action"
                progress = self._progress_text(text[index:next_action])
                if progress:
                    actions.append({"type": "progress", "text": progress})
                index = next_action

    def _progress_text(self, text: str) -> str:
        text = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
        text = text.replace("```", "")
        return _shorten(" ".join(text.split()), 500)

    def _plain_progress_text(self, text: str) -> str:
        progress = self._progress_text(text)
        if not progress or "{" in progress or "}" in progress:
            return ""
        starters = (
            "let me ",
            "i need ",
            "i will ",
            "i'll ",
            "now ",
            "next ",
            "我需要",
            "让我",
            "我会",
            "现在",
            "接下来",
        )
        return progress if progress.lower().startswith(starters) else ""

    def _decode_json_array_text(self, text: str, index: int) -> tuple[JsonValue, int]:
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(text, index)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return value, cursor
        value = json_repair.loads(text[index:])
        if not isinstance(value, list):
            raise ValueError("expected JSON action array")
        return value, len(text)

    def _repair_single_json_action(self, text: str) -> tuple[list[Json], str]:
        try:
            value = json_repair.loads(text)
        except Exception as error:
            return [], str(error)
        if isinstance(value, list):
            return [], "unexpected text after JSON action"
        return self._actions_from_json_value(value)

    def _should_repair_json_decode_error(self, error: str, text: str) -> bool:
        return "Invalid control character" in error or re.fullmatch(r".*[}\]]\s*[}\]]+\s*", text, re.DOTALL) is not None

    def _should_repair_trailing_json_text(self, text: str) -> bool:
        return re.fullmatch(r"\s*[}\]]+\s*", text) is not None

    def _has_action_frame_end(self, line: str) -> bool:
        return self.ACTION_FRAME_END_SPLIT_PATTERN.search(line) is not None

    def _strip_json_fence(self, text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _strip_leaked_think_tags(self, text: str) -> str:
        text = text.strip()
        while text.startswith("</think>"):
            text = text[len("</think>") :].lstrip()
        while text.startswith("<think>"):
            end = text.find("</think>")
            if end < 0:
                return text
            text = text[end + len("</think>") :].lstrip()
            while text.startswith("</think>"):
                text = text[len("</think>") :].lstrip()
        return text

    def _invalid_model_response(self, content: str, reason: str = "expected one JSON object matching the Output JSON schema") -> Json:
        guidance = ""
        if self._strip_leaked_think_tags(content.strip()).startswith("<tool_call>"):
            guidance = (
                " Native tool_call syntax is not supported; return an action frame like "
                '{"type":"tool","name":"Read","intention":"...","args":["nanocode.py","0,100"]}\n__END_ACTION__.'
            )
        return {
            "actions": [],
            "_format_bad_output": content,
            "_format_error": "Invalid model output: " + reason + ". Return action frames only. Bad output: " + _shorten(content) + guidance,
        }

    def _message_content(self, result: JsonValue) -> str | None:
        data = _json_dict(result)
        choices = _json_list(data.get("choices"))
        if not choices:
            raise LLMError("API response missing choices")
        message = _json_dict(_json_dict(choices[0]).get("message"))
        content = message.get("content")
        if not isinstance(content, str):
            return None
        return content

    def _format_missing_message_content(self, result: JsonValue) -> str:
        choice = _json_dict(_json_list(_json_dict(result).get("choices"))[0])
        message = _json_dict(choice.get("message"))
        details: Json = {
            "finish_reason": choice.get("finish_reason"),
            "message_keys": sorted(str(key) for key in message.keys()),
        }
        return "API response missing message content: " + json.dumps(details, ensure_ascii=False)

    def _record_usage(self, usage: Json, config: ProviderConfig) -> None:
        prompt_tokens = self._json_int(usage.get("prompt_tokens"))
        completion_tokens = self._json_int(usage.get("completion_tokens"))
        total_tokens = self._json_int(usage.get("total_tokens"))
        self.session.state.last_prompt_tokens = prompt_tokens
        self.session.state.last_completion_tokens = completion_tokens
        self.session.state.last_total_tokens = total_tokens
        self.session.state.session_prompt_tokens += prompt_tokens
        self.session.state.session_completion_tokens += completion_tokens
        self.session.state.session_total_tokens += total_tokens
        self.session.state.model_usage.setdefault(config.model or "(empty)", ModelUsage()).add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _json_int(value: JsonValue) -> int:
        return value if isinstance(value, int) else 0


############################
# ToolCallRunner
############################


class ToolCallDisplayFormatter:
    DISPLAY_LIMIT: ClassVar[int] = 5

    @classmethod
    def latest_report(cls, executions: list[ToolCallExecution]) -> str:
        if not executions:
            return ""
        offset = max(0, len(executions) - cls.DISPLAY_LIMIT)
        visible = executions[offset:]
        lines = []
        if offset:
            lines.append("  ... " + str(offset) + " older")
        for execution in visible:
            lines.append(cls._format_execution(execution))
        return "\n".join(lines)

    @classmethod
    def _format_execution(cls, execution: ToolCallExecution) -> str:
        marker = "[success]" if execution.outcome == "success" else "[failure]"
        text = marker + " " + cls._format_call(execution.call)
        if execution.result_key:
            text += " -> " + execution.result_key
        if execution.outcome != "success":
            error = cls._compact_tool_error(execution.output)
            if error:
                text += " | " + error
        elif execution.result_excerpted:
            text += " | excerpt"
        return text

    @classmethod
    def _format_call(cls, call: ParsedToolCall) -> str:
        tool_class = TOOL_REGISTRY.get(call.name)
        tokens = tool_class.cli_args(call.args) if tool_class is not None else [Tool.cli_token(arg) for arg in call.args]
        return " ".join([call.name] + tokens)

    @staticmethod
    def _compact_tool_error(output: str) -> str:
        if "* reason: user_ctrl_c" in output or "* interrupted: true" in output:
            return "interrupted by user"
        text = " ".join(output.split())
        prefix = "ToolCallError: "
        if text.startswith(prefix):
            text = text[len(prefix) :]
        return _shorten(text, 180)


class ToolCallRunner:
    MAX_TOOL_RESULT_STORE_ITEMS: ClassVar[int] = 256

    def __init__(self, session: Session, protected_result_keys: Callable[[], set[str]] | None = None):
        self.session = session
        self.protected_result_keys = protected_result_keys or (lambda: set())
        self.latest_executions: list[ToolCallExecution] = []
        self.skipped_after_failure_count = 0
        self.skipped_after_failure_key = ""

    def execute(
        self,
        tool_calls: list[JsonValue],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_live_output: ToolLiveOutputCallback | None = None,
        on_live_done: ToolLiveDoneCallback | None = None,
    ) -> None:
        executions = []
        self.skipped_after_failure_count = 0
        self.skipped_after_failure_key = ""
        items = self._merge_adjacent_tool_calls(self._dedupe_readonly_tool_calls(tool_calls))
        for index, item in enumerate(items):
            call: ParsedToolCall | None = None
            outcome = "success"
            output = ""
            error_type: Type[Exception] | None = None
            requires_confirmation = False
            requires_verification = False
            try:
                if isinstance(item, PreparedToolCall):
                    call = item.call
                    tool = item.tool
                else:
                    call = item if isinstance(item, ParsedToolCall) else self.parse_tool_call(item)
                    tool = self._make_tool(call)
                requires_verification = tool.effect() == ToolEffect.EDIT
                preview_error = getattr(tool, "preview_error", None)
                if callable(preview_error):
                    preview_error_text = str(preview_error())
                    if preview_error_text:
                        raise ToolCallError("preview unavailable: " + preview_error_text)
                requires_confirmation = tool.requires_confirmation(self.session)
                if requires_confirmation:
                    if self.session.settings.yolo:
                        if on_auto_approve is not None:
                            on_auto_approve(call, tool)
                    elif confirm is None:
                        raise Cancellation("user confirmation required")
                    else:
                        confirmation = confirm(call, tool)
                        if confirmation is not True:
                            reason = " ".join(confirmation.split()) if isinstance(confirmation, str) else ""
                            if reason:
                                raise Cancellation("user refused: " + reason)
                            raise Cancellation("user refused")
                output = self._call_tool(tool, call, on_live_output=on_live_output, on_live_done=on_live_done)
                exit_match = re.search(r"^\* exit_code: (-?\d+)$", output, re.MULTILINE)
                if exit_match and int(exit_match.group(1)) != 0:
                    outcome = "failure"
            except Cancellation as error:
                outcome = "failure"
                output = "Cancelled: " + str(error)
                error_type = type(error)
            except Exception as error:
                outcome = "failure"
                output = "ToolCallError: " + str(error)
                error_type = type(error)
            if call is None:
                call = self._invalid_tool_call(item)
            result_key = ""
            result_excerpted = False
            if call.name != ToolResultTool.name():
                result_key = self._store_tool_result(call, outcome, output)
                item = self.session.state.tool_result_store[result_key]
                output = item.value
                result_excerpted = item.excerpted

            execution = ToolCallExecution(
                call=call,
                outcome=outcome,
                output=output,
                error_type=error_type,
                result_key=result_key,
                result_excerpted=result_excerpted,
                requires_verification=outcome == "success" and requires_verification,
            )
            executions.append(execution)
            if outcome == "failure" and error_type is not Cancellation:
                self.skipped_after_failure_count = len(items) - index - 1
                self.skipped_after_failure_key = result_key or _format_tool_call_summary(call)
                break
            if error_type is Cancellation:
                break

        self.latest_executions = executions

    def _readonly_call_key(self, call: ParsedToolCall) -> tuple[str, tuple[str, ...]] | None:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None or tool_class.EFFECT != ToolEffect.READONLY:
            return None
        return call.name, tuple(call.args)

    def _call_tool(
        self,
        tool: Tool,
        call: ParsedToolCall,
        *,
        on_live_output: ToolLiveOutputCallback | None,
        on_live_done: ToolLiveDoneCallback | None,
    ) -> str:
        live_started = False

        def sink(chunk: str) -> None:
            nonlocal live_started
            if not chunk:
                return
            live_started = True
            if on_live_output is not None:
                on_live_output(call, chunk)

        try:
            return tool.call_live(sink if on_live_output is not None else None)
        finally:
            if live_started and on_live_done is not None:
                on_live_done(call)

    def _dedupe_readonly_tool_calls(self, tool_calls: list[JsonValue]) -> list[JsonValue | ParsedToolCall]:
        filtered: list[JsonValue | ParsedToolCall] = []
        for item in tool_calls:
            try:
                call = self.parse_tool_call(item)
            except ToolCallArgError:
                filtered.append(item)
                continue
            key = self._readonly_call_key(call)
            if key is not None and filtered and isinstance(filtered[-1], ParsedToolCall) and self._readonly_call_key(filtered[-1]) == key:
                filtered[-1] = call
                continue
            if call.name == ToolResultTool.name() and filtered and isinstance(filtered[-1], ParsedToolCall) and filtered[-1].name == call.name:
                merged_args = list(filtered[-1].args)
                merged_args.extend(arg for arg in call.args if arg not in merged_args)
                filtered[-1] = ParsedToolCall(name=call.name, intention=call.intention, args=merged_args)
                continue
            filtered.append(call)
        return filtered

    def _merge_adjacent_tool_calls(self, tool_calls: list[JsonValue | ParsedToolCall]) -> list[JsonValue | ParsedToolCall | PreparedToolCall]:
        merged: list[JsonValue | ParsedToolCall | PreparedToolCall] = []
        index = 0
        while index < len(tool_calls):
            item = tool_calls[index]
            merge_key = self._merge_key(item)
            if merge_key is None:
                merged.append(item)
                index += 1
                continue

            group = [item]
            index += 1
            while index < len(tool_calls):
                next_item = tool_calls[index]
                if self._merge_key(next_item) != merge_key:
                    break
                group.append(next_item)
                index += 1

            if len(group) == 1:
                merged.append(item)
                continue

            prepared = self._merge_calls(group)
            if prepared is None:
                merged.extend(group)
            else:
                merged.append(prepared)
        return merged

    def _merge_key(self, item: JsonValue | ParsedToolCall) -> tuple[str, tuple[str, ...]] | None:
        if not isinstance(item, ParsedToolCall) or item.name != ReplaceRangeTool.name():
            return None
        key = ReplaceRangeTool.merge_key(item)
        if key is None:
            return None
        return (item.name, key)

    def _merge_calls(self, group: list[JsonValue | ParsedToolCall]) -> PreparedToolCall | None:
        parsed_group = [item for item in group if isinstance(item, ParsedToolCall)]
        if len(parsed_group) != len(group):
            return None
        if parsed_group[0].name != ReplaceRangeTool.name():
            return None
        return ReplaceRangeTool.merge_calls(self.session, parsed_group)

    def _store_tool_result(self, call: ParsedToolCall, outcome: str, output: str) -> str:
        self.session.state.tool_result_counter += 1
        key = "tr." + str(self.session.state.tool_result_counter)
        description = outcome + " " + ToolCallDisplayFormatter._format_call(call)
        if call.intention:
            description += " - " + call.intention
        log_path = self._write_tool_result_log(key, output)
        bounded = _bound_tool_output(output, log_path=log_path)
        self.session.state.tool_result_store[key] = ToolResultItem(
            description=description,
            value=bounded.value,
            log_path=log_path,
            original_lines=bounded.original_lines,
            original_chars=bounded.original_chars,
            excerpted=bounded.excerpted,
        )
        self._trim_tool_result_store()
        return key

    def _write_tool_result_log(self, key: str, output: str) -> str:
        directory = self.session.tool_results_dir()
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        pid = os.getpid()
        for attempt in range(3):
            suffix = "" if attempt == 0 else f"-{attempt}"
            filepath = os.path.join(directory, f"{timestamp}-{pid}-{key}{suffix}.log")
            try:
                with open(filepath, "x", encoding="utf-8") as fp:
                    fp.write(output)
                return os.path.relpath(filepath, self.session.cwd) if self.session.is_path_in_cwd(filepath) else filepath
            except FileExistsError:
                continue
        return ""

    def _trim_tool_result_store(self) -> None:
        keep = self.protected_result_keys()
        for old_key in list(self.session.state.tool_result_store):
            if len(self.session.state.tool_result_store) <= self.MAX_TOOL_RESULT_STORE_ITEMS:
                return
            if old_key in keep:
                continue
            self.session.state.tool_result_store.pop(old_key)

    def parse_tool_call(self, value: JsonValue) -> ParsedToolCall:
        item = _json_dict(value)
        name = _json_str(item.get("name"))
        if not name:
            raise ToolCallArgError('tool action missing required field: name. Use {"type":"tool","name":"Read","intention":"...","args":["path"]}.')
        if name not in TOOL_REGISTRY and name == name.lower():
            name = next((registered_name for registered_name in TOOL_REGISTRY if registered_name.lower() == name), name)
        intention = _json_str(item.get("intention")) or ""
        args = [_json_str(arg) or "" for arg in _json_list(item.get("args"))]
        return ParsedToolCall(name=name, intention=intention, args=args)

    def _invalid_tool_call(self, value: JsonValue) -> ParsedToolCall:
        item = _json_dict(value)
        summary = "invalid tool action"
        if _json_str(item.get("type")) == "tool" and not _json_str(item.get("name")):
            summary += ": missing required field name"
        return ParsedToolCall(name="InvalidToolCall", intention=summary, args=[])

    def _make_tool(self, call: ParsedToolCall) -> Tool:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            raise ToolCallArgError("tool not found: " + call.name)
        return tool_class.make(self.session, call.args)


############################
# Agent State
############################


STABLE_KNOWLEDGE_CATEGORIES: tuple[str, ...] = ("stack", "structure", "workflow", "convention", "gotcha")


class AgentStateUpdater:
    DISPLAY_LIMIT: ClassVar[int] = 5
    COMPACT_DISPLAY_LIMIT: ClassVar[int] = 3
    MAX_KNOWN_ITEMS: ClassVar[int] = 500
    MAX_STABLE_KNOWLEDGE_ITEMS_PER_CATEGORY: ClassVar[int] = 30
    VERIFY_STATUS_ACTIONS: ClassVar[dict[str, VerificationStatus]] = {
        "passed": VerificationStatus.DONE,
        "failed": VerificationStatus.FAILED,
        "blocked": VerificationStatus.BLOCKED,
    }

    def __init__(
        self,
        session: Session,
        blackboard: Blackboard,
    ):
        self.session = session
        self.blackboard = blackboard
        self.latest_report = ""
        self.latest_compact_plan_rows: list[str] = []
        self.changed = False

    def apply(self, response: Json) -> None:
        actions = self._actions(response)
        before_goal = self.blackboard.goal
        before_plan = [item.format() for item in self.blackboard.plan]
        before_hypotheses = [item.format() for item in self.blackboard.hypotheses]
        before_known = [KnownItem.format_item(item) for item in self.blackboard.known]
        before_user_rules = self.session.state.user_rules.format()
        before_extra_state = self._before_extra_state()
        goal_changed = self._apply_goal(actions)
        plan_replaced = self._apply_plan(actions)
        if goal_changed and not plan_replaced:
            self.blackboard.plan = []
        self._apply_work_mode(actions)
        self._apply_known(actions)
        self._apply_hypotheses(actions)
        self._apply_user_rules(actions)
        self._apply_extra_state(actions, goal_changed=goal_changed, plan_replaced=plan_replaced)
        self._apply_task_code(actions)
        self.latest_report = self._format_state_report(
            before_goal,
            before_plan,
            before_hypotheses,
            before_known,
            before_user_rules,
            before_extra_state,
        )
        self.changed = bool(self.latest_report)

    def _actions(self, response: Json) -> list[Json]:
        return [action for action in (_json_dict(item) for item in _json_list(response.get("actions"))) if action]

    def _format_state_report(
        self,
        before_goal: str,
        before_plan: list[str],
        before_hypotheses: list[str],
        before_known: list[str],
        before_user_rules: str,
        before_extra_state: str,
    ) -> str:
        current = self.blackboard
        lines = []
        if current.goal != before_goal:
            self._append_state_section(lines, "  Goal    " + self._compact(current.goal or "(empty)"))
        plan = [item.format() for item in current.plan]
        self.latest_compact_plan_rows = []
        if plan != before_plan:
            self.latest_compact_plan_rows = self._compact_changed_plan_rows(before_plan, plan)
            self._append_state_section(lines, "  Plan", self._format_plan_rows())
        hypotheses = [item.format() for item in current.hypotheses]
        if hypotheses != before_hypotheses:
            self._append_state_section(lines, "  Hypotheses", self._format_hypothesis_rows())
        known = [KnownItem.format_item(item) for item in current.known]
        if known != before_known:
            self._append_state_section(lines, "  Known", self._format_known_rows())
        user_rules = self.session.state.user_rules.format()
        if user_rules != before_user_rules:
            self._append_state_section(lines, "  User_Rules    updated")
        self._append_extra_state_report(lines, before_extra_state)
        return "\n".join(lines)

    def _format_plan_rows(self) -> list[str]:
        items = self.blackboard.plan
        if not items:
            return ["    (empty)"]
        offset = max(0, len(items) - self.DISPLAY_LIMIT)
        rows = ["    ... " + str(offset) + " older"] if offset else []
        for index, item in enumerate(items[offset:], start=offset + 1):
            rows.append("    " + str(index) + ". [" + str(item.status) + "] " + self._compact(item.text))
            if item.context:
                rows.append("       context: " + self._compact(item.context))
        return rows

    def _format_known_rows(self) -> list[str]:
        items = self.blackboard.known
        if not items:
            return ["    (empty)"]
        offset = max(0, len(items) - self.DISPLAY_LIMIT)
        rows = ["    ... " + str(offset) + " older"] if offset else []
        for index, item in enumerate(items[offset:], start=offset + 1):
            rows.append("    " + str(index) + ". " + self._compact(KnownItem.format_item(item)))
        return rows

    def _format_hypothesis_rows(self) -> list[str]:
        items = self.blackboard.hypotheses
        if not items:
            return ["    (empty)"]
        offset = max(0, len(items) - self.DISPLAY_LIMIT)
        rows = ["    ... " + str(offset) + " older"] if offset else []
        for index, item in enumerate(items[offset:], start=offset + 1):
            rows.append("    " + str(index) + ". " + self._compact(item.format()))
        return rows

    def compact_report(self) -> str:
        sections = []
        if "  Plan" in self.latest_report and self.blackboard.plan:
            sections.append("Plan")
        if "  Hypotheses" in self.latest_report and self.blackboard.hypotheses:
            sections.append("Hypotheses")
        if "  Known" in self.latest_report and self.blackboard.known:
            sections.append("Known")
        if not sections:
            return ""
        lines = [" + ".join(sections) + " Updated"]
        grouped = len(sections) > 1
        if "Plan" in sections:
            if grouped:
                lines.append("Plan")
            lines.extend(self.latest_compact_plan_rows or self._compact_plan_rows())
        if "Hypotheses" in sections:
            if grouped:
                lines.append("Hypotheses")
            lines.extend(self._compact_hypothesis_rows())
        if "Known" in sections:
            if grouped:
                lines.append("Known")
            lines.extend(self._compact_known_rows())
        return "\n".join(lines)

    def _compact_plan_rows(self) -> list[str]:
        items = self.blackboard.plan
        offset = max(0, len(items) - self.COMPACT_DISPLAY_LIMIT)
        rows = ["  ... " + str(offset) + " older"] if offset else []
        rows.extend(self._compact_plan_row(index, item) for index, item in enumerate(items[offset:], start=offset + 1))
        return rows

    def _compact_changed_plan_rows(self, before_plan: list[str], plan: list[str]) -> list[str]:
        if not before_plan:
            return self._compact_plan_rows()
        indexes = [
            index
            for index in range(max(len(before_plan), len(plan)))
            if (before_plan[index] if index < len(before_plan) else None) != (plan[index] if index < len(plan) else None)
        ]
        if not indexes or any(index >= len(self.blackboard.plan) for index in indexes):
            return self._compact_plan_rows()
        offset = max(0, len(indexes) - self.COMPACT_DISPLAY_LIMIT)
        rows = ["  ... " + str(offset) + " changed older"] if offset else []
        rows.extend(self._compact_plan_row(index + 1, self.blackboard.plan[index]) for index in indexes[offset:])
        return rows

    def _compact_plan_row(self, index: int, item: PlanItem) -> str:
        return "  " + str(index) + ". [" + str(item.status) + "] " + self._compact(item.text, 90)

    def _compact_known_rows(self) -> list[str]:
        items = self.blackboard.known
        offset = max(0, len(items) - self.COMPACT_DISPLAY_LIMIT)
        rows = ["  ... " + str(offset) + " older"] if offset else []
        rows.extend("  " + str(index) + ". " + self._compact(KnownItem.format_item(item), 100) for index, item in enumerate(items[offset:], start=offset + 1))
        return rows

    def _compact_hypothesis_rows(self) -> list[str]:
        items = self.blackboard.hypotheses
        offset = max(0, len(items) - self.COMPACT_DISPLAY_LIMIT)
        rows = ["  ... " + str(offset) + " older"] if offset else []
        rows.extend("  " + str(index) + ". " + self._compact(item.format(), 100) for index, item in enumerate(items[offset:], start=offset + 1))
        return rows

    def _compact(self, text: str, limit: int = 140) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _apply_goal(self, actions: list[Json]) -> bool:
        changed = False
        for action in actions:
            action_type = _json_str(action.get("type"))
            if action_type == "start":
                update = _json_str(action.get("goal"))
                if update:
                    goal_changed = update != self.blackboard.goal
                    changed = changed or goal_changed
                    self.blackboard.goal = update
                    self.blackboard.goal_reached = False
            if action_type == "goal":
                update = _json_str(action.get("text"))
                complete = action.get("complete")
                if update is not None:
                    goal_changed = update != self.blackboard.goal
                    changed = changed or (goal_changed and complete is not True)
                    self.blackboard.goal = update
                if isinstance(complete, bool):
                    self.blackboard.goal_reached = complete
        return changed

    def _apply_plan(self, actions: list[Json]) -> bool:
        replaced = False
        for start in [action for action in actions if _json_str(action.get("type")) == "start"]:
            items = [item for item in (self._plan_item_from_json(raw) for raw in _json_list(start.get("plan"))) if item]
            if items:
                self.blackboard.plan = items
                replaced = True
        for update in [action for action in actions if _json_str(action.get("type")) == "plan"]:
            items = _json_list(update.get("items"))
            if update.get("mode") != "patch":
                if not items:
                    continue
                self.blackboard.plan = [item for item in (self._plan_item_from_json(raw) for raw in items) if item]
                replaced = True
                continue
            self._apply_plan_patches(self.blackboard.plan, items)
        return replaced

    def _apply_plan_patches(self, plan: list[PlanItem], value: JsonValue) -> bool:
        changed = False
        for raw in _json_list(value):
            patch = _json_dict(raw)
            op = _json_str(patch.get("op")) or "add"
            item_id = _json_str(patch.get("id")) or ""
            if op == "remove":
                before = len(plan)
                plan[:] = [item for item in plan if item.id != item_id]
                changed = changed or len(plan) != before
                continue
            existing = next((item for item in plan if item.id == item_id and item.id), None)
            if existing:
                text = _json_str(patch.get("text")) if "text" in patch else None
                status = _json_str(patch.get("status")) if "status" in patch else None
                context = _json_str(patch.get("context")) if "context" in patch else existing.context
                updated = (
                    text or existing.text,
                    PlanStatus(status) if status in ALL_PLAN_STATUSES else existing.status,
                    context or "",
                )
                changed = changed or (existing.text, existing.status, existing.context) != updated
                existing.text, existing.status, existing.context = updated
                continue
            plan_item = self._plan_item_from_json(patch)
            if plan_item is None:
                continue
            plan.append(plan_item)
            changed = True
        return changed

    def _plan_item_from_json(self, value: JsonValue) -> PlanItem | None:
        item = _json_dict(value)
        text = _json_str(item.get("text"))
        if not text:
            return None
        status = _json_str(item.get("status")) or PlanStatus.TODO
        if status not in ALL_PLAN_STATUSES:
            status = PlanStatus.TODO
        return PlanItem(
            text=text,
            status=PlanStatus(status),
            id=_json_str(item.get("id")) or "",
            context=_json_str(item.get("context")) or "",
        )

    def _apply_known(self, actions: list[Json]) -> None:
        for action in actions:
            values = _json_list(action.get("items")) if _json_str(action.get("type")) == "known" else []
            for raw in values:
                item = KnownItem.from_json(raw)
                if item is not None:
                    self._add_known_item(item.text, item.source)

    def _apply_hypotheses(self, actions: list[Json]) -> None:
        for action in actions:
            values = _json_list(action.get("items")) if _json_str(action.get("type")) == "hypothesis" else []
            for raw in values:
                item = Hypothesis.from_json(raw)
                if item is not None:
                    self._add_hypothesis(item)

    def _apply_work_mode(self, actions: list[Json]) -> None:
        for action in actions:
            if _json_str(action.get("type")) != "start":
                continue
            mode = _json_str(action.get("work_mode")) or WorkMode.NORMAL
            self.blackboard.work_mode = WorkMode(mode) if mode in ALL_WORK_MODES else WorkMode.NORMAL

    def _add_hypothesis(self, item: Hypothesis) -> None:
        for index, existing in enumerate(self.blackboard.hypotheses):
            same_id = item.id and item.id == existing.id
            same_text = self._hypothesis_key(item.text) == self._hypothesis_key(existing.text)
            if not same_id and not same_text:
                continue
            source = tuple(dict.fromkeys((*existing.source, *item.source)))
            self.blackboard.hypotheses[index] = Hypothesis(
                text=item.text or existing.text,
                status=item.status,
                id=item.id or existing.id,
                source=source,
                context=item.context or existing.context,
            )
            return
        self.blackboard.hypotheses.append(item)

    def _hypothesis_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" \t\r\n。.;；").lower()

    def _apply_user_rules(self, actions: list[Json]) -> None:
        changed = False
        for action in actions:
            if _json_str(action.get("type")) != "user_rule":
                continue
            rule = (_json_str(action.get("text")) or "").strip()
            changed = self.session.state.user_rules.add(rule) or changed
        if changed:
            self.session.save_user_rules()

    def _add_known_item(self, fact: str, source: tuple[str, ...] = ()) -> None:
        fact = _shorten(" ".join(fact.split()))
        for index, existing in enumerate(self.blackboard.known):
            if self._known_facts_overlap(existing, fact):
                text = KnownItem.text_of(existing)
                merged_source = tuple(dict.fromkeys((*KnownItem.source_of(existing), *source)))
                if len(fact) > len(text):
                    self.blackboard.known[index] = KnownItem(text=fact, source=merged_source)
                elif merged_source != KnownItem.source_of(existing):
                    self.blackboard.known[index] = KnownItem(text=text, source=merged_source)
                return
        self.blackboard.known.append(KnownItem(text=fact, source=source))
        del self.blackboard.known[: max(0, len(self.blackboard.known) - self.MAX_KNOWN_ITEMS)]

    def _known_facts_overlap(self, left: KnownItem | str, right: KnownItem | str) -> bool:
        left_key = self._known_fact_key(left)
        right_key = self._known_fact_key(right)
        if left_key == right_key:
            return True
        return min(len(left_key), len(right_key)) >= 32 and (left_key in right_key or right_key in left_key)

    def _known_fact_key(self, fact: KnownItem | str) -> str:
        return re.sub(r"\s+", " ", KnownItem.text_of(fact)).strip(" \t\r\n。.;；").lower()

    def _before_extra_state(self) -> str:
        return json.dumps(
            {
                "verification": self.blackboard.verification.format(),
                "stable_knowledge": self.blackboard.stable_knowledge,
            },
            ensure_ascii=False,
        )

    def _apply_extra_state(self, actions: list[Json], *, goal_changed: bool, plan_replaced: bool) -> None:
        self._apply_stable_knowledge(actions)
        if goal_changed:
            self.blackboard.verification_required = False
        self._reset_stale_verification(actions, goal_changed=goal_changed, plan_replaced=plan_replaced)
        self._apply_verification(actions)
        self._bind_verification_goal()

    def _apply_task_code(self, actions: list[Json]) -> None:
        action_types = {_json_str(action.get("type")) for action in actions}
        if self.blackboard.verification_required or self.blackboard.verification.status == VerificationStatus.REQUIRED:
            self.blackboard.task_code = TaskCode.VERIFYING
            return
        if "verify" in action_types:
            self.blackboard.task_code = TaskCode.WORKING
            return
        if "start" in action_types:
            self.blackboard.task_code = TaskCode.WORKING
            return
        if any(action_type in action_types for action_type in ("goal", "plan", "known", "stable_knowledge", "progress", "tool")) and not self.blackboard.goal_reached:
            self.blackboard.task_code = TaskCode.WORKING

    def _append_state_section(self, lines: list[str], title: str, rows: list[str] | None = None) -> None:
        if not lines:
            lines.append("State Updated | VERIFY:" + self.blackboard.verification.status)
        lines.append(title)
        lines.extend(rows or [])

    def _append_extra_state_report(self, lines: list[str], before_extra_state: str) -> None:
        try:
            before = _json_dict(json.loads(before_extra_state))
        except json.JSONDecodeError:
            before = {}
        if self.blackboard.stable_knowledge != before.get("stable_knowledge", []):
            self._append_state_section(lines, "  Stable_Knowledge", self._format_stable_knowledge_rows())
        verification = self.blackboard.verification.format()
        if verification == before.get("verification", ""):
            return
        self._append_state_section(lines, "  Verify  " + self._format_verification())

    def _format_stable_knowledge_rows(self) -> list[str]:
        knowledge = self.blackboard.stable_knowledge
        if not any(knowledge.values()):
            return ["    (empty)"]
        rows = []
        for category in STABLE_KNOWLEDGE_CATEGORIES:
            items = knowledge.get(category, [])
            if not items:
                continue
            rows.append("    " + category)
            offset = max(0, len(items) - self.DISPLAY_LIMIT)
            if offset:
                rows.append("      ... " + str(offset) + " older")
            for index, item in enumerate(items[offset:], start=offset + 1):
                rows.append("      " + str(index) + ". " + self._compact(item))
        return rows

    def _apply_stable_knowledge(self, actions: list[Json]) -> None:
        for action in actions:
            values = _json_list(action.get("items")) if _json_str(action.get("type")) == "stable_knowledge" else []
            for raw in values:
                category, fact = self._stable_knowledge_item_from_json(raw)
                if fact:
                    self._add_stable_knowledge_item(category, fact)

    def _stable_knowledge_item_from_json(self, value: JsonValue) -> tuple[str, str]:
        item = _json_dict(value)
        if item:
            category = _json_str(item.get("category")) or "gotcha"
            fact = (_json_str(item.get("text")) or _json_str(item.get("fact")) or "").strip()
        else:
            category = "gotcha"
            fact = (_json_str(value) or "").strip()
        if category not in STABLE_KNOWLEDGE_CATEGORIES:
            category = "gotcha"
        return category, fact

    def _add_stable_knowledge_item(self, category: str, fact: str) -> None:
        knowledge = self.blackboard.stable_knowledge
        items = knowledge.setdefault(category, [])
        if fact in items:
            return
        items.append(fact)
        del items[: max(0, len(items) - self.MAX_STABLE_KNOWLEDGE_ITEMS_PER_CATEGORY)]

    def _format_verification(self) -> str:
        verification = self.blackboard.verification
        parts = [verification.status]
        parts.extend(
            part
            for part in (
                verification.kind,
                self._compact(verification.method) if verification.method else "",
                "criteria: " + self._compact("; ".join(verification.criteria)) if verification.criteria else "",
                "context: " + self._compact(verification.context) if verification.context else "",
                "blocker: " + verification.blocker if verification.blocker else "",
            )
            if part
        )
        return " | ".join(parts)

    def _apply_verification(self, actions: list[Json]) -> None:
        for data in [action for action in actions if _json_str(action.get("type")) == "verify"]:
            kind = _json_str(data.get("kind"))
            if kind is not None:
                self.blackboard.verification.kind = kind if kind and all(part in VALID_VERIFICATION_KINDS for part in kind.split("+")) else ""
            criteria = [item for item in ((_json_str(raw) or "").strip() for raw in _json_list(data.get("criteria"))) if item]
            if "criteria" in data:
                self.blackboard.verification.criteria = criteria
            method = _json_str(data.get("method"))
            if method is not None:
                if method != self.blackboard.verification.method:
                    self.blackboard.verification.context = ""
                self.blackboard.verification.method = method
            status = self.VERIFY_STATUS_ACTIONS.get(_json_str(data.get("status")) or "")
            if status is not None:
                self.blackboard.verification.status = status
                self.blackboard.verification_required = False
                if status != VerificationStatus.BLOCKED:
                    self.blackboard.verification.blocker = VerificationBlocker.NONE
            blocker = _json_str(data.get("blocker"))
            if blocker is not None:
                self.blackboard.verification.blocker = VerificationBlocker(blocker) if blocker in ALL_VERIFICATION_BLOCKERS else VerificationBlocker.NONE
            context = _json_str(data.get("context"))
            if context is not None:
                self.blackboard.verification.context = context

    def _reset_stale_verification(self, actions: list[Json], *, goal_changed: bool, plan_replaced: bool) -> None:
        verification = self.blackboard.verification
        if goal_changed:
            verification.reset()
            return
        if verification.goal and verification.goal != self.blackboard.goal:
            verification.reset()
            return
        if (
            plan_replaced
            and not any(_json_str(action.get("type")) == "verify" for action in actions)
            and verification.status
            in {
                VerificationStatus.REQUIRED,
                VerificationStatus.DONE,
                VerificationStatus.FAILED,
                VerificationStatus.BLOCKED,
            }
        ):
            verification.reset()

    def _bind_verification_goal(self) -> None:
        verification = self.blackboard.verification
        if not verification.has_context():
            verification.goal = ""
            return
        if self.blackboard.goal:
            verification.goal = self.blackboard.goal


############################
# ConversationCompactor
############################


class ConversationCompactor:
    KEEP_RECENT: ClassVar[int] = 5
    MAX_COMPACTED_KNOWN_ITEMS: ClassVar[int] = 150

    def __init__(self, session: Session, model_client: ModelClient, blackboard: Blackboard):
        self.session = session
        self.model_client = model_client
        self.blackboard = blackboard

    def compact(self) -> int:
        count = len(self.session.state.conversation)
        if count <= self.KEEP_RECENT:
            return 0
        old_items = self.session.state.conversation[: -self.KEEP_RECENT]
        keep_items = self.session.state.conversation[-self.KEEP_RECENT :]
        summary, known = self._summarize(old_items)
        self.session.state.conversation = [AssistantMessage(content="Conversation compact summary:\n" + summary)] + keep_items
        self.blackboard.known = known
        return count

    def maybe_compact(self) -> bool:
        if self.session.settings.compact_at <= 0:
            return False
        if len(self.session.state.conversation) <= self.session.settings.compact_at:
            return False
        return self.compact() > 0

    def _summarize(self, items: list[ConversationItem]) -> tuple[str, list[KnownItem]]:
        user_prompt = COMPACT_USER_PROMPT_TEMPLATE.format(
            known="\n".join(KnownItem.format_item(item) for item in self.blackboard.known) or "(empty)",
            conversation="\n\n".join(item.format() for item in items),
        ).strip()
        kwargs = {"parse_actions": False} if isinstance(self.model_client, ModelClient) else {}
        response = self.model_client.request(COMPACTOR_PROMPT.strip(), user_prompt, activity="compact", **kwargs)
        summary = _json_str(response.get("summary"))
        if not summary:
            raise LLMError("compact response missing summary")
        known = [item for item in (KnownItem.from_json(raw) for raw in _json_list(response.get("known"))) if item]
        if not known:
            known = list(self.blackboard.known)
        return summary, known[-self.MAX_COMPACTED_KNOWN_ITEMS :]


############################
# Verification
############################


VALID_VERIFICATION_KINDS: set[str] = {"syntax_check", "change_syntax_check", "lint", "test", "build", "change_check", "other"}


############################
# Agent
############################


@dataclass(frozen=True)
class ResponseContext:
    response: Json
    actions: list[Json]
    goal_was_empty: bool
    plan_was_empty: bool
    plan_was_complete: bool
    verification_was_settled: bool
    goal_will_change: bool
    chat_message: str | None
    tool_calls: list[JsonValue]
    pending_verify_requested: bool
    progress_messages: list[str]
    user_rule_message: str | None
    completion_message: str
    has_goal_action: bool
    has_plan_action: bool
    has_fresh_plan_action: bool
    has_user_rule_action: bool
    state_or_work_requested: bool


############################
# Agent Runtime
############################


class Agent:
    MAX_CONSECUTIVE_FORMAT_ERRORS: ClassVar[int] = 3
    MAX_AGENT_FEEDBACK_ERRORS: ClassVar[int] = 8
    MAX_AGENT_FEEDBACK_ERROR_LEN: ClassVar[int] = 220
    MODEL_TIMEOUT_RETRY_DELAYS: ClassVar[tuple[int, ...]] = (3, 10, 20, 30, 60, 120)
    blackboard: Blackboard
    ACT_ACTION_TYPES: ClassVar[set[str]] = {
        "chat",
        "start",
        "goal",
        "plan",
        "hypothesis",
        "known",
        "stable_knowledge",
        "progress",
        "tool",
        "verify",
        "user_rule",
        "forget",
    }
    PLAN_ACTION_TYPES: ClassVar[set[str]] = ACT_ACTION_TYPES - {"chat", "user_rule", "forget"}
    OBSERVE_ACTION_TYPES: ClassVar[set[str]] = {"keep", "hypothesis", "known", "stable_knowledge", "forget"}
    COMPLETED_PLAN_STATUSES: ClassVar[set[PlanStatus]] = {PlanStatus.DONE, PlanStatus.BLOCKED}
    MAX_COMPLETED_GOAL_TOOL_RESULTS: ClassVar[int] = 50
    RECENT_EDITS: ClassVar[int] = 20
    RECENT_TOOL_CALL_CHARS: ClassVar[int] = 72_000
    KEPT_TOOL_RESULT_CHARS: ClassVar[int] = 96_000
    RECENT_TOOL_CALL_SUMMARIES: ClassVar[int] = 40
    PENDING_OBSERVE_RESULTS: ClassVar[int] = 8
    PENDING_OBSERVE_CHAR_RATIO: ClassVar[float] = 0.4
    PENDING_OBSERVE_TOOL_TURNS: ClassVar[int] = 2
    PLAN_MODE_GIT_READONLY: ClassVar[frozenset[str]] = GIT_READONLY_COMMANDS

    def __init__(self, session: Session):
        self.session = session
        self.blackboard = Blackboard()
        self.runtime = AgentRuntime()
        self.tool_context = ToolResultContext()
        self.prompt_builder = PromptBuilder(
            session,
            blackboard=self.blackboard,
            runtime=self.runtime,
            tool_context=self.tool_context,
        )
        self.model_client = ModelClient(session)
        self.tool_runner = ToolCallRunner(session, self._protected_tool_result_keys)
        self.state_updater = AgentStateUpdater(session, self.blackboard)
        self.compactor = ConversationCompactor(session, self.model_client, self.blackboard)
        self.failed_tool_call_key: tuple[str, tuple[str, ...]] | None = None
        self.failed_tool_call_count = 0
        self.agent_feedback_errors: list[str] = []
        self.observe_feedback_errors: list[str] = []
        self.mode = AgentMode.ACT

    def build_user_prompt(self) -> str:
        return self.prompt_builder.user_prompt(
            self._format_recent_tool_call_context(),
            self._format_agent_feedback(),
        )

    def build_observe_prompt(self) -> str:
        return self.prompt_builder.observe_user_prompt(
            self._format_recent_tool_call_context(),
            self._format_observe_feedback(),
        )

    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        activity: str = "agent",
        on_message: MessageCallback | None = None,
    ) -> Json:
        for attempt in range(len(self.MODEL_TIMEOUT_RETRY_DELAYS) + 1):
            try:
                self.session.state.turn_model_calls += 1
                return self.model_client.request(system_prompt, user_prompt, activity=activity)
            except LLMError as error:
                timeout_reason = str(error)
                if timeout_reason not in ("request model timeout", "request first token timeout") or attempt >= len(self.MODEL_TIMEOUT_RETRY_DELAYS):
                    raise
                delay = self.MODEL_TIMEOUT_RETRY_DELAYS[attempt]
                self._set_status_notice("err:first_token" if timeout_reason == "request first token timeout" else "err:timeout")
                if on_message is not None and self.session.settings.debug:
                    on_message(
                        "Retrying: " + timeout_reason + "; retry "
                        + str(attempt + 1)
                        + "/"
                        + str(len(self.MODEL_TIMEOUT_RETRY_DELAYS))
                        + " in "
                        + str(delay)
                        + "s."
                    )
                time.sleep(delay)
        raise LLMError("request model timeout")

    def _set_status_notice(self, text: str, ttl: float = 5.0) -> None:
        self.session.state.status_notice = text
        self.session.state.status_notice_until = time.monotonic() + ttl

    def compact_history(self) -> int:
        return self.compactor.compact()

    def cancel_current_goal(self) -> None:
        self._finish_current_goal()

    def run_loop(
        self,
        *,
        max_steps: int,
        on_message: MessageCallback | None = None,
        on_step: Callable[[Json], AgentRunResult],
        on_step_limit: Callable[[], JsonValue],
        on_before_step: Callable[[int, int], None] | None = None,
        on_format_error_limit: Callable[[Json, str], JsonValue] | None = None,
    ) -> JsonValue:
        consecutive_format_errors = 0
        try:
            for index in range(max_steps):
                if on_before_step is not None:
                    on_before_step(index, max_steps)
                response = self.step(on_message=on_message)
                format_error = _json_str(response.get("_format_error"))
                if format_error:
                    consecutive_format_errors += 1
                    self._set_status_notice("err:format")
                    remember_error = self._remember_observe_error if self.mode == AgentMode.OBSERVE else self._remember_agent_error
                    remember_error(
                        self._format_gate_user_message("Error: model returned invalid output", format_error) + " Rule: return valid JSON action frames only."
                    )
                    if consecutive_format_errors >= self.MAX_CONSECUTIVE_FORMAT_ERRORS:
                        if on_format_error_limit is not None:
                            return on_format_error_limit(response, format_error)
                        self._report_gate(
                            on_message,
                            "Stopped: model returned invalid output " + str(self.MAX_CONSECUTIVE_FORMAT_ERRORS) + " times in a row.",
                            "Format_Gate: stopped after "
                            + str(self.MAX_CONSECUTIVE_FORMAT_ERRORS)
                            + " consecutive invalid model outputs. "
                            + self._format_gate_debug_details(response, format_error),
                        )
                        raise LLMError(
                            "model returned invalid output " + str(self.MAX_CONSECUTIVE_FORMAT_ERRORS) + " times in a row: " + _shorten(format_error, 300)
                        )
                    self._report_gate(
                        on_message,
                        self._format_gate_user_message("Retrying: model returned invalid output", format_error),
                        "Format_Gate: retrying model response. " + self._format_gate_debug_details(response, format_error),
                    )
                    continue
                consecutive_format_errors = 0
                result = on_step(response)
                if result.done:
                    return result.value
            return on_step_limit()
        except KeyboardInterrupt:
            self.cancel_current_goal()
            raise

    def _finish_current_goal(self) -> None:
        self.blackboard.task_code = TaskCode.DONE
        self.blackboard.goal_reached = False
        self.blackboard.verification_required = False

    def _format_recent_tool_call_context(self) -> str:
        if self.mode == AgentMode.OBSERVE and self.tool_context.pending_observe:
            return "\n\n".join(self.tool_context.pending_observe)
        return "\n\n".join(self.tool_context.recent + self.tool_context.latest)

    def _prune_tool_result_store(self) -> None:
        keep = self._protected_tool_result_keys()
        while len(self.session.state.tool_result_store) > self.MAX_COMPLETED_GOAL_TOOL_RESULTS:
            key = next((item for item in self.session.state.tool_result_store if item not in keep), "")
            if not key:
                return
            self.session.state.tool_result_store.pop(key)

    def _protected_tool_result_keys(self) -> set[str]:
        keys = self.blackboard.source_result_keys()
        keys.update(ToolResultContext.blocks_by_key(self.tool_context.kept_results))
        return keys

    def _remember_feedback_error(self, errors: list[str], text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        text = _shorten(text, self.MAX_AGENT_FEEDBACK_ERROR_LEN)
        if text in errors:
            return
        errors.append(text)
        del errors[: max(0, len(errors) - self.MAX_AGENT_FEEDBACK_ERRORS)]

    def _remember_agent_error(self, text: str) -> None:
        self._remember_feedback_error(self.agent_feedback_errors, text)

    def _remember_observe_error(self, text: str) -> None:
        self._remember_feedback_error(self.observe_feedback_errors, text)

    def _format_agent_feedback(self) -> str:
        if not self.agent_feedback_errors:
            return ""
        return "\n".join("- " + error for error in self.agent_feedback_errors)

    def _format_observe_feedback(self) -> str:
        if not self.observe_feedback_errors:
            return ""
        return "\n".join("- " + error for error in self.observe_feedback_errors)

    def _report_gate(self, on_message: MessageCallback | None, message: str, debug_message: str) -> None:
        if on_message is None:
            return
        if message.startswith(("Retrying:", "Continuing:")) and self.session.state.status_notice_until <= time.monotonic():
            self._set_status_notice("err:gate")
        if self.session.settings.debug:
            on_message(debug_message)
            return
        if not message.startswith(("Retrying:", "Continuing:")):
            on_message(message)

    def _format_gate_user_message(self, prefix: str, format_error: str) -> str:
        detail = format_error
        for marker in (". Bad output:", " Bad output:"):
            if marker in detail:
                detail = detail.split(marker, 1)[0]
                break
        if detail.startswith("Invalid model output: "):
            detail = detail[len("Invalid model output: ") :]
        return prefix + ": " + _shorten(detail, 180)

    def _format_gate_debug_details(self, response: Json, format_error: str) -> str:
        bad_output = _json_str(response.get("_format_bad_output"))
        if bad_output is None:
            return _shorten(format_error, 180)
        return _shorten(format_error, 180) + "\nFull bad output:\n" + bad_output

    def step(self, *, on_message: MessageCallback | None = None) -> Json:
        if self.mode == AgentMode.OBSERVE:
            system_prompt = self.prompt_builder.system_prompt(AGENT_OBSERVE_SYSTEM_PROMPT, tools=())
            user_prompt = self.build_observe_prompt()
            activity = "observe"
        else:
            system_prompt = self.prompt_builder.system_prompt(
                AGENT_PLAN_SYSTEM_PROMPT if self.session.settings.plan_mode else None,
                tools=PLAN_MODE_TOOLS if self.session.settings.plan_mode else None,
            )
            user_prompt = self.build_user_prompt()
            activity = "agent"
        response = self.request(system_prompt, user_prompt, activity=activity, on_message=on_message)
        if _json_str(response.get("_format_error")):
            return response
        invalid_response = self._validate_action_response(response)
        if invalid_response is not None:
            return invalid_response
        return response

    def apply_response(self, response: Json) -> list[str]:
        actions = self._response_actions(response)
        if self._start_changes_goal(actions):
            self.tool_context.kept_results = []
            self.tool_context.pending_observe = []
            self.blackboard.hypotheses = []
        self.state_updater.apply(response)
        forgotten = self.tool_context.forget_results(ToolResultContext.forget_result_keys_from_actions(actions))
        if self.mode != AgentMode.OBSERVE and self._has_memory_update_action(actions):
            self._mark_memory_checkpoint()
        return forgotten

    def _start_changes_goal(self, actions: list[Json]) -> bool:
        return any(
            _json_str(action.get("type")) == "start"
            and bool(goal := _json_str(action.get("goal")))
            and goal != self.blackboard.goal
            for action in actions
        )

    def _mark_memory_checkpoint(self, counter: int = 0) -> None:
        checkpoint = counter or self.tool_context.visible_counter(self.mode) or self.session.state.tool_result_counter
        self.blackboard.memory_checkpoint_tool_result_counter = max(self.blackboard.memory_checkpoint_tool_result_counter, checkpoint)
        self.tool_context.mark_checkpoint(self.blackboard.memory_checkpoint_tool_result_counter)

    def _has_memory_update_action(self, actions: list[Json]) -> bool:
        for action in actions:
            action_type = _json_str(action.get("type"))
            if action_type == "keep" and _source_from_json(action):
                return True
            if action_type == "hypothesis" and _json_list(action.get("items")):
                return True
            if action_type == "known" and any(_memory_fact_from_json(raw) for raw in _json_list(action.get("items"))):
                return True
            if action_type == "stable_knowledge" and _json_list(action.get("items")):
                return True
        return False

    def execute_tool_calls(
        self,
        tool_calls: list[JsonValue],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_live_output: ToolLiveOutputCallback | None = None,
        on_live_done: ToolLiveDoneCallback | None = None,
    ) -> str:
        self.tool_runner.execute(
            tool_calls,
            confirm=confirm,
            on_auto_approve=on_auto_approve,
            on_live_output=on_live_output,
            on_live_done=on_live_done,
        )
        self.tool_context.append_latest(
            self.tool_runner.latest_executions,
            max_summaries=self.RECENT_TOOL_CALL_SUMMARIES,
            max_chars=self.RECENT_TOOL_CALL_CHARS,
        )
        self.session.state.turn_tool_calls += len(self.tool_runner.latest_executions)
        self.session.state.session_tool_calls += len(self.tool_runner.latest_executions)
        for execution in self.tool_runner.latest_executions:
            self._after_tool_execution(execution)
        self.runtime.consecutive_tool_turns += 1
        queued = self.tool_context.queue_observation(
            [block for block in self.tool_context.latest if ToolResultContext.is_full_block(block)],
            checkpoint=self.blackboard.memory_checkpoint_tool_result_counter,
        )
        if queued and self._should_observe_after_tools():
            self.mode = AgentMode.OBSERVE
        return "\n\n".join(self.tool_context.latest)

    def _should_observe_after_tools(self) -> bool:
        pending = self.tool_context.pending_observe
        if not pending:
            return False
        if any(self._tool_failure_needs_observe(execution) for execution in self.tool_runner.latest_executions):
            return True
        if len(pending) >= self.PENDING_OBSERVE_RESULTS:
            return True
        if len("\n\n".join(pending)) >= int(self.RECENT_TOOL_CALL_CHARS * self.PENDING_OBSERVE_CHAR_RATIO):
            return True
        return self.runtime.consecutive_tool_turns >= self.PENDING_OBSERVE_TOOL_TURNS

    def _tool_failure_needs_observe(self, execution: ToolCallExecution) -> bool:
        if execution.outcome == "success":
            return False
        if execution.error_type is not None and issubclass(execution.error_type, (ToolCallArgError, Cancellation)):
            return False
        return True

    def _after_tool_execution(self, execution: ToolCallExecution) -> None:
        self._remember_tool_failure(execution)
        if execution.error_type is not None and issubclass(execution.error_type, ToolCallArgError):
            detail = self._format_tool_arg_error(execution)
            rule = "Rule: use the tool signature exactly."
            if execution.call.name in {EditTool.name(), ReplaceRangeTool.name()}:
                rule = "Rule: use ReplaceRange for read ranges or repeated text, and use the exact tool signature."
            self._remember_agent_error(
                "Error: tool call args invalid: "
                + _format_tool_call_summary(execution.call)
                + " -> "
                + detail
                + ". "
                + rule
                + ((" Tool output: " + execution.output) if detail != execution.output else "")
            )
        if execution.requires_verification:
            self.blackboard.verification_required = True
            self.blackboard.task_code = TaskCode.VERIFYING
            self._remember_recent_edit(execution)

    def _remember_tool_failure(self, execution: ToolCallExecution) -> None:
        if execution.outcome != "failure":
            self.failed_tool_call_key = None
            self.failed_tool_call_count = 0
            return
        key = (execution.call.name, tuple(execution.call.args))
        if key == self.failed_tool_call_key:
            self.failed_tool_call_count += 1
        else:
            self.failed_tool_call_key = key
            self.failed_tool_call_count = 1
        if self.failed_tool_call_count >= 2:
            self._remember_agent_error(
                "Error: repeated same failed tool call: "
                + _format_tool_call_summary(execution.call)
                + ". Rule: do not retry the same tool with identical args; correct the args or switch tools."
            )

    def _format_tool_arg_error(self, execution: ToolCallExecution) -> str:
        call = execution.call
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return execution.output
        params = self._exact_signature_params(tool_class.SIGNATURE)
        if not params or len(call.args) == len(params):
            return execution.output
        detail = "got " + str(len(call.args)) + " args, expected " + str(len(params))
        if len(call.args) < len(params):
            detail += ", missing: " + ", ".join(params[len(call.args) :])
        else:
            detail += ", extra: " + str(len(call.args) - len(params))
        return detail

    def _exact_signature_params(self, signature: str) -> list[str]:
        match = re.search(r"\(([^)]*)\)", signature)
        if not match:
            return []
        value = match.group(1)
        if "[" in value or "]" in value or "*" in value or "..." in value:
            return []
        return [part.strip().split("=", 1)[0].strip() for part in value.split(",") if part.strip()]

    def _remember_recent_edit(self, execution: ToolCallExecution) -> None:
        if not execution.call.args:
            return
        filepath = self.session.resolve_path(execution.call.args[0])
        try:
            path = os.path.relpath(filepath, self.session.cwd)
        except ValueError:
            path = filepath
        intention = " ".join(execution.call.intention.split()) or execution.call.name
        self.runtime.recent_edits.append("- " + path + ": " + _shorten(intention, 160))
        self.runtime.recent_edits = self.runtime.recent_edits[-self.RECENT_EDITS :]

    def _invalid_action_response(self, response: Json, reason: str) -> Json:
        return {
            "actions": [],
            "_format_error": "Invalid model output: "
            + reason
            + ". Return action frames only. Bad output: "
            + _shorten(json.dumps(response, ensure_ascii=False)),
        }

    def _validate_action_response(self, response: Json) -> Json | None:
        if not isinstance(response.get("actions"), list):
            return self._invalid_action_response(response, "expected actions array")
        extra_keys = sorted(str(key) for key in response.keys() if key != "actions" and not str(key).startswith("_format_"))
        if extra_keys:
            return self._invalid_action_response(response, "unexpected top-level keys: " + ", ".join(extra_keys))
        return None

    def _format_frame_error_report(self, response: Json) -> str:
        errors = [_json_str(error) or "" for error in _json_list(response.get("_format_frame_errors"))]
        errors = [error for error in errors if error]
        if not errors:
            return ""
        return "Format_Warning: ignored invalid action frame(s).\n" + "\n".join("- " + _shorten(error, 220) for error in errors)

    def _response_actions(self, response: Json) -> list[Json]:
        return [action for action in (_json_dict(item) for item in _json_list(response.get("actions"))) if action]

    def _gate_action_types(
        self,
        actions: list[Json],
        *,
        allowed: set[str],
        on_message: MessageCallback | None,
        retry_message: str,
        feedback_message: str,
        remember_error: Callable[[str], None] | None = None,
    ) -> AgentRunResult | None:
        invalid = sorted({action_type for action_type in (_json_str(action.get("type")) for action in actions) if action_type} - allowed)
        if not invalid:
            return None
        (remember_error or self._remember_agent_error)(feedback_message + " Invalid action(s): " + ", ".join(invalid) + ".")
        self._report_gate(
            on_message,
            retry_message,
            "ActionType_Gate: use action types: " + ", ".join(sorted(allowed)) + "; got: " + ", ".join(invalid) + ".",
        )
        return AgentRunResult()

    def _chat_message_from_actions(self, actions: list[Json]) -> str | None:
        for action in actions:
            action_type = _json_str(action.get("type"))
            if action_type == "chat":
                return _json_str(action.get("text")) or ""
            return None
        return None

    def _progress_messages_from_actions(self, actions: list[Json]) -> list[str]:
        messages = []
        for action in actions:
            if _json_str(action.get("type")) == "progress":
                message = _json_str(action.get("text")) or _json_str(action.get("message")) or ""
            else:
                message = ""
            if message:
                messages.append(message)
        return messages

    def _completion_message_from_actions(self, actions: list[Json]) -> str:
        for action in reversed(actions):
            if _json_str(action.get("type")) == "goal" and action.get("complete") is True:
                return _json_str(action.get("message_for_complete")) or ""
        return ""

    def _incomplete_goal_update_from_actions(self, actions: list[Json]) -> str:
        update = ""
        for action in actions:
            action_type = _json_str(action.get("type"))
            if action_type == "start":
                update = _json_str(action.get("goal")) or update
            elif action_type == "goal" and action.get("complete") is not True:
                update = _json_str(action.get("text")) or update
        return update

    def _has_fresh_plan_action(self, actions: list[Json]) -> bool:
        for action in actions:
            action_type = _json_str(action.get("type"))
            if action_type == "start" and self._has_plan_items(action.get("plan")):
                return True
            if action_type == "plan" and action.get("mode") != "patch" and self._has_plan_items(action.get("items")):
                return True
        return False

    def _has_plan_items(self, value: JsonValue) -> bool:
        return any(_json_str(_json_dict(raw).get("text")) for raw in _json_list(value))

    def _plan_is_complete(self) -> bool:
        return bool(self.blackboard.plan) and all(
            item.status in self.COMPLETED_PLAN_STATUSES and item.context.strip() for item in self.blackboard.plan
        )

    def _verification_is_settled(self) -> bool:
        return self.blackboard.verification.status in {VerificationStatus.DONE, VerificationStatus.BLOCKED}

    def _latest_tool_results_clean(self) -> bool:
        return not any(execution.outcome != "success" for execution in self.tool_runner.latest_executions)

    def _completion_plan_error(self, ctx: ResponseContext) -> str:
        if not self.blackboard.goal_reached:
            return ""
        if not self.blackboard.plan:
            return "plan was removed before completion" if not ctx.plan_was_empty and ctx.has_plan_action else ""
        unfinished = [item for item in self.blackboard.plan if item.status not in self.COMPLETED_PLAN_STATUSES]
        if unfinished:
            return "unfinished plan items: " + self._format_plan_gate_items(unfinished)
        missing_context = [item for item in self.blackboard.plan if not item.context.strip()]
        if missing_context:
            return "plan items missing context: " + self._format_plan_gate_items(missing_context)
        return ""

    def _blocked_verification_completion_error(self) -> str:
        if not self.blackboard.goal_reached or self.blackboard.verification.status != VerificationStatus.BLOCKED:
            return ""
        if self.blackboard.verification.blocker == VerificationBlocker.USER:
            return ""
        return "verify blocked requires blocker=user before completion"

    def _format_plan_gate_items(self, items: list[PlanItem]) -> str:
        rendered = []
        for item in items[:3]:
            label = item.id or item.text
            rendered.append(str(item.status) + " " + _shorten(" ".join(label.split()), 80))
        if len(items) > 3:
            rendered.append("+" + str(len(items) - 3) + " more")
        return "; ".join(rendered)

    def _user_rule_message_from_actions(self, actions: list[Json]) -> str | None:
        for action in actions:
            if _json_str(action.get("type")) == "user_rule":
                return _json_str(action.get("message")) or "Rule saved."
        return None

    def _pending_verification_error(self, actions: list[Json]) -> str:
        if any(_json_str(action.get("type")) == "verify" and _json_str(action.get("status")) == "pending" for action in actions):
            return "status=pending is not supported in single-agent mode"
        return ""

    def _investigate_completion_error(self) -> str:
        if self.blackboard.work_mode != WorkMode.INVESTIGATE or not self.blackboard.goal_reached:
            return ""
        return "" if any(item.status == HypothesisStatus.CONFIRMED for item in self.blackboard.hypotheses) else "investigate completion requires a confirmed hypothesis"

    def _forget_active_hypothesis_error(self, actions: list[Json]) -> str:
        forgotten = set(ToolResultContext.forget_result_keys_from_actions(actions))
        if not forgotten:
            return ""
        released = set()
        for action in actions:
            values = _json_list(action.get("items")) if _json_str(action.get("type")) == "hypothesis" else []
            for raw in values:
                item = Hypothesis.from_json(raw)
                if item is not None and item.status != HypothesisStatus.ACTIVE:
                    released.update(key for key in item.source if key.startswith("tr."))
        protected = {
            key
            for item in self.blackboard.hypotheses
            if item.status == HypothesisStatus.ACTIVE
            for key in item.source
            if key.startswith("tr.")
        }
        conflict = sorted((forgotten & protected) - released)
        return "active hypothesis source: " + ", ".join(conflict) if conflict else ""

    def _plan_items_from_json(self, value: JsonValue) -> list[PlanItem]:
        return [item for item in (self.state_updater._plan_item_from_json(raw) for raw in _json_list(value)) if item]

    def _repeated_tool_retry_error(self, tool_calls: list[JsonValue]) -> str:
        if self.failed_tool_call_key is None or self.failed_tool_call_count < 2:
            return ""
        for value in tool_calls:
            try:
                call = self.tool_runner.parse_tool_call(value)
            except ToolCallArgError:
                continue
            if (call.name, tuple(call.args)) == self.failed_tool_call_key:
                return "same failed tool call repeated after " + str(self.failed_tool_call_count) + " failures: " + _format_tool_call_summary(call)
        return ""

    def _plan_mode_tool_error(self, tool_calls: list[JsonValue]) -> str:
        if not self.session.settings.plan_mode:
            return ""
        for value in tool_calls:
            try:
                call = self.tool_runner.parse_tool_call(value)
            except ToolCallArgError:
                continue
            tool_class = TOOL_REGISTRY.get(call.name)
            if tool_class is None:
                return "plan mode allows registered readonly tools only; blocked " + _format_tool_call_summary(call)
            if tool_class.effect() == ToolEffect.READONLY:
                continue
            if tool_class is GitTool:
                args = call.args[1:] if call.args and call.args[0].startswith("cwd=") else call.args
                if args and args[0] in self.PLAN_MODE_GIT_READONLY:
                    continue
            return "plan mode allows readonly discovery only; blocked " + _format_tool_call_summary(call)
        return ""

    def _build_response_context(self, response: Json) -> ResponseContext:
        actions = self._response_actions(response)
        tool_calls = [action for action in actions if _json_str(action.get("type")) == "tool"]
        pending_verify_requested = any(_json_str(action.get("type")) == "verify" and _json_str(action.get("status")) == "pending" for action in actions)
        progress_messages = self._progress_messages_from_actions(actions)
        has_goal_action = any(_json_str(action.get("type")) in {"goal", "start"} for action in actions)
        has_plan_action = any(_json_str(action.get("type")) in {"plan", "start"} for action in actions)
        has_forget_action = any(_json_str(action.get("type")) == "forget" for action in actions)
        has_hypothesis_action = any(_json_str(action.get("type")) == "hypothesis" for action in actions)
        goal_update = self._incomplete_goal_update_from_actions(actions)
        return ResponseContext(
            response=response,
            actions=actions,
            goal_was_empty=not self.blackboard.goal,
            plan_was_empty=not self.blackboard.plan,
            plan_was_complete=self._plan_is_complete(),
            verification_was_settled=self._verification_is_settled(),
            goal_will_change=bool(self.blackboard.goal and goal_update and goal_update != self.blackboard.goal),
            chat_message=self._chat_message_from_actions(actions),
            tool_calls=tool_calls,
            pending_verify_requested=pending_verify_requested,
            progress_messages=progress_messages,
            user_rule_message=self._user_rule_message_from_actions(actions),
            completion_message=self._completion_message_from_actions(actions),
            has_goal_action=has_goal_action,
            has_plan_action=has_plan_action,
            has_fresh_plan_action=self._has_fresh_plan_action(actions),
            has_user_rule_action=any(_json_str(action.get("type")) == "user_rule" for action in actions),
            state_or_work_requested=bool(tool_calls or pending_verify_requested or progress_messages or has_plan_action or has_forget_action or has_hypothesis_action),
        )

    def _handle_chat_response(self, ctx: ResponseContext, on_message: MessageCallback | None) -> AgentRunResult | None:
        if ctx.chat_message is None:
            return None
        self.blackboard.task_code = TaskCode.DONE
        self.session.append_conversation(AssistantMessage(content=ctx.chat_message))
        if on_message is not None:
            on_message(ctx.chat_message)
        return AgentRunResult(done=True, value=ctx.response)

    def _gate_before_apply(self, ctx: ResponseContext, on_message: MessageCallback | None) -> bool:
        action_gate = self._gate_action_types(
            ctx.actions,
            allowed=self.PLAN_ACTION_TYPES if self.session.settings.plan_mode else self.ACT_ACTION_TYPES,
            on_message=on_message,
            retry_message="Retrying: use a valid agent action.",
            feedback_message="Error: this step only accepts agent work actions.",
        )
        if action_gate is not None:
            return True
        forget_error = self._forget_tool_result_error(ctx.actions)
        if forget_error:
            self._remember_agent_error("Error: forget is invalid: " + forget_error + ". Rule: forget only visible tool result source keys.")
            self._report_gate(
                on_message,
                "Retrying: forget only visible tool result keys.",
                "ToolResult_Gate: " + forget_error + ".",
            )
            return True
        forget_hypothesis_error = self._forget_active_hypothesis_error(ctx.actions)
        if forget_hypothesis_error:
            self._remember_agent_error(
                "Error: forget would remove a tool result used by an active hypothesis: "
                + forget_hypothesis_error
                + ". Rule: mark the hypothesis ruled_out, dropped, or confirmed before forgetting its source."
            )
            self._report_gate(
                on_message,
                "Retrying: close hypothesis before forgetting its source result.",
                "ToolResult_Gate: " + forget_hypothesis_error + ".",
            )
            return True
        repeated_tool_retry_error = self._repeated_tool_retry_error(ctx.tool_calls)
        if repeated_tool_retry_error:
            self._remember_agent_error(
                "Error: repeated failed tool call is blocked: "
                + repeated_tool_retry_error
                + ". Rule: correct the args or switch tools; for local edit failures, prefer ReplaceRange after Read."
            )
            self._report_gate(
                on_message,
                "Retrying: change the failed tool call instead of repeating it.",
                "ToolRetry_Gate: " + repeated_tool_retry_error + ".",
            )
            return True
        plan_mode_tool_error = self._plan_mode_tool_error(ctx.tool_calls)
        if plan_mode_tool_error:
            self._remember_agent_error("Error: " + plan_mode_tool_error + ". Rule: produce a proposed plan without executing mutations.")
            self._report_gate(
                on_message,
                "Retrying: plan mode only allows readonly discovery.",
                "PlanMode_Gate: " + plan_mode_tool_error + ".",
            )
            return True
        if self.blackboard.task_code != TaskCode.NEW and any(_json_str(action.get("type")) == "start" for action in ctx.actions):
            self._remember_agent_error(
                "Error: repeated start is invalid after the current task is active. Rule: follow Current Task Code and continue with plan/tool/verify/goal."
            )
            self._report_gate(
                on_message,
                "Retrying: current task is already active; continue without start.",
                "GoalPlan_Gate: repeated start while task code is " + self.blackboard.task_code + ".",
            )
            return True
        if self.blackboard.task_code != TaskCode.NEW and ctx.goal_will_change and not ctx.has_fresh_plan_action:
            self._remember_agent_error(
                "Error: rewriting Goal is invalid after the current task is active. Rule: follow Current Task Code and continue the existing Goal/Plan."
            )
            self._report_gate(
                on_message,
                "Retrying: current task is already active; continue without rewriting goal.",
                "GoalPlan_Gate: goal rewrite while task code is " + self.blackboard.task_code + ".",
            )
            return True
        pending_verification_error = self._pending_verification_error(ctx.actions)
        if pending_verification_error:
            self._remember_agent_error(
                "Error: pending verify is invalid: "
                + pending_verification_error
                + '. Rule: run verification with tool actions directly, then return verify status="passed"|"failed"|"blocked".'
            )
            self._report_gate(
                on_message,
                "Retrying: run verification tools directly.",
                "Verification_Gate: pending verify is invalid: " + pending_verification_error + ".",
            )
            return True
        if ctx.goal_was_empty and not ctx.has_goal_action and ctx.state_or_work_requested:
            self._remember_agent_error(
                "Error: started task state/work before Goal and Plan were ready. Rule: set goal complete=false and create a short plan before tools."
            )
            self._report_gate(
                on_message,
                "Retrying: set goal and plan before tools.",
                "GoalPlan_Gate: Goal is empty before task state/work.",
            )
            return True
        if ctx.goal_will_change and not ctx.has_fresh_plan_action and (ctx.tool_calls or ctx.pending_verify_requested):
            self._remember_agent_error("Error: changed Goal without replacing Plan. Rule: include start.plan or a full plan action with the new goal.")
            self._report_gate(
                on_message,
                "Retrying: new goal requires a fresh plan.",
                "GoalPlan_Gate: Goal changed without replacing Plan.",
            )
            return True
        return False

    def _emit_debug_frame_errors(self, response: Json, on_message: MessageCallback | None) -> None:
        if not self.session.settings.debug or on_message is None:
            return
        frame_error_report = self._format_frame_error_report(response)
        if frame_error_report:
            on_message(frame_error_report)

    def _emit_state_and_progress(self, ctx: ResponseContext, on_message: MessageCallback | None) -> None:
        if on_message is not None and self.state_updater.latest_report:
            report = self.state_updater.latest_report if self.session.settings.debug else self.state_updater.compact_report()
            if report:
                on_message(report)
        if on_message is not None:
            for message in ctx.progress_messages:
                on_message(message)

    def _gate_after_apply(self, ctx: ResponseContext, on_message: MessageCallback | None) -> AgentRunResult | None:
        if ctx.plan_was_empty and not self.blackboard.plan and (ctx.tool_calls or ctx.pending_verify_requested):
            self._remember_agent_error("Error: attempted tool/verify while Plan is empty. Rule: create a short plan first, then do the next smallest step.")
            self._report_gate(
                on_message,
                "Retrying: create a short plan before tools.",
                "GoalPlan_Gate: Plan is empty before tool/verify.",
            )
            return AgentRunResult()

        if ctx.tool_calls and self._latest_tool_results_clean() and self._verification_is_settled():
            if self._plan_is_complete():
                self._remember_agent_error(
                    "Error: Plan and verification are complete. Rule: finish with goal.complete=true; "
                    "if more tools are needed, first reopen Plan with a todo/doing item and context explaining why completion is insufficient."
                )
                self._report_gate(
                    on_message,
                    "Retrying: finish the completed task or reopen the plan before more tools.",
                    "Completion_Gate: completed plan and verification cannot continue tools without reopening Plan.",
                )
                return AgentRunResult()
            if ctx.plan_was_complete and ctx.verification_was_settled:
                missing_context = [
                    item
                    for item in self.blackboard.plan
                    if item.status not in self.COMPLETED_PLAN_STATUSES and not item.context.strip()
                ]
                if missing_context:
                    self._remember_agent_error(
                        "Error: continuing after completed Plan requires a reopened todo/doing Plan item with context. "
                        "Rule: explain why existing completion is insufficient before tool calls."
                    )
                    self._report_gate(
                        on_message,
                        "Retrying: reopen the plan with context before more tools.",
                        "Completion_Gate: reopened plan item missing context: " + self._format_plan_gate_items(missing_context) + ".",
                    )
                    return AgentRunResult()

        if (
            not ctx.tool_calls
            and not self.blackboard.goal_reached
            and self.blackboard.verification.status in (VerificationStatus.DONE, VerificationStatus.BLOCKED)
        ):
            self._remember_agent_error(
                "Error: verification is done but goal.complete is not true. Rule: if finished, return goal complete=true with message_for_complete; otherwise continue with tool/plan/verify."
            )
            self._report_gate(
                on_message,
                "Retrying: verification is done but goal is not complete.",
                "Completion_Gate: verification is done but goal.complete is not true.",
            )
            return AgentRunResult()
        if not ctx.tool_calls and not ctx.plan_was_complete and self._plan_is_complete() and not self.blackboard.goal_reached:
            if not self._verification_is_settled():
                self._remember_agent_error(
                    'Error: Plan is complete but verification is not recorded. Rule: return verify status="passed"|"blocked" with context, or reopen Plan before more work.'
                )
                self._report_gate(
                    on_message,
                    "Retrying: record verification before completing.",
                    "Completion_Gate: completed plan requires verification status.",
                )
                return AgentRunResult()
            self._remember_agent_error(
                "Error: Plan and verification are complete but goal.complete is not true. Rule: finish with goal.complete=true and message_for_complete."
            )
            self._report_gate(
                on_message,
                "Retrying: finish the completed task.",
                "Completion_Gate: completed plan and verification require goal.complete=true.",
            )
            return AgentRunResult()
        if (
            ctx.state_or_work_requested
            and not ctx.tool_calls
            and not ctx.pending_verify_requested
            and not ctx.progress_messages
            and not ctx.completion_message
            and not self.state_updater.changed
        ):
            self._remember_agent_error(
                "Error: response made no effective state change. Rule: do not repeat state updates; continue with tool, verify, or goal."
            )
            self._report_gate(
                on_message,
                "Retrying: continue with tool, verify, or goal.",
                "Progress_Gate: state-only response made no effective change.",
            )
            return AgentRunResult()
        return None

    def _plan_mode_completion_error(self, message: str) -> str:
        if not self.session.settings.plan_mode:
            return ""
        text = message.strip()
        if not text.startswith("<proposed_plan>") or not text.endswith("</proposed_plan>"):
            return "final plan must be wrapped in <proposed_plan>...</proposed_plan>"
        if text.count("<proposed_plan>") != 1 or text.count("</proposed_plan>") != 1:
            return "final plan must contain exactly one proposed_plan block"
        if not text.removeprefix("<proposed_plan>").removesuffix("</proposed_plan>").strip():
            return "final plan block is empty"
        return ""

    def _promote_required_verification(self, ctx: ResponseContext) -> None:
        verification = self.blackboard.verification
        if not self.blackboard.verification_required or not self.blackboard.goal_reached:
            return
        if verification.status in {VerificationStatus.REQUIRED, VerificationStatus.DONE, VerificationStatus.BLOCKED}:
            return
        self.blackboard.task_code = TaskCode.VERIFYING
        verification.status = VerificationStatus.REQUIRED
        verification.kind = verification.kind or "change_syntax_check"
        verification.method = verification.method or self.blackboard.goal or self.blackboard.user_input
        if not verification.criteria:
            verification.criteria = ["changed files pass the smallest relevant syntax or compile check"]
        verification.context = verification.context or ctx.completion_message or self.blackboard.goal

    def _run_tool_actions(
        self,
        ctx: ResponseContext,
        *,
        confirm: ConfirmCallback | None,
        on_auto_approve: ToolDisplayCallback | None,
        on_live_output: ToolLiveOutputCallback | None,
        on_live_done: ToolLiveDoneCallback | None,
        on_message: MessageCallback | None,
    ) -> bool:
        if not ctx.tool_calls:
            return False
        self.execute_tool_calls(
            ctx.tool_calls,
            confirm=confirm,
            on_auto_approve=on_auto_approve,
            on_live_output=on_live_output,
            on_live_done=on_live_done,
        )
        if on_message is not None:
            report = ToolCallDisplayFormatter.latest_report(self.tool_runner.latest_executions)
            if report:
                on_message(report)
            if self.session.settings.debug and self.tool_runner.skipped_after_failure_count:
                on_message(
                    "Tool Calls Skipped: "
                    + str(self.tool_runner.skipped_after_failure_count)
                    + " after "
                    + self.tool_runner.skipped_after_failure_key
                    + " failed"
                )
        self.compactor.maybe_compact()
        return True

    def _handle_observe_response(
        self,
        ctx: ResponseContext,
        response: Json,
        *,
        on_message: MessageCallback | None,
    ) -> AgentRunResult:
        repeated_tool_retry_error = self._repeated_tool_retry_error(ctx.tool_calls)
        if repeated_tool_retry_error:
            self._remember_observe_error(
                "Error: repeated failed tool call is blocked: "
                + repeated_tool_retry_error
                + ". Rule: first observe latest results, then correct the args or switch tools."
            )
            self._report_gate(
                on_message,
                "Retrying: change the failed tool call instead of repeating it.",
                "ToolRetry_Gate: " + repeated_tool_retry_error + ".",
            )
            return AgentRunResult()
        gate_result = self._gate_action_types(
            ctx.actions,
            allowed=self.OBSERVE_ACTION_TYPES,
            on_message=on_message,
            retry_message="Retrying: observe latest results.",
            feedback_message="Error: latest results must be observed before more work.",
            remember_error=self._remember_observe_error,
        )
        if gate_result is not None:
            return gate_result
        forget_error = self._forget_tool_result_error(ctx.actions)
        if forget_error:
            self._remember_observe_error("Error: forget is invalid: " + forget_error + ". Rule: forget only visible tool result source keys.")
            self._report_gate(
                on_message,
                "Retrying: forget only visible tool result keys.",
                "ToolResult_Gate: " + forget_error + ".",
            )
            return AgentRunResult()
        forget_hypothesis_error = self._forget_active_hypothesis_error(ctx.actions)
        if forget_hypothesis_error:
            self._remember_observe_error(
                "Error: forget would remove a tool result used by an active hypothesis: "
                + forget_hypothesis_error
                + ". Rule: mark the hypothesis ruled_out, dropped, or confirmed before forgetting its source."
            )
            self._report_gate(
                on_message,
                "Retrying: close hypothesis before forgetting its source result.",
                "ToolResult_Gate: " + forget_hypothesis_error + ".",
            )
            return AgentRunResult()
        if any(_json_str(action.get("type")) == "verify" and _json_str(action.get("status")) == "pending" for action in ctx.actions):
            self._remember_observe_error("Error: cannot request new verification before observing latest results. Rule: keep or forget latest results first.")
            self._report_gate(
                on_message,
                "Retrying: observe latest results before new verification.",
                "Verification_Gate: verify status=pending is not allowed while observing latest results.",
            )
            return AgentRunResult()
        if not ctx.actions:
            self._remember_observe_error("Error: observe returned no actions. Rule: keep useful tool results or forget latest raw results with a reason.")
            self._report_gate(
                on_message,
                "Retrying: keep or forget latest results.",
                "Observe_Gate: empty actions are not a checkpoint; return keep or forget.",
            )
            return AgentRunResult()
        observed_blocks = list(self.tool_context.pending_observe)
        observed_counter = ToolResultContext.max_counter(observed_blocks)
        covered = {
            key
            for action in ctx.actions
            if _json_str(action.get("type")) in {"keep", "forget"}
            for key in _source_from_json(action)
            if key.startswith("tr.")
        }
        missing_observe_keys = [key for key in ToolResultContext.blocks_by_key(observed_blocks) if key not in covered]
        if missing_observe_keys:
            self._remember_observe_error(
                "Error: observe did not cover latest result keys: "
                + ", ".join(missing_observe_keys)
                + ". Rule: every latest result key must be covered by keep or forget."
            )
            self._report_gate(
                on_message,
                "Retrying: cover every latest result key with keep or forget.",
                "Observe_Gate: missing coverage for result keys: " + ", ".join(missing_observe_keys) + ".",
            )
            return AgentRunResult()
        self._emit_debug_frame_errors(response, on_message)
        forgotten_keys = self.apply_response(response)
        self._emit_state_and_progress(ctx, on_message)
        kept_keys: list[str] = []
        if any(_json_str(action.get("type")) in {"keep", "forget", "known", "stable_knowledge"} for action in ctx.actions):
            self.mode = AgentMode.ACT
            self.runtime.consecutive_tool_turns = 0
            kept_keys = self.tool_context.keep_results(ctx.actions, observed_blocks, max_chars=self.KEPT_TOOL_RESULT_CHARS)
            self.tool_context.compact_observed(observed_blocks)
            self._mark_memory_checkpoint(observed_counter)
            self.observe_feedback_errors = []
        else:
            self.mode = AgentMode.OBSERVE
        self._emit_tool_context_update(kept_keys, forgotten_keys, on_message)
        self._promote_required_verification(ctx)
        return AgentRunResult()

    def _forget_tool_result_error(self, actions: list[Json]) -> str:
        keys = ToolResultContext.forget_result_keys_from_actions(actions)
        if not any(_json_str(action.get("type")) == "forget" for action in actions):
            return ""
        if not keys:
            return "missing tr.* source"
        visible_keys = set(
            ToolResultContext.blocks_by_key(
                self.tool_context.kept_results + self.tool_context.pending_observe + self.tool_context.latest + self.tool_context.recent
            )
        )
        missing = [key for key in keys if key not in visible_keys]
        return "not in visible tool results: " + ", ".join(missing) if missing else ""

    def _emit_tool_context_update(self, kept: list[str], forgotten: list[str], on_message: MessageCallback | None) -> None:
        if on_message is None or not (kept or forgotten):
            return
        parts = []
        if kept:
            parts.append(" ".join("+" + key for key in kept))
        if forgotten:
            parts.append(" ".join("-" + key for key in forgotten))
        on_message("Tool Result Context: " + " / ".join(parts))

    def _finish_or_continue(self, ctx: ResponseContext, on_message: MessageCallback | None) -> AgentRunResult:
        if self.blackboard.verification.status == VerificationStatus.REQUIRED:
            self.blackboard.goal_reached = False
            if self.blackboard.verification_required:
                self._remember_agent_error(
                    'Error: edited files must be verified before completion. Rule: run the smallest relevant check, then return verify status="passed"|"blocked" with context before goal complete=true.'
                )
                retry_message = "Retrying: verify edited files before completion."
                debug_message = "Verification_Gate: edit completion requires verification."
            else:
                self._remember_agent_error(
                    'Error: completion is blocked until verification passes or is blocked. Rule: run the needed verification tool, then return verify status="passed"|"blocked" with context before goal complete=true.'
                )
                retry_message = "Retrying: verification is required before completion."
                debug_message = "Verification_Gate: retrying until verification is passed or blocked."
            self._report_gate(
                on_message,
                retry_message,
                debug_message,
            )
            return AgentRunResult()
        if self.blackboard.verification.status == VerificationStatus.FAILED and self.blackboard.goal_reached:
            self.blackboard.goal_reached = False
            self._report_gate(
                on_message,
                "Retrying: verification failed; fix the reported issue first.",
                "Verification_Gate: verification failed; fix before completion.",
            )
            return AgentRunResult()
        completion_plan_error = self._completion_plan_error(ctx)
        if completion_plan_error:
            self.blackboard.goal_reached = False
            self._remember_agent_error(
                "Error: returned goal.complete=true before Plan was complete. Rule: every existing Plan item must be done or blocked with result context before completion."
            )
            self._report_gate(
                on_message,
                "Retrying: finish the plan before completing.",
                "Completion_Gate: " + completion_plan_error + ".",
            )
            return AgentRunResult()
        blocked_completion_error = self._blocked_verification_completion_error()
        if blocked_completion_error:
            self.blackboard.goal_reached = False
            self._remember_agent_error(
                "Error: returned goal.complete=true with verify blocked, but "
                + blocked_completion_error
                + ". Rule: continue verification if possible; only complete blocked verification when blocker=user."
            )
            self._report_gate(
                on_message,
                "Retrying: blocked verification needs blocker=user.",
                "Verification_Gate: " + blocked_completion_error + ".",
            )
            return AgentRunResult()
        investigate_completion_error = self._investigate_completion_error()
        if investigate_completion_error:
            self.blackboard.goal_reached = False
            self._remember_agent_error("Error: " + investigate_completion_error + ". Rule: mark a hypothesis confirmed before completing.")
            self._report_gate(
                on_message,
                "Retrying: confirm a hypothesis before completing.",
                "Completion_Gate: " + investigate_completion_error + ".",
            )
            return AgentRunResult()
        if self.blackboard.goal_reached and not ctx.completion_message:
            self.blackboard.goal_reached = False
            self._remember_agent_error(
                "Error: returned goal.complete=true without message_for_complete. Rule: finish with goal complete=true and non-empty message_for_complete."
            )
            self._report_gate(
                on_message,
                "Retrying: goal is complete but message_for_complete is missing.",
                "Completion_Gate: goal.complete=true requires non-empty message_for_complete.",
            )
            return AgentRunResult()
        plan_mode_completion_error = self._plan_mode_completion_error(ctx.completion_message) if self.blackboard.goal_reached else ""
        if plan_mode_completion_error:
            self.blackboard.goal_reached = False
            self._remember_agent_error(
                "Error: invalid plan-mode completion: " + plan_mode_completion_error + ". Rule: return the proposed plan as the final message."
            )
            self._report_gate(
                on_message,
                "Retrying: finish plan mode with a proposed_plan block.",
                "PlanMode_Gate: " + plan_mode_completion_error + ".",
            )
            return AgentRunResult()
        if self.blackboard.goal_reached:
            self.session.append_conversation(AssistantMessage(content=ctx.completion_message))
            if on_message is not None:
                on_message(ctx.completion_message)
            self._finish_current_goal()
            return AgentRunResult(done=True, value=ctx.response)
        self.blackboard.goal_reached = False
        if not ctx.actions:
            self._remember_agent_error(
                "Error: returned no actions while the goal is incomplete. Rule: continue with a useful agent action and optional progress field, or final goal action."
            )
            self._report_gate(
                on_message,
                "Continuing: assistant must set current task's goal.",
                "GoalPlan_Gate: goal not reached; retrying next useful action.",
            )
        return AgentRunResult()

    def run(
        self,
        user_input: str,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_live_output: ToolLiveOutputCallback | None = None,
        on_live_done: ToolLiveDoneCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> Json:
        self.agent_feedback_errors = []
        self.failed_tool_call_key = None
        self.failed_tool_call_count = 0
        self.runtime.consecutive_tool_turns = 0
        self.tool_context.prune_recent(max_summaries=self.RECENT_TOOL_CALL_SUMMARIES, max_chars=self.RECENT_TOOL_CALL_CHARS)
        self.tool_context.pending_observe = []
        self._prune_tool_result_store()
        # Range fingerprints are tied to previously read file content; require a fresh read before later edits.
        self.session.state.range_fingerprints.clear()
        self.mode = AgentMode.ACT
        self.session.state.turn_tool_calls = 0
        self.session.state.turn_model_calls = 0
        self.blackboard.user_input = user_input
        previous_task_done = self.blackboard.task_code == TaskCode.DONE
        if previous_task_done:
            self.blackboard.work_mode = WorkMode.NORMAL
        self.blackboard.task_code = TaskCode.NEW
        self.blackboard.goal_reached = False
        self.blackboard.verification_required = False
        self.observe_feedback_errors = []
        self.blackboard.verification.reset()
        self.compactor.maybe_compact()
        self.session.append_conversation(UserMessage(content=user_input))

        return self.run_loop(
            max_steps=self.session.settings.max_agent_steps,
            on_message=on_message,
            on_step=lambda response: self.handle_response(
                response,
                confirm=confirm,
                on_auto_approve=on_auto_approve,
                on_live_output=on_live_output,
                on_live_done=on_live_done,
                on_message=on_message,
            ),
            on_step_limit=lambda: (_ for _ in ()).throw(LLMError("agent step limit reached")),
        )

    def handle_response(
        self,
        response: Json,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_live_output: ToolLiveOutputCallback | None = None,
        on_live_done: ToolLiveDoneCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> AgentRunResult:
        ctx = self._build_response_context(response)
        if self.mode == AgentMode.OBSERVE:
            return self._handle_observe_response(
                ctx,
                response,
                on_message=on_message,
            )

        if self._gate_before_apply(ctx, on_message):
            return AgentRunResult()

        chat_result = self._handle_chat_response(ctx, on_message)
        if chat_result is not None:
            return chat_result

        self._emit_debug_frame_errors(response, on_message)
        forgotten_keys = self.apply_response(response)
        self._emit_state_and_progress(ctx, on_message)
        self._emit_tool_context_update([], forgotten_keys, on_message)
        if ctx.has_user_rule_action and not ctx.tool_calls and not ctx.pending_verify_requested:
            message = ctx.user_rule_message or "Rule saved."
            self.session.append_conversation(AssistantMessage(content=message))
            if on_message is not None:
                on_message(message)
            self._finish_current_goal()
            return AgentRunResult(done=True, value=response)

        gate_result = self._gate_after_apply(ctx, on_message)
        if gate_result is not None:
            return gate_result

        self._promote_required_verification(ctx)
        if self._run_tool_actions(
            ctx,
            confirm=confirm,
            on_auto_approve=on_auto_approve,
            on_live_output=on_live_output,
            on_live_done=on_live_done,
            on_message=on_message,
        ):
            return AgentRunResult()

        self.runtime.consecutive_tool_turns = 0
        return self._finish_or_continue(ctx, on_message)


############################
# Commands
############################


class CommandStatus(StrEnum):
    HANDLED = "handled"
    EXIT = "exit"
    UNHANDLED = "unhandled"


@dataclass(frozen=True)
class CommandResult:
    status: CommandStatus
    message: str = ""


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    category: str
    usage: str


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "Show commands or ask about nanocode", "Info", "/help [question]"),
    CommandSpec("/status", "Show session status", "Info", "/status"),
    CommandSpec("/rules", "Show long-term user rules", "Info", "/rules"),
    CommandSpec("/knowledge", "Show stable knowledge", "Info", "/knowledge"),
    CommandSpec("/compact", "Compact conversation history", "Info", "/compact"),
    CommandSpec("/config", "Show resolved runtime config", "Config", "/config"),
    CommandSpec("/set", "Set a runtime config override", "Config", "/set <key> <value>"),
    CommandSpec("/model", "Show or set model and reasoning", "Config", "/model [model_name]"),
    CommandSpec("/reason", "Set reasoning effort", "Config", "/reason"),
    CommandSpec("/provider", "Show or switch provider", "Config", "/provider [name]"),
    CommandSpec("/plan", "Toggle plan mode or ask for a readonly plan", "Config", "/plan [on|off|question]"),
    CommandSpec("/yolo", "Toggle yolo mode (skip confirmations)", "Config", "/yolo"),
    CommandSpec("/clean", "Clean all session tool result logs", "Maintenance", "/clean"),
    CommandSpec("/exit", "Exit nanocode", "Control", "/exit"),
    CommandSpec("/quit", "Exit nanocode", "Control", "/quit"),
)


############################
# Runtime Config Keys
############################


CONFIG_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
CONFIG_PROVIDER_ATTRS: dict[str, str] = {
    "provider.model": "model",
    "provider.reasoning": "reasoning",
    "provider.effort": "reasoning_effort",
    "provider.stream": "stream",
    "provider.temperature": "temperature",
    "provider.timeout": "timeout",
    "provider.first_token_timeout": "first_token_timeout",
}
CONFIG_RUNTIME_ATTRS: dict[str, str] = {
    "runtime.compact_at": "compact_at",
    "runtime.shell_timeout": "shell_timeout",
    "runtime.max_agent_steps": "max_agent_steps",
    "runtime.plan_timeout": "plan_timeout",
    "runtime.plan_first_token_timeout": "plan_first_token_timeout",
    "runtime.yolo": "yolo",
}
CONFIG_SET_KEYS: tuple[str, ...] = tuple(CONFIG_PROVIDER_ATTRS) + tuple(CONFIG_RUNTIME_ATTRS)
CONFIG_VALUE_COMPLETIONS: dict[str, tuple[str, ...]] = {
    "provider.reasoning": ("on", "off"),
    "provider.effort": CONFIG_EFFORTS,
    "provider.stream": ("on", "off"),
    "provider.temperature": ("off",),
    "runtime.yolo": ("on", "off"),
}
CONFIG_BOOL_KEYS: set[str] = {"provider.reasoning", "provider.stream", "runtime.yolo"}
CONFIG_INT_KEYS: set[str] = {
    "provider.timeout",
    "provider.first_token_timeout",
    "runtime.compact_at",
    "runtime.shell_timeout",
    "runtime.max_agent_steps",
    "runtime.plan_timeout",
    "runtime.plan_first_token_timeout",
}
CONFIG_SET_USAGE = "Usage: /set <key> <value>"


class CommandDispatcher:
    MODEL_CONFIGURED_LABEL = "---- Configured models ----"
    MODEL_DISCOVERED_LABEL = "---- Discovered models ----"
    MODEL_LABELS = frozenset((MODEL_CONFIGURED_LABEL, MODEL_DISCOVERED_LABEL))

    def __init__(
        self,
        agent: Agent,
        run_agent: MessageCallback | None = None,
        run_with_status: StatusRunner | None = None,
        select_reasoning: ReasoningSelector | None = None,
        select_model: ModelSelector | None = None,
        select_provider: ProviderSelector | None = None,
    ):
        self.agent = agent
        self.run_agent = run_agent
        self.run_with_status = run_with_status
        self.select_reasoning = select_reasoning
        self.select_model = select_model
        self.select_provider = select_provider
        self.handlers: dict[str, Callable[[str], str]] = {
            "/help": self._help,
            "/status": self._status,
            "/rules": self._rules,
            "/compact": self._compact,
            "/config": self._config,
            "/set": self._set,
            "/clean": self._clean,
            "/model": self._model,
            "/reason": self._reason,
            "/provider": self._provider,
            "/plan": self._plan,
            "/yolo": self._yolo,
            "/knowledge": self._knowledge,
        }

    def dispatch(self, user_input: str) -> CommandResult:
        command, _, args = user_input.strip().partition(" ")
        args = args.strip()
        if command in {"/exit", "/quit", "exit", "quit"}:
            return CommandResult(CommandStatus.EXIT, "Exit")
        handler = self.handlers.get(command)
        if handler is None:
            return CommandResult(CommandStatus.UNHANDLED, "")
        return CommandResult(CommandStatus.HANDLED, handler(args))

    def _help(self, args: str) -> str:
        if args:
            question = self._format_source_help_question(args)
            if self.run_agent is not None:
                self.run_agent(question)
            else:
                self.agent.run(question)
            return ""
        lines = ["Commands:"]
        current_category = ""
        for spec in COMMANDS:
            if spec.category != current_category:
                current_category = spec.category
                lines.append(current_category + ":")
            lines.append("  " + spec.usage + " - " + spec.description)
        lines.append("")
        lines.append("Tip: use @path to autocomplete file paths in prompts.")
        return "\n".join(lines)

    def _format_source_help_question(self, question: str) -> str:
        source_path = os.path.abspath(__file__)
        project_metadata = os.path.join(os.path.dirname(source_path), "pyproject.toml")
        return "\n".join(
            [
                "Answer this question about nanocode itself.",
                "First inspect the nanocode source file at this exact path:",
                source_path,
                "Inspect this project metadata file too when useful, if it exists:",
                project_metadata,
                "Base the answer on the source you inspected, cite concrete functions/classes/options when relevant, and keep the answer concise.",
                "",
                "Question:",
                question,
            ]
        )

    def _model(self, args: str) -> str:
        model = args.strip()
        if not model:
            provider = self.agent.session.config.provider
            models = self._model_choices(provider)
            if models and self.select_model is not None:
                while True:
                    selected = self.select_model(models, provider.model)
                    if selected is SELECTION_BACK:
                        return "No change"
                    if not isinstance(selected, str):
                        return self._set("provider.model")
                    if selected in self.MODEL_LABELS:
                        continue
                    result = self._set_model(selected, back_to_model=True)
                    if result is SELECTION_BACK:
                        continue
                    return result
            return self._set("provider.model")
        if " " in model:
            return "Usage: /model [model_name]"
        return self._set_model(model)

    def _model_choices(self, provider: ProviderConfig) -> tuple[str, ...]:
        configured = provider.available_models
        remote = tuple(model for model in self._fetch_remote_models(provider) if model not in configured)
        choices: list[str] = []
        if configured:
            choices.extend((self.MODEL_CONFIGURED_LABEL, *configured))
        if remote:
            choices.extend((self.MODEL_DISCOVERED_LABEL, *remote))
        return tuple(choices)

    def _fetch_remote_models(self, provider: ProviderConfig) -> tuple[str, ...]:
        if not provider.url or not provider.key:
            return ()
        base_url = provider.url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        request = urllib.request.Request(
            base_url + "/models",
            headers={"Authorization": "Bearer " + provider.key, "User-Agent": HTTP_USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return ()
        ids = []
        for item in _json_list(_json_dict(data).get("data")):
            model_id = _json_dict(item).get("id")
            if isinstance(model_id, str) and model_id:
                ids.append(model_id)
        return tuple(dict.fromkeys(sorted(ids)))

    def _set_model(self, model: str, *, back_to_model: bool = False) -> str | SelectionBack:
        message = "Set provider.model = " + model
        choice = self.select_reasoning() if self.select_reasoning is not None else None
        if choice is SELECTION_BACK:
            return SELECTION_BACK if back_to_model else "No change"
        self.agent.session.config.provider.model = model
        return message + (("\n" + self._apply_reasoning_choice(choice)) if isinstance(choice, str) else "")

    def _reason(self, args: str) -> str:
        if args.strip():
            return "Usage: /reason"
        if self.select_reasoning is None:
            return "Reasoning selection not available"
        choice = self.select_reasoning()
        if not isinstance(choice, str):
            return "No change"
        return self._apply_reasoning_choice(choice)

    def _apply_reasoning_choice(self, choice: str) -> str:
        provider = self.agent.session.config.provider
        if choice == "off":
            provider.reasoning = False
            return "Set provider.reasoning = off"
        if choice not in CONFIG_EFFORTS:
            return "Invalid reasoning effort: " + choice
        provider.reasoning = True
        provider.reasoning_effort = choice
        return "Set provider.reasoning = on\nSet provider.effort = " + choice

    def _provider(self, args: str) -> str:
        name = args.strip()
        config = self.agent.session.config
        providers = ", ".join(sorted(config.providers))
        if not name:
            if self.select_provider is not None:
                selected = self.select_provider(tuple(config.providers), config.active_provider)
                if isinstance(selected, str):
                    return self._set_provider(selected)
            return "provider: " + config.active_provider + "\nproviders: " + providers
        if " " in name:
            return "Usage: /provider [name]"
        if name not in config.providers:
            return "Unknown provider: " + name + "\nproviders: " + providers
        return self._set_provider(name)

    def _set_provider(self, name: str) -> str:
        config = self.agent.session.config
        if name not in config.providers:
            return "Unknown provider: " + name + "\nproviders: " + ", ".join(sorted(config.providers))
        config.active_provider = name
        return "Set provider = " + name

    def _yolo(self, args: str) -> str:
        if not args.strip():
            current = self.agent.session.settings.yolo
            return self._set("runtime.yolo " + ("off" if current else "on"))
        return self._set("runtime.yolo " + args)

    def _plan(self, args: str) -> str:
        text = args.strip()
        if not text:
            current = self.agent.session.settings.plan_mode
            self.agent.session.settings.plan_mode = not current
            return "Set plan mode = " + self._format_bool(self.agent.session.settings.plan_mode)
        if text in {"on", "off"}:
            self.agent.session.settings.plan_mode = text == "on"
            return "Set plan mode = " + text
        previous = self.agent.session.settings.plan_mode
        self.agent.session.settings.plan_mode = True
        try:
            if self.run_agent is not None:
                self.run_agent(text)
            else:
                self.agent.run(text)
        finally:
            self.agent.session.settings.plan_mode = previous
        return ""

    def _rules(self, args: str) -> str:
        if args:
            return "Usage: /rules"
        return self.agent.session.state.user_rules.format()

    def _status(self, args: str) -> str:
        if args:
            return "Usage: /status"
        session = self.agent.session
        blackboard = self.agent.blackboard
        provider = session.config.provider
        reasoning = provider.reasoning_effort if provider.reasoning else "off"
        model_usage = (
            "\n".join(
                "  " + (model.rsplit("/", 1)[-1] or model) + ": calls=" + str(usage.calls) + " tokens=" + _format_count(usage.total_tokens)
                for model, usage in session.state.model_usage.items()
            )
            if session.state.model_usage
            else "  (empty)"
        )
        verification_status = blackboard.verification.status
        return "\n".join(
            [
                "provider: " + session.config.active_provider,
                "model: " + (provider.model or "(empty)") + " reasoning=" + (reasoning or "(empty)") + " stream=" + self._format_bool(provider.stream),
                "session: " + session.session_id,
                "runtime: yolo="
                + self._format_bool(session.settings.yolo)
                + " plan="
                + self._format_bool(session.settings.plan_mode)
                + " compact_at="
                + str(session.settings.compact_at),
                "conversation: " + str(len(session.state.conversation)) + "/" + str(session.settings.compact_at),
                "tool_calls: turn=" + str(session.state.turn_tool_calls) + " session=" + str(session.state.session_tool_calls),
                "tokens: last=" + _format_count(session.state.last_total_tokens) + " session=" + _format_count(session.state.session_total_tokens),
                "models:",
                model_usage,
                "task: " + blackboard.task_code,
                "goal: " + (blackboard.goal or "(empty)"),
                "verification: " + verification_status,
            ]
        )

    def _compact(self, args: str) -> str:
        if args:
            return "Usage: /compact"
        return self._with_status(self._compact_history)

    def _compact_history(self) -> str:
        before = len(self.agent.session.state.conversation)
        count = self.agent.compact_history()
        if count:
            return "Compacted conversation history: " + str(count) + " item(s) -> " + str(len(self.agent.session.state.conversation)) + " item(s)"
        return (
            "Conversation history is empty"
            if before == 0
            else "Nothing to compact: " + str(before) + " item(s), keeping recent " + str(ConversationCompactor.KEEP_RECENT) + "."
        )

    def _config(self, args: str) -> str:
        if args:
            return "Usage: /config"
        session = self.agent.session
        provider_config = session.config.provider
        return "\n".join(
            [
                "config: " + ConfigFile.path(),
                "provider.active: " + session.config.active_provider,
                "provider.available: " + ", ".join(sorted(session.config.providers)),
                "provider.url: " + (provider_config.url or "(empty)"),
                "provider.key: " + ("(set)" if provider_config.key else "(empty)"),
                "provider.model: " + (provider_config.model or "(empty)"),
                "provider.available_models: " + (", ".join(provider_config.available_models) or "(empty)"),
                "provider.reasoning: " + self._format_bool(provider_config.reasoning),
                "provider.effort: " + (provider_config.reasoning_effort or "(empty)"),
                "provider.reasoning_payload: " + (provider_config.reasoning_payload or "(empty)"),
                "provider.stream: " + self._format_bool(provider_config.stream),
                "provider.temperature: " + self._format_optional(provider_config.temperature),
                "provider.timeout: " + self._format_optional(provider_config.timeout),
                "provider.first_token_timeout: " + self._format_optional(provider_config.first_token_timeout),
                "paths.data_dir: " + session.data_path(),
                "paths.project_dir: " + session.project_dir(),
                "paths.session_dir: " + session.session_dir(),
                "paths.history: " + session.history_path(),
                "runtime.compact_at: " + str(session.settings.compact_at),
                "runtime.shell_timeout: " + str(session.settings.shell_timeout),
                "runtime.max_agent_steps: " + str(session.settings.max_agent_steps),
                "runtime.plan_timeout: " + str(session.settings.plan_timeout),
                "runtime.plan_first_token_timeout: " + str(session.settings.plan_first_token_timeout),
                "runtime.auto_clean_recent: " + session.settings.auto_clean_recent,
                "runtime.yolo: " + self._format_bool(session.settings.yolo),
                "runtime.plan_mode: " + self._format_bool(session.settings.plan_mode),
            ]
        )

    def _knowledge(self, args: str) -> str:
        if args:
            return "Usage: /knowledge"
        knowledge = self.agent.blackboard.stable_knowledge
        if not any(knowledge.values()):
            return "No stable knowledge stored."
        lines = ["Stable knowledge:"]
        for category in STABLE_KNOWLEDGE_CATEGORIES:
            items = knowledge.get(category, [])
            if not items:
                continue
            lines.append(category + ":")
            lines.extend("- " + item for item in items)
        return "\n".join(lines)

    def _set(self, args: str) -> str:
        key, value = self._parse_set_args(args)
        if not key:
            return CONFIG_SET_USAGE
        if key not in CONFIG_SET_KEYS:
            return "Unknown config key: " + key
        if value is None:
            return "Current " + key + " is " + self._config_value(key)
        error = self._apply_config_value(key, value)
        if error:
            return error
        suffix = ""
        if key == "runtime.compact_at":
            compacted = self._with_status(lambda: "yes" if self.agent.compactor.maybe_compact() else "") == "yes"
            suffix = " and compacted history" if compacted else ""
        return "Set " + key + " = " + self._config_value(key) + suffix

    def _parse_set_args(self, args: str) -> tuple[str, str | None]:
        key, separator, value = args.partition(" ")
        return key.strip(), (value.strip() or None) if separator else None

    def _config_value(self, key: str) -> str:
        target, attr = self._config_target(key)
        value = getattr(target, attr)
        if key in CONFIG_BOOL_KEYS:
            return self._format_bool(value)
        if key == "provider.model":
            return value or "(empty)"
        if key == "provider.temperature":
            return self._format_optional(value)
        return str(value)

    def _apply_config_value(self, key: str, value: str) -> str:
        target, attr = self._config_target(key)
        if key in CONFIG_BOOL_KEYS:
            if value not in {"on", "off"}:
                return "Usage: /set " + key + " [on|off]"
            setattr(target, attr, value == "on")
            return ""
        if key == "provider.effort":
            if value not in CONFIG_EFFORTS:
                return "Usage: /set " + key + " [" + "|".join(CONFIG_EFFORTS) + "]"
            setattr(target, attr, value)
            return ""
        if key == "provider.temperature":
            if value == "off":
                setattr(target, attr, None)
                return ""
            try:
                parsed_float = float(value)
            except ValueError:
                return "Usage: /set " + key + " <number|off>"
            if parsed_float < 0:
                return "Usage: /set " + key + " <number|off>"
            setattr(target, attr, parsed_float)
            return ""
        if key in CONFIG_INT_KEYS:
            try:
                parsed_int = int(value)
            except ValueError:
                return "Usage: /set " + key + " <positive-number>"
            if parsed_int <= 0:
                return "Usage: /set " + key + " <positive-number>"
            setattr(target, attr, parsed_int)
            return ""
        setattr(target, attr, value)
        return ""

    def _config_target(self, key: str) -> tuple[object, str]:
        if key in CONFIG_PROVIDER_ATTRS:
            return self.agent.session.config.provider, CONFIG_PROVIDER_ATTRS[key]
        return self.agent.session.settings, CONFIG_RUNTIME_ATTRS[key]

    def _clean(self, args: str) -> str:
        if args:
            return "Usage: /clean"
        sessions_dir = self.agent.session.data_path("sessions")
        if not os.path.isdir(sessions_dir):
            return f"No session logs directory found at {sessions_dir}"
        result = SessionLogCleaner(self.agent.session).clean()
        msg = f"Cleaned {result.cleaned} log file(s) from {sessions_dir}"
        if result.skipped:
            msg += f" ({result.skipped} active session(s) skipped)"
        if result.failed:
            msg += f" ({result.failed} failed)"
        return msg

    def _format_bool(self, value: bool | None) -> str:
        return "(fallback)" if value is None else ("on" if value else "off")

    def _format_optional(self, value: object) -> str:
        return str(value) if value is not None else "(fallback)"

    def _with_status(self, action: StatusAction) -> str:
        return action() if self.run_with_status is None else self.run_with_status(action)


def _format_count(value: int) -> str:
    if value <= 0:
        return "-"
    if value >= 1_000_000:
        return str(value // 1_000_000) + "m"
    if value >= 1_000:
        return str(value // 1_000) + "k"
    return str(value)


############################
# Interactive Loop
############################


class StatusBar:
    INTERVAL: ClassVar[float] = 0.2

    def __init__(self, session: Session):
        self.session = session
        self.started_at = 0.0
        self.last_elapsed = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rendered = False
        self.output = create_output(sys.stderr)

    def __enter__(self) -> Self:
        self.started_at = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.pause()

    def reset_timer(self) -> None:
        self.started_at = time.monotonic()
        self.last_elapsed = 0.0

    def elapsed(self) -> float:
        if self.started_at <= 0:
            return 0.0
        return time.monotonic() - self.started_at

    def is_running(self) -> bool:
        return self.thread is not None

    def snapshot(self, turn_elapsed: float = 0.0) -> str:
        return "".join(text for _, text in self._fragments(turn_elapsed, now=time.monotonic(), show_sweep=False, show_elapsed=False))

    def resume(self) -> None:
        if self.thread is not None or not sys.stderr.isatty():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        if self.thread is None:
            return
        self.last_elapsed = self.elapsed()
        self.stop_event.set()
        self.thread.join()
        self.thread = None
        self._clear()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            now = time.monotonic()
            elapsed = self.elapsed()
            self.last_elapsed = elapsed
            self.output.write_raw("\r")
            self.output.erase_end_of_line()
            print_formatted_text(FormattedText(self._fragments(elapsed, now=now, show_sweep=True, show_elapsed=True)), output=self.output, end="", flush=True)
            self.rendered = True
            self.stop_event.wait(self.INTERVAL)

    def _clear(self) -> None:
        if not self.rendered:
            return
        self.output.write_raw("\r")
        self.output.erase_end_of_line()
        self.output.flush()
        self.rendered = False

    def _fragments(self, turn_elapsed: float, *, now: float, show_sweep: bool, show_elapsed: bool) -> list[tuple[str, str]]:
        text = self._format_line(turn_elapsed, now=now, show_elapsed=show_elapsed)
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(text) >= columns:
            text = text[: max(0, columns - 4)] + "..."
        return self._sweep_fragments(text, now) if show_sweep else [("ansicyan", text)]

    def _format_line(self, turn_elapsed: float, *, now: float, show_elapsed: bool) -> str:
        session = self.session
        active_model = session.state.current_model_call_label or session.config.provider.model
        model = active_model.rsplit("/", 1)[-1] or active_model or "(no model)"
        reasoning = session.state.current_model_call_reasoning_label or (
            session.config.provider.reasoning_effort if session.config.provider.reasoning else "off"
        )
        modes = "".join(" | " + label for label, enabled in (("yolo", session.settings.yolo), ("plan", session.settings.plan_mode)) if enabled)
        context = str(len(session.state.conversation)) + "/" + str(session.settings.compact_at)
        last_tokens = _format_count(session.state.last_total_tokens)
        session_tokens = _format_count(session.state.session_total_tokens)
        tokens = "last:" + last_tokens + " session:" + session_tokens
        parts = [model + " (" + reasoning + ")" + modes, "ctx:" + context, "tools:" + str(session.state.turn_tool_calls), "tok:" + tokens]
        if show_elapsed:
            parts.append(f"{turn_elapsed:.1f}s")
        if session.state.current_model_call_started_at > 0:
            activity = self._activity_label(session.state.current_model_call_activity)
            if session.state.current_model_call_has_content:
                activity += "*"
            parts.append(
                activity
                + "("
                + str(session.state.turn_model_calls)
                + "):"
                + f"{max(0.0, now - session.state.current_model_call_started_at):.1f}s"
            )
        if session.state.status_notice and session.state.status_notice_until > now:
            parts.append(session.state.status_notice)
        return " | ".join(parts)

    @staticmethod
    def _activity_label(activity: str) -> str:
        return {"compact": "compacting", "observe": "observing"}.get(activity, "working")

    def _sweep_fragments(self, text: str, now: float) -> list[tuple[str, str]]:
        if not text:
            return [("", "")]
        width = max(1, len(text) - 1)
        sweep = (now * 0.55) % 1.0
        fragments = []
        for index, char in enumerate(text):
            ratio = index / width
            red = round(75 + (180 - 75) * ratio)
            green = round(180 + (130 - 180) * ratio)
            blue = 235
            distance = abs(ratio - sweep)
            intensity = max(0.0, 1.0 - distance * 5.0) ** 2
            red = round(red + (230 - red) * intensity)
            green = round(green + (245 - green) * intensity)
            blue = round(blue + (255 - blue) * intensity)
            fragments.append((f"#{red:02x}{green:02x}{blue:02x}", char))
        return fragments


class AgentLoop:
    LIVE_PREVIEW_MAX_LINES: ClassVar[int] = 10
    LIVE_PREVIEW_MAX_CHARS: ClassVar[int] = 20_000
    LIVE_PREVIEW_REFRESH_INTERVAL: ClassVar[float] = 0.12
    LIVE_PREVIEW_INTERRUPT_HINT_AFTER: ClassVar[float] = 3.0

    def __init__(
        self,
        agent: Agent,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: MessageCallback = print,
        prompt_session=None,
    ):
        self.agent = agent
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.status_bar = StatusBar(agent.session)
        self.history_path = agent.session.history_path()
        self.prompt_session = prompt_session
        self._live_preview_active = False
        self._live_preview_resume_status = False
        self._live_preview_text = ""
        self._live_preview_rendered_lines = 0
        self._live_preview_last_render = 0.0
        self._live_preview_started_at = 0.0
        self._live_preview_hint_shown = False
        if self.prompt_session is None and input_fn is input and sys.stdin.isatty():
            self.prompt_session = self._make_prompt_session()

    def run(self) -> int:
        self._print_welcome()
        with SessionLock(self.agent.session.lock_path()), self.status_bar:
            self._auto_clean_logs()
            dispatcher = CommandDispatcher(
                self.agent,
                run_agent=self._run_agent,
                run_with_status=self._run_with_status,
                select_reasoning=self._select_reasoning,
                select_model=self._select_model,
                select_provider=self._select_provider,
            )
            while True:
                try:
                    user_input = self._read_input(self._prompt()).strip()
                except EOFError:
                    self._emit("")
                    return 0
                except KeyboardInterrupt:
                    self._emit("Cancelled")
                    continue
                if not user_input:
                    continue
                try:
                    result = dispatcher.dispatch(user_input)
                except Exception as error:
                    self._emit("Error: " + str(error))
                    continue
                if result.status == CommandStatus.EXIT:
                    return 0
                if result.status == CommandStatus.HANDLED:
                    if result.message:
                        self._emit(result.message)
                    continue
                self._run_agent(user_input)

    def _auto_clean_logs(self) -> None:
        seconds = RuntimeSettings.clean_retention_seconds(self.agent.session.settings.auto_clean_recent)
        if seconds > 0:
            SessionLogCleaner(self.agent.session).clean(older_than_seconds=seconds)

    def _prompt(self) -> str:
        labels = []
        if self.agent.session.settings.yolo:
            labels.append("yolo")
        if self.agent.session.settings.plan_mode:
            labels.append("plan")
        return "[" + ",".join(labels) + "] > " if labels else "> "

    def _read_input(self, prompt: str) -> str:
        if self.prompt_session is None:
            return self.input_fn(prompt)
        with patch_stdout():
            return self.prompt_session.prompt(
                prompt,
                multiline=False,
                enable_history_search=True,
                refresh_interval=StatusBar.INTERVAL,
                bottom_toolbar=lambda: self.status_bar._fragments(
                    0.0,
                    now=time.monotonic(),
                    show_sweep=False,
                    show_elapsed=False,
                ),
            )

    def _choice_style(self) -> Style:
        return Style.from_dict(
            {
                "selected-option": "bold #0f4c5c bg:#e6f2f3",
                "choice-hint": "#6b7280",
                "bottom-toolbar": "noreverse bg:default fg:default",
                "bottom-toolbar.text": "noreverse bg:default fg:default",
            }
        )

    def _choice_bottom_toolbar(self):
        return self.status_bar._fragments(
            0.0,
            now=time.monotonic(),
            show_sweep=False,
            show_elapsed=False,
        )

    def _visible_choices(self, choices: tuple[str, ...], labels: dict[str, str], disabled: set[str], query: str) -> tuple[str, ...]:
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

    def _choice_enabled(self, choices: tuple[str, ...], disabled: set[str]) -> tuple[str, ...]:
        return tuple(choice for choice in choices if choice not in disabled)

    def _choice_initial_index(self, enabled_choices: tuple[str, ...], current: str) -> int:
        return enabled_choices.index(current) if current in enabled_choices else 0

    def _run_choice_application(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str],
        current: str,
        disabled: set[str],
    ) -> SelectionResult:
        state: dict[str, str | int | bool] = {"query": "", "selected": 0, "searching": False}

        def enabled() -> tuple[str, ...]:
            return self._choice_enabled(self._visible_choices(choices, labels, disabled, str(state["query"])), disabled)

        def clamp_selection() -> None:
            options = enabled()
            if not options:
                state["selected"] = 0
                return
            state["selected"] = min(max(int(state["selected"]), 0), len(options) - 1)

        def choice_fragments():
            query = str(state["query"])
            visible = self._visible_choices(choices, labels, disabled, query)
            options = self._choice_enabled(visible, disabled)
            clamp_selection()
            suffix = (" /" + query) if query else ""
            if query and not state["searching"]:
                suffix += " (filtered)"
            fragments = [
                ("", title + suffix + "\n"),
                ("class:choice-hint", "  j/k move, / search, Esc back/cancel\n"),
            ]
            if query and not options:
                fragments.append(("", "  No matches\n"))
                return fragments[:-1]
            number = 1
            for choice in visible:
                label = labels.get(choice, choice)
                if choice in disabled:
                    fragments.append(("", "  " + label + "\n"))
                    continue
                selected = number - 1 == int(state["selected"])
                style = "class:selected-option" if selected else ""
                if selected:
                    fragments.append(("[SetCursorPosition]", ""))
                fragments.append((style, ("> " if selected else "  ") + f"{number:2d}. " + label + "\n"))
                number += 1
            if state["searching"]:
                fragments.append(("", "/" + query))
            return fragments[:-1] if fragments and fragments[-1][1] == "\n" else fragments

        bindings = KeyBindings()
        searching = Condition(lambda: bool(state["searching"]))

        @bindings.add("up", eager=True)
        def _up(event):
            state["selected"] = max(0, int(state["selected"]) - 1)
            event.app.invalidate()

        @bindings.add("k", filter=~searching, eager=True)
        def _k(event):
            state["selected"] = max(0, int(state["selected"]) - 1)
            event.app.invalidate()

        @bindings.add("down", eager=True)
        def _down(event):
            options = enabled()
            if options:
                state["selected"] = min(len(options) - 1, int(state["selected"]) + 1)
            event.app.invalidate()

        @bindings.add("j", filter=~searching, eager=True)
        def _j(event):
            options = enabled()
            if options:
                state["selected"] = min(len(options) - 1, int(state["selected"]) + 1)
            event.app.invalidate()

        @bindings.add("/", eager=True)
        def _search(event):
            state["query"] = ""
            state["searching"] = True
            state["selected"] = 0
            event.app.invalidate()

        @bindings.add("backspace", filter=searching, eager=True)
        @bindings.add("c-h", filter=searching, eager=True)
        def _backspace(event):
            state["query"] = str(state["query"])[:-1]
            state["selected"] = 0
            event.app.invalidate()

        @bindings.add("escape", eager=True)
        def _cancel_search(event):
            if state["searching"]:
                state["searching"] = False
                event.app.invalidate()
                return
            if state["query"]:
                state["query"] = ""
                state["selected"] = 0
                event.app.invalidate()
                return
            event.app.exit(result=SELECTION_BACK)

        @bindings.add("enter", eager=True)
        def _accept(event):
            options = enabled()
            if options:
                event.app.exit(result=options[int(state["selected"])])

        for index in range(1, 10):

            @bindings.add(str(index), eager=True)
            def _select_number(event, number: int = index):
                if state["searching"]:
                    state["query"] = str(state["query"]) + event.data
                    state["selected"] = 0
                    event.app.invalidate()
                    return
                options = enabled()
                if number <= len(options):
                    state["selected"] = number - 1
                    event.app.invalidate()

        @bindings.add("c-c", eager=True)
        @bindings.add("<sigint>", eager=True)
        def _interrupt(event):
            event.app.exit(exception=KeyboardInterrupt())

        @bindings.add(Keys.Any, filter=searching)
        def _type(event):
            if not event.data or event.data in "\r\n":
                return
            state["query"] = str(state["query"]) + event.data
            state["selected"] = 0
            event.app.invalidate()

        options = enabled()
        state["selected"] = self._choice_initial_index(options, current) if options else 0
        content = FormattedTextControl(choice_fragments, focusable=True)
        choice_window = Window(content, dont_extend_height=True)
        app = Application(
            layout=Layout(
                HSplit(
                    [
                        choice_window,
                        Window(
                            FormattedTextControl(lambda: self._choice_bottom_toolbar(), style="class:bottom-toolbar.text"),
                            style="class:bottom-toolbar",
                            dont_extend_height=True,
                            height=Dimension(min=1),
                        ),
                    ]
                ),
                focused_element=choice_window,
            ),
            style=self._choice_style(),
            full_screen=False,
            key_bindings=bindings,
            refresh_interval=StatusBar.INTERVAL,
            erase_when_done=True,
        )
        return app.run()

    def _select_choice(
        self,
        title: str,
        choices: tuple[str, ...],
        labels: dict[str, str] | None = None,
        current: str = "",
        disabled: set[str] | None = None,
    ) -> SelectionResult:
        labels = labels or {}
        disabled = disabled or set()
        query = ""
        while True:
            visible_choices = self._visible_choices(choices, labels, disabled, query)
            enabled_choices = tuple(choice for choice in visible_choices if choice not in disabled)
            if query and not enabled_choices:
                self._emit("No matches: " + query)
                query = ""
                continue
            if self.prompt_session is not None and sys.stdin.isatty():
                try:
                    selected = self._run_choice_application(title, choices, labels, current, disabled)
                except (EOFError, KeyboardInterrupt):
                    self._emit("Cancelled")
                    return None
                if not isinstance(selected, str) or selected not in disabled:
                    return selected
                continue

            lines = []
            index = 1
            for choice in visible_choices:
                if choice in disabled:
                    lines.append("  " + labels.get(choice, choice))
                    continue
                lines.append("  " + str(index) + ". " + labels.get(choice, choice))
                index += 1
            self._emit(title + ((" /" + query) if query else "") + ":\n" + "\n".join(lines))
            prompt = "Select " + title.lower() + " [1-" + str(len(enabled_choices)) + "] or /keyword "
            try:
                raw_choice = self._read_input(prompt).strip()
                choice = raw_choice.lower()
            except (EOFError, KeyboardInterrupt):
                self._emit("Cancelled")
                return None
            if not choice:
                return None
            if raw_choice.startswith("/"):
                query = raw_choice[1:].strip()
                continue
            if choice.isdigit() and 1 <= int(choice) <= len(enabled_choices):
                return enabled_choices[int(choice) - 1]
            if raw_choice in enabled_choices:
                return raw_choice
            if choice in enabled_choices:
                return choice
            self._emit("Invalid selection: " + raw_choice)

    def _select_model(self, models: tuple[str, ...], current_model: str) -> SelectionResult:
        labels = {current_model: current_model + " (current)"} if current_model in models else {}
        for label in CommandDispatcher.MODEL_LABELS:
            if label in models:
                labels[label] = label
        while True:
            selected = self._select_choice("Model", models, labels, current=current_model, disabled=set(CommandDispatcher.MODEL_LABELS))
            if not isinstance(selected, str) or selected not in CommandDispatcher.MODEL_LABELS:
                return selected

    def _select_provider(self, providers: tuple[str, ...], current_provider: str) -> SelectionResult:
        labels = {current_provider: current_provider + " (current)"}
        return self._select_choice("Provider", providers, labels, current=current_provider)

    def _select_reasoning(self) -> SelectionResult:
        provider = self.agent.session.config.provider
        current = provider.reasoning_effort if provider.reasoning else "off"
        labels = {"off": "off - disable reasoning"}
        if current == "off":
            labels["off"] = "off - disable reasoning (current)"
        elif current in CONFIG_EFFORTS:
            labels[current] = current + " (current)"
        return self._select_choice("Reasoning effort", ("off", *CONFIG_EFFORTS), labels, current=current)

    def _discard_pending_tty_input(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            import termios
        except ImportError:
            return
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (AttributeError, OSError, termios.error):
            pass

    def _make_prompt_session(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        return PromptSession(
            history=FileHistory(self.history_path),
            completer=ReferenceFileCompleter(
                self.agent.session.cwd,
                CommandCompleter(
                    lambda: self.agent.session.config.providers,
                    lambda: self.agent.session.config.provider.available_models,
                ),
            ),
            complete_while_typing=True,
            style=Style.from_dict(
                {
                    "bottom-toolbar": "noreverse bg:default fg:default",
                    "bottom-toolbar.text": "noreverse bg:default fg:default",
                }
            ),
        )

    def _run_agent(self, user_input: str) -> None:
        try:
            self.status_bar.reset_timer()
            self.status_bar.resume()
            self.agent.run(
                user_input,
                confirm=self._confirm_tool_call,
                on_auto_approve=self._show_auto_tool_call,
                **self._live_preview_callbacks(),
                on_message=self._emit,
            )
        except KeyboardInterrupt:
            self.agent.cancel_current_goal()
            self._emit("Cancelled")
        except Cancellation as error:
            self.agent.cancel_current_goal()
            self._emit("Cancelled: " + str(error))
        except Exception as error:
            self._emit("Error: " + str(error))
        finally:
            self._finish_live_tool_output()
            self.status_bar.pause()

    def _live_preview_callbacks(self) -> dict[str, ToolLiveOutputCallback | ToolLiveDoneCallback]:
        if not self._live_preview_enabled():
            return {}
        return {"on_live_output": self._show_live_tool_output, "on_live_done": self._finish_live_tool_output}

    def _live_preview_enabled(self) -> bool:
        return self.output_fn is print and sys.stderr.isatty()

    def _show_live_tool_output(self, call: ParsedToolCall, chunk: str) -> None:
        if not self._live_preview_enabled() or not chunk:
            return
        if not self._live_preview_active:
            self._start_live_tool_output()
        self._live_preview_text = (self._live_preview_text + chunk)[-self.LIVE_PREVIEW_MAX_CHARS :]
        self._render_live_tool_output(throttled=True)

    def _start_live_tool_output(self) -> None:
        self._live_preview_active = True
        self._live_preview_text = ""
        self._live_preview_rendered_lines = 0
        self._live_preview_last_render = 0.0
        self._live_preview_started_at = time.monotonic()
        self._live_preview_hint_shown = False
        self._live_preview_resume_status = self.status_bar.is_running()
        if self._live_preview_resume_status:
            self.status_bar.pause()

    def _finish_live_tool_output(self, call: ParsedToolCall | None = None) -> None:
        if not self._live_preview_active:
            return
        self._render_live_tool_output(throttled=False)
        # Keep the final live preview in terminal history instead of treating it
        # as an active redraw region.
        self._live_preview_rendered_lines = 0
        self._live_preview_active = False
        self._live_preview_text = ""
        self._live_preview_started_at = 0.0
        self._live_preview_hint_shown = False
        if self._live_preview_resume_status:
            self._live_preview_resume_status = False
            self.status_bar.resume()

    def _render_live_tool_output(self, *, throttled: bool) -> None:
        lines = self._live_preview_lines()
        if not any(line.strip() for line in lines):
            return
        now = time.monotonic()
        if throttled and now - self._live_preview_last_render < self.LIVE_PREVIEW_REFRESH_INTERVAL:
            return
        self._live_preview_last_render = now
        self._clear_live_tool_output()
        segments: list[tuple[str, str]] = []
        hint_visible = self._live_preview_interrupt_hint(now)
        if hint_visible:
            segments.append(("ansibrightblack", "  Ctrl-C interrupts current Bash; press again after it stops to cancel the session.\n"))
        for line in lines:
            segments.extend([("ansibrightblack", "  "), ("ansibrightblack", line + "\n")])
        print_formatted_text(FormattedText(segments), output=self.status_bar.output, end="", flush=True)
        self._live_preview_rendered_lines = len(lines) + (1 if hint_visible else 0)

    def _live_preview_interrupt_hint(self, now: float) -> bool:
        if self._live_preview_hint_shown:
            return True
        if self._live_preview_started_at <= 0:
            return False
        if now - self._live_preview_started_at < self.LIVE_PREVIEW_INTERRUPT_HINT_AFTER:
            return False
        self._live_preview_hint_shown = True
        return True

    def _clear_live_tool_output(self) -> None:
        if self._live_preview_rendered_lines <= 0:
            return
        self.status_bar.output.cursor_up(self._live_preview_rendered_lines)
        self.status_bar.output.erase_down()
        self.status_bar.output.flush()
        self._live_preview_rendered_lines = 0

    def _live_preview_lines(self) -> list[str]:
        text = self._live_preview_text.replace("\r", "\n")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        lines = [line for line in text.splitlines() if line.strip()][-self.LIVE_PREVIEW_MAX_LINES :]
        width = max(20, shutil.get_terminal_size((120, 20)).columns - 6)
        return [_shorten(line, width) for line in lines]

    def _run_with_status(self, action: StatusAction) -> str:
        self.status_bar.reset_timer()
        self.status_bar.resume()
        try:
            return action()
        finally:
            self.status_bar.pause()

    def _confirm_tool_call(self, call: ParsedToolCall, tool: Tool) -> ConfirmationResult:
        def action() -> ConfirmationResult:
            self._print_tool_call_display("Confirm Tool Call", "manual approval required", call, tool, title_style="bold ansiyellow")
            return self._wait_confirm("Proceed?", default=True)

        return self._with_status_paused(action)

    def _show_auto_tool_call(self, call: ParsedToolCall, tool: Tool) -> None:
        self._with_status_paused(lambda: self._print_tool_call_display("Auto Tool Call", "auto approved", call, tool, title_style="bold ansiblue"))

    def _with_status_paused(self, action: Callable[[], JsonValue]) -> JsonValue:
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.pause()
        try:
            return action()
        finally:
            if was_running:
                self.status_bar.resume()

    def _print_tool_call_display(
        self,
        title: str,
        status: str,
        call: ParsedToolCall,
        tool: Tool,
        *,
        title_style: str,
    ) -> None:
        self._emit_segments(
            [
                ("ansibrightblack", "-" * 48 + "\n"),
                (title_style, title),
                ("ansibrightblack", " | " + status + "\n"),
                ("ansibrightblack", "  Run     "),
                ("ansicyan", call.executed + "\n"),
            ],
            title + " | " + status + "\n  Run     " + call.executed,
        )
        if call.intention:
            self._emit_segments(
                [("ansibrightblack", "  Why     "), ("ansimagenta", call.intention + "\n")],
                "  Why     " + call.intention,
            )
        if tool.effect() == ToolEffect.EDIT:
            preview = tool.preview()
            if preview:
                self._emit_segments(self._preview_segments(preview), "  Preview\n" + preview)

    def _emit(self, message: str) -> None:
        self._with_status_paused(lambda: self._print_message(message))

    def _print_welcome(self) -> None:
        self._emit_segments([("bold ansicyan", "nanocode"), ("ansiwhite", " - AI coding assistant\n")], "nanocode - AI coding assistant")
        self._emit_segments(
            [("ansibrightblack", "  "), ("ansicyan", "/help [question]"), ("ansiwhite", " for help or source-aware questions\n")],
            "  /help [question] for help or source-aware questions",
        )
        self._emit_segments(
            [("ansibrightblack", "  "), ("ansicyan", "/status"), ("ansiwhite", " for current session state\n")],
            "  /status for current session state",
        )

    def _wait_confirm(self, prompt: str, *, default: bool) -> ConfirmationResult:
        self._discard_pending_tty_input()
        suffix = "[Y/n/reason]" if default else "[y/N/reason]"
        while True:
            raw_answer = self._read_input(prompt + " " + suffix + " ").strip()
            answer = raw_answer.lower()
            if not answer:
                self.output_fn("Answer: " + ("yes" if default else "no"))
                return default
            if answer in {"y", "yes"}:
                self.output_fn("Answer: yes")
                return True
            if answer in {"n", "no"}:
                self.output_fn("Answer: no")
                return False
            self.output_fn("Answer: no - " + raw_answer)
            return raw_answer

    def _print_message(self, message: str) -> None:
        if message.startswith("State Updated"):
            self._emit_segments(self._state_segments(message), message)
            return
        if message.startswith(("Plan Updated", "Known Updated", "Hypotheses Updated", "Plan + Known Updated", "Plan + Hypotheses Updated", "Hypotheses + Known Updated", "Plan + Hypotheses + Known Updated")):
            self._emit_segments(self._compact_state_segments(message), message)
            return
        if message.startswith("Tool Result Context:"):
            plain = "  ctx: " + message.removeprefix("Tool Result Context:").strip()
            self._emit_segments([("ansibrightblack", plain + "\n")], plain)
            return
        if message.startswith("Tool Calls Skipped:"):
            plain = "  skipped: " + message.removeprefix("Tool Calls Skipped:").strip()
            self._emit_segments([("ansibrightblack", plain + "\n")], plain)
            return
        if self._is_tool_report(message):
            self._emit_segments(self._indent_segments(self._tool_segments(message), "  "), self._tool_plain(message, indent="  "), end="")
            return
        if message.startswith("Retrying:"):
            self._emit_segments([("ansibrightblack", message + "\n")], message)
            return
        if message.startswith("Error:"):
            self._emit_segments([("bold ansired", message + "\n")], message)
            return
        if message.startswith("Cancelled"):
            tip = "Context is kept; send a follow-up to continue."
            self._emit_segments(
                [("ansiyellow", message + "\n"), ("ansibrightblack", "  " + tip + "\n")],
                message + "\n  " + tip,
            )
            return
        self._emit_segments([("ansicyan", message + "\n")], message)

    def _tool_plain(self, message: str, *, indent: str) -> str:
        return "\n".join(indent + line.replace("[success] ", "").replace("[failure] ", "") for line in message.splitlines())

    def _is_tool_report(self, message: str) -> bool:
        lines = message.splitlines()
        if not lines:
            return False
        first = lines[0]
        return first.startswith("  ...") or self._is_tool_call_line(first)

    def _is_tool_call_line(self, line: str) -> bool:
        return line.startswith("[success] ") or line.startswith("[failure] ")

    def _emit_segments(self, segments: list[tuple[str, str]], plain: str, *, end: str = "\n") -> None:
        if self.output_fn is print:
            print_formatted_text(FormattedText(segments), end=end, flush=True)
        else:
            self.output_fn(plain)

    def _preview_segments(self, preview: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = [("ansibrightblack", "  Preview\n")]
        content_indent = "  "
        diff_start = self._unified_diff_start(preview)
        if diff_start >= 0:
            prefix = "\n".join(preview.splitlines()[:diff_start])
            diff = "\n".join(preview.splitlines()[diff_start:])
            if prefix:
                segments += self._indented_text_segments(prefix, indent=content_indent, style="ansiyellow")
            return segments + self._indent_segments(self._diff_segments(diff), content_indent)
        return segments + self._indented_text_segments(preview, indent=content_indent, style="ansicyan")

    def _unified_diff_start(self, text: str) -> int:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            body = "\n".join(lines[index:])
            if line.startswith("--- ") and "\n+++ " in body and "\n@@ " in body:
                return index
        return -1

    def _diff_segments(self, text: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        lines = text.splitlines()
        old_line: int | None = None
        new_line: int | None = None

        def parse_hunk_start(part: str, prefix: str) -> int | None:
            if not part.startswith(prefix):
                return None
            value = part[1:].split(",", 1)[0]
            try:
                return int(value)
            except ValueError:
                return None

        def add_line_number(old_number: int | None, new_number: int | None) -> None:
            old_text = "" if old_number is None else str(old_number)
            new_text = "" if new_number is None else str(new_number)
            segments.append(("ansibrightblack", f"{old_text:>4} {new_text:>4} │ "))

        for index, line in enumerate(lines):
            suffix = "\n" if index < len(lines) - 1 else ""
            if line.startswith("@@"):
                parts = line.split()
                if len(parts) >= 3:
                    old_line = parse_hunk_start(parts[1], "-")
                    new_line = parse_hunk_start(parts[2], "+")
                add_line_number(None, None)
                segments.append(("ansicyan", line + suffix))
            elif line.startswith(("---", "+++")):
                add_line_number(None, None)
                segments.append(("ansibrightblack", line + suffix))
            elif line.startswith("+"):
                add_line_number(None, new_line)
                segments.append(("ansigreen", line + suffix))
                if new_line is not None:
                    new_line += 1
            elif line.startswith("-"):
                add_line_number(old_line, None)
                segments.append(("ansired", line + suffix))
                if old_line is not None:
                    old_line += 1
            elif line.startswith(" "):
                add_line_number(old_line, new_line)
                segments.append(("ansiwhite", line + suffix))
                if old_line is not None:
                    old_line += 1
                if new_line is not None:
                    new_line += 1
            else:
                add_line_number(None, None)
                segments.append(("ansiwhite", line + suffix))
        return segments

    def _indented_text_segments(self, text: str, *, indent: str, style: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        for line in text.splitlines() or [""]:
            segments.extend([("ansibrightblack", indent), (style, line + "\n")])
        return segments

    def _indent_segments(self, segments: list[tuple[str, str]], indent: str) -> list[tuple[str, str]]:
        indented: list[tuple[str, str]] = []
        at_line_start = True
        for style, text in segments:
            for part in text.splitlines(keepends=True):
                if at_line_start:
                    indented.append(("ansibrightblack", indent))
                indented.append((style, part))
                at_line_start = part.endswith("\n")
        return indented

    def _state_segments(self, message: str) -> list[tuple[str, str]]:
        lines = message.splitlines()
        segments: list[tuple[str, str]] = [("ansibrightblack", "-" * 48 + "\n")]
        for index, line in enumerate(lines):
            if index == 0:
                title, _, badge = line.partition("|")
                badge = badge.strip()
                segments.extend([("bold ansicyan", title.strip()), ("ansibrightblack", " | "), (self._verify_style(badge), badge), ("", "\n")])
            elif line.startswith("  Goal"):
                segments.extend([("ansibrightblack", line[:10]), ("bold ansigreen", line[10:] + "\n")])
            elif line.startswith("  Plan"):
                segments.extend([("ansibrightblack", "  "), ("bold ansicyan", line.strip()), ("", "\n")])
            elif line.startswith("  Hypotheses"):
                segments.extend([("ansibrightblack", "  "), ("bold ansimagenta", line.strip()), ("", "\n")])
            elif line.startswith("  Known"):
                segments.extend([("ansibrightblack", "  "), ("bold ansiyellow", line.strip()), ("", "\n")])
            elif line.startswith("  Verify"):
                status = line[10:].strip().split(" ", 1)[0]
                segments.extend([("ansibrightblack", line[:10]), (self._verify_style("VERIFY:" + status), line[10:] + "\n")])
            elif line.startswith("    ..."):
                segments.extend([("ansibrightblack", line + "\n")])
            elif line.startswith("    "):
                segments.extend([("ansibrightblack", "    "), ("ansiwhite", line[4:] + "\n")])
            else:
                segments.extend([("ansiwhite", line + "\n")])
        return segments

    def _compact_state_segments(self, message: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        for line in message.splitlines():
            if line.endswith("Updated"):
                segments.append(("bold ansicyan", line + "\n"))
            elif line in {"Plan", "Hypotheses", "Known"}:
                segments.append(("ansicyan", line + "\n"))
            elif line.startswith("  ..."):
                segments.append(("ansibrightblack", line + "\n"))
            else:
                segments.append(("ansiwhite", line + "\n"))
        return segments

    def _tool_segments(self, message: str) -> list[tuple[str, str]]:
        lines = message.splitlines()
        segments: list[tuple[str, str]] = []
        for line in lines:
            if self._is_tool_call_line(line):
                marker, _, tail = line.partition(" ")
                status_style = "ansigreen" if marker == "[success]" else "ansired"
                segments.extend(self._tool_call_segments(tail, status_style))
            else:
                segments.extend([("ansibrightblack", line + "\n")])
        return segments

    def _tool_call_segments(self, tail: str, status_style: str) -> list[tuple[str, str]]:
        head, sep, rest = tail.partition(" -> ")
        if not sep:
            return [(status_style, tail + "\n")]
        key, detail_sep, detail = rest.partition(" | ")
        detail_text = detail_sep + detail if detail_sep else ""
        detail_style = "ansibrightblack" if detail == "excerpt" else status_style
        segments = [(status_style, head), ("ansibrightblack", sep + key)]
        if detail_text:
            segments.append((detail_style, detail_text))
        segments.append(("", "\n"))
        return segments

    def _verify_style(self, badge: str) -> str:
        if "required" in badge:
            return "bold ansimagenta"
        if "done" in badge:
            return "bold ansigreen"
        if "failed" in badge or "blocked" in badge:
            return "bold ansired"
        return "ansibrightblack"


############################
# Helpers
############################


def _format_lines(lines: list[str], indent: str) -> str:
    return "\n".join(indent + line for line in lines)


def _make_unified_diff(old_content: str, new_content: str, filepath: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=filepath,
            tofile=filepath,
        )
    )


def _format_process_result(tag: str, exit_code: int, stdout: str, stderr: str) -> str:
    lines = [f"<{tag}>", f"* exit_code: {exit_code}"]
    if stdout:
        lines.extend(["<stdout>", stdout.rstrip("\n"), "</stdout>"])
    if stderr:
        lines.extend(["<stderr>", stderr.rstrip("\n"), "</stderr>"])
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def _json_dict(value: JsonValue) -> Json:
    return value if isinstance(value, dict) else {}


def _json_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _json_str(value: JsonValue) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def _memory_fact_from_json(value: JsonValue) -> str | None:
    item = _json_dict(value)
    if item:
        fact = (_json_str(item.get("text")) or _json_str(item.get("fact")) or "").strip()
    else:
        fact = (_json_str(value) or "").strip()
    if not fact:
        return None
    if fact.startswith("<") and fact.endswith(">"):
        inner = fact[1:-1].strip().lower()
        if inner and any(word in inner for word in ("fact", "target", "arg", "path", "criterion", "result", "context", "message", "goal")):
            return None
    return fact


def _source_from_json(item: Json) -> tuple[str, ...]:
    source_values = _json_list(item.get("source")) or _json_list(item.get("sources"))
    source = [(_json_str(raw) or "").strip() for raw in source_values]
    for key in ("result_key", "key"):
        value = (_json_str(item.get(key)) or "").strip()
        if value:
            source.append(value)
    return tuple(dict.fromkeys(item for item in source if item))


def _shorten(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class CommandCompleter(Completer):
    def __init__(self, providers: Iterable[str] | Callable[[], Iterable[str]] = (), models: Iterable[str] | Callable[[], Iterable[str]] = ()):
        self.providers = providers
        self.models = models

    def _values(self, values: Iterable[str] | Callable[[], Iterable[str]]) -> Iterable[str]:
        return values() if callable(values) else values

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/set "):
            text = text[len("/set ") :]
            if " " not in text:
                for key in CONFIG_SET_KEYS:
                    if key.startswith(text):
                        yield Completion(key, start_position=-len(text))
                return
            key, _, value_prefix = text.partition(" ")
            for value in CONFIG_VALUE_COMPLETIONS.get(key, ()):
                if value.startswith(value_prefix):
                    yield Completion(value, start_position=-len(value_prefix))
            return
        if text.startswith("/provider "):
            text = text[len("/provider ") :]
            for provider in self._values(self.providers):
                if provider.startswith(text):
                    yield Completion(provider, start_position=-len(text))
            return
        if text.startswith("/model "):
            text = text[len("/model ") :]
            for model in self._values(self.models):
                if model.startswith(text):
                    yield Completion(model, start_position=-len(text))
            return
        if text.startswith("/plan "):
            text = text[len("/plan ") :]
            for value in ("on", "off"):
                if value.startswith(text):
                    yield Completion(value, start_position=-len(text))
            return
        if text.startswith("/") and " " not in text:
            for spec in COMMANDS:
                if spec.name.startswith(text):
                    yield Completion(spec.name, start_position=-len(text))


class ReferenceFileCompleter(Completer):
    def __init__(self, cwd: str, command_completer: Completer):
        self.cwd = cwd
        self.command_completer = command_completer

    def get_completions(self, document, complete_event):
        match = re.search(r"(?:^|\s)@([^\s]*)$", document.text_before_cursor)
        if match is None:
            yield from self.command_completer.get_completions(document, complete_event)
            return

        partial = match.group(1)
        dirname, prefix = os.path.split(partial)
        base_dir = os.path.abspath(os.path.join(self.cwd, dirname))
        try:
            names = sorted(os.listdir(base_dir))
        except OSError:
            return

        for name in names:
            if not name.startswith(prefix):
                continue
            full_path = os.path.join(base_dir, name)
            suffix = "/" if os.path.isdir(full_path) else ""
            candidate = os.path.join(dirname, name) + suffix if dirname else name + suffix
            yield Completion(candidate, start_position=-len(partial), display="@" + candidate)


############################
# Entrypoint
############################


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="nanocode: AI coding assistant")
        parser.add_argument("-v", "--version", action="version", version=__version__)
        parser.add_argument("--yolo", action="store_true", help="Skip tool execution confirmations")
        parser.add_argument("--plan", action="store_true", help="Plan changes without editing or running commands")
        parser.add_argument("--debug", action="store_true", help="Write request prompts to the current session debug directory")
        parser.add_argument("--config", default=None, help="Path to config file (default: ~/.nanocode/config.toml)")
        parser.add_argument("--init-config", action="store_true", help="Create a default config file at --config or ~/.nanocode/config.toml")
        args = parser.parse_args(argv)
        if args.init_config:
            config_path, created = ConfigFile.init(args.config)
            print(("Created config: " if created else "Config already exists: ") + config_path)
            return 0
        session = Session.from_config_file(path=args.config, yolo=args.yolo, plan_mode=args.plan, debug=args.debug)
        missing = session.missing_required_config()
        if missing:
            print("Missing config: " + ", ".join(missing), file=sys.stderr)
            print("Edit " + (os.path.expanduser(args.config) if args.config else ConfigFile.path()) + " or run `nanocode --init-config`.", file=sys.stderr)
            return 2
        return AgentLoop(Agent(session)).run()
    except ConfigError as error:
        print("Error: " + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
