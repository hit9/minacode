"""
nanocode
~~~~~~~~
A lightweight terminal-based AI coding assistant
https://github.com/hit9/nanocode
Install: uv tool install nanocode-cli
"""

import argparse
import fnmatch
import hashlib
import itertools
import json
import json_repair
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import difflib
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from abc import abstractmethod
from enum import StrEnum
from typing import Any, Callable, ClassVar, final, Iterator, Protocol, Self, Type, TypeAlias
from typing_extensions import override
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.patch_stdout import patch_stdout


JsonValue: TypeAlias = Any
Json: TypeAlias = dict[str, JsonValue]
MAX_TOOL_OUTPUT_CHARS = 12_000
EXPLORE_MESSAGE_PREFIX = "[explore] "
__version__ = "0.3.1"


class Error(Exception): ...


class ToolCallError(Exception): ...


class ToolCallArgError(ToolCallError): ...


class LLMError(Exception): ...


class ConfigError(Exception): ...


class ModelRequestTimeout(Exception): ...


class Cancellation(Exception): ...


class PromptItem:
    @abstractmethod
    def format(self, indent: str = "") -> str:
        raise NotImplementedError


############################
# Conversation (dataclasses)
############################


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationItem(PromptItem):
    role: Role
    time: datetime = field(default_factory=datetime.now)

    def format_ts(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M:%S")


@final
@dataclass
class UserMessage(ConversationItem):
    role: Role = Role.USER
    content: str = ""

    @override
    def format(self, indent: str = "") -> str:
        lines = [f'<UserMessage at="{self.format_ts()}">', f"{self.content}", "</UserMessage>"]
        return _format_lines(lines, indent)


@final
@dataclass
class AssistantMessage(ConversationItem):
    role: Role = Role.ASSISTANT
    content: str = ""

    @override
    def format(self, indent: str = "") -> str:
        lines = [f'<AssistantMessage at="{self.format_ts()}">', self.content, "</AssistantMessage>"]
        return _format_lines(lines, indent)


############################
# Blackboard (dataclasses)
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


@final
@dataclass
class PlanItem(PromptItem):
    text: str
    status: PlanStatus = PlanStatus.TODO
    id: str = ""
    context: str = ""

    @override
    def format(self, indent: str = "") -> str:
        parts = [f"({self.status})"]
        if self.id:
            parts.append("id=" + self.id)
        parts.append(self.text)
        if self.context:
            parts.append("context=" + self.context)
        return indent + "<PlanItem>" + " ".join(parts) + "</PlanItem>"


class VerificationStatus(StrEnum):
    IDLE = "idle"
    PLANNED = "planned"
    REQUIRED = "required"
    DONE = "done"
    BLOCKED = "blocked"


@final
@dataclass
class Verification(PromptItem):
    goal: str = ""
    status: VerificationStatus = VerificationStatus.IDLE
    method: str = ""
    context: str = ""

    @override
    def format(self, indent: str = "") -> str:
        lines = [
            "<Verification>",
            "  <goal>" + self.goal + "</goal>",
            "  <status>" + self.status + "</status>",
            "  <method>" + self.method + "</method>",
            "  <context>" + self.context + "</context>",
            "</Verification>",
        ]
        return _format_lines(lines, indent)

    def reset(self) -> None:
        self.goal = ""
        self.status = VerificationStatus.IDLE
        self.method = ""
        self.context = ""

    def has_context(self) -> bool:
        return bool(self.goal or self.method or self.context or self.status != VerificationStatus.IDLE)


@final
@dataclass
class ToolResultItem(PromptItem):
    description: str
    value: str
    log_path: str = ""
    original_lines: int = 0
    original_chars: int = 0
    excerpted: bool = False

    @override
    def format(self, indent: str = "") -> str:
        lines = ["<ToolResultItem>", "  <description>" + self.description + "</description>"]
        if self.log_path:
            lines.append("  <log_path>" + self.log_path + "</log_path>")
        if self.original_lines or self.original_chars:
            lines.append("  <original_lines>" + str(self.original_lines) + "</original_lines>")
            lines.append("  <original_chars>" + str(self.original_chars) + "</original_chars>")
        lines.append("  <excerpted>" + str(self.excerpted).lower() + "</excerpted>")
        lines.append("</ToolResultItem>")
        return _format_lines(lines, indent)


@dataclass
class Blackboard:
    user_input: str = ""
    goal: str = ""
    goal_reached: bool = False
    plan: list[PlanItem] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    verification: Verification = field(default_factory=Verification)


@dataclass
class ModelConfig:
    model: str = ""
    temperature: float | None = None
    reasoning: bool | None = None
    reasoning_effort: str = ""
    stream: bool | None = None
    timeout: int | None = None
    prompt_price_per_1m_tokens: float | None = None
    completion_price_per_1m_tokens: float | None = None

    def resolved(self, fallback: "ModelConfig") -> "ModelConfig":
        return ModelConfig(
            model=self.model or fallback.model,
            temperature=self.temperature if self.temperature is not None else fallback.temperature,
            reasoning=self.reasoning if self.reasoning is not None else fallback.reasoning,
            reasoning_effort=self.reasoning_effort or fallback.reasoning_effort,
            stream=self.stream if self.stream is not None else fallback.stream,
            timeout=self.timeout if self.timeout is not None else fallback.timeout,
            prompt_price_per_1m_tokens=(
                self.prompt_price_per_1m_tokens if self.prompt_price_per_1m_tokens is not None else fallback.prompt_price_per_1m_tokens
            ),
            completion_price_per_1m_tokens=(
                self.completion_price_per_1m_tokens if self.completion_price_per_1m_tokens is not None else fallback.completion_price_per_1m_tokens
            ),
        )


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, *, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.cost += cost


############################
# Config
############################


DEFAULT_MODEL_CONFIG = ModelConfig(
    temperature=0.7,
    reasoning=True,
    reasoning_effort="medium",
    stream=True,
    timeout=60,
    prompt_price_per_1m_tokens=0.0,
    completion_price_per_1m_tokens=0.0,
)


@final
class ConfigFile:
    DEFAULT_TEXT: ClassVar[str] = """# nanocode configuration
# Location: ~/.nanocode/config.toml

[api]
# OpenAI-compatible chat completions base URL, for example "https://api.openai.com/v1".
url = ""
# API key for the configured provider.
key = ""

[main_model]
# Default model used by the main interactive agent.
model = ""
temperature = 0.7
reasoning = true
reasoning_effort = "medium"
stream = true
timeout = 60
# Optional usage pricing per 1M tokens. Leave 0.0 if unknown.
prompt_price_per_1m_tokens = 0.0
completion_price_per_1m_tokens = 0.0

[worker_model]
# Default model config for worker agents. Empty model falls back to main_model.model.
model = ""
temperature = 0.7
reasoning = true
reasoning_effort = "medium"
stream = true
timeout = 60
prompt_price_per_1m_tokens = 0.0
completion_price_per_1m_tokens = 0.0

[explore_agent]
# ExploreAgent removes uncertainty about unknown file/code targets before editing.
max_turns = 50

[paths]
# Relative paths are resolved from the current project directory.
nanocode_dir = ".nanocode"

[runtime]
shell_timeout = 60
compact_at = 50
max_agent_steps = 50
"""

    @classmethod
    def path(cls) -> str:
        return os.path.join(os.path.expanduser("~"), ".nanocode", "config.toml")

    @classmethod
    def init(cls, path: str | None = None) -> tuple[str, bool]:
        config_path = path or cls.path()
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
        config_path = path or cls.path()
        try:
            with open(config_path, "rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError as error:
            raise ConfigError(f"Config file not found: {config_path}. Run `nanocode --init-config` to create one.") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"Invalid config file {config_path}: {error}") from error
        return data if isinstance(data, dict) else {}

    @classmethod
    def table(cls, config: Json, name: str) -> Json:
        value = config.get(name)
        return value if isinstance(value, dict) else {}

    @classmethod
    def str(cls, config: Json, key: str, default: str = "") -> str:
        value = config.get(key)
        if value is None:
            return default
        return str(value)

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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config value `{key}` must be a number")
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
    def model_config(cls, config: Json, defaults: ModelConfig) -> ModelConfig:
        return ModelConfig(
            model=cls.str(config, "model", defaults.model),
            temperature=cls.float(config, "temperature", defaults.temperature),
            reasoning=cls.bool(config, "reasoning", defaults.reasoning),
            reasoning_effort=cls.str(config, "reasoning_effort", defaults.reasoning_effort),
            stream=cls.bool(config, "stream", defaults.stream),
            timeout=cls.int(config, "timeout", defaults.timeout),
            prompt_price_per_1m_tokens=cls.float(config, "prompt_price_per_1m_tokens", defaults.prompt_price_per_1m_tokens),
            completion_price_per_1m_tokens=cls.float(
                config,
                "completion_price_per_1m_tokens",
                defaults.completion_price_per_1m_tokens,
            ),
        )


@dataclass
class AgentRuntime:
    tool_result_store: dict[str, ToolResultItem] = field(default_factory=dict)
    tool_result_counter: int = 0


@dataclass
class PromptContext:
    blackboard: Blackboard
    runtime: AgentRuntime
    parent_known: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    done: bool = False
    value: JsonValue = None


@final
@dataclass(frozen=True)
class ExploreReport(PromptItem):
    targets: list[Json]
    known: list[str]
    verification: Verification

    @override
    def format(self, indent: str = "") -> str:
        lines = ["<ExploreReport>"]
        lines.append("  <targets>")
        if self.targets:
            for item in self.targets:
                lines.append("    " + json.dumps(item, ensure_ascii=False))
        else:
            lines.append("    (empty)")
        lines.append("  </targets>")
        lines.append("  <known>")
        if self.known:
            for item in self.known:
                lines.append("    " + item)
        else:
            lines.append("    (empty)")
        lines.append("  </known>")
        lines.append("  " + self.verification.format().replace("\n", "\n  "))
        lines.append("</ExploreReport>")
        return _format_lines(lines, indent)


@final
class RangeFingerprintStore:
    MAX_ENTRIES: ClassVar[int] = 200

    @final
    @dataclass
    class Entry:
        fingerprint: str
        filepath: str
        start: int
        end: int
        content: str

    @final
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
        if not any(
            existing.fingerprint == entry.fingerprint
            and existing.filepath == entry.filepath
            and existing.start == entry.start
            and existing.end == entry.end
            and existing.content == entry.content
            for existing in self._entries
        ):
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


@final
@dataclass
class Session:
    # ---- system ----
    system: str = field(default_factory=platform.system)
    arch: str = field(default_factory=platform.machine)
    cwd: str = field(default_factory=os.getcwd)
    bash: str = field(default_factory=lambda: shutil.which("bash") or "")

    # ---- configs ----
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    nanocode_dir: str = ".nanocode"
    temperature: float = 0.7
    reasoning: bool = True
    reasoning_effort: str = "medium"
    stream: bool = True
    model_timeout: int = 60
    shell_timeout: int = 60
    compact_at: int = 50
    max_agent_steps: int = 50
    prompt_price_per_1m_tokens: float = 0.0
    completion_price_per_1m_tokens: float = 0.0
    worker_model_config: ModelConfig = field(default_factory=ModelConfig)
    explore_agent_max_turns: int = 50

    # ---- runtime variables ----
    yolo: bool = False
    debug: bool = False
    debug_prompt_count: int = 0

    # ---- stats ---
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    last_cost: float = 0.0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    session_cost: float = 0.0
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    current_model_call_started_at: float = 0.0
    current_model_call_label: str = ""
    current_model_call_reasoning_label: str = ""

    # ---- conversation ---
    conversation: list[ConversationItem] = field(default_factory=list)
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)
    tool_result_store: dict[str, ToolResultItem] = field(default_factory=dict)
    tool_result_counter: int = 0
    turn_tool_calls: int = 0
    turn_model_calls: int = 0

    @classmethod
    def from_config_file(cls, *, path: str | None = None, yolo: bool = False, debug: bool = False) -> "Session":
        return cls.from_config_data(ConfigFile.load(path), yolo=yolo, debug=debug)

    @classmethod
    def from_config_data(cls, config: Json, *, yolo: bool = False, debug: bool = False) -> "Session":
        api = ConfigFile.table(config, "api")
        paths = ConfigFile.table(config, "paths")
        runtime = ConfigFile.table(config, "runtime")
        main_model = ConfigFile.model_config(ConfigFile.table(config, "main_model"), DEFAULT_MODEL_CONFIG)
        worker_model = ConfigFile.model_config(ConfigFile.table(config, "worker_model"), ModelConfig())
        explore_agent = ConfigFile.table(config, "explore_agent")
        shell_timeout = ConfigFile.int(runtime, "shell_timeout", 60)
        compact_at = ConfigFile.int(runtime, "compact_at", 50)
        max_agent_steps = ConfigFile.int(runtime, "max_agent_steps", 50)
        explore_agent_max_turns = ConfigFile.int(explore_agent, "max_turns", 50)
        return cls(
            api_url=ConfigFile.str(api, "url"),
            api_key=ConfigFile.str(api, "key"),
            model=main_model.model,
            nanocode_dir=ConfigFile.str(paths, "nanocode_dir", ".nanocode"),
            temperature=main_model.temperature if main_model.temperature is not None else 0.7,
            reasoning=main_model.reasoning if main_model.reasoning is not None else True,
            reasoning_effort=main_model.reasoning_effort or "medium",
            stream=main_model.stream if main_model.stream is not None else True,
            model_timeout=main_model.timeout if main_model.timeout is not None else 60,
            shell_timeout=shell_timeout if shell_timeout is not None else 60,
            compact_at=compact_at if compact_at is not None else 50,
            max_agent_steps=max_agent_steps if max_agent_steps is not None else 50,
            prompt_price_per_1m_tokens=(
                main_model.prompt_price_per_1m_tokens if main_model.prompt_price_per_1m_tokens is not None else 0.0
            ),
            completion_price_per_1m_tokens=(
                main_model.completion_price_per_1m_tokens if main_model.completion_price_per_1m_tokens is not None else 0.0
            ),
            worker_model_config=worker_model,
            explore_agent_max_turns=max(1, explore_agent_max_turns if explore_agent_max_turns is not None else 50),
            yolo=yolo,
            debug=debug,
        )

    def resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return os.path.abspath(path)

    def is_path_in_cwd(self, path: str) -> bool:
        cwd = os.path.realpath(self.cwd)
        path = os.path.realpath(path)
        try:
            return os.path.commonpath([cwd, path]) == cwd
        except ValueError:
            return False

    def append_conversation(self, item: ConversationItem) -> None:
        self.conversation.append(item)

    def debug_dir(self) -> str:
        return self.resolve_path(os.path.join(self.nanocode_dir, "debug"))

    def tool_results_dir(self) -> str:
        return self.resolve_path(os.path.join(self.nanocode_dir, "tool_results"))

    def missing_required_config(self) -> list[str]:
        missing = []
        if not self.api_url:
            missing.append("api.url")
        if not self.api_key:
            missing.append("api.key")
        if not self.model:
            missing.append("main_model.model")
        return missing

    @property
    def main_model_config(self) -> ModelConfig:
        return ModelConfig(
            model=self.model,
            temperature=self.temperature,
            reasoning=self.reasoning,
            reasoning_effort=self.reasoning_effort,
            stream=self.stream,
            timeout=self.model_timeout,
            prompt_price_per_1m_tokens=self.prompt_price_per_1m_tokens,
            completion_price_per_1m_tokens=self.completion_price_per_1m_tokens,
        )

    def model_config_for(self, activity: str, override: ModelConfig | None = None) -> ModelConfig:
        config = self.main_model_config
        if activity in {"worker", "explore"}:
            config = self.worker_model_config.resolved(config)
        if override is not None:
            config = override.resolved(config)
        return config


###########
# Tools
###########


class ToolEffect(StrEnum):
    READONLY = "readonly"
    EDIT = "edit"
    OTHER = "other"


class Tool(Protocol):
    @classmethod
    def name(cls) -> str: ...
    @classmethod
    def description(cls) -> list[str]: ...
    @classmethod
    def signature(cls) -> str: ...
    @classmethod
    def example(cls) -> list[str]: ...
    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.OTHER

    @classmethod
    def is_readonly(cls) -> bool:
        return cls.effect() == ToolEffect.READONLY

    @classmethod
    def is_editing(cls) -> bool:
        return cls.effect() == ToolEffect.EDIT

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self: ...
    def requires_confirmation(self, session: Session) -> bool: ...
    def display(self) -> str: ...
    def call(self) -> str: ...


ToolClass: TypeAlias = Type[Tool]


@final
@dataclass
class ParsedToolCall:
    name: str
    intention: str
    args: list[str]

    @property
    def executed(self) -> str:
        return self.name + "(" + ", ".join(json.dumps(arg, ensure_ascii=False) for arg in self.args) + ")"


@final
@dataclass
class ToolCallExecution:
    call: ParsedToolCall
    outcome: str
    output: str
    error_type: Type[Exception] | None = None
    result_key: str = ""


@final
@dataclass
class PreparedToolCall:
    call: ParsedToolCall
    tool: Tool


@final
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
        "original_lines: " + str(original_lines) + "\noriginal_chars: " + str(original_chars) + ("\nfull_log: " + log_path if log_path else "") + "\n"
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


def _format_recent_tool_calls(executions: list[ToolCallExecution]) -> str:
    blocks = _format_recent_tool_call_blocks(executions)
    return _join_tool_call_blocks(blocks) or "(empty)"


def _format_recent_tool_call_blocks(executions: list[ToolCallExecution]) -> list[str]:
    return [_format_recent_tool_call(execution) for execution in executions]


def _join_tool_call_blocks(blocks: list[str]) -> str:
    return "\n\n".join(blocks)


def _format_recent_tool_call(execution: ToolCallExecution) -> str:
    status = "ok" if execution.outcome == "success" else "fail"
    key = (' key="' + execution.result_key + '"') if execution.result_key else ""
    lines = [
        "<ToolCall " + status + key + ">",
        "call: " + execution.call.executed,
    ]
    if execution.call.intention:
        lines.append("why: " + execution.call.intention)
    lines.extend(["result:", execution.output, "</ToolCall>"])
    return "\n".join(lines)


ConfirmationResult: TypeAlias = bool | str
ConfirmCallback: TypeAlias = Callable[[ParsedToolCall, Tool], ConfirmationResult]
ToolDisplayCallback: TypeAlias = Callable[[ParsedToolCall, Tool], None]
MessageCallback: TypeAlias = Callable[[str], None]
ActionCallback: TypeAlias = Callable[[Json], None]
StatusAction: TypeAlias = Callable[[], str]
StatusRunner: TypeAlias = Callable[[StatusAction], str]


####################
# Tools Helpers
####################


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


def _parse_line_range_token(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[-:,]\s*(\d+)\s*", value)
    if match is None:
        raise ToolCallArgError("invalid range: use a comma token like 0,120")
    return _parse_line_range(match.group(1), match.group(2))


def _range_fingerprint(content: str) -> str:
    return hashlib.blake2s(content.encode("utf-8"), digest_size=3).hexdigest()


####################
# Tools Impl
####################


@final
@dataclass
class ReadTool(Tool):
    MAX_LINES: ClassVar[int] = 600

    filepath: str = ""
    start: int = 0
    end: int = 0
    ranges: list[tuple[int, int]] = field(default_factory=list)
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def name(cls) -> str:
        return "Read"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.READONLY

    @classmethod
    def description(cls) -> list[str]:
        return [
            "Read UTF-8 file lines and cache fingerprints for range edits.",
            "Range tokens are 0-based start,end; end=0 means EOF; for batches pass each range as its own token.",
            "Use LineCount/Search before broad reads; each range returns at most 600 lines.",
            "Before ReplaceRange, Read the exact target range and reuse that fingerprint.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "Read(filepath[, range_token...]) -> ReadToolResult<fingerprint, content>"

    @classmethod
    def example(cls) -> list[str]:
        return [
            'Example args: ["code.py", "0,120"]',
            'Example args: ["code.py", "0,40", "200,260"]',
            'Example args: ["code.py"]',
        ]

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
            ranges = [_parse_line_range_token(arg) for arg in args[1:]]
        elif len(args) == 3:
            ranges = [_parse_line_range(args[1], args[2])]
        elif len(args) == 2:
            raise ToolCallArgError('Read args error: invalid range token; expected ["filepath", "start,end"]. Example: Read("nanocode.py", "2065,2095").')
        else:
            raise ToolCallArgError('Read args error: for multiple ranges use comma tokens. Example: Read("nanocode.py", "0,40", "200,260").')
        start, end = ranges[0]
        return cls(filepath=filepath, start=start, end=end, ranges=ranges, cwd=session.cwd, range_fingerprints=session.range_fingerprints)

    def requires_confirmation(self, session: Session) -> bool:
        return not session.is_path_in_cwd(self.filepath)

    def display(self) -> str:
        if len(self.ranges) > 1:
            ranges = ", ".join(str(start) + ":" + str(end) for start, end in self.ranges)
            return f"Read({self.filepath}, {ranges})"
        return f"Read({self.filepath}, {self.start}, {self.end})"

    def call(self) -> str:
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

    def _read_range(self, start: int, end: int) -> tuple[str, int, int, str, bool, int]:
        total_lines = 0
        selected_lines = []
        truncated = False
        bounded_read_lines = end - start if end else 0
        if end and bounded_read_lines <= self.MAX_LINES:
            with open(self.filepath, "r", encoding="utf-8") as f:
                selected_lines = list(itertools.islice(f, start, end))
        else:
            with open(self.filepath, "r", encoding="utf-8") as f:
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
            filepath=self.filepath,
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
            lines.extend(
                [
                    indent + "<truncated>true</truncated>",
                    indent + "<total_lines>" + str(total_lines) + "</total_lines>",
                    indent
                    + "<note>Read returned "
                    + str(returned_end - start)
                    + " lines from "
                    + str(start)
                    + ":"
                    + str(returned_end)
                    + " of "
                    + str(total_lines)
                    + " total lines. Use Search to locate relevant text or Read smaller ranges in batches.</note>",
                ]
            )
        lines.extend(
            [
                indent + "<content no-indention>",
                content,
                indent + "</content>",
            ]
        )
        return lines


@final
@dataclass
class LineCountTool(Tool):
    filepath: str = ""

    @classmethod
    def name(cls) -> str:
        return "LineCount"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.READONLY

    @classmethod
    def description(cls) -> list[str]:
        return ["Count lines in one file before choosing Read range tokens."]

    @classmethod
    def signature(cls) -> str:
        return "LineCount(filepath) -> LineCountToolResult<line_count>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 1:
            raise ToolCallArgError("requires exactly one arg: filepath")
        return cls(filepath=session.resolve_path(args[0]))

    def requires_confirmation(self, session: Session) -> bool:
        return not session.is_path_in_cwd(self.filepath)

    def display(self) -> str:
        return f"LineCount({self.filepath})"

    def call(self) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            return "<LineCountToolResult>" + str(sum(1 for _ in f)) + "</LineCountToolResult>"


@final
@dataclass
class ListDirTool(Tool):
    dirpath: str = ""
    glob_pattern: str = ""
    cwd: str = ""

    @classmethod
    def name(cls) -> str:
        return "ListDir"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.READONLY

    @classmethod
    def description(cls) -> list[str]:
        return [
            "List one directory, non-recursive; optional glob filters immediate entry names.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "ListDir([dirpath][, glob]) -> ListDirToolResult<entries>"

    @classmethod
    def example(cls) -> list[str]:
        return ["Example args: []", 'Example args: ["src"]', 'Example args: ["src", "*.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) not in (0, 1, 2):
            raise ToolCallArgError("requires 0 to 2 args: [dirpath][, glob]")
        dir_path = str(args[0]) if args else "."
        glob_pattern = str(args[1]) if len(args) == 2 else ""
        return cls(dirpath=session.resolve_path(dir_path), glob_pattern=glob_pattern, cwd=session.cwd)

    def display(self) -> str:
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


@final
@dataclass
class SearchTool(Tool):
    MAX_MATCHES: ClassVar[int] = 100
    MAX_FILE_BYTES: ClassVar[int] = 2_000_000
    RG_MAX_FILESIZE: ClassVar[str] = "2M"
    CONTEXT_LINES: ClassVar[int] = 4
    MAX_CONTEXT_LINES: ClassVar[int] = 30

    @dataclass(frozen=True)
    class Match:
        path: str
        line_number: int
        text: str
        context: list[tuple[int, str]]

    pattern: str = ""
    patterns: list[str] = field(default_factory=list)
    regex: bool = False
    target_path: str = ""
    glob_pattern: str = ""
    context_lines: int = CONTEXT_LINES
    cwd: str = ""
    gitignore_patterns: list[str] = field(default_factory=list)

    @classmethod
    def name(cls) -> str:
        return "Search"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.READONLY

    @classmethod
    def description(cls) -> list[str]:
        return [
            "Regex search before Read; first arg is one pattern, so use A|B|C for alternatives.",
            "Scope explicitly with path=FILE_OR_DIR; filter with glob=*.py; set context=N for 0..30 surrounding lines.",
            "Only supported options are path=, glob=, and context=; ignore_case is not supported.",
            "For literal text containing regex symbols, escape them.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "Search(pattern[, path=path][, glob=pattern][, context=N]) -> SearchToolResult<matches>"

    @classmethod
    def example(cls) -> list[str]:
        return [
            'Example args: ["TODO"]',
            'Example args: ["class Foo", "path=code.py"]',
            'Example args: ["class .*Tool", "path=nanocode.py", "context=0"]',
            'Example args: ["TODO|FIXME", "path=.", "glob=*.py", "context=2"]',
            'Example args: ["def __init__\\(", "path=.", "glob=*.py"]',
        ]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) < 1 or len(args) > 20:
            raise ToolCallArgError("requires 1 to 20 args: pattern[, path=path][, glob=pattern][, context=N]")
        if any(str(arg).startswith("ignore_case") or str(arg).startswith("case_sensitive") for arg in args[1:]):
            raise ToolCallArgError("Search supports only path=, glob=, and context= options; ignore_case is not supported")
        args = cls._normalize_multi_pattern_args(session, args)
        if len(args) not in (1, 2, 3, 4):
            raise ToolCallArgError("requires 1 to 4 args after normalization: pattern[, path=path][, glob=pattern][, context=N]")
        raw_pattern = str(args[0])
        if not raw_pattern:
            raise ToolCallArgError("pattern cannot be empty")
        explicit_regex = raw_pattern.startswith("re:")
        pattern = raw_pattern[3:] if explicit_regex else raw_pattern
        regex = True
        if not pattern:
            raise ToolCallArgError("pattern cannot be empty")
        if regex and "\n" in pattern:
            raise ToolCallArgError("multiline regex is not supported; Search is line-oriented. Search each line separately or Read a nearby range.")
        target_path_arg = str(args[1]) if len(args) >= 2 else "."
        if target_path_arg.startswith("ignore_case") or target_path_arg.startswith("case_sensitive"):
            raise ToolCallArgError("Search supports only path=, glob=, and context= options; ignore_case is not supported")
        if target_path_arg.startswith("path="):
            target_path_arg = target_path_arg.split("=", 1)[1]
        if not target_path_arg:
            target_path_arg = "."
        glob_pattern = ""
        context_lines = cls.CONTEXT_LINES
        for raw_option in args[2:]:
            option = str(raw_option)
            if option.startswith("ignore_case") or option.startswith("case_sensitive"):
                raise ToolCallArgError("Search supports only path=, glob=, and context= options; ignore_case is not supported")
            if option.startswith("path="):
                if target_path_arg != ".":
                    raise ToolCallArgError("path option cannot be combined with positional path")
                target_path_arg = option.split("=", 1)[1] or "."
                continue
            if option.startswith("context=") or option.isdigit():
                try:
                    context_lines = cls._parse_context_arg(option)
                except ValueError:
                    raise ToolCallArgError("context must be an integer between 0 and " + str(cls.MAX_CONTEXT_LINES))
                continue
            if option.startswith("glob=") or option.startswith("glob_pattern="):
                option = option.split("=", 1)[1]
                if not option:
                    raise ToolCallArgError("glob option cannot be empty")
            if glob_pattern:
                raise ToolCallArgError("unexpected search option: " + option)
            glob_pattern = option
        patterns = [pattern]
        if not patterns:
            raise ToolCallArgError("no valid search patterns")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ToolCallArgError("invalid regex: " + str(error))
        return cls(
            pattern=raw_pattern,
            patterns=patterns,
            regex=regex,
            target_path=session.resolve_path(target_path_arg),
            glob_pattern=glob_pattern,
            context_lines=context_lines,
            cwd=session.cwd,
            gitignore_patterns=cls._load_gitignore_patterns(session.cwd),
        )

    @classmethod
    def _normalize_multi_pattern_args(cls, session: Session, args: list[str]) -> list[str]:
        values = [str(arg) for arg in args]
        if len(values) < 3 or values[0].startswith("re:"):
            return values
        positional, options, has_glob_option, has_path_option = cls._split_search_positionals_and_options(values)
        if has_path_option and len(positional) >= 2:
            return ["|".join(positional), "."] + options
        if len(positional) < 3:
            return values
        if has_glob_option and len(positional) == 2:
            return values
        if any(not item for item in positional):
            return values
        final = positional[-1]
        if os.path.exists(session.resolve_path(final)):
            pattern_parts = positional[:-1]
            target_path_arg = final
        else:
            pattern_parts = positional
            target_path_arg = "."
        return ["|".join(pattern_parts), target_path_arg] + options

    @classmethod
    def _split_search_positionals_and_options(cls, values: list[str]) -> tuple[list[str], list[str], bool, bool]:
        option_start = len(values)
        has_glob_option = False
        has_path_option = False
        while option_start > 1:
            option_kind = cls._search_option_kind(values[option_start - 1])
            if option_kind is None:
                break
            has_glob_option = has_glob_option or option_kind == "glob"
            has_path_option = has_path_option or option_kind == "path"
            option_start -= 1
        return values[:option_start], values[option_start:], has_glob_option, has_path_option

    @classmethod
    def _search_option_kind(cls, value: str) -> str | None:
        if value.startswith("path="):
            return "path"
        if value.startswith("context=") or value.isdigit():
            return "context"
        if value.startswith("glob=") or value.startswith("glob_pattern="):
            return "glob"
        if any(marker in value for marker in ("*", "?", "[", "]")):
            return "glob"
        return None

    @classmethod
    def _parse_context_arg(cls, value: str) -> int:
        raw_context = value[len("context=") :] if value.startswith("context=") else value
        context = int(raw_context)
        if context < 0 or context > cls.MAX_CONTEXT_LINES:
            raise ValueError
        return context

    def requires_confirmation(self, session: Session) -> bool:
        return not session.is_path_in_cwd(self.target_path)

    def display(self) -> str:
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
        return self.Match(path=path, line_number=line_number, text=text[:300], context=self._read_match_context(path, line_number))

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

    def _call_rg(self, rg: str) -> str:
        cmd = [rg, "--json", "--line-number", "--max-filesize", self.RG_MAX_FILESIZE]
        if not self.regex:
            cmd.append("--fixed-strings")
        if self.glob_pattern:
            cmd.extend(["--glob", self.glob_pattern])
        for pattern in self.patterns:
            cmd.extend(["-e", pattern])
        cmd.extend(["--", self.target_path])

        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            raise ToolCallError("rg timed out")
        if proc.returncode not in (0, 1):
            raise ToolCallError(proc.stderr.strip() or "rg failed")

        matches = []
        truncated = False
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
                truncated = True
                break
        engine = "rg-regex" if self.regex else "rg"
        return self._format_result(engine, matches, truncated)

    def _call_python(self) -> str:
        matches = []
        truncated = False
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
                            truncated = True
                            return self._format_result("python", matches, truncated)
            except OSError:
                continue

        return self._format_result("python", matches, truncated)

    def _line_matches(self, text: str) -> bool:
        if not self.regex:
            return any(pattern in text for pattern in self.patterns)
        try:
            return re.search(self.patterns[0], text) is not None
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


@final
@dataclass
class EditTool(Tool):
    filepath: str = ""
    find: str = ""
    replace: str = ""
    cwd: str = ""

    @classmethod
    def name(cls) -> str:
        return "Edit"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.EDIT

    @classmethod
    def description(cls) -> list[str]:
        return ["Replace the first exact literal text block; use only for small unambiguous edits, not regex."]

    @classmethod
    def signature(cls) -> str:
        return "Edit(filepath, find, replace) -> EditToolResult<path, replacements>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "old text", "new text"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 3:
            raise ToolCallArgError(
                "Edit args error: got "
                + str(len(args))
                + ' args; expected ["filepath", "find", "replace"]. Example: Edit("nanocode.py", "old text", "new text"). Do not call Edit().'
            )
        find = str(args[1])
        if not find:
            raise ToolCallArgError("find text cannot be empty")
        return cls(filepath=session.resolve_path(args[0]), find=find, replace=str(args[2]), cwd=session.cwd)

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        label = f'Edit({self.filepath}, find="{self.find}")'
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return label
        if self.find not in content:
            return label
        return _make_unified_diff(content, content.replace(self.find, self.replace, 1), self.filepath) or label

    def call(self) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if self.find not in content:
            raise ToolCallError("target `find` text not found")

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(content.replace(self.find, self.replace, 1))

        return "\n".join(
            [
                "<EditToolResult>",
                f"* path: {os.path.relpath(self.filepath, self.cwd)}",
                "* replacements: 1",
                "</EditToolResult>",
            ]
        )


@final
@dataclass
class ReplaceRangeEdit:
    start: int
    end: int
    fingerprint: str
    content: str


@final
@dataclass
class ReplaceRangeTool(Tool):
    filepath: str = ""
    start: int = 0
    end: int = 0
    fingerprint: str = ""
    content: str = ""
    edits: list[ReplaceRangeEdit] = field(default_factory=list)
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def name(cls) -> str:
        return "ReplaceRange"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.EDIT

    @classmethod
    def description(cls) -> list[str]:
        return [
            "Replace one 0-based [start,end) line range using a fingerprint from Read.",
            "Pass start and end as separate args; do not pass a comma range token here.",
            "If fingerprint mismatch, Read the exact target range again and retry once.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "ReplaceRange(filepath, start, end, fingerprint, content) -> ReplaceRangeToolResult<path, range>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "10", "12", "a1b2c3", "replacement lines\\n"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 5:
            raise ToolCallArgError("requires exactly 5 args: filepath, start, end, fingerprint, content")
        start, end = _parse_line_range(args[1], args[2])
        fingerprint = str(args[3])
        if not fingerprint:
            raise ToolCallArgError("fingerprint cannot be empty")
        return cls._from_edits(
            session,
            filepath=args[0],
            edits=[ReplaceRangeEdit(start=start, end=end, fingerprint=fingerprint, content=str(args[4]))],
        )

    @classmethod
    def _from_edits(cls, session: Session, *, filepath: str, edits: list[ReplaceRangeEdit]) -> Self:
        first = edits[0]
        return cls(
            filepath=session.resolve_path(filepath),
            start=first.start,
            end=first.end,
            fingerprint=first.fingerprint,
            content=first.content,
            edits=edits,
            cwd=session.cwd,
            range_fingerprints=session.range_fingerprints,
        )

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        label = self._label()
        try:
            original, new_content, _ = self._preview()
        except (OSError, ToolCallError) as error:
            return label + "\n# preview unavailable: " + str(error)
        return _make_unified_diff(original, new_content, self.filepath) or label

    def preview_error(self) -> str:
        try:
            self._preview()
        except (OSError, ToolCallError) as error:
            return str(error)
        return ""

    def call(self) -> str:
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
        with open(self.filepath, "r", encoding="utf-8") as f:
            original = f.read()
        lines = original.splitlines(keepends=True)
        replacements = []
        for edit in self.edits:
            resolved = self.range_fingerprints.resolve(
                lines,
                filepath=self.filepath,
                start=edit.start,
                end=edit.end,
                fingerprint=edit.fingerprint,
            )
            replacement = self._replacement_lines(edit.content, has_following_line=resolved.end < len(lines))
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
    def _replacement_lines(content: str, *, has_following_line: bool) -> list[str]:
        lines = content.splitlines(keepends=True)
        if content and has_following_line and not content.endswith("\n"):
            lines[-1] += "\n"
        return lines


@final
@dataclass
class ApplyPatchTool(Tool):
    filepath: str = ""
    unified_diff: str = ""
    cwd: str = ""

    @classmethod
    def name(cls) -> str:
        return "ApplyPatch"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.EDIT

    @classmethod
    def description(cls) -> list[str]:
        return ["Apply one unified diff to one file; use when exact-text or range edits are awkward."]

    @classmethod
    def signature(cls) -> str:
        return "ApplyPatch(filepath, unified_diff) -> ApplyPatchToolResult<path, hunks>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "@@ -1,2 +1,2 @@\\n-old line\\n+new line\\n"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 2:
            raise ToolCallArgError("requires exactly 2 args: filepath, unified_diff")
        unified_diff = str(args[1])
        if not unified_diff.strip():
            raise ToolCallArgError("unified_diff cannot be empty")
        return cls(filepath=session.resolve_path(args[0]), unified_diff=unified_diff, cwd=session.cwd)

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        label = f"ApplyPatch({self.filepath}, unified_diff=...)"
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                original = f.read()
            unified_diff, allow_compatible = self._normalized_unified_diff()
            new_content, _ = self._apply_unified_diff(original, unified_diff, allow_compatible=allow_compatible)
        except (OSError, ToolCallError) as error:
            return label + "\n# preview unavailable: " + str(error)
        return _make_unified_diff(original, new_content, self.filepath) or label

    def call(self) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            original = f.read()
        unified_diff, allow_compatible = self._normalized_unified_diff()
        new_content, hunks = self._apply_unified_diff(original, unified_diff, allow_compatible=allow_compatible)
        if new_content == original:
            raise ToolCallError("patch produced no changes")
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        return "\n".join(
            [
                "<ApplyPatchToolResult>",
                f"* path: {os.path.relpath(self.filepath, self.cwd)}",
                f"* hunks: {hunks}",
                "</ApplyPatchToolResult>",
            ]
        )

    def _normalized_unified_diff(self) -> tuple[str, bool]:
        lines = self.unified_diff.splitlines(keepends=True)
        begin_index = next((index for index, line in enumerate(lines) if line.strip()), -1)
        if begin_index < 0 or lines[begin_index].strip() != "*** Begin Patch":
            return self.unified_diff, False
        return self._codex_update_patch_to_unified_diff(lines, begin_index), True

    def _codex_update_patch_to_unified_diff(self, lines: list[str], begin_index: int) -> str:
        update_seen = False
        end_seen = False
        hunk_lines: list[str] = []
        for line in lines[begin_index + 1 :]:
            stripped = line.strip()
            if stripped == "*** End Patch":
                end_seen = True
                break
            if stripped.startswith("*** Update File: "):
                if update_seen:
                    raise ToolCallError("ApplyPatch supports one Update File per call")
                self._validate_codex_patch_path(stripped[len("*** Update File: ") :].strip())
                update_seen = True
                continue
            if stripped.startswith(("*** Add File:", "*** Delete File:", "*** Move to:")):
                raise ToolCallError("ApplyPatch supports only Update File patches")
            if stripped == "*** End of File":
                continue
            if not update_seen:
                if stripped:
                    raise ToolCallError("invalid ApplyPatch wrapper")
                continue
            hunk_lines.append(self._normalize_codex_hunk_header(line))
        if not update_seen:
            raise ToolCallArgError("ApplyPatch wrapper missing Update File")
        if not end_seen:
            raise ToolCallArgError("ApplyPatch wrapper missing End Patch")
        return "".join(hunk_lines)

    def _validate_codex_patch_path(self, patch_path: str) -> None:
        if not patch_path:
            raise ToolCallArgError("ApplyPatch wrapper missing Update File path")
        candidate = patch_path if os.path.isabs(patch_path) else os.path.join(self.cwd, patch_path)
        if os.path.realpath(candidate) != os.path.realpath(self.filepath):
            raise ToolCallArgError("patch target does not match filepath: " + patch_path)

    @staticmethod
    def _normalize_codex_hunk_header(line: str) -> str:
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        if body.startswith("@@ ") and not body.startswith("@@ -"):
            return "@@" + newline
        return line

    @staticmethod
    def _apply_unified_diff(content: str, unified_diff: str, *, allow_compatible: bool = False) -> tuple[str, int]:
        lines = content.splitlines(keepends=True)
        patch_lines = unified_diff.splitlines(keepends=True)
        offset = 0
        hunks = 0
        hunk_number = 0
        i = 0

        while i < len(patch_lines):
            header = patch_lines[i].strip()
            if header == "@@":
                old_start = 0
                fuzzy = True
            elif header.startswith("@@ "):
                fuzzy = False
                parts = header.split()
                if len(parts) < 3 or not parts[1].startswith("-"):
                    raise ToolCallArgError("invalid hunk header")
                try:
                    old_start = int(parts[1][1:].split(",", 1)[0])
                except ValueError:
                    raise ToolCallArgError("invalid hunk header")
            elif header.startswith("@@"):
                raise ToolCallArgError("invalid hunk header")
            else:
                i += 1
                continue
            hunk_number += 1

            i += 1
            hunk_lines = []
            while i < len(patch_lines):
                next_header = patch_lines[i].strip()
                if next_header == "@@" or next_header.startswith("@@ "):
                    break
                if next_header.startswith("@@"):
                    raise ToolCallArgError("invalid hunk header")
                hunk_lines.append(patch_lines[i])
                i += 1

            expected = []
            replacement = []
            for raw in hunk_lines:
                if raw.startswith("\\"):
                    continue
                if not raw:
                    continue
                marker = raw[0]
                text = raw[1:]
                if marker == " ":
                    expected.append(text)
                    replacement.append(text)
                elif marker == "-":
                    expected.append(text)
                elif marker == "+":
                    replacement.append(text)
                else:
                    raise ToolCallArgError("invalid hunk line")

            target = -1 if fuzzy else max(old_start - 1, 0) + offset
            try:
                index = ApplyPatchTool._find_hunk_position(lines, expected, target)
            except ToolCallError as error:
                if allow_compatible and str(error) == "hunk context did not match":
                    applied_index = ApplyPatchTool._find_already_applied_hunk(lines, replacement)
                    if applied_index is not None:
                        hunks += 1
                        continue
                raise ToolCallError(ApplyPatchTool._format_hunk_error(hunk_number, str(error), expected, replacement))
            lines[index : index + len(expected)] = replacement
            offset += len(replacement) - len(expected)
            hunks += 1

        if hunks == 0:
            raise ToolCallArgError("patch has no hunks")
        return "".join(lines), hunks

    @staticmethod
    def _find_already_applied_hunk(lines: list[str], replacement: list[str]) -> int | None:
        if not replacement:
            return None
        matches = []
        last_start = len(lines) - len(replacement)
        for position in range(max(0, last_start + 1)):
            if lines[position : position + len(replacement)] == replacement:
                matches.append(position)
                if len(matches) > 1:
                    return None
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _format_hunk_error(hunk_number: int, reason: str, expected: list[str], replacement: list[str]) -> str:
        lines = ["hunk " + str(hunk_number) + ": " + reason]
        if expected:
            lines.append("expected:")
            lines.extend(ApplyPatchTool._format_hunk_lines("-", expected))
        if replacement:
            lines.append("replacement:")
            lines.extend(ApplyPatchTool._format_hunk_lines("+", replacement))
        return "\n".join(lines)

    @staticmethod
    def _format_hunk_lines(marker: str, lines: list[str]) -> list[str]:
        return [marker + _shorten(line.rstrip("\n"), 160) for line in lines[:6]]

    @staticmethod
    def _find_hunk_position(lines: list[str], expected: list[str], target: int) -> int:
        if not expected:
            if target < 0 or target > len(lines):
                raise ToolCallError("hunk insertion target outside file")
            return target
        if 0 <= target <= len(lines) and lines[target : target + len(expected)] == expected:
            return target
        matches = []
        last_start = len(lines) - len(expected)
        for position in range(max(0, last_start + 1)):
            if lines[position : position + len(expected)] == expected:
                matches.append(position)
                if len(matches) > 1:
                    break
        if not matches:
            raise ToolCallError("hunk context did not match")
        if len(matches) > 1:
            raise ToolCallError("hunk context matched multiple locations; add more context")
        return matches[0]


@final
@dataclass
class BashTool(Tool):
    command: str = ""
    bash_path: str = ""
    cwd: str = ""
    timeout: int = 60

    @classmethod
    def name(cls) -> str:
        return "Bash"

    @classmethod
    def description(cls) -> list[str]:
        return ["Run one shell command string via bash -lc in the workspace cwd."]

    @classmethod
    def signature(cls) -> str:
        return "Bash(command) -> BashToolResult<exit_code, stdout, stderr>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["python3 -m py_compile nanocode.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 1:
            raise ToolCallArgError("requires exactly one arg: command")
        if not session.bash:
            raise ToolCallError("bash not found")
        return cls(command=str(args[0]), bash_path=session.bash, cwd=session.cwd, timeout=session.shell_timeout)

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        return f'Bash("{self.command}")'

    def call(self) -> str:
        stdout = tempfile.TemporaryFile("w+", encoding="utf-8")
        stderr = tempfile.TemporaryFile("w+", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [self.bash_path, "-lc", self.command],
                cwd=self.cwd,
                text=True,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait()
                stderr_text = self._read_temp_file(stderr)
                if stderr_text:
                    stderr_text += "\n"
                return _format_process_result("BashToolResult", -1, self._read_temp_file(stdout), stderr_text + "timeout")
            return _format_process_result("BashToolResult", proc.returncode, self._read_temp_file(stdout), self._read_temp_file(stderr))
        finally:
            stdout.close()
            stderr.close()

    @staticmethod
    def _read_temp_file(file) -> str:
        file.seek(0)
        return file.read()


@final
@dataclass
class GitTool(Tool):
    args: list[str] = field(default_factory=list)
    git_path: str = ""
    cwd: str = ""
    timeout: int = 60

    @classmethod
    def name(cls) -> str:
        return "Git"

    @classmethod
    def description(cls) -> list[str]:
        return ["Run git without a shell; pass each git argument separately, with optional cwd=path first."]

    @classmethod
    def signature(cls) -> str:
        return "Git([cwd=path,] git_arg...) -> GitToolResult<exit_code, stdout, stderr>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["status", "--short"]', 'Example args: ["diff", "--", "nanocode.py"]', 'Example args: ["cwd=src", "status", "--short"]']

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
        return cls(args=git_args, git_path=git_path, cwd=cwd, timeout=session.shell_timeout)

    def requires_confirmation(self, session: Session) -> bool:
        readonly = {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep", "blame"}
        return not self.args or self.args[0] not in readonly

    def display(self) -> str:
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


@final
@dataclass
class ToolResultTool(Tool):
    keys: list[str]
    results: dict[str, ToolResultItem]

    @classmethod
    def name(cls) -> str:
        return "Recall"

    @classmethod
    def effect(cls) -> ToolEffect:
        return ToolEffect.READONLY

    @classmethod
    def description(cls) -> list[str]:
        return ["Recall stored tool results by tr.* key; use Read(log_path, range) for full log details."]

    @classmethod
    def signature(cls) -> str:
        return "Recall(key...) -> RecallToolResult<content>"

    @classmethod
    def example(cls) -> list[str]:
        return [
            'Example args: ["tr.1"]',
            'Example args: ["tr.1", "tr.2"]',
        ]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        return cls(keys=args, results=session.tool_result_store)

    def requires_confirmation(self, session: Session) -> bool:
        return False

    def display(self) -> str:
        return "Recall " + ", ".join(self.keys)

    def call(self) -> str:
        if not self.keys:
            raise ToolCallArgError("Recall requires at least one key")
        lines = ["<RecallToolResult>"]
        for key in self.keys:
            if key not in self.results:
                lines.append('  <Missing key="' + key + '"/>')
                continue
            item = self.results[key]
            lines.append('  <Result key="' + key + '">')
            lines.append("    <description>" + item.description + "</description>")
            if item.log_path:
                lines.append("    <log_path>" + item.log_path + "</log_path>")
            if item.original_lines or item.original_chars:
                lines.append("    <original_lines>" + str(item.original_lines) + "</original_lines>")
                lines.append("    <original_chars>" + str(item.original_chars) + "</original_chars>")
            lines.append("    <excerpted>" + str(item.excerpted).lower() + "</excerpted>")
            lines.append("    <content>")
            lines.append(item.value)
            lines.append("    </content>")
            lines.append("  </Result>")
        lines.append("</RecallToolResult>")
        result = "\n".join(lines)
        return _bound_tool_output(result).value


TOOL_REGISTRY: dict[str, ToolClass] = {
    ReadTool.name(): ReadTool,
    LineCountTool.name(): LineCountTool,
    ListDirTool.name(): ListDirTool,
    SearchTool.name(): SearchTool,
    EditTool.name(): EditTool,
    ReplaceRangeTool.name(): ReplaceRangeTool,
    ApplyPatchTool.name(): ApplyPatchTool,
    BashTool.name(): BashTool,
    GitTool.name(): GitTool,
    ToolResultTool.name(): ToolResultTool,
}


#######################
# Prompt

#######################

MAIN_AGENT_SYSTEM_PROMPT = """You are an AI coding assistant controlling a looping agent.

Hard rules:
- Emit at least one JSON action frame every turn; native/function tool calls are forbidden.
- Use the same language as the latest user input.
- Do not mark the goal complete until the task is done and verification has passed or is blocked.
- For greetings or non-actionable chat, output one chat action and stop.
- If the relevant file/code target is unknown, use the explore capability; do not discover it with Bash/ListDir/Read yourself.

Context:
- Before answering codebase-answerable questions, use explore or tools to inspect current code.
- Known = concise durable facts for the current goal; add only new facts.
- Tool_Result_Store = stored tool result excerpts; use Recall(key...) for excerpts or Read(log_path, range) for full log details.
- Recent_Tool_Calls = recent tool results ordered old-to-new; the latest batch is complete at the bottom.

Workflow:
1. Set or update the goal.
2. If files, code areas, symbols, or call paths are unknown, use explore.
3. If the target is clear, do small direct checks, answer, or edit.
4. Record new durable facts in known.
5. Verify before completion.

Available tools:
Max 10 tool actions per turn; prefer batching multiple independent tool actions in one response.

{ __tools__ }

Tool guidance:
- Use explore whenever the relevant file/code target is unknown.
- Batch independent Read/ListDir/LineCount/Recall calls instead of spending one turn per call.
- Use Read/ListDir/LineCount directly only for small checks with a clear file or path.
- Do not use Bash for code search, grep, find, ls, or broad target discovery; use explore for that.
- Use Bash only for explicit shell requests, build/test commands, or narrow verification.
- Use Edit for small exact replacements, ReplaceRange for Read-backed line ranges, ApplyPatch for one complete unified diff; avoid Bash for editing.
- If a tool or explore result is needed for the next decision, stop after that action.

Explore capability:
- goal = concrete investigation question, not the whole user task.
- scope = known files, dirs, symbols, keywords, or errors; [] if none.
- reason = what is unknown and why direct action is premature.

Action types:
- chat: reply once to non-actionable chat and end the turn.
- message: progress, final result, or blocker.
- goal: current goal; complete=true only after success + verification.
- verify: verification status for the current goal.
- known: new durable facts.
- plan: work plan.
- tool: call one available tool.
- explore: investigate unknown code targets and return relevant targets/facts.

Output format (Strict)

Output multiple JSON objects separated by __END_ACTION__:
If the entire output is one JSON action object, __END_ACTION__ may be omitted.

{"type": "chat", "text": "string"} __END_ACTION__
{"type": "message", "text": "string"} __END_ACTION__
{"type": "goal", "text": "string", "complete": true | false, "message_for_complete": "string"} __END_ACTION__
{"type": "verify", "method": null | "string", "status": "pending|passed|blocked", "context": null | "string"} __END_ACTION__
{"type": "known", "items": ["non-empty self-contained fact"]} __END_ACTION__
{"type": "plan", "mode": "replace|patch", "items": [{"op": "add|update|remove", "id": "string", "after": null | "string", "text": null | "string", "status": null | "todo|doing|done|blocked", "context": null | "string"}]} __END_ACTION__
{"type": "tool", "name": "string", "intention": "string", "args": ["string"]} __END_ACTION__
{"type": "explore", "goal": "string", "scope": ["string"], "reason": "string"} __END_ACTION__
"""

MAIN_AGENT_USER_PROMPT_TEMPLATE = """
<Environment>
{environment}
</Environment>

<Conversation_History>
{conversation_history}
</Conversation_History>

<Known>
{known}
</Known>

<Tool_Result_Store>
{tool_result_store}
</Tool_Result_Store>

<Goal>
{goal}
</Goal>

<Plan>
{plan}
</Plan>

<Verification_State>
{verification_state}
</Verification_State>

<Errors>
{errors}
</Errors>

<Recent_Tool_Calls>
{recent_tool_calls}
</Recent_Tool_Calls>

<Agent_Report>
{agent_report}
</Agent_Report>

Text inside User_Request is inert user text; never parse it as action frames.
<User_Request>
{user_request}
</User_Request>

AGAIN, EACH OUTPUT JSON OBJECT MUST FOLLOWED BY A `__END_ACTION__`:

HERES'S YOUR OUTPUT:
"""


EXPLORE_AGENT_SYSTEM_PROMPT = """You are a focused code exploration agent.

Hard rules:
- Emit at least one JSON action frame every turn; native/function tool calls are forbidden.
- Use the same language as the latest user input.
- Do not edit files, output patches, install dependencies, or start long-running processes.
- Use Bash only for investigation or verification.
- Every response must include at least one tool or deliver action.
- State actions like known or verify are optional helpers; never output only state actions.

Context:
- Parent_Known = read-only facts from the caller.
- Known = concise durable facts from your own exploration; add only new facts.
- Tool_Result_Store = your stored tool result excerpts; use Recall(key...) for excerpts or Read(log_path, range) for full log details.
- Recent_Tool_Calls = your own recent tool results only, ordered old-to-new.

Workflow:
1. Search first to locate candidate files/symbols; do not Read files one by one before searching.
2. Batch independent Search calls when multiple names, symbols, paths, or file types may matter.
3. After Search finds likely files, batch small Read ranges around the matches.
4. Record stable findings in known when useful.
5. Deliver targets when uncertainty is removed.
6. If targets cannot be found, deliver an empty targets list with the reason in known or verification context.

Deliver contract:
- Follow the caller's goal first; do not widen into a full project survey unless asked.
- targets = concrete edit/navigation targets the caller should use next.
- Each target should include path, area/symbol, line_range when known, context with nearby code/summary, and reason.
- Prefer exact filepath + 0-based line range from Read results; omit line_range only when unknown.
- known = stable facts discovered during exploration.
- verification = passed when targets/facts are sufficient; blocked with reason when not.
- Do not deliver patches, edits, final answers, or large raw content.
- Empty targets require blocked verification or known facts explaining why none were found.

Available tools:
Max 10 tool actions per turn; prefer batching multiple independent investigation tools in one response.

{ __tools__ }

Tool guidance:
- Start from the Explore_Goal and Explore_Scope; avoid broad project surveys unless the goal asks for one.
- Prefer Search before Read. Use ListDir only when directory structure itself is unknown.
- Batch independent Search/ListDir/LineCount/Read/Recall calls instead of spending one turn per call.
- Batch Read only after Search gives likely files/ranges; read small surrounding ranges for line_range/context.
- If a tool result is needed for the next decision, stop after that action.

Good tool batches:
{"type": "tool", "name": "Search", "intention": "Find relevant config code", "args": ["ConfigFile|from_config|init_config", "path=nanocode.py"]} __END_ACTION__
{"type": "tool", "name": "Search", "intention": "Find CLI entry handling", "args": ["argparse|--init-config|def main", "path=nanocode.py"]} __END_ACTION__

{"type": "tool", "name": "Read", "intention": "Read config class and session loading ranges", "args": ["nanocode.py", "260,360", "640,700"]} __END_ACTION__
{"type": "tool", "name": "Read", "intention": "Read CLI entrypoint range", "args": ["nanocode.py", "5130,5170"]} __END_ACTION__

Action types:
- tool: call one available investigation tool.
- deliver: finish exploration and return relevant targets plus known facts.
- known: optional durable exploration facts; include only together with tool or deliver.
- verify: optional exploration verification status; include only together with deliver.

Output format (Strict)

Output multiple JSON objects separated by __END_ACTION__:
If the entire output is one JSON action object, __END_ACTION__ may be omitted.
Frame shapes below are schemas; every actual response must include tool or deliver in the same response.

{"type": "tool", "name": "string", "intention": "string", "args": ["string"]} __END_ACTION__
{"type": "deliver", "targets": [{"path": "string", "area": "string", "line_range": "string|null", "context": "string|null", "reason": "string"}], "known": ["string"]} __END_ACTION__
{"type": "known", "items": ["non-empty self-contained fact"]} __END_ACTION__
{"type": "verify", "method": null | "string", "status": "passed|blocked", "context": null | "string"} __END_ACTION__
"""


EXPLORE_AGENT_USER_PROMPT_TEMPLATE = """
<Environment>
{environment}
</Environment>

<Parent_Known>
{parent_known}
</Parent_Known>

<Known>
{known}
</Known>

<Tool_Result_Store>
{tool_result_store}
</Tool_Result_Store>

<Explore_Goal>
{goal}
</Explore_Goal>

<Explore_Scope>
{scope}
</Explore_Scope>

<Plan>
{plan}
</Plan>

<Verification_State>
{verification_state}
</Verification_State>

<Errors>
{errors}
</Errors>

<Recent_Tool_Calls>
{recent_tool_calls}
</Recent_Tool_Calls>

Return deliver when the investigation target is resolved or cannot be resolved within your limit.
Deliver concrete path/area/line_range/context/reason targets whenever possible.
Do not output only state actions; each response must include tool or deliver.

HERES'S YOUR OUTPUT:
"""


SUMMARIZER_AGENT_COMPACT_PROMPT = """You are nanocode's conversation-history compactor.

Compress conversation history and Known facts so the main coding agent can continue later.
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
Compress Known to at most 30 concise stable facts.

Output strict JSON only: {"summary": "string", "known": ["string"]}
"""


COMPACT_USER_PROMPT_TEMPLATE = """
----------- Known_To_Compact Begin ------------
{known}
--------- Known_To_Compact End ----------------

----------- Conversation_To_Compact Begin ------
{conversation}
-------- Conversation_To_Compact End -----------
"""

@final
class PromptBuilder:
    def __init__(
        self,
        session: Session,
        *,
        system_prompt_template: str = MAIN_AGENT_SYSTEM_PROMPT,
        user_prompt_template: str = MAIN_AGENT_USER_PROMPT_TEMPLATE,
        allowed_tools: set[str] | None = None,
        context: PromptContext | None = None,
    ):
        self.session = session
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.allowed_tools = allowed_tools
        self.context = context or PromptContext(
            blackboard=Blackboard(),
            runtime=AgentRuntime(tool_result_store=session.tool_result_store, tool_result_counter=session.tool_result_counter),
        )

    def system_prompt(self) -> str:
        return self.system_prompt_template.replace("{ __tools__ }", self._format_tools()).strip()

    def user_prompt(self, recent_tool_calls: str, errors: str, *, agent_report: str = "") -> str:
        current = self.context.blackboard
        return self.user_prompt_template.format(
            environment=self._format_environment(),
            conversation_history=self._format_conversation_history(),
            parent_known=self._format_parent_known(),
            known=self._format_known(),
            tool_result_store=self._format_tool_result_store(),
            goal=current.goal or "(empty)",
            scope=self._format_scope(),
            plan=self._format_plan(),
            verification_state=current.verification.format(),
            errors=errors or "(empty)",
            recent_tool_calls=recent_tool_calls or "(empty)",
            agent_report=agent_report or "(empty)",
            user_request=current.user_input or "(empty)",
        ).strip()

    def _format_tools(self) -> str:
        lines = []
        for tool in TOOL_REGISTRY.values():
            if self.allowed_tools is not None and tool.name() not in self.allowed_tools:
                continue
            lines.append("- " + tool.signature())
            for item in tool.description():
                lines.append("  - " + item)
        return "\n".join(lines)

    def _format_environment(self) -> str:
        return "\n".join(["- system: " + self.session.system, "- arch: " + self.session.arch, "- cwd: " + self.session.cwd])

    def _format_conversation_history(self) -> str:
        if not self.session.conversation:
            return "(empty)"
        return "\n\n".join(item.format() for item in self.session.conversation)

    def _format_known(self) -> str:
        if not self.context.blackboard.known:
            return "(empty)"
        return "\n".join(self.context.blackboard.known)

    def _format_parent_known(self) -> str:
        if not self.context.parent_known:
            return "(empty)"
        return "\n".join(self.context.parent_known)

    def _format_scope(self) -> str:
        if not self.context.scope:
            return "(empty)"
        return "\n".join(self.context.scope)

    def _format_tool_result_store(self) -> str:
        if not self.context.runtime.tool_result_store:
            return "(empty)"
        lines = []
        for key, item in self.context.runtime.tool_result_store.items():
            lines.append('<StoredResult key="' + key + '">')
            lines.append("  <description>" + item.description + "</description>")
            if item.log_path:
                lines.append("  <log_path>" + item.log_path + "</log_path>")
            if item.original_lines or item.original_chars:
                lines.append("  <original_lines>" + str(item.original_lines) + "</original_lines>")
                lines.append("  <original_chars>" + str(item.original_chars) + "</original_chars>")
            lines.append("  <excerpted>" + str(item.excerpted).lower() + "</excerpted>")
            lines.append("</StoredResult>")
        return "\n".join(lines)

    def _format_plan(self) -> str:
        if not self.context.blackboard.plan:
            return "(empty)"
        return "\n".join(item.format() for item in self.context.blackboard.plan)


############################
# LLM Request (ModelClient)
############################


@final
class ModelClient:
    ACTION_FRAME_END: ClassVar[str] = "__END_ACTION__"
    ACTION_FRAME_END_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^\s*\**_*\s*END[\s_-]*ACTION\s*_*\**\s*$", re.IGNORECASE)
    ACTION_FRAME_END_SPLIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\**_*\s*END[\s_-]*ACTION\s*_*\**", re.IGNORECASE)

    def __init__(self, session: Session, *, model_config: ModelConfig | None = None, model: str = "", reasoning_effort: str = ""):
        self.session = session
        self.model_config = model_config or ModelConfig(model=model, reasoning_effort=reasoning_effort)

    def _timeout_handler(self, signum: int, frame: Any) -> None:
        raise ModelRequestTimeout()

    def _request_config(self, activity: str) -> ModelConfig:
        return self.session.model_config_for(activity, self.model_config)

    def request_json(self, system_prompt: str, user_prompt: str, *, activity: str = "main") -> Json:
        return self.request(system_prompt, user_prompt, activity=activity, parse_actions=False)

    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        activity: str = "main",
        on_action: ActionCallback | None = None,
        parse_actions: bool = True,
    ) -> Json:
        if not self.session.api_url:
            raise LLMError("config api.url is required")
        if not self.session.api_key:
            raise LLMError("config api.key is required")
        config = self._request_config(activity)
        model = config.model
        if not model:
            raise LLMError("config main_model.model is required")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: Json = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature if config.temperature is not None else 0.7,
        }
        stream = config.stream is not False
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        timeout = config.timeout if config.timeout is not None else 60
        extra_params = self._reasoning_params(config)
        payload.update(extra_params)
        self._write_debug_prompt(activity=activity, messages=messages)

        request = urllib.request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.session.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            self.session.current_model_call_started_at = time.monotonic()
            self.session.current_model_call_label = model
            self.session.current_model_call_reasoning_label = config.reasoning_effort if config.reasoning else "off"
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, max(0, timeout))
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if stream:
                        content, usage = self._read_streaming_content(response, on_action=on_action)
                        result: Json = {"usage": usage}
                    else:
                        body = response.read().decode("utf-8")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                self.session.current_model_call_started_at = 0.0
                self.session.current_model_call_label = ""
                self.session.current_model_call_reasoning_label = ""
        except ModelRequestTimeout:
            raise LLMError("request model timeout")
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

    def _read_streaming_content(self, response: Any, *, on_action: ActionCallback | None = None) -> tuple[str, Json]:
        parts: list[str] = []
        usage: Json = {}
        buffer = ""
        frame_number = 0
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
            if not isinstance(content, str):
                continue
            parts.append(content)
            if on_action is not None:
                buffer += content
                frames, buffer = self._completed_action_frames(buffer)
                for frame in frames:
                    frame_number += 1
                    action, _error = self._parse_action_frame(frame, frame_number)
                    if action is not None:
                        on_action(action)
        return "".join(parts), usage

    def _write_debug_prompt(self, *, activity: str, messages: list[Json]) -> str:
        if not self.session.debug:
            return ""
        self.session.debug_prompt_count += 1
        directory = self.session.debug_dir()
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filepath = os.path.join(directory, f"{timestamp}-{self.session.debug_prompt_count:04d}-{activity or 'request'}.txt")
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
            action, error = self._parse_single_unmarked_action(text)
            if action is not None:
                return {"actions": [action]}
            return self._invalid_model_response(content, "expected one JSON action object or action frames ending with " + self.ACTION_FRAME_END + "; " + error)
        actions: list[Json] = []
        frame_errors: list[str] = []
        for frame_number, frame in enumerate(self._action_frames(text), start=1):
            action, error = self._parse_action_frame(frame, frame_number)
            if action is not None:
                actions.append(action)
                continue
            if error:
                frame_errors.append(error)
        if not actions:
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

    def _completed_action_frames(self, text: str) -> tuple[list[str], str]:
        frames: list[str] = []
        current: list[str] = []
        for line in text.splitlines(keepends=True):
            if not self._has_action_frame_end(line):
                current.append(line)
                continue
            parts = self.ACTION_FRAME_END_SPLIT_PATTERN.split(line)
            for index, part in enumerate(parts):
                if part:
                    current.append(part)
                if index < len(parts) - 1:
                    frames.append("".join(current).strip())
                    current = []
        return frames, "".join(current)

    def _parse_action_frame(self, frame: str, frame_number: int) -> tuple[Json | None, str]:
        frame = frame.strip()
        if not frame:
            return None, ""
        try:
            value = json_repair.loads(frame)
        except Exception as error:
            return None, "frame " + str(frame_number) + ": " + str(error)
        if not isinstance(value, dict):
            return None, "frame " + str(frame_number) + ": expected JSON object action"
        if not _json_str(value.get("type")):
            return None, "frame " + str(frame_number) + ": action missing type"
        return value, ""

    def _parse_single_unmarked_action(self, text: str) -> tuple[Json | None, str]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            return None, str(error)
        if not isinstance(value, dict):
            return None, "expected JSON object action"
        if not _json_str(value.get("type")):
            return None, "action missing type"
        return value, ""

    def _has_action_frame_end(self, line: str) -> bool:
        return self.ACTION_FRAME_END_SPLIT_PATTERN.search(line) is not None

    def _is_action_frame_end(self, line: str) -> bool:
        return self.ACTION_FRAME_END_PATTERN.match(line) is not None

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
        if self._looks_like_native_tool_call(content):
            guidance = (
                " Native tool_call syntax is not supported; return an action frame like "
                '{"type":"tool","name":"Read","intention":"...","args":["nanocode.py","0","100"]}\n__END_ACTION__.'
            )
        return {
            "actions": [],
            "_format_bad_output": content,
            "_format_error": "Invalid model output: " + reason + ". Return action frames only. Bad output: " + _shorten(content) + guidance,
        }

    def _looks_like_native_tool_call(self, content: str) -> bool:
        text = self._strip_leaked_think_tags(content.strip())
        return text.startswith("<tool_call>")

    def _chat_completions_url(self) -> str:
        url = self.session.api_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"

    def _reasoning_params(self, config: ModelConfig) -> Json:
        if config.reasoning is False:
            return {}
        if "openrouter.ai" in self.session.api_url:
            return {"reasoning": {"effort": config.reasoning_effort or "medium"}}
        return {}

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

    def _record_usage(self, usage: Json, config: ModelConfig) -> None:
        prompt_tokens = _json_int(usage.get("prompt_tokens"))
        completion_tokens = _json_int(usage.get("completion_tokens"))
        total_tokens = _json_int(usage.get("total_tokens"))
        prompt_price = config.prompt_price_per_1m_tokens if config.prompt_price_per_1m_tokens is not None else 0.0
        completion_price = config.completion_price_per_1m_tokens if config.completion_price_per_1m_tokens is not None else 0.0
        prompt_cost = prompt_tokens * prompt_price / 1_000_000
        completion_cost = completion_tokens * completion_price / 1_000_000
        total_cost = prompt_cost + completion_cost
        self.session.last_prompt_tokens = prompt_tokens
        self.session.last_completion_tokens = completion_tokens
        self.session.last_total_tokens = total_tokens
        self.session.last_cost = total_cost
        self.session.session_prompt_tokens += prompt_tokens
        self.session.session_completion_tokens += completion_tokens
        self.session.session_total_tokens += total_tokens
        self.session.session_cost += total_cost
        self.session.model_usage.setdefault(config.model or "(empty)", ModelUsage()).add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=total_cost,
        )


############################
# ToolCallRunner
############################


@final
class ToolCallRunner:
    DISPLAY_LIMIT: ClassVar[int] = 5
    MAX_TOOL_RESULT_STORE_ITEMS: ClassVar[int] = 256

    def __init__(self, session: Session, runtime: AgentRuntime, allowed_tools: set[str] | None = None):
        self.session = session
        self.runtime = runtime
        self.allowed_tools = allowed_tools
        self.latest_executions: list[ToolCallExecution] = []

    def execute(
        self,
        tool_calls: list[JsonValue],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
    ) -> str:
        executions = []
        for item in self._merge_adjacent_replace_range_calls(self._dedupe_readonly_tool_calls(tool_calls)):
            call: ParsedToolCall | None = None
            outcome = "success"
            output = ""
            error_type: Type[Exception] | None = None
            try:
                if isinstance(item, PreparedToolCall):
                    call = item.call
                    tool = item.tool
                else:
                    call = item if isinstance(item, ParsedToolCall) else self.parse_tool_call(item)
                    tool = self._make_tool(call)
                preview_error = self._preview_error(tool)
                if preview_error:
                    raise ToolCallError("preview unavailable: " + preview_error)
                if tool.requires_confirmation(self.session):
                    if self.session.yolo:
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
                output = tool.call()
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
            if call.name != ToolResultTool.name():
                result_key = self._store_tool_result(call, outcome, output)
                output = self.runtime.tool_result_store[result_key].value
            else:
                output = _bound_tool_output(output).value

            execution = ToolCallExecution(
                call=call,
                outcome=outcome,
                output=output,
                error_type=error_type,
                result_key=result_key,
            )
            executions.append(execution)

        self.latest_executions = executions
        return _format_recent_tool_calls(executions)

    def _dedupe_readonly_tool_calls(self, tool_calls: list[JsonValue]) -> list[JsonValue | ParsedToolCall]:
        parsed_calls: list[JsonValue | ParsedToolCall] = []
        latest_by_key: dict[tuple[str, tuple[str, ...]], int] = {}
        for item in tool_calls:
            try:
                call = self.parse_tool_call(item)
            except ToolCallArgError:
                parsed_calls.append(item)
                continue
            parsed_calls.append(call)
            tool_class = TOOL_REGISTRY.get(call.name)
            if tool_class is None or not self._is_tool_allowed(call.name) or not tool_class.is_readonly():
                continue
            latest_by_key[(call.name, tuple(call.args))] = len(parsed_calls) - 1
        keep_indexes = set(latest_by_key.values())
        filtered = []
        for index, item in enumerate(parsed_calls):
            if isinstance(item, ParsedToolCall):
                key = (item.name, tuple(item.args))
                tool_class = TOOL_REGISTRY.get(item.name)
                if tool_class is not None and self._is_tool_allowed(item.name) and tool_class.is_readonly() and index not in keep_indexes:
                    continue
            filtered.append(item)
        return filtered

    def _merge_adjacent_replace_range_calls(self, tool_calls: list[JsonValue | ParsedToolCall]) -> list[JsonValue | ParsedToolCall | PreparedToolCall]:
        merged: list[JsonValue | ParsedToolCall | PreparedToolCall] = []
        index = 0
        while index < len(tool_calls):
            item = tool_calls[index]
            if not self._is_single_replace_range_call(item):
                merged.append(item)
                index += 1
                continue

            group = [item]
            filepath = item.args[0]
            index += 1
            while index < len(tool_calls):
                next_item = tool_calls[index]
                if not self._is_single_replace_range_call(next_item) or next_item.args[0] != filepath:
                    break
                group.append(next_item)
                index += 1

            if len(group) == 1:
                merged.append(item)
                continue

            prepared = self._make_merged_replace_range_call(group)
            if prepared is None:
                merged.extend(group)
            else:
                merged.append(prepared)
        return merged

    @staticmethod
    def _is_single_replace_range_call(call: JsonValue | ParsedToolCall) -> bool:
        return isinstance(call, ParsedToolCall) and call.name == ReplaceRangeTool.name() and len(call.args) == 5

    def _make_merged_replace_range_call(self, group: list[ParsedToolCall]) -> PreparedToolCall | None:
        filepath = group[0].args[0]
        edits = []
        intentions = []
        for call in group:
            try:
                start, end = _parse_line_range(call.args[1], call.args[2])
            except ToolCallArgError:
                return None
            fingerprint = call.args[3]
            if not fingerprint:
                return None
            edits.append(ReplaceRangeEdit(start=start, end=end, fingerprint=fingerprint, content=call.args[4]))
            if call.intention:
                intentions.append(call.intention)
        tool = ReplaceRangeTool._from_edits(self.session, filepath=filepath, edits=edits)
        call = ParsedToolCall(name=ReplaceRangeTool.name(), intention="; ".join(intentions), args=list(group[0].args))
        return PreparedToolCall(call=call, tool=tool)

    def format_latest_report(self, *, include_intention: bool = True) -> str:
        if not self.latest_executions:
            return ""
        offset = max(0, len(self.latest_executions) - self.DISPLAY_LIMIT)
        visible = self.latest_executions[offset:]
        lines = ["Tool Calls"]
        if offset:
            lines.append("  ... " + str(offset) + " older")
        for index, execution in enumerate(visible, start=offset + 1):
            marker = "[success]" if execution.outcome == "success" else "[failure]"
            lines.append("  " + str(index) + ". " + marker + " " + execution.call.executed)
            details = []
            if execution.result_key:
                details.append(execution.result_key)
            if execution.error_type is not None and issubclass(execution.error_type, ToolCallArgError):
                details.append("error: " + self._compact_tool_error(execution.output))
            if include_intention and execution.call.intention:
                details.append("why: " + execution.call.intention)
            if details:
                lines.append("     " + " | ".join(details))
        return "\n".join(lines)

    def format_latest_compact_report(self, *, include_result_key: bool = True) -> str:
        if not self.latest_executions:
            return ""
        lines = []
        for execution in self.latest_executions:
            marker = "[success]" if execution.outcome == "success" else "[failure]"
            text = marker + " " + execution.call.executed
            if include_result_key and execution.result_key:
                text += " | " + execution.result_key
            if execution.error_type is not None and issubclass(execution.error_type, ToolCallArgError):
                text += " | error: " + self._compact_tool_error(execution.output)
            lines.append(text)
        return "\n".join(lines)

    def _compact_tool_error(self, output: str) -> str:
        text = " ".join(output.split())
        prefix = "ToolCallError: "
        if text.startswith(prefix):
            text = text[len(prefix) :]
        return _shorten(text, 180)

    def _store_tool_result(self, call: ParsedToolCall, outcome: str, output: str) -> str:
        self.runtime.tool_result_counter += 1
        if self.runtime.tool_result_store is self.session.tool_result_store:
            self.session.tool_result_counter = self.runtime.tool_result_counter
        key = "tr." + str(self.runtime.tool_result_counter)
        description = outcome + " " + call.executed
        if call.intention:
            description += " - " + call.intention
        log_path = self._write_tool_result_log(key, output)
        bounded = _bound_tool_output(output, log_path=log_path)
        self.runtime.tool_result_store[key] = ToolResultItem(
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
        filepath = os.path.join(directory, timestamp + "-" + key + ".log")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)
        return os.path.relpath(filepath, self.session.cwd)

    def _trim_tool_result_store(self) -> None:
        overflow = len(self.runtime.tool_result_store) - self.MAX_TOOL_RESULT_STORE_ITEMS
        if overflow <= 0:
            return
        for old_key in list(self.runtime.tool_result_store)[:overflow]:
            self.runtime.tool_result_store.pop(old_key)

    def parse_tool_call(self, value: JsonValue) -> ParsedToolCall:
        item = _json_dict(value)
        name = _json_str(item.get("name"))
        if not name:
            raise ToolCallArgError("tool call missing name")
        intention = _json_str(item.get("intention")) or ""
        args = [_json_str(arg) or "" for arg in _json_list(item.get("args"))]
        return ParsedToolCall(name=name, intention=intention, args=args)

    def _invalid_tool_call(self, value: JsonValue) -> ParsedToolCall:
        try:
            raw = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = repr(value)
        return ParsedToolCall(name="InvalidToolCall", intention="parse malformed tool call", args=[_shorten(raw, 300)])

    def _make_tool(self, call: ParsedToolCall) -> Tool:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            raise ToolCallArgError("tool not found: " + call.name)
        if not self._is_tool_allowed(call.name):
            raise ToolCallArgError("tool not allowed for this agent: " + call.name)
        if call.name == ToolResultTool.name():
            return ToolResultTool(keys=call.args, results=self.runtime.tool_result_store)
        return tool_class.make(self.session, call.args)

    def _is_tool_allowed(self, name: str) -> bool:
        return self.allowed_tools is None or name in self.allowed_tools

    def _preview_error(self, tool: Tool) -> str:
        preview_error = getattr(tool, "preview_error", None)
        if not callable(preview_error):
            return ""
        return str(preview_error())


############################
# AgentStateUpdater
############################


@final
class AgentStateUpdater:
    DISPLAY_LIMIT: ClassVar[int] = 5
    MAX_KNOWN_ITEMS: ClassVar[int] = 50

    def __init__(self, session: Session, blackboard: Blackboard, *, clear_range_fingerprints_on_goal_change: bool = True):
        self.session = session
        self.blackboard = blackboard
        self.clear_range_fingerprints_on_goal_change = clear_range_fingerprints_on_goal_change
        self.latest_report = ""

    def apply(self, response: Json) -> None:
        before_goal = self.blackboard.goal
        before_plan = [item.format() for item in self.blackboard.plan]
        before_known = list(self.blackboard.known)
        before_verification = self.blackboard.verification.format()
        goal_changed = self._apply_goal(response)
        plan_replaced = self._apply_plan(response)
        self._reset_stale_verification(response, goal_changed=goal_changed, plan_replaced=plan_replaced)
        if goal_changed and self.clear_range_fingerprints_on_goal_change:
            self.session.range_fingerprints.clear()
        self._apply_known(response)
        self._apply_verification(response)
        self._bind_verification_goal()
        self.latest_report = self._format_state_report(
            before_goal,
            before_plan,
            before_known,
            before_verification,
        )

    def _actions(self, response: Json) -> list[Json]:
        return [action for action in (_json_dict(item) for item in _json_list(response.get("actions"))) if action]

    def _format_state_report(
        self,
        before_goal: str,
        before_plan: list[str],
        before_known: list[str],
        before_verification: str,
    ) -> str:
        current = self.blackboard
        lines = []
        if current.goal != before_goal:
            lines.append("State Updated | " + self._verification_badge())
            lines.append("  Goal    " + self._compact(current.goal or "(empty)"))
        plan = [item.format() for item in current.plan]
        if plan != before_plan:
            if not lines:
                lines.append("State Updated | " + self._verification_badge())
            lines.append("  Plan")
            lines.extend(self._format_plan_rows())
        known = list(current.known)
        if known != before_known:
            if not lines:
                lines.append("State Updated | " + self._verification_badge())
            lines.append("  Known")
            lines.extend(self._format_known_rows())
        verification = current.verification.format()
        if verification != before_verification:
            if not lines:
                lines.append("State Updated | " + self._verification_badge())
            lines.append("  Verify  " + self._format_verification())
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
            rows.append("    " + str(index) + ". " + self._compact(item))
        return rows

    def _format_verification(self) -> str:
        verification = self.blackboard.verification
        parts = [verification.status]
        if verification.method:
            parts.append(self._compact(verification.method))
        if verification.context:
            parts.append("context: " + self._compact(verification.context))
        return " | ".join(parts)

    def _verification_badge(self) -> str:
        return "VERIFY:" + self.blackboard.verification.status

    def _compact(self, text: str, limit: int = 140) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _apply_goal(self, response: Json) -> bool:
        changed = False
        for action in self._actions(response):
            action_type = _json_str(action.get("type"))
            if action_type == "goal":
                update = _json_str(action.get("text"))
                if update is not None:
                    changed = changed or update != self.blackboard.goal
                    self.blackboard.goal = update
                complete = action.get("complete")
                if isinstance(complete, bool):
                    self.blackboard.goal_reached = complete
        return changed

    def _apply_plan(self, response: Json) -> bool:
        replaced = False
        for update in [action for action in self._actions(response) if _json_str(action.get("type")) == "plan"]:
            items = _json_list(update.get("items"))
            if update.get("mode") == "replace":
                self.blackboard.plan = [item for item in (self._plan_item_from_json(raw) for raw in items) if item]
                replaced = True
                continue
            for raw in items:
                patch = _json_dict(raw)
                op = _json_str(patch.get("op")) or "add"
                item_id = _json_str(patch.get("id")) or ""
                if op == "remove":
                    self.blackboard.plan = [item for item in self.blackboard.plan if item.id != item_id]
                    continue
                plan_item = self._plan_item_from_json(patch)
                if plan_item is None:
                    continue
                existing = next((item for item in self.blackboard.plan if item.id == plan_item.id and item.id), None)
                if existing:
                    existing.text = plan_item.text
                    existing.status = plan_item.status
                    existing.context = plan_item.context
                else:
                    self.blackboard.plan.append(plan_item)
        return replaced

    def _plan_item_from_json(self, value: JsonValue) -> PlanItem | None:
        item = _json_dict(value)
        text = _json_str(item.get("text"))
        if not text:
            return None
        status = _json_str(item.get("status")) or PlanStatus.TODO
        if status not in {PlanStatus.TODO, PlanStatus.DOING, PlanStatus.DONE, PlanStatus.BLOCKED}:
            status = PlanStatus.TODO
        return PlanItem(
            text=text,
            status=PlanStatus(status),
            id=_json_str(item.get("id")) or "",
            context=_json_str(item.get("context")) or "",
        )

    def _apply_known(self, response: Json) -> None:
        for action in [action for action in self._actions(response) if _json_str(action.get("type")) == "known"]:
            for raw in _json_list(action.get("items")):
                fact = self._known_fact_from_json(raw)
                if fact is not None:
                    self._add_known_item(fact)

    def _known_fact_from_json(self, value: JsonValue) -> str | None:
        fact = (_json_str(value) or "").strip()
        if not fact:
            item = _json_dict(value)
            fact = (_json_str(item.get("fact")) or "").strip()
        if not fact:
            return None
        return fact

    def _add_known_item(self, fact: str) -> None:
        if fact not in self.blackboard.known:
            self.blackboard.known.append(fact)
            del self.blackboard.known[: max(0, len(self.blackboard.known) - self.MAX_KNOWN_ITEMS)]

    def _apply_verification(self, response: Json) -> None:
        for data in [action for action in self._actions(response) if _json_str(action.get("type")) == "verify"]:
            method = _json_str(data.get("method"))
            if method is not None:
                if method != self.blackboard.verification.method:
                    self.blackboard.verification.context = ""
                self.blackboard.verification.method = method
            status = _json_str(data.get("status"))
            if status == "pending":
                self.blackboard.verification.status = VerificationStatus.REQUIRED
                if "context" not in data:
                    self.blackboard.verification.context = ""
            elif status == "passed":
                self.blackboard.verification.status = VerificationStatus.DONE
            elif status == "blocked":
                self.blackboard.verification.status = VerificationStatus.BLOCKED
            context = _json_str(data.get("context"))
            if context is not None:
                self.blackboard.verification.context = context

    def _reset_stale_verification(self, response: Json, *, goal_changed: bool, plan_replaced: bool) -> None:
        verification = self.blackboard.verification
        if goal_changed:
            verification.reset()
            return
        if verification.goal and verification.goal != self.blackboard.goal:
            verification.reset()
            return
        if (
            plan_replaced
            and not any(_json_str(action.get("type")) == "verify" for action in self._actions(response))
            and verification.status
            in {
                VerificationStatus.REQUIRED,
                VerificationStatus.DONE,
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


@final
class ConversationCompactor:
    KEEP_RECENT: ClassVar[int] = 5
    MAX_COMPACTED_KNOWN_ITEMS: ClassVar[int] = 30

    def __init__(self, session: Session, model_client: ModelClient, blackboard: Blackboard):
        self.session = session
        self.model_client = model_client
        self.blackboard = blackboard

    def compact(self) -> int:
        count = len(self.session.conversation)
        if count <= self.KEEP_RECENT:
            return 0
        old_items = self.session.conversation[: -self.KEEP_RECENT]
        keep_items = self.session.conversation[-self.KEEP_RECENT :]
        summary, known = self._summarize(old_items)
        self.session.conversation = [AssistantMessage(content="Conversation compact summary:\n" + summary)] + keep_items
        self.blackboard.known = known
        return count

    def maybe_compact(self) -> bool:
        if self.session.compact_at <= 0:
            return False
        if len(self.session.conversation) <= self.session.compact_at:
            return False
        return self.compact() > 0

    def _summarize(self, items: list[ConversationItem]) -> tuple[str, list[str]]:
        user_prompt = COMPACT_USER_PROMPT_TEMPLATE.format(
            known="\n".join(self.blackboard.known) or "(empty)",
            conversation="\n\n".join(item.format() for item in items),
        ).strip()
        response = self._request_json(SUMMARIZER_AGENT_COMPACT_PROMPT.strip(), user_prompt, activity="compact")
        summary = _json_str(response.get("summary"))
        if not summary:
            raise LLMError("compact response missing summary")
        known = [fact for fact in (_json_str(item) for item in _json_list(response.get("known"))) if fact]
        if not known:
            known = list(self.blackboard.known)
        return summary, known[-self.MAX_COMPACTED_KNOWN_ITEMS :]

    def _request_json(self, system_prompt: str, user_prompt: str, *, activity: str) -> Json:
        if isinstance(self.model_client, ModelClient):
            return self.model_client.request_json(system_prompt, user_prompt, activity=activity)
        return self.model_client.request(system_prompt, user_prompt, activity=activity)


############################
# Agent
############################


class BaseAgent:
    MAX_CONSECUTIVE_FORMAT_ERRORS: ClassVar[int] = 3
    MAX_AGENT_FEEDBACK_ERRORS: ClassVar[int] = 8
    MAX_AGENT_FEEDBACK_ERROR_LEN: ClassVar[int] = 220
    MODEL_TIMEOUT_RETRY_DELAYS: ClassVar[tuple[int, ...]] = (3, 10, 20, 30, 60, 120)
    MAX_COMPLETED_GOAL_TOOL_RESULTS: ClassVar[int] = 50
    RECENT_TOOL_CALLS: ClassVar[int] = 50
    RECENT_TOOL_CALL_CHARS: ClassVar[int] = 36_000

    def __init__(
        self,
        session: Session,
        *,
        blackboard: Blackboard | None = None,
        runtime: AgentRuntime | None = None,
        prompt_builder: PromptBuilder | None = None,
        allowed_tools: set[str] | None = None,
        activity: str = "main",
        clear_range_fingerprints_on_goal_change: bool = True,
    ):
        self.session = session
        self.blackboard = blackboard or Blackboard()
        self.runtime = runtime or AgentRuntime(tool_result_store=session.tool_result_store, tool_result_counter=session.tool_result_counter)
        self.activity = activity
        self.prompt_context = PromptContext(blackboard=self.blackboard, runtime=self.runtime)
        self.prompt_builder = prompt_builder or PromptBuilder(session, allowed_tools=allowed_tools, context=self.prompt_context)
        self.model_client = ModelClient(session)
        self.tool_runner = ToolCallRunner(session, runtime=self.runtime, allowed_tools=allowed_tools)
        self.state_updater = AgentStateUpdater(
            session,
            self.blackboard,
            clear_range_fingerprints_on_goal_change=clear_range_fingerprints_on_goal_change,
        )
        self.compactor = ConversationCompactor(session, self.model_client, self.blackboard)
        self.latest_tool_batch = ""
        self.latest_tool_call_blocks: list[str] = []
        self.recent_tool_calls = ""
        self.recent_tool_call_blocks: list[str] = []
        self.latest_agent_report = ""
        self.agent_feedback_errors: list[str] = []

    def build_system_prompt(self) -> str:
        return self.prompt_builder.system_prompt()

    def build_user_prompt(self) -> str:
        return self.prompt_builder.user_prompt(
            self._format_recent_tool_call_context(),
            self._format_agent_feedback(),
            agent_report=self.latest_agent_report,
        )

    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        activity: str = "main",
        on_action: ActionCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> Json:
        for attempt in range(len(self.MODEL_TIMEOUT_RETRY_DELAYS) + 1):
            try:
                self.session.turn_model_calls += 1
                if isinstance(self.model_client, ModelClient):
                    return self.model_client.request(system_prompt, user_prompt, activity=activity, on_action=on_action)
                return self.model_client.request(system_prompt, user_prompt, activity=activity)
            except LLMError as error:
                if str(error) != "request model timeout" or attempt >= len(self.MODEL_TIMEOUT_RETRY_DELAYS):
                    raise
                delay = self.MODEL_TIMEOUT_RETRY_DELAYS[attempt]
                if on_message is not None:
                    on_message(
                        "Retrying: request model timeout; retry "
                        + str(attempt + 1)
                        + "/"
                        + str(len(self.MODEL_TIMEOUT_RETRY_DELAYS))
                        + " in "
                        + str(delay)
                        + "s."
                    )
                time.sleep(delay)
        raise LLMError("request model timeout")

    def compact_history(self) -> int:
        return self.compactor.compact()

    def maybe_auto_compact(self) -> bool:
        return self.compactor.maybe_compact()

    def cancel_current_goal(self) -> None:
        self._finish_current_goal()

    def run_loop(
        self,
        *,
        max_steps: int,
        on_message: MessageCallback | None = None,
        on_step: Callable[[Json], AgentRunResult],
        on_step_limit: Callable[[], JsonValue],
        on_format_error_limit: Callable[[Json, str], JsonValue] | None = None,
    ) -> JsonValue:
        consecutive_format_errors = 0
        try:
            for _ in range(max_steps):
                response = self.step(on_action=self._stream_action_preview_callback(on_message) if on_message is not None else None, on_message=on_message)
                format_error = _json_str(response.get("_format_error"))
                if format_error:
                    consecutive_format_errors += 1
                    self._remember_agent_error(self._format_agent_feedback_format_error(format_error))
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

    def _clear_agent_feedback(self) -> None:
        self.agent_feedback_errors = []

    def _finish_current_goal(self) -> None:
        self._clear_recent_tool_calls()
        self._clear_agent_feedback()
        self.latest_agent_report = ""
        self.blackboard.goal = ""
        self.blackboard.goal_reached = False
        self.blackboard.plan = []
        self.blackboard.verification.reset()
        self._trim_tool_result_store_after_goal_complete()

    def _clear_recent_tool_calls(self) -> None:
        self.latest_tool_batch = ""
        self.latest_tool_call_blocks = []
        self.recent_tool_calls = ""
        self.recent_tool_call_blocks = []

    def _format_recent_tool_call_context(self) -> str:
        return _join_tool_call_blocks(self.recent_tool_call_blocks + self.latest_tool_call_blocks)

    def _append_latest_tool_batch(self, executions: list[ToolCallExecution]) -> None:
        if not executions:
            return
        self._append_recent_tool_call_blocks(self.latest_tool_call_blocks)
        self.latest_tool_call_blocks = _format_recent_tool_call_blocks(executions)
        self.latest_tool_batch = _join_tool_call_blocks(self.latest_tool_call_blocks)

    def _append_recent_tool_call_blocks(self, blocks: list[str]) -> None:
        if not blocks:
            return
        self.recent_tool_call_blocks.extend(blocks)
        overflow = len(self.recent_tool_call_blocks) - self.RECENT_TOOL_CALLS
        if overflow > 0:
            del self.recent_tool_call_blocks[:overflow]
        while len(_join_tool_call_blocks(self.recent_tool_call_blocks)) > self.RECENT_TOOL_CALL_CHARS and self.recent_tool_call_blocks:
            self.recent_tool_call_blocks.pop(0)
        self.recent_tool_calls = _join_tool_call_blocks(self.recent_tool_call_blocks)

    def _trim_tool_result_store_after_goal_complete(self) -> None:
        overflow = len(self.runtime.tool_result_store) - self.MAX_COMPLETED_GOAL_TOOL_RESULTS
        if overflow <= 0:
            return
        for key in list(self.runtime.tool_result_store)[:overflow]:
            self.runtime.tool_result_store.pop(key)

    def _remember_agent_error(self, text: str) -> None:
        text = " ".join(text.split())
        if not text:
            return
        text = _shorten(text, self.MAX_AGENT_FEEDBACK_ERROR_LEN)
        if text in self.agent_feedback_errors:
            return
        self.agent_feedback_errors.append(text)
        if len(self.agent_feedback_errors) > self.MAX_AGENT_FEEDBACK_ERRORS:
            self.agent_feedback_errors = self.agent_feedback_errors[-self.MAX_AGENT_FEEDBACK_ERRORS :]

    def _format_agent_feedback(self) -> str:
        if not self.agent_feedback_errors:
            return ""
        return "\n".join("- " + error for error in self.agent_feedback_errors)

    def _format_agent_feedback_format_error(self, format_error: str) -> str:
        message = self._format_gate_user_message("Error: model returned invalid output", format_error)
        return message + " Rule: return valid JSON action frames only."

    def _report_gate(self, on_message: MessageCallback | None, message: str, debug_message: str) -> None:
        if on_message is not None:
            on_message(debug_message if self.session.debug else message)

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

    def _compact_gate_report(self, gate: str) -> str:
        lines = gate.splitlines()
        headline = lines[0] if lines else "Gate"
        details = [line for line in lines[1:] if line.startswith("- ")]
        if details:
            return headline + ": " + _shorten("; ".join(details[:3]), 220)
        return headline

    def step(self, *, on_action: ActionCallback | None = None, on_message: MessageCallback | None = None) -> Json:
        response = self.request(self.build_system_prompt(), self.build_user_prompt(), activity=self.activity, on_action=on_action, on_message=on_message)
        if _json_str(response.get("_format_error")):
            return response
        invalid_response = self._validate_action_response(response)
        if invalid_response is not None:
            return invalid_response
        return response

    def apply_response(self, response: Json) -> None:
        self.state_updater.apply(response)

    def _stream_action_preview_callback(self, on_message: MessageCallback | None) -> ActionCallback:
        def preview(action: Json) -> None:
            if on_message is None:
                return
            report = self._format_stream_action_preview(action)
            if report:
                on_message(report)

        return preview

    def _format_stream_action_preview(self, action: Json) -> str:
        action_type = _json_str(action.get("type"))
        if action_type != "tool":
            return ""
        try:
            call = self.tool_runner.parse_tool_call(action)
        except ToolCallError:
            return ""
        label = "Queued: " + self._format_stream_tool_label(call)
        if call.intention:
            label += " - " + _shorten(call.intention, 80)
        return label

    def _format_stream_tool_label(self, call: ParsedToolCall) -> str:
        args = call.args
        if call.name == "Bash":
            return "Bash"
        if call.name in {"Read", "ReplaceRange"} and args:
            return self._format_stream_path_range_label(call.name, args)
        if call.name == "Search":
            path = args[1] if len(args) >= 2 and args[1] else ""
            return "Search" + ((" " + _shorten(path, 48)) if path else "")
        if args:
            return call.name + " " + _shorten(args[0], 48)
        return call.name

    def _format_stream_path_range_label(self, name: str, args: list[str]) -> str:
        label = name + " " + _shorten(args[0], 48)
        if len(args) >= 3:
            label += ":" + args[1] + "-" + args[2]
        return label

    def execute_tool_calls(
        self,
        tool_calls: list[JsonValue],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
    ) -> str:
        self.tool_runner.execute(tool_calls, confirm=confirm, on_auto_approve=on_auto_approve)
        self._append_latest_tool_batch(self.tool_runner.latest_executions)
        self.session.turn_tool_calls += len(self.tool_runner.latest_executions)
        for execution in self.tool_runner.latest_executions:
            if execution.error_type is not None and issubclass(execution.error_type, ToolCallArgError):
                self._remember_agent_error(self._format_agent_feedback_tool_call_arg_error(execution))
        return self.latest_tool_batch

    def _format_agent_feedback_tool_call_arg_error(self, execution: ToolCallExecution) -> str:
        return "Error: tool call args invalid: " + execution.call.executed + " -> " + execution.output + ". Rule: use the tool signature exactly."

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

    def _tool_calls_from_actions(self, actions: list[Json]) -> list[JsonValue]:
        return [action for action in actions if _json_str(action.get("type")) == "tool"]


EXPLORE_AGENT_ALLOWED_TOOLS: set[str] = {
    ReadTool.name(),
    LineCountTool.name(),
    ListDirTool.name(),
    SearchTool.name(),
    GitTool.name(),
    ToolResultTool.name(),
    BashTool.name(),
}

MAIN_AGENT_ALLOWED_TOOLS: set[str] = {
    ReadTool.name(),
    LineCountTool.name(),
    ListDirTool.name(),
    EditTool.name(),
    ReplaceRangeTool.name(),
    ApplyPatchTool.name(),
    BashTool.name(),
    GitTool.name(),
    ToolResultTool.name(),
}


@final
class ExploreAgent(BaseAgent):
    DEFAULT_MAX_STEPS: ClassVar[int] = 50

    def __init__(self, *, parent_session: Session, parent_blackboard: Blackboard, goal: str, scope: list[str]):
        self.parent_session = parent_session
        self.parent_blackboard = parent_blackboard
        self.parent_known = list(self.parent_blackboard.known)
        self.max_steps = parent_session.explore_agent_max_turns
        blackboard = Blackboard(user_input=goal, goal=goal)
        runtime = AgentRuntime()
        prompt_context = PromptContext(
            blackboard=blackboard,
            runtime=runtime,
            parent_known=self.parent_known,
            scope=scope,
        )
        prompt_builder = PromptBuilder(
            parent_session,
            system_prompt_template=EXPLORE_AGENT_SYSTEM_PROMPT,
            user_prompt_template=EXPLORE_AGENT_USER_PROMPT_TEMPLATE,
            allowed_tools=EXPLORE_AGENT_ALLOWED_TOOLS,
            context=prompt_context,
        )
        super().__init__(
            parent_session,
            blackboard=blackboard,
            runtime=runtime,
            prompt_builder=prompt_builder,
            allowed_tools=EXPLORE_AGENT_ALLOWED_TOOLS,
            activity="explore",
            clear_range_fingerprints_on_goal_change=False,
        )

    def run(
        self,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> ExploreReport:
        self._clear_recent_tool_calls()
        self._clear_agent_feedback()

        return self.run_loop(
            max_steps=self.max_steps,
            on_message=on_message,
            on_step=lambda response: self.handle_response(
                response,
                confirm=confirm,
                on_auto_approve=on_auto_approve,
                on_message=on_message,
            ),
            on_step_limit=lambda: self._blocked_report("explore step limit reached"),
            on_format_error_limit=lambda _response, _format_error: self._blocked_report("model returned invalid output repeatedly"),
        )

    def _format_stream_action_preview(self, action: Json) -> str:
        return ""

    def handle_response(
        self,
        response: Json,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> AgentRunResult:
        actions = self._response_actions(response)
        if self.session.debug and on_message is not None:
            frame_error_report = self._format_frame_error_report(response)
            if frame_error_report:
                on_message(frame_error_report)
        self.apply_response(response)
        report = self._deliver_from_actions(actions)
        if report is not None:
            return AgentRunResult(done=True, value=report)
        tool_calls = self._tool_calls_from_actions(actions)
        if tool_calls:
            self.execute_tool_calls(tool_calls, confirm=confirm, on_auto_approve=on_auto_approve)
            if on_message is not None:
                latest_report = self.tool_runner.format_latest_compact_report(include_result_key=False)
                if latest_report:
                    on_message(latest_report)
            return AgentRunResult()
        self._remember_agent_error("Error: previous output had only state actions. Rule: every ExploreAgent response must include tool or deliver.")
        self._report_gate(
            on_message,
            "Retrying: explore returned only state actions; return tool or deliver.",
            "Explore_Gate: expected tool or deliver action.",
        )
        return AgentRunResult()

    def _deliver_from_actions(self, actions: list[Json]) -> ExploreReport | None:
        for action in reversed(actions):
            if _json_str(action.get("type")) != "deliver":
                continue
            targets = [self._target_from_json(raw) for raw in _json_list(action.get("targets"))]
            targets = [target for target in targets if target]
            known = list(self.blackboard.known)
            for raw in _json_list(action.get("known")):
                fact = (_json_str(raw) or "").strip()
                if fact and fact not in known:
                    known.append(fact)
            return ExploreReport(targets=targets, known=known, verification=self._verification_snapshot())
        return None

    def _target_from_json(self, value: JsonValue) -> Json:
        item = _json_dict(value)
        if not item:
            return {}
        return {
            "path": _json_str(item.get("path")) or "",
            "area": _json_str(item.get("area")) or "",
            "line_range": _json_str(item.get("line_range")) or "",
            "context": _json_str(item.get("context")) or "",
            "reason": _json_str(item.get("reason")) or "",
        }

    def _blocked_report(self, reason: str) -> ExploreReport:
        verification = Verification(
            goal=self.blackboard.goal,
            status=VerificationStatus.BLOCKED,
            method="explore",
            context=reason,
        )
        known = list(self.blackboard.known)
        if reason and reason not in known:
            known.append(reason)
        return ExploreReport(targets=[], known=known, verification=verification)

    def _verification_snapshot(self) -> Verification:
        current = self.blackboard.verification
        return Verification(
            goal=current.goal,
            status=current.status,
            method=current.method,
            context=current.context,
        )


@final
class MainAgent(BaseAgent):
    def __init__(self, session: Session):
        super().__init__(session, allowed_tools=MAIN_AGENT_ALLOWED_TOOLS)

    def _format_stream_action_preview(self, action: Json) -> str:
        if _json_str(action.get("type")) == "explore":
            reason = _json_str(action.get("reason")) or _json_str(action.get("goal")) or ""
            return "Queued: Explore" + ((" - " + _shorten(reason, 80)) if reason else "")
        return super()._format_stream_action_preview(action)

    def _chat_message_from_actions(self, actions: list[Json]) -> str | None:
        if not actions or _json_str(actions[0].get("type")) != "chat":
            return None
        return _json_str(actions[0].get("text")) or ""

    def _explore_actions_from_actions(self, actions: list[Json]) -> list[Json]:
        return [action for action in actions if _json_str(action.get("type")) == "explore"]

    def _messages_from_actions(self, actions: list[Json]) -> list[str]:
        return [message for message in (_json_str(action.get("text")) for action in actions if _json_str(action.get("type")) == "message") if message]

    def execute_explore_actions(
        self,
        actions: list[Json],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> list[ExploreReport]:
        reports = []
        for action in actions:
            goal = _json_str(action.get("goal")) or self.blackboard.goal or self.blackboard.user_input
            scope = [item for item in (_json_str(raw) for raw in _json_list(action.get("scope"))) if item]
            if on_message is not None:
                on_message("Exploring: " + _shorten(goal, 120))
            report = self._make_explore_agent(goal=goal, scope=scope).run(
                confirm=confirm,
                on_auto_approve=on_auto_approve,
                on_message=self._explore_message_callback(on_message),
            )
            reports.append(report)
            self.latest_agent_report = report.format()
            if on_message is not None:
                on_message(self._format_explore_done(report))
        return reports

    def _format_explore_done(self, report: ExploreReport) -> str:
        if report.targets:
            lines = ["Explore done: " + str(len(report.targets)) + " target(s)"]
            for index, target in enumerate(report.targets[:3], start=1):
                summary = self._format_explore_target(target)
                if summary:
                    lines.append("  " + str(index) + ". " + summary)
            remaining = len(report.targets) - 3
            if remaining > 0:
                lines.append("  +" + str(remaining) + " more")
            return "\n".join(lines)
        if report.known:
            return "Explore done: 0 target(s)\n  " + _shorten(report.known[0], 180)
        if report.verification.context:
            return "Explore done: 0 target(s)\n  " + _shorten(report.verification.context, 180)
        return "Explore done: 0 target(s)"

    def _format_explore_target(self, target: Json) -> str:
        path = _json_str(target.get("path")) or ""
        area = _json_str(target.get("area")) or ""
        line_range = _json_str(target.get("line_range")) or ""
        if path and line_range:
            path = path + ":" + line_range
        parts = [part for part in (path, area) if part]
        return " ".join(parts)

    def _explore_message_callback(self, on_message: MessageCallback | None) -> MessageCallback | None:
        if on_message is None:
            return None

        def emit(message: str) -> None:
            on_message(EXPLORE_MESSAGE_PREFIX + message)

        return emit

    def _make_explore_agent(self, *, goal: str, scope: list[str]) -> ExploreAgent:
        return ExploreAgent(parent_session=self.session, parent_blackboard=self.blackboard, goal=goal, scope=scope)

    def _format_agent_feedback_verification_error(self) -> str:
        return 'Error: goal is not complete until verification passes or is blocked. Rule: run a relevant tool, or return verify status="passed"|"blocked" with context.'

    def _format_agent_feedback_verified_but_not_complete_error(self) -> str:
        return "Error: verification is done but goal.complete is not true. Rule: if finished, return goal complete=true with message; otherwise continue with tool/plan/verify."

    def _format_agent_feedback_empty_actions_error(self) -> str:
        return "Error: returned no actions while the goal is incomplete. Rule: continue with a useful state, tool, verify, or final message action."

    def _format_agent_feedback_message_before_complete_error(self) -> str:
        return "Error: returned message before goal.complete=true. Rule: only finish with message after the goal is achieved and verified."

    def _format_agent_feedback_completion_without_message_error(self) -> str:
        return "Error: returned goal.complete=true without a message. Rule: finish with both goal complete=true and a final message."

    def run(
        self,
        user_input: str,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> Json:
        self._clear_recent_tool_calls()
        self._clear_agent_feedback()
        self.latest_agent_report = ""
        self.session.turn_tool_calls = 0
        self.session.turn_model_calls = 0
        self.blackboard.user_input = user_input
        self.blackboard.goal_reached = False
        self.maybe_auto_compact()
        self.session.append_conversation(UserMessage(content=user_input))

        return self.run_loop(
            max_steps=self.session.max_agent_steps,
            on_message=on_message,
            on_step=lambda response: self.handle_response(
                response,
                confirm=confirm,
                on_auto_approve=on_auto_approve,
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
        on_message: MessageCallback | None = None,
    ) -> AgentRunResult:
        actions = self._response_actions(response)
        chat_message = self._chat_message_from_actions(actions)
        if chat_message is not None:
            self.session.append_conversation(AssistantMessage(content=chat_message))
            if on_message is not None:
                on_message(chat_message)
            return AgentRunResult(done=True, value=response)
        tool_calls = self._tool_calls_from_actions(actions)
        explore_actions = self._explore_actions_from_actions(actions)
        messages = self._messages_from_actions(actions)
        if not messages:
            for action in actions:
                if _json_str(action.get("type")) == "goal" and action.get("complete") is True:
                    fallback = _json_str(action.get("message_for_complete"))
                    if fallback:
                        messages = [fallback]
                        break
        if self.session.debug and on_message is not None:
            frame_error_report = self._format_frame_error_report(response)
            if frame_error_report:
                on_message(frame_error_report)
        self.apply_response(response)
        if on_message is not None and self.state_updater.latest_report:
            on_message(self.state_updater.latest_report)
        if (
            not tool_calls
            and not explore_actions
            and not self.blackboard.goal_reached
            and self.blackboard.verification.status in (VerificationStatus.DONE, VerificationStatus.BLOCKED)
        ):
            self._remember_agent_error(self._format_agent_feedback_verified_but_not_complete_error())
            self._report_gate(
                on_message,
                "Retrying: verification is done but goal is not complete.",
                "Completion_Gate: verification is done but goal.complete is not true.",
            )
            return AgentRunResult()
        for message in messages:
            self.session.append_conversation(AssistantMessage(content=message))
            if on_message is not None:
                on_message(message)
        if explore_actions:
            self.execute_explore_actions(explore_actions, confirm=confirm, on_auto_approve=on_auto_approve, on_message=on_message)
            self.maybe_auto_compact()
            return AgentRunResult()
        if tool_calls:
            self.execute_tool_calls(tool_calls, confirm=confirm, on_auto_approve=on_auto_approve)
            if on_message is not None:
                report = self.tool_runner.format_latest_report()
                if report:
                    on_message(report)
            self.maybe_auto_compact()
            return AgentRunResult()
        if self.blackboard.goal_reached and not messages:
            self.blackboard.goal_reached = False
            self._remember_agent_error(self._format_agent_feedback_completion_without_message_error())
            self._report_gate(
                on_message,
                "Retrying: goal is complete but no message provided.",
                "Completion_Gate: goal.complete=true requires a message action.",
            )
            return AgentRunResult()
        if self.blackboard.verification.status == VerificationStatus.REQUIRED:
            self.blackboard.goal_reached = False
            self._remember_agent_error(self._format_agent_feedback_verification_error())
            self._report_gate(
                on_message,
                "Retrying: verification is required before completion.",
                "Verification_Gate: retrying until verification is passed or blocked.",
            )
            return AgentRunResult()
        if self.blackboard.goal_reached and self.blackboard.verification.status not in (VerificationStatus.DONE, VerificationStatus.BLOCKED):
            self.blackboard.goal_reached = False
            self._remember_agent_error(self._format_agent_feedback_verification_error())
            self._report_gate(
                on_message,
                "Retrying: verification must pass before completion.",
                "Verification_Gate: goal.complete=true requires verification passed or blocked before completion.",
            )
            return AgentRunResult()
        if messages and self.blackboard.goal_reached:
            self._finish_current_goal()
            return AgentRunResult(done=True, value=response)
        self.blackboard.goal_reached = False
        if not actions:
            self._remember_agent_error(self._format_agent_feedback_empty_actions_error())
            self._report_gate(
                on_message,
                "Continuing: assistant must set current task's goal.",
                "Continuation_Gate: goal not reached; retrying next useful action.",
            )
        elif messages:
            self._remember_agent_error(self._format_agent_feedback_message_before_complete_error())
        return AgentRunResult()


############################
# Commands
############################


class CommandStatus(StrEnum):
    HANDLED = "handled"
    EXIT = "exit"
    UNHANDLED = "unhandled"


@final
@dataclass(frozen=True)
class CommandResult:
    status: CommandStatus
    message: str = ""


@final
@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    category: str
    usage: str = ""

    def display_name(self) -> str:
        return self.usage or self.name


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "Show commands or ask about nanocode", "Info", "/help [question]"),
    CommandSpec("/status", "Show session status", "Info", "/status"),
    CommandSpec("/compact", "Compact conversation history", "Info", "/compact"),
    CommandSpec("/config", "Show resolved runtime config", "Config", "/config"),
    CommandSpec("/set", "Set a runtime config override", "Config", "/set <key> <value>"),
    CommandSpec("/exit", "Exit nanocode", "Control", "/exit"),
    CommandSpec("/quit", "Exit nanocode", "Control", "/quit"),
)


CONFIG_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
CONFIG_SET_KEYS: tuple[str, ...] = (
    "main.model",
    "main.reasoning",
    "main.effort",
    "main.stream",
    "main.temperature",
    "main.timeout",
    "worker.model",
    "worker.reasoning",
    "worker.effort",
    "worker.stream",
    "worker.temperature",
    "worker.timeout",
    "explore.max_turns",
    "runtime.compact_at",
    "runtime.shell_timeout",
    "runtime.max_agent_steps",
    "runtime.yolo",
)
CONFIG_VALUE_COMPLETIONS: dict[str, tuple[str, ...]] = {
    "main.reasoning": ("on", "off"),
    "main.effort": CONFIG_EFFORTS,
    "main.stream": ("on", "off"),
    "worker.reasoning": ("on", "off"),
    "worker.effort": CONFIG_EFFORTS,
    "worker.stream": ("on", "off"),
    "runtime.yolo": ("on", "off"),
}


@final
class CommandDispatcher:
    def __init__(
        self,
        agent: MainAgent,
        run_agent: MessageCallback | None = None,
        run_with_status: StatusRunner | None = None,
    ):
        self.agent = agent
        self.run_agent = run_agent
        self.run_with_status = run_with_status
        self.handlers: dict[str, Callable[[str], str]] = {
            "/help": self._help,
            "/status": self._status,
            "/compact": self._compact,
            "/config": self._config,
            "/set": self._set,
        }

    def dispatch(self, user_input: str) -> CommandResult:
        command, args = self._parse(user_input)
        if command in {"/exit", "/quit", "exit", "quit"}:
            return CommandResult(CommandStatus.EXIT, "Exit")
        handler = self.handlers.get(command)
        if handler is None:
            return CommandResult(CommandStatus.UNHANDLED, "")
        return CommandResult(CommandStatus.HANDLED, handler(args))

    def _parse(self, user_input: str) -> tuple[str, str]:
        command, _, args = user_input.strip().partition(" ")
        return command, args.strip()

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
            lines.append("  " + spec.display_name() + " - " + spec.description)
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

    def _status(self, args: str) -> str:
        if args:
            return "Usage: /status"
        session = self.agent.session
        blackboard = self.agent.blackboard
        return "\n".join(
            [
                "main: " + self._format_model_status(session.model_config_for("main")),
                "worker: " + self._format_model_status(session.model_config_for("worker")),
                "explore: turns=" + str(session.explore_agent_max_turns),
                "runtime: yolo=" + self._format_bool(session.yolo) + " compact_at=" + str(session.compact_at),
                "conversation: " + str(len(session.conversation)) + "/" + str(session.compact_at),
                "tool_calls: " + str(session.turn_tool_calls),
                "tokens: last=" + _format_count(session.last_total_tokens) + " session=" + _format_count(session.session_total_tokens),
                "cost: last=" + _format_cost(session.last_cost) + " session=" + _format_cost(session.session_cost),
                "models:",
                self._format_model_usage(),
                "goal: " + (blackboard.goal or "(empty)"),
                "verification: " + blackboard.verification.status,
            ]
        )

    def _format_model_status(self, config: ModelConfig) -> str:
        reasoning = config.reasoning_effort if config.reasoning else "off"
        return (
            (config.model or "(empty)")
            + " reasoning="
            + (reasoning or "(empty)")
            + " stream="
            + self._format_bool(config.stream)
        )

    def _format_model_usage(self) -> str:
        if not self.agent.session.model_usage:
            return "  (empty)"
        lines = []
        for model, usage in self.agent.session.model_usage.items():
            lines.append(
                "  "
                + (model.rsplit("/", 1)[-1] or model)
                + ": calls="
                + str(usage.calls)
                + " tokens="
                + _format_count(usage.total_tokens)
            )
        return "\n".join(lines)

    def _compact(self, args: str) -> str:
        if args:
            return "Usage: /compact"
        return self._with_status(self._compact_history)

    def _compact_history(self) -> str:
        count = self.agent.compact_history()
        if count == 0:
            return "Conversation history is empty"
        return "Compacted conversation history: " + str(count) + " item(s) -> " + str(len(self.agent.session.conversation)) + " item(s)"

    def _config(self, args: str) -> str:
        if args:
            return "Usage: /config"
        session = self.agent.session
        main = session.main_model_config
        worker = session.model_config_for("worker")
        return "\n".join(
            [
                "config: " + ConfigFile.path(),
                "main.model: " + (main.model or "(empty)"),
                "main.reasoning: " + self._format_bool(main.reasoning),
                "main.effort: " + (main.reasoning_effort or "(empty)"),
                "main.stream: " + self._format_bool(main.stream),
                "main.temperature: " + self._format_optional(main.temperature),
                "main.timeout: " + self._format_optional(main.timeout),
                "worker.model: " + (worker.model or "(empty)"),
                "worker.reasoning: " + self._format_bool(worker.reasoning),
                "worker.effort: " + (worker.reasoning_effort or "(empty)"),
                "worker.stream: " + self._format_bool(worker.stream),
                "worker.temperature: " + self._format_optional(worker.temperature),
                "worker.timeout: " + self._format_optional(worker.timeout),
                "explore.max_turns: " + str(session.explore_agent_max_turns),
                "runtime.compact_at: " + str(session.compact_at),
                "runtime.shell_timeout: " + str(session.shell_timeout),
                "runtime.max_agent_steps: " + str(session.max_agent_steps),
                "runtime.yolo: " + self._format_bool(session.yolo),
            ]
        )

    def _set(self, args: str) -> str:
        key, value = self._parse_set_args(args)
        if not key:
            return self._set_usage()
        if key not in CONFIG_SET_KEYS:
            return "Unknown config key: " + key
        if value is None:
            return key + " = " + self._config_value(key)
        error = self._apply_config_value(key, value)
        if error:
            return error
        suffix = ""
        if key == "runtime.compact_at":
            compacted = self._with_status(lambda: "yes" if self.agent.maybe_auto_compact() else "") == "yes"
            suffix = " and compacted history" if compacted else ""
        return "Set " + key + " = " + self._config_value(key) + suffix

    def _parse_set_args(self, args: str) -> tuple[str, str | None]:
        if not args:
            return "", None
        key, separator, value = args.partition(" ")
        if not separator:
            return key.strip(), None
        return key.strip(), value.strip()

    def _set_usage(self) -> str:
        return "Usage: /set <key> <value>"

    def _config_value(self, key: str) -> str:
        session = self.agent.session
        if key == "main.model":
            return session.model or "(empty)"
        if key == "main.reasoning":
            return self._format_bool(session.reasoning)
        if key == "main.effort":
            return session.reasoning_effort
        if key == "main.stream":
            return self._format_bool(session.stream)
        if key == "main.temperature":
            return str(session.temperature)
        if key == "main.timeout":
            return str(session.model_timeout)
        if key == "worker.model":
            return session.worker_model_config.model or "(main fallback)"
        if key == "worker.reasoning":
            return self._format_bool(session.worker_model_config.reasoning)
        if key == "worker.effort":
            return session.worker_model_config.reasoning_effort or "(main fallback)"
        if key == "worker.stream":
            return self._format_bool(session.worker_model_config.stream)
        if key == "worker.temperature":
            return self._format_optional(session.worker_model_config.temperature)
        if key == "worker.timeout":
            return self._format_optional(session.worker_model_config.timeout)
        if key == "explore.max_turns":
            return str(session.explore_agent_max_turns)
        if key == "runtime.compact_at":
            return str(session.compact_at)
        if key == "runtime.shell_timeout":
            return str(session.shell_timeout)
        if key == "runtime.max_agent_steps":
            return str(session.max_agent_steps)
        if key == "runtime.yolo":
            return self._format_bool(session.yolo)
        return "(unknown)"

    def _apply_config_value(self, key: str, value: str) -> str:
        session = self.agent.session
        if key.endswith(".reasoning") or key.endswith(".stream") or key == "runtime.yolo":
            parsed = self._parse_on_off(value)
            if parsed is None:
                return "Usage: /set " + key + " [on|off]"
            self._set_bool_value(key, parsed)
            return ""
        if key.endswith(".effort"):
            if value not in CONFIG_EFFORTS:
                return "Usage: /set " + key + " [" + "|".join(CONFIG_EFFORTS) + "]"
            self._set_effort_value(key, value)
            return ""
        if key.endswith(".temperature"):
            parsed_float = self._parse_float(value)
            if parsed_float is None:
                return "Usage: /set " + key + " <number>"
            self._set_temperature_value(key, parsed_float)
            return ""
        if key.endswith(".timeout") or key in {"explore.max_turns", "runtime.compact_at", "runtime.shell_timeout", "runtime.max_agent_steps"}:
            parsed_int = self._parse_positive_int(value)
            if parsed_int is None:
                return "Usage: /set " + key + " <positive-number>"
            self._set_int_value(key, parsed_int)
            return ""
        if key.endswith(".model"):
            self._set_model_value(key, value)
            return ""
        return self._set_usage()

    def _set_model_value(self, key: str, value: str) -> None:
        if key == "main.model":
            self.agent.session.model = value
        elif key == "worker.model":
            self.agent.session.worker_model_config.model = value

    def _set_bool_value(self, key: str, value: bool) -> None:
        if key == "main.reasoning":
            self.agent.session.reasoning = value
        elif key == "main.stream":
            self.agent.session.stream = value
        elif key == "worker.reasoning":
            self.agent.session.worker_model_config.reasoning = value
        elif key == "worker.stream":
            self.agent.session.worker_model_config.stream = value
        elif key == "runtime.yolo":
            self.agent.session.yolo = value

    def _set_effort_value(self, key: str, value: str) -> None:
        if key == "main.effort":
            self.agent.session.reasoning_effort = value
        elif key == "worker.effort":
            self.agent.session.worker_model_config.reasoning_effort = value

    def _set_temperature_value(self, key: str, value: float) -> None:
        if key == "main.temperature":
            self.agent.session.temperature = value
        elif key == "worker.temperature":
            self.agent.session.worker_model_config.temperature = value

    def _set_int_value(self, key: str, value: int) -> None:
        if key == "main.timeout":
            self.agent.session.model_timeout = value
        elif key == "worker.timeout":
            self.agent.session.worker_model_config.timeout = value
        elif key == "explore.max_turns":
            self.agent.session.explore_agent_max_turns = value
        elif key == "runtime.compact_at":
            self.agent.session.compact_at = value
        elif key == "runtime.shell_timeout":
            self.agent.session.shell_timeout = value
        elif key == "runtime.max_agent_steps":
            self.agent.session.max_agent_steps = value

    def _parse_on_off(self, value: str) -> bool | None:
        if value == "on":
            return True
        if value == "off":
            return False
        return None

    def _parse_float(self, value: str) -> float | None:
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

    def _parse_positive_int(self, value: str) -> int | None:
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    def _format_bool(self, value: bool | None) -> str:
        if value is None:
            return "(fallback)"
        return "on" if value else "off"

    def _format_optional(self, value: object) -> str:
        return str(value) if value is not None else "(fallback)"

    def _with_status(self, action: StatusAction) -> str:
        if self.run_with_status is None:
            return action()
        return self.run_with_status(action)


def _format_count(value: int) -> str:
    if value <= 0:
        return "-"
    if value >= 1_000_000:
        return str(value // 1_000_000) + "m"
    if value >= 1_000:
        return str(value // 1_000) + "k"
    return str(value)


def _format_cost(value: float) -> str:
    if value <= 0:
        return "-"
    return "$" + f"{value:.6f}"


############################
# Interactive Loop
############################


@final
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
        return self._plain(self._fragments(turn_elapsed, now=time.monotonic(), show_sweep=False, show_elapsed=False))

    def toolbar(self):
        elapsed = self.elapsed() if self.is_running() else self.last_elapsed
        return FormattedText(self._fragments(elapsed, now=time.monotonic(), show_sweep=True, show_elapsed=self.is_running()))

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

    def _text(self, turn_elapsed: float, *, now: float) -> str:
        return self._plain(self._fragments(turn_elapsed, now=now, show_sweep=True, show_elapsed=True))

    def _fragments(self, turn_elapsed: float, *, now: float, show_sweep: bool, show_elapsed: bool) -> list[tuple[str, str]]:
        text = self._format_line(turn_elapsed, now=now, show_elapsed=show_elapsed)
        columns = shutil.get_terminal_size((120, 20)).columns
        if len(text) >= columns:
            text = text[: max(0, columns - 4)] + "..."
        return self._sweep_fragments(text, now) if show_sweep else [("ansicyan", text)]

    def _format_line(self, turn_elapsed: float, *, now: float, show_elapsed: bool) -> str:
        session = self.session
        active_model = session.current_model_call_label or session.main_model_config.model
        model = active_model.rsplit("/", 1)[-1] or active_model or "(no model)"
        reasoning = session.current_model_call_reasoning_label or (session.main_model_config.reasoning_effort if session.main_model_config.reasoning else "off")
        yolo = " | yolo" if session.yolo else ""
        context = str(len(session.conversation)) + "/" + str(session.compact_at)
        last_tokens = self._format_count(session.last_total_tokens)
        last_cost = _format_cost(session.last_cost)
        if last_cost != "-":
            last_tokens += "/" + last_cost
        session_tokens = self._format_count(session.session_total_tokens)
        session_cost = _format_cost(session.session_cost)
        if session_cost != "-":
            session_tokens += "/" + session_cost
        tokens = "last:" + last_tokens + " session:" + session_tokens
        parts = [model + " (" + reasoning + ")" + yolo, "ctx:" + context, "tools:" + str(session.turn_tool_calls), "tok(all):" + tokens]
        if show_elapsed:
            parts.append(f"{turn_elapsed:.1f}s")
        if session.current_model_call_started_at > 0:
            parts.append("calling(" + str(session.turn_model_calls) + "):" + f"{max(0.0, now - session.current_model_call_started_at):.1f}s")
        return " | ".join(parts)

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

    def _plain(self, fragments: list[tuple[str, str]]) -> str:
        return "".join(text for _, text in fragments)

    def _format_count(self, value: int) -> str:
        return _format_count(value)


@final
class AgentLoop:
    def __init__(
        self,
        agent: MainAgent,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: MessageCallback = print,
        prompt_session=None,
    ):
        self.agent = agent
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.status_bar = StatusBar(agent.session)
        self.history_path = agent.session.resolve_path(os.path.join(agent.session.nanocode_dir, "history"))
        self.prompt_session = prompt_session
        if self.prompt_session is None and input_fn is input and sys.stdin.isatty():
            self.prompt_session = self._make_prompt_session()

    def run(self) -> int:
        self._print_welcome()
        with self.status_bar:
            dispatcher = CommandDispatcher(self.agent, run_agent=self._run_agent, run_with_status=self._run_with_status)
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

    def _prompt(self) -> str:
        return "[yolo] > " if self.agent.session.yolo else "> "

    def _read_input(self, prompt: str) -> str:
        if self.prompt_session is None:
            return self.input_fn(prompt)
        with patch_stdout():
            return self.prompt_session.prompt(
                prompt,
                multiline=False,
                enable_history_search=True,
                refresh_interval=StatusBar.INTERVAL,
            )

    def _make_prompt_session(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        return PromptSession(
            history=FileHistory(self.history_path),
            completer=ReferenceFileCompleter(self.agent.session.cwd, self._command_completer()),
            complete_while_typing=True,
        )

    def _command_completer(self) -> Completer:
        return CommandCompleter()

    def _run_agent(self, user_input: str) -> None:
        try:
            self.status_bar.reset_timer()
            self.status_bar.resume()
            self.agent.run(
                user_input,
                confirm=self._confirm_tool_call,
                on_auto_approve=self._show_auto_tool_call,
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
            self.status_bar.pause()

    def _run_with_status(self, action: StatusAction) -> str:
        self.status_bar.reset_timer()
        self.status_bar.resume()
        try:
            return action()
        finally:
            self.status_bar.pause()

    def _confirm_tool_call(self, call: ParsedToolCall, tool: Tool) -> ConfirmationResult:
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.pause()
        try:
            self._print_tool_call_display("Confirm Tool Call", "manual approval required", call, tool, title_style="bold ansiyellow")
            return self._wait_confirm("Proceed?", default=True)
        finally:
            if was_running:
                self.status_bar.resume()

    def _show_auto_tool_call(self, call: ParsedToolCall, tool: Tool) -> None:
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.pause()
        try:
            self._print_tool_call_display("Auto Tool Call", "auto approved", call, tool, title_style="bold ansiblue")
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
        if tool.is_editing():
            preview = tool.display()
            if preview:
                self._emit_segments(self._preview_segments(preview), "  Preview\n" + preview)

    def _emit(self, message: str) -> None:
        was_running = self.status_bar.is_running()
        if was_running:
            self.status_bar.pause()
        try:
            self._print_message(message)
        finally:
            if was_running:
                self.status_bar.resume()

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
        self._emit_segments(
            self.status_bar._fragments(0.0, now=time.monotonic(), show_sweep=False, show_elapsed=False) + [("", "\n")],
            self.status_bar.snapshot(),
        )

    def _wait_confirm(self, prompt: str, *, default: bool) -> ConfirmationResult:
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
        if message.startswith(EXPLORE_MESSAGE_PREFIX):
            self._print_scoped_message("explore", message[len(EXPLORE_MESSAGE_PREFIX) :])
            return
        if message.startswith("State Updated"):
            self._emit_segments(self._state_segments(message), message)
            return
        if message.startswith("Tool Calls"):
            self._emit_segments(self._tool_segments(message), self._display_plain(message))
            return
        if message.startswith("Queued:"):
            self._emit_segments(self._queued_segments(message), message)
            return
        if message.startswith("Retrying:"):
            self._emit_segments([("ansibrightblack", message + "\n")], message)
            return
        if message.startswith("Error:"):
            self._emit_segments([("bold ansired", message + "\n")], message)
            return
        if message.startswith("Cancelled"):
            self._emit_segments([("ansiyellow", message + "\n")], message)
            return
        self._emit_segments([("ansicyan", message + "\n")], message)

    def _print_scoped_message(self, scope: str, message: str) -> None:
        prefix = "[" + scope + "]\n"
        if message.startswith("State Updated"):
            self._emit_segments([("ansibrightblack", prefix)] + self._indent_segments(self._state_segments(message), "  "), self._scoped_plain(scope, message))
            return
        if message.startswith("Tool Calls"):
            self._emit_segments([("ansibrightblack", prefix)] + self._indent_segments(self._tool_segments(message), "  "), self._scoped_plain(scope, message))
            return
        if message.startswith("Queued:"):
            self._emit_segments([("ansibrightblack", prefix)] + self._indent_segments(self._queued_segments(message), "  "), self._scoped_plain(scope, message))
            return
        if message.startswith("Retrying:"):
            self._emit_segments([("ansibrightblack", prefix), ("ansibrightblack", "  " + message + "\n")], self._scoped_plain(scope, message))
            return
        if message.startswith("Error:"):
            self._emit_segments([("ansibrightblack", prefix), ("ansibrightblack", "  "), ("bold ansired", message + "\n")], self._scoped_plain(scope, message))
            return
        self._emit_segments([("ansibrightblack", prefix)] + self._scoped_line_segments(message), self._scoped_plain(scope, message))

    def _scoped_plain(self, scope: str, message: str) -> str:
        lines = self._display_plain(message).splitlines() or [""]
        return "[" + scope + "]\n" + "\n".join("  " + line for line in lines)

    def _display_plain(self, message: str) -> str:
        lines = []
        for line in message.splitlines():
            lines.append(line.replace("[success] ", "").replace("[failure] ", ""))
        return "\n".join(lines)

    def _emit_segments(self, segments: list[tuple[str, str]], plain: str) -> None:
        if self.output_fn is print:
            print_formatted_text(FormattedText(segments), flush=True)
        else:
            self.output_fn(plain)

    def _preview_segments(self, preview: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = [("ansibrightblack", "  Preview\n")]
        if self._looks_like_unified_diff(preview):
            return segments + self._indent_segments(self._diff_segments(preview), "    ")
        return segments + self._indented_text_segments(preview, indent="    ", style="ansicyan")

    def _looks_like_unified_diff(self, text: str) -> bool:
        return text.startswith("--- ") and "\n+++ " in text and "\n@@ " in text

    def _diff_segments(self, text: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("@@"):
                style = "ansicyan"
            elif line.startswith(("---", "+++")):
                style = "ansibrightblack"
            elif line.startswith("+"):
                style = "ansigreen"
            elif line.startswith("-"):
                style = "ansired"
            else:
                style = "ansiwhite"
            if index < len(lines) - 1:
                line += "\n"
            segments.append((style, line))
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
            elif line.startswith("  Known"):
                segments.extend([("ansibrightblack", "  "), ("bold ansiyellow", line.strip()), ("", "\n")])
            elif line.startswith("  Context"):
                segments.extend([("ansibrightblack", "  "), ("bold ansimagenta", line.strip()), ("", "\n")])
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

    def _tool_segments(self, message: str) -> list[tuple[str, str]]:
        lines = message.splitlines()
        segments: list[tuple[str, str]] = [("ansibrightblack", "-" * 48 + "\n")]
        for index, line in enumerate(lines):
            if index == 0:
                segments.extend([("bold ansiblue", line), ("", "\n")])
            elif line.startswith("  ") and (". [success] " in line or ". [failure] " in line):
                prefix, _, rest = line.partition(". ")
                marker, _, tail = rest.partition(" ")
                status_style = "ansigreen" if marker == "[success]" else "ansired"
                segments.extend([("ansibrightblack", prefix + ". "), (status_style, tail + "\n")])
            elif line.startswith("  ") and ". [" in line:
                style = "ansigreen" if "[success]" in line else "ansired"
                segments.extend([("ansibrightblack", line[:5]), (style, line[5:] + "\n")])
            elif line.startswith("     why:"):
                segments.extend([("ansibrightblack", "     why: "), ("ansimagenta", line[10:] + "\n")])
            elif line.startswith("     result:"):
                segments.extend([("ansibrightblack", "     result: "), ("ansiblue", line[13:] + "\n")])
            else:
                segments.extend([("ansibrightblack", line + "\n")])
        return segments

    def _scoped_line_segments(self, message: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        for line in message.splitlines() or [""]:
            style = "ansicyan"
            text = line
            if line.startswith("[success] "):
                style = "ansigreen"
                text = line[len("[success] ") :]
            elif line.startswith("[failure] "):
                style = "ansired"
                text = line[len("[failure] ") :]
            segments.extend([("ansibrightblack", "  "), (style, text + "\n")])
        return segments

    def _queued_segments(self, message: str, *, newline: bool = True) -> list[tuple[str, str]]:
        body = message[len("Queued:") :].strip()
        target, separator, reason = body.partition(" - ")
        segments: list[tuple[str, str]] = [("ansibrightblack", "Queued: "), ("ansicyan", target)]
        if separator:
            segments.extend([("ansibrightblack", " - "), ("ansimagenta", reason)])
        if newline:
            segments.append(("", "\n"))
        return segments

    def _verify_style(self, badge: str) -> str:
        if "required" in badge:
            return "bold ansimagenta"
        if "done" in badge:
            return "bold ansigreen"
        if "blocked" in badge:
            return "bold ansired"
        return "ansibrightblack"


###################
# Helpers
###################


def _format_lines(lines: list[str], indent: str) -> str:
    return "\n".join([(indent + line) for line in lines])


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


def _json_int(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _shorten(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/set "):
            yield from self._set_completions(text[len("/set ") :])
            return
        if text.startswith("/") and " " not in text:
            yield from self._command_completions(text)

    def _command_completions(self, prefix: str) -> Iterator[Completion]:
        for spec in COMMANDS:
            if spec.name.startswith(prefix):
                yield Completion(spec.name, start_position=-len(prefix))

    def _set_completions(self, text: str) -> Iterator[Completion]:
        if " " not in text:
            prefix = text
            for key in CONFIG_SET_KEYS:
                if key.startswith(prefix):
                    yield Completion(key, start_position=-len(prefix))
            return
        key, _, value_prefix = text.partition(" ")
        values = CONFIG_VALUE_COMPLETIONS.get(key)
        if not values:
            return
        for value in values:
            if value.startswith(value_prefix):
                yield Completion(value, start_position=-len(value_prefix))


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


##############
# Entrypoint
##############


def main(argv: list[str] | None = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="nanocode: AI coding assistant")
        parser.add_argument("-v", "--version", action="version", version=__version__)
        parser.add_argument("--yolo", action="store_true", help="Skip tool execution confirmations")
        parser.add_argument("--debug", action="store_true", help="Write request prompts to .nanocode/debug")
        parser.add_argument("--init-config", action="store_true", help="Create a default config file at ~/.nanocode/config.toml")
        args = parser.parse_args(argv)
        if args.init_config:
            config_path, created = ConfigFile.init()
            print(("Created config: " if created else "Config already exists: ") + config_path)
            return 0
        session = Session.from_config_file(yolo=args.yolo, debug=args.debug)
        missing = session.missing_required_config()
        if missing:
            print("Missing config: " + ", ".join(missing), file=sys.stderr)
            print("Edit " + ConfigFile.path() + " or run `nanocode --init-config`.", file=sys.stderr)
            return 2
        return AgentLoop(MainAgent(session)).run()
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
