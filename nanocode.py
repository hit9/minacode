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
__version__ = "0.2.9"


class Error(Exception): ...


class ToolCallError(Exception): ...


class LLMError(Exception): ...


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
# Current (dataclasses)
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
class ContextItem(PromptItem):
    description: str
    value: str

    @override
    def format(self, indent: str = "") -> str:
        return _format_lines(["<ContextItem>", "  <description>" + self.description + "</description>", "</ContextItem>"], indent)


@final
@dataclass
class Current:
    user_input: str = ""
    goal: str = ""
    goal_reached: bool = False
    plan: list[PlanItem] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    verification: Verification = field(default_factory=Verification)


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
    REQUIRED_ENVS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("NANOCODE_API_URL", "api_url"),
        ("NANOCODE_API_KEY", "api_key"),
        ("NANOCODE_MODEL", "model"),
    )

    # ---- system ----
    system: str = field(default_factory=platform.system)
    arch: str = field(default_factory=platform.machine)
    cwd: str = field(default_factory=os.getcwd)
    bash: str = field(default_factory=lambda: shutil.which("bash") or "")

    # ---- env configs ----
    api_url: str = field(default_factory=lambda: os.environ.get("NANOCODE_API_URL", ""))  # reqiured
    api_key: str = field(default_factory=lambda: os.environ.get("NANOCODE_API_KEY", ""))  # reqiured
    model: str = field(default_factory=lambda: os.environ.get("NANOCODE_MODEL", ""))  # reqiured
    nanocode_dir: str = field(default_factory=lambda: os.environ.get("NANOCODE_DIR", ".nanocode"))
    temperature: float = field(default_factory=lambda: float(os.environ.get("NANOCODE_TEMPERATURE", "0.7")))
    reasoning: bool = field(default_factory=lambda: os.environ.get("NANOCODE_REASONING", "on") == "on")
    reasoning_effort: str = field(default_factory=lambda: os.environ.get("NANOCODE_REASONING_EFFORT", "medium"))
    stream: bool = field(default_factory=lambda: os.environ.get("NANOCODE_STREAM", "on") == "on")
    model_timeout: int = field(default_factory=lambda: int(os.environ.get("NANOCODE_MODEL_TIMEOUT", "60")))
    stream_first_token_timeout: int = field(default_factory=lambda: int(os.environ.get("NANOCODE_STREAM_FIRST_TOKEN_TIMEOUT", "30")))
    shell_timeout: int = field(default_factory=lambda: int(os.environ.get("NANOCODE_SHELL_TIMEOUT", "60")))
    compact_at: int = field(default_factory=lambda: int(os.environ.get("NANOCODE_COMPACT_AT", "50")))
    max_agent_steps: int = field(default_factory=lambda: int(os.environ.get("NANOCODE_MAX_AGENT_STEPS", "50")))
    prompt_price_per_1m_tokens: float = field(default_factory=lambda: float(os.environ.get("NANOCODE_PROMPT_PRICE_PER_1M_TOKENS", "0")))
    completion_price_per_1m_tokens: float = field(default_factory=lambda: float(os.environ.get("NANOCODE_COMPLETION_PRICE_PER_1M_TOKENS", "0")))

    # ---- runtime variables ----
    yolo: bool = False
    debug: bool = False
    debug_prompt_count: int = 0

    # ---- stats ---
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    last_cost_usd: float = 0.0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    session_cost_usd: float = 0.0
    current_model_call_started_at: float = 0.0

    # ---- current and conversation ---
    current: Current = field(default_factory=Current)
    conversation: list[ConversationItem] = field(default_factory=list)
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)
    context_store: dict[str, ContextItem] = field(default_factory=dict)

    @property
    def context(self) -> dict[str, ContextItem]:
        return self.context_store

    @context.setter
    def context(self, value: dict[str, ContextItem] | dict[str, str]) -> None:
        self.context_store = {key: item if isinstance(item, ContextItem) else ContextItem(description=key, value=str(item)) for key, item in value.items()}

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

    def missing_required_envs(self) -> list[str]:
        return [env_name for env_name, attr_name in self.REQUIRED_ENVS if not getattr(self, attr_name)]


###########
# Tools
###########


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


def _format_last_tool_calls(executions: list[ToolCallExecution]) -> str:
    blocks = [_format_last_tool_call(execution) for execution in executions]
    return "\n\n".join(blocks) or "(empty)"


def _format_last_tool_call(execution: ToolCallExecution) -> str:
    return "\n".join(
        [
            "<Last_Tool_Call>",
            "  <tool>" + execution.call.name + "</tool>",
            "  <intention>" + execution.call.intention + "</intention>",
            "  <executed>" + execution.call.executed + "</executed>",
            "  <outcome>" + execution.outcome + "</outcome>",
            "  <raw_result>",
            execution.output,
            "  </raw_result>",
            "</Last_Tool_Call>",
        ]
    )


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
        raise ToolCallError("invalid start: should be an integer")
    try:
        end = max(0, int(end_arg))
    except (ValueError, TypeError):
        raise ToolCallError("invalid end: should be an integer")
    if end:
        end = max(end, start)
    return start, end


def _parse_line_range_token(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if match is None:
        raise ToolCallError("invalid range: should be start-end")
    return _parse_line_range(match.group(1), match.group(2))


def _replacement_lines(content: str, *, has_following_line: bool) -> list[str]:
    lines = content.splitlines(keepends=True)
    if content and has_following_line and not content.endswith("\n"):
        lines[-1] += "\n"
    return lines


def _range_fingerprint(content: str) -> str:
    return hashlib.blake2s(content.encode("utf-8"), digest_size=3).hexdigest()


####################
# Tools Impl
####################


@final
@dataclass
class ReadTool(Tool):
    MAX_LINES: ClassVar[int] = 1000

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
    def description(cls) -> list[str]:
        return [
            "Read file lines and cache fingerprints for range edits.",
            "Ranges are 0-based [start,end); end=0 means EOF; pass repeated start/end pairs or start-end tokens.",
            "Returns at most 1000 lines per range; use Search/LineCount before broad reads.",
            "For ReplaceRange, read an exact or covering range first; empty inserts require an exact empty-range Read.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "Read(filepath[, start, end... | start-end...]) -> ReadToolResult<fingerprint, content>"

    @classmethod
    def example(cls) -> list[str]:
        return [
            'Example: ["code.py", "0", "120"]',
            'Example: ["code.py", "0", "40", "200", "260"]',
            'Example: ["code.py", "0-40", "200-260"]',
            'Example: ["code.py"]',
        ]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) == 0:
            raise ToolCallError("requires filepath optionally followed by start/end pairs")
        filepath = session.resolve_path(args[0])
        if len(args) == 1:
            ranges = [(0, 0)]
        elif all(re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", arg) for arg in args[1:]):
            ranges = [_parse_line_range_token(arg) for arg in args[1:]]
        else:
            if len(args) % 2 == 0:
                raise ToolCallError("requires filepath optionally followed by start/end pairs")
            ranges = [_parse_line_range(args[index], args[index + 1]) for index in range(1, len(args), 2)]
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
    def description(cls) -> list[str]:
        return ["Count file lines before choosing Read ranges."]

    @classmethod
    def signature(cls) -> str:
        return "LineCount(filepath) -> LineCountToolResult<lines>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 1:
            raise ToolCallError("requires exactly one arg: filepath")
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
    def description(cls) -> list[str]:
        return [
            "List immediate directory entries; optional glob filters entry names.",
        ]

    @classmethod
    def signature(cls) -> str:
        return 'ListDir(dir_path?: "."[, glob_pattern]) -> ListDirToolResult<entries>'

    @classmethod
    def example(cls) -> list[str]:
        return ["Example args: []", 'Example args: ["."]', 'Example args: ["src", "*.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) not in (0, 1, 2):
            raise ToolCallError("requires 0 to 2 args: [dir_path][, glob_pattern]")
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
    def description(cls) -> list[str]:
        return [
            "Search files before Read; fixed text by default, auto-regex for regex-looking patterns, or prefix re:.",
            "Use A|B|C or 3+ plain args for fixed-text OR; final existing path narrows scope.",
            "Options: path=string, context=N|N, glob=*.py or bare glob.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "Search(pattern[, path][, option...]) -> SearchToolResult<matches>; options: path, context=N|N (0..30), glob"

    @classmethod
    def example(cls) -> list[str]:
        return [
            'Example args: ["TODO"]',
            'Example args: ["class Foo", "code.py"]',
            'Example args: ["class .*Tool", "nanocode.py", "0"]',
            'Example args: ["TODO", ".", "*.py"]',
            'Example args: ["class Bar|def main", "nanocode.py", "6"]',
            'Example args: ["TODO", ".", "*.py", "8"]',
            'Example args: ["def __init__\\([^)]*,[^)]*\\)", ".", "*.py"]',
            'Example args: ["re:^class .*Tool", "nanocode.py"]',
        ]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) < 1 or len(args) > 20:
            raise ToolCallError("requires 1 to 20 args: pattern[, path][, glob_pattern][, context=N]")
        args = cls._normalize_multi_pattern_args(session, args)
        if len(args) not in (1, 2, 3, 4):
            raise ToolCallError("requires 1 to 4 args after normalization: pattern[, path][, glob_pattern][, context=N]")
        raw_pattern = str(args[0])
        if not raw_pattern:
            raise ToolCallError("pattern cannot be empty")
        explicit_regex = raw_pattern.startswith("re:")
        pattern = raw_pattern[3:] if explicit_regex else raw_pattern
        regex = explicit_regex or cls._looks_like_regex_pattern(pattern)
        if not pattern:
            raise ToolCallError("pattern cannot be empty")
        if regex and "\n" in pattern:
            raise ToolCallError("multiline regex is not supported; Search is line-oriented. Search each line separately or Read a nearby range.")
        target_path_arg = str(args[1]) if len(args) >= 2 else "."
        if target_path_arg.startswith("path="):
            target_path_arg = target_path_arg.split("=", 1)[1]
        if not target_path_arg:
            target_path_arg = "."
        glob_pattern = ""
        context_lines = cls.CONTEXT_LINES
        for raw_option in args[2:]:
            option = str(raw_option)
            if option.startswith("path="):
                if target_path_arg != ".":
                    raise ToolCallError("path option cannot be combined with positional path")
                target_path_arg = option.split("=", 1)[1] or "."
                continue
            if option.startswith("context=") or option.isdigit():
                try:
                    context_lines = cls._parse_context_arg(option)
                except ValueError:
                    raise ToolCallError("context must be an integer between 0 and " + str(cls.MAX_CONTEXT_LINES))
                continue
            if option.startswith("glob=") or option.startswith("glob_pattern="):
                option = option.split("=", 1)[1]
                if not option:
                    raise ToolCallError("glob option cannot be empty")
            if glob_pattern:
                raise ToolCallError("unexpected search option: " + option)
            glob_pattern = option
        patterns = [pattern] if regex else [part for part in pattern.split("|") if part]
        if not patterns:
            raise ToolCallError("no valid search patterns")
        if regex:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ToolCallError("invalid regex: " + str(error))
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

    @classmethod
    def _looks_like_regex_pattern(cls, pattern: str) -> bool:
        return any(marker in pattern for marker in ("\\", ".*", ".+", "\\b", "\\s", "\\d", "^", "$", "[", "]", "(", ")"))

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
            raise ToolCallError("invalid regex: " + str(error))

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
    def description(cls) -> list[str]:
        return ["Replace the first exact text block in a file; use for small unambiguous edits."]

    @classmethod
    def signature(cls) -> str:
        return "Edit(filepath, find, replace) -> EditToolResult<path, replacements>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "old text", "new text"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 3:
            raise ToolCallError("requires exactly 3 args: filepath, find, replace")
        find = str(args[1])
        if not find:
            raise ToolCallError("find text cannot be empty")
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
class ReplaceRangeTool(Tool):
    filepath: str = ""
    start: int = 0
    end: int = 0
    fingerprint: str = ""
    content: str = ""
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def name(cls) -> str:
        return "ReplaceRange"

    @classmethod
    def description(cls) -> list[str]:
        return [
            "Replace one 0-based line range using a Read fingerprint.",
            "Read an exact or covering range first; if fingerprint mismatch, Read target range and retry once.",
            "Can relocate shifted old content only when it still matches exactly once.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "ReplaceRange(filepath, start: 0-N, end: 0-N, fingerprint, content) -> ReplaceRangeToolResult<path, range>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "10", "12", "a1b2c3", "new text\\n"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 5:
            raise ToolCallError("requires exactly 5 args: filepath, start, end, fingerprint, content")
        start, end = _parse_line_range(args[1], args[2])
        fingerprint = str(args[3])
        if not fingerprint:
            raise ToolCallError("fingerprint cannot be empty")
        return cls(
            filepath=session.resolve_path(args[0]),
            start=start,
            end=end,
            fingerprint=fingerprint,
            content=str(args[4]),
            cwd=session.cwd,
            range_fingerprints=session.range_fingerprints,
        )

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        label = f"ReplaceRange({self.filepath}, {self.start}, {self.end}, {self.fingerprint})"
        try:
            original, new_content, _, _ = self._preview()
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
        original, new_content, resolved, _ = self._preview()
        if new_content == original:
            raise ToolCallError("range replacement produced no changes")
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        lines = [
            "<ReplaceRangeToolResult>",
            f"* path: {os.path.relpath(self.filepath, self.cwd)}",
            f"* range: {resolved.start}:{resolved.end}",
            f"* fingerprint: {resolved.fingerprint}",
        ]
        if resolved.relocated_from:
            old_start, old_end = resolved.relocated_from
            lines.append(f"* relocated_from: {old_start}:{old_end}")
        lines.append("</ReplaceRangeToolResult>")
        return "\n".join(lines)

    def _preview(self) -> tuple[str, str, RangeFingerprintStore.Resolved, list[str]]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            original = f.read()
        lines = original.splitlines(keepends=True)
        resolved = self.range_fingerprints.resolve(
            lines,
            filepath=self.filepath,
            start=self.start,
            end=self.end,
            fingerprint=self.fingerprint,
        )
        replacement = _replacement_lines(self.content, has_following_line=resolved.end < len(lines))
        new_lines = lines[: resolved.start] + replacement + lines[resolved.end :]
        return original, "".join(new_lines), resolved, replacement


@final
@dataclass
class BatchReplaceRangesTool(Tool):
    @final
    @dataclass
    class Edit:
        start: int
        end: int
        fingerprint: str
        content: str

    filepath: str = ""
    edits: list[Edit] = field(default_factory=list)
    cwd: str = ""
    range_fingerprints: RangeFingerprintStore = field(default_factory=RangeFingerprintStore)

    @classmethod
    def name(cls) -> str:
        return "BatchReplaceRanges"

    @classmethod
    def description(cls) -> list[str]:
        return [
            "Replace multiple non-overlapping ranges in one file using Read fingerprints.",
            "Use for several edits in the same file; ranges refer to one snapshot and do not shift within the call.",
            "Fingerprints may come from exact or covering Read ranges; same-range cached content can relocate.",
        ]

    @classmethod
    def signature(cls) -> str:
        return "BatchReplaceRanges(filepath, edits_json) -> BatchReplaceRangesToolResult<path, edits>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "[{\\"start\\":10,\\"end\\":12,\\"fingerprint\\":\\"a1b2c3\\",\\"content\\":\\"new text\\\\n\\"}]"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 2:
            raise ToolCallError("requires exactly 2 args: filepath, edits_json")
        try:
            raw_edits = json.loads(str(args[1]))
        except json.JSONDecodeError as error:
            raise ToolCallError("invalid edits_json: " + str(error))
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ToolCallError("edits_json must be a non-empty JSON array")
        edits = [cls._edit_from_json(raw) for raw in raw_edits]
        return cls(
            filepath=session.resolve_path(args[0]),
            edits=edits,
            cwd=session.cwd,
            range_fingerprints=session.range_fingerprints,
        )

    @classmethod
    def _edit_from_json(cls, raw: JsonValue) -> Edit:
        if not isinstance(raw, dict):
            raise ToolCallError("each edit must be a JSON object")
        start, end = _parse_line_range(str(raw.get("start", "")), str(raw.get("end", "")))
        fingerprint = str(raw.get("fingerprint", ""))
        if not fingerprint:
            raise ToolCallError("edit fingerprint cannot be empty")
        content = raw.get("content")
        if not isinstance(content, str):
            raise ToolCallError("edit content must be a string")
        return cls.Edit(start=start, end=end, fingerprint=fingerprint, content=content)

    def requires_confirmation(self, session: Session) -> bool:
        return True

    def display(self) -> str:
        label = f"BatchReplaceRanges({self.filepath}, edits={len(self.edits)})"
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
        original, new_content, resolved_edits = self._preview()
        if new_content == original:
            raise ToolCallError("range replacements produced no changes")
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        lines = [
            "<BatchReplaceRangesToolResult>",
            f"* path: {os.path.relpath(self.filepath, self.cwd)}",
            f"* edits: {len(resolved_edits)}",
        ]
        for index, (resolved, _) in enumerate(resolved_edits, start=1):
            line = f"* range {index}: {resolved.start}:{resolved.end}"
            if resolved.relocated_from:
                old_start, old_end = resolved.relocated_from
                line += f" relocated_from={old_start}:{old_end}"
            lines.append(line)
        lines.append("</BatchReplaceRangesToolResult>")
        return "\n".join(lines)

    def _preview(self) -> tuple[str, str, list[tuple[RangeFingerprintStore.Resolved, list[str]]]]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            original = f.read()
        lines = original.splitlines(keepends=True)
        resolved_edits = []
        for edit in self.edits:
            resolved = self.range_fingerprints.resolve(
                lines,
                filepath=self.filepath,
                start=edit.start,
                end=edit.end,
                fingerprint=edit.fingerprint,
            )
            resolved_edits.append((resolved, _replacement_lines(edit.content, has_following_line=resolved.end < len(lines))))
        self._ensure_non_overlapping(resolved_edits)

        new_lines = list(lines)
        for resolved, replacement in sorted(resolved_edits, key=lambda item: item[0].start, reverse=True):
            new_lines[resolved.start : resolved.end] = replacement
        return original, "".join(new_lines), resolved_edits

    def _ensure_non_overlapping(self, resolved_edits: list[tuple[RangeFingerprintStore.Resolved, list[str]]]) -> None:
        last_end = -1
        for resolved, _ in sorted(resolved_edits, key=lambda item: (item[0].start, item[0].end)):
            if resolved.start < last_end:
                raise ToolCallError("resolved ranges overlap")
            last_end = max(last_end, resolved.end)


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
    def description(cls) -> list[str]:
        return ["Apply one single-file unified diff; use when range fingerprints are awkward."]

    @classmethod
    def signature(cls) -> str:
        return "ApplyPatch(filepath, unified_diff) -> ApplyPatchToolResult<path, hunks>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["code.py", "@@ -1,2 +1,2 @@\\n-old\\n+new\\n"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 2:
            raise ToolCallError("requires exactly 2 args: filepath, unified_diff")
        unified_diff = str(args[1])
        if not unified_diff.strip():
            raise ToolCallError("unified_diff cannot be empty")
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
            raise ToolCallError("ApplyPatch wrapper missing Update File")
        if not end_seen:
            raise ToolCallError("ApplyPatch wrapper missing End Patch")
        return "".join(hunk_lines)

    def _validate_codex_patch_path(self, patch_path: str) -> None:
        if not patch_path:
            raise ToolCallError("ApplyPatch wrapper missing Update File path")
        candidate = patch_path if os.path.isabs(patch_path) else os.path.join(self.cwd, patch_path)
        if os.path.realpath(candidate) != os.path.realpath(self.filepath):
            raise ToolCallError("patch target does not match filepath: " + patch_path)

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
                    raise ToolCallError("invalid hunk header")
                try:
                    old_start = int(parts[1][1:].split(",", 1)[0])
                except ValueError:
                    raise ToolCallError("invalid hunk header")
            elif header.startswith("@@"):
                raise ToolCallError("invalid hunk header")
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
                    raise ToolCallError("invalid hunk header")
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
                    raise ToolCallError("invalid hunk line")

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
            raise ToolCallError("patch has no hunks")
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
        return ["Run a shell command with bash -lc."]

    @classmethod
    def signature(cls) -> str:
        return "Bash(command) -> BashToolResult<exit_code, stdout, stderr>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["python3 -m py_compile nanocode.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if len(args) != 1:
            raise ToolCallError("requires exactly one arg: command")
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
        return ["Run git without a shell; pass each argument separately."]

    @classmethod
    def signature(cls) -> str:
        return "Git(args...[, cwd=path]) -> GitToolResult<exit_code, stdout, stderr>"

    @classmethod
    def example(cls) -> list[str]:
        return ['Example args: ["status", "--short"]', 'Example args: ["diff", "--", "nanocode.py"]']

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if not args:
            raise ToolCallError("requires at least one git arg")
        git_path = shutil.which("git")
        if not git_path:
            raise ToolCallError("git not found")

        cwd = session.cwd
        git_args = [str(arg) for arg in args]
        if git_args[0].startswith("cwd="):
            cwd_arg = git_args.pop(0)[len("cwd=") :]
            if not cwd_arg:
                raise ToolCallError("cwd= requires a path")
            cwd = session.resolve_path(cwd_arg)
            if not session.is_path_in_cwd(cwd):
                raise ToolCallError(f"path outside cwd: {cwd_arg}")
            if not os.path.isdir(cwd):
                raise ToolCallError(f"cwd is not a directory: {cwd_arg}")
        if not git_args:
            raise ToolCallError("requires at least one git arg")
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
class ContextTool(Tool):
    MAX_OUTPUT_CHARS: ClassVar[int] = 20_000

    keys: list[str]
    context: dict[str, ContextItem]

    @classmethod
    def name(cls) -> str:
        return "Context"

    @classmethod
    def description(cls) -> list[str]:
        return ["Read hidden context values by key; batch multiple keys when useful."]

    @classmethod
    def signature(cls) -> str:
        return "Context(key[, key...]) -> ContextToolResult<content>"

    @classmethod
    def example(cls) -> list[str]:
        return [
            '{"name": "Context", "intention": "Read stored context", "args": ["parser.notes", "other.key"]}',
        ]

    @classmethod
    def make(cls, session: Session, args: list[str]) -> Self:
        if args and args[0].strip().lower() in {"get", "read"}:
            args = args[1:]
        return cls(keys=args, context=session.context_store)

    def requires_confirmation(self, session: Session) -> bool:
        return False

    def display(self) -> str:
        return "Context " + ", ".join(self.keys)

    def call(self) -> str:
        if not self.keys:
            raise ToolCallError("Context requires at least one key")
        lines = ["<ContextToolResult>"]
        for key in self.keys:
            if key not in self.context:
                lines.append('  <Missing key="' + key + '"/>')
                continue
            lines.extend(['  <ContextItem key="' + key + '">', self.context[key].value, "  </ContextItem>"])
        lines.append("</ContextToolResult>")
        result = "\n".join(lines)
        if len(result) <= self.MAX_OUTPUT_CHARS:
            return result
        return result[: self.MAX_OUTPUT_CHARS] + "\n...<truncated>\n</ContextToolResult>"


TOOL_REGISTRY: dict[str, ToolClass] = {
    ReadTool.name(): ReadTool,
    LineCountTool.name(): LineCountTool,
    ListDirTool.name(): ListDirTool,
    SearchTool.name(): SearchTool,
    EditTool.name(): EditTool,
    ReplaceRangeTool.name(): ReplaceRangeTool,
    BatchReplaceRangesTool.name(): BatchReplaceRangesTool,
    ApplyPatchTool.name(): ApplyPatchTool,
    BashTool.name(): BashTool,
    GitTool.name(): GitTool,
    ContextTool.name(): ContextTool,
}


#######################
# Prompt

#######################

MAIN_AGENT_SYSTEM_PROMPT = """You are an AI coding assistant controlling a looping Agent Loop.

NEVER MARK THE GOAL AS COMPLETE UNLESS THE GOAL IS ACTUALLY ACHIEVED AND VERIFICATION HAS PASSED; OTHERWISE CONTINUE THE LOOP.
USE ONLY JSON ACTION FRAMES FOR TOOL CALLS; NATIVE/FUNCTION TOOL CALLS ARE FORBIDDEN.

Memory:
- Known = concise, self-contained facts.
- Context = hidden raw support: code snippets, logs, source text, long outputs.
- Use Context(key...) to fetch hidden context by key.
- Before Read, prefer batched Context(...) if stored context may answer the question.
- Tool results are one-shot; immediately save useful facts as known and raw support as context.

STEPS:

1. Goal:
   - If the goal is not set, output goal first.

2. Fresh tool results:
   - Extract only new, stable known facts from latest tool results.
   - Store supporting raw text/logs/code snippets as context.
   - Do this before any next tool/message.

3. Memory check:
   - Use Known and Context keys/descriptions first.
   - If needed context is hidden, call batched Context(key...).
   - Only Read files when memory is missing or insufficient.
   - Context description must say what the value contains and when to reuse it.

4. Plan:
   - Create or revise the plan based on facts and the goal.

5. Act:
   - Use tools, verify, or message.
   - Verify before marking the goal as complete.
   - Report progress with message when appropriate.

Memory Tools:

{ __memory_tools__ }

Available tools:

{ __other_tools__ }

READ GATE:
- Do not Read a file if relevant Context keys exist.
- First call batched Context(key...).
- Read only after Context is missing, insufficient, or stale.

Rules:

1. Every turn must emit at least one action frame.
2. Output known only for new durable facts; do not repeat or rephrase existing Known.
3. Call at most 10 tools in one turn.
4. Prefer batched Search/Read/Context when useful.
5. Batch only independent tools.
6. If a tool result is needed for the next decision, stop after that tool batch.
7. Do not Read before checking relevant stored Context when available.

Action types:
* message: tell the user progress, result, or blocker.
* goal: set/update the current goal; complete=true only after success + verification.
* verify: record verification status for the current goal.
* known: save new durable facts with raw context.
* plan: create or update the work plan.
* tool: call a tool through JSON action frame only.

Output format (Strict)

Output multiple JSON objects separated by __END_ACTION__:

{"type": "message", "text": "string"} __END_ACTION__
{"type": "goal", "text": "string", "complete": true | false} __END_ACTION__
{"type": "verify", "method": null | "string", "status": "pending|passed|blocked", "context": null | "string"} __END_ACTION__
{"type": "known", "items": [{"fact": "non-empty self-contained string", "context": [{"key": "non-empty context key", "description": "non-empty description", "value": "non-empty raw context"}]}]} __END_ACTION__
{"type": "plan", "mode": "replace|patch", "items": [{"op": "add|update|remove", "id": "string", "after": null | "string", "text": null | "string", "status": null | "todo|doing|done|blocked", "context": null | "string"}]} __END_ACTION__
{"type": "tool", "name": "string", "intention": "string", "args": ["string"]} __END_ACTION__
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

<Context_Store>
{context}
</Context_Store>

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

<Last_Tool_Calls>
{last_tool_calls}
</Last_Tool_Calls>

<Latest_User_Input>
{latest_user_input}
</Latest_User_Input>
"""


SUMMARIZER_AGENT_COMPACT_PROMPT = """You are nanocode's conversation-history compactor.

Compress conversation history so the main coding agent can continue later.
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

Output strict JSON only: {"summary": "string"}
"""


COMPACT_USER_PROMPT_TEMPLATE = """
----------- Conversation_To_Compact Begin ------
{conversation}
-------- Conversation_To_Compact End -----------
"""


@final
class PromptBuilder:
    def __init__(self, session: Session):
        self.session = session

    def system_prompt(self) -> str:
        return (
            MAIN_AGENT_SYSTEM_PROMPT.replace("{ __memory_tools__ }", self._format_tools(memory=True))
            .replace("{ __other_tools__ }", self._format_tools(memory=False))
            .replace("{ __tools__ }", self._format_tools())
            .strip()
        )

    def user_prompt(self, last_tool_calls: str, errors: str) -> str:
        current = self.session.current
        return MAIN_AGENT_USER_PROMPT_TEMPLATE.format(
            environment=self._format_environment(),
            conversation_history=self._format_conversation_history(),
            known=self._format_known(),
            context=self._format_context(),
            goal=current.goal or "(empty)",
            plan=self._format_plan(),
            verification_state=current.verification.format(),
            errors=errors or "(empty)",
            last_tool_calls=last_tool_calls or "(empty)",
            latest_user_input=current.user_input or "(empty)",
        ).strip()

    def _format_tools(self, memory: bool | None = None) -> str:
        lines = []
        for tool in TOOL_REGISTRY.values():
            is_memory_tool = tool.name() == ContextTool.name()
            if memory is not None and is_memory_tool != memory:
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
        if not self.session.current.known:
            return "(empty)"
        return "\n".join(self.session.current.known)

    def _format_context(self) -> str:
        if not self.session.context_store:
            return "(empty)"
        lines = []
        for key, item in self.session.context_store.items():
            lines.extend(['<ContextItem key="' + key + '">', "  <description>" + item.description + "</description>", "</ContextItem>"])
        return "\n".join(lines)

    def _format_plan(self) -> str:
        if not self.session.current.plan:
            return "(empty)"
        return "\n".join(item.format() for item in self.session.current.plan)


############################
# LLM Request (ModelClient)
############################


@final
class ModelClient:
    ACTION_FRAME_END: ClassVar[str] = "__END_ACTION__"
    ACTION_FRAME_END_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^\s*\**_*\s*END[\s_-]*ACTION\s*_*\**\s*$", re.IGNORECASE)
    ACTION_FRAME_END_SPLIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\**_*\s*END[\s_-]*ACTION\s*_*\**", re.IGNORECASE)

    def __init__(self, session: Session):
        self.session = session

    def request(self, system_prompt: str, user_prompt: str, *, activity: str = "main", on_action: ActionCallback | None = None) -> Json:
        if not self.session.api_url:
            raise LLMError("NANOCODE_API_URL is required")
        if not self.session.api_key:
            raise LLMError("NANOCODE_API_KEY is required")
        if not self.session.model:
            raise LLMError("NANOCODE_MODEL is required")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: Json = {
            "model": self.session.model,
            "messages": messages,
            "temperature": self.session.temperature,
        }
        if self.session.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        extra_params = self._reasoning_params()
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
            try:
                with urllib.request.urlopen(request, timeout=self.session.model_timeout) as response:
                    if self.session.stream:
                        content, usage = self._read_streaming_content(response, on_action=on_action)
                        result: Json = {"usage": usage}
                    else:
                        body = response.read().decode("utf-8")
            finally:
                self.session.current_model_call_started_at = 0.0
        except socket.timeout:
            raise LLMError("request model timeout")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise LLMError("API request failed: HTTP " + str(error.code) + ": " + _shorten(body))
        except Exception as error:
            raise LLMError(str(error))

        if not self.session.stream:
            try:
                result = json.loads(body)
            except json.JSONDecodeError:
                raise LLMError("API response is not JSON: " + _shorten(body))

        self._record_usage(_json_dict(result.get("usage") if isinstance(result, dict) else None))
        if not self.session.stream:
            content = self._message_content(result)
        if content is None:
            return self._invalid_model_response(self._format_missing_message_content(result))
        return self._parse_model_content(content)

    def _read_streaming_content(self, response: Any, *, on_action: ActionCallback | None = None) -> tuple[str, Json]:
        parts: list[str] = []
        usage: Json = {}
        buffer = ""
        frame_number = 0
        first_content = True
        if self.session.stream_first_token_timeout > 0:
            self._set_stream_read_timeout(response, self.session.stream_first_token_timeout)
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
            if first_content:
                first_content = False
                self._set_stream_read_timeout(response, self.session.model_timeout)
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

    def _set_stream_read_timeout(self, response: Any, timeout: int) -> bool:
        raw = getattr(response, "fp", None)
        raw = getattr(raw, "raw", raw)
        candidates = [
            response,
            getattr(response, "sock", None),
            raw,
            getattr(raw, "_sock", None),
        ]
        for candidate in candidates:
            setter = getattr(candidate, "settimeout", None)
            if not callable(setter):
                continue
            try:
                setter(timeout)
            except Exception:
                continue
            return True
        return False

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

    def _reasoning_params(self) -> Json:
        if not self.session.reasoning:
            return {}
        if "openrouter.ai" in self.session.api_url:
            return {"reasoning": {"effort": self.session.reasoning_effort}}
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

    def _record_usage(self, usage: Json) -> None:
        prompt_tokens = _json_int(usage.get("prompt_tokens"))
        completion_tokens = _json_int(usage.get("completion_tokens"))
        total_tokens = _json_int(usage.get("total_tokens"))
        prompt_cost = prompt_tokens * self.session.prompt_price_per_1m_tokens / 1_000_000
        completion_cost = completion_tokens * self.session.completion_price_per_1m_tokens / 1_000_000
        total_cost = prompt_cost + completion_cost
        self.session.last_prompt_tokens = prompt_tokens
        self.session.last_completion_tokens = completion_tokens
        self.session.last_total_tokens = total_tokens
        self.session.last_cost_usd = total_cost
        self.session.session_prompt_tokens += prompt_tokens
        self.session.session_completion_tokens += completion_tokens
        self.session.session_total_tokens += total_tokens
        self.session.session_cost_usd += total_cost


############################
# ToolCallRunner
############################


@final
class ToolCallRunner:
    DISPLAY_LIMIT: ClassVar[int] = 5

    def __init__(self, session: Session):
        self.session = session
        self.latest_executions: list[ToolCallExecution] = []

    def execute(
        self,
        tool_calls: list[JsonValue],
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
    ) -> str:
        executions = []
        for item in tool_calls:
            call: ParsedToolCall | None = None
            outcome = "success"
            output = ""
            try:
                call = self.parse_tool_call(item)
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
            except Exception as error:
                outcome = "failure"
                output = "ToolCallError: " + str(error)
            if call is None:
                call = self._invalid_tool_call(item)

            execution = ToolCallExecution(
                call=call,
                outcome=outcome,
                output=output,
            )
            executions.append(execution)

        self.latest_executions = executions
        return _format_last_tool_calls(executions)

    def format_latest_report(self) -> str:
        if not self.latest_executions:
            return ""
        offset = max(0, len(self.latest_executions) - self.DISPLAY_LIMIT)
        visible = self.latest_executions[offset:]
        lines = ["Tool Calls"]
        if offset:
            lines.append("  ... " + str(offset) + " older")
        for index, execution in enumerate(visible, start=offset + 1):
            lines.append("  " + str(index) + ". [" + execution.outcome + "] " + execution.call.executed)
            if execution.call.intention:
                lines.append("     why: " + execution.call.intention)
        return "\n".join(lines)

    def parse_tool_call(self, value: JsonValue) -> ParsedToolCall:
        item = _json_dict(value)
        name = _json_str(item.get("name"))
        if not name:
            raise ToolCallError("tool call missing name")
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
            raise ToolCallError("tool not found: " + call.name)
        return tool_class.make(self.session, call.args)

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
    MAX_CONTEXT_ITEMS: ClassVar[int] = 300
    MAX_CONTEXT_VALUE_CHARS: ClassVar[int] = 12_000

    def __init__(self, session: Session):
        self.session = session
        self.latest_report = ""

    def apply(self, response: Json) -> None:
        before_goal = self.session.current.goal
        before_plan = [item.format() for item in self.session.current.plan]
        before_known = list(self.session.current.known)
        before_context = dict(self.session.context_store)
        before_verification = self.session.current.verification.format()
        goal_changed = self._apply_goal(response)
        plan_replaced = self._apply_plan(response)
        self._reset_stale_verification(response, goal_changed=goal_changed, plan_replaced=plan_replaced)
        if goal_changed:
            self.session.range_fingerprints.clear()
        self._apply_known(response)
        self._apply_verification(response)
        self._bind_verification_goal()
        self.latest_report = self._format_state_report(
            before_goal,
            before_plan,
            before_known,
            before_context,
            before_verification,
        )

    def _actions(self, response: Json) -> list[Json]:
        return [action for action in (_json_dict(item) for item in _json_list(response.get("actions"))) if action]

    def _format_state_report(
        self,
        before_goal: str,
        before_plan: list[str],
        before_known: list[str],
        before_context: dict[str, ContextItem],
        before_verification: str,
    ) -> str:
        current = self.session.current
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
        if self.session.context_store != before_context:
            if not lines:
                lines.append("State Updated | " + self._verification_badge())
            lines.append("  Context " + f"({len(self.session.context_store)})")
            lines.extend(self._format_context_rows(before_context))
        verification = current.verification.format()
        if verification != before_verification:
            if not lines:
                lines.append("State Updated | " + self._verification_badge())
            lines.append("  Verify  " + self._format_verification())
        return "\n".join(lines)

    def _format_plan_rows(self) -> list[str]:
        items = self.session.current.plan
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
        items = self.session.current.known
        if not items:
            return ["    (empty)"]
        offset = max(0, len(items) - self.DISPLAY_LIMIT)
        rows = ["    ... " + str(offset) + " older"] if offset else []
        for index, item in enumerate(items[offset:], start=offset + 1):
            rows.append("    " + str(index) + ". " + self._compact(item))
        return rows

    def _format_context_rows(self, before_context: dict[str, ContextItem]) -> list[str]:
        changed = [key for key, value in self.session.context_store.items() if before_context.get(key) != value]
        if not changed:
            return ["    (empty)"]
        offset = max(0, len(changed) - self.DISPLAY_LIMIT)
        rows = ["    ... " + str(offset) + " older"] if offset else []
        for index, key in enumerate(changed[offset:], start=offset + 1):
            item = self.session.context_store[key]
            rows.append("    " + str(index) + ". " + self._compact(key) + " - " + self._compact(item.description))
        return rows

    def _format_verification(self) -> str:
        verification = self.session.current.verification
        parts = [verification.status]
        if verification.method:
            parts.append(self._compact(verification.method))
        if verification.context:
            parts.append("context: " + self._compact(verification.context))
        return " | ".join(parts)

    def _verification_badge(self) -> str:
        return "VERIFY:" + self.session.current.verification.status

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
                    changed = changed or update != self.session.current.goal
                    self.session.current.goal = update
                complete = action.get("complete")
                if isinstance(complete, bool):
                    self.session.current.goal_reached = complete
        return changed

    def _apply_plan(self, response: Json) -> bool:
        replaced = False
        for update in [action for action in self._actions(response) if _json_str(action.get("type")) == "plan"]:
            items = _json_list(update.get("items"))
            if update.get("mode") == "replace":
                self.session.current.plan = [item for item in (self._plan_item_from_json(raw) for raw in items) if item]
                replaced = True
                continue
            for raw in items:
                patch = _json_dict(raw)
                op = _json_str(patch.get("op")) or "add"
                item_id = _json_str(patch.get("id")) or ""
                if op == "remove":
                    self.session.current.plan = [item for item in self.session.current.plan if item.id != item_id]
                    continue
                plan_item = self._plan_item_from_json(patch)
                if plan_item is None:
                    continue
                existing = next((item for item in self.session.current.plan if item.id == plan_item.id and item.id), None)
                if existing:
                    existing.text = plan_item.text
                    existing.status = plan_item.status
                    existing.context = plan_item.context
                else:
                    self.session.current.plan.append(plan_item)
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
        item = _json_dict(value)
        if not item:
            return None
        fact = (_json_str(item.get("fact")) or "").strip()
        if not fact:
            return None
        self._store_context_from_known_item(item)
        return fact

    def _store_context_from_known_item(self, item: Json) -> None:
        for raw in _json_list(item.get("context")):
            context = _json_dict(raw)
            if not context:
                continue
            key = (_json_str(context.get("key")) or "").strip()
            description = _json_str(context.get("description"))
            value = _json_str(context.get("value"))
            self._store_context(key, description, value)

    def _store_context(self, key: str | None, description: str | None, value: str | None) -> None:
        key = (key or "").strip()
        description = (description or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        if not description:
            description = key
        if len(value) > self.MAX_CONTEXT_VALUE_CHARS:
            value = value[: self.MAX_CONTEXT_VALUE_CHARS] + "\n...<truncated>"
        if key in self.session.context_store:
            self.session.context_store.pop(key)
        elif len(self.session.context_store) >= self.MAX_CONTEXT_ITEMS:
            self.session.context_store.pop(next(iter(self.session.context_store)))
        self.session.context_store[key] = ContextItem(description=description, value=value)

    def _add_known_item(self, fact: str) -> None:
        if fact not in self.session.current.known:
            self.session.current.known.append(fact)

    def _apply_verification(self, response: Json) -> None:
        for data in [action for action in self._actions(response) if _json_str(action.get("type")) == "verify"]:
            method = _json_str(data.get("method"))
            if method is not None:
                if method != self.session.current.verification.method:
                    self.session.current.verification.context = ""
                self.session.current.verification.method = method
            status = _json_str(data.get("status"))
            if status == "pending":
                self.session.current.verification.status = VerificationStatus.REQUIRED
                if "context" not in data:
                    self.session.current.verification.context = ""
            elif status == "passed":
                self.session.current.verification.status = VerificationStatus.DONE
            elif status == "blocked":
                self.session.current.verification.status = VerificationStatus.BLOCKED
            context = _json_str(data.get("context"))
            if context is not None:
                self.session.current.verification.context = context

    def _reset_stale_verification(self, response: Json, *, goal_changed: bool, plan_replaced: bool) -> None:
        verification = self.session.current.verification
        if goal_changed:
            verification.reset()
            return
        if verification.goal and verification.goal != self.session.current.goal:
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
        verification = self.session.current.verification
        if not verification.has_context():
            verification.goal = ""
            return
        if self.session.current.goal:
            verification.goal = self.session.current.goal


############################
# ConversationCompactor
############################


@final
class ConversationCompactor:
    KEEP_RECENT: ClassVar[int] = 5

    def __init__(self, session: Session, model_client: ModelClient):
        self.session = session
        self.model_client = model_client

    def compact(self) -> int:
        count = len(self.session.conversation)
        if count <= self.KEEP_RECENT:
            return 0
        old_items = self.session.conversation[: -self.KEEP_RECENT]
        keep_items = self.session.conversation[-self.KEEP_RECENT :]
        summary = self._summarize(old_items)
        self.session.conversation = [AssistantMessage(content="Conversation compact summary:\n" + summary)] + keep_items
        return count

    def maybe_compact(self) -> bool:
        if self.session.compact_at <= 0:
            return False
        if len(self.session.conversation) <= self.session.compact_at:
            return False
        return self.compact() > 0

    def _summarize(self, items: list[ConversationItem]) -> str:
        user_prompt = COMPACT_USER_PROMPT_TEMPLATE.format(conversation="\n\n".join(item.format() for item in items)).strip()
        response = self.model_client.request(SUMMARIZER_AGENT_COMPACT_PROMPT.strip(), user_prompt, activity="compact")
        summary = _json_str(response.get("summary"))
        if not summary:
            raise LLMError("compact response missing summary")
        return summary


############################
# Agent
############################


@final
class Agent:
    MAX_CONSECUTIVE_FORMAT_ERRORS: ClassVar[int] = 3
    MAX_AGENT_FEEDBACK_ERRORS: ClassVar[int] = 8
    MAX_AGENT_FEEDBACK_ERROR_LEN: ClassVar[int] = 220
    MODEL_TIMEOUT_RETRY_DELAYS: ClassVar[tuple[int, ...]] = (3, 6, 10)

    def __init__(self, session: Session):
        self.session = session
        self.prompt_builder = PromptBuilder(session)
        self.model_client = ModelClient(session)
        self.tool_runner = ToolCallRunner(session)
        self.state_updater = AgentStateUpdater(session)
        self.compactor = ConversationCompactor(session, self.model_client)
        self.last_tool_calls = ""
        self.agent_feedback_errors: list[str] = []

    def build_system_prompt(self) -> str:
        return self.prompt_builder.system_prompt()

    def build_user_prompt(self) -> str:
        return self.prompt_builder.user_prompt(self.last_tool_calls, self._format_agent_feedback())

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
        self._clear_agent_feedback()
        self.session.current.goal_reached = False

    def run(
        self,
        user_input: str,
        *,
        confirm: ConfirmCallback | None = None,
        on_auto_approve: ToolDisplayCallback | None = None,
        on_message: MessageCallback | None = None,
    ) -> Json:
        self.last_tool_calls = ""
        self._clear_agent_feedback()
        self.session.current.user_input = user_input
        self.session.current.goal_reached = False
        self.maybe_auto_compact()
        self.session.append_conversation(UserMessage(content=user_input))
        consecutive_format_errors = 0

        try:
            for _ in range(self.session.max_agent_steps):
                response = self.step(on_action=self._stream_action_preview_callback(on_message) if on_message is not None else None, on_message=on_message)
                format_error = _json_str(response.get("_format_error"))
                if format_error:
                    consecutive_format_errors += 1
                    self._remember_agent_error(self._format_agent_feedback_format_error(format_error))
                    if consecutive_format_errors >= self.MAX_CONSECUTIVE_FORMAT_ERRORS:
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
                actions = self._response_actions(response)
                tool_calls = self._tool_calls_from_actions(actions)
                messages = self._messages_from_actions(actions)
                if self.session.debug and on_message is not None:
                    frame_error_report = self._format_frame_error_report(response)
                    if frame_error_report:
                        on_message(frame_error_report)
                self.apply_response(response)
                if on_message is not None and self.state_updater.latest_report:
                    on_message(self.state_updater.latest_report)
                for message in messages:
                    self.session.append_conversation(AssistantMessage(content=message))
                    if on_message is not None:
                        on_message(message)
                if tool_calls:
                    self.execute_tool_calls(tool_calls, confirm=confirm, on_auto_approve=on_auto_approve)
                    if on_message is not None:
                        report = self.tool_runner.format_latest_report()
                        if report:
                            on_message(report)
                    self.maybe_auto_compact()
                    continue
                if self.session.current.verification.status == VerificationStatus.REQUIRED:
                    self.session.current.goal_reached = False
                    self._remember_agent_error(self._format_agent_feedback_verification_error())
                    self._report_gate(
                        on_message,
                        "Retrying: verification is required before completion.",
                        "Verification_Gate: retrying until verification is passed or blocked.",
                    )
                    continue
                if messages and self.session.current.goal_reached:
                    self._clear_agent_feedback()
                    return response
                self.session.current.goal_reached = False
                if not actions:
                    self._remember_agent_error(self._format_agent_feedback_empty_actions_error())
                    self._report_gate(
                        on_message,
                        "Continuing: goal is not complete yet.",
                        "Continuation_Gate: goal not reached; retrying next useful action.",
                    )
                elif messages:
                    self._remember_agent_error(self._format_agent_feedback_message_before_complete_error())
                continue
        except KeyboardInterrupt:
            self.cancel_current_goal()
            raise
        raise LLMError("agent step limit reached")

    def _clear_agent_feedback(self) -> None:
        self.agent_feedback_errors = []

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

    def _format_agent_feedback_verification_error(self) -> str:
        return 'Error: goal is not complete until verification passes or is blocked. Rule: run a relevant tool, or return verify status="passed"|"blocked" with context.'

    def _format_agent_feedback_empty_actions_error(self) -> str:
        return "Error: returned no actions while the goal is incomplete. Rule: continue with a useful state, tool, verify, or final message action."

    def _format_agent_feedback_message_before_complete_error(self) -> str:
        return "Error: returned message before goal.complete=true. Rule: only finish with message after the goal is achieved and verified."

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
        response = self.request(self.build_system_prompt(), self.build_user_prompt(), activity="main", on_action=on_action, on_message=on_message)
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
        if _json_str(action.get("type")) != "tool":
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
        self.last_tool_calls = self.tool_runner.execute(tool_calls, confirm=confirm, on_auto_approve=on_auto_approve)
        return self.last_tool_calls

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

    def _messages_from_actions(self, actions: list[Json]) -> list[str]:
        return [message for message in (_json_str(action.get("text")) for action in actions if _json_str(action.get("type")) == "message") if message]


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
    CommandSpec("/context", "Show or clear hidden context store", "Info", "/context [clear]"),
    CommandSpec("/compact", "Compact conversation history", "Info", "/compact"),
    CommandSpec("/model", "Show or set the model", "Config", "/model [name]"),
    CommandSpec("/compact-at", "Show or set auto-compact threshold", "Config", "/compact-at [number]"),
    CommandSpec("/reason", "Show or toggle reasoning", "Config", "/reason [on|off|status]"),
    CommandSpec("/reason_effort", "Show or set reasoning effort", "Config", "/reason_effort [minimal|low|medium|high|xhigh]"),
    CommandSpec("/stream", "Show or toggle streaming responses", "Config", "/stream [on|off|status]"),
    CommandSpec("/yolo", "Show or toggle confirmation bypass", "Config", "/yolo [on|off|status]"),
    CommandSpec("/exit", "Exit nanocode", "Control", "/exit"),
    CommandSpec("/quit", "Exit nanocode", "Control", "/quit"),
)


@final
class CommandDispatcher:
    EFFORTS: ClassVar[set[str]] = {"minimal", "low", "medium", "high", "xhigh"}

    def __init__(
        self,
        agent: Agent,
        run_agent: MessageCallback | None = None,
        run_with_status: StatusRunner | None = None,
    ):
        self.agent = agent
        self.run_agent = run_agent
        self.run_with_status = run_with_status
        self.handlers: dict[str, Callable[[str], str]] = {
            "/help": self._help,
            "/status": self._status,
            "/context": self._context,
            "/compact": self._compact,
            "/model": self._model,
            "/compact-at": self._compact_at,
            "/reason": self._reason,
            "/reason_effort": self._reason_effort,
            "/stream": self._stream,
            "/yolo": self._yolo,
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

    def _context(self, args: str) -> str:
        if args == "clear":
            count = len(self.agent.session.context_store)
            self.agent.session.context_store.clear()
            return "Cleared context: " + str(count)
        if args:
            return "Usage: /context [clear]"
        if not self.agent.session.context_store:
            return "Context: 0"
        lines = ["Context: " + str(len(self.agent.session.context_store))]
        for key, item in self.agent.session.context_store.items():
            lines.append("  " + key + " - " + item.description)
        return "\n".join(lines)

    def _status(self, args: str) -> str:
        if args:
            return "Usage: /status"
        session = self.agent.session
        reasoning = session.reasoning_effort if session.reasoning else "off"
        stream = "on" if session.stream else "off"
        yolo = "on" if session.yolo else "off"
        return "\n".join(
            [
                "model: " + (session.model or "(empty)"),
                "reasoning: " + reasoning,
                "stream: " + stream,
                "yolo: " + yolo,
                "conversation: " + str(len(session.conversation)) + "/" + str(session.compact_at),
                "context: " + str(len(session.context_store)),
                "tokens: last=" + _format_count(session.last_total_tokens) + " session=" + _format_count(session.session_total_tokens),
                "cost(usd): last=" + _format_cost(session.last_cost_usd) + " session=" + _format_cost(session.session_cost_usd),
                "goal: " + (session.current.goal or "(empty)"),
                "verification: " + session.current.verification.status,
            ]
        )

    def _compact(self, args: str) -> str:
        if args:
            return "Usage: /compact"
        return self._with_status(self._compact_history)

    def _compact_history(self) -> str:
        count = self.agent.compact_history()
        if count == 0:
            return "Conversation history is empty"
        return "Compacted conversation history: " + str(count) + " item(s) -> " + str(len(self.agent.session.conversation)) + " item(s)"

    def _model(self, args: str) -> str:
        if not args:
            return "Current model: " + (self.agent.session.model or "(empty)")
        self.agent.session.model = args
        return "Model set to: " + args

    def _compact_at(self, args: str) -> str:
        if not args:
            return "Current auto-compact threshold: " + str(self.agent.session.compact_at)
        try:
            value = int(args)
        except ValueError:
            return "Usage: /compact-at [number]"
        if value <= 0:
            return "Usage: /compact-at [number] (must be positive)"
        self.agent.session.compact_at = value
        compacted = self._with_status(lambda: "yes" if self.agent.maybe_auto_compact() else "") == "yes"
        suffix = " and compacted history" if compacted else ""
        return "Auto-compact threshold set to: " + str(value) + suffix

    def _with_status(self, action: StatusAction) -> str:
        if self.run_with_status is None:
            return action()
        return self.run_with_status(action)

    def _reason(self, args: str) -> str:
        if args == "on":
            self.agent.session.reasoning = True
            return "Reasoning enabled"
        if args == "off":
            self.agent.session.reasoning = False
            return "Reasoning disabled"
        if args in {"", "status"}:
            return "Reasoning is " + ("on" if self.agent.session.reasoning else "off")
        return "Usage: /reason [on|off|status]"

    def _reason_effort(self, args: str) -> str:
        if not args:
            return "Current reasoning effort: " + self.agent.session.reasoning_effort
        if args not in self.EFFORTS:
            return "Usage: /reason_effort [minimal|low|medium|high|xhigh]"
        self.agent.session.reasoning_effort = args
        return "Reasoning effort set to: " + args

    def _stream(self, args: str) -> str:
        if args == "on":
            self.agent.session.stream = True
            return "Streaming enabled"
        if args == "off":
            self.agent.session.stream = False
            return "Streaming disabled"
        if args in {"", "status"}:
            return "Streaming is " + ("on" if self.agent.session.stream else "off")
        return "Usage: /stream [on|off|status]"

    def _yolo(self, args: str) -> str:
        if args == "on":
            self.agent.session.yolo = True
            return "YOLO enabled"
        if args == "off":
            self.agent.session.yolo = False
            return "YOLO disabled"
        if args in {"", "status"}:
            return "YOLO is " + ("on" if self.agent.session.yolo else "off")
        return "Usage: /yolo [on|off|status]"


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
        model = session.model.rsplit("/", 1)[-1] or session.model or "(no model)"
        reasoning = session.reasoning_effort if session.reasoning else "off"
        yolo = " | yolo" if session.yolo else ""
        context = str(len(session.conversation)) + "/" + str(session.compact_at)
        last_tokens = self._format_count(session.last_total_tokens)
        last_cost = _format_cost(session.last_cost_usd)
        if last_cost != "-":
            last_tokens += "/" + last_cost
        session_tokens = self._format_count(session.session_total_tokens)
        session_cost = _format_cost(session.session_cost_usd)
        if session_cost != "-":
            session_tokens += "/" + session_cost
        tokens = "last:" + last_tokens + " session:" + session_tokens
        parts = [model + " (" + reasoning + ")" + yolo, "ctx:" + context, "context:" + str(len(session.context_store)), "tok:" + tokens]
        if show_elapsed:
            parts.append(f"{turn_elapsed:.1f}s")
        if session.current_model_call_started_at > 0:
            parts.append("calling:" + f"{max(0.0, now - session.current_model_call_started_at):.1f}s")
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

    def _command_completer(self) -> WordCompleter:
        return WordCompleter([spec.name for spec in COMMANDS], ignore_case=False, WORD=True)

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
        if message.startswith("State Updated"):
            self._emit_segments(self._state_segments(message), message)
            return
        if message.startswith("Tool Calls"):
            self._emit_segments(self._tool_segments(message), message)
            return
        if message.startswith("Queued:"):
            self._emit_segments(self._queued_segments(message), message)
            return
        if message.startswith("Error:"):
            self._emit_segments([("bold ansired", message + "\n")], message)
            return
        if message.startswith("Cancelled"):
            self._emit_segments([("ansiyellow", message + "\n")], message)
            return
        self._emit_segments([("ansicyan", message + "\n")], message)

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
            elif line.startswith("  ") and ". [" in line:
                style = "ansigreen" if "[success]" in line else "ansired"
                segments.extend([("ansibrightblack", line[:5]), (style, line[5:] + "\n")])
            elif line.startswith("     why:"):
                segments.extend([("ansibrightblack", "     why: "), ("ansimagenta", line[10:] + "\n")])
            elif line.startswith("     log:"):
                segments.extend([("ansibrightblack", "     log: "), ("ansiblue", line[10:] + "\n")])
            else:
                segments.extend([("ansibrightblack", line + "\n")])
        return segments

    def _queued_segments(self, message: str) -> list[tuple[str, str]]:
        body = message[len("Queued:") :].strip()
        target, separator, reason = body.partition(" - ")
        segments: list[tuple[str, str]] = [("ansibrightblack", "Queued: "), ("ansicyan", target)]
        if separator:
            segments.extend([("ansibrightblack", " - "), ("ansimagenta", reason)])
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


class ReferenceFileCompleter(Completer):
    def __init__(self, cwd: str, command_completer: WordCompleter):
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
        args = parser.parse_args(argv)
        session = Session(yolo=args.yolo, debug=args.debug)
        missing = session.missing_required_envs()
        if missing:
            print("Missing env: " + ", ".join(missing), file=sys.stderr)
            return 2
        return AgentLoop(Agent(session)).run()
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
