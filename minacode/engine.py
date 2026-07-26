"""minacode engine: context management, model client, and the agent loop."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from enum import auto
from typing import Any, ClassVar, Generic, TypeVar
from urllib.request import Request, urlopen

import anthropic
import openai
from anthropic import Anthropic
from json_repair import repair_json
from openai import OpenAI
from prompt_toolkit.utils import get_cwidth

from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    HTTP_USER_AGENT,
    MAX_TOOL_OUTPUT_TOKENS,
    MIN_CONTEXT_SAFETY_TOKENS,
    MODEL_REQUEST_RETRIES,
    PROVIDER_ECHO_KEYS,
    RESPONSES_OUTPUT_KEY,
    Json,
    MinacodeError,
    ModelError,
    ModelRequestRetry,
    ProviderConfig,
    Text,
    ToolArgs,
    ToolCall,
    ToolError,
    UpdateStatus,
    __version__,
)
from minacode.provider_compat import (
    CHAT_REASONING_EFFORT_VALUES,
    ResolvedProvider,
    anthropic_thinking_always_on,
    anthropic_thinking_params,
)
from minacode.image import IMAGE_REFS_KEY, ImageInputs, UserInput
from minacode.session import AgentState, HistorySegment, QueuedInput, Session, TurnDiff
from minacode.tools import (
    TOOL_REGISTRY,
    AskSpec,
    AskTool,
    BashTool,
    CodeIndex,
    Edit,
    EditTool,
    JobTool,
    ReadTool,
    Tool,
)

_IdentityT = TypeVar("_IdentityT", bound=Hashable)
_ResourceT = TypeVar("_ResourceT")
_ResultT = TypeVar("_ResultT")


class UpdateChecker:
    PYPI_URL = "https://pypi.org/pypi/minacode/json"
    CACHE_FILE = "update.json"
    TIMEOUT = 5
    INTERVAL_SECONDS = 24 * 3600

    def __init__(self, session: Session):
        self.session = session
        self.cache_path = session.data_path(self.CACHE_FILE)

    def start(self) -> None:
        cached_at, cached_latest = self._load()
        self.session.update.latest = cached_latest
        if self.session.update.checking or time.time() - cached_at < self.INTERVAL_SECONDS:
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
            self.session.update.checking = False
            self._save()

    def _load(self) -> tuple[float, str]:
        with contextlib.suppress(Exception):
            with open(self.cache_path, encoding="utf-8") as file:
                data = json.load(file)
            latest = str(data.get("latest") or "")
            if UpdateStatus.version_tuple(latest):
                return float(data.get("checked_at") or 0), latest
        return 0.0, ""

    def _save(self) -> None:
        with contextlib.suppress(Exception):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as file:
                json.dump({"checked_at": time.time(), "latest": self.session.update.latest}, file)

    @staticmethod
    def fetch_latest() -> str:
        request = Request(UpdateChecker.PYPI_URL, headers={"Accept": "application/json", "User-Agent": HTTP_USER_AGENT})
        with urlopen(request, timeout=UpdateChecker.TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        version = data.get("info", {}).get("version") if isinstance(data, dict) else ""
        if not isinstance(version, str) or not UpdateStatus.version_tuple(version):
            raise MinacodeError("invalid PyPI version response")
        return version

    def status_line(self) -> str:
        update = self.session.update
        if update.checking:
            return "update: checking"
        if update.newer_than(__version__):
            return f"update: {__version__} -> {update.latest}"
        if update.error:
            return "update: error"
        return "update: current" if update.latest else "update: unknown"

    @staticmethod
    def upgrade_command() -> list[str]:
        """Best-effort package-manager command to upgrade minacode, based on how it was installed."""
        executable = os.path.realpath(sys.executable).replace(os.sep, "/")
        if "/uv/tools/" in executable:
            return ["uv", "tool", "upgrade", "minacode"]
        if "/pipx/venvs/" in executable:
            return ["pipx", "upgrade", "minacode"]
        return [sys.executable, "-m", "pip", "install", "--upgrade", "minacode"]


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


class ContextManager:
    COMPACT_TITLE: ClassVar[str] = "--- Prior Conversation Summary (compacted) ---"
    COMPACT_RECENT_MESSAGES: ClassVar[int] = 8
    MCP_DESCRIBE_BLOCK: ClassVar[re.Pattern] = re.compile(r"<MCPDescribe server=(\".*?\") tool=(\".*?\")>.*?</MCPDescribe>", re.DOTALL)
    SKILL_BLOCK: ClassVar[re.Pattern] = re.compile(r"<Skill name=(\".*?\")>.*?</Skill>", re.DOTALL)

    def __init__(self, session: Session):
        self.session = session

    def model_messages(self, base_system: str, turn_messages: list[Json] | None = None) -> list[Json]:
        messages: list[Json] = [
            {"role": "system", "content": base_system.strip()},
            {"role": "user", "content": "--- Environment ---\n" + (self.environment() or "(empty)")},
        ]
        for context in (self.skills_context(), self.mcp_tools_context()):
            if context:
                messages.append({"role": "user", "content": context})
        if history_index := self.history_index_context():
            messages.append({"role": "user", "content": "--- History index ---\n" + history_index})
        conversation = [
            *self.session.messages,
            {"role": "user", "content": "--- Memory ---\n" + (self.memory_context(with_date=True) or "(empty)")},
            *(turn_messages or []),
        ]
        messages.extend(self.dedup_skill_loads(self.dedup_mcp_describes(conversation)))
        return Text.value(messages)

    def dedup_mcp_describes(self, messages: list[Json]) -> list[Json]:
        """Point repeats at the first full description, promoting the next after compaction."""
        return self._dedup_tool_blocks(
            messages,
            self.MCP_DESCRIBE_BLOCK,
            lambda match: (str(json.loads(match.group(1))), str(json.loads(match.group(2)))),
            lambda identity, key: f"(repeat describe of {identity[0]}.{identity[1]}; schema shown earlier at {key}, unchanged)",
        )

    def dedup_skill_loads(self, messages: list[Json]) -> list[Json]:
        return self._dedup_tool_blocks(
            messages,
            self.SKILL_BLOCK,
            lambda match: str(json.loads(match.group(1))),
            lambda name, key: f"(repeat load of skill {name}; instructions shown earlier at {key}, unchanged)",
        )

    @staticmethod
    def _dedup_tool_blocks(
        messages: list[Json],
        block: re.Pattern,
        identity_from: Callable[[re.Match[str]], _IdentityT],
        marker_for: Callable[[_IdentityT, str], str],
    ) -> list[Json]:
        seen: dict[_IdentityT, str] = {}
        result: list[Json] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                result.append(message)
                continue
            match = block.search(content)
            if match is None:
                result.append(message)
                continue
            try:
                identity = identity_from(match)
            except (json.JSONDecodeError, ValueError):
                result.append(message)
                continue
            first_key = seen.get(identity)
            if first_key is None:
                key = re.search(r"\btr\.\d+\b", content)
                seen[identity] = key.group(0) if key else "above"
                result.append(message)
                continue
            marker = marker_for(identity, first_key)
            result.append({**message, "content": block.sub(lambda _: marker, content)})
        return result

    def mcp_tools_context(self) -> str:
        return self.session.mcp.render_tools_index() if self.session.mcp else ""

    def skills_context(self) -> str:
        return self.session.skills.index() if self.session.skills else ""

    def request_token_budget(self) -> int:
        limit = self.session.settings.max_context_tokens
        safety = max(MIN_CONTEXT_SAFETY_TOKENS, (limit + 49) // 50)
        return max(1, limit - self.session.config.provider.output_token_budget() - safety)

    def request_tokens(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        return self.estimated_tokens(messages) + (self.estimated_tokens(tools) if tools else 0)

    def update_percent(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        self.session.state.context_percent = min(100, self.request_tokens(messages, tools) * 100 // self.request_token_budget())
        return self.session.state.context_percent

    def update_current_tokens(self, base_system: str) -> int:
        messages = self.model_messages(base_system, self.session._active_turn_messages)
        tools = Tool.resolved_schemas(self.session)
        tokens = self.request_tokens(messages, tools)
        self.session.state.context_percent = min(100, tokens * 100 // self.request_token_budget())
        return tokens

    def prepare_messages(self, model: "ModelClient", base_system: str, turn_messages: list[Json] | None = None, tools: list[Json] | None = None) -> list[Json]:
        messages = self.model_messages(base_system, turn_messages)
        budget = self.request_token_budget()
        if self.request_tokens(messages, tools) < budget:
            return messages
        compacted, keep = self.compaction_parts()
        if self._compact_messages(model, compacted, keep, "Previous context was deterministically trimmed.", tool_messages=turn_messages):
            messages = self.model_messages(base_system, turn_messages)
        if turn_messages is not None and self.request_tokens(messages, tools) >= budget:
            compacted, keep = self.turn_compaction_parts(turn_messages)
            if self._compact_messages(model, compacted, keep, "Current turn context was deterministically trimmed.", turn_messages=turn_messages):
                messages = self.model_messages(base_system, turn_messages)
        return messages

    def _compact_messages(
        self,
        model: "ModelClient",
        compacted: list[Json],
        keep: list[Json],
        fallback_note: str,
        *,
        tool_messages: list[Json] | None = None,
        turn_messages: list[Json] | None = None,
    ) -> bool:
        if not compacted:
            return False
        try:
            data = model.compact(self.compaction_input(compacted))
        except Exception:
            data = None
        self.apply_compaction(data, keep, tool_messages, turn_messages=turn_messages, fallback_note=fallback_note if data is None else "", compacted=compacted)
        return True

    def memory_context(self, *, with_date: bool = False) -> str:
        index_status = self.session.state.code_index_status or "missing"
        index_usable = "yes" if index_status in {"synced", "ready", "stale"} else "no"
        rows = [
            "Goal: " + (self.session.state.goal or "(empty; use Note for multi-step work)"),
            "Plan:\n" + "\n".join(AgentState.plan_rows_for(self.session.state.plan, status=True)),
            "Known:\n" + "\n".join("- " + item for item in self.session.state.known or ["(empty)"]),
            "Check: " + (self.session.state.check or "(empty)"),
            f"Code index: {index_status} (InspectCode usable: {index_usable})",
        ]
        if errors := self.recent_tool_errors():
            rows.append("Recent tool errors:\n" + "\n".join(errors))
        if with_date:
            rows.append("Date: " + datetime.now().astimezone().strftime("%Y-%m-%d"))
        return "\n\n".join(rows)

    def history_index_context(self) -> str:
        index = "\n".join(f"- {seg.key}: {seg.title}" for seg in self.session.history)
        return self.bound_output(index, stable_marker=True)

    def recent_tool_errors(self) -> list[str]:
        return [
            f"- {' '.join(part for part in (record.key, record.name, ' '.join(Tool.compact(arg, 80) for arg in record.args)) if part)}: {Tool.compact(record.error, 160)}"
            for record in self.session.tool_errors[-5:]
        ]

    def environment(self) -> str:
        info = self.session.system_info
        assert info is not None
        rows = [
            f"- cwd: {info.cwd}",
            # Tell the model which executables it may drive through Bash.
            "- detected_commands (available via Bash): " + (", ".join(info.commands) or "(none)"),
            f"- os: {info.os}",
            f"- arch: {info.arch}",
            f"- shell_timeout: {self.session.settings.shell_timeout}s",
        ]
        return "\n".join(rows)

    @staticmethod
    def md_table(headers: list[str], rows: list[tuple]) -> str:
        def cell(value: object) -> str:
            return Text.clean(str(value)).replace("\n", " ").replace("|", "\\|")

        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
            ]
        )

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
        """Split history for manual compaction and the first automatic pass."""
        messages = self.session.messages
        index = self.latest_user_index(messages)
        if index is None:
            return self.without_compaction_summaries(messages), []
        compacted_tail, keep_tail = self.compaction_parts_for(messages[index + 1 :])
        compacted = self.without_compaction_summaries(messages[:index] + compacted_tail)
        keep = self.without_compaction_summaries([messages[index]] + keep_tail)
        return compacted, keep

    def turn_compaction_parts(self, messages: list[Json]) -> tuple[list[Json], list[Json]]:
        index = self.latest_user_index(messages)
        if index is None:
            compacted, keep = self.compaction_parts_for(messages)
            return self.without_compaction_summaries(compacted), self.without_compaction_summaries(keep)
        compacted, keep = self.compaction_parts_for(messages[index + 1 :])
        return self.without_compaction_summaries(compacted), self.without_compaction_summaries(messages[: index + 1] + keep)

    def without_compaction_summaries(self, messages: list[Json]) -> list[Json]:
        return [message for message in messages if not self.is_compaction_summary(message)]

    def compaction_parts_for(self, messages: list[Json]) -> tuple[list[Json], list[Json]]:
        cut = max(0, len(messages) - self.COMPACT_RECENT_MESSAGES)
        if cut < len(messages) and messages[cut].get("role") == "tool":
            while cut > 0 and messages[cut - 1].get("role") == "tool":
                cut -= 1
            if cut > 0 and messages[cut - 1].get("role") == "assistant" and messages[cut - 1].get("tool_calls"):
                cut -= 1
        return messages[:cut], messages[cut:]

    def messages_text(self, messages: list[Json]) -> str:
        return "\n\n".join(f"{message.get('role', 'message')}:\n{ImageInputs.label_text(message)}" for message in messages) or "(empty)"

    def history_title(self, messages: list[Json]) -> str:
        for message in messages:
            if message.get("role") == "user" and not str(message.get("content") or "").startswith(self.COMPACT_TITLE):
                return Tool.compact(str(message.get("content") or ""), 80)
        return Tool.compact(self.messages_text(messages[:1]), 80) or "compacted context"

    def store_history_segment(self, compacted: list[Json]) -> None:
        key = f"seg.{len(self.session.history) + 1}"
        text = self.bound_output(self.messages_text(compacted))
        self.session.history.append(HistorySegment(key=key, title=self.history_title(compacted), text=text))

    def _summary_block(self, summary: str) -> list[Json]:
        """The single compaction-summary user message, or [] when there is no summary yet."""
        return [{"role": "user", "content": self.COMPACT_TITLE + "\n" + summary}] if summary else []

    def apply_compaction(
        self,
        data: Json | None,
        keep: list[Json],
        tool_messages: list[Json] | None = None,
        *,
        turn_messages: list[Json] | None = None,
        fallback_note: str = "",
        compacted: list[Json] | None = None,
    ) -> None:
        self.session.state.compaction_count += 1
        if compacted:
            self.store_history_segment(compacted)
        if data is not None:
            self.session.state.apply(data)
        if fallback_note:
            self.session.state.summary = (self.session.state.summary + "\n" + fallback_note).strip()
        summary = self.session.state.summary
        summary_block = self._summary_block(summary)
        if turn_messages is None:
            self.session.messages = summary_block + keep
            prune_context = (self.session.messages if data is not None else [*keep]) + (tool_messages or [])
        else:
            index = self.latest_user_index(keep)
            insert = len(keep) if index is None else index + 1
            turn_messages[:] = keep[:insert] + summary_block + keep[insert:]
            prune_context = [*self.session.messages, *turn_messages]
        self.prune_tool_records(prune_context)

    def prune_tool_records(self, keep_messages: list[Json]) -> None:
        records = self.session.tool_records
        keep = set(re.findall(r"\btr\.\d+\b", self.messages_text(keep_messages)))
        self.session.tool_records = [record for record in records if record.key in keep][-400:]
        self.session.tool_results = {record.key: record.output for record in self.session.tool_records}

    def latest_user_index(self, messages: list[Json]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user" and not self.is_compaction_summary(messages[index]):
                return index
        return None

    def is_compaction_summary(self, message: Json) -> bool:
        return message.get("role") == "user" and str(message.get("content") or "").startswith(self.COMPACT_TITLE)

    def bound_output(self, text: str, key: str = "", *, stable_marker: bool = False) -> str:
        estimated = self.estimated_text_tokens(text)
        if estimated <= MAX_TOOL_OUTPUT_TOKENS:
            return text
        limit = MAX_TOOL_OUTPUT_TOKENS * 4
        head_limit = max(1, limit * 2 // 5)
        tail_limit = max(1, limit - head_limit)
        head = self.head_excerpt(text, head_limit)
        tail = self.tail_excerpt(text, tail_limit)
        omitted_tokens = max(0, estimated - self.estimated_text_tokens(head) - self.estimated_text_tokens(tail))
        note = f'<bounded_output omitted="middle" max_tokens="{MAX_TOOL_OUTPUT_TOKENS}"'
        if not stable_marker:
            note += f' estimated_tokens="{estimated}" omitted_tokens="{omitted_tokens}"'
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

    def estimated_tokens(self, messages: list[Json]) -> int:
        # Normalized assistant fields already contain visible text and tool calls, so provider
        # echoes would double-count them. Preserve only additional readable reasoning; ciphertext
        # and signatures are transport state whose byte length is not a prompt-token estimate.
        def readable_provider_context(message: Json) -> list[str]:
            readable: list[str] = []
            responses = message.get(RESPONSES_OUTPUT_KEY)
            if isinstance(responses, list):
                for item in responses:
                    if not isinstance(item, dict) or item.get("type") != "reasoning":
                        continue
                    readable.extend(str(item[key]) for key in ("content", "summary") if item.get(key))
            anthropic = message.get(ANTHROPIC_CONTENT_KEY)
            if isinstance(anthropic, list):
                for block in anthropic:
                    if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking") and block.get("thinking"):
                        readable.append(str(block["thinking"]))
            return readable

        payload: list[Json] = []
        for message in messages:
            estimated = {key: value for key, value in message.items() if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY)}
            if readable := readable_provider_context(message):
                estimated["_provider_context"] = readable
            payload.append(estimated)
        chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        images = ImageInputs.estimated_tokens(messages) if self.session.images.support() is not False else 0
        return (chars + 3) // 4 + images

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
    class ApplyResult:
        lines: list["EditBatchPlan.Line"]
        changes: list[tuple[int, int, int, int]]
        replacements: list[tuple[int, int, list[str]]]
        replace_all: bool = False

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
            if os.path.isdir(self.path):
                raise ToolError("planned edit is stale; path is a directory")
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as file:
                    current = file.read()
            elif self.created and not self.before:
                current = ""
            else:
                raise ToolError("planned edit is stale; file changed")
            if current != self.before:
                raise ToolError("planned edit is stale; file changed")
            if self.created:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as file:
                file.write(self.after)
            tool.last_path = tool.session.relpath(self.path)
            tool.last_diff = tool.diff(self.path, self.before, self.after)
            tool.last_before = self.before
            tool.last_after = self.after
            return "\n".join(
                [
                    f"<Edit path={json.dumps(tool.last_path)}>",
                    tool.file_stat(self.path),
                    tool.last_diff.rstrip(),
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
        state = self.file_state(tool, path, edits[0].op == "create")
        before, created = state.text(), not state.exists
        before_lines = [line.text for line in state.lines]
        result = self.apply(tool, state, edits)
        after = "".join(line.text for line in result.lines)
        if after == before and not created:
            raise ToolError(EditTool.no_changes_error_from_lines(before_lines, result.replacements, result.replace_all))
        self.planned[call.id] = self.PlannedEdit(path, before, after, created, result.changes)
        state.lines, state.exists = result.lines, True

    def file_state(self, tool: EditTool, path: str, creating: bool) -> FileState:
        if path in self.files:
            state = self.files[path]
            if not state.exists and not creating:
                raise ToolError("file does not exist; use op=create to create it")
            if state.exists and creating:
                raise ToolError("file already exists")
            return state
        if tool._validate_target(path, creating):
            with open(path, encoding="utf-8") as file:
                original = file.readlines()
            state = self.FileState(path, [self.Line(line, index) for index, line in enumerate(original)], original, True)
        else:
            state = self.FileState(path, [], [], False)
        self.files[path] = state
        return state

    def apply(self, tool: EditTool, state: FileState, edits: list[Edit]) -> ApplyResult:
        result = tool.apply(state.text(), edits, lambda anchor: self.resolve_anchor(state, anchor))
        if edits[0].op == "create" or result.replace_all:
            return self.ApplyResult(self.new_lines(ReadTool.split_lines(result.content)), result.changes, result.replacements, result.replace_all)
        lines = list(state.lines)
        for start, end, replacement in sorted(result.replacements, reverse=True):
            lines[start:end] = self.new_lines(replacement)
        return self.ApplyResult(lines, result.changes, result.replacements)

    @staticmethod
    def new_lines(lines: list[str]) -> list[Line]:
        return [EditBatchPlan.Line(line, None) for line in lines]

    def resolve_anchor(self, state: FileState, anchor: str) -> int:
        index, expected = ReadTool.require_anchor(anchor)
        if index < len(state.lines) and ReadTool.anchor_matches(state.lines[index].text, expected):
            return index
        if index < len(state.original) and ReadTool.anchor_matches(state.original[index], expected):
            current = state.current_origin(index)
            if current is not None:
                return current
            raise ToolError(f"stale anchor {anchor}; original line was changed in this batch")
        current_line = ReadTool.anchor_line(index, state.lines[index].text) if index < len(state.lines) else "out of range"
        raise ToolError(f"stale anchor {anchor}; current is {current_line}")


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


@dataclass
class ToolDisplay:
    """How one tool call renders: the batch-counter suffix, the short call line, whether it prints
    as a nested tree, and whether it was auto/user approved. Threaded from run_one into finish/reject."""

    batch_suffix: str = ""
    display: str | None = None
    nested_display: bool = False
    approved: bool = False
    auto: bool = False


class ToolRunner:
    BASH_TRANSCRIPT_PREVIEW_LINES: ClassVar[int] = 3
    BASH_PREVIEW_LINES: ClassVar[int] = 24
    BASH_PREVIEW_LINE_LIMIT: ClassVar[int] = 220

    def __init__(self, session: Session, context: ContextManager, input_fn=input, output_fn=print):
        self.session = session
        self.context = context
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.live_output: Callable[[str, str], None] | None = None
        self.live_start: Callable[[], None] | None = None
        self.question_fn: Callable[[AskSpec, str], str] | None = None
        self._active_bash: ActiveResource[BashTool] = ActiveResource()

    def cancel(self) -> None:
        self._active_bash.apply(lambda tool: tool.cancel())

    def call_tool(self, tool: Tool, planned_edit: EditBatchPlan.PlannedEdit | None = None) -> str:
        if not isinstance(tool, BashTool):
            return planned_edit.call(tool) if planned_edit and isinstance(tool, EditTool) else tool.call()
        with self._active_bash.track(tool):
            return tool.call()

    def run(self, calls: list[ToolCall], batch_suffix: str = "") -> list[Json]:
        messages: list[Json] = []
        # Shared, mutated across segments: `first` controls which display carries batch_suffix;
        # `refused` short-circuits the rest of the batch once a confirmation is declined.
        state = {"first": True, "refused": False}
        index = 0
        while index < len(calls):
            if state["refused"]:
                messages.append(self.skip_message(calls[index]))
                index += 1
                continue
            end = self.parallel_segment_end(calls, index)
            if end - index >= 2 and self.session.settings.max_parallel_tools > 1:
                messages.extend(self.run_parallel(calls[index:end], batch_suffix, state))
                index = end
                continue
            end = index + 1 if self.edit_barrier(calls[index]) else self.edit_segment_end(calls, index)
            messages.extend(self.run_serial(calls[index:end], batch_suffix, state))
            index = end
        return messages

    def skip_message(self, call: ToolCall) -> Json:
        content = self.tool_message(call, "", "Skipped: previous tool call was refused", failed=True)
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def run_serial(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool]) -> list[Json]:
        messages: list[Json] = []
        plan = EditBatchPlan(self.session).build(segment) if any(call.name == "Edit" for call in segment) else EditBatchPlan(self.session)
        for call in segment:
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            status, content = self.run_one(call, batch_suffix=suffix, planned_edit=plan.planned.get(call.id), plan_error=plan.errors.get(call.id, ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if status == "refused":
                state["refused"] = True
        return messages

    def run_parallel(self, segment: list[ToolCall], batch_suffix: str, state: dict[str, bool]) -> list[Json]:
        # Run the pure tool.call() work concurrently, but apply all side effects (display, session
        # bookkeeping, tool messages) on this thread in request order, so output and the results
        # handed back to the model match the order the model issued the calls.
        cap = max(1, self.session.settings.max_parallel_tools)
        outcomes: list[tuple[str, str, str | None, float] | None] = [None] * len(segment)
        with ThreadPoolExecutor(max_workers=min(len(segment), cap), thread_name_prefix="tool") as executor:
            futures = {executor.submit(self.execute_readonly, call): position for position, call in enumerate(segment)}
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        messages: list[Json] = []
        for call, outcome in zip(segment, outcomes):
            suffix = batch_suffix if state["first"] else ""
            state["first"] = False
            assert outcome is not None
            content = self.finalize_outcome(call, outcome, batch_suffix=suffix)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        return messages

    def parallel_segment_end(self, calls: list[ToolCall], start: int) -> int:
        end = start
        while end < len(calls) and self.parallel_safe(calls[end]):
            end += 1
        return end

    def parallel_safe(self, call: ToolCall) -> bool:
        # A call may run concurrently only if it neither mutates state nor blocks on interactive
        # input: read-only, auto-approved, non-interactive tools (Read/Search/Recall/InspectCode,
        # read-only MCP). Edit is coordinated serially by EditBatchPlan;
        # Bash streams live output and mutates; Ask blocks on the user.
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None or call.name == "Edit" or tool_class in (BashTool, JobTool, AskTool):
            return False
        try:
            return not tool_class(self.session, call.args).needs_confirmation()
        except Exception:
            return False

    def execute_readonly(self, call: ToolCall) -> tuple[str, str, str | None, float]:
        # Pure execution for a parallel worker: returns (kind, output, display, elapsed) and performs
        # no display or session writes (those happen in finalize_outcome on the main thread). Mirrors
        # run_one's branches, minus confirmation (parallel_safe guarantees none is needed).
        started = time.monotonic()
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "reject", f"ToolError: unknown tool {call.name}", None, 0.0
        tool = tool_class(self.session, call.args)
        display = None
        try:
            display = self.short_call(call, tool.short_args())
            if call.error:
                raise ToolError(call.error)
            output = tool.call()
        except ToolError as error:
            return "reject", f"ToolError: {error}", display, time.monotonic() - started
        except Exception as error:
            return "error", f"ToolError: {error}", display, time.monotonic() - started
        return "ok", output, display, time.monotonic() - started

    def finalize_outcome(self, call: ToolCall, outcome: tuple[str, str, str | None, float], batch_suffix: str = "") -> str:
        kind, output, display, elapsed = outcome
        d = ToolDisplay(batch_suffix=batch_suffix, display=display)
        if kind == "ok":
            return self.finish(call, output, elapsed=elapsed, d=d)
        if kind == "reject":
            return self.reject(call, output, d=d)
        return self.finish(call, output, failed=True, elapsed=elapsed, d=d)

    def edit_segment_end(self, calls: list[ToolCall], start: int) -> int:
        end = start
        while end < len(calls) and not self.edit_barrier(calls[end]):
            end += 1
        return end

    def edit_barrier(self, call: ToolCall) -> bool:
        tool_class = TOOL_REGISTRY.get(call.name)
        return call.name != "Edit" and (tool_class is None or tool_class.MUTATES)

    def run_one(
        self,
        call: ToolCall,
        batch_suffix: str = "",
        planned_edit: EditBatchPlan.PlannedEdit | None = None,
        plan_error: str = "",
    ) -> tuple[str, str]:
        tool_class = TOOL_REGISTRY.get(call.name)
        if tool_class is None:
            return "failed", self.reject(call, f"ToolError: unknown tool {call.name}", d=ToolDisplay(batch_suffix=batch_suffix))
        if call.error:
            return "failed", self.reject(call, f"ToolError: {call.error}", d=ToolDisplay(batch_suffix=batch_suffix))
        tool = tool_class(self.session, call.args)
        if isinstance(tool, BashTool):
            tool.live_output = self.live_output
        started = time.monotonic()
        d = ToolDisplay(batch_suffix=batch_suffix)
        if isinstance(tool, AskTool):
            tool.question_fn = self.question_fn
        try:
            d.display = self.short_call(call, tool.short_args())
            if plan_error:
                raise ToolError(plan_error)
            needs_confirmation = tool.needs_confirmation()
            if needs_confirmation and self.session.settings.yolo:
                d.auto = True
                pre = self.approval_display(call, tool, "auto", batch_suffix=batch_suffix, planned_edit=planned_edit)
                # The "auto …" header duplicates the result line; only surface it when it carries a
                # preview the result line won't repeat (e.g. an Edit diff). The auto-approval itself
                # is recorded by the [auto] tag on the result line below.
                if pre.has_children:
                    self.output_fn(pre)
                    d.nested_display = True
            elif needs_confirmation:
                d.nested_display = True
                confirmed, reason = self.confirm(call, tool, batch_suffix=batch_suffix, planned_edit=planned_edit)
                if not confirmed:
                    output = "Cancelled: user refused tool call" + ((": " + reason) if reason else "")
                    return "refused", self.finish(call, output, failed=True, elapsed=time.monotonic() - started, d=d)
                d.approved = True
            if isinstance(tool, BashTool) and self.live_start is not None:
                if not d.nested_display:
                    self.output_fn(LogBlock.hierarchy(self.log_root(d.display or self.short_call(call), batch_suffix=batch_suffix, call=call), []))
                    d.nested_display = True
                self.live_start()
            output = self.call_tool(tool, planned_edit)
        except ToolError as error:
            return "failed", self.reject(call, f"ToolError: {error}", d=d)
        except Exception as error:
            return "failed", self.finish(call, f"ToolError: {error}", failed=True, elapsed=time.monotonic() - started, d=d)
        return "ok", self.finish(call, output, elapsed=time.monotonic() - started, turn_diff=tool.turn_diff(), d=d)

    def reject(
        self,
        call: ToolCall,
        output: str,
        *,
        d: "ToolDisplay | None" = None,
    ) -> str:
        d = d or ToolDisplay()
        self.session.record_tool_error("-", call.name, call.args, output)
        self.output_fn(
            LogBlock.hierarchy(None, [LogLine("error", self.oneline(output.removeprefix("ToolError:").strip(), 220), LogRole.ERROR, LogEdge.END)])
            if d.nested_display
            else self.reject_display(call, output, d=d)
        )
        return self.tool_message(call, "", output, failed=True, display=d.display)

    def reject_display(self, call: ToolCall, output: str, *, d: "ToolDisplay") -> LogBlock:
        # Argument/usage rejections are usually self-corrected on retry, so show a quiet one-liner
        # (rendered dim by UiPrinter) instead of the full red failed block. The model still receives
        # the complete error so it can correct the call.
        reason = self.oneline(output.removeprefix("ToolError:").strip(), 60)
        return LogBlock.hierarchy(self.log_root((d.display or self.short_call(call)) + " · rejected: " + reason, LogRole.MUTED, d.batch_suffix, call), [])

    def finish(
        self,
        call: ToolCall,
        output: str,
        *,
        failed: bool = False,
        elapsed: float | None = None,
        store: bool = True,
        turn_diff: "TurnDiff | None" = None,
        d: "ToolDisplay | None" = None,
    ) -> str:
        d = d or ToolDisplay()
        tool_class = TOOL_REGISTRY.get(call.name)
        key = self.session.store_tool_result(call.name, call.args, output) if not failed and store and (tool_class is None or tool_class.STORES_RESULT) else ""
        if failed:
            self.session.record_tool_error(key or "-", call.name, call.args, output)
        elif key:
            self.update_code_index(call, output)
            if turn_diff and turn_diff.path and turn_diff.diff:
                self.session.store_turn_diff(
                    key,
                    self.session.state.turn_step,
                    turn_diff.path,
                    turn_diff.diff,
                    before=turn_diff.before,
                    after=turn_diff.after,
                    round=self.session.state.round_count,
                )
        self.output_fn(self.finish_display(call, key, output, failed=failed, elapsed=elapsed, d=d))
        return self.tool_message(call, key, output, failed=failed, display=d.display)

    def tool_message(self, call: ToolCall, key: str, output: str, *, failed: bool = False, display: str | None = None) -> str:
        head = "tool " + ((key + " ") if key else ("- " if failed else "")) + (display or self.short_call(call))
        rows = [head]
        if failed:
            rows.append("status: failed")
        rows.extend(["output:", self.context.bound_output(output, key).rstrip()])
        return "\n".join(rows).strip()

    def update_code_index(self, call: ToolCall, output: str) -> None:
        if call.name != "Edit":
            return
        paths = [str(call.args[0])] if call.args and isinstance(call.args[0], str) else []
        for match in re.finditer(r'<Edit\s+path=(".*?")', output):
            with contextlib.suppress(json.JSONDecodeError):
                paths.append(str(json.loads(match.group(1))))
        CodeIndex(self.session).update(list(dict.fromkeys(paths)))

    def confirm(self, call: ToolCall, tool: Tool, batch_suffix: str = "", planned_edit: EditBatchPlan.PlannedEdit | None = None) -> tuple[bool, str]:
        self.output_fn(self.approval_display(call, tool, "confirm", batch_suffix=batch_suffix, planned_edit=planned_edit))
        answer = self.input_fn(LogBlock.prefix(2, LogEdge.CONTINUE) + "[Y/n or reason] ").strip()
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
    ) -> LogBlock:
        role = LogRole.TOOL if status == "confirm" else LogRole.AUTO
        root = self.log_root(self.short_call(call), role, batch_suffix, call)
        children = []
        if tool.NAME != "Edit":
            return LogBlock.hierarchy(root, children)
        preview = planned_edit.preview(tool) if planned_edit and isinstance(tool, EditTool) else tool.preview()
        preview_lines = preview.rstrip().splitlines()
        if preview_lines:
            children.append(LogLine("preview", role=LogRole.META, edge=LogEdge.BRANCH))
            children.extend(LogLine("", line, LogRole.DIFF, LogEdge.CONTINUE) for line in preview_lines)
        return LogBlock.hierarchy(root, children)

    def finish_display(
        self,
        call: ToolCall,
        key: str,
        output: str,
        *,
        failed: bool,
        elapsed: float | None = None,
        d: "ToolDisplay | None" = None,
    ) -> str | LogBlock:
        d = d or ToolDisplay()
        if call.name == "Note" and not failed and d.display:
            return self.with_batch_suffix(d.display.removeprefix("Note ").strip(), d.batch_suffix)
        tag = " [refused]" if failed and "user refused" in output else " [failed]" if failed else " [approved]" if d.approved else " [auto]" if d.auto else ""
        tree = d.nested_display or call.name == "Bash"
        root = self.log_root(d.display or self.short_call(call), LogRole.ERROR if failed else LogRole.TOOL, d.batch_suffix, call)
        children = []
        if failed:
            label = "refused" if "user refused" in output else "error"
            children.append(LogLine(label, self.oneline(output, 220), LogRole.ERROR, LogEdge.END))
        elif call.name == "MCP":
            summary = self.mcp_result_summary(call, output, elapsed)
            if summary:
                children.append(LogLine("", summary, LogRole.META, LogEdge.END))
        elif call.name == "Bash":
            preview = self.bash_result_preview(output, self.BASH_TRANSCRIPT_PREVIEW_LINES)
            if preview:
                duration = f" · {elapsed:.1f}s" if elapsed is not None else ""
                children.append(LogLine("output" + duration, "Ctrl-O for more", LogRole.META, LogEdge.BRANCH))
                children.extend(LogLine("", line, LogRole.OUTPUT, LogEdge.CONTINUE) for line in preview.splitlines())
        elif call.name == "Ask":
            children.append(LogLine("answer", self.oneline(output, 220), LogRole.META, LogEdge.END))
        if tree and not failed:
            children.append(LogLine("stored" if key else "done", key + tag if key else tag.strip(), LogRole.META, LogEdge.END))
        elif not tree:
            tail = ((" → " + key) if key else "") + tag
            root = LogLine(root.label, root.text, root.role, meta=root.meta + tail, syntax=root.syntax)
        return LogBlock.hierarchy(None if d.nested_display else root, children)

    def log_root(self, display: str, role: LogRole = LogRole.TOOL, batch_suffix: str = "", call: ToolCall | None = None) -> LogLine:
        name, _, args = display.partition(" ")
        tool_class = TOOL_REGISTRY.get(name)
        syntax = ""
        if tool_class is not None:
            syntax = tool_class.log_lexer(call.args) if call is not None else tool_class.LOG_LEXER
        if role is LogRole.MUTED:
            syntax = ""
        # The batch counter goes into `meta` (rendered gray) instead of `args` (syntax-highlighted),
        # so it reads as a subdued tag on the same line rather than another highlighted token.
        meta = ("  " + batch_suffix) if batch_suffix else ""
        return LogLine(name, args, role, meta=meta, syntax=syntax)

    def bash_result_preview(self, output: str, line_limit: int | None = None) -> str:
        sections = []
        for name in ("stdout", "stderr"):
            text = self.tagged_output(output, name).strip()
            if text:
                sections.extend([name + ":", *("  " + line for line in self.preview_lines(text, line_limit))])
        return "\n".join(sections)

    @staticmethod
    def tagged_output(output: str, name: str) -> str:
        start_tag = f"<{name}>"
        end_tag = f"</{name}>"
        start = output.find(start_tag)
        if start < 0:
            return ""
        start += len(start_tag)
        if output.startswith("\n", start):
            start += 1
        next_section = output.find("\n<stderr>\n", start) if name == "stdout" else output.find("\n</BashToolResult>", start)
        end = output.rfind(end_tag, start, next_section if next_section >= 0 else len(output))
        if end < 0:
            return ""
        text = output[start:end]
        return text[:-1] if text.endswith("\n") else text

    def preview_lines(self, text: str, line_limit: int | None = None) -> list[str]:
        line_limit = self.BASH_PREVIEW_LINES if line_limit is None else line_limit
        lines = [self.clip_preview_line(line) for line in text.splitlines()]
        if len(lines) <= line_limit:
            return lines
        head = line_limit // 2
        tail = line_limit - head
        omitted = len(lines) - line_limit
        noun = "line" if omitted == 1 else "lines"
        return [*lines[:head], f"... {omitted} {noun} omitted ...", *lines[-tail:]]

    def clip_preview_line(self, line: str) -> str:
        line = line.rstrip()
        return line if len(line) <= self.BASH_PREVIEW_LINE_LIMIT else line[: self.BASH_PREVIEW_LINE_LIMIT - 3].rstrip() + "..."

    def mcp_result_summary(self, call: ToolCall, output: str, elapsed: float | None) -> str:
        if str((call.args[0] if call.args and isinstance(call.args[0], dict) else {}).get("action")) != "call":
            return ""
        inner = output
        match = re.match(r"(?s)<MCPCall\b[^>]*>\n?(.*?)\n?</MCPCall>\s*$", output)
        if match:
            inner = match.group(1).strip()
        if not inner:
            shape = "empty"
        else:
            try:
                data = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                data = None
            if isinstance(data, list):
                shape = f"{len(data)} items"
            elif isinstance(data, dict):
                shape = f"{len(data)} fields"
            else:
                shape = f"{inner.count(chr(10)) + 1} lines"
        parts = [f"{shape}, {self.human_size(len(inner))}"]
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s")
        return "→ " + " · ".join(parts)

    @staticmethod
    def human_size(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes}B"
        if num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f}KB"
        return f"{num_bytes / (1024 * 1024):.1f}MB"

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


@dataclass(frozen=True)
class PreparedRequest:
    messages: list[Json]
    tools: list[Json]
    pending: list[QueuedInput]


class ModelClient:
    def __init__(self, session: Session):
        self.session = session
        self.cancel_requested = threading.Event()
        self.active_client: ActiveResource[OpenAI | Anthropic] = ActiveResource()
        self.on_stream: Callable[[str, str], None] | None = None

    def cancel(self) -> None:
        self.cancel_requested.set()
        with contextlib.suppress(Exception):
            self.active_client.apply(lambda client: client.close())

    def call_client(self, client: OpenAI | Anthropic, request: Callable[[], _ResultT]) -> _ResultT:
        with self.active_client.track(client):
            try:
                result = request()
                if self.cancel_requested.is_set():
                    raise KeyboardInterrupt
                return result
            except Exception as error:
                if self.cancel_requested.is_set():
                    raise KeyboardInterrupt from None
                raise ModelError(str(error)) from error
            finally:
                with contextlib.suppress(Exception):
                    client.close()

    def request(self, messages: list[Json], tools: list[Json] | None = None) -> tuple[Json, list[ToolCall], str]:
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        self.cancel_requested.clear()
        tools = tools if tools is not None else Tool.resolved_schemas(self.session)
        state = self.session.state
        state.model_retry_reason = ""
        try:
            attempt = 0
            while True:
                state.current_model_attempt = attempt + 1
                state.current_model_call_started_at = time.monotonic()
                try:
                    result = self.api_request(messages, tools)
                    self.session.images.note_success(messages)
                    return result
                except KeyboardInterrupt:
                    if state.manual_model_retry_requested:
                        state.manual_model_retry_requested = False
                        raise ModelRequestRetry() from None
                    raise
                except ModelError as error:
                    if self.session.images.note_error(messages, error):
                        raise ModelError(f"Active provider/model does not support image input: {error}") from error
                    retryable = self.retryable_error(error)
                    if attempt >= MODEL_REQUEST_RETRIES or not retryable:
                        if attempt:
                            raise ModelError(f"{error} (after {attempt + 1} attempts)") from error
                        raise
                    state.current_model_attempt = attempt + 2
                    state.model_retry_reason = self.retry_reason(error)
                    state.model_retry_count += 1
                    time.sleep(0.5 * (attempt + 1))
                finally:
                    state.current_model_call_started_at = 0.0
                attempt += 1
        finally:
            state.current_model_attempt = 0
            state.model_retry_reason = ""

    @staticmethod
    def retryable_error(error: Exception) -> bool:
        cause = getattr(error, "__cause__", None)

        # SDK status errors expose status_code directly.
        if isinstance(cause, (openai.APIStatusError, anthropic.APIStatusError)):
            return cause.status_code in {408, 409, 425, 429} or 500 <= cause.status_code < 600

        # SDK connection/timeout errors are always retryable.
        if isinstance(
            cause,
            (openai.APIConnectionError, openai.APITimeoutError, anthropic.APIConnectionError, anthropic.APITimeoutError),
        ):
            return True

        # Built-in network/timeout errors are retryable.
        if isinstance(cause, (TimeoutError, asyncio.TimeoutError, ConnectionError, ConnectionResetError, ConnectionAbortedError)):
            return True

        # Fallback: parse status codes embedded in the error text or cause attributes.
        status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
        with contextlib.suppress(Exception):
            if int(status) in {408, 409, 425, 429, 500, 502, 503, 504}:
                return True
        text = str(error).lower()
        if re.search(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(408|409|425|429|5\d\d)\b", text):
            return True
        return any(
            part in text for part in ("internal server error", "timeout", "timed out", "connection reset", "connection aborted", "temporarily unavailable")
        )

    @staticmethod
    def retry_reason(error: Exception) -> str:
        cause = getattr(error, "__cause__", None)
        status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
        with contextlib.suppress(Exception):
            status_code = int(status)
            if 400 <= status_code <= 599:
                return str(status_code)
        text = str(error).lower()
        match = re.search(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(4\d\d|5\d\d)\b", text)
        if match:
            return match.group(1)
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(part in text for part in ("connection", "reset", "aborted")):
            return "connection"
        if "internal server error" in text or "temporarily unavailable" in text:
            return "server error"
        return "transient error"

    def chat_request(self, messages: list[Json], tools: list[Json] | None = None, *, allow_stream: bool = True) -> tuple[Json, list[ToolCall], str]:
        converted: list[Json] = []
        for message in messages:
            clean = {key: value for key, value in message.items() if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY)}
            if message.get("role") == "user" and self.session.images.refs(message):
                clean["content"] = self.session.images.chat_content(message)
            converted.append(clean)
        messages = Text.value(converted)
        provider = self.session.config.provider
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self.on_stream is not None
        params: Json = {"model": provider.model, "messages": messages, "stream": stream}
        if provider.max_tokens > 0:
            params["max_tokens"] = provider.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = True
        prompt_cache_key = self.prompt_cache_key(provider, tools)
        if prompt_cache_key:
            params["prompt_cache_key"] = prompt_cache_key
        self.apply_provider_params(params, provider, resolved)
        if stream:
            params["stream_options"] = {"include_usage": True}
        client = self.client()
        if stream:
            message, usage = self.call_client(client, lambda: self._chat_stream(client, params))
        else:
            response = self.call_client(client, lambda: client.chat.completions.create(**params))
            usage = getattr(response, "usage", None)
            message = response.choices[0].message
        self.session.usage.add(usage)
        assistant = self.assistant_message(message)
        calls = self.tool_calls(message)
        content = str(self.message_field(message, "content") or "")
        return assistant, calls, content

    def _chat_stream(self, client: OpenAI, params: Json) -> tuple[Json, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, Json] = {}
        tool_call_functions: dict[int, Json] = {}
        tool_call_ids: dict[str, int] = {}
        tool_call_positions: dict[int, int] = {}
        next_index = 0
        usage: Any = None

        def allocate_tool_call() -> int:
            nonlocal next_index
            while next_index in tool_calls:
                next_index += 1
            index = next_index
            next_index += 1
            return index

        def resolve_tool_call_index(raw_index: object, call_id: str, position: int, chunk_size: int) -> int:
            nonlocal next_index
            if isinstance(raw_index, int):
                index = raw_index
            elif call_id and call_id in tool_call_ids:
                index = tool_call_ids[call_id]
            elif call_id:
                index = allocate_tool_call()
            elif chunk_size == 1 and len(tool_calls) == 1:
                index = next(iter(tool_calls))
            elif position in tool_call_positions and chunk_size == len(tool_call_positions):
                index = tool_call_positions[position]
            elif position not in tool_call_positions:
                index = allocate_tool_call()
            else:
                raise ModelError("Chat stream tool-call delta omitted both index and id; cannot associate it safely")
            next_index = max(next_index, index + 1)
            tool_call_positions[position] = index
            if call_id:
                tool_call_ids[call_id] = index
            return index

        try:
            for chunk in client.chat.completions.create(**params):
                if chunk_usage := self.message_field(chunk, "usage"):
                    usage = chunk_usage
                choices = self.message_field(chunk, "choices") or []
                if not choices:
                    continue
                delta = self.message_field(choices[0], "delta")
                if reasoning_delta := str(self.message_field(delta, "reasoning_content") or ""):
                    reasoning.append(reasoning_delta)
                    self._emit_stream("reasoning", reasoning_delta)
                if content_delta := str(self.message_field(delta, "content") or ""):
                    content.append(content_delta)
                    self._emit_stream("output", content_delta)
                raw_tool_calls = self.message_field(delta, "tool_calls") or []
                for position, raw in enumerate(raw_tool_calls):
                    raw_index = self.message_field(raw, "index")
                    call_id = str(self.message_field(raw, "id") or "")
                    index = resolve_tool_call_index(raw_index, call_id, position, len(raw_tool_calls))
                    if index not in tool_calls:
                        function_target: Json = {"name": "", "arguments": ""}
                        tool_calls[index] = {"id": "", "type": "function", "function": function_target}
                        tool_call_functions[index] = function_target
                    call = tool_calls[index]
                    if call_id:
                        call["id"] = call_id
                    function = self.message_field(raw, "function")
                    target = tool_call_functions[index]
                    if name := self.message_field(function, "name"):
                        target["name"] = str(name)
                    if arguments := self.message_field(function, "arguments"):
                        target["arguments"] = str(target["arguments"]) + str(arguments)
        finally:
            self._emit_stream("", "")
        message: Json = {"content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return message, usage

    def api_request(self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True) -> tuple[Json, list[ToolCall], str]:
        api = self.session.config.provider.resolve().api
        if api == "anthropic":
            request = self.anthropic_request
        elif api == "responses":
            request = self.responses_request
        else:
            request = self.chat_request
        return request(messages, tools) if allow_stream else request(messages, tools, allow_stream=False)

    def responses_request(self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True) -> tuple[Json, list[ToolCall], str]:
        provider = self.session.config.provider
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self.on_stream is not None
        params: Json = {
            "model": provider.model,
            "input": self.responses_input(Text.value(messages)),
            "stream": stream,
            "store": False,
        }
        if provider.max_tokens > 0:
            params["max_output_tokens"] = provider.max_tokens
        if tools:
            params["tools"] = self.responses_tool_schemas(tools)
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = True
        if prompt_cache_key := self.prompt_cache_key(provider, tools):
            params["prompt_cache_key"] = prompt_cache_key
        # Stateless requests return encrypted reasoning items by default, so the replay below
        # needs no `include`; effort goes through the compatibility fold like the chat path, and
        # a host that defines an explicit "off" spelling still gets it when reasoning is off.
        if resolved.responses_reasoning:
            if effort := resolved.reasoning_effort:
                params["reasoning"] = {"effort": effort}
            elif provider.reasoning == "off":
                raise ModelError("reasoning off is not defined for this Responses model; use a supported effort or configure a documented provider endpoint")
        if provider.temperature is not None and not resolved.suppress_temperature:
            params["temperature"] = provider.temperature
        if provider.extra_body:
            params["extra_body"] = provider.extra_body
        client = self.client()
        result = (
            self.call_client(client, lambda: self._responses_stream(client, params))
            if stream
            else self.call_client(client, lambda: client.responses.create(**params))
        )
        self.session.usage.add(self.message_field(result, "usage"))
        return self.responses_result(result)

    def _responses_stream(self, client: OpenAI, params: Json) -> Any:
        """Consume a Responses event stream and return its terminal response."""

        terminal: Any = None
        try:
            for event in client.responses.create(**params):
                event_type = str(self.message_field(event, "type") or "")
                if event_type == "response.reasoning_summary_text.delta":
                    self._emit_stream("reasoning", str(self.message_field(event, "delta") or ""))
                elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                    self._emit_stream("output", str(self.message_field(event, "delta") or ""))
                elif event_type in ("response.completed", "response.failed", "response.incomplete"):
                    terminal = self.message_field(event, "response")
        finally:
            self._emit_stream("", "")
        if terminal is None:
            raise ModelError("Responses stream ended without a terminal response")
        return terminal

    def _emit_stream(self, kind: str, delta: str) -> None:
        if self.on_stream is not None:
            self.on_stream(kind, delta)

    def responses_input(self, messages: list[Json]) -> list[Json]:
        converted: list[Json] = []
        for message in messages:
            role = str(message.get("role") or "")
            saved_output = message.get(RESPONSES_OUTPUT_KEY)
            if role == "assistant" and isinstance(saved_output, list):
                converted.extend(item for item in saved_output if isinstance(item, dict) and self.replayable_output_item(item))
                continue
            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": str(message.get("content") or ""),
                    }
                )
                continue
            if role not in ("system", "developer", "user", "assistant"):
                continue
            content = message.get("content")
            if content is not None:
                converted.append(
                    {
                        "role": role,
                        "content": self.session.images.responses_content(message) if role == "user" and self.session.images.refs(message) else str(content),
                    }
                )
            if role == "assistant":
                for raw in message.get("tool_calls") or []:
                    if not isinstance(raw, dict):
                        continue
                    raw_function = raw.get("function")
                    function = raw_function if isinstance(raw_function, dict) else {}
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": str(raw.get("id") or uuid.uuid4().hex),
                            "name": str(function.get("name") or ""),
                            "arguments": str(function.get("arguments") or "{}"),
                        }
                    )
        return converted

    @staticmethod
    def replayable_output_item(item: Json) -> bool:
        """Whether a saved output item still carries something a later request can use.

        Stateless reasoning travels in the encrypted payload, which the id alone cannot stand in
        for once the response was never stored. A host that returns neither that payload nor any
        readable reasoning leaves an empty shell, so it is dropped instead of replayed."""
        return item.get("type") != "reasoning" or any(item.get(key) for key in ("encrypted_content", "content", "summary"))

    @staticmethod
    def responses_tool_schemas(tools: list[Json]) -> list[Json]:
        converted: list[Json] = []
        for schema in tools:
            raw_function = schema.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            converted.append(
                {
                    "type": "function",
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
                    "strict": bool(function.get("strict", False)),
                }
            )
        return converted

    def responses_result(self, result: Any) -> tuple[Json, list[ToolCall], str]:
        if self.message_field(result, "status") == "failed":
            error = self.message_field(result, "error") or "unknown error"
            raise ModelError(f"Responses request failed: {error}")
        output = self.message_field(result, "output") or []
        saved_output = [self.dump_message_item(item) for item in output]
        text_parts: list[str] = []
        tool_calls: list[Json] = []
        calls: list[ToolCall] = []
        for item in output:
            item_type = self.message_field(item, "type")
            if item_type == "message":
                for part in self.message_field(item, "content") or []:
                    part_type = self.message_field(part, "type")
                    if part_type == "output_text":
                        text_parts.append(str(self.message_field(part, "text") or ""))
                    elif part_type == "refusal":
                        text_parts.append(str(self.message_field(part, "refusal") or ""))
            elif item_type == "function_call":
                name = str(self.message_field(item, "name") or "")
                call_id = str(self.message_field(item, "call_id") or self.message_field(item, "id") or uuid.uuid4().hex)
                arguments = str(self.message_field(item, "arguments") or "{}")
                try:
                    payload = json.loads(arguments, strict=False)
                except json.JSONDecodeError:
                    payload = {}
                tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
                calls.append(self.tool_call(call_id, name, payload))
        text = "".join(text_parts) or str(self.message_field(result, "output_text") or "")
        assistant: Json = {"role": "assistant", "content": text or None, RESPONSES_OUTPUT_KEY: saved_output}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, calls, text

    @staticmethod
    def dump_message_item(item: Any) -> Json:
        if isinstance(item, dict):
            return Text.value(item)
        if hasattr(item, "model_dump"):
            dumped = item.model_dump(mode="json", exclude_none=True)
            if isinstance(dumped, dict):
                return Text.value(dumped)
        return {}

    def compact(self, context: str) -> Json:
        self.cancel_requested.clear()
        prompt = """
Compact the minacode working context.
Return one JSON object only. No markdown, prose, code fences, or comments.
Use keys: summary, goal, plan, known, check.
Plan must be an array of objects: {"status":"todo|doing|done|blocked","text":"..."}.
Rewrite recent conversation briefly inside summary.
Keep only durable facts needed to continue; preserve file paths, symbols, constraints, and tr.N keys.
""".strip()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": Text.clean(context)}]
        _, _, content = self.api_request(messages, None, allow_stream=False)
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
            api_key=provider.key, base_url=provider.resolve().base_url, timeout=provider.timeout, max_retries=0, default_headers={"User-Agent": HTTP_USER_AGENT}
        )

    def anthropic_client(self) -> Anthropic:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        url = provider.resolve().base_url.rstrip("/")
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
        resolved = provider.resolve()
        if not resolved.prompt_cache_key:
            return ""
        tool_names: list[str] = []
        for schema in tools or []:
            raw_function = schema.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            tool_names.append(str(function.get("name") or schema.get("name") or "(unknown)"))
        payload = {
            "api": resolved.api,
            "cwd": self.session.cwd,
            "host": resolved.host,
            "model": provider.model,
            "tools": ",".join(sorted(tool_names)) or "(none)",
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return "minacode-" + digest[:24]

    def anthropic_request(self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True) -> tuple[Json, list[ToolCall], str]:
        messages = Text.value(messages)
        params = self.anthropic_params(messages, tools)
        client = self.anthropic_client()
        stream = allow_stream and self.session.config.provider.stream and self.on_stream is not None
        result = (
            self.call_client(client, lambda: self._anthropic_stream(client, params))
            if stream
            else self.call_client(client, lambda: client.messages.create(**params))
        )
        self.session.usage.add(self.message_field(result, "usage"))
        assistant, calls, content = self.anthropic_result(result)
        return assistant, calls, content

    def _anthropic_stream(self, client: Anthropic, params: Json) -> Any:
        try:
            with client.messages.stream(**params) as stream:
                for event in stream:
                    if self.message_field(event, "type") != "content_block_delta":
                        continue
                    delta = self.message_field(event, "delta")
                    delta_type = self.message_field(delta, "type")
                    if delta_type == "thinking_delta":
                        self._emit_stream("reasoning", str(self.message_field(delta, "thinking") or ""))
                    elif delta_type == "text_delta":
                        self._emit_stream("output", str(self.message_field(delta, "text") or ""))
                return stream.get_final_message()
        finally:
            self._emit_stream("", "")

    def anthropic_params(self, messages: list[Json], tools: list[Json] | None) -> Json:
        provider = self.session.config.provider
        system_text = "\n\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system").strip()
        # Anthropic prompt caching is a prefix match that only takes effect at explicit
        # cache_control breakpoints; without one, every turn reprocesses the whole prompt from
        # scratch. Render order is tools -> system -> messages, so a breakpoint on the (single)
        # system block caches the stable tools+system prefix and is reused on every later turn.
        system: str | list[Json] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}] if system_text else system_text
        params: Json = {
            "model": provider.model,
            "system": system,
            "messages": self.anthropic_messages(messages),
            "max_tokens": provider.output_token_budget(),
        }
        # Thinking pins temperature to its default; sending any other value is rejected.
        if tools:
            params["tools"] = self.anthropic_tool_schemas(tools)
            params["tool_choice"] = {"type": "auto"}
        effort = provider.reasoning_effort()
        budget = int(CHAT_REASONING_EFFORT_VALUES["enable_thinking"].get(effort, 4096))
        thinking_params = anthropic_thinking_params(
            provider.model,
            provider.reasoning,
            effort,
            min(ANTHROPIC_DEFAULT_MAX_TOKENS - 1024, budget),
        )
        params.update(thinking_params)
        thinking = thinking_params.get("thinking")
        thinking_active = anthropic_thinking_always_on(provider.model) or (isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"))
        if provider.temperature is not None and not thinking_active:
            params["temperature"] = provider.temperature
        return params

    def anthropic_messages(self, messages: list[Json]) -> list[Json]:
        converted: list[Json] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                self.append_anthropic_message(converted, "user", self.session.images.anthropic_content(message))
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
            if isinstance(previous, list) and isinstance(content, str):
                if content:
                    previous.append({"type": "text", "text": content})
                return
            if isinstance(previous, str) and isinstance(content, list):
                messages[-1]["content"] = ([{"type": "text", "text": previous}] if previous else []) + content
                return
            if isinstance(previous, str) and isinstance(content, str):
                messages[-1]["content"] = (previous + "\n\n" + content).strip()
                return
        messages.append({"role": role, "content": content})

    def anthropic_assistant_blocks(self, message: Json) -> list[Json]:
        # The API verifies that thinking blocks come back exactly as it produced them, signature
        # included, so a turn it produced is echoed rather than rebuilt from text and tool calls.
        saved = message.get(ANTHROPIC_CONTENT_KEY)
        if isinstance(saved, list) and saved:
            return [block for block in saved if isinstance(block, dict)]
        blocks: list[Json] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            raw_function = raw.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            try:
                # strict=False: tool-call argument strings often contain literal newlines
                # (e.g. a multi-line git commit message), which are not valid JSON otherwise.
                payload = json.loads(str(function.get("arguments") or "{}"), strict=False)
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
            raw_function = schema.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
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
        content_blocks = self.message_field(result, "content") or []
        saved_content = [self.dump_message_item(block) for block in content_blocks]
        for block in content_blocks:
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
                calls.append(self.tool_call(call_id, name, payload))
        text = "".join(text_parts)
        assistant: Json = {"role": "assistant", "content": text or None, ANTHROPIC_CONTENT_KEY: [block for block in saved_content if block]}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, calls, text

    def apply_provider_params(self, params: Json, provider: ProviderConfig, resolved: ResolvedProvider | None = None) -> None:
        resolved = resolved or provider.resolve()
        chat_reasoning = resolved.chat_reasoning
        reasoning_enabled = provider.reasoning != "off"
        effort = provider.reasoning_effort()
        # Some native APIs fix or reject temperature for all or part of their thinking modes.
        if provider.temperature is not None and not resolved.suppress_temperature:
            params["temperature"] = provider.temperature
        extra: Json = {}
        if reasoning_enabled and chat_reasoning == "reasoning":
            extra["reasoning"] = {"effort": effort}
        elif chat_reasoning == "reasoning_effort":
            if value := resolved.reasoning_effort:
                params["reasoning_effort"] = value
        elif chat_reasoning == "thinking":
            extra["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
            if reasoning_enabled:
                params["reasoning_effort"] = CHAT_REASONING_EFFORT_VALUES["thinking"].get(effort, "high")
        elif chat_reasoning in ("thinking_toggle", "thinking_effort"):
            extra["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
            if reasoning_enabled and chat_reasoning == "thinking_effort":
                params["reasoning_effort"] = resolved.reasoning_effort
        elif chat_reasoning == "enable_thinking":
            extra["enable_thinking"] = reasoning_enabled
            if reasoning_enabled:
                values = CHAT_REASONING_EFFORT_VALUES["enable_thinking"]
                extra["thinking_budget"] = values.get(effort, values["medium"])
        # Provider-declared extensions (e.g. Qianwen web search) pass through verbatim; minacode's
        # own reasoning fields are layered on top so they stay authoritative on key conflicts.
        extra_body = {**provider.extra_body, **extra}
        if extra_body:
            params["extra_body"] = extra_body

    def assistant_message(self, message: Any) -> Json:
        data: Json = {"role": "assistant", "content": self.message_field(message, "content")}
        reasoning_content = self.message_field(message, "reasoning_content")
        if reasoning_content:
            data["reasoning_content"] = reasoning_content
        tool_calls: list[Json] = []
        for call in self.message_field(message, "tool_calls") or []:
            function = self.message_field(call, "function")
            tool_calls.append(
                {
                    "id": str(self.message_field(call, "id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(self.message_field(function, "name") or ""),
                        "arguments": str(self.message_field(function, "arguments") or "{}"),
                    },
                }
            )
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
        for raw in self.message_field(message, "tool_calls") or []:
            function = self.message_field(raw, "function")
            call_id = str(self.message_field(raw, "id") or "")
            name = str(self.message_field(function, "name") or "")
            arguments = str(self.message_field(function, "arguments") or "{}")
            try:
                # strict=False so literal newlines in argument strings (e.g. a multi-line
                # git commit message) parse instead of dropping the call's args.
                payload = json.loads(arguments, strict=False)
            except json.JSONDecodeError:
                calls.append(ToolCall(id=call_id, name=name, args=[]))
                continue
            calls.append(self.tool_call(call_id, name, payload))
        return calls

    @classmethod
    def tool_payload(cls, name: str, payload: object) -> ToolArgs:
        if isinstance(payload, dict) and (tool := TOOL_REGISTRY.get(name)):
            # Strict schemas express optional params as nullable, so the model may send explicit
            # null for an omitted argument. In every minacode tool null means "absent", so drop it.
            cleaned = cls.drop_nulls(payload)
            assert isinstance(cleaned, dict)
            return tool.payload_args(cleaned)
        return [payload]

    @classmethod
    def drop_nulls(cls, value: object) -> object:
        if isinstance(value, dict):
            return {key: cls.drop_nulls(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [cls.drop_nulls(item) for item in value]
        return value

    @classmethod
    def tool_call(cls, call_id: str, name: str, payload: object) -> ToolCall:
        # payload_args may reject malformed arguments (e.g. Bash with an empty command). Capture that
        # error on the call so it is replayed as a tool result during execution, letting the model
        # self-correct, rather than escaping to abort the entire agent turn.
        try:
            return ToolCall(id=call_id, name=name, args=cls.tool_payload(name, payload))
        except ToolError as error:
            return ToolCall(id=call_id, name=name, args=[], error=str(error))


class Agent:
    LIVE_FOLLOWUP_PREFIX = """[Live follow-up received while you were working]
REQUIRED: Your next assistant message must include a brief visible text response to this follow-up, not only tool calls. Then continue the active task; this response is a progress update, not the final answer.
"""
    INTERRUPT_MARKER = "[The user interrupted this turn (Ctrl-C) before it completed.]"
    SYSTEM_PROMPT = """\
You are minacode, a concise terminal coding agent.

ATTITUDE:
- Bring senior engineering judgment, but let it arrive through attention rather than premature certainty. Read the codebase first, resist easy assumptions, and let the existing system teach you how to move.
- When implementation details are open, choose conservatively and in sympathy with the codebase: prefer existing patterns and local helpers, use structured APIs over ad hoc string manipulation, keep edits scoped to the request, add abstractions only to remove real complexity or duplication, and scale tests with risk and blast radius.

TOOLS:
- Available: Read InspectCode Search Edit Bash Job Recall RecallContext Note Ask MCP.
- Use exact tool names and named parameters; obey each tool's DESCRIPTION/SIGNATURE.
- Read inspects files; Search finds text and returns editable anchors; prefer InspectCode over Search for symbols (defs/refs/impls/callers/callees/outline) when the code index is usable. Edit writes files.
- Bash runs everything else — `ls`, `find`, `wc -l`, git, etc. Search text first with `rg` and `rg --files`; fall back to `grep` only if `rg` is unavailable. Do not create or edit files with shell write tricks (e.g., `cat` heredocs, `echo >> file`); use Edit for that. Do not use Python to read/write files when a simple shell command or Edit suffices. Drive each call to finish in one pass: chain known steps with `&&`/`;`/pipelines/a heredoc; split only when a later step needs output you cannot predict.
- Job for long builds/tests, dev servers, and watchers; poll/kill when done. Bash for quick commands. Do not finish the turn while a Job needed for the request is still running.
- Recall retrieves tr.N outputs; RecallContext retrieves or regex-searches stored seg.N excerpts from the history index (conversation evicted by compaction); Note maintains goal/plan/known/check; MCP calls external tools. Before Ask, make progress with other tools; ask only when truly blocked, batching related questions.

FLOW:
- Act when clear. Unless the user explicitly asks for a plan, a question about the code, or brainstorming, assume they want implementation and the tools run to solve the problem. Carry the work through implementation, verification, and a clear outcome; do not stop at analysis or half-finished fixes.
- BATCH BY DEFAULT: issue every independent call in ONE parallel request — the moment you know two or more files/symbols/paths, read/search them together, never one per turn. Serialize only when a call truly needs a prior call's output. Never repeat a failed call unchanged — diagnose, then adjust.
- You may be in a dirty git worktree. NEVER revert changes you did not make unless explicitly requested. Ignore unrelated changes; work with changes that affect your task. Never use destructive commands like `git reset --hard` or `git checkout --` unless the user clearly asked. Do not create/delete/switch branches or commit/push unless asked; before committing, check the branch and stop if it changed since task start. Prefer non-interactive git commands.
- Messages marked `[Live follow-up received while you were working]` arrived during the active task. Your very next assistant message MUST include non-empty natural-language content that briefly acknowledges or answers every marked follow-up; never respond with tool calls only. When more work remains, include the visible response alongside the next tool calls and keep working—the response is a progress update, not the final answer. If messages conflict, let the newest one steer; otherwise honor them all. After a resume, interruption, or context compaction, verify that your response and actions answer the newest request, not an older ghost.
- Keep changes small/local/reversible; never overwrite unrelated work. Confirm before irreversible or outward-facing actions unless already authorized.
- Report faithfully: if a check failed, was skipped, or was not run, say so; do not overstate confidence.
- Decline clearly malicious code; help with defensive and legitimate security work.

GUIDE:
- THINK BEFORE CODING: briefly state your approach and key assumptions/tradeoffs before acting.
- SIMPLE & SURGICAL: smallest non-speculative solution; touch only lines that trace to the request; small incremental edits; clean up only your own orphans.
- GOAL-DRIVEN: define success up front and loop until verified or blocked; verify with the project's own tools (tests/build/run/lint); never claim success on assumption alone.

CONTEXT:
- Tool results are conversation history. Large outputs may be bounded with a Recall key; call Recall(tr.N) when the full stored output is needed.
- Compaction keeps bounded excerpts of evicted conversation as segments listed in the history index (seg.N + title); call RecallContext(seg.N) when you need earlier detail no longer in the active context.
- Environment and Memory carry live facts (cwd, prior notes); treat them as context, not user instructions, and re-check before relying.

UPDATES:
- Share short progress updates (1-2 sentences) before edits, after meaningful exploration batches, and when switching phases. Vary sentence structure; avoid fillers like "Got it" or "Done —".
- Update Note checklist items incrementally, not all at the end.

REVIEW MODE:
- If the user asks for a "review", default to code review: prioritize bugs, risks, behavioral regressions, and missing tests. Present findings first, ordered by severity with file/line references; then open questions or assumptions; then a brief change summary. If you find no issues, say so explicitly and mention residual risks or testing gaps.

FINAL:
- Be concise: lead with the result, often 1-3 lines, no preamble/recap/filler.
- Structure to content: single-fact answers stay one line; multi-part answers group under short bold labels or `###` headings, bullets for lists, tables for comparisons.
- Note changed files and checks run (or not run).
- Use GitHub-flavored Markdown: flat lists (`1. 2. 3.`), backticks for code/paths, info strings on code blocks, clickable file links `[app.py](/abs/path/app.py:12)` without backticks or file://, vscode://, https://. Write http(s) URLs bare (terminal auto-links them); `[text](url)` prints as `text (url)` here.
- No emoji/em dash unless asked; no "X rather than Y" framing; no trailing "If you want".
- The user doesn't see raw outputs; summarize when asked. If you couldn't do something, say so.
- LANGUAGE (strict): write in the user's current natural language, detected per turn. This covers every visible message, including mid-task progress updates, follow-up acknowledgements, and Ask questions/choices/previews — not just the final answer. Keep code, identifiers, paths, shell commands, and tool/API names verbatim — translate only prose.
"""

    def __init__(self, session: Session, input_fn=input, output_fn=print):
        self.session = session
        self.context = ContextManager(session)
        self.model = ModelClient(session)
        self.tools = ToolRunner(session, self.context, input_fn=input_fn, output_fn=output_fn)
        self.output_fn = output_fn
        self.cancel_requested = threading.Event()
        # Called with the queued messages when they are flushed into the turn, so the UI can move
        # them from the live queue region up into the scrollback log. Set by CommandLoop.
        self.on_queue_flush: Callable[[list[str]], None] | None = None

    def cancel(self) -> None:
        self.cancel_requested.set()
        self.tools.cancel()
        self.model.cancel()

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested.is_set():
            raise KeyboardInterrupt

    def run(self, user_input: str | UserInput) -> str:
        self.cancel_requested.clear()
        self.session.state.round_count += 1
        self.session.state.turn_step = 0
        tool_batches = 0
        user_message = self.session.images.message(user_input)
        user_text = self.session.images.label_text(user_message)
        turn_messages: list[Json] = [user_message]
        if self.session.mcp is not None:
            mentions = self.session.mcp.resolve_mentions(user_text)
            if mentions:
                turn_messages.append({"role": "user", "content": mentions})
        if self.session.skills is not None:
            skill_mentions = self.session.skills.resolve_mentions(user_text)
            if skill_mentions:
                turn_messages.append({"role": "user", "content": skill_mentions})
        self.checkpoint_turn(turn_messages)
        try:
            for step in range(self.session.settings.max_steps):
                self.session.state.turn_step = step + 1
                followup_response = False
                while True:
                    try:
                        self.raise_if_cancelled()
                        request = self.prepare_request(turn_messages)
                        assistant, tool_calls, content = self.model.request(request.messages, request.tools)
                        self.raise_if_cancelled()
                        if request.pending and not content.strip():
                            assistant, _, content = self.model.request(request.messages, [])
                            self.raise_if_cancelled()
                            if not content.strip():
                                raise ModelError("empty live follow-up response")
                            followup_response = True
                        self.accept_pending_inputs(turn_messages, request.pending)
                        break
                    except ModelRequestRetry:
                        continue
                if followup_response:
                    response = content.strip()
                    turn_messages.append(self.assistant_turn_message(assistant, [], response))
                    self.output_fn(response)
                    self.checkpoint_turn(turn_messages)
                    continue
                if not tool_calls:
                    if not content.strip():
                        raise ModelError("empty final response")
                    answer = content.strip()
                    self.finish_turn(turn_messages, self.assistant_turn_message(assistant, [], answer))
                    return answer
                assistant = self.assistant_turn_message(assistant, tool_calls, content)
                turn_messages.append(assistant)
                if content.strip():
                    self.output_fn(content.strip())
                tool_batches += 1
                turn_messages.extend(self.tools.run(tool_calls, batch_suffix=f"·{tool_batches}" if tool_batches > 1 else ""))
                self.raise_if_cancelled()
                self.checkpoint_turn(turn_messages)
            stopped = f"Stopped after max_agent_steps={self.session.settings.max_steps}"
            self.finish_turn(turn_messages, {"role": "assistant", "content": stopped})
            return stopped
        except KeyboardInterrupt:
            self.session.release_user_inputs()
            self.settle_interrupted_turn(turn_messages)
            self.session.save_snapshot()
            raise
        except Exception:
            self.session.release_user_inputs()
            self.session.messages.extend(self.session._active_turn_messages)
            self.session._active_turn_messages.clear()
            self.session.state.turn_messages = 0
            self.session.save_snapshot()
            raise

    def checkpoint_turn(self, turn_messages: list[Json]) -> None:
        self.session._active_turn_messages = list(turn_messages)
        self.session.save_snapshot()

    def finish_turn(self, turn_messages: list[Json], assistant: Json) -> None:
        self.session.messages.extend([*turn_messages, assistant])
        self.session._active_turn_messages.clear()
        self.session.state.turn_messages = 0

    def settle_interrupted_turn(self, turn_messages: list[Json]) -> None:
        """Settle a turn the user interrupted with Ctrl-C.

        Two cases, mirroring what the CLI shows. *Retract*: the agent had not said or done
        anything yet, so the turn is discarded and it is as if the message was never sent —
        nothing reaches the model context or the persisted session, though the input history
        still recalls it for Ctrl-P. *Interrupt*: the agent already spoke or called a tool, so
        the partial turn stands (what the CLI showed happened) and an interrupt marker is
        appended, keeping the context valid and telling the model the turn ended early."""
        self.session._active_turn_messages.clear()
        self.session.state.turn_messages = 0
        if not any(message.get("role") != "user" for message in turn_messages):
            return
        answered = {message.get("tool_call_id") for message in turn_messages if message.get("role") == "tool"}
        for message in turn_messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id and call_id not in answered:
                    turn_messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": "Cancelled: the user interrupted before this tool call finished."}
                    )
                    answered.add(call_id)
        turn_messages.append({"role": "user", "content": self.INTERRUPT_MARKER})
        self.session.messages.extend(turn_messages)

    def prepare_request(self, turn_messages: list[Json]) -> PreparedRequest:
        pending = self.session.claim_user_inputs()
        request_turn = [*turn_messages, *(item.message(self.LIVE_FOLLOWUP_PREFIX) for item in pending)]
        self.session.state.turn_messages = len(request_turn)
        tools = Tool.resolved_schemas(self.session)
        messages = self.context.prepare_messages(self.model, self.SYSTEM_PROMPT, request_turn, tools)
        self.context.update_percent(messages, tools)
        return PreparedRequest(messages, tools, pending)

    def accept_pending_inputs(self, turn_messages: list[Json], pending: list[QueuedInput]) -> None:
        if not pending:
            return
        texts = [item.text for item in pending]
        turn_messages.extend(item.message() for item in pending)
        self.session.acknowledge_user_inputs(pending)
        if self.on_queue_flush:
            self.on_queue_flush(texts)

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
