"""minacode context: model message projection, deduplication, and compaction."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Hashable
from typing import ClassVar, TypeVar

from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    MAX_TOOL_OUTPUT_TOKENS,
    PROVIDER_ECHO_KEYS,
    RESPONSES_OUTPUT_KEY,
    SESSION_EVENT_KEY,
    Json,
    Text,
    request_budget_for,
)
from minacode.image import IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, ImageInputs
from minacode.model import ModelClient
from minacode.prompts import (
    COMPACTION_SUMMARY_TITLE,
    CURRENT_TURN_CONTEXT_TRIMMED,
    PREVIOUS_CONTEXT_TRIMMED,
)
from minacode.prompts import (
    compaction_input as format_compaction_input,
)
from minacode.session import HistorySegment, Session
from minacode.tools import (
    Tool,
)

_IdentityT = TypeVar("_IdentityT", bound=Hashable)


class ContextManager:
    """Project session state into one request's messages and compact it to fit the budget.

    Request projection is derived at the send boundary and never stored: replay transforms must not
    write back into history. Compaction is the deliberate persisted exception. When the projected
    request exceeds its budget, older messages are captured in retained history and replaced by one
    summary checkpoint. Layer order exists for prompt-cache stability — version-stable system and
    tools, then session-stable environment/capability context, then the append-only conversation and
    active turn. Mutable working state is written as tool history or compaction checkpoints; inserting
    a rebuilt block into the conversation prefix would invalidate later cache reuse.

    Request-local transforms belong here rather than in stored messages: repeated MCP schemas and
    skill loads collapse to a pointer at the first copy, re-promoted when compaction removes it.

    The budget is the context limit less the provider's output reserve and a safety margin, measured
    against the payload that actually crosses the wire. Over budget compacts prior history first, and
    the current turn only if still over.
    """

    COMPACT_RECENT_MESSAGES: ClassVar[int] = 8
    MCP_DESCRIBE_BLOCK: ClassVar[re.Pattern] = re.compile(r"<MCPDescribe server=(\".*?\") tool=(\".*?\")>.*?</MCPDescribe>", re.DOTALL)
    SKILL_BLOCK: ClassVar[re.Pattern] = re.compile(r"<Skill name=(\".*?\")>.*?</Skill>", re.DOTALL)
    TOOL_RECORD_KEY: ClassVar[re.Pattern] = re.compile(r"\btr\.\d+\b")

    def __init__(self, session: Session, model: ModelClient | None = None):
        self.session = session
        self.model = model
        # Automatic compaction runs inside request projection, below the UI layer. This lifecycle
        # hook lets orchestration expose that real phase without making context depend on a renderer.
        # False is emitted in a finally block, including model failures that fall back to trimming.
        self.on_compaction: Callable[[bool], None] | None = None

    def model_messages(self, base_system: str, turn_messages: list[Json] | None = None) -> list[Json]:
        messages: list[Json] = [
            {"role": "system", "content": base_system.strip()},
            {"role": "user", "content": "--- Environment ---\n" + (self.environment() or "(empty)")},
        ]
        for context in (self.skills_context(), self.mcp_tools_context()):
            if context:
                messages.append({"role": "user", "content": context})
        conversation = [*self.session.messages, *(turn_messages or [])]
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
                key = ContextManager.TOOL_RECORD_KEY.search(content)
                seen[identity] = key.group(0) if key else "above"
                result.append(message)
                continue
            marker = marker_for(identity, first_key)
            result.append({**message, "content": block.sub(lambda _, marker=marker: marker, content)})
        return result

    def mcp_tools_context(self) -> str:
        return self.session.mcp.render_tools_index() if self.session.mcp else ""

    def skills_context(self) -> str:
        return self.session.skills.index() if self.session.skills else ""

    def request_token_budget(self) -> int:
        return request_budget_for(
            self.session.settings.max_context_tokens,
            self.session.config.provider.output_token_budget(),
        )

    def request_tokens(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        if self.model is not None:
            return self.model.estimated_request_tokens(messages, tools)
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

    def prepare_messages(self, model: ModelClient, base_system: str, turn_messages: list[Json] | None = None, tools: list[Json] | None = None) -> list[Json]:
        messages = self.model_messages(base_system, turn_messages)
        budget = self.request_token_budget()
        if self.request_tokens(messages, tools) < budget:
            return messages
        compacted, keep = self.compaction_parts()
        if self._compact_messages(model, compacted, keep, PREVIOUS_CONTEXT_TRIMMED, tool_messages=turn_messages):
            messages = self.model_messages(base_system, turn_messages)
        if turn_messages is not None and self.request_tokens(messages, tools) >= budget:
            compacted, keep = self.turn_compaction_parts(turn_messages)
            if self._compact_messages(model, compacted, keep, CURRENT_TURN_CONTEXT_TRIMMED, turn_messages=turn_messages):
                messages = self.model_messages(base_system, turn_messages)
        return messages

    def _compact_messages(
        self,
        model: ModelClient,
        compacted: list[Json],
        keep: list[Json],
        fallback_note: str,
        *,
        tool_messages: list[Json] | None = None,
        turn_messages: list[Json] | None = None,
    ) -> bool:
        if not compacted:
            return False
        on_compaction = self.on_compaction
        if on_compaction is not None:
            on_compaction(True)
        try:
            try:
                data = model.compact(self.compaction_input(compacted))
            except Exception:  # noqa: BLE001 - compaction degrades to deterministic trimming on any model failure.
                data = None
            self.apply_compaction(
                data,
                keep,
                tool_messages,
                turn_messages=turn_messages,
                fallback_note=fallback_note if data is None else "",
                compacted=compacted,
            )
        finally:
            if on_compaction is not None:
                on_compaction(False)
        return True

    def environment(self) -> str:
        info = self.session.system_info
        assert info is not None
        rows = [
            f"- cwd: {info.cwd}",
            f"- session_started_at: {self.session.created_at}",
            # Tell the model which executables it may drive through Bash.
            "- detected_commands (available via Bash): " + (", ".join(info.commands) or "(none)"),
            f"- os: {info.os}",
            f"- arch: {info.arch}",
            f"- shell_timeout: {self.session.settings.shell_timeout}s",
        ]
        return "\n".join(rows)

    def compaction_input(self, messages: list[Json]) -> str:
        older, recent = self.compaction_parts_for(messages)
        return format_compaction_input(
            state=self.session.state.format(),
            previous_summary=self.session.state.summary,
            older_messages=self.messages_text(older),
            recent_messages=self.messages_text(recent),
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
        """Split messages into a compactable head and a recent tail, never inside a tool exchange.

        The cut walks back past a run of tool results and the assistant message that called them, since
        a history with tool calls whose results were summarized away — or results whose call is gone —
        is rejected by every provider. Giving a few extra messages to the summary is the cheaper loss.
        """
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
            if (
                message.get("role") == "user"
                and not message.get(SESSION_EVENT_KEY)
                and not str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)
                and not ImageInputs.is_tool_observation(message)
            ):
                return Tool.compact(str(message.get("content") or ""), 80)
        return Tool.compact(self.messages_text(messages[:1]), 80) or "compacted context"

    def store_history_segment(self, compacted: list[Json]) -> HistorySegment:
        key = f"seg.{len(self.session.history) + 1}"
        text = self.bound_output(self.messages_text(compacted))
        segment = HistorySegment(key=key, title=self.history_title(compacted), text=text)
        self.session.history.append(segment)
        return segment

    def _summary_block(self, segment: HistorySegment | None) -> list[Json]:
        """One durable checkpoint containing everything needed after the compacted prefix."""
        rows = [
            COMPACTION_SUMMARY_TITLE,
            "Summary:",
            self.session.state.summary or "(empty)",
            "",
            "Working state:",
            self.session.state.format(),
        ]
        if segment is not None:
            rows.extend(("", f"Stored history segment: {segment.key}: {segment.title}"))
        return [{"role": "user", "content": "\n".join(rows), SESSION_EVENT_KEY: "compaction_checkpoint"}]

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
        segment = self.store_history_segment(compacted) if compacted else None
        if data is not None:
            self.session.state.apply(data)
        if fallback_note:
            self.session.state.summary = (self.session.state.summary + "\n" + fallback_note).strip()
        summary_block = self._summary_block(segment)
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
        keep = set(self.TOOL_RECORD_KEY.findall(self.messages_text(keep_messages)))
        self.session.tool_records = [record for record in records if record.key in keep][-400:]
        self.session.tool_results = {record.key: record.output for record in self.session.tool_records}

    def latest_user_index(self, messages: list[Json]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if (
                messages[index].get("role") == "user"
                and not messages[index].get(SESSION_EVENT_KEY)
                and not self.is_compaction_summary(messages[index])
                and not ImageInputs.is_tool_observation(messages[index])
            ):
                return index
        return None

    def is_compaction_summary(self, message: Json) -> bool:
        return message.get("role") == "user" and str(message.get("content") or "").startswith(COMPACTION_SUMMARY_TITLE)

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
            estimated = {
                key: value for key, value in message.items() if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, SESSION_EVENT_KEY)
            }
            if readable := readable_provider_context(message):
                estimated["_provider_context"] = readable
            payload.append(estimated)
        chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        images = ImageInputs.estimated_tokens(messages) if self.session.images.support() is not False else 0
        return (chars + 3) // 4 + images

    @staticmethod
    def estimated_text_tokens(text: str) -> int:
        return (len(text) + 3) // 4
