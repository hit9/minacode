"""minacode engine: the agent turn loop that composes context, model, and tools."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable

from minacode.base import (
    Json,
    MalformedToolCallError,
    ModelError,
    ModelRequestRetry,
    Text,
    ToolCall,
)
from minacode.context import ContextManager
from minacode.image import UserInput
from minacode.model import ModelClient, PreparedRequest
from minacode.prompts import (
    INTERRUPT_MARKER,
    LIVE_FOLLOWUP_PREFIX,
    SYSTEM_PROMPT,
)
from minacode.runner import ToolRunner
from minacode.session import QueuedInput, Session
from minacode.tools import (
    Tool,
)

_TEXTUAL_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*(?P<quote>[\"'])(?P<name>[A-Za-z0-9_.:-]{1,128})(?P=quote)\s*>"
    r"(?:(?!<invoke\b).)*</invoke>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r" {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
_BLOCKQUOTE_RE = re.compile(r" {0,3}>")
MAX_TEXTUAL_TOOL_CORRECTIONS = 5


class Agent:
    """Run one user turn to a final answer, composing context, model, and tools.

    A turn is a transaction: messages accumulate in a local list, checkpoint into the session's
    active-turn buffer, and reach durable history only on commit, on settle after an interrupt, or
    on an error flush. Nothing else may append to that history mid-turn.

    The loop alternates model requests and tool batches until the model answers without calling a
    tool, `max_steps` runs out, or the user cancels. Cancellation arrives from another thread and is
    observed only at those boundaries.

    Queued input is claimed per request and acknowledged only once that request succeeds, so a retry
    never swallows a follow-up.
    """

    def __init__(self, session: Session, input_fn=input, output_fn=print):
        self.session = session
        self.model = ModelClient(session)
        self.context = ContextManager(session, self.model)
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
        self.session.clear_quick_hints()  # a new turn invalidates whatever the previous turn offered
        self.session.state.round_count += 1
        self.session.state.turn_step = 0
        tool_batches = 0
        malformed_tool_names: list[str] = []
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
                        while not tool_calls and (textual_tool := self.textual_tool_call(content, request.tools)):
                            self.start_textual_tool_correction(malformed_tool_names, textual_tool)
                            correction_messages = [
                                *request.messages,
                                {"role": "user", "content": self.tool_call_correction(textual_tool)},
                            ]
                            while True:
                                try:
                                    assistant, tool_calls, content = self.model.request(correction_messages, request.tools)
                                    break
                                except ModelRequestRetry:
                                    continue
                            self.raise_if_cancelled()
                        if request.pending and not content.strip():
                            assistant, followup_tool_calls, content = self.model.request(request.messages, [])
                            self.raise_if_cancelled()
                            while not followup_tool_calls and (textual_tool := self.textual_tool_call(content, request.tools)):
                                self.start_textual_tool_correction(malformed_tool_names, textual_tool)
                                correction_messages = [
                                    *request.messages,
                                    {"role": "user", "content": self.followup_tool_call_correction(textual_tool)},
                                ]
                                while True:
                                    try:
                                        assistant, followup_tool_calls, content = self.model.request(correction_messages, [])
                                        break
                                    except ModelRequestRetry:
                                        continue
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
                if content.strip() and self.terminal_next_hints(tool_calls):
                    return self.finish_with_next_hints(turn_messages, assistant, tool_calls, content, tool_batches)
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

    def terminal_next_hints(self, tool_calls: list[ToolCall]) -> bool:
        """True when a batch is nothing but NextHints calls — a terminal batch that ends the turn."""
        return bool(tool_calls) and all(call.name == "NextHints" for call in tool_calls)

    def finish_with_next_hints(self, turn_messages: list[Json], assistant: Json, tool_calls: list[ToolCall], content: str, tool_batches: int) -> str:
        """Run an all-NextHints batch and finish the turn with `content` in a single model call.

        The tool-bearing assistant message keeps only the calls; the answer becomes its own final
        message so it appears exactly once in history."""
        answer = content.strip()
        tool_message = dict(assistant or {})
        tool_message["content"] = None
        tool_message.pop("tool_calls", None)
        turn_messages.append(self.assistant_turn_message(tool_message, tool_calls, ""))
        batches = tool_batches + 1
        turn_messages.extend(self.tools.run(tool_calls, batch_suffix=f"\u00b7{batches}" if batches > 1 else ""))
        self.raise_if_cancelled()
        self.finish_turn(turn_messages, {"role": "assistant", "content": answer})
        return answer

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
        turn_messages.append({"role": "user", "content": INTERRUPT_MARKER})
        self.session.messages.extend(turn_messages)

    def prepare_request(self, turn_messages: list[Json]) -> PreparedRequest:
        pending = self.session.claim_user_inputs()
        request_turn = [*turn_messages, *(item.message(LIVE_FOLLOWUP_PREFIX) for item in pending)]
        self.session.state.turn_messages = len(request_turn)
        tools = Tool.resolved_schemas(self.session)
        messages = self.context.prepare_messages(self.model, SYSTEM_PROMPT, request_turn, tools)
        self.context.update_percent(messages, tools)
        return PreparedRequest(messages, tools, pending)

    @classmethod
    def textual_tool_call(cls, content: str, tools: list[Json]) -> str | None:
        """Recognize a terminal textual invoke without interpreting any of its arguments."""

        match = _TEXTUAL_INVOKE_RE.search(content)
        if match is None or cls.inside_markdown_literal(content, match.start()):
            return None
        known = {str(function.get("name") or "") for schema in tools if isinstance(schema, dict) and isinstance((function := schema.get("function")), dict)}
        name = match.group("name")
        return name if name in known else None

    @staticmethod
    def inside_markdown_literal(content: str, offset: int) -> bool:
        line_start = content.rfind("\n", 0, offset) + 1
        prefix = content[line_start:offset]
        leading_whitespace = prefix[: len(prefix) - len(prefix.lstrip(" \t"))]
        if len(leading_whitespace.expandtabs(4)) >= 4 or _BLOCKQUOTE_RE.match(prefix):
            return True

        fence: tuple[str, int] | None = None
        for line in content[:offset].splitlines():
            match = _FENCE_RE.match(line)
            if match is None:
                continue
            marker = match.group("marker")
            rest = match.group("rest")
            if fence is None:
                if marker[0] == "`" and "`" in rest:
                    continue
                fence = marker[0], len(marker)
            elif marker[0] == fence[0] and len(marker) >= fence[1] and not rest.strip():
                fence = None
        return fence is not None

    def start_textual_tool_correction(self, names: list[str], name: str) -> None:
        if len(names) >= MAX_TEXTUAL_TOOL_CORRECTIONS:
            raise self.malformed_tool_call_error([*names, name])
        names.append(name)
        on_stream = getattr(self.model, "on_stream", None)
        if callable(on_stream):
            on_stream(f"correcting malformed tool call {len(names)}/{MAX_TEXTUAL_TOOL_CORRECTIONS} · {name}", "")

    @staticmethod
    def tool_call_correction(name: str) -> str:
        return "\n".join(
            [
                "[Runtime protocol correction]",
                f"The previous generation printed a textual <invoke> for {name}. Nothing was executed.",
                "Continue the same task using the native tool interface. Do not output tool markup.",
            ]
        )

    @staticmethod
    def followup_tool_call_correction(name: str) -> str:
        return "\n".join(
            [
                "[Runtime protocol correction]",
                f"The previous generation printed a textual <invoke> for {name}. Nothing was executed.",
                "Respond briefly in natural language to the live follow-up. Do not call a tool or output tool markup in this response.",
            ]
        )

    @staticmethod
    def malformed_tool_call_error(names: list[str]) -> MalformedToolCallError:
        count = len(names)
        if len(set(names)) == 1:
            return MalformedToolCallError(f"Model emitted {names[0]} as text {count} times; none of the textual calls were executed.")
        sequence = ", then ".join(names)
        return MalformedToolCallError(f"Model emitted tool calls as text {count} times ({sequence}); none of the textual calls were executed.")

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
