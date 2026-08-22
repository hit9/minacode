"""minacode engine: the agent turn loop that composes context, model, and tools."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable

from minacode.base import (
    IMAGE_ROUTE_UNKNOWN,
    PAUSED_TURN_KEY,
    SEARCH_SOURCES_KEY,
    SESSION_EVENT_KEY,
    Json,
    MalformedToolCallError,
    ModelError,
    ModelRequestRetry,
    Text,
    ToolCall,
)
from minacode.context import ContextManager
from minacode.image import ImageInputs, UserInput
from minacode.model import ModelClient, PreparedRequest, resilience
from minacode.prompts import (
    FAILED_TOOL_CALL_RESULT,
    FAILED_TURN_MARKER,
    INTERRUPT_MARKER,
    LIVE_FOLLOWUP_PREFIX,
)
from minacode.runner import ToolRunner
from minacode.session import QueuedInput, Session, SessionSnapshotCodec
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

    def __init__(self, session: Session, input_fn=input, output_fn=print, final_output_fn=None):
        self.session = session
        self.model = ModelClient(session)
        self.context = ContextManager(session, self.model)
        self.tools = ToolRunner(session, self.context, input_fn=input_fn, output_fn=output_fn)
        self.output_fn = output_fn
        # How a turn's final answer is published, when it should look different from interim
        # text (e.g. markdown rendering). None publishes through output_fn like interim text.
        self.final_output_fn = final_output_fn
        self.cancel_requested = threading.Event()
        # Image-bearing semantic messages introduced into the active turn since its last accepted
        # main-model request (opening attachment, claimed queued attachment, ViewImage
        # observation). Cleared when a request is accepted; used to decide 400 eligibility and to
        # observe exactly the current occurrences, never older accepted history.
        self._current_image_messages: list[Json] = []
        # Presentation hook for image-routing notices (one gray line per unknown->learned
        # transition). Wired by the CLI; never enters model context.
        self.on_image_route_notice: Callable[[str], None] | None = None
        # Sources the provider's own search reported during the last turn, in the order they appeared.
        # The UI renders them under the answer; the turn's stored messages are left untouched.
        self.turn_sources: list[Json] = []
        # Set when the last run ended because max_steps ran out (not because the model answered).
        # Runtime fact for callers like the Delegate tool; never derived from the answer's wording.
        self.stopped_at_max_steps = False
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
        self.stopped_at_max_steps = False
        self.turn_sources = []
        self.session.clear_quick_hints()  # a new turn invalidates whatever the previous turn offered
        self.session.state.round_count += 1
        self.session.state.turn_step = 0
        tool_batches = 0
        malformed_tool_names: list[str] = []
        self._current_image_messages = []
        user_message = self._initial_user_message(user_input)
        if ImageInputs.input_refs(user_message):
            # The opening attachment is a current image occurrence for the first request of the turn.
            self._current_image_messages.append(user_message)
        # Mentions belong to the user's typed input, never to projected image content.
        user_text = user_input.display_text() if isinstance(user_input, UserInput) else self.session.images.label_text(user_message)
        turn_messages = [user_message, *self.mention_messages(user_text)]
        transcript_messages: list[Json] = [self.transcript_message(user_message)]
        self.checkpoint_turn(turn_messages, transcript_messages)
        failed_request: PreparedRequest | None = None
        try:
            for step in range(self.session.settings.max_steps):
                self.session.state.turn_step = step + 1
                self.session.clear_quick_hints()  # a later step supersedes hints from a non-terminal batch; only the terminal batch keeps its hints
                while True:
                    try:
                        self.raise_if_cancelled()
                        request = self.prepare_request(turn_messages)
                        failed_request = request
                        # One main-model request, with an eligible 400 converting into exactly one
                        # vision fallback (see _image_fallback). `accepted` is the request whose
                        # acceptance commits the turn — the retry on a recovered 400.
                        assistant, tool_calls, content, accepted = self._main_request(request, turn_messages)
                        self.record_sources(assistant)
                        self.raise_if_cancelled()
                        # The request reached the provider, so its follow-ups belong to history from
                        # here on, and any correction sent next lands after them — history keeps the
                        # order the provider saw, because a sent message can never be taken back.
                        self.accept_pending_inputs(turn_messages, transcript_messages, accepted.pending, accepted.turn_messages)
                        failed_request = None
                        assistant, tool_calls, content = self.correct_textual_tool_calls(
                            assistant,
                            tool_calls,
                            content,
                            base_messages=accepted.messages,
                            tools=accepted.tools,
                            names=malformed_tool_names,
                            turn_messages=turn_messages,
                            transcript_messages=transcript_messages,
                        )
                        break
                    except ModelRequestRetry:
                        continue
                if assistant.get(PAUSED_TURN_KEY) and not tool_calls:
                    # The provider paused a long server-side tool run rather than ending the turn.
                    # Resuming means sending this message back unchanged and asking again, so it
                    # joins the turn like any other step: bounded by max_steps, checkpointed, and
                    # never mistaken for the answer even though it carries no tool call of ours.
                    assistant_message = self.assistant_turn_message(assistant, [], content)
                    turn_messages.append(assistant_message)
                    transcript_messages.append(self.transcript_message(assistant_message))
                    if content.strip():
                        self.output_fn(content.strip())
                    self.checkpoint_turn(turn_messages, transcript_messages)
                    continue
                if not tool_calls:
                    if not content.strip():
                        raise ModelError("empty final response")
                    answer = content.strip()
                    # Publish the final answer through the same output channel as interim text, so
                    # every agent (the parent's and a worker's) reports its final answer the same
                    # way; callers that used to print the return value themselves no longer do.
                    (self.final_output_fn or self.output_fn)(answer)
                    self.finish_turn(turn_messages, transcript_messages, self.assistant_turn_message(assistant, [], answer))
                    return answer
                if self.terminal_next_hints(tool_calls):
                    answer = self.finish_with_next_hints(
                        turn_messages,
                        assistant,
                        tool_calls,
                        content,
                        tool_batches,
                        transcript_messages=transcript_messages,
                    )
                    if answer is not None:
                        return answer
                    # The batch produced neither text nor hints (every call failed). Its error
                    # results are already in the turn history; continue so the model reads them
                    # and corrects, instead of ending on a blank turn. The failed batch still
                    # counts as a tool batch, so the next ordinary batch is numbered ·2 instead
                    # of presenting as the first.
                    tool_batches += 1
                    self.checkpoint_turn(turn_messages, transcript_messages)
                    continue
                assistant = self.assistant_turn_message(assistant, tool_calls, content)
                turn_messages.append(assistant)
                transcript_messages.append(self.transcript_message(assistant))
                if content.strip():
                    self.output_fn(content.strip())
                tool_batches += 1
                tool_messages = self.tools.run(tool_calls, batch_suffix=f"·{tool_batches}" if tool_batches > 1 else "")
                turn_messages.extend(tool_messages)
                # ViewImage observations produced by this batch are current image occurrences for
                # the next main-model request; their refs feed the 400-eligibility check and the
                # fallback observation, never older accepted history.
                self._current_image_messages.extend(message for message in tool_messages if ImageInputs.input_refs(message))
                transcript_messages.extend(SessionSnapshotCodec.transcript_messages(tool_messages))
                self.raise_if_cancelled()
                self.checkpoint_turn(turn_messages, transcript_messages)
            stopped = f"Stopped after max_agent_steps={self.session.settings.max_steps}"
            self.stopped_at_max_steps = True
            self.finish_turn(turn_messages, transcript_messages, {"role": "assistant", "content": stopped})
            (self.final_output_fn or self.output_fn)(stopped)
            return stopped
        except KeyboardInterrupt:
            self.session.release_user_inputs()
            self.settle_interrupted_turn(turn_messages, transcript_messages)
            self.session.save_snapshot()
            raise
        except Exception as error:
            if isinstance(error, ModelError):
                # A queued follow-up was part of the rejected request too. Commit the exact sent
                # turn before settling its images, rather than releasing it to repeat the same
                # rejected image on every later request.
                if failed_request is not None and failed_request.pending:
                    self.accept_pending_inputs(
                        turn_messages,
                        transcript_messages,
                        failed_request.pending,
                        failed_request.turn_messages,
                    )
                self.session.images.settle_failed_messages(turn_messages)
            self.session.release_user_inputs()
            # A turn that died from an error still has to leave a legal, marked history: tool
            # calls that never got results are settled so the next request is not rejected for a
            # dangling call, and a marker records where the turn ended. Settling only keeps the
            # history valid; the failure still propagates unchanged.
            self.settle_unanswered_tool_calls(turn_messages, transcript_messages, FAILED_TOOL_CALL_RESULT)
            # The error is bounded before it is written down. Unlike INTERRUPT_MARKER this marker
            # interpolates a value nobody controls -- a provider can answer with a whole HTTP body --
            # and it lands in permanent history, so it would ride every later request and the
            # compaction payload with it. What the marker is for is where the turn stopped, and a
            # line of that fits.
            turn_messages.append({"role": "user", "content": FAILED_TURN_MARKER.format(error=ToolRunner.oneline(str(error), 300))})
            self.session.messages.extend(turn_messages)
            self.session.transcript_messages.extend(transcript_messages)
            self.session._active_turn_messages.clear()
            self.session._active_transcript_messages.clear()
            self.session.state.turn_messages = 0
            self.session.save_snapshot()
            raise

    def _initial_user_message(self, user_input: str | UserInput) -> Json:
        """Build the turn's opening user message, preserving image refs for direct projection."""

        return self.session.images.message(user_input)

    def correct_textual_tool_calls(
        self,
        assistant: Json,
        tool_calls: list[ToolCall],
        content: str,
        *,
        base_messages: list[Json],
        tools: list[Json],
        names: list[str],
        turn_messages: list[Json],
        transcript_messages: list[Json],
    ) -> tuple[Json, list[ToolCall], str]:
        """Retry terminal textual tool markup with a protocol correction sent as a real message.

        Each correction joins the turn before it is sent and the retry keeps the same tool list:
        what reached the provider must reach history, and the tool block is part of the cached
        prefix, so neither may be reshaped for a single request."""
        corrections: list[Json] = []
        while not tool_calls and (textual_tool := self.textual_tool_call(content, tools)):
            self.start_textual_tool_correction(names, textual_tool)
            correction: Json = {"role": "user", "content": self.tool_call_correction(textual_tool), SESSION_EVENT_KEY: "tool_call_correction"}
            corrections.append(correction)
            turn_messages.append(correction)
            self.checkpoint_turn(turn_messages, transcript_messages)
            correction_messages = [*base_messages, *corrections]
            while True:
                try:
                    assistant, tool_calls, content = self.model.request(correction_messages, tools)
                    self.record_sources(assistant)
                    break
                except ModelRequestRetry:
                    continue
            self.raise_if_cancelled()
        return assistant, tool_calls, content

    def record_sources(self, assistant: Json) -> None:
        """Accumulate provider-side search sources across every request the turn makes.

        A search can happen in any step, not only the one that answers, so collecting per request
        is what lets the footer describe the whole turn. Duplicates are resolved by URL at render
        time, which also covers a request that was retried or corrected."""
        for source in assistant.get(SEARCH_SOURCES_KEY) or []:
            if isinstance(source, dict):
                self.turn_sources.append(source)

    def checkpoint_turn(self, turn_messages: list[Json], transcript_messages: list[Json]) -> None:
        self.session._active_turn_messages = list(turn_messages)
        self.session._active_transcript_messages = list(transcript_messages)
        self.session.save_snapshot()

    def finish_turn(self, turn_messages: list[Json], transcript_messages: list[Json], assistant: Json | None = None) -> None:
        if assistant is not None:
            self.session.messages.extend([*turn_messages, assistant])
            self.session.transcript_messages.extend([*transcript_messages, self.transcript_message(assistant)])
        else:
            self.session.messages.extend(turn_messages)
            self.session.transcript_messages.extend(transcript_messages)
        self.session._active_turn_messages.clear()
        self.session._active_transcript_messages.clear()
        self.session.state.turn_messages = 0

    def terminal_next_hints(self, tool_calls: list[ToolCall]) -> bool:
        """True when a batch is nothing but NextHints calls — a terminal batch that ends the turn."""
        return bool(tool_calls) and all(call.name == "NextHints" for call in tool_calls)

    def finish_with_next_hints(
        self,
        turn_messages: list[Json],
        assistant: Json,
        tool_calls: list[ToolCall],
        content: str,
        tool_batches: int,
        *,
        transcript_messages: list[Json] | None = None,
    ) -> str | None:
        """Run an all-NextHints batch, finishing the turn when it actually produced output.

        Returns the turn's answer (possibly "") when the turn ends: the answer text, or — with
        no text — the NextHints tool result that stored suggestions. Returns None when neither
        happened, i.e. every call failed: the error results stay in the turn history and the
        caller must continue to the next step so the model can correct instead of ending on a
        blank turn.

        The tool-bearing assistant message keeps only the calls; with answer text, the answer
        becomes its own final message so it appears exactly once in history. Without answer
        text the tool result ends the history and the turn returns an empty string."""
        answer = content.strip()
        transcript_messages = transcript_messages if transcript_messages is not None else SessionSnapshotCodec.transcript_messages(turn_messages)
        tool_message = dict(assistant or {})
        tool_message["content"] = None
        tool_message.pop("tool_calls", None)
        assistant_message = self.assistant_turn_message(tool_message, tool_calls, "")
        turn_messages.append(assistant_message)
        transcript_messages.append(self.transcript_message(assistant_message))
        batches = tool_batches + 1
        result_messages = self.tools.run(tool_calls, batch_suffix=f"\u00b7{batches}" if batches > 1 else "")
        turn_messages.extend(result_messages)
        transcript_messages.extend(SessionSnapshotCodec.transcript_messages(result_messages))
        self.raise_if_cancelled()
        if not answer and not self.session.quick_hints:
            # The batch produced neither text nor suggestions (every call failed). Ending here
            # would be a blank turn the user sees as nothing happening; keep the error results
            # in history and let the next step read them and correct.
            return None
        if answer:
            # Text exists: the answer becomes its own final message and is published exactly
            # once, unchanged from the plain final-answer path.
            self.finish_turn(turn_messages, transcript_messages, {"role": "assistant", "content": answer})
            (self.final_output_fn or self.output_fn)(answer)
        else:
            # Tool-only terminal batch: the NextHints tool result ends the history. All three
            # adapters replay a turn whose last message is that tool result once the next
            # user message is appended, so no empty closing assistant message is stored, and
            # nothing is published (an empty answer must not reach the visible output).
            self.finish_turn(turn_messages, transcript_messages)
        return answer

    def settle_interrupted_turn(self, turn_messages: list[Json], transcript_messages: list[Json]) -> None:
        """Settle a turn the user interrupted with Ctrl-C.

        Two cases, mirroring what the CLI shows. *Retract*: the agent had not said or done
        anything yet, so the turn is discarded and it is as if the message was never sent —
        nothing reaches the model context or the persisted session, though the input history
        still recalls it for Ctrl-P. *Interrupt*: the agent already spoke or called a tool, so
        the partial turn stands (what the CLI showed happened) and an interrupt marker is
        appended, keeping the context valid and telling the model the turn ended early."""
        self.session._active_turn_messages.clear()
        self.session._active_transcript_messages.clear()
        self.session.state.turn_messages = 0
        if not any(message.get("role") != "user" for message in transcript_messages):
            return

        def cancelled_text(call: Json) -> str:
            if (call.get("function") or {}).get("name") == "Delegate":
                # Name who was cancelled and that the worker's context survives: the parent
                # cannot see the worker, so the interrupt line is its only notice.
                return "Cancelled: the worker's turn was interrupted; its context is kept, reset it with /worker reset."
            return "Cancelled: the user interrupted before this tool call finished."

        self.settle_unanswered_tool_calls(turn_messages, transcript_messages, cancelled_text)
        turn_messages.append({"role": "user", "content": INTERRUPT_MARKER})
        self.session.messages.extend(turn_messages)
        self.session.transcript_messages.extend(transcript_messages)

    def settle_unanswered_tool_calls(
        self,
        turn_messages: list[Json],
        transcript_messages: list[Json],
        text: str | Callable[[Json], str],
    ) -> None:
        """Give every tool call of the turn that never got a result a synthetic failed one.

        A turn that ends early — interrupted or dead from an error — may leave an assistant
        message whose tool_calls have no matching tool results, and providers reject a messages
        list with dangling calls: one such turn would fail every later request on this session.
        `text` is the result content for each unanswered call; a callable receives the call so
        the wording can depend on the tool (the interrupt path's Delegate line)."""
        answered = {message.get("tool_call_id") for message in turn_messages if message.get("role") == "tool"}
        for message in turn_messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id and call_id not in answered:
                    content = text(call) if callable(text) else text
                    turn_messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
                    transcript_messages.append({"role": "tool", "tool_call_id": call_id, "result_key": "", "status": "failed"})
                    answered.add(call_id)

    @staticmethod
    def transcript_message(message: Json) -> Json:
        projected = SessionSnapshotCodec.transcript_message(message)
        if projected is None:
            raise ValueError("internal messages cannot be added to the visible transcript")
        return projected

    def prepare_request(self, turn_messages: list[Json]) -> PreparedRequest:
        pending = self.session.claim_user_inputs()
        # Without a queued follow-up this must be the real active-turn list: current-turn compaction
        # rewrites it in place, and a throwaway copy would make the next step compact the same prefix
        # again. Pending input stays transactional in a copy until the provider accepts it.
        request_turn = turn_messages
        current: list[Json] = list(self._current_image_messages)
        if pending:
            request_turn = [*turn_messages]
            for item in pending:
                mentions = self.mention_messages(item.text)
                pending_message = item.message(LIVE_FOLLOWUP_PREFIX)
                request_turn.append(pending_message)
                request_turn.extend(mentions)
                if ImageInputs.input_refs(pending_message):
                    # A claimed queued attachment is a current image occurrence.
                    current.append(pending_message)
        # On a text-only route with [vision], observe the current image occurrences before the
        # main request: no doomed raw image is ever sent, and the observation is durable text.
        # Older accepted image history is never redescribed.
        current_raw = [message for message in current if ImageInputs.input_refs(message)]
        if self.session.image_route.delivery() == "vision" and current_raw:
            request_turn = self.session.images.observe_current(request_turn, current, self.model.vision_observe)
            if not pending:
                # No accept_pending_inputs will commit the copy later, so keep the live list in
                # sync: a failure after the paid observation must settle the converted message,
                # not the raw one whose observation text would be lost.
                turn_messages[:] = request_turn
        self.session.state.turn_messages = len(request_turn)
        tools = Tool.resolved_schemas(self.session)
        messages = self.context.prepare_messages(self.model, self.session.system_prompt, request_turn, tools)
        self.context.update_percent(messages, tools)
        return PreparedRequest(messages, tools, pending, request_turn, tuple(current_raw))

    def _main_request(
        self,
        request: PreparedRequest,
        turn_messages: list[Json],
    ) -> tuple[Json, list[ToolCall], str, PreparedRequest]:
        """Send one main-model request; an eligible 400 converts into exactly one vision fallback.

        Returns `(assistant, tool_calls, content, accepted)`, where `accepted` is the request
        whose acceptance commits the turn — the text-only retry when an eligible 400 recovered,
        otherwise `request` itself. The fallback retry cannot trigger another fallback: its route
        is now learned and it carries no current raw image, so both eligibility gates fail.
        """

        try:
            assistant, tool_calls, content = self.model.request(request.messages, request.tools)
            return assistant, tool_calls, content, request
        except ModelError as error:
            if not self._eligible_image_fallback(error, request):
                raise
            self.session.image_route.learn_text_only()
            vision_entry = self.session.config.vision_provider
            if not vision_entry:
                self._emit_image_route_notice("main model rejected image input (400); no vision provider configured")
                # No fallback is available: the original error propagates and the normal
                # replay-safe failure settlement runs, exactly as for any other rejected request.
                raise
            provider = self.session.config.providers[vision_entry]
            self._emit_image_route_notice(f"main model rejected image input (400); using {vision_entry}/{provider.model or '(empty)'}")
            # Observe the eligible current occurrences through [vision] and convert them to
            # durable text observations in the turn, then retry once without raw image blocks
            # (the route is now learned text-only, so projection suppresses every older raw
            # image as well, not only the failed occurrence).
            converted = self.session.images.observe_current(request.turn_messages, list(request.current_image_messages), self.model.vision_observe)
            turn_messages[:] = converted
            tools = Tool.resolved_schemas(self.session)
            messages = self.context.prepare_messages(self.model, self.session.system_prompt, turn_messages, tools)
            self.context.update_percent(messages, tools)
            retry = PreparedRequest(messages, tools, request.pending, turn_messages)
            self.session.state.turn_messages = len(turn_messages)
            # The successful fallback request is the one accepted for transaction ordering: current
            # and queued messages commit once, checkpoint once, and the loop continues normally —
            # no failed-turn marker is appended for a recovered 400.
            assistant, tool_calls, content = self.model.request(messages, tools)
            return assistant, tool_calls, content, retry

    def _eligible_image_fallback(self, error: ModelError, request: PreparedRequest) -> bool:
        """Whether a failed main request may learn text-only and fall back through [vision].

        All conditions must hold: the exhausted request failed with HTTP status exactly 400; it
        projected at least one current image occurrence as a raw image block; the active route
        was unknown when it was sent; and the request has not already consumed its one vision
        fallback (the retry fails the first two gates, so no explicit budget flag is needed).
        """

        if resilience.error_status(error) != 400:
            return False
        if not request.current_image_messages:
            return False
        return self.session.image_route.state() == IMAGE_ROUTE_UNKNOWN

    def _emit_image_route_notice(self, text: str) -> None:
        """Publish one gray, non-model routing notice; never enters model context."""

        if self.on_image_route_notice is not None:
            self.on_image_route_notice(text)

    def mention_messages(self, text: str) -> list[Json]:
        """Session-event context blocks attached after one user message, initial or queued."""
        blocks: list[Json] = []
        for event, resolver in (
            ("mcp_mentions", self.session.mcp.resolve_mentions if self.session.mcp is not None else None),
            ("skill_mentions", self.session.skills.resolve_mentions if self.session.skills is not None else None),
            ("file_mentions", self.session.mentions.resolve_mentions if self.session.mentions is not None else None),
        ):
            content = resolver(text) if resolver is not None else ""
            if content:
                # Expansions are not new requests. Marking them keeps compaction's latest-user
                # boundary on the raw message that caused them, including queued follow-ups.
                blocks.append({"role": "user", "content": content, SESSION_EVENT_KEY: event})
        return blocks

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
    def malformed_tool_call_error(names: list[str]) -> MalformedToolCallError:
        count = len(names)
        if len(set(names)) == 1:
            return MalformedToolCallError(f"Model emitted {names[0]} as text {count} times; none of the textual calls were executed.")
        sequence = ", then ".join(names)
        return MalformedToolCallError(f"Model emitted tool calls as text {count} times ({sequence}); none of the textual calls were executed.")

    def accept_pending_inputs(
        self,
        turn_messages: list[Json],
        transcript_messages: list[Json],
        pending: list[QueuedInput],
        prepared_turn_messages: list[Json],
    ) -> None:
        # A request reached the provider: whatever current image occurrences it carried are no
        # longer current, whether or not queued input was part of it.
        self._current_image_messages = []
        if not pending:
            return
        texts = [item.text for item in pending]
        # Committed with the marker the provider was sent, not the bare text: dropping it here would
        # rewrite a message already in the prefix and leave the model's acknowledgement unexplained.
        messages = [item.message(LIVE_FOLLOWUP_PREFIX) for item in pending]
        turn_messages[:] = prepared_turn_messages
        transcript_messages.extend(SessionSnapshotCodec.transcript_messages(messages))
        self.session.acknowledge_user_inputs(pending)
        if self.on_queue_flush:
            self.on_queue_flush(texts)

    @staticmethod
    def assistant_turn_message(assistant: Json, tool_calls: list[ToolCall], content: str) -> Json:
        message = dict(assistant or {})
        message["role"] = "assistant"
        message["content"] = message.get("content") if message.get("content") is not None else (content.strip() or None)
        if not tool_calls:
            message.pop("tool_calls", None)
        elif not message.get("tool_calls"):
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps({"args": call.args}, ensure_ascii=False)}}
                for call in tool_calls
            ]
        return Text.value(message)
