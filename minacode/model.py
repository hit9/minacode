"""minacode model client: provider request protocols, streaming, and retry policy."""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import json
import random
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from json_repair import repair_json

from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    HTTP_USER_AGENT,
    MODEL_REQUEST_RETRIES,
    PAUSED_TURN_KEY,
    PROVIDER_ECHO_KEYS,
    RESPONSES_OUTPUT_KEY,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    SEARCH_SOURCES_KEY,
    SESSION_EVENT_KEY,
    ActiveResource,
    Json,
    ModelError,
    ModelOutputTruncated,
    ModelRequestRetry,
    ModelResponseTimeout,
    ModelUsage,
    ProviderConfig,
    Text,
    ToolArgs,
    ToolCall,
    ToolError,
    builtin_tool_label,
    request_budget_for,
)
from minacode.image import IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, ImageInputs
from minacode.model_catalog import THINKING_BUDGETS
from minacode.prompts import (
    COMPACTION_PROMPT,
)
from minacode.provider_compat import (
    ResolvedProvider,
    anthropic_keeps_prior_thinking,
    anthropic_thinking_always_on,
    anthropic_thinking_params,
    builtin_tools_issue,
)

if TYPE_CHECKING:
    # The provider SDKs cost ~0.8s to import and are not needed until the first request;
    # the runtime imports below keep them off the startup path (see MCPManager for the same pattern).
    from anthropic import Anthropic
    from openai import OpenAI

from minacode.session import AgentState, QueuedInput, Session
from minacode.tools import (
    TOOL_REGISTRY,
    Tool,
)

_ResultT = TypeVar("_ResultT")

# Retry-wait granularity: sleeping in ~0.1s slices lets the wait observe the UI-thread cancel flag
# instead of relying on a signal interrupting one long sleep.
_RETRY_SLEEP_SLICE = 0.1
_COMPACTION_RESPONSE_TIMEOUT = 60.0


@dataclass(frozen=True)
class PreparedRequest:
    messages: list[Json]
    tools: list[Json]
    pending: list[QueuedInput]


class ModelClient:
    """Send one request over the selected provider protocol and normalize the reply.

    Chat Completions, Responses, and Anthropic Messages all return the same (assistant message, tool
    calls, text) triple, so callers never learn which ran. History stays one normalized model;
    continuation data such as reasoning blocks round-trips through namespaced opaque fields, because
    providers verify that what they produced comes back unchanged — flattening it into text breaks
    the next request.

    Retries are invisible to the caller: bounded backoff on transport and 5xx failures, with progress
    published through session state for the status bar. A missing model or a refused modality is a
    decision rather than a glitch and surfaces at once. Streaming is the same call, not a second path.

    Cancelling closes the in-flight client, so a blocked read ends instead of waiting out its timeout.
    """

    _RETRYABLE_STATUS_RE: ClassVar[re.Pattern] = re.compile(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(408|409|425|429|5\d\d)\b")
    _STATUS_CODE_RE: ClassVar[re.Pattern] = re.compile(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(4\d\d|5\d\d)\b")
    _JSON_FENCE_RE: ClassVar[re.Pattern] = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)

    def __init__(self, session: Session):
        self.session = session
        self.cancel_requested = threading.Event()
        self.active_client: ActiveResource[OpenAI | Anthropic] = ActiveResource()
        self.on_stream: Callable[[str, str], None] | None = None
        # Called with (label, detail) for each provider-side tool call a response reports. Reported
        # from the parsed result rather than the stream, so a search is logged the same way when
        # streaming is off and on a frontend that shows no live status at all.
        self.on_builtin_call: Callable[[str, str], None] | None = None
        # Lifecycle hook, mirroring ContextManager.on_compaction: True while a retry backoff wait is in
        # progress, False in a finally block. Lets the orchestration label the phase without model
        # depending on a renderer.
        self.on_retry_wait: Callable[[bool], None] | None = None

    def cancel(self) -> None:
        self.cancel_requested.set()
        with contextlib.suppress(Exception):
            self.active_client.apply(lambda client: client.close())

    def chat_messages(self, messages: list[Json]) -> list[Json]:
        """Build Chat Completions history using the provider's documented replay contract."""

        provider = self.session.config.provider
        resolved = provider.resolve()
        history = resolved.chat_reasoning_history
        thinking = provider.extra_body.get("thinking")
        if provider.extra_body.get("preserve_thinking") is True or (
            isinstance(thinking, dict) and (thinking.get("keep") == "all" or thinking.get("clear_thinking") is False)
        ):
            history = "all"

        converted: list[Json] = []
        latest_user = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user" and not ImageInputs.is_tool_observation(message)),
            default=-1,
        )
        for index, message in enumerate(messages):
            clean = {
                key: value for key, value in message.items() if key not in (*PROVIDER_ECHO_KEYS, IMAGE_REFS_KEY, TOOL_IMAGE_OBSERVATION_KEY, SESSION_EVENT_KEY)
            }
            keep_reasoning = history == "all" or (
                bool(message.get("tool_calls")) and (history == "tool_calls" or (history == "current_turn" and index > latest_user))
            )
            if message.get("role") == "assistant" and not keep_reasoning:
                for key in ("reasoning_content", "reasoning", "reasoning_details"):
                    clean.pop(key, None)
            if message.get("role") == "user" and self.session.images.refs(message):
                clean["content"] = self.session.images.chat_content(message)
            converted.append(clean)
        return Text.value(converted)

    def estimated_request_tokens(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        """Estimate the actual protocol payload instead of minacode's normalized history."""

        resolved = self.session.config.provider.resolve()
        api = resolved.api
        # Measuring a payload must never fail on it: an entry this wire rejects is the request's
        # error to raise, not something that should break the status bar, /status, or resume.
        builtin = self.builtin_tools(resolved, strict=False)
        # Payload builders would otherwise expand every local image to base64 merely to throw the
        # bytes away below. Labels preserve the surrounding wire shape; image tiles are added once.
        projected = [{key: value for key, value in message.items() if key != IMAGE_REFS_KEY} for message in messages]
        if api == "responses":
            payload: Json = {"input": self.responses_input(Text.value(projected))}
            if request_tools := [*self.responses_tool_schemas(tools or []), *builtin]:
                payload["tools"] = request_tools
        elif api == "anthropic":
            system = "\n\n".join(str(message.get("content") or "") for message in projected if message.get("role") == "system").strip()
            estimated_messages = projected
            if not anthropic_keeps_prior_thinking(self.session.config.provider.model):
                latest_user = max(
                    (index for index, message in enumerate(projected) if message.get("role") == "user" and not ImageInputs.is_tool_observation(message)),
                    default=-1,
                )
                active_assistants = [index for index, message in enumerate(projected) if index > latest_user and message.get("role") == "assistant"]
                keep_from = (
                    latest_user
                    if active_assistants
                    else max((index for index, message in enumerate(projected) if message.get("role") == "assistant"), default=len(projected))
                )
                estimated_messages = []
                for index, message in enumerate(projected):
                    estimated = dict(message)
                    saved = estimated.get(ANTHROPIC_CONTENT_KEY)
                    if index < keep_from and isinstance(saved, list):
                        estimated[ANTHROPIC_CONTENT_KEY] = [
                            block for block in saved if not isinstance(block, dict) or block.get("type") not in ("thinking", "redacted_thinking")
                        ]
                    estimated_messages.append(estimated)
            payload = {"system": system, "messages": self.anthropic_messages(Text.value(estimated_messages))}
            if request_tools := [*self.anthropic_tool_schemas(tools or []), *builtin]:
                payload["tools"] = request_tools
        else:
            payload = {"messages": self.chat_messages(projected)}
            if request_tools := [*(tools or []), *builtin]:
                payload["tools"] = request_tools

        def prompt_value(value: object) -> object:
            if isinstance(value, list):
                return [prompt_value(item) for item in value]
            if not isinstance(value, dict):
                return value
            kind = value.get("type")
            clean: Json = {}
            for key, item in value.items():
                if key in ("encrypted_content", "signature"):
                    continue
                if key == "data" and kind in ("reasoning.encrypted", "redacted_thinking"):
                    continue
                if (key == "data" and kind == "base64") or (key in ("image_url", "url") and isinstance(item, str) and item.startswith("data:")):
                    clean[key] = ""
                else:
                    clean[key] = prompt_value(item)
            return clean

        chars = len(json.dumps(prompt_value(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        images = ImageInputs.estimated_tokens(messages) if self.session.images.support() is not False else 0
        return (chars + 3) // 4 + images

    def call_client(self, client: OpenAI | Anthropic, request: Callable[[], _ResultT], *, response_timeout: float | None = None) -> _ResultT:
        response_timeout = self.session.config.provider.response_timeout if response_timeout is None else response_timeout
        outcome: list[tuple[bool, object]] = []
        completed = threading.Event()

        def invoke() -> None:
            try:
                outcome.append((True, request()))
            except BaseException as error:  # noqa: BLE001 - preserve the provider call's exact outcome across the deadline thread boundary.
                outcome.append((False, error))
            finally:
                completed.set()

        worker = threading.Thread(target=invoke, name="model-request", daemon=True)
        with self.active_client.track(client):
            worker.start()
            try:
                deadline = time.monotonic() + response_timeout if response_timeout else 0.0
                while not completed.wait(_RETRY_SLEEP_SLICE):
                    if self.cancel_requested.is_set():
                        raise KeyboardInterrupt
                    if deadline and time.monotonic() >= deadline:
                        raise ModelResponseTimeout(
                            f"Model response exceeded provider.response_timeout={response_timeout:g}s; set it to 0 to disable the total-generation limit"
                        )
                succeeded, value = outcome[0]
                if not succeeded:
                    raise cast(BaseException, value)
                if self.cancel_requested.is_set():
                    raise KeyboardInterrupt
                return cast(_ResultT, value)
            except ModelResponseTimeout:
                raise
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
                        provider = self.session.config.provider
                        identity = f"{self.session.config.active_provider}/{provider.model or '(no model)'}"
                        raise ModelError(
                            f"{identity} does not support image input. Switch to an image-capable model, or continue with image labels only."
                        ) from error
                    retryable = self.retryable_error(error)
                    if attempt >= MODEL_REQUEST_RETRIES or not retryable:
                        if attempt:
                            raise ModelError(f"{error} (after {attempt + 1} attempts)") from error
                        raise
                    state.current_model_attempt = attempt + 2
                    state.model_retry_reason = self.retry_reason(error)
                    state.model_retry_count += 1
                    self._wait_before_retry(self.retry_delay(error, attempt), state)
                finally:
                    state.current_model_call_started_at = 0.0
                attempt += 1
        finally:
            state.current_model_attempt = 0
            state.model_retry_reason = ""

    def retry_delay(self, error: Exception, attempt: int) -> float:
        """Single-wait pacing: provider Retry-After wins when parseable, else exponential backoff + jitter."""
        if (retry_after := self.retry_after_delay(error)) is not None:
            return retry_after
        delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * 2**attempt)
        # jitter 0.5x-1.5x is not optional: without it, parallel read-only tool batches (and worker vs
        # parent requests) would retry in lockstep and spike the provider exactly when it is weakest.
        return delay * (0.5 + random.random())

    def _wait_before_retry(self, delay: float, state: AgentState) -> None:
        """Sleep in ~0.1s slices, watching the UI-thread cancel signal, and publish the wait as facts:
        model_retry_until (monotonic deadline, the renderer formats it) and the on_retry_wait phase
        hook. The retry decision is unchanged (see retryable_error); only the pacing is here."""
        on_retry_wait = self.on_retry_wait
        if on_retry_wait is not None:
            on_retry_wait(True)
        try:
            state.model_retry_until = time.monotonic() + delay
            deadline = state.model_retry_until
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(_RETRY_SLEEP_SLICE, remaining))
                # Cancellation is a control signal from the UI thread; a signal interrupting
                # time.sleep is not guaranteed. A /resend racing the start of this wait remains a
                # retry request, not a user cancellation of the whole turn.
                if self.cancel_requested.is_set():
                    if state.manual_model_retry_requested:
                        state.manual_model_retry_requested = False
                        raise ModelRequestRetry() from None
                    raise KeyboardInterrupt
        finally:
            state.model_retry_until = 0.0
            if on_retry_wait is not None:
                on_retry_wait(False)

    @staticmethod
    def retryable_error(error: Exception) -> bool:
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        import anthropic
        import httpx
        import openai

        # A truncated generation is deterministic: the same request hits the same output cap again.
        if isinstance(error, (ModelResponseTimeout, ModelOutputTruncated)):
            return False
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

        # Streaming reads surface httpx transport errors unwrapped: the provider SDKs' Stream.__stream__
        # iterates the response directly and re-raises httpx failures (a dropped connection mid-stream is
        # httpx.ReadError, an interrupted chunked body is httpx.RemoteProtocolError) rather than wrapping
        # them as APIConnectionError. They're the same class of transient failure. httpx transport errors
        # don't inherit OSError, so the isinstance above misses them.
        if isinstance(cause, httpx.TransportError):
            return True

        # Fallback: parse status codes embedded in the error text or cause attributes.
        status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
        with contextlib.suppress(Exception):
            if int(status) in {408, 409, 425, 429, 500, 502, 503, 504}:
                return True
        text = str(error).lower()
        if ModelClient._RETRYABLE_STATUS_RE.search(text):
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
        match = ModelClient._STATUS_CODE_RE.search(text)
        if match:
            return match.group(1)
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(part in text for part in ("connection", "reset", "aborted")):
            return "connection"
        if "internal server error" in text or "temporarily unavailable" in text:
            return "server error"
        return "transient error"

    @staticmethod
    def retry_after_delay(error: Exception) -> float | None:
        """Provider Retry-After (seconds or HTTP-date) from the SDK cause, clamped to RETRY_MAX_DELAY.

        Returns None to fall back to the backoff algorithm when the header is missing, empty,
        malformed, or negative. Any parsed value is clamped so a single aberrant header cannot stall
        the CLI for minutes; the retry decision itself is unchanged (see retryable_error)."""
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        import anthropic
        import openai

        cause = getattr(error, "__cause__", None)
        if not isinstance(cause, (openai.APIStatusError, anthropic.APIStatusError)):
            return None
        headers = getattr(cause.response, "headers", None) or {}
        value = headers.get("retry-after")
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        text = str(value).strip()
        if not text:
            return None
        try:
            seconds = int(text)
        except ValueError:
            # HTTP-date form (RFC 7231; no zone means GMT). parsedate_to_datetime raises on some
            # malformed inputs instead of returning None, so treat either outcome as "unparseable".
            try:
                parsed = email.utils.parsedate_to_datetime(text)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed is None:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            seconds = (parsed - datetime.now(UTC)).total_seconds()
        if seconds < 0:
            return None
        return min(seconds, RETRY_MAX_DELAY)

    def truncated_output_error(self, usage: Any) -> ModelOutputTruncated:
        """Report a generation the provider cut off at the output cap before it produced anything.

        Reasoning counts against that cap on the Responses and Anthropic wires, so a high effort can
        consume the whole budget and return neither text nor a tool call. The turn then fails with
        nothing to show, which is indistinguishable from an empty answer unless the cap is named.
        A truncation that still carried text is left alone: the partial answer is visible, and the
        cut is its own evidence."""
        provider = self.session.config.provider
        cap = f"provider.max_tokens={provider.max_tokens}" if provider.max_tokens > 0 else "the provider's own default output limit"
        completion = ModelUsage.field(usage, "completion_tokens", "output_tokens")
        reasoning = ModelUsage.field(usage, "completion_tokens_details.reasoning_tokens", "output_tokens_details.reasoning_tokens")
        spent = f" after {completion} output tokens" if completion else ""
        spent += f" ({reasoning} of them reasoning)" if reasoning else ""
        return ModelOutputTruncated(f"Model output was truncated at {cap}{spent}. Raise provider.max_tokens, or lower provider.reasoning.")

    def empty_length_error(self, usage: Any) -> ModelError:
        """`finish_reason=length` with nothing generated on the Chat wire is ambiguous: the output may
        have hit the cap, or the input may have exceeded the model's context window (some
        OpenAI-compatible providers report the latter as `length` too). Only the cap case is verifiable
        from usage; anything else names both settings instead of pushing max_tokens blindly."""
        provider = self.session.config.provider
        completion = ModelUsage.field(usage, "completion_tokens", "output_tokens")
        if provider.max_tokens > 0 and completion >= provider.max_tokens:
            return self.truncated_output_error(usage)
        cap = f"provider.max_tokens={provider.max_tokens}" if provider.max_tokens > 0 else "the provider's default output cap"
        spent = f" after {completion} output tokens" if completion else ""
        return ModelError(
            f"Generation stopped empty with `finish_reason=length`{spent}: either the output hit {cap} or "
            f"the input exceeded the model's context window. Check provider.max_tokens and runtime.max_context_tokens."
        )

    def _record_usage(self, usage: Any) -> None:
        """Add a completed request to session usage, keeping the budget it was prepared against so the
        status fill uses the request-time denominator instead of today's configuration."""
        self.session.usage.add(
            usage,
            request_budget_for(self.session.settings.max_context_tokens, self.session.config.provider.output_token_budget()),
        )

    def chat_request(
        self, messages: list[Json], tools: list[Json] | None = None, *, allow_stream: bool = True, response_timeout: float | None = None
    ) -> tuple[Json, list[ToolCall], str]:
        messages = self.chat_messages(messages)
        provider = self.session.config.provider
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self.on_stream is not None
        params: Json = {"model": provider.model, "messages": messages, "stream": stream}
        if provider.max_tokens > 0:
            params["max_tokens"] = provider.max_tokens
        if request_tools := [*(tools or []), *self.builtin_tools(resolved)]:
            params["tools"] = request_tools
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
            message, usage, finish_reason = self.call_client(client, lambda: self._chat_stream(client, params), response_timeout=response_timeout)
        else:
            response = self.call_client(client, lambda: client.chat.completions.create(**params), response_timeout=response_timeout)
            usage = getattr(response, "usage", None)
            message = response.choices[0].message
            finish_reason = str(self.message_field(response.choices[0], "finish_reason") or "")
        self._record_usage(usage)
        assistant = self.assistant_message(message)
        calls = self.tool_calls(message)
        content = str(self.message_field(message, "content") or "")
        # Raised outside call_client, which flattens every exception into a plain ModelError.
        if finish_reason == "length" and not calls and not content.strip():
            raise self.empty_length_error(usage)
        return assistant, calls, content

    def _chat_stream(self, client: OpenAI, params: Json) -> tuple[Json, Any, str]:
        """Reassemble a streamed chat completion into one assistant message and its finish reason.

        Tool calls are the hard part. The spec streams them as deltas keyed by `index`, but providers
        variously omit it, restart it, or send only `id`. `resolve_tool_call_index` recovers the
        association from whatever a chunk carries, in decreasing order of reliability, and raises
        instead of guessing when nothing identifies the call: a wrong association concatenates two
        calls' argument fragments into one call with corrupt JSON, which the model cannot correct
        because it looks like something it wrote.

        Unlike Responses, Chat has no separate text-done event. Do not promote on the first tool
        delta: compatible providers can vary their delta order. `finish_reason=tool_calls` is the
        first protocol boundary that proves this assistant message is complete.
        """
        content: list[str] = []
        reasoning_content: list[str] = []
        reasoning: list[str] = []
        reasoning_details: list[Json] = []
        tool_calls: dict[int, Json] = {}
        tool_call_functions: dict[int, Json] = {}
        tool_call_ids: dict[str, int] = {}
        tool_call_positions: dict[int, int] = {}
        next_index = 0
        usage: Any = None
        output_promoted = False
        finish_reason = ""

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
                choice = choices[0]
                delta = self.message_field(choice, "delta")
                reasoning_content_delta = str(self.message_field(delta, "reasoning_content") or "")
                reasoning_delta = str(self.message_field(delta, "reasoning") or "")
                if reasoning_content_delta:
                    reasoning_content.append(reasoning_content_delta)
                    self._emit_stream("reasoning", reasoning_content_delta)
                elif reasoning_delta:
                    reasoning.append(reasoning_delta)
                    self._emit_stream("reasoning", reasoning_delta)
                raw_details = self.message_field(delta, "reasoning_details") or []
                details = [self.dump_message_item(item) for item in raw_details]
                reasoning_details.extend(item for item in details if item)
                if not reasoning_content_delta and not reasoning_delta:
                    for detail in details:
                        text = detail.get("text") if detail.get("type") == "reasoning.text" else detail.get("summary")
                        if text:
                            self._emit_stream("reasoning", str(text))
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
                if chunk_finish_reason := str(self.message_field(choice, "finish_reason") or ""):
                    finish_reason = chunk_finish_reason
                if finish_reason == "tool_calls" and content and tool_calls and not output_promoted:
                    self._emit_stream("output_done", "".join(content))
                    output_promoted = True
        finally:
            self._emit_stream("", "")
        message: Json = {"content": "".join(content) or None}
        if reasoning_content:
            message["reasoning_content"] = "".join(reasoning_content)
        if reasoning:
            message["reasoning"] = "".join(reasoning)
        if reasoning_details:
            # OpenRouter defines the complete sequence as the ordered concatenation of each
            # delta's reasoning_details array; replay it unchanged on the assistant message.
            # Evidence: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
            message["reasoning_details"] = reasoning_details
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return message, usage, finish_reason

    def api_request(
        self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True, response_timeout: float | None = None
    ) -> tuple[Json, list[ToolCall], str]:
        api = self.session.config.provider.resolve().api
        if api == "anthropic":
            request = self.anthropic_request
        elif api == "responses":
            request = self.responses_request
        else:
            request = self.chat_request
        if allow_stream and response_timeout is None:
            return request(messages, tools)
        return request(messages, tools, allow_stream=allow_stream, response_timeout=response_timeout)

    def responses_request(
        self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True, response_timeout: float | None = None
    ) -> tuple[Json, list[ToolCall], str]:
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
        if request_tools := [*self.responses_tool_schemas(tools or []), *self.builtin_tools(resolved)]:
            params["tools"] = request_tools
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
        if stream:
            result = self.call_client(client, lambda: self._responses_stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self.call_client(client, lambda: client.responses.create(**params), response_timeout=response_timeout)
            streamed = False
        self._record_usage(self.message_field(result, "usage"))
        return self.responses_result(result, streamed)

    def _responses_stream(self, client: OpenAI, params: Json) -> Any:
        """Consume a Responses stream, promoting completed text before tool arguments finish.

        Text completion and function-call discovery are independent events and either can arrive
        first. Promotion is therefore a two-condition state transition, not an ordering assumption;
        the terminal response is still consumed normally for history, tool calls, and usage.
        """

        terminal: Any = None
        output: list[str] = []
        text_done = handoff_seen = output_promoted = False

        def promote_output() -> None:
            nonlocal output_promoted
            if text_done and handoff_seen and output and not output_promoted:
                self._emit_stream("output_done", "".join(output))
                output_promoted = True

        try:
            for event in client.responses.create(**params):
                event_type = str(self.message_field(event, "type") or "")
                # Two spellings of the same event: hosts that summarize reasoning stream the summary,
                # hosts that expose the raw chain stream the text. DeepSeek only ever sends the
                # latter and documents that it generates no summary at all, so listening for one
                # spelling leaves a thinking model with no preview.
                # Evidence: https://api-docs.deepseek.com/guides/responses_api
                if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    self._emit_stream("reasoning", str(self.message_field(event, "delta") or ""))
                elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                    delta = str(self.message_field(event, "delta") or "")
                    output.append(delta)
                    self._emit_stream("output", delta)
                elif event_type in ("response.output_text.done", "response.refusal.done"):
                    text_done = True
                    promote_output()
                elif event_type == "response.output_item.added":
                    item = self.message_field(event, "item")
                    item_type = str(self.message_field(item, "type") or "")
                    if item_type == "function_call":
                        handoff_seen = True
                        promote_output()
                    elif item_type.endswith("_call"):
                        # A provider-side tool is the same durable tool boundary as a local function
                        # call: completed text before it must be handed off now, so the live status
                        # below never covers a finished answer. A provider-side tool also runs inside
                        # the request with no local tool line to show for it, so the status label is
                        # the only sign the turn is still moving.
                        handoff_seen = True
                        promote_output()
                        self._emit_stream(builtin_tool_label(item_type), "")
                elif event_type == "response.output_item.done":
                    item = self.message_field(event, "item")
                    item_type = str(self.message_field(item, "type") or "")
                    # A provider-side call has no local tool line of its own, so report it the moment
                    # the stream completes it and the transcript shows it live. The stream and the
                    # terminal output carry the same calls, so the parsed-result scan stays silent on
                    # streaming requests; reporting here is the one and only record for them.
                    if item_type.endswith("_call") and item_type != "function_call":
                        # Some compatible providers omit the matching output_item.added event, so the
                        # durable report below must also establish the promotion boundary itself.
                        handoff_seen = True
                        promote_output()
                        action = self.message_field(item, "action")
                        query = self.message_field(action, "query") if action is not None else ""
                        self.report_builtin_call(item_type, str(query or ""))
                elif event_type == "response.function_call_arguments.delta":
                    handoff_seen = True
                    promote_output()
                elif event_type in ("response.completed", "response.incomplete"):
                    # Compatible providers may omit response.output_text.done; the accepted terminal
                    # response proves the streamed text is final, so it is the terminal fallback for
                    # text completion. The tool boundary guard keeps plain responses unpromoted.
                    text_done = True
                    promote_output()
                    terminal = self.message_field(event, "response")
                elif event_type == "response.failed":
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
        seen_output_ids: set[str] = set()
        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            saved_output = message.get(RESPONSES_OUTPUT_KEY)
            if role == "assistant" and isinstance(saved_output, list):
                for item in saved_output:
                    if not isinstance(item, dict) or not self.replayable_output_item(item):
                        continue
                    if content is None and item.get("type") == "message":
                        continue
                    item_id = str(item.get("id") or "")
                    if item_id and item_id in seen_output_ids:
                        continue
                    if item_id:
                        seen_output_ids.add(item_id)
                    converted.append(item)
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

    def responses_result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
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
        # Streaming already reported every provider-side call live, and the stream and the terminal
        # output carry the same calls, so scanning again would double each one — and a call without an
        # id could not be de-duplicated at all. The scan is the only source for non-streaming requests.
        if not streamed:
            for item in saved_output:
                item_type = str(item.get("type") or "")
                if item_type.endswith("_call") and item_type != "function_call":
                    action = item.get("action")
                    query = action.get("query") if isinstance(action, dict) else ""
                    self.report_builtin_call(item_type, query if isinstance(query, str) else "")
        if not calls and not text.strip() and self.message_field(result, "status") == "incomplete":
            details = self.message_field(result, "incomplete_details")
            if self.message_field(details, "reason") == "max_output_tokens":
                raise self.truncated_output_error(self.message_field(result, "usage"))
        assistant: Json = {"role": "assistant", "content": text or None, RESPONSES_OUTPUT_KEY: saved_output}
        if sources := self.responses_sources(saved_output):
            assistant[SEARCH_SOURCES_KEY] = sources
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, calls, text

    @classmethod
    def responses_sources(cls, saved_output: list[Json]) -> list[Json]:
        """Sources a Responses host attached to one response.

        Two hosts, two places: OpenAI cites inline through `url_citation` annotations on the
        message, while Qwen returns no citations at all and reports sources only on the search
        call. Reading both keeps one renderer honest across them."""
        groups: list[Any] = []
        for item in saved_output:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict):
                        groups.append(part.get("annotations"))
                continue
            action = item.get("action")
            groups.append(action.get("sources") if isinstance(action, dict) else None)
            groups.append(item.get("results"))
        return cls.collect_sources(*groups)

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
        messages = [{"role": "system", "content": COMPACTION_PROMPT}, {"role": "user", "content": Text.clean(context)}]
        configured_timeout = self.session.config.provider.response_timeout
        response_timeout = min(configured_timeout, _COMPACTION_RESPONSE_TIMEOUT) if configured_timeout else _COMPACTION_RESPONSE_TIMEOUT
        _, _, content = self.api_request(messages, None, allow_stream=False, response_timeout=response_timeout)
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
        match = ModelClient._JSON_FENCE_RE.match(text)
        return (match.group(1) if match else text).strip()

    def client(self) -> OpenAI:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        from openai import OpenAI

        return OpenAI(
            api_key=provider.key, base_url=provider.resolve().base_url, timeout=provider.timeout, max_retries=0, default_headers={"User-Agent": HTTP_USER_AGENT}
        )

    def anthropic_client(self) -> Anthropic:
        provider = self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        url = provider.resolve().base_url.rstrip("/")
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        from anthropic import Anthropic

        return Anthropic(
            api_key=provider.key,
            base_url=url.removesuffix("/v1"),
            timeout=provider.timeout,
            max_retries=0,
            default_headers={"User-Agent": HTTP_USER_AGENT},
        )

    def report_builtin_call(self, name: str, detail: object) -> None:
        if self.on_builtin_call is not None:
            self.on_builtin_call(builtin_tool_label(name), str(detail or "").strip())

    @staticmethod
    def collect_sources(*groups: Any) -> list[Json]:
        """Flatten provider-side search sources into `{"url", "title"}` records, first mention wins.

        Every host reports the same two facts under a different name, so the shapes are normalized
        here rather than at each call site. A record without a URL is dropped: it cannot be shown
        as a source, and a title alone would suggest attribution that isn't there."""
        sources: dict[str, Json] = {}
        for group in groups:
            for raw in group or []:
                item = raw if isinstance(raw, dict) else ModelClient.dump_message_item(raw)
                if not isinstance(item, dict):
                    continue
                # OpenAI and OpenRouter nest the fields one level down under `url_citation`.
                nested = item.get("url_citation")
                if isinstance(nested, dict):
                    item = nested
                url = str(item.get("url") or "")
                if url and url not in sources:
                    sources[url] = {"url": url, "title": str(item.get("title") or "")}
        return list(sources.values())

    def builtin_tools(self, resolved: ResolvedProvider | None = None, *, strict: bool = True) -> list[Json]:
        """Provider-side tool entries, copied so a request cannot mutate the loaded config.

        These reach every protocol's `tools` array unchanged. Each host expresses its builtin
        tools in the shape of the active protocol — including OpenRouter on both Chat and Responses
        — so one pass-through serves all of them. Qwen Chat configures search through
        `extra_body.enable_search` instead.

        Documented providers restrict which wire may carry each provider-native entry. Entries
        outside that wire stay configured but inactive, so switching models never requires
        destructive config edits. A malformed or unsupported entry on the active wire still fails
        locally. Unknown hosts keep the generic pass-through.

        ``strict=False`` reports the same entries without raising, for read-only accounting such as
        token estimation: refusing an unsupported entry belongs to the request that would send it,
        not to the status bar, `/status`, or the resume that merely measures the payload.
        """

        provider = self.session.config.provider
        entries = provider.builtin_tools
        if not entries:
            return []
        resolved = resolved or provider.resolve()
        issue = builtin_tools_issue(resolved, entries)
        if issue is not None:
            if issue.reason == "wire":
                return []
            if not strict:
                return [dict(entry) for entry in entries]
            raise ModelError(
                f"provider.builtin_tools {', '.join(issue.configured)} are not supported on the {resolved.api} wire "
                f"for {provider.model or '(no model)'} ({resolved.host or 'this provider'}) yet; "
                f"supported provider tools: {', '.join(issue.supported_entries) or '(none)'}"
            )
        return [dict(entry) for entry in entries]

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
        # Builtin tools are part of the cached prefix too: enabling search changes the tool block
        # the host renders ahead of the system prompt, so it must change the cache key with it.
        tool_names.extend(str(entry.get("type") or "(unknown)") for entry in self.builtin_tools(resolved))
        payload = {
            "api": resolved.api,
            "cwd": self.session.cwd,
            "host": resolved.host,
            "model": provider.model,
            "tools": ",".join(sorted(tool_names)) or "(none)",
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return "minacode-" + digest[:24]

    def anthropic_request(
        self, messages: list[Json], tools: list[Json] | None, *, allow_stream: bool = True, response_timeout: float | None = None
    ) -> tuple[Json, list[ToolCall], str]:
        messages = Text.value(messages)
        params = self.anthropic_params(messages, tools)
        client = self.anthropic_client()
        stream = allow_stream and self.session.config.provider.stream and self.on_stream is not None
        if stream:
            result = self.call_client(client, lambda: self._anthropic_stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self.call_client(client, lambda: client.messages.create(**params), response_timeout=response_timeout)
            streamed = False
        self._record_usage(self.message_field(result, "usage"))
        assistant, calls, content = self.anthropic_result(result, streamed)
        return assistant, calls, content

    def _anthropic_stream(self, client: Anthropic, params: Json) -> Any:
        """Consume Messages blocks and promote text once both text and tool blocks are known.

        Content blocks need not put text before `tool_use`, so block start/stop events feed the same
        order-independent transition as Responses. Input JSON may continue after promotion when the
        completed text block came first.
        """
        output: list[str] = []
        text_blocks: set[int] = set()
        server_tools: dict[int, dict[str, str]] = {}
        text_done = handoff_seen = output_promoted = False

        def promote_output() -> None:
            nonlocal output_promoted
            if text_done and handoff_seen and output and not output_promoted:
                self._emit_stream("output_done", "".join(output))
                output_promoted = True

        try:
            with client.messages.stream(**params) as stream:
                for event in stream:
                    event_type = self.message_field(event, "type")
                    if event_type == "content_block_start":
                        block = self.message_field(event, "content_block")
                        block_type = self.message_field(block, "type")
                        if block_type == "text":
                            text_blocks.add(int(self.message_field(event, "index") or 0))
                        elif block_type == "tool_use":
                            handoff_seen = True
                            promote_output()
                        elif block_type == "server_tool_use":
                            # A provider-side tool is the same durable tool boundary as a local
                            # tool_use: completed text before it is final and must be handed off now,
                            # before the live status below covers the preview.
                            handoff_seen = True
                            promote_output()
                            self._emit_stream(builtin_tool_label(str(self.message_field(block, "name") or "")), "")
                            # The query streams in via input_json_delta and is only whole at content_block_stop,
                            # so register the block now and report it there, showing the search in the transcript live.
                            # Some hosts put the whole input on content_block_start instead of streaming
                            # it via input_json_delta; keep that query as the fallback the stop handler
                            # uses when no partial_json ever arrived.
                            start_input = self.message_field(block, "input")
                            server_tools[int(self.message_field(event, "index") or 0)] = {
                                "id": str(self.message_field(block, "id") or ""),
                                "name": str(self.message_field(block, "name") or ""),
                                "json": "",
                                "query": str(start_input.get("query") or "") if isinstance(start_input, dict) else "",
                            }
                        continue
                    if event_type == "content_block_stop":
                        index = int(self.message_field(event, "index") or 0)
                        if index in text_blocks:
                            text_done = True
                            promote_output()
                        elif index in server_tools:
                            info = server_tools.pop(index)
                            # Defensively establish the boundary here too; the durable report below
                            # must not be the first durable tool signal after a finished text block.
                            handoff_seen = True
                            promote_output()
                            query = info["query"]
                            if info["json"]:
                                with contextlib.suppress(json.JSONDecodeError):
                                    parsed = json.loads(info["json"])
                                    if isinstance(parsed, dict) and parsed.get("query"):
                                        query = str(parsed["query"])
                            self.report_builtin_call(info["name"], query)
                        continue
                    if event_type != "content_block_delta":
                        continue
                    delta = self.message_field(event, "delta")
                    delta_type = self.message_field(delta, "type")
                    if delta_type == "thinking_delta":
                        self._emit_stream("reasoning", str(self.message_field(delta, "thinking") or ""))
                    elif delta_type == "text_delta":
                        text = str(self.message_field(delta, "text") or "")
                        output.append(text)
                        self._emit_stream("output", text)
                    elif delta_type == "input_json_delta":
                        index = int(self.message_field(event, "index") or 0)
                        if index in server_tools:
                            server_tools[index]["json"] += str(self.message_field(delta, "partial_json") or "")
                return stream.get_final_message()
        finally:
            self._emit_stream("", "")

    def anthropic_params(self, messages: list[Json], tools: list[Json] | None) -> Json:
        provider = self.session.config.provider
        resolved = provider.resolve()
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
            "max_tokens": provider.anthropic_output_cap(),
        }
        # Thinking pins temperature to its default; sending any other value is rejected.
        if request_tools := [*self.anthropic_tool_schemas(tools or []), *self.builtin_tools(resolved)]:
            params["tools"] = request_tools
            params["tool_choice"] = {"type": "auto"}
        effort = provider.reasoning_effort()
        thinking_params = anthropic_thinking_params(
            provider.model,
            provider.reasoning,
            effort,
            self.manual_thinking_budget(effort, provider.anthropic_output_cap()),
        )
        params.update(thinking_params)
        thinking = thinking_params.get("thinking")
        thinking_active = anthropic_thinking_always_on(provider.model) or (isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"))
        if provider.temperature is not None and not thinking_active:
            params["temperature"] = provider.temperature
        return params

    @staticmethod
    def manual_thinking_budget(effort: str, max_tokens: int) -> int:
        """The manual thinking budget for one effort, kept inside the request's own output budget.

        Every host with an integer budget rejects one that is not strictly below the output cap —
        Anthropic on `max_tokens`, the OpenAI-compatible `enable_thinking` hosts on the
        `max_completion_tokens` they fold that cap into — so a smaller configured
        `provider.max_tokens` has to lower the budget with it rather than fail the request. The
        1,024-token floor is the documented minimum; below that the budget cannot be satisfied at
        all and the provider's own error is the honest answer.
        Evidence: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
                  https://docs.qwencloud.com/api-reference/chat/openai-chat"""
        budget = THINKING_BUDGETS.get(effort, THINKING_BUDGETS["medium"])
        return max(1024, min(max_tokens - 1024, budget))

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
            return [block for block in saved if isinstance(block, dict) and (message.get("content") is not None or block.get("type") != "text")]
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

    def anthropic_result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
        text_parts: list[str] = []
        tool_calls: list[Json] = []
        calls: list[ToolCall] = []
        content_blocks = self.message_field(result, "content") or []
        saved_content = [self.dump_message_item(block) for block in content_blocks]
        for block in content_blocks:
            block_type = self.message_field(block, "type")
            # Streaming already reported each server tool live; the scan is the only source otherwise.
            if block_type == "server_tool_use" and not streamed:
                raw_input = self.message_field(block, "input")
                query = raw_input.get("query") if isinstance(raw_input, dict) else ""
                self.report_builtin_call(str(self.message_field(block, "name") or ""), query)
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
        if not calls and not text.strip() and self.message_field(result, "stop_reason") == "max_tokens":
            raise self.truncated_output_error(self.message_field(result, "usage"))
        assistant: Json = {"role": "assistant", "content": text or None, ANTHROPIC_CONTENT_KEY: [block for block in saved_content if block]}
        # A long server-side tool run can be paused and handed back mid-turn. The turn continues by
        # sending this message back unchanged, which the saved content blocks above already do.
        # Evidence: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        if self.message_field(result, "stop_reason") == "pause_turn":
            assistant[PAUSED_TURN_KEY] = True
        if sources := self.anthropic_sources(saved_content):
            assistant[SEARCH_SOURCES_KEY] = sources
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return assistant, calls, text

    @classmethod
    def anthropic_sources(cls, saved_content: list[Json]) -> list[Json]:
        """Sources from a Messages response: cited text first, then the raw search results.

        A `web_search_tool_result` carries an error object rather than a result list when the
        search itself failed, which `collect_sources` skips as having no URL."""
        groups: list[Any] = []
        for block in saved_content:
            if not isinstance(block, dict):
                continue
            groups.append(block.get("citations"))
            if block.get("type") == "web_search_tool_result":
                content = block.get("content")
                groups.append(content if isinstance(content, list) else None)
        return cls.collect_sources(*groups)

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
            # The resolved effort, like every other control below: a host that documents a reduced
            # scale must fold this one too, instead of the fold silently applying to its siblings.
            extra["reasoning"] = {"effort": resolved.reasoning_effort or effort}
        elif chat_reasoning == "reasoning_effort":
            if value := resolved.reasoning_effort:
                params["reasoning_effort"] = value
        elif chat_reasoning == "thinking":
            extra["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
            if reasoning_enabled:
                params["reasoning_effort"] = resolved.reasoning_effort
        elif chat_reasoning in ("thinking_toggle", "thinking_effort"):
            extra["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
            if reasoning_enabled and chat_reasoning == "thinking_effort":
                params["reasoning_effort"] = resolved.reasoning_effort
        elif chat_reasoning == "enable_thinking":
            extra["enable_thinking"] = reasoning_enabled
            if reasoning_enabled:
                # An unset max_tokens leaves the cap to the host, which sizes its own budget under it.
                extra["thinking_budget"] = (
                    self.manual_thinking_budget(effort, provider.max_tokens)
                    if provider.max_tokens > 0
                    else THINKING_BUDGETS.get(effort, THINKING_BUDGETS["medium"])
                )
        # Provider-declared extensions (e.g. Qianwen web search) pass through verbatim; minacode's
        # own reasoning fields are layered on top so they stay authoritative on key conflicts.
        extra_body = {**provider.extra_body, **extra}
        configured_thinking = provider.extra_body.get("thinking")
        managed_thinking = extra.get("thinking")
        if isinstance(configured_thinking, dict) and isinstance(managed_thinking, dict):
            extra_body["thinking"] = {**configured_thinking, **managed_thinking}
        if extra_body:
            params["extra_body"] = extra_body

    def assistant_message(self, message: Any) -> Json:
        data: Json = {"role": "assistant", "content": self.message_field(message, "content")}
        for key in ("reasoning_content", "reasoning"):
            value = self.message_field(message, key)
            if value:
                data[key] = Text.value(value)
        raw_details = self.message_field(message, "reasoning_details") or []
        details = [item for item in (self.dump_message_item(raw) for raw in raw_details) if item]
        if details:
            data["reasoning_details"] = details
        # Chat hosts that cite (OpenAI's search models, OpenRouter's web plugin) hang annotations
        # off the message. Hosts that report search on the response instead of the message are not
        # covered here; their sources stay where the provider put them.
        if sources := self.collect_sources(self.message_field(message, "annotations")):
            data[SEARCH_SOURCES_KEY] = sources
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
