"""minacode model client: provider request protocols, streaming, and retry policy."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from json_repair import repair_json

# Aliased because the module name `anthropic` shadows the third-party SDK package of the same
# name imported inside function bodies.
import minacode.model.anthropic as anthropic_module
from minacode.base import (
    ANTHROPIC_CONTENT_KEY,
    HTTP_USER_AGENT,
    MODEL_REQUEST_RETRIES,
    PROVIDER_ORIGIN_KEY,
    SEARCH_SOURCES_KEY,
    SESSION_EVENT_KEY,
    ActiveResource,
    Json,
    ModelError,
    ModelOutputTruncated,
    ModelRequestRetry,
    ModelResponseTimeout,
    ModelUsage,
    Text,
    ToolArgs,
    ToolCall,
    ToolError,
    builtin_tool_label,
)
from minacode.config import (
    ProviderConfig,
    compaction_provider_config,
    vision_provider_config,
)
from minacode.image import IMAGE_REFS_KEY, ImageInputs, ImageRef
from minacode.model import chat, resilience, responses
from minacode.prompts import (
    COMPACTION_ECHO_RETRY,
    COMPACTION_PROMPT,
    COMPACTION_REQUEST_EVENT,
    COMPACTION_RETRY,
    VISION_OBSERVE_DEFAULT_QUESTION,
    VISION_OBSERVE_PROMPT,
)
from minacode.providers.catalog import THINKING_BUDGETS
from minacode.providers.compat import (
    ResolvedProvider,
    anthropic_keeps_prior_thinking,
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


@dataclass(frozen=True)
class PreparedRequest:
    messages: list[Json]
    tools: list[Json]
    pending: list[QueuedInput]
    turn_messages: list[Json]


class _RequestLease:
    """Keep callbacks inside the lifetime of the background request that produced them."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = True


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

    _JSON_FENCE_RE: ClassVar[re.Pattern] = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)

    def __init__(self, session: Session):
        self.session = session
        self.cancel_requested = threading.Event()
        self.active_client: ActiveResource[OpenAI | Anthropic] = ActiveResource()
        self._request_local = threading.local()
        self.on_stream: Callable[[str, str], None] | None = None
        # Called with (label, detail) for each provider-side tool call a response reports. Reported
        # from the parsed result rather than the stream, so a search is logged the same way when
        # streaming is off and on a frontend that shows no live status at all.
        self.on_builtin_call: Callable[[str, str], None] | None = None
        # Called with (label, detail) when a vision-bridge observation runs, before the request is
        # sent: the attachment bridge fires before the turn's first model call, a stretch where
        # nothing else reports, so without it the vision request is invisible in the transcript.
        # (A bridged ViewImage draws its trace inside the tool's own finish block instead.)
        self.on_vision_observe: Callable[[str, str], None] | None = None
        # Lifecycle hook, mirroring ContextManager.on_compaction: True while a retry backoff wait is in
        # progress, False in a finally block. Lets the orchestration label the phase without model
        # depending on a renderer.
        self.on_retry_wait: Callable[[bool], None] | None = None
        # The effective model the last compaction summary ran on; "" when the last compaction fell
        # back to deterministic trimming or never ran. Recorded on the HistorySegment by callers.
        self.last_compaction_model = ""

    def cancel(self) -> None:
        self.cancel_requested.set()
        with contextlib.suppress(Exception):
            self.active_client.apply(lambda client: client.close())

    def provider_origin(self, provider: ProviderConfig | None = None) -> str:
        """Identity of the endpoint that issues — and alone can verify — a provider echo.

        Everything that decides who can verify the ciphertext belongs here. A hostname alone does
        not: two local gateways separated only by port, or two entries on one host holding keys for
        different organizations, would share an identity and defeat the check. The full base URL
        covers scheme, port, and path routing; the key is reduced to a fingerprint so a credential
        never reaches a session snapshot. Rotating a key costs one turn of reasoning continuity.
        """
        provider = provider if provider is not None else self.session.config.provider
        credential = hashlib.sha256(provider.key.encode("utf-8")).hexdigest()[:12] if provider.key else "-"
        return f"{provider.resolve().base_url}/{provider.model.lower()}#{credential}"

    @staticmethod
    def replayable_echo(message: Json, origin: str) -> bool:
        """Whether this turn's provider echo may be replayed to the endpoint now in use.

        Sending another issuer's ciphertext back is a verification failure, not a silent drop, so a
        mismatch falls through to rebuilding the turn from its normalized text and tool calls — the
        same degradation a protocol switch already takes. Turns recorded before origins were stamped
        carry no identity and stay replayable: an unmarked session is almost always still on the
        provider that wrote it, and dropping thinking blocks mid tool loop would break it outright.
        """
        saved = message.get(PROVIDER_ORIGIN_KEY)
        return not isinstance(saved, str) or not saved or saved == origin

    @staticmethod
    def latest_user_position(messages: list[Json]) -> int:
        """Index of the message that ends the current turn for reasoning replay, or -1.

        Exposed rather than inlined because compaction places its appended instruction relative to
        this exact boundary: whether that instruction counts as the boundary decides whether the
        summary request strips the same reasoning the live request did, and a second expression of
        the rule is how that came apart twice.

        A message carrying COMPACTION_REQUEST_EVENT is excluded -- see
        ContextManager.compaction_request, which marks its instruction only when the live
        projection's own boundary already falls inside the slice it is appending to."""
        return max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user" and not ImageInputs.is_tool_observation(message) and message.get(SESSION_EVENT_KEY) != COMPACTION_REQUEST_EVENT
            ),
            default=-1,
        )

    def chat_messages(self, messages: list[Json], provider: ProviderConfig | None = None) -> list[Json]:
        """Build Chat Completions history using the provider's documented replay contract."""

        provider = provider if provider is not None else self.session.config.provider
        return chat.chat_messages(messages, provider, provider.resolve(), self.session.images, self.latest_user_position)

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
        lease = _RequestLease()

        def invoke() -> None:
            self._request_local.lease = lease
            try:
                outcome.append((True, request()))
            except BaseException as error:  # noqa: BLE001 - preserve the provider call's exact outcome across the deadline thread boundary.
                outcome.append((False, error))
            finally:
                del self._request_local.lease
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
                with lease.lock:
                    lease.active = False
                with contextlib.suppress(Exception):
                    client.close()

    def _request_active(self) -> bool:
        lease = getattr(self._request_local, "lease", None)
        if lease is None:
            return not self.cancel_requested.is_set()
        with lease.lock:
            return lease.active and not self.cancel_requested.is_set()

    def _raise_if_request_inactive(self) -> None:
        if not self._request_active():
            raise KeyboardInterrupt

    def _request_callback(self, callback: Callable[[], None]) -> None:
        lease = getattr(self._request_local, "lease", None)
        if lease is None:
            if not self.cancel_requested.is_set():
                callback()
            return
        with lease.lock:
            if lease.active and not self.cancel_requested.is_set():
                callback()

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
                state.stream_started_at = state.stream_chars = 0
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
                    retryable = resilience.retryable_error(error)
                    if attempt >= MODEL_REQUEST_RETRIES or not retryable:
                        if attempt:
                            raise ModelError(f"{error} (after {attempt + 1} attempts)") from error
                        raise
                    state.current_model_attempt = attempt + 2
                    state.model_retry_reason = resilience.retry_reason(error)
                    state.model_retry_count += 1
                    self._wait_before_retry(resilience.retry_delay(error, attempt), state)
                finally:
                    state.current_model_call_started_at = 0.0
                    # Cleared, not kept: a rate with no stream behind it would freeze on the divider
                    # for the length of the tool call that follows.
                    state.stream_started_at = 0.0
                attempt += 1
        finally:
            state.current_model_attempt = 0
            state.model_retry_reason = ""

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
        status fill uses the request-time denominator instead of today's configuration.

        A summary request goes to its own counter, told apart by the label compact() publishes for
        the length of that request: it can be billed to a different account at a different price,
        and its prefix reuse is worth reading on its own -- it now rides the conversation's cached
        prefix deliberately, and blending the two would hide whether that worked."""
        counter = self.session.compaction_usage if self.session.state.compaction_entry else self.session.usage
        counter.add(usage, self.session.request_token_budget())

    def chat_request(
        self,
        messages: list[Json],
        tools: list[Json] | None = None,
        *,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        provider: ProviderConfig | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self.session.config.provider
        messages = self.chat_messages(messages, provider=provider)
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self.on_stream is not None
        params = chat.chat_params(
            messages,
            tools,
            provider,
            resolved,
            stream=stream,
            json_object=json_object,
            builtin_tools=self.builtin_tools,
            derive_cache_key=self.prompt_cache_key,
            apply_provider_params=self.apply_provider_params,
        )
        client = self.client(provider=provider)
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
        return chat.reassemble_stream(
            client,
            params,
            message_field=self.message_field,
            dump_message_item=self.dump_message_item,
            raise_if_inactive=self._raise_if_request_inactive,
            emit=self._emit_stream,
        )

    def api_request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        provider: ProviderConfig | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self.session.config.provider
        api = provider.resolve().api
        if api == "anthropic":
            request = self.anthropic_request
        elif api == "responses":
            request = self.responses_request
        else:
            request = self.chat_request
        # json_object reaches the Chat wire only. Responses spells the same thing differently and
        # Anthropic spells it differently again (output_format); neither is wired yet, and passing
        # an unknown keyword to them would be an error rather than a no-op.
        extra: Json = {"json_object": True} if json_object and api not in ("anthropic", "responses") else {}
        if allow_stream and response_timeout is None:
            return request(messages, tools, provider=provider, **extra)
        return request(messages, tools, allow_stream=allow_stream, response_timeout=response_timeout, provider=provider, **extra)

    def responses_request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        provider: ProviderConfig | None = None,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self.session.config.provider
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self.on_stream is not None
        params: Json = {
            "model": provider.model,
            "input": responses.responses_input(
                Text.value(messages),
                self.provider_origin(provider),
                provider_origin=self.provider_origin,
                replayable_echo=self.replayable_echo,
                images=self.session.images,
            ),
            "stream": stream,
            "store": False,
        }
        if provider.max_tokens > 0:
            params["max_output_tokens"] = provider.max_tokens
        if request_tools := [*responses.responses_tool_schemas(tools or []), *self.builtin_tools(resolved)]:
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
        if provider.extra_body and (extra_body := responses.responses_extra_body(provider.extra_body, params)):
            params["extra_body"] = extra_body
        client = self.client(provider=provider)
        if stream:
            result = self.call_client(client, lambda: self._responses_stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self.call_client(client, lambda: client.responses.create(**params), response_timeout=response_timeout)
            streamed = False
        self._record_usage(self.message_field(result, "usage"))
        assistant, calls, text = self.responses_result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self.provider_origin(provider)
        return assistant, calls, text

    @staticmethod
    def responses_extra_body(extra_body: Json, params: Json) -> Json:
        """Fold configured `reasoning` fields into the managed object instead of replacing it.

        `extra_body` is merged over the request body, so a whole object configured there would drop
        the fields minacode manages inside it — settling `reasoning.context` would silently take the
        resolved `effort` with it. Merging per field keeps a documented extra reachable while
        `/reason` stays authoritative, mirroring how the Chat path folds `thinking`.
        """
        return responses.responses_extra_body(extra_body, params)

    def _responses_stream(self, client: OpenAI, params: Json) -> Any:
        """Consume a Responses stream, promoting completed text before tool arguments finish.

        Text completion and function-call discovery are independent events and either can arrive
        first. Promotion is therefore a two-condition state transition, not an ordering assumption;
        the terminal response is still consumed normally for history, tool calls, and usage.
        """

        return responses.reassemble_stream(
            client,
            params,
            message_field=self.message_field,
            raise_if_inactive=self._raise_if_request_inactive,
            emit=self._emit_stream,
            report_builtin_call=self.report_builtin_call,
        )

    def _emit_stream(self, kind: str, delta: str) -> None:
        # Counted here rather than at each protocol's stream reader: this is the one funnel every
        # API shape passes through, and reasoning deltas are part of what the wait is made of.
        state = self.session.state
        if not state.stream_started_at:
            state.stream_started_at = time.monotonic()
        state.stream_chars += len(delta)
        if self.on_stream is not None:
            self._request_callback(lambda: self.on_stream(kind, delta) if self.on_stream is not None else None)

    def responses_input(self, messages: list[Json], origin: str = "") -> list[Json]:
        return responses.responses_input(
            messages,
            origin,
            provider_origin=self.provider_origin,
            replayable_echo=self.replayable_echo,
            images=self.session.images,
        )

    @staticmethod
    def replayable_output_item(item: Json) -> bool:
        """Whether a saved output item still carries something a later request can use.

        Stateless reasoning travels in the encrypted payload, which the id alone cannot stand in
        for once the response was never stored. A host that returns neither that payload nor any
        readable reasoning leaves an empty shell, so it is dropped instead of replayed."""
        return responses.replayable_output_item(item)

    @staticmethod
    def responses_tool_schemas(tools: list[Json]) -> list[Json]:
        return responses.responses_tool_schemas(tools)

    def responses_result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
        return responses.responses_result(
            result,
            streamed,
            message_field=self.message_field,
            dump_message_item=self.dump_message_item,
            tool_call=self.tool_call,
            report_builtin_call=self.report_builtin_call,
            truncated_output_error=self.truncated_output_error,
            collect_sources=self.collect_sources,
        )

    @classmethod
    def responses_sources(cls, saved_output: list[Json]) -> list[Json]:
        """Sources a Responses host attached to one response.

        Two hosts, two places: OpenAI cites inline through `url_citation` annotations on the
        message, while Qwen returns no citations at all and reports sources only on the search
        call. Reading both keeps one renderer honest across them."""
        return responses.responses_sources(saved_output, cls.collect_sources)

    @staticmethod
    def dump_message_item(item: Any) -> Json:
        return responses.dump_message_item(item)

    def vision_observe(self, images: tuple[ImageRef, ...], question: str = "") -> str:
        """Ask the [vision]-configured entry to observe images, bypassing the active provider's
        image gate.

        Mirrors compact(): the [vision] entry is resolved per call and validated locally -- a
        missing field would otherwise surface as a generic SDK credentials error naming nothing
        the user can act on -- then served by one non-streaming api_request with pre-built image
        blocks. Perception only: no tools, no coding task; the main model does the reasoning.
        """

        provider = vision_provider_config(self.session.config)
        entry_name = self.session.config.vision_provider
        if missing := provider.missing_fields():
            raise ModelError(f"vision provider `{entry_name}` is missing {', '.join(missing)}; check [vision] and [provider.{entry_name}]")
        if self.on_vision_observe is not None:
            self.on_vision_observe(
                f"{entry_name}/{provider.model}",
                f"described {len(images)} attached image{'s' if len(images) != 1 else ''}",
            )
        messages = [
            {"role": "system", "content": VISION_OBSERVE_PROMPT},
            {
                "role": "user",
                "content": self.session.images.vision_content(images, provider.resolve().api, question.strip() or VISION_OBSERVE_DEFAULT_QUESTION),
            },
        ]
        _, _, content = self.api_request(messages, tools=None, allow_stream=False, response_timeout=provider.response_timeout, provider=provider)
        return content.strip()

    def compact(self, context: str, inline_messages: list[Json] | None = None, tools: list[Json] | None = None, echo_source: str = "") -> Json:
        self.cancel_requested.clear()
        # The summary request runs on the [compaction]-resolved provider entry (empty [compaction]
        # = the active provider), resolved per call so a runtime /provider switch applies next
        # time. The context budget is untouched: compaction still measures against the main
        # provider's window, only the summary request itself uses this entry.
        provider = compaction_provider_config(self.session.config)
        entry_name = self.session.config.compaction_provider or self.session.config.active_provider
        # The client's own gate checks the active provider, which is the wrong entry when a
        # summary runs elsewhere: an incomplete [compaction] entry would otherwise reach the SDK
        # and come back as "Missing credentials", naming nothing the user can act on. Compaction
        # still degrades to deterministic trimming -- this only makes the fallback say why.
        if missing := provider.missing_fields():
            raise ModelError(f"compaction provider `{entry_name}` is missing {', '.join(missing)}; check [compaction] and [provider.{entry_name}]")
        self.last_compaction_model = ""
        # Two shapes. The inline form is the agent's own request with a compaction instruction
        # appended: same tools, same system, same conversation, so the provider's prefix cache --
        # already warm from the turn that just ran -- covers everything but the tail, and the
        # compactor sees real messages instead of a flattened re-rendering that drops tool calls.
        # The flattened form stays for every case the inline one cannot serve.
        inline = inline_messages is not None
        messages = inline_messages if inline else [{"role": "system", "content": COMPACTION_PROMPT}, {"role": "user", "content": Text.clean(context)}]
        # Compaction honors the configured total-generation limit instead of a hidden cap: a
        # summary is worth the user's configured wait, and the deterministic trim fallback still
        # catches whatever the provider rejects.
        response_timeout = provider.response_timeout
        # The status bar reads this to name the entry actually serving the summary; cleared in the
        # finally so a timeout, a cancel, or a provider error leaves no stale row behind.
        entry_label = f"{entry_name}/{provider.model}"
        self.session.state.compaction_entry = entry_label
        try:
            data = self.compact_attempts(messages, provider, response_timeout, entry_label, tools=tools if inline else None, echo_source=echo_source)
        finally:
            self.session.state.compaction_entry = ""
        self.last_compaction_model = provider.model
        return data

    def compact_attempts(
        self,
        messages: list[Json],
        provider: ProviderConfig,
        response_timeout: float,
        entry_label: str,
        tools: list[Json] | None = None,
        echo_source: str = "",
    ) -> Json:
        """Ask for the summary, and ask once more if what came back was not a JSON object.

        The failure this retries is a model ignoring the format and replying in prose -- usually by
        continuing the conversation it was handed instead of summarizing it. A bare resend would
        likely reproduce it, so the second attempt carries the previous reply and a correction,
        which is the same shape a person would use. Every error names the entry that served the
        request: compaction can run on its own `[compaction]` provider, and a cheaper model there is
        exactly the one that fails this way, so the message has to say which model to look at."""
        attempt_messages = list(messages)
        for attempt in (1, 2):
            try:
                # Tools ride along, and tool_choice is deliberately left exactly as an ordinary
                # request sets it. Forcing "none" here looks safer and is not: changing tool_choice
                # invalidates the messages cache, which is the whole 100k-token conversation this
                # request exists to reuse -- it would spend the prize to buy the guarantee. The
                # instruction not to call tools lives in the appended message instead, and a model
                # that calls one anyway returns no text, which the retry below already handles.
                _, _, content = self.api_request(
                    attempt_messages, tools, allow_stream=False, response_timeout=response_timeout, provider=provider, json_object=True
                )
            except ModelResponseTimeout:
                raise ModelResponseTimeout(
                    f"compaction summary on `{entry_label}` exceeded provider.response_timeout={response_timeout:g}s; "
                    "set it to 0 to disable the total-generation limit"
                ) from None
            try:
                data = self.parse_json_object(content)
            except ModelError as error:
                if attempt == 2:
                    raise ModelError(f"{error} (compaction provider `{entry_label}`)") from None
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": Tool.compact(content, 400)},
                    {"role": "user", "content": COMPACTION_RETRY},
                ]
                continue
            if not isinstance(data, dict):
                raise ModelError(f"compactor returned non-object JSON (compaction provider `{entry_label}`)")
            # Checked after parsing, not instead of it: this is the same failure the retry above
            # exists for, only it arrived shaped like a valid answer. Left unchecked it applies
            # cleanly into session.state and is fed back into every later compaction.
            if self.echoes_source(data.get("summary") or "", echo_source):
                if attempt == 2:
                    raise ModelError(f"compactor echoed the conversation instead of summarizing it (compaction provider `{entry_label}`)")
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": Tool.compact(content, 400)},
                    {"role": "user", "content": COMPACTION_ECHO_RETRY},
                ]
                continue
            return data
        raise ModelError(f"compactor returned no usable JSON (compaction provider `{entry_label}`)")

    # A summary that is mostly one verbatim run copied out of the conversation is the echo failure
    # wearing valid JSON: constrained decoding forces the shape, never the task. Deliberately blunt
    # thresholds -- a real summary paraphrases, so 80% of it being a single unbroken copy is not a
    # close call, and quoting a path or an identifier is far below that.
    #
    # The floor is counted in characters, which are not equal: the same sentence is 138 characters
    # of English and 68 of Chinese. It is set for the denser script, because a floor tuned to
    # English would have excluded from this check every summary written in the language the failure
    # was first seen in.
    ECHO_MIN_CHARS: ClassVar[int] = 40
    ECHO_RATIO: ClassVar[float] = 0.8
    ECHO_COMPARE_CHARS: ClassVar[int] = 4000

    @classmethod
    def echoes_source(cls, summary: str, source: str) -> bool:
        """True when `summary` reproduces `source` rather than describing it."""
        summary = " ".join(str(summary).split())[: cls.ECHO_COMPARE_CHARS]
        source = " ".join(str(source).split())[-cls.ECHO_COMPARE_CHARS :]
        if len(summary) < cls.ECHO_MIN_CHARS or not source:
            return False
        match = SequenceMatcher(None, summary, source, autojunk=False).find_longest_match(0, len(summary), 0, len(source))
        return match.size >= len(summary) * cls.ECHO_RATIO

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

    def client(self, provider: ProviderConfig | None = None) -> OpenAI:
        provider = provider if provider is not None else self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        from openai import OpenAI

        return OpenAI(
            api_key=provider.key, base_url=provider.resolve().base_url, timeout=provider.timeout, max_retries=0, default_headers={"User-Agent": HTTP_USER_AGENT}
        )

    def anthropic_client(self, provider: ProviderConfig | None = None) -> Anthropic:
        provider = provider if provider is not None else self.session.config.provider
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
            label = builtin_tool_label(name)
            text = str(detail or "").strip()
            self._request_callback(lambda: self.on_builtin_call(label, text) if self.on_builtin_call is not None else None)

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
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        provider: ProviderConfig | None = None,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self.session.config.provider
        messages = Text.value(messages)
        params = self.anthropic_params(messages, tools, provider=provider)
        client = self.anthropic_client(provider=provider)
        stream = allow_stream and provider.stream and self.on_stream is not None
        if stream:
            result = self.call_client(client, lambda: self._anthropic_stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self.call_client(client, lambda: client.messages.create(**params), response_timeout=response_timeout)
            streamed = False
        self._record_usage(self.message_field(result, "usage"))
        assistant, calls, content = self.anthropic_result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self.provider_origin(provider)
        return assistant, calls, content

    def _anthropic_stream(self, client: Anthropic, params: Json) -> Any:
        """Consume Messages blocks and promote text once both text and tool blocks are known.

        Content blocks need not put text before `tool_use`, so block start/stop events feed the same
        order-independent transition as Responses. Input JSON may continue after promotion when the
        completed text block came first.
        """
        return anthropic_module.reassemble_stream(
            client,
            params,
            message_field=self.message_field,
            raise_if_inactive=self._raise_if_request_inactive,
            emit=self._emit_stream,
            report_builtin_call=self.report_builtin_call,
        )

    def anthropic_params(self, messages: list[Json], tools: list[Json] | None, provider: ProviderConfig | None = None) -> Json:
        provider = provider if provider is not None else self.session.config.provider
        return anthropic_module.anthropic_params(
            messages,
            tools,
            provider,
            provider.resolve(),
            provider_origin=self.provider_origin,
            replayable_echo=self.replayable_echo,
            images=self.session.images,
            builtin_tools=self.builtin_tools,
        )

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
        return anthropic_module.manual_thinking_budget(effort, max_tokens)

    def anthropic_messages(self, messages: list[Json], origin: str = "") -> list[Json]:
        return anthropic_module.anthropic_messages(
            messages,
            origin,
            provider_origin=self.provider_origin,
            replayable_echo=self.replayable_echo,
            images=self.session.images,
        )

    @staticmethod
    def append_anthropic_message(messages: list[Json], role: str, content: str | list[Json]) -> None:
        anthropic_module.append_anthropic_message(messages, role, content)

    def anthropic_assistant_blocks(self, message: Json, origin: str = "") -> list[Json]:
        return anthropic_module.anthropic_assistant_blocks(
            message,
            origin,
            provider_origin=self.provider_origin,
            replayable_echo=self.replayable_echo,
        )

    @staticmethod
    def anthropic_tool_schemas(tools: list[Json]) -> list[Json]:
        return anthropic_module.anthropic_tool_schemas(tools)

    def anthropic_result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
        return anthropic_module.anthropic_result(
            result,
            streamed,
            message_field=self.message_field,
            dump_message_item=self.dump_message_item,
            tool_call=self.tool_call,
            report_builtin_call=self.report_builtin_call,
            truncated_output_error=self.truncated_output_error,
            collect_sources=self.collect_sources,
        )

    @classmethod
    def anthropic_sources(cls, saved_content: list[Json]) -> list[Json]:
        """Sources from a Messages response: cited text first, then the raw search results.

        A `web_search_tool_result` carries an error object rather than a result list when the
        search itself failed, which `collect_sources` skips as having no URL."""
        return anthropic_module.anthropic_sources(saved_content, cls.collect_sources)

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
