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
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from json_repair import repair_json

# Aliased because the module name `anthropic` shadows the third-party SDK package of the same
# name imported inside function bodies.
from minacode.base import (
    HTTP_USER_AGENT,
    MODEL_REQUEST_RETRIES,
    PROVIDER_ORIGIN_KEY,
    SEARCH_SOURCES_KEY,
    SESSION_EVENT_KEY,
    ActiveResource,
    Billing,
    Json,
    ModelError,
    ModelOutputTruncated,
    ModelRequestRetry,
    ModelResponseTimeout,
    ModelStreamIncomplete,
    ModelUsage,
    Text,
    ToolCall,
    ToolError,
    builtin_tool_label,
)
from minacode.config import ProviderConfig
from minacode.image import IMAGE_REFS_KEY, ImageInputs
from minacode.model import resilience, responses
from minacode.model.protocol import AnthropicWire, ChatWire, ResponsesWire, WireProtocol
from minacode.prompts import COMPACTION_REQUEST_EVENT
from minacode.providers.compat import (
    ProviderPolicy,
    ResolvedProvider,
    builtin_tools_issue,
    bundled_policy,
)

if TYPE_CHECKING:
    # The provider SDKs cost ~0.8s to import and are not needed until the first request;
    # the runtime imports below keep them off the startup path (see MCPManager for the same pattern).
    from anthropic import Anthropic
    from openai import OpenAI

from minacode.session import AgentState, QueuedInput, Session
from minacode.tools import (
    Tool,
    tool_payload,
)

_ResultT = TypeVar("_ResultT")

# Retry-wait granularity: sleeping in ~0.1s slices lets the wait observe the UI-thread cancel flag
# instead of relying on a signal interrupting one long sleep.
_RETRY_SLEEP_SLICE = 0.1


def prompt_value(value: object) -> object:
    """Strip secrets and inline image bytes from a payload tree before measuring its size.

    Encrypted reasoning, signatures, and base64 image data would otherwise dominate the byte count
    they are meant to inflate, so they are dropped (or blanked for inline data) before the estimate.
    api-agnostic, so it lives here rather than in any wire.
    """

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


@dataclass(frozen=True)
class PreparedRequest:
    messages: list[Json]
    tools: list[Json]
    pending: list[QueuedInput]
    turn_messages: list[Json]
    # Exact semantic messages whose raw image occurrences entered this request since the last
    # accepted main-model request (opening attachment, claimed queued attachment, or a ViewImage
    # observation from the current tool loop). Eligibility for 400 learning requires at least one
    # of these; the fallback observes exactly these and nothing older.
    current_image_messages: tuple[Json, ...] = ()


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
        # Lifecycle hook, mirroring ContextManager.on_compaction: True while a retry backoff wait is in
        # progress, False in a finally block. Lets the orchestration label the phase without model
        # depending on a renderer.
        self.on_retry_wait: Callable[[bool], None] | None = None
        # The effective model the last compaction summary ran on; "" when the last compaction fell
        # back to deterministic trimming or never ran. Recorded on the HistorySegment by callers.
        self.last_compaction_model = ""
        self._wires: dict[str, WireProtocol] = {
            "chat": ChatWire(self),
            "responses": ResponsesWire(self),
            "anthropic": AnthropicWire(self),
        }

    def policy(self) -> ProviderPolicy:
        """The active catalog policy, or the bundled one before a session is bootstrapped."""

        catalog = self.session.catalog
        return catalog.policy if catalog is not None else bundled_policy()

    def resolved(self, provider: ProviderConfig) -> ResolvedProvider:
        return self.policy().resolve(provider)

    def apply_request(self, params: Json, provider: ProviderConfig, resolved: ResolvedProvider, *, wire: str) -> Json:
        """Run the resolved request recipe over a body this client assembled."""

        return self.policy().apply_request(params, provider, resolved, wire=wire)

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
        return f"{self.resolved(provider).base_url}/{provider.model.lower()}#{credential}"

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

        A message carrying COMPACTION_REQUEST_EVENT is excluded -- see Compactor.request, which
        marks its instruction only when the live projection's own boundary already falls inside
        the slice it is appending to."""
        return max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user" and not ImageInputs.is_tool_observation(message) and message.get(SESSION_EVENT_KEY) != COMPACTION_REQUEST_EVENT
            ),
            default=-1,
        )

    def estimated_request_tokens(self, messages: list[Json], tools: list[Json] | None = None) -> int:
        """Estimate the actual protocol payload instead of minacode's normalized history."""

        resolved = self.resolved(self.session.config.provider)
        # Measuring a payload must never fail on it: an entry this wire rejects is the request's
        # error to raise, not something that should break the status bar, /status, or resume.
        builtin = self.builtin_tools(resolved, strict=False)
        # Payload builders would otherwise expand every local image to base64 merely to throw the
        # bytes away below. Labels preserve the surrounding wire shape; image tiles are added once.
        projected = [{key: value for key, value in message.items() if key != IMAGE_REFS_KEY} for message in messages]
        payload = self.wire(self.session.config.provider).estimation_payload(projected, tools, builtin)
        chars = len(json.dumps(prompt_value(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        # A text-only route never projects raw image blocks, so its tiles must not count against
        # the budget either; labels and the asset-context line are already inside `chars`.
        images = self.session.images.estimated_tokens(messages) if not self.session.image_route.is_text_only() else 0
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
            # ModelStreamIncomplete is raised by the Responses stream reassembler and must reach
            # the retry classifier as itself: flattening it here would lose the retry decision.
            except (ModelResponseTimeout, ModelStreamIncomplete):
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
                    return self.api_request(messages, tools)
                except KeyboardInterrupt:
                    if state.manual_model_retry_requested:
                        state.manual_model_retry_requested = False
                        raise ModelRequestRetry() from None
                    raise
                except ModelError as error:
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

    def _record_usage(self, usage: Any, billing: Billing = Billing.MAIN) -> None:
        """Add a completed request to session usage, keeping the budget it was prepared against so the
        status fill uses the request-time denominator instead of today's configuration.

        `billing` routes a secondary-request cost onto its own counter and its own last-request
        snapshot. A summary request goes to its own counter and account: it can be billed at a
        different price, and its prefix reuse is worth reading on its own -- it rides the cached
        prefix deliberately, and blending the two would hide whether that worked. A vision
        observation joins the main totals but must not overwrite the main counter's last-request
        ctx/cache snapshot the status bar reads."""
        counter = self.session.compaction_usage if billing == Billing.COMPACTION else self.session.usage
        counter.add(usage, self.session.request_token_budget(), touch_last=billing != Billing.VISION)

    def wire(self, provider: ProviderConfig) -> WireProtocol:
        """The adapter for a provider's wire api, selected once per request.

        A direct lookup rather than a chat fallback: `resolve()` only ever yields one of the three
        wire names -- `provider.api` is checked against PROVIDER_API_CHOICES wherever it is set
        (config load, /api, the [worker] and [compaction] overrides) and `auto` resolves through
        the catalog to one of them. An unknown key here would be a bug worth raising, not a
        request quietly sent on the wrong wire."""
        return self._wires[self.resolved(provider).api]

    def api_request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        provider: ProviderConfig | None = None,
        json_object: bool = False,
        billing: Billing = Billing.MAIN,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self.session.config.provider
        return self.wire(provider).request(
            messages,
            tools,
            provider=provider,
            allow_stream=allow_stream,
            response_timeout=response_timeout,
            json_object=json_object,
            billing=billing,
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

    @staticmethod
    def request_headers(provider: ProviderConfig) -> dict[str, str]:
        """Default headers for one entry: minacode's own, then the entry's `headers` over them.

        Both wires share this so a header configured for an entry follows it across a `/provider`
        switch and into the worker and compaction entries, which are copies of it."""
        return {"User-Agent": HTTP_USER_AGENT, **provider.headers}

    def client(self, provider: ProviderConfig | None = None) -> OpenAI:
        provider = provider if provider is not None else self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        from openai import OpenAI

        return OpenAI(
            api_key=provider.key,
            base_url=self.resolved(provider).base_url,
            timeout=provider.timeout,
            max_retries=0,
            default_headers=self.request_headers(provider),
        )

    def anthropic_client(self, provider: ProviderConfig | None = None) -> Anthropic:
        provider = provider if provider is not None else self.session.config.provider
        if missing := self.session.missing_config():
            raise ModelError("missing config: " + ", ".join(missing))
        url = self.resolved(provider).base_url.rstrip("/")
        # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
        from anthropic import Anthropic

        return Anthropic(
            api_key=provider.key,
            base_url=url.removesuffix("/v1"),
            timeout=provider.timeout,
            max_retries=0,
            default_headers=self.request_headers(provider),
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
                item = raw if isinstance(raw, dict) else responses.dump_message_item(raw)
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
        resolved = resolved or self.resolved(provider)
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
        """The routing hint for prefix caching, or "" when the entry does not define one.

        It steers which machine serves a request so that requests sharing a prefix land together;
        it does not pin routing and never substitutes for a byte-identical prefix. Everything that
        changes the rendered prefix therefore belongs in the key -- notably the tool set below.
        See DESIGN.md "Cache epochs and breakpoints"."""
        configured = provider.prompt_cache_key
        if configured == "off":
            return ""
        if configured != "auto":
            return configured
        resolved = self.resolved(provider)
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

    def apply_provider_params(self, params: Json, provider: ProviderConfig, resolved: ResolvedProvider | None = None) -> None:
        resolved = resolved or self.resolved(provider)
        # Provider-declared extra body first, so the recipe engine's path merge keeps the user's
        # configured fields authoritative on key conflicts outside the managed path.
        if provider.extra_body:
            params["extra_body"] = {**provider.extra_body, **(params.get("extra_body") or {})}
        # Some native APIs fix or reject temperature for all or part of their thinking modes.
        if provider.temperature is not None and not resolved.suppress_temperature:
            params["temperature"] = provider.temperature
        self.apply_request(params, provider, resolved, wire=resolved.api)

    def assistant_message(self, message: Any) -> Json:
        data: Json = {"role": "assistant", "content": self.message_field(message, "content")}
        for key in ("reasoning_content", "reasoning"):
            value = self.message_field(message, key)
            if value:
                data[key] = Text.value(value)
        raw_details = self.message_field(message, "reasoning_details") or []
        details = [item for item in (responses.dump_message_item(raw) for raw in raw_details) if item]
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
    def tool_call(cls, call_id: str, name: str, payload: object) -> ToolCall:
        # payload_args may reject malformed arguments (e.g. Bash with an empty command). Capture that
        # error on the call so it is replayed as a tool result during execution, letting the model
        # self-correct, rather than escaping to abort the entire agent turn.
        try:
            return ToolCall(id=call_id, name=name, args=tool_payload(name, payload))
        except ToolError as error:
            return ToolCall(id=call_id, name=name, args=[], error=str(error))
