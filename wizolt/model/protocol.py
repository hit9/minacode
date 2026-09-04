"""Wire adapters: one complete adapter per provider wire.

Each adapter owns that wire's construction (messages/params/tool schemas), sending (request), and
parsing (result/sources). ModelClient keeps the shared request lifecycle and hands each adapter a
reference to itself at construction, so the client's shared services are bound once per wire
instead of re-injected on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import wizolt.model.anthropic as anthropic_module
import wizolt.model.chat as chat_module
import wizolt.model.responses as responses_module
from wizolt.base import PROVIDER_ORIGIN_KEY, Billing, Json, ModelError, Text, ToolCall
from wizolt.config import ProviderConfig

if TYPE_CHECKING:
    from wizolt.model.client import ModelClient


def omit_request_fields(params: Json, names: tuple[str, ...]) -> Json:
    """Remove configured endpoint-rejected fields from a completed wire request, in place."""

    extra = params.get("extra_body")
    for name in names:
        params.pop(name, None)
        if isinstance(extra, dict):
            extra.pop(name, None)
    if isinstance(extra, dict) and not extra:
        params.pop("extra_body", None)
    return params


class WireProtocol(Protocol):
    """One provider wire's complete adapter: construction, sending, and parsing.

    `request` is what api_request hands every request to; `messages` builds the wire's history
    payload and is shared by all three adapters (the token estimator uses it). Per-wire
    construction/parsing the other adapters do not share stays off the interface.

    `json_object` reaches the Chat wire only. Responses spells the same thing differently and
    Anthropic spells it differently again (output_format); neither is wired yet, so both accept
    the flag and ignore it rather than passing an unknown keyword to their SDK.
    """

    async def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
        billing: Billing = Billing.MAIN,
    ) -> tuple[Json, list[ToolCall], str]: ...

    def messages(self, messages: list[Json], *, text_only: bool | None = None) -> list[Json]: ...

    def estimation_payload(self, messages: list[Json], tools: list[Json] | None, builtin: list[Json]) -> Json: ...


class ChatWire:
    """The chat-completions wire (the default/else branch)."""

    def __init__(self, client: ModelClient):
        self._client = client

    async def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
        billing: Billing = Billing.MAIN,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        messages = self.messages(messages, provider=provider)
        resolved = self._client.resolved(provider)
        stream = allow_stream and provider.stream and self._client.on_stream is not None
        params = chat_module.chat_params(
            messages,
            tools,
            provider,
            resolved,
            stream=stream,
            json_object=json_object,
            builtin_tools=self._client.builtin_tools,
            derive_cache_key=self._client.prompt_cache_key,
            apply_provider_params=self._client.apply_provider_params,
        )
        omit_request_fields(params, provider.omit_body)
        client = self._client.client(provider=provider)
        if stream:
            message, usage, finish_reason = await self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
        else:
            response = await self._client.call_client(client, lambda: client.chat.completions.create(**params), response_timeout=response_timeout)
            usage = getattr(response, "usage", None)
            message = response.choices[0].message
            finish_reason = str(self._client.message_field(response.choices[0], "finish_reason") or "")
        self._client._record_usage(usage, billing=billing)
        assistant = self._client.assistant_message(message)
        calls = self._client.tool_calls(message)
        content = str(self._client.message_field(message, "content") or "")
        # Raised outside call_client, which flattens every exception into a plain ModelError.
        if finish_reason == "length" and not calls and not content.strip():
            raise self._client.empty_length_error(usage)
        return assistant, calls, content

    def messages(self, messages: list[Json], *, text_only: bool | None = None, provider: ProviderConfig | None = None) -> list[Json]:
        """Build Chat Completions history using the provider's documented replay contract.

        `text_only` defaults to the session's current main-route image verdict; pass an explicit
        value to override it (the main request path recomputes it per call).
        """

        provider = provider if provider is not None else self._client.session.config.provider
        if text_only is None:
            text_only = self._client.session.image_route.is_text_only()
        return chat_module.chat_messages(
            messages, provider, self._client.resolved(provider), self._client.session.images, self._client.latest_user_position, text_only=text_only
        )

    def estimation_payload(self, messages: list[Json], tools: list[Json] | None, builtin: list[Json]) -> Json:
        """The wire-shaped payload for one token estimate, matching what request() would send."""
        payload: Json = {"messages": self.messages(messages)}
        if request_tools := [*(tools or []), *builtin]:
            payload["tools"] = request_tools
        return payload

    async def _stream(self, client: Any, params: Json) -> tuple[Json, Any, str]:
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
        return await chat_module.reassemble_stream(
            client,
            params,
            message_field=self._client.message_field,
            dump_message_item=responses_module.dump_message_item,
            raise_if_inactive=self._client._raise_if_request_inactive,
            emit=self._client._emit_stream,
        )


class ResponsesWire:
    """The OpenAI Responses wire."""

    def __init__(self, client: ModelClient):
        self._client = client

    async def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
        billing: Billing = Billing.MAIN,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        resolved = self._client.resolved(provider)
        stream = allow_stream and provider.stream and self._client.on_stream is not None
        params: Json = {
            "model": provider.model,
            "input": self.messages(Text.value(messages), self._client.provider_origin(provider), provider=provider),
            "stream": stream,
            # Stateless by design, not to save anything: the wire can retain the conversation and
            # let a later request name it by id, but session messages are the source of truth, a
            # sent message is irrevocable, and a resume must rebuild from the snapshot alone.
            # Server-held state moves the truth off the machine that owns it. Prefix caching is
            # unaffected -- it keys on the rendered prefix, not on stored conversations. See
            # DESIGN.md "Cache epochs and breakpoints".
            "store": False,
        }
        if provider.max_tokens > 0:
            params["max_output_tokens"] = provider.max_tokens
        if request_tools := [*responses_module.responses_tool_schemas(tools or []), *self._client.builtin_tools(resolved)]:
            params["tools"] = request_tools
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = True
        if prompt_cache_key := self._client.prompt_cache_key(provider, tools):
            params["prompt_cache_key"] = prompt_cache_key
        # Stateless requests return encrypted reasoning items by default, so the replay below needs
        # no `include`; effort goes through the request-recipe fold like the chat path, and a host
        # that defines an explicit "off" spelling still gets it when reasoning is off.
        if resolved.responses_reasoning and provider.reasoning == "off" and resolved.reasoning_effort is None:
            raise ModelError("reasoning off is not defined for this Responses model; use a supported effort or configure a documented provider endpoint")
        # Fold user extensions first. A catalog recipe may then add managed extra_body paths
        # without a later assignment replacing them, matching the Chat wire's precedence.
        if provider.extra_body and (extra_body := responses_module.responses_extra_body(provider.extra_body, params)):
            params["extra_body"] = extra_body
        self._client.apply_request(params, provider, resolved, wire="responses")
        if provider.temperature is not None and not resolved.suppress_temperature:
            params["temperature"] = provider.temperature
        omit_request_fields(params, provider.omit_body)
        client = self._client.client(provider=provider)
        if stream:
            result = await self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = await self._client.call_client(client, lambda: client.responses.create(**params), response_timeout=response_timeout)
            streamed = False
        self._client._record_usage(self._client.message_field(result, "usage"), billing=billing)
        assistant, calls, text = self.result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self._client.provider_origin(provider)
        return assistant, calls, text

    async def _stream(self, client: Any, params: Json) -> Any:
        """Consume a Responses stream, promoting completed text before tool arguments finish."""
        return await responses_module.reassemble_stream(
            client,
            params,
            message_field=self._client.message_field,
            raise_if_inactive=self._client._raise_if_request_inactive,
            emit=self._client._emit_stream,
            report_builtin_call=self._client.report_builtin_call,
        )

    def messages(self, messages: list[Json], origin: str = "", *, text_only: bool | None = None, provider: ProviderConfig | None = None) -> list[Json]:
        provider = provider if provider is not None else self._client.session.config.provider
        resolved = self._client.resolved(provider)
        text_only = self._client.session.image_route.is_text_only() if text_only is None else text_only
        return responses_module.responses_input(
            messages,
            origin,
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            reasoning_history=resolved.reasoning_history,
            latest_user_position=self._client.latest_user_position,
            text_only=text_only,
        )

    def estimation_payload(self, messages: list[Json], tools: list[Json] | None, builtin: list[Json]) -> Json:
        """The wire-shaped payload for one token estimate, matching what request() would send."""
        payload: Json = {"input": self.messages(Text.value(messages))}
        if request_tools := [*responses_module.responses_tool_schemas(tools or []), *builtin]:
            payload["tools"] = request_tools
        return payload

    def result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
        return responses_module.responses_result(
            result,
            streamed,
            message_field=self._client.message_field,
            dump_message_item=responses_module.dump_message_item,
            tool_call=self._client.tool_call,
            report_builtin_call=self._client.report_builtin_call,
            truncated_output_error=self._client.truncated_output_error,
            collect_sources=self._client.collect_sources,
        )


class AnthropicWire:
    """The Anthropic Messages wire."""

    def __init__(self, client: ModelClient):
        self._client = client

    async def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
        billing: Billing = Billing.MAIN,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        messages = Text.value(messages)
        params = omit_request_fields(self.params(messages, tools, provider), provider.omit_body)
        client = self._client.anthropic_client(provider=provider)
        stream = allow_stream and provider.stream and self._client.on_stream is not None
        if stream:
            result = await self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = await self._client.call_client(client, lambda: client.messages.create(**params), response_timeout=response_timeout)
            streamed = False
        self._client._record_usage(self._client.message_field(result, "usage"), billing=billing)
        assistant, calls, content = self.result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self._client.provider_origin(provider)
        return assistant, calls, content

    async def _stream(self, client: Any, params: Json) -> Any:
        """Consume Messages blocks and promote text once both text and tool blocks are known."""
        return await anthropic_module.reassemble_stream(
            client,
            params,
            message_field=self._client.message_field,
            raise_if_inactive=self._client._raise_if_request_inactive,
            emit=self._client._emit_stream,
            report_builtin_call=self._client.report_builtin_call,
        )

    def params(self, messages: list[Json], tools: list[Json] | None, provider: ProviderConfig | None = None) -> Json:
        provider = provider if provider is not None else self._client.session.config.provider
        return anthropic_module.anthropic_params(
            messages,
            tools,
            provider,
            self._client.resolved(provider),
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            builtin_tools=self._client.builtin_tools,
            apply_request=lambda params, entry, resolved: self._client.apply_request(params, entry, resolved, wire="anthropic"),
            latest_user_position=self._client.latest_user_position,
            text_only=self._client.session.image_route.is_text_only(),
        )

    def messages(self, messages: list[Json], origin: str = "", *, text_only: bool | None = None) -> list[Json]:
        text_only = self._client.session.image_route.is_text_only() if text_only is None else text_only
        resolved = self._client.resolved(self._client.session.config.provider)
        return anthropic_module.anthropic_messages(
            messages,
            origin,
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            reasoning_history=resolved.reasoning_history,
            latest_user_position=self._client.latest_user_position,
            text_only=text_only,
        )

    def estimation_payload(self, messages: list[Json], tools: list[Json] | None, builtin: list[Json]) -> Json:
        """The wire-shaped payload for one token estimate, matching what request() would send."""
        system = "\n\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "system").strip()
        payload: Json = {"system": system, "messages": self.messages(Text.value(messages))}
        if request_tools := [*anthropic_module.anthropic_tool_schemas(tools or []), *builtin]:
            payload["tools"] = request_tools
        return payload

    def assistant_blocks(self, message: Json, origin: str = "") -> list[Json]:
        return anthropic_module.anthropic_assistant_blocks(
            message,
            origin,
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
        )

    def result(self, result: Any, streamed: bool = False) -> tuple[Json, list[ToolCall], str]:
        return anthropic_module.anthropic_result(
            result,
            streamed,
            message_field=self._client.message_field,
            dump_message_item=responses_module.dump_message_item,
            tool_call=self._client.tool_call,
            report_builtin_call=self._client.report_builtin_call,
            truncated_output_error=self._client.truncated_output_error,
            collect_sources=self._client.collect_sources,
        )

    def sources(self, saved_content: list[Json]) -> list[Json]:
        """Sources from a Messages response: cited text first, then the raw search results."""
        return anthropic_module.anthropic_sources(saved_content, self._client.collect_sources)
