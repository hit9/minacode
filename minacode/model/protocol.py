"""Wire adapters: one complete adapter per provider wire.

Each adapter owns that wire's construction (messages/params/tool schemas), sending (request), and
parsing (result/sources). ModelClient keeps the shared request lifecycle and hands each adapter a
reference to itself at construction, so the client's shared services are bound once per wire
instead of re-injected on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import minacode.model.anthropic as anthropic_module
import minacode.model.chat as chat_module
import minacode.model.responses as responses_module
from minacode.base import PROVIDER_ORIGIN_KEY, Json, ModelError, Text, ToolCall
from minacode.config import ProviderConfig

if TYPE_CHECKING:
    from minacode.model.client import ModelClient


class WireProtocol(Protocol):
    """One provider wire's complete adapter: construction, sending, and parsing.

    Only `request` is required of every wire so far; the per-wire construction/parsing methods are
    added to the interface step by step as each adapter's logic moves out of ModelClient and the
    other adapters grow the same method (Parts C4c-C4e and B).
    """

    def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]: ...


class ChatWire:
    """The chat-completions wire (the default/else branch)."""

    def __init__(self, client: ModelClient):
        self._client = client

    def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        messages = self.messages(messages, provider=provider)
        resolved = provider.resolve()
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
        client = self._client.client(provider=provider)
        if stream:
            message, usage, finish_reason = self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
        else:
            response = self._client.call_client(client, lambda: client.chat.completions.create(**params), response_timeout=response_timeout)
            usage = getattr(response, "usage", None)
            message = response.choices[0].message
            finish_reason = str(self._client.message_field(response.choices[0], "finish_reason") or "")
        self._client._record_usage(usage)
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
            messages, provider, provider.resolve(), self._client.session.images, self._client.latest_user_position, text_only=text_only
        )

    def _stream(self, client: Any, params: Json) -> tuple[Json, Any, str]:
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
        return chat_module.reassemble_stream(
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

    def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        resolved = provider.resolve()
        stream = allow_stream and provider.stream and self._client.on_stream is not None
        params: Json = {
            "model": provider.model,
            "input": self.messages(Text.value(messages), self._client.provider_origin(provider)),
            "stream": stream,
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
        # no `include`; effort goes through the compatibility fold like the chat path.
        if resolved.responses_reasoning:
            if effort := resolved.reasoning_effort:
                params["reasoning"] = {"effort": effort}
            elif provider.reasoning == "off":
                raise ModelError("reasoning off is not defined for this Responses model; use a supported effort or configure a documented provider endpoint")
        if provider.temperature is not None and not resolved.suppress_temperature:
            params["temperature"] = provider.temperature
        if provider.extra_body and (extra_body := responses_module.responses_extra_body(provider.extra_body, params)):
            params["extra_body"] = extra_body
        client = self._client.client(provider=provider)
        if stream:
            result = self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self._client.call_client(client, lambda: client.responses.create(**params), response_timeout=response_timeout)
            streamed = False
        self._client._record_usage(self._client.message_field(result, "usage"))
        assistant, calls, text = self.result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self._client.provider_origin(provider)
        return assistant, calls, text

    def _stream(self, client: Any, params: Json) -> Any:
        """Consume a Responses stream, promoting completed text before tool arguments finish."""
        return responses_module.reassemble_stream(
            client,
            params,
            message_field=self._client.message_field,
            raise_if_inactive=self._client._raise_if_request_inactive,
            emit=self._client._emit_stream,
            report_builtin_call=self._client.report_builtin_call,
        )

    def messages(self, messages: list[Json], origin: str = "", *, text_only: bool | None = None) -> list[Json]:
        text_only = self._client.session.image_route.is_text_only() if text_only is None else text_only
        return responses_module.responses_input(
            messages,
            origin,
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            text_only=text_only,
        )

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

    def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig | None = None,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        provider = provider if provider is not None else self._client.session.config.provider
        messages = Text.value(messages)
        params = self.params(messages, tools, provider)
        client = self._client.anthropic_client(provider=provider)
        stream = allow_stream and provider.stream and self._client.on_stream is not None
        if stream:
            result = self._client.call_client(client, lambda: self._stream(client, params), response_timeout=response_timeout)
            streamed = True
        else:
            result = self._client.call_client(client, lambda: client.messages.create(**params), response_timeout=response_timeout)
            streamed = False
        self._client._record_usage(self._client.message_field(result, "usage"))
        assistant, calls, content = self.result(result, streamed)
        assistant[PROVIDER_ORIGIN_KEY] = self._client.provider_origin(provider)
        return assistant, calls, content

    def _stream(self, client: Any, params: Json) -> Any:
        """Consume Messages blocks and promote text once both text and tool blocks are known."""
        return anthropic_module.reassemble_stream(
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
            provider.resolve(),
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            builtin_tools=self._client.builtin_tools,
            text_only=self._client.session.image_route.is_text_only(),
        )

    def messages(self, messages: list[Json], origin: str = "", *, text_only: bool | None = None) -> list[Json]:
        text_only = self._client.session.image_route.is_text_only() if text_only is None else text_only
        return anthropic_module.anthropic_messages(
            messages,
            origin,
            provider_origin=self._client.provider_origin,
            replayable_echo=self._client.replayable_echo,
            images=self._client.session.images,
            text_only=text_only,
        )

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
