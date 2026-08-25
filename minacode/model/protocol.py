"""Wire adapters: one complete adapter per provider wire.

Each adapter owns that wire's construction (messages/params/tool schemas), sending (request), and
parsing (result/sources). ModelClient keeps the shared request lifecycle and hands each adapter a
reference to itself at construction, so the client's shared services are bound once per wire
instead of re-injected on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import minacode.model.anthropic as anthropic_module
from minacode.base import PROVIDER_ORIGIN_KEY, Json, Text, ToolCall
from minacode.config import ProviderConfig
from minacode.model.responses import dump_message_item

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
        return self._client.chat_request(
            messages,
            tools,
            allow_stream=allow_stream,
            response_timeout=response_timeout,
            provider=provider,
            json_object=json_object,
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
        return self._client.responses_request(
            messages,
            tools,
            allow_stream=allow_stream,
            response_timeout=response_timeout,
            provider=provider,
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
            dump_message_item=dump_message_item,
            tool_call=self._client.tool_call,
            report_builtin_call=self._client.report_builtin_call,
            truncated_output_error=self._client.truncated_output_error,
            collect_sources=self._client.collect_sources,
        )

    def sources(self, saved_content: list[Json]) -> list[Json]:
        """Sources from a Messages response: cited text first, then the raw search results."""
        return anthropic_module.anthropic_sources(saved_content, self._client.collect_sources)
