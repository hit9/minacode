"""Wire adapters: one complete adapter per provider wire.

Each adapter owns that wire's construction (messages/params/tool schemas), sending (request), and
parsing (result/sources). ModelClient keeps the shared request lifecycle and hands each adapter a
reference to itself at construction, so the client's shared services are bound once per wire
instead of re-injected on every call.

The request method is the dispatch seam api_request keys on; the remaining construction and
parsing methods are introduced step by step (Parts C4b-C4e and B) as their logic moves out of
ModelClient, so for now each wire is a thin forwarder back to the client's methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minacode.base import Json, ToolCall
from minacode.config import ProviderConfig

if TYPE_CHECKING:
    from minacode.model.client import ModelClient


class WireProtocol(Protocol):
    """One provider wire's complete adapter: construction, sending, and parsing."""

    def request(
        self,
        messages: list[Json],
        tools: list[Json] | None,
        *,
        provider: ProviderConfig,
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
        provider: ProviderConfig,
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
        provider: ProviderConfig,
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
        provider: ProviderConfig,
        allow_stream: bool = True,
        response_timeout: float | None = None,
        json_object: bool = False,
    ) -> tuple[Json, list[ToolCall], str]:
        return self._client.anthropic_request(
            messages,
            tools,
            allow_stream=allow_stream,
            response_timeout=response_timeout,
            provider=provider,
        )
