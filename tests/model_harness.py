"""Shared harness for the ModelClient test modules.

The mock client factories intercept OpenAI/Anthropic SDK HTTP calls with httpx.MockTransport so
the wire formats can be exercised without hitting real providers."""

import json

import httpx
from anthropic import Anthropic
from openai import OpenAI

from minacode.config import (
    Config,
    ProviderConfig,
)
from minacode.session import Session


class _MockClientFactory:
    """Factory that returns a fresh OpenAI client on each call, all sharing one request log."""

    def __init__(self, responses: list, base_url: str = "http://test"):
        self.responses = list(responses)
        self.calls: list[httpx.Request] = []
        self.base_url = base_url

    def _next_response(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, httpx.Response):
            return response
        if isinstance(response, int):
            return httpx.Response(response)
        status, body = response
        return httpx.Response(status, json=body)

    def __call__(self, **kwargs) -> OpenAI:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.Client(transport=transport)
        return OpenAI(
            api_key="sk-test",
            base_url=kwargs.get("base_url", self.base_url),
            http_client=http_client,
            max_retries=0,
        )


class _StreamClientFactory:
    def __init__(self, events: list[dict], base_url: str = "http://test", failures: int = 0):
        self.events = events
        self.calls: list[httpx.Request] = []
        self.base_url = base_url
        self.failures = failures

    def __call__(self, **kwargs) -> OpenAI:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            if self.failures:
                self.failures -= 1
                return httpx.Response(500, json={"error": {"message": "temporary failure", "type": "server_error"}})
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in self.events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


class _AnthropicMockClientFactory(_MockClientFactory):
    """Factory that returns fresh Anthropic clients over the shared mocked response queue."""

    def __call__(self, **kwargs) -> Anthropic:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.Client(transport=transport)
        return Anthropic(
            api_key="sk-test",
            base_url=kwargs.get("base_url", self.base_url),
            http_client=http_client,
            max_retries=0,
        )


class _AnthropicStreamClientFactory:
    def __init__(self, events: list[tuple[str, dict]], base_url: str = "http://test"):
        self.events = events
        self.calls: list[httpx.Request] = []
        self.base_url = base_url

    def __call__(self, **kwargs) -> Anthropic:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            body = "".join(f"event: {name}\ndata: {json.dumps(event)}\n\n" for name, event in self.events)
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return Anthropic(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


def _session(tmp_path, **provider_kwargs):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    provider_kwargs.setdefault("model", "gpt-4")
    provider_kwargs.setdefault("url", "http://test")
    provider_kwargs.setdefault("key", "sk-test")
    config.providers = {"default": ProviderConfig(**provider_kwargs)}
    return Session(cwd=str(tmp_path), config=config)
