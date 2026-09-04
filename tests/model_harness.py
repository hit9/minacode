"""Shared harness for the ModelClient test modules.

The mock client factories intercept OpenAI/Anthropic SDK HTTP calls with httpx.MockTransport so
the wire formats can be exercised without hitting real providers."""

import json

import anthropic._base_client as _bc
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# Anthropic 1.x validates the client against its own vendored httpx2 module, so the mock
# transport, the requests it delivers, and the responses built from the fixture queue must
# all come from that same module. Older SDKs use plain httpx; fall back to it.
sdk_httpx = getattr(_bc, "httpx2", None) or httpx

from wizolt.config import (
    Config,
    ProviderConfig,
)
from wizolt.model import resilience
from wizolt.session import Session, bootstrap_features


class _MockClientFactory:
    """Factory that returns a fresh async OpenAI client on each call, all sharing one request log."""

    # The httpx module whose Request/Response types this factory speaks. The OpenAI path uses
    # plain httpx; the Anthropic subclass switches to the SDK's own module.
    response_module = httpx

    def __init__(self, responses: list, base_url: str = "http://test"):
        self.responses = list(responses)
        self.calls: list[httpx.Request] = []
        self.base_url = base_url

    def _next_response(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, self.response_module.Response):
            return response
        if isinstance(response, int):
            return self.response_module.Response(response)
        status, body = response
        return self.response_module.Response(status, json=body)

    def __call__(self, **kwargs) -> AsyncOpenAI:
        transport = httpx.MockTransport(self._next_response)
        http_client = httpx.AsyncClient(transport=transport)
        return AsyncOpenAI(
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

    def __call__(self, **kwargs) -> AsyncOpenAI:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            if self.failures:
                self.failures -= 1
                return httpx.Response(500, json={"error": {"message": "temporary failure", "type": "server_error"}})
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in self.events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return AsyncOpenAI(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


class _AnthropicMockClientFactory(_MockClientFactory):
    """Factory that returns fresh async Anthropic clients over the shared mocked response queue."""

    response_module = sdk_httpx

    def __call__(self, **kwargs) -> AsyncAnthropic:
        transport = sdk_httpx.MockTransport(self._next_response)
        http_client = sdk_httpx.AsyncClient(transport=transport)
        return AsyncAnthropic(
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

    def __call__(self, **kwargs) -> AsyncAnthropic:
        def respond(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            body = "".join(f"event: {name}\ndata: {json.dumps(event)}\n\n" for name, event in self.events)
            return sdk_httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        http_client = sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(respond))
        return AsyncAnthropic(api_key="sk-test", base_url=self.base_url, http_client=http_client, max_retries=0)


def record_backoff(monkeypatch, on_wait=None):
    """Record every backoff delay the retry policy asks for, and spend none of it.

    The wait itself is one cancellable `await asyncio.sleep(delay)`, so what a timing test can
    observe -- and all these tests ever asserted -- is the delay the policy chose. `on_wait` is
    called before each recorded wait, for tests that sample the status facts published with it.
    Returns a reader for the delays, in order."""

    delays: list[float] = []
    retry_delay = resilience.retry_delay

    def recorded(error, attempt):
        delay = retry_delay(error, attempt)
        delays.append(delay)
        if on_wait is not None:
            on_wait()
        return 0.0

    monkeypatch.setattr(resilience, "retry_delay", recorded)
    return lambda: list(delays)


def async_events(items):
    """The async iterator an SDK stream hands the wires, built from a synchronous test iterable."""

    class _Iterator:
        def __init__(self):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    return _Iterator()


def async_create(build):
    """Turn a synchronous fake `create(**params)` into the coroutine function the async SDK exposes.

    `build` returns whatever the real call would resolve to: a response object, or an iterable of
    stream events, which is adapted to the async iterator the wire consumes."""

    async def create(**params):
        result = build(**params)
        return async_events(result) if hasattr(result, "__iter__") and not hasattr(result, "__aiter__") else result

    return create


class AsyncStreamContext:
    """The async context-managed stream Anthropic's Messages SDK returns, over a test iterable."""

    def __init__(self, events, final_message=None):
        self._events = events
        self._final_message = final_message if final_message is not None else {"content": []}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def __aiter__(self):
        return async_events(self._events)

    async def get_final_message(self):
        return self._final_message


class AsyncCloseable:
    """A stand-in provider client whose close() is awaited, like the async SDK clients'."""

    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def _session(tmp_path, **provider_kwargs):
    config = Config()
    config.data_dir = str(tmp_path / "data")
    provider_kwargs.setdefault("model", "gpt-4")
    provider_kwargs.setdefault("url", "http://test")
    provider_kwargs.setdefault("key", "sk-test")
    config.providers = {"default": ProviderConfig(**provider_kwargs)}
    session = Session(cwd=str(tmp_path), config=config)
    bootstrap_features(session)
    return session
