"""`omit_body`: request fields an endpoint rejects, dropped on the way out."""

import json

import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory, _session

from wizolt.base import ConfigError
from wizolt.config import ProviderConfig
from wizolt.model import ModelClient
from wizolt.model.protocol import omit_request_fields

CHAT_BODY = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.5",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
}
ANTHROPIC_BODY = {"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "hi"}]}


async def sent_body(model: ModelClient, factory, monkeypatch, attribute: str = "client") -> dict:
    monkeypatch.setattr(model, attribute, factory)
    await model.request([{"role": "user", "content": "hi"}], None)
    return json.loads(factory.calls[0].content)


async def test_a_named_field_never_reaches_the_chat_request(tmp_path, monkeypatch):
    """The reason the setting exists: a gateway answers 400 for a field wizolt sends, and
    `extra_body` can only add fields."""
    s = _session(tmp_path, url="https://api.openai.com/v1", model="gpt-5.5", stream=False, reasoning="high")
    body = await sent_body(ModelClient(s), _MockClientFactory([(200, CHAT_BODY)]), monkeypatch)
    assert body["reasoning_effort"] == "high"

    s = _session(tmp_path, url="https://api.openai.com/v1", model="gpt-5.5", stream=False, reasoning="high", omit_body=("reasoning_effort",))
    body = await sent_body(ModelClient(s), _MockClientFactory([(200, CHAT_BODY)]), monkeypatch)
    assert "reasoning_effort" not in body
    assert body["messages"] and body["model"] == "gpt-5.5"


async def test_a_named_field_never_reaches_the_anthropic_request(tmp_path, monkeypatch):
    s = _session(tmp_path, url="https://api.anthropic.com", api="anthropic", model="claude-sonnet-4-6", stream=False, reasoning="off", temperature=0.2)
    body = await sent_body(ModelClient(s), _AnthropicMockClientFactory([(200, ANTHROPIC_BODY)]), monkeypatch, attribute="anthropic_client")
    assert body["temperature"] == 0.2

    s = _session(
        tmp_path,
        url="https://api.anthropic.com",
        api="anthropic",
        model="claude-sonnet-4-6",
        stream=False,
        reasoning="off",
        temperature=0.2,
        omit_body=("temperature",),
    )
    body = await sent_body(ModelClient(s), _AnthropicMockClientFactory([(200, ANTHROPIC_BODY)]), monkeypatch, attribute="anthropic_client")
    assert "temperature" not in body


async def test_a_field_is_dropped_wherever_the_request_puts_it(tmp_path, monkeypatch):
    """A provider's 400 names the field, not the place wizolt happened to put it, so a name
    configured here is matched in `extra_body` as well as at the top level."""
    s = _session(tmp_path, url="https://gw.example/v1", model="m", stream=False, extra_body={"enable_search": True}, omit_body=("enable_search",))
    body = await sent_body(ModelClient(s), _MockClientFactory([(200, CHAT_BODY)]), monkeypatch)

    assert "enable_search" not in body
    assert "extra_body" not in body


def test_omitting_the_last_extra_body_field_leaves_no_empty_object():
    provider = ProviderConfig(omit_body=("enable_search",))
    assert omit_request_fields({"model": "m", "extra_body": {"enable_search": True}}, provider.omit_body) == {"model": "m"}

    kept = omit_request_fields(
        {"model": "m", "extra_body": {"enable_search": True, "keep": 1}},
        ProviderConfig(omit_body=("enable_search",)).omit_body,
    )
    assert kept == {"model": "m", "extra_body": {"keep": 1}}


def test_the_fields_carrying_the_request_itself_are_refused():
    """Dropping one of these does not adjust a request, it empties it, and the provider error
    would point anywhere but at this setting."""
    for name in ("model", "messages", "input", "stream"):
        with pytest.raises(ConfigError):
            ProviderConfig.from_dict({"omit_body": [name]})

    assert ProviderConfig.from_dict({}).omit_body == ()
    assert ProviderConfig.from_dict({"omit_body": "reasoning_effort, stream_options"}).omit_body == ("reasoning_effort", "stream_options")


def test_an_unsent_field_is_not_an_error():
    """Named fields are what an endpoint rejects, not what wizolt promises to send: a name that
    never appears must stay harmless as the request shape changes."""
    assert omit_request_fields({"model": "m"}, ProviderConfig(omit_body=("nothing_like_this",)).omit_body) == {"model": "m"}
