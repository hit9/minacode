"""Offline provider contracts through the real SDK serializers and mocked HTTP transport."""

import json

import anthropic
import openai
import pytest
from model_harness import _AnthropicMockClientFactory, _MockClientFactory, _session
from provider_cases import PROVIDER_CONTRACTS, ProviderContract

from wizolt.model import ModelClient


def _chat_response(model: str) -> dict:
    return {
        "id": "chatcmpl-contract",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }


def _responses_response(model: str) -> dict:
    return {
        "id": "resp_contract",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _anthropic_response(model: str) -> dict:
    return {
        "id": "msg_contract",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _assert_subset(actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        assert key in actual
        if isinstance(value, dict):
            assert isinstance(actual[key], dict)
            _assert_subset(actual[key], value)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("case", PROVIDER_CONTRACTS, ids=lambda case: case.id)
def test_provider_wire_contracts_are_serialized_by_the_real_sdks(tmp_path, monkeypatch, case: ProviderContract):
    session = _session(
        tmp_path,
        url=case.url,
        model=case.model,
        api=case.api,
        chat_reasoning=case.chat_reasoning,
        reasoning=case.reasoning,
        temperature=case.temperature,
        stream=False,
    )
    model = ModelClient(session)
    resolved = session.catalog.policy.resolve(session.config.provider)

    assert resolved.api == case.expected_api
    if case.expected_api == "anthropic":
        factory = _AnthropicMockClientFactory([(200, _anthropic_response(case.model))])
        monkeypatch.setattr(anthropic, "Anthropic", factory)
        monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: pytest.fail("provider resolved to OpenAI SDK instead of Anthropic"))
    else:
        response = _responses_response(case.model) if case.expected_api == "responses" else _chat_response(case.model)
        factory = _MockClientFactory([(200, response)])
        monkeypatch.setattr(openai, "OpenAI", factory)
        monkeypatch.setattr(anthropic, "Anthropic", lambda **_kwargs: pytest.fail("provider resolved to Anthropic instead of OpenAI SDK"))

    model.request([{"role": "user", "content": "hello"}], [])

    assert len(factory.calls) == 1
    request = factory.calls[0]
    assert request.url.path == case.expected_path
    body = json.loads(request.content)
    _assert_subset(body, case.expected_body)
    for key in case.absent_body_keys:
        assert key not in body
