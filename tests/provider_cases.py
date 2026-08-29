"""Representative wire contracts independent of the provider catalog's fact table."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderContract:
    id: str
    url: str
    model: str
    reasoning: str
    expected_api: str
    expected_path: str
    expected_body: dict[str, object]
    api: str = "auto"
    chat_reasoning: str = "auto"
    temperature: float | None = None
    absent_body_keys: tuple[str, ...] = ()


PROVIDER_CONTRACTS = (
    ProviderContract(
        id="chat",
        url="https://gateway.example/v1",
        model="custom-chat",
        reasoning="max",
        chat_reasoning="reasoning_effort",
        temperature=0.4,
        expected_api="chat",
        expected_path="/v1/chat/completions",
        expected_body={"model": "custom-chat", "reasoning_effort": "max", "temperature": 0.4},
    ),
    ProviderContract(
        id="responses",
        url="https://gateway.example/v1",
        model="custom-responses",
        reasoning="high",
        api="responses",
        temperature=0.4,
        expected_api="responses",
        expected_path="/v1/responses",
        expected_body={"model": "custom-responses", "reasoning": {"effort": "high"}, "temperature": 0.4},
    ),
    ProviderContract(
        id="messages",
        url="https://gateway.example/v1",
        model="custom-messages",
        reasoning="off",
        api="anthropic",
        temperature=0.4,
        expected_api="anthropic",
        expected_path="/v1/messages",
        expected_body={"model": "custom-messages", "temperature": 0.4},
    ),
)
