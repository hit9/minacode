"""Independent expected contracts for provider compatibility smoke tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderContract:
    id: str
    provider: str
    url: str
    model: str
    reasoning: str
    expected_api: str
    expected_path: str
    expected_body: dict[str, object]
    api: str = "auto"
    temperature: float | None = None
    absent_body_keys: tuple[str, ...] = ()


PROVIDER_CONTRACTS = (
    ProviderContract(
        id="openai-chat",
        provider="openai",
        url="https://api.openai.com/v1",
        model="gpt-5.6",
        reasoning="max",
        temperature=0.4,
        expected_api="chat",
        expected_path="/v1/chat/completions",
        expected_body={"model": "gpt-5.6", "reasoning_effort": "max"},
        absent_body_keys=("temperature",),
    ),
    ProviderContract(
        id="openrouter-chat",
        provider="openrouter",
        url="https://openrouter.ai/api/v1",
        model="vendor/model",
        reasoning="max",
        temperature=0.4,
        expected_api="chat",
        expected_path="/api/v1/chat/completions",
        expected_body={"model": "vendor/model", "reasoning": {"effort": "max"}, "temperature": 0.4},
    ),
    ProviderContract(
        id="opencode-deepseek-chat",
        provider="opencode",
        url="https://opencode.ai/zen/v1",
        model="deepseek-v4-flash",
        reasoning="medium",
        temperature=0.4,
        expected_api="chat",
        expected_path="/zen/v1/chat/completions",
        expected_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}},
        absent_body_keys=("temperature",),
    ),
    ProviderContract(
        id="opencode-gpt-responses",
        provider="opencode",
        url="https://opencode.ai/zen/v1",
        model="gpt-5.5",
        reasoning="max",
        temperature=0.4,
        expected_api="responses",
        expected_path="/zen/v1/responses",
        expected_body={"reasoning": {"effort": "xhigh"}, "temperature": 0.4},
    ),
    ProviderContract(
        id="opencode-claude-messages",
        provider="opencode",
        url="https://opencode.ai/zen/v1",
        model="claude-sonnet",
        reasoning="off",
        temperature=0.4,
        expected_api="anthropic",
        expected_path="/zen/v1/messages",
        expected_body={"model": "claude-sonnet", "temperature": 0.4},
    ),
    ProviderContract(
        id="deepseek-chat",
        provider="deepseek",
        url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        reasoning="medium",
        temperature=0.4,
        expected_api="chat",
        expected_path="/v1/chat/completions",
        expected_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}},
        absent_body_keys=("prompt_cache_key", "temperature"),
    ),
    ProviderContract(
        id="qwen-chat",
        provider="qwen",
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max-preview",
        reasoning="off",
        temperature=0.4,
        expected_api="chat",
        expected_path="/compatible-mode/v1/chat/completions",
        expected_body={"reasoning_effort": "none", "temperature": 0.4},
    ),
    ProviderContract(
        id="kimi-open-chat",
        provider="kimi_open",
        url="https://api.moonshot.ai/v1",
        model="kimi-k3",
        reasoning="medium",
        temperature=0.2,
        expected_api="chat",
        expected_path="/v1/chat/completions",
        expected_body={"reasoning_effort": "high"},
        absent_body_keys=("temperature",),
    ),
    ProviderContract(
        id="kimi-code-chat",
        provider="kimi_code",
        url="https://api.kimi.com/coding/v1",
        model="k3",
        reasoning="medium",
        temperature=0.2,
        expected_api="chat",
        expected_path="/coding/v1/chat/completions",
        expected_body={"reasoning_effort": "high", "temperature": 0.2},
    ),
    ProviderContract(
        id="zai-chat",
        provider="zai",
        url="https://api.z.ai/api/paas/v4",
        model="glm-5.2",
        reasoning="xhigh",
        temperature=0.6,
        expected_api="chat",
        expected_path="/api/paas/v4/chat/completions",
        expected_body={"reasoning_effort": "xhigh", "thinking": {"type": "enabled"}, "temperature": 0.6},
        absent_body_keys=("prompt_cache_key",),
    ),
    ProviderContract(
        id="bigmodel-chat",
        provider="bigmodel",
        url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-5.2",
        reasoning="max",
        temperature=0.6,
        expected_api="chat",
        expected_path="/api/paas/v4/chat/completions",
        expected_body={"reasoning_effort": "max", "thinking": {"type": "enabled"}, "temperature": 0.6},
        absent_body_keys=("prompt_cache_key",),
    ),
    ProviderContract(
        id="anthropic-messages",
        provider="anthropic",
        url="https://api.anthropic.com/v1/messages",
        model="claude-sonnet-4-7",
        reasoning="max",
        temperature=0.2,
        api="anthropic",
        expected_api="anthropic",
        expected_path="/v1/messages",
        expected_body={
            "model": "claude-sonnet-4-7",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        absent_body_keys=("temperature",),
    ),
)
