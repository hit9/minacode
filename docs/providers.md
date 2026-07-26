# Providers

minacode talks to OpenAI-compatible Chat Completions and Responses APIs, and to the
Anthropic Messages API. The default path is deliberately generic: for an unknown host,
minacode sends standard Chat Completions requests and only applies behavior you configure.
Small compatibility profiles exist for documented provider differences.

## Provider blocks

Start with the [minimal provider configuration](configuration.md#provider-selection). Each
`[provider.<name>]` block describes one endpoint, and its name is local to your config. Use
`/provider [NAME]` to switch blocks inside a session, `/model [MODEL]` to switch models, and
`/config` to inspect configured and resolved values.

### Provider options

| Key | Default | Meaning |
|---|---|---|
| `url` | — | API base URL; a trailing `/chat/completions`, `/responses`, or `/messages` is accepted and stripped before SDK use |
| `key` | — | API key |
| `model` | — | Model name |
| `api` | `auto` | Request format: `auto`, `chat`, `responses`, or `anthropic` |
| `reasoning` | `medium` | Normalized reasoning effort: `off`, `minimal`, `low`, `medium`, `high`, or `xhigh` |
| `chat_reasoning` | `auto` | Chat-only reasoning wire format; normally leave this on `auto` |
| `prompt_cache_key` | `auto` | Stable prompt-cache key; use `off` to omit it or provide a custom key |
| `timeout` | `120` | Request timeout in seconds |
| `available_models` | — | Models shown by `/model` in addition to models discovered elsewhere |
| `temperature` | — | Sampling temperature; omit it to use the provider default |
| `max_tokens` | — | Output-token cap; also reserves that space during automatic compaction |
| `strict_tools` | `false` | Emit strict function schemas on hosts known to support them; toggle with `/strict` |
| `extra_body` | `{}` | Provider extension fields merged into an OpenAI-compatible request body |

Explicit settings win over compatibility profiles. In most configurations, `url`, `key`,
and `model` are enough.

## API protocols

| Value | Request path | Use it when |
|---|---|---|
| `auto` | Resolved from the endpoint suffix or a documented compatibility profile; otherwise Chat Completions | You want minacode to select the normal protocol |
| `chat` | `/chat/completions` | The provider exposes an OpenAI-compatible Chat Completions API |
| `responses` | `/responses` | The provider exposes an OpenAI-compatible Responses API |
| `anthropic` | `/messages` | The provider exposes an Anthropic-compatible Messages API |

An explicit endpoint suffix also selects the matching protocol while preserving the SDK base
URL. For example, a URL ending in `/responses` resolves to `responses`; setting `api`
explicitly takes precedence.

Responses requests use `store = false`. minacode keeps the returned reasoning and function-call
items in its own transcript and replays them on later requests, so tool loops and resumed
sessions remain stateless. See OpenAI's [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).

## Reasoning

`reasoning` is minacode's provider-independent effort setting. Change it live with
`/reason [EFFORT]`. Native Responses and Anthropic requests receive their standard reasoning
fields; Chat-compatible providers receive the documented form selected by `chat_reasoning`.

Leave `chat_reasoning = "auto"` unless an otherwise unknown provider needs a specific wire
format. Supported explicit formats are `off`, `reasoning`, `reasoning_effort`, `thinking`,
`thinking_toggle`, `thinking_effort`, `enable_thinking`, and `mandatory_thinking` — every form
`auto` can select is also selectable by hand, for gateways and model names the rules do not
recognize. A compatibility profile may also suppress `temperature` where a provider's thinking
mode fixes or rejects it.

## Caching and strict tools

`prompt_cache_key = "auto"` creates a stable key from the workspace, host, model, protocol, and
tool set. Providers that cache automatically can disable this through their compatibility
profile. `/status` reports the cache-token counts returned by the provider; it does not estimate
cache hits locally.

`strict_tools = true` only becomes active for a host and protocol known to support strict
function schemas. `/strict` reports whether the setting is active. On other providers the
normal schemas remain unchanged.

## Compatibility profiles

Profiles are domain-wide where the provider contract is domain-wide, and model-family rules are
used only when the provider documents different behavior by family.

| Host | Automatic compatibility behavior | Official reference |
|---|---|---|
| `api.openai.com` | `reasoning_effort` for `o*` and `gpt-5*` Chat models; strict function schemas | [Reasoning](https://developers.openai.com/api/docs/guides/reasoning), [strict mode](https://developers.openai.com/api/docs/guides/function-calling#strict-mode) |
| `openrouter.ai` | OpenRouter's top-level `reasoning` object | [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) |
| `opencode.ai` | Anthropic Messages routing for Claude and Qwen families on the shared Zen endpoint | [OpenCode Zen](https://opencode.ai/docs/zen) |
| `api.deepseek.com` | DeepSeek thinking controls, automatic prefix caching, and the beta endpoint when strict tools are active | [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/), [tool calls](https://api-docs.deepseek.com/guides/tool_calls) |
| `*.aliyuncs.com` | `reasoning_effort` for the documented Qwen3.8 family, including `none` for reasoning off | [Qwen OpenAI Chat](https://docs.qwencloud.com/api-reference/chat/openai-chat) |
| `*.moonshot.ai`, `*.moonshot.cn` | Kimi open-platform thinking tiers, fixed-temperature behavior, strict tools, and prompt-cache keys | [Thinking models](https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model), [model parameters](https://platform.kimi.ai/docs/api/models-overview) |
| `*.kimi.com` | Kimi Code model-family reasoning controls; kept separate from the open platform | [Kimi Code models](https://www.kimi.com/code/docs/kimi-code/models.html) |
| `*.z.ai`, `*.bigmodel.cn` | GLM thinking controls and automatic context caching | [Z.AI thinking](https://docs.z.ai/guides/capabilities/thinking), [BigModel cache](https://docs.bigmodel.cn/cn/guide/capabilities/cache) |

This table documents compatibility behavior, not a fixed list of supported providers. Other
OpenAI-compatible and Anthropic-compatible endpoints work through the generic protocol paths.

## Troubleshooting

- Run `/config` to see the configured and resolved API modes.
- Run `/status` to confirm the active provider/model and returned cache usage.
- Set `api` explicitly if an endpoint cannot be inferred safely.
- Keep a provider's API key paired with its documented base URL; keys from different billing
  products or regions are not necessarily interchangeable.
