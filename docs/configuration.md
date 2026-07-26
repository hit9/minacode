# Configuration

minacode reads a single TOML file, `~/.minacode/config.toml` by default. Generate a
commented starter with `minacode --init-config`, or point at another file with
`--config <path>`.

<span class="marker">Only the `[provider]` block is required.</span> Every other key falls back to
a built-in default, so a minimal config is just a provider. Inspect the resolved configuration
at any time with `/config`.

## Providers

minacode supports OpenAI-compatible Chat Completions and Responses APIs, plus the Anthropic
Messages API. Define one or more `[provider.<name>]` blocks and select one with
`[provider] active`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

These three fields are enough for most endpoints. minacode selects the usual protocol and applies
only necessary, documented compatibility adjustments. Explicit settings always take precedence.
Use `/config` to inspect the result.

Define additional blocks to use more providers. Switch between them with `/provider [NAME]`, and
switch the active model with `/model [MODEL]`.

### API protocol

Leave `api = "auto"` unless your endpoint needs an explicit protocol:

| Value | Meaning |
|---|---|
| `auto` | Infer the protocol when possible; otherwise use Chat Completions |
| `chat` | OpenAI-compatible Chat Completions |
| `responses` | OpenAI-compatible Responses |
| `anthropic` | Anthropic-compatible Messages |

A URL ending in `/chat/completions`, `/responses`, or `/messages` also selects that protocol.

### Optional provider settings

Most users can leave these unset.

| Key | Default | Meaning |
|---|---|---|
| `api` | `auto` | API protocol shown above |
| `image_input` | `auto` | Image capability: learn automatically, force `on`, or disable with `off` |
| `reasoning` | `medium` | Reasoning effort; change it during a session with `/reason` |
| `available_models` | — | Additional models shown by `/model` |
| `temperature` | — | Sampling temperature; omitted by default |
| `max_tokens` | — | Output-token cap and reserved compaction space |
| `timeout` | `120` | Request timeout in seconds |
| `prompt_cache_key` | `auto` | Stable prompt-cache key; set `off` to omit it |
| `strict_tools` | `false` | Request strict function schemas where supported; toggle with `/strict` |
| `extra_body` | `{}` | Extra fields for an OpenAI-compatible request body |
| `chat_reasoning` | `auto` | Provider-specific Chat reasoning format; normally leave on `auto` |

Unknown OpenAI-compatible endpoints stay on the generic path. If automatic protocol selection is
wrong for an endpoint, set `api` explicitly. `/status` shows the active model and cache usage
reported by the provider.

With `image_input = "auto"`, minacode sends attached images using the selected standard API. A
successful image request is remembered for that provider and model during the session; only an
explicit image-not-supported response disables later image submissions. Set `on` or `off` when the
endpoint's capability is already known. Historical images remain readable as text labels after
switching to a provider or model with image input disabled.

## Runtime

Optional; the defaults shown are used when omitted.

| Key | Default | Meaning |
|---|---|---|
| `yolo` | `false` | Start without confirmation prompts |
| `max_context_tokens` | `245760` (240K) | Total context ceiling used to calculate the automatic-compaction budget |
| `max_agent_steps` | `200` | Maximum tool steps in one turn |
| `shell_timeout` | `60` | Maximum shell-command lifetime, in seconds |
| `bash_wait_timeout` | `10` | Foreground wait before a running command becomes a background job; `0` disables promotion |
| `max_parallel_tools` | `4` | Maximum read-only tool calls executed concurrently; `1` disables parallelism |
| `session_retention_days` | `7` | Delete inactive saved sessions older than this at startup; `0` keeps them indefinitely |
| `theme` | `auto` | Terminal theme: `auto`, `light`, or `dark`; overridden by `--theme` |

Selected tuning values can be changed for the current session with `/set` (Tab completion
lists the supported keys). `/yolo` toggles `yolo`.

## Data location

```toml
[paths]
data_dir = "~/.minacode"   # sessions, code index, OAuth tokens, user skills, update cache
```

Sessions live under `<data_dir>/projects/<project>/`, one directory per working directory. Each
holds that project's session logs and a `latest` pointer, so a resume stays scoped to the project
it belongs to. A project directory is removed once its last session expires.
