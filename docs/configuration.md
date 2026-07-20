# Configuration

nanocode reads a single TOML file, `~/.nanocode/config.toml` by default. Generate a
commented starter with `nanocode --init-config`, or point at another file with
`--config <path>`.

<span class="marker">Only the `[provider]` block is required.</span> Every other key falls back to
a built-in default, so a minimal config is just a provider. Inspect the resolved configuration
at any time with `/config`.

## Providers

Define one or more `[provider.<name>]` blocks and pick the active one with
`[provider] active`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

Switch providers within a session with `/provider [NAME]` and models with `/model [MODEL]`.

### Provider options

| Key | Default | Meaning |
|---|---|---|
| `url` | — | Base URL of the API |
| `key` | — | API key |
| `model` | — | Model name |
| `api` | `auto` | Request format: `auto`, `anthropic`, or `chat`. `auto` picks the right one for the host and model. |
| `reasoning` | `medium` | Reasoning effort: `off`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `chat_reasoning` | `auto` | Chat-API reasoning format: `auto`, `off`, `reasoning`, `reasoning_effort`, `thinking`, or `enable_thinking` |
| `prompt_cache_key` | `auto` | Stable prompt-cache key; use `off` to disable or provide a custom key |
| `timeout` | `180` | Request timeout, in seconds |
| `available_models` | — | Models offered by `/model`'s picker |
| `temperature` | — | Sampling temperature (omit to use the server default) |
| `max_tokens` | — | Cap on output tokens (omit to use the server default) |
| `strict_tools` | `false` | Emit strict tool-call schemas where supported (OpenAI, DeepSeek); toggle live with `/strict` |

For well-known hosts, `auto` settings pick sensible defaults for you, so
<span class="marker">`url`, `key`, and `model` are usually all you need</span>. Anything you set
explicitly is always respected. Tested with DeepSeek, OpenCode, Alibaba Cloud, and ZenMux;
other OpenAI-compatible and Anthropic endpoints work too.

## Runtime

Optional; the defaults shown are used when omitted.

| Key | Default | Meaning |
|---|---|---|
| `yolo` | `false` | Start without confirmation prompts |
| `max_context_tokens` | `128000` | Context budget before automatic compaction |
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
data_dir = "~/.nanocode"   # sessions, code index, OAuth tokens, user skills, update cache
```

Sessions live under `<data_dir>/projects/<project>/`, one directory per working directory. Each
holds that project's session logs and a `latest` pointer, so a resume stays scoped to the project
it belongs to. A project directory is removed once its last session expires.
