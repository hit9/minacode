# Configuration

nanocode reads a single TOML file, `~/.nanocode/config.toml` by default. Generate a
commented starter with `nanocode --init-config`, or point at another file with
`--config <path>`.

Only the `[provider]` block is required. Every other key falls back to a built-in default, so
a minimal config is just a provider. Inspect the resolved configuration at any time with
`/config`.

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
| `api` | `auto` | Request format: `auto`, `anthropic`, `openai`/`chat`. `auto` picks the right one for the host and model. |
| `reasoning` | `medium` | Reasoning effort: `off`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `timeout` | `180` | Request timeout, in seconds |
| `available_models` | — | Models offered by `/model`'s picker |
| `temperature` | — | Sampling temperature (omit to use the server default) |
| `max_tokens` | — | Cap on output tokens (omit to use the server default) |
| `strict_tools` | `false` | Emit strict tool-call schemas where supported (OpenAI, DeepSeek); toggle live with `/strict` |

For well-known hosts, `auto` settings pick sensible defaults for you, so `url`, `key`, and
`model` are usually all you need. Anything you set explicitly is always respected. Tested
with DeepSeek, OpenCode, Alibaba Cloud, and ZenMux; other OpenAI-compatible and Anthropic
endpoints work too.

## Runtime

Optional; the defaults shown are used when omitted.

```toml
[runtime]
yolo = false               # start with confirmations off
max_context_tokens = 128000  # budget before auto-compaction
max_agent_steps = 200      # max tool steps per turn
shell_timeout = 60         # default timeout for shell tools, in seconds
```

You can change these live for the current session with `/set runtime.KEY VALUE` (and provider
values with `/set provider.KEY VALUE`). `/yolo` toggles `yolo`.

## Data location

```toml
[paths]
data_dir = "~/.nanocode"   # sessions, code index, OAuth tokens
```

Sessions live under `<data_dir>/sessions/`.

## MCP servers

MCP servers are configured under `[mcp.<name>]` blocks — see [MCP](mcp.md) for the full
reference.

## Skills

Skills aren't configured in TOML; they're discovered from the filesystem — see
[Skills](skills.md).
