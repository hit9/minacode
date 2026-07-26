# Configuration

minacode reads a single TOML file, `~/.minacode/config.toml` by default. Generate a
commented starter with `minacode --init-config`, or point at another file with
`--config <path>`.

<span class="marker">Only the `[provider]` block is required.</span> Every other key falls back to
a built-in default, so a minimal config is just a provider. Inspect the resolved configuration
at any time with `/config`.

## Provider selection

Define one or more `[provider.<name>]` blocks and select one with `[provider] active`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

For provider fields, API protocols, live switching, reasoning, caching, strict tools, and
documented compatibility behavior, see [Providers](providers.md).

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
