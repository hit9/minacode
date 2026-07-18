# Slash commands

Type `/` commands at the prompt to inspect state, switch models, manage the
session, or configure runtime behavior on the fly. Run `/help` for the built-in
reference.

## Looking around

### `/status`

Shows everything about the runtime at a glance:

- workspace path and session id
- active provider, model, API kind, and reasoning effort
- context window fill percentage
- conversation history length, current turn messages, stored tool results
- prompt-cache hit ratio (total and last request)
- code index state
- background jobs count
- whether an update is available

Run it when you want to know "where am I right now."

### `/diff`

Opens an interactive diff viewer with two tabs:

- **Latest** — what changed during the most recent turn
- **Session** — the net diff for everything since the session began

Navigate with `j`/`k`, `g`/`G`, and `/` search; press `Esc` to close. Outside of
interactive mode it prints the diffs as plain text.

### `/ps`

Lists active background jobs — shell commands that outlive the initial `Bash` call
(see [Background jobs](tools.md#built-in-tools)). Each row shows the job id,
state, command, and elapsed time.

### `/skills`

Lists every installed [skill](skills.md) by name, source, and description. Skills
are loaded with `Skill(name)` or referenced inline with `$name`.

### `/config`

Shows the active configuration: every provider block, runtime settings, and their
resolved values. Useful for verifying what the agent sees right now.

## The code index

### `/index [force]`

Build or rebuild the code symbol index that powers `InspectCode`. The first build
walks every source file; later syncs are fast. Add `force` to rebuild from scratch.
See [Code symbol index](code-index.md) for a full explanation of what the index is
and how it stays current.

## Switching models

### `/provider [NAME]`

Show or switch the active provider. Without an argument it lists every configured
provider (from your [configuration](configuration.md#providers)) and marks the
active one — use the arrow keys to pick one interactively. With a name it switches
immediately.

### `/model [MODEL]`

Show or switch the model for the current provider. Without an argument it opens an
interactive picker showing configured models plus any models discovered from the
provider's API. With a name it sets the model directly.

Changing the model also prompts you to pick a reasoning effort.

### `/reason [EFFORT]`

Show or set the reasoning effort for providers that support it (OpenAI o-series and
DeepSeek R1). Accepted values:

| Level | Description |
|---|---|
| `off` | Disable reasoning |
| `minimal` | Minimal reasoning |
| `low` | Low effort |
| `medium` | Medium effort (default) |
| `high` | High effort |
| `xhigh` | Maximum effort |

Without an argument it opens a picker to choose interactively.

## Managing the session

### `/compact`

Summarize and shrink the conversation immediately. nanocode keeps long sessions
within a working budget on its own, but `/compact` lets you trim on demand. After
compaction a summary of earlier context is preserved so the agent retains the
important parts.

### `/yolo`

Toggle confirmation prompts for mutating tools. When on, the agent edits files and
runs commands without asking. See [Safety](safety.md) before turning this off
permanently.

### `/strict`

Toggle strict tool-call schemas. Strict mode emits the `strict: true` field in
OpenAI/DeepSeek function-calling requests. Some models require it; some reject
requests that include it.

### `/set KEY VALUE`

Set a supported configuration value for the current session without editing the
config file. Covers `provider.*` and `runtime.*` keys. Example:

```
/set provider.model deepseek-v4-flash
```

### `/resend`

Re-send the in-flight model request. Type this while a turn is working — the
provider receives the same request again. Useful when a request times out or returns
an incomplete response.

## MCP

### `/mcp`

Manage external [MCP](mcp.md) server connections. Sub-commands:

| Usage | Effect |
|---|---|
| `/mcp` | List configured servers and their connection status |
| `/mcp connect <server> [server ...]` | Connect one or more servers now |
| `/mcp disconnect <server>` | Disconnect a server |
| `/mcp tools [server]` | List tools and resources exposed by a connected server |

Connections auto-start when the agent references a server (via `@server` or
`Skill(name)`).

## Help and exit

### `/help`

Show the built-in command and tool reference — the same text you see here, compact.

### `/exit`, `/quit`

Leave nanocode. Your session is saved automatically and can be resumed with
`-c` or `--resume`.
