# The interactive session

nanocode runs as a conversation in your terminal. You type a request, the agent works
through it with [tools](tools.md), and you stay in the loop the whole time — steering,
answering questions, and reviewing changes.

## Interaction

### Follow-ups: typing while it works

You don't have to wait for the agent to finish. Press `Enter` while it works to queue a
follow-up for the next step, or `Ctrl-C` to interrupt the current turn so queued input takes
over immediately. Press `Up` (or `Ctrl-P`) to bring back the newest queued message and edit
it.

### Commands

Type `/` commands at the prompt. Run `/help` for the built-in reference.

| Command | What it does |
|---|---|
| `/help` | Show the command and tool reference |
| `/status` | Runtime status: model, context and cache use, MCP, index, jobs, updates |
| `/config` | Show the active configuration |
| `/diff` | Review the latest edits and the whole-session diff |
| `/ps` | List active background jobs |
| `/skills` | List installed skills |
| `/compact` | Summarize and shrink the conversation now |
| `/index [force]` | Sync or rebuild the code symbol index |
| `/provider [NAME]` | Show or switch the active provider |
| `/model [MODEL]` | Show or switch the model |
| `/reason [EFFORT]` | Show or set reasoning effort (`minimal`…`xhigh`, or `off`) |
| `/set KEY VALUE` | Set a supported provider or runtime tuning value for this session |
| `/yolo` | Toggle confirmation prompts on or off |
| `/strict` | Toggle strict tool-call schemas (OpenAI / DeepSeek) |
| `/mcp` | Manage [MCP](mcp.md) server connections |
| `/resend` | Re-send the in-flight model request (while a turn is working) |
| `/exit`, `/quit` | Leave nanocode |

### Key commands in detail

`/status`

Shows everything about the runtime at a glance: workspace path, session id, model
and provider in use, context window fill percentage, conversation history length,
prompt-cache hit ratio, code index state, background jobs, and whether an update is
available. Run it when you want to know "where am I right now."

`/diff`

Opens an interactive diff viewer with two tabs — **Latest** (what changed in the
most recent turn) and **Session** (the net diff since the session started). Navigate
with `j`/`k`, `g`/`G`, and `/` search; press `Esc` to close. See
[Reviewing changes](#reviewing-changes) for more.

`/ps`

Lists active background jobs — shell commands that are still running after the
initial `Bash` call timed out. Each job shows its id, state (`running` / `done`),
command, and elapsed time. Use `Job` tool actions to inspect or manage them.

`/index [force]`

Build or rebuild the code symbol index that powers `InspectCode`. The first build
scans every source file; subsequent syncs are fast. Add `force` to rebuild from
scratch. See [Code symbol index](code-index.md) for the full picture.

`/provider [NAME]`

Show or switch the active provider. With no argument, it lists every configured
provider and marks the active one. With a name, it switches immediately — all
subsequent model calls use that provider. Providers are defined in
[Configuration](configuration.md#providers).

`/model [MODEL]`

Show or switch the model for the current provider. Without an argument it lists
available models (from your config) and lets you pick one with arrow keys. With a
model name, it switches to that model directly.

`/reason [EFFORT]`

Show or set the reasoning effort for providers that support it (OpenAI o-series and
DeepSeek R1). Values are `off`, `minimal`, `low`, `medium`, `high`, or `xhigh`.
Without an argument it opens a picker to choose interactively.

### Mentions

Two inline references, both Tab-completed as you type:

- `@server` or `@server.tool` — point the agent at an [MCP](mcp.md) server or tool, connecting
  it on demand for this message.
- `$skill` — inject a [skill](skills.md)'s full instructions into the current turn.

### Keys and input editing

**Interactive selectors** (model picker, MCP manager, diff viewer) support:

- `j` / `k` or arrow keys to move
- `g` / `G` to jump to top / bottom
- `/` to search, `Enter` to accept, `Esc` to cancel

**The input line** supports:

- history recall and completion
- `Ctrl-C` — clear idle input; while running, interrupt the current turn
- `Ctrl-D` — exit from an empty prompt
- `Ctrl-R` — reverse-search your history
- `Ctrl-X Ctrl-E` or `Ctrl-G` — edit the current input in `$VISUAL` / `$EDITOR` (falls back to vim)

## Sessions

Your work is saved automatically — the conversation, edits, and diffs are tied to the
project directory you started in, so an interrupted session picks up where it stopped.
Inactive sessions older than seven days are removed at startup by default; set
`runtime.session_retention_days = 0` to keep them indefinitely.

Resume from the command line:

```sh
nanocode -c            # resume the latest session in this project
nanocode --resume      # same, explicit
nanocode --resume UID  # resume a specific session by id
```

### Reviewing changes

`/diff` opens an interactive, tabbed viewer with two views:

- **Latest** — what changed during the most recent round of your requests
- **Session** — the net diff for everything since the session began

Navigate with `j`/`k`, `g`/`G`, and `/` search; press `Esc` to close.

### Long sessions

nanocode keeps long conversations within a working budget on its own, summarizing older
context as needed so a session can run indefinitely. Run `/compact` to trim it now, or
`/status` to see current context and token usage.
