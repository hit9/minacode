# Interaction

nanocode runs as a conversation in your terminal. You type a request, the agent works
through it with [tools](tools.md), and you stay in the loop the whole time — steering,
answering questions, and reviewing changes.

## Follow-ups

Press `Enter` to queue a message for the next turn, `Ctrl-C` to interrupt and
take over immediately.

| Key | When | Effect |
|---|---|---|
| `Enter` | Typing a message | Queue for next turn |
| `Ctrl-C` | Typing a message | Interrupt agent now |
| `Ctrl-C` | Idle prompt | Clear input line |
| `Up` / `Ctrl-P` | Typing | Recall newest queued message |
## Commands

Type `/` commands at the prompt to inspect state, switch models, manage the
session, or configure runtime behavior on the fly. Run `/help` for the built-in
reference.

### Looking around

**`/status`** — Shows everything about the runtime at a glance: workspace path,
session id, active provider and model, context window fill percentage,
conversation history, prompt-cache hit ratio, code index state, background jobs,
and whether an update is available.

```{figure} ../snapshots/nanocode-status-command.png
:alt: The /status command showing workspace, session, provider, context, and code index state
:width: 600px
:align: center

The /status output at a glance.
```

**`/diff`** — Opens an interactive diff viewer with two tabs:

- **Latest** — what changed during the most recent turn
- **Session** — the net diff for everything since the session began

Navigate with `j`/`k`, `g`/`G`, and `/` search; press `Esc` to close. Outside of
interactive mode it prints the diffs as plain text.

**`/ps`** — Lists active background jobs (see [Tools](tools.md#built-in-tools)).
Each row shows job id, state, command, and elapsed time.

**`/skills`** — Lists every installed [skill](skills.md) by name, source, and
description.

**`/config`** — Shows the active configuration: provider blocks, runtime settings,
and their resolved values.

### The code index

**`/index [force]`** — Build or rebuild the code symbol index that powers
`InspectCode`. The first build walks every source file; later syncs are fast. Add
`force` to rebuild from scratch. See [Code symbol index](tools.md#code-symbol-index)
for details.

### Switching models

**`/provider [NAME]`** — Show or switch the active provider. Without an argument it
lists every configured provider (from your [configuration](configuration.md#providers))
and lets you pick one interactively. With a name it switches immediately.

**`/model [MODEL]`** — Show or switch the model for the current provider. Without
an argument it opens an interactive picker with configured and discovered models.
Changing the model also prompts you to pick a reasoning effort.

**`/reason [EFFORT]`** — Show or set reasoning effort (OpenAI o-series and DeepSeek
R1). Values: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`. Without an
argument it opens a picker.

```{figure} ../snapshots/nanocode-demo-switching-providers-models.gif
:alt: Switching providers and models interactively during a session
:width: 600px
:align: center

Switching providers and models mid-session.
```

### Managing the session

**`/compact`** — Summarize and shrink the conversation immediately. nanocode keeps
long sessions within budget on its own, but `/compact` trims on demand.

**`/yolo`** — Toggle confirmation prompts. See [Safety](safety.md) before turning
this off permanently.

**`/strict`** — Toggle strict tool-call schemas (OpenAI / DeepSeek).

**`/set KEY VALUE`** — Set `provider.*` or `runtime.*` for the session. Example:
`/set provider.model deepseek-v4-flash`.

**`/resend`** — Re-send the in-flight model request. Type this while a turn is
working.

### MCP

**`/mcp`** — Manage [MCP](mcp.md) server connections. Sub-commands:

| Usage | Effect |
|---|---|
| `/mcp` | List servers and connection status |
| `/mcp connect <server> [server ...]` | Connect servers now |
| `/mcp disconnect <server>` | Disconnect a server |
| `/mcp tools [server]` | List tools from a connected server |

### Help and exit

**`/help`** — Show the built-in command and tool reference.

**`/exit`, `/quit`** — Leave nanocode. Your session is saved automatically and can
be resumed with `-c` or `--resume`.

## Mentions

Two inline references, both Tab-completed as you type:

- `@server` or `@server.tool` — point the agent at an [MCP](mcp.md) server or tool, connecting
  it on demand for this message.
- `$skill` — inject a [skill](skills.md)'s full instructions into the current turn.

## Keys and input editing

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

```{figure} ../snapshots/nanocode-working-input-editor.png
:alt: Editing a follow-up message in an external editor
:width: 600px
:align: center

Typing a follow-up message in an external editor.
```

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

```{figure} ../snapshots/nanocode-diff-list.png
:alt: Interactive diff list showing changed files from the latest turn
:width: 600px
:align: center

Choosing a file to diff.
```

```{figure} ../snapshots/nanocode-diff-file-detail.png
:alt: Side-by-side file diff with syntax highlighting
:width: 600px
:align: center

Side-by-side detail view of a changed file.
```

### Long sessions

nanocode keeps long conversations within a working budget on its own, summarizing older
context as needed so a session can run indefinitely. Run `/compact` to trim it now, or
`/status` to see current context and token usage.
