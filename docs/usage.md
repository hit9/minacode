# Interaction

nanocode runs as a conversation in your terminal. You type a request, the agent works
through it with [tools](tools.md), and you stay in the loop the whole time — steering,
answering questions, and reviewing changes.

## Follow-ups

You can keep typing while nanocode works. A submitted follow-up joins the current task if
another model step begins; otherwise it becomes the next task. Interrupting does not submit a
draft that is still in the editor.

<div class="term-shot" role="img" aria-label="Terminal view: nanocode is working on a request while two follow-up messages wait below a divider reading 'working, 2 queued'."><span class="fs-user">• refactor the MCP manager</span><span class="fs-tool">  Read nanocode.py</span><span class="fs-tool">  Edit nanocode.py</span><span class="fs-divider">──── working (12s) [ 2 queued ] ─────────────</span><span class="fs-queued">+ also update the tests</span><span class="fs-queued">+ and bump the version</span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span class="fs-hint">  ↑ recalls queued · Ctrl-C interrupts</span></div>

Everything below the divider is waiting. The agent picks those messages up at its next step,
and they move up into the log above the divider once they are in.

| Key | When | Effect |
|---|---|---|
| `Enter` | While the agent works | Queue a follow-up for the next model step |
| `Ctrl-C` | While the agent works | Interrupt the current task; keep any draft in the editor |
| `Ctrl-C` | Idle prompt | Clear input line |
| `Up` / `Ctrl-P` | While working, with an empty editor | Recall the newest queued message |

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

**`/diff`** — Review changes from the latest turn or the whole session. See
[Reviewing changes](#reviewing-changes) below.

<div class="term-shot" role="img" aria-label="The diff viewer: a Latest and Session tab above a list of changed files, each with added and removed line counts, and a key hint along the bottom."><span><span class="fs-i fs-tab-on"> Latest </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-off"> Session </span></span><span> </span><span class="fs-sel">&gt; <span class="fs-i fs-add">+45</span> <span class="fs-i fs-del">-12</span> docs/usage.md</span><span class="fs-dim">  <span class="fs-i fs-add">+12</span> <span class="fs-i fs-del">- 3</span> nanocode.py</span><span class="fs-dim">  <span class="fs-i fs-add">+ 4</span> <span class="fs-i fs-del">- 0</span> tests/test_mcp.py</span><span> </span><span class="fs-dim">  [list] ↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close [1/3]</span></div>

The two tabs pick the range; each row is one changed file with its added and removed line
counts. `Enter` opens the selected file's diff.

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

- `@server` or `@server.tool` — connect to an [MCP](mcp.md) server on demand and point the
  agent at that server or tool. <span class="marker">The connection remains active until you
  disconnect it.</span>
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

<span class="marker">Your work is saved automatically</span> — the conversation, edits, and diffs
are tied to the project directory you started in, so an interrupted session picks up where it
stopped. Inactive sessions older than seven days are removed at startup by default; set
`runtime.session_retention_days = 0` to keep them indefinitely.

Resume from the command line:

```sh
nanocode -c            # resume the latest session in this project
nanocode --resume      # same, explicit
nanocode --resume UID  # resume a specific session by id, from any directory
```

Sessions are stored per project, so `-c` and a bare `--resume` never reach into another project's
history — even when your most recent session anywhere was somewhere else. A `UID` is looked up
across every project, so you can resume one by id from wherever you are.

Resuming replays the conversation into your scrollback, including the diff each edit made. Long
diffs are trimmed there; `/diff` always has the full text.

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
