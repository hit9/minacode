# Interaction

minacode runs as a conversation in your terminal. You type a request, the agent works
through it with [tools](tools.md), and you stay in the loop the whole time — steering,
answering questions, and reviewing changes.

## Follow-ups

You can keep typing while minacode works. A submitted follow-up joins the current task if
another model step begins; otherwise it becomes the next task. A draft still in the editor is
never submitted by interrupting — the first `Ctrl-C` discards it instead.

<div class="term-shot" role="img" aria-label="Terminal view: minacode is working on a request while two follow-up messages wait below a divider reading 'working, 2 queued'."><span class="fs-user">• refactor the MCP manager</span><span class="fs-tool">  Read minacode.py</span><span class="fs-tool">  Edit minacode.py</span><span class="fs-divider">──── working (12s) [ 2 queued ] ─────────────</span><span class="fs-queued">+ also update the tests</span><span class="fs-queued">+ and bump the version</span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span class="fs-hint">  ↑ recalls queued · Ctrl-C interrupts</span></div>

Everything below the divider is waiting. The agent picks those messages up at its next step,
and they move up into the log above the divider once they are in.

| Key | When | Effect |
|---|---|---|
| `Enter` | While the agent works | Queue a follow-up for the next model step |
| `Ctrl-C` | While the agent works | Discard a draft in the editor; with the editor empty, interrupt the task — retracting the message if the agent has not answered yet, or recording the interrupt once it has |
| `Ctrl-C` | Idle prompt | Clear input line |
| `Ctrl-U` | Any prompt | Clear the whole input line, leaving the turn running |
| `Up` / `Ctrl-P` | While working, with an empty editor | Recall the newest queued message |

Interrupting splits two ways. If the agent has not answered yet, `Ctrl-C` *retracts* the
message: it is discarded and never reaches the conversation record or the saved session, as
if it was never sent (your input history still recalls it with `Ctrl-P`). Once the agent has
spoken or run a tool, `Ctrl-C` *interrupts*: the work already shown stays, and the turn is
marked as interrupted so minacode knows it ended early.

## Bash output

While Bash runs, its live output stays above the `working` divider. When the command
is running, a blank row keeps its last live-output line clear of that divider. When the
command finishes, minacode keeps up to three lines from each output stream in the transcript.
The gray `output · Ctrl-O for more` row opens a larger, bounded preview: press `Ctrl-O`
to browse the ten most recent completed Bash previews, newest first. Use `j`/`k` or the arrows
to select one and `Enter` to open it; `Esc` returns to the list, while `Ctrl-O` or `q` closes
the viewer. The complete result remains stored under its `tr.N` key. Each viewer screen leaves
a blank row above its subdued labeled rule, separating it from terminal scrollback.

<div class="term-shot" role="img" aria-label="A completed Bash command with bounded output, followed by the Ctrl-O list of recent Bash commands and one larger output preview, each separated from scrollback by a labeled rule."><span class="fs-tool">  Bash  pytest -q</span><span class="fs-dim">    ├ output · 14.7s Ctrl-O for more</span><span class="fs-dim">    │ stdout:</span><span class="fs-output">    │   708 passed in 14.84s</span><span class="fs-dim">    └ stored tr.18</span><span> </span><span class="fs-divider">──── Bash outputs · latest 3 ───────────────</span><span class="fs-sel">&gt;  1. tr.18  Bash pytest -q</span><span class="fs-dim">   2. tr.17  Bash git diff --check</span><span class="fs-dim">   3. tr.16  Bash git status --short</span><span> </span><span class="fs-divider">──── Bash output · tr.18 ──────────────────</span><span class="fs-dim">  Bash pytest -q</span><span class="fs-dim"> </span><span class="fs-dim">  stdout:</span><span class="fs-dim">    708 passed in 14.84s</span><span class="fs-dim"> </span><span class="fs-dim">  Esc / ← back · Ctrl-O / q closes</span></div>

## Commands

Type `/` commands at the prompt to inspect state, switch models, manage the
session, or configure runtime behavior on the fly. Run `/help` for the built-in
reference.

### Looking around

**`/status`** — Shows everything about the runtime at a glance: workspace path,
session id, active provider and model, calculated compaction-budget fill percentage,
conversation history, prompt-cache hit ratio, code index state, background jobs,
and whether an update is available.

```{figure} ../snapshots/minacode-status-command.png
:alt: The /status command showing workspace, session, provider, context, and code index state
:width: 600px
:align: center

The /status output at a glance.
```

**`/diff`** — Review changes from the latest turn or the whole session. See
[Reviewing changes](#reviewing-changes) below.

<div class="term-shot" role="img" aria-label="The diff viewer: a Latest and Session tab above a list of changed files, each with added and removed line counts, and a key hint along the bottom."><span><span class="fs-i fs-tab-on"> Latest </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-off"> Session </span></span><span> </span><span class="fs-sel">&gt; <span class="fs-i fs-add">+45</span> <span class="fs-i fs-del">-12</span> docs/usage.md</span><span class="fs-dim">  <span class="fs-i fs-add">+12</span> <span class="fs-i fs-del">- 3</span> minacode.py</span><span class="fs-dim">  <span class="fs-i fs-add">+ 4</span> <span class="fs-i fs-del">- 0</span> tests/test_mcp.py</span><span> </span><span class="fs-dim">  [list] ↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close [1/3]</span></div>

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

```{figure} ../snapshots/minacode-demo-switching-providers-models.gif
:alt: Switching providers and models interactively during a session
:width: 600px
:align: center

Switching providers and models mid-session.
```

### Managing the session

**`/compact`** — Summarize and shrink the conversation immediately. minacode keeps
long sessions within budget on its own, but `/compact` trims on demand.

**`/yolo`** — Toggle confirmation prompts. See [Safety](safety.md) before turning
this off permanently.

**`/strict`** — Toggle strict tool-call schemas (OpenAI / DeepSeek).

**`/set KEY VALUE`** — Set `provider.*` or `runtime.*` for the session. Example:
`/set provider.model deepseek-v4-flash`.

**`/resend`** — Cancel and re-send the in-flight model request without restarting the
turn. Type it while a model request is waiting; it has no effect while the agent is
running a tool or otherwise between model calls. The divider briefly reports the retry,
keeps its elapsed timer and waiting pulse, then returns to `working` for the replacement
request. Automatic retries also show their attempt and concise reason, such as
`retrying 2/6 · timeout`, followed by `working · attempt 2/6` while that request continues:

<div class="term-shot" role="img" aria-label="The running divider briefly changes from working to retrying while preserving its green waiting pulse and elapsed timer, then returns to working as the replacement model request continues."><span class="fs-divider">──── <span class="fs-i fs-add">●</span> working (11s) ────────────────────</span><span class="fs-prompt">+&gt; /resend</span><span class="fs-divider">──── <span class="fs-i fs-add">●</span> retrying (12s) ──────────────────</span><span class="fs-divider">──── <span class="fs-i fs-add">●</span> working (14s) ────────────────────</span></div>

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

**`/exit`, `/quit`** — Leave minacode. Your session is saved automatically and can
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
- `Ctrl-C` — clear the current input; with the input empty while running, interrupt the turn (retracting it if the agent has not answered yet)
- `Ctrl-U` — clear the whole input line, in the idle prompt and the follow-up editor alike
- `Ctrl-D` — exit from an empty prompt
- `Ctrl-R` — reverse-search your history
- `Ctrl-O` — browse the ten most recent completed Bash output previews; press it again to close
- `Ctrl-X Ctrl-E` or `Ctrl-G` — edit the current input in `$VISUAL` / `$EDITOR` (falls back to vim)

```{figure} ../snapshots/minacode-working-input-editor.png
:alt: Editing a follow-up message in an external editor
:width: 600px
:align: center

Typing a follow-up message in an external editor.
```

When you open the editor in reply to the agent, its most recent reply is appended below a git-style scissors line, so you can read what you are answering while you compose (the full-screen editor hides that scrollback):

<div class="term-shot" role="img" aria-label="External editor view: the draft being composed on top, a git-style scissors line, then the agent's most recent reply below it for reference; everything below the scissors line is stripped before the message is sent."><span class="fs-user">yes, add the reconnect test and cap the backoff at 30s</span><span class="fs-dim">&nbsp;</span><span class="fs-divider"># ------------------------ &gt;8 ------------------------</span><span class="fs-dim"># Reference only: everything below the scissors line is stripped before your</span><span class="fs-dim"># message is sent. The agent's most recent reply follows for reference.</span><span class="fs-dim">&nbsp;</span><span class="fs-prompt">I split McpManager into StdioTransport and HttpTransport, each closing its own</span><span class="fs-prompt">client in close(). Want me to add a test for the reconnect path?</span></div>

Everything from the scissors line down is stripped before the message is sent; a scissors line you type yourself is left untouched. Long replies are capped to their most recent lines.

## Sessions

<span class="marker">Your work is saved automatically</span> — the conversation, edits, and diffs
are tied to the project directory you started in, so an interrupted session picks up where it
stopped. Inactive sessions older than seven days are removed at startup by default; set
`runtime.session_retention_days = 0` to keep them indefinitely.

Resume from the command line:

```sh
minacode -c            # resume the latest session in this project
minacode --resume      # same, explicit
minacode --resume UID  # resume a specific session by id, from any directory
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

```{figure} ../snapshots/minacode-diff-list.png
:alt: Interactive diff list showing changed files from the latest turn
:width: 600px
:align: center

Choosing a file to diff.
```

```{figure} ../snapshots/minacode-diff-file-detail.png
:alt: Side-by-side file diff with syntax highlighting
:width: 600px
:align: center

Side-by-side detail view of a changed file.
```

### Long sessions

minacode keeps long conversations within a working budget on its own, summarizing older
context as needed so a session can run indefinitely. Run `/compact` to trim it now, or
`/status` to see current context and token usage.
