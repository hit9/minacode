# Interaction

wizolt runs as a conversation in your terminal. You type a request, the agent works
through it with [tools](tools.md), and you stay in the loop the whole time — steering,
answering questions, and reviewing changes.

## Follow-ups

You can keep typing while wizolt works. A submitted follow-up joins the current task if
another model step begins; otherwise it becomes the next task. A draft still in the editor is
never submitted by interrupting — the first `Ctrl-C` discards it instead.

<div class="term-shot" role="img" aria-label="Terminal view: wizolt is working on a request while two follow-up messages wait below a divider reading 'working, 2 queued'."><span class="fs-user">• refactor the MCP manager</span><span class="fs-tool">  Read wizolt.py</span><span class="fs-tool">  Edit wizolt.py</span><span><span class="fs-i fs-rule">--</span><span class="fs-i fs-glow">-</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-add">●</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-working">working (12s) [ 2 queued ]</span><span class="fs-i fs-rule"> ------------------------------</span></span><span class="fs-queued">+ also update the tests</span><span class="fs-queued">+ and bump the version</span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span class="fs-hint">  ↑ recalls queued · Ctrl-C interrupts</span></div>

A `+` below the divider is waiting for the next model step. At that boundary — after the current
tool-call batch, when there is one — all waiting follow-ups are sent together, in order, with the
next model request and move above the divider as normal user messages. They remain retryable
internally until that request completes successfully; a failed request moves them back below the
divider as queued input.

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
marked as interrupted so wizolt knows it ended early.

## Streaming model output

Model output streams by default in the interactive terminal over OpenAI-compatible Chat
Completions, OpenAI Responses, and Anthropic Messages. Text appears as it arrives above the
divider, and the divider names the phase: `thinking` while the model reasons, if it exposes
reasoning at all, then `responding` while it writes the answer.

<div class="term-shot" role="img" aria-label="The same divider line at three moments of one turn: while reasoning streams above it the label reads thinking, while the answer streams it reads responding, and when the turn completes the preview is replaced by the final answer above the idle prompt."><span class="fs-dim">  I should inspect the existing implementation first.</span><span> </span><span><span class="fs-i fs-rule">--</span><span class="fs-i fs-glow">-</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-add">●</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-working">thinking (4s)</span><span class="fs-i fs-rule"> ------------------------------</span></span><span> </span><span class="fs-dim">  ⋮</span><span class="fs-dim">  I found the issue in the request path.</span><span> </span><span><span class="fs-i fs-rule">--</span><span class="fs-i fs-glow">-</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-add">●</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-working">responding (7s)</span><span class="fs-i fs-rule"> ------------------------------</span></span><span> </span><span class="fs-dim">  ⋮</span><span>The retry loop reused a closed client. Reconnecting per attempt fixes it.</span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span></div>

The live text is a bounded preview, not a second conversation entry. On completion it clears
and the final answer is rendered once in the normal Rich transcript. Tool-call arguments stay
buffered until the call is complete, so partial JSON never appears as user-facing output. No
streaming setting is required for the usual case; endpoints that reject streaming can disable it
with `provider.stream = false` or `/set provider.stream off`.

## Bash output

While Bash runs, its live output stays above the `working` divider. When the command
finishes, up to three lines from each stream stay in the transcript, and the complete result
is stored under its `tr.N` key.

Press `Ctrl-O` to browse stored results, newest first: `j`/`k` or the arrows
select, `/` searches, `Enter` opens, and `Ctrl-O` or `q` closes. Long lists scroll inside a window
about ten rows tall, and a counter under it says which rows you are looking at. Opening one shows
what was run above what it returned, in a read-only scrolling viewer — a Bash command with both
its streams, a `ToolScript` with its complete script and result, a background `Job` with its log,
or a delegation order with the worker's answer below it. For a script this is the only way to read
it under `--yolo`, where no confirmation prompt stops to offer `v`; for an order it is where you
judge the answer against what was actually asked, since the transcript keeps only the `Delegate
send` line. A Job log is available while that job remains in the current session; after a resume
or after the log is removed, the viewer shows the stored Job result instead.

A `ToolScript` that is **running right now** leads the list, marked `running` instead of a `tr.N`
key. A long batch is exactly when its script is worth reading, and until it returns there is no
stored result to open. It leaves the list when the batch finishes and its real entry takes over.

Very large results are bounded rather than rendered whole: the viewer keeps the head and tail of
a long result and clips individual lines past ~1000 characters, and the header says so whenever
it did. Stored tool results remain complete under their `tr.N` keys; live Job logs are read through
a separate fixed-size snapshot.

<div class="term-shot" role="img" aria-label="A completed Bash command with bounded output, followed by the Ctrl-O list of recent results — a running ToolScript at the top marked running in magenta, then stored entries whose dim tr.N key, green tool name and plain arguments are coloured the way the transcript colours the same call, with the selected row highlighted whole — and the read-only viewer one of them opens, showing the command above its output."><span class="fs-tool">  Bash  pytest -q</span><span class="fs-dim">    ├ output · 14.7s Ctrl-O for more</span><span class="fs-dim">    │ stdout:</span><span class="fs-output">    │   708 passed in 14.84s</span><span class="fs-dim">    └ stored tr.18</span><span> </span><span class="fs-divider">──── Tool output · latest 4 ────────────────</span><span><span class="fs-i fs-dim">   1. </span><span class="fs-i fs-working">running  </span><span class="fs-i fs-tool">ToolScript </span><span class="fs-i">call 24 lines (938 chars)</span></span><span class="fs-sel">&gt;  2. tr.18  Bash pytest -q</span><span><span class="fs-i fs-dim">   3. tr.17  </span><span class="fs-i fs-tool">Bash </span><span class="fs-i">git diff --check</span></span><span><span class="fs-i fs-dim">   4. tr.16  </span><span class="fs-i fs-tool">Bash </span><span class="fs-i">git status --short</span></span><span> </span><span class="fs-divider">  Output · tr.18 · read-only</span><span class="fs-dim">  key   tr.18</span><span class="fs-dim">  exit  0</span><span> </span><span class="fs-divider">  ──────────────────────────────────────────</span><span> </span><span class="fs-output">  1  pytest -q</span><span> </span><span class="fs-divider">  ── result ─────────────────────────────────</span><span> </span><span class="fs-dim">  stdout:</span><span class="fs-dim">    708 passed in 14.84s</span><span class="fs-dim"> </span><span class="fs-dim">  ↑/↓ scroll · g/G top/bottom · Esc/q close</span></div>

## Status bar

A single line beneath the prompt summarizes the runtime: the active provider and model, the
reasoning level, context fill with the latest request's cache ratio as `ctx 23% · cache 98%`,
the code index state, and any background jobs. MCP and skill counts and an update notice appear
when relevant.

The cache ratio fills in after the first request and refreshes with each response. `/status`
reports the same figures for the whole session.

While the agent works, the role colors give way to a blue-to-purple sweep that scrolls
across the line, and the live counters join it: a retry or attempt notice, and the
`step N/M` counter once the turn reaches the final fifth of `max_agent_steps`, signaling
that the turn is about to be cut off.

A `[compaction]` marker leads the line while a summary runs, so the pause reads as compaction
rather than as a slow reply.

The working divider above the prompt names the current phase — `thinking`, `responding`, or
`web search` while a [provider-side tool](tools.md#provider-side-tools) runs inside the request —
with the time spent so far beside it, and an estimated output speed while text is arriving:
`responding (12s · ↓ 48 tok/s)`. The `↓` marks the speed as the model's incoming stream; it is
still an estimate, and it disappears between requests and on providers that do not stream.

<div class="term-shot" role="img" aria-label="The status bar in two states. At rest: provider and model, reasoning, context fill with the cache ratio, and index, each in its role color. While working: the same line rendered as a blue-to-purple sweep with a bright band, plus a step counter near the cap."><span><span class="fs-i fs-dim">idle    </span><span class="fs-i sb-base">dashscope/qwen3.7-plus</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-reason">high</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-ctx">ctx 23% · cache 98%</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-index">index ✓</span></span><span><span class="fs-i fs-dim">working </span><span class="fs-i sb-sweep-a">dashscope/qwen3.7-plus | high | </span><span class="fs-i sb-sweep-hi">ctx 41% · cache 95%</span><span class="fs-i sb-sweep-b"> | index ✓ | step 320/400</span></span></div>

## Quick hints

After an answer, the model may suggest two or three next steps as chips under the prompt. They are
suggestions, not commands: ignore them and keep typing, or take one. Chips flow left to right,
up to three per line, and wrap to new lines when the terminal is too narrow, so every
suggestion stays visible.

<div class="term-shot" role="img" aria-label="The idle prompt after an answer: the answer text, an empty prompt with a caret, and one row of three suggestion chips separated by grey bars, the middle one highlighted in reverse."><span>Everything is ready to review.</span><span> </span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span> </span><span><span class="fs-i fs-sel"> run the tests </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-on"> show the diff </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-sel"> commit the work </span></span></div>

`Tab` cycles between the input and the chips. `Enter` on a chip picks it into the input and
returns to the prompt, so `Tab` to the next chip and `Enter` again combines several suggestions;
a final `Enter` sends. Focus a picked chip and press `Enter` to unpick it. Editing the text
normally clears the chip selection state. Quick hints are always available at the TUI prompt.

## Commands

Type `/` commands at the prompt to inspect state, switch models, manage the
session, or configure runtime behavior on the fly. See the
[command reference](commands.md) for the full list, or run `/help` in a session.

## Mentions

Mentions pull something into the turn. Type `@` at the prompt for a list of the three kinds;
picking one opens its candidates, which narrow as you keep typing. `Tab` highlights a row; `Enter` commits it into the input without sending, and a second `Enter` sends.

<div class="term-shot" role="img" aria-label="The mention menu in two moments. After typing an at sign the prompt lists the three kinds with a one-line description each. After choosing at-file the file picker opens between two rules, with its own query line, a match counter, a key hint, and two ranked file paths, the first one pointed at."><span class="fs-prompt">&gt; add tests for @<span class="fs-caret">▏</span></span><span><span class="fs-i">                </span><span class="fs-i"> @file:   </span><span class="fs-i fs-dim"> files in this repo </span></span><span><span class="fs-i">                </span><span class="fs-i"> @mcp:    </span><span class="fs-i fs-dim"> MCP servers and tools </span></span><span><span class="fs-i">                </span><span class="fs-i"> @skill:  </span><span class="fs-i fs-dim"> installed skills </span></span><span> </span><span class="fs-prompt">&gt; add tests for @file:<span class="fs-caret">▏</span></span><span class="fs-divider">  ──── file picker ─────────────────────────────────</span><span><span class="fs-i fs-sel">  files&gt; </span><span class="fs-i">mention</span><span class="fs-i fs-caret">▏</span></span><span class="fs-dim">      3/812</span><span class="fs-dim">      Ctrl-N/P or ↑/↓ move · Enter select · Esc close</span><span class="fs-sel">  &gt;   tests/test_mentions.py</span><span>      wizolt/mentions.py</span><span class="fs-divider">  ──────────────────────────────────────────────────</span></div>

| Mention | Also written | Effect |
|---|---|---|
| `@file:path` | — | Points the agent at a file in the project |
| `@mcp:server`, `@mcp:server.tool` | `@server`, `@server.tool` | Connects an [MCP](mcp.md) server on demand and points the agent at that server or tool. <span class="marker">The connection remains active until you disconnect it.</span> |
| `@skill:name` | `$name` | Injects a [skill](skills.md)'s full instructions into the current turn |

**Files.** Small in-workspace files are inlined as a FILE MENTIONS block; large ones and anything
outside the workspace become a pointer telling the agent to Read it. A missing path reports itself
instead of being silently dropped.

**Picking files.** Files stay out of the first `@` list: typing or selecting `@file:` opens fzf
immediately, and `Tab` does the same from an active file mention. Without fzf, a bounded literal
fallback runs in the background. Candidates include hidden, tracked, and untracked files, but never
`.git` directories or paths ignored by Git; in a non-Git directory, nested `.gitignore` files and
negation rules are honored. A path with spaces, Unicode, or ambiguous punctuation is inserted in a
quoted, round-trippable form such as `@file:"docs/design notes/中文.txt"`.

Mentions are expanded in follow-ups queued while the agent is working, too.

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
- `Ctrl-R` — reverse-search your history; `Enter` puts the match in the input to edit, a second
  `Enter` sends it
- `Ctrl-O` — browse recent Bash outputs, ToolScript scripts, background Job logs, and delegation
  orders; press it again to close
- `Ctrl-X Ctrl-E` or `Ctrl-G` — edit the current input in `$VISUAL` / `$EDITOR` (falls back to
  vim), as a temporary Markdown file

```{figure} ../snapshots/wizolt-working-input-editor.png
:alt: Editing a follow-up message in an external editor
:width: 600px
:align: center

Typing a follow-up message in an external editor.
```

When you open the editor in reply to the agent, its most recent reply is appended below a git-style scissors line, so you can read what you are answering while you compose (the full-screen editor hides that scrollback):

<div class="term-shot" role="img" aria-label="The message open in vim: numbered lines holding the draft with the cursor at its end, a blank line, the scissors line with its unique marker, two comment lines explaining that everything below is stripped — the first wrapping at the window edge — then the agent's most recent reply, filler tildes, a status line naming the modified temporary Markdown file with the cursor position, and the INSERT mode message."><span><span class="fs-i fs-vim-fill">  1 </span><span class="fs-i">yes, add the reconnect test and cap the backoff at 30s</span><span class="fs-i fs-caret">▏</span></span><span><span class="fs-i fs-vim-fill">  2 </span><span class="fs-i">&nbsp;</span></span><span><span class="fs-i fs-vim-fill">  3 </span><span class="fs-i fs-dim"># ------------------------ &gt;8 ------------------------ (4f2a9c1b77d0)</span></span><span><span class="fs-i fs-vim-fill">  4 </span><span class="fs-i fs-dim"># Reference only: everything below the scissors line is stripped before </span></span><span><span class="fs-i fs-vim-fill">    </span><span class="fs-i fs-dim">your</span></span><span><span class="fs-i fs-vim-fill">  5 </span><span class="fs-i fs-dim"># message is sent. The agent&#x27;s most recent reply follows for reference.</span></span><span><span class="fs-i fs-vim-fill">  6 </span><span class="fs-i">&nbsp;</span></span><span><span class="fs-i fs-vim-fill">  7 </span><span class="fs-i">I split McpManager into StdioTransport and HttpTransport.</span></span><span><span class="fs-i fs-vim-fill">  8 </span><span class="fs-i">Want me to add a test for the reconnect path?</span></span><span><span class="fs-i fs-vim-fill">    </span><span class="fs-i fs-vim-fill">~</span></span><span><span class="fs-i fs-vim-fill">    </span><span class="fs-i fs-vim-fill">~</span></span><span class="fs-vim-status">wizolt-input-a1b2c3d4.md [+]                  markdown    1,55     All</span><span class="fs-dim">-- INSERT --</span></div>

Everything from the scissors line down is stripped before the message is sent; a scissors line you type yourself is left untouched. Long replies are capped to their most recent lines.

### Image input

Paste or type the path of an existing local image directly into the prompt. wizolt replaces
the path with an inline label such as `[Image #1 · screenshot.png]`, so you can see exactly which
images will be submitted while continuing to edit the surrounding text. Relative paths resolve
from the workspace; quoted paths and backslash-escaped spaces are accepted.

<div class="term-shot" role="img" aria-label="The input prompt after recognizing a local screenshot path as an editable inline image label."><span class="fs-prompt">&gt; explain <span class="fs-i fs-sel">[Image #1 · screenshot.png]</span> and fix the layout<span class="fs-caret">▏</span></span></div>

PNG, JPEG, WebP, and single-frame GIF files are supported. wizolt sends each new attachment to
the active model using the selected standard API. If the provider rejects that turn, the image
remains stored at a session-owned path and later requests replay a readable label instead of the
same image blocks; the agent can inspect the stored path explicitly with `ViewImage`. A configured
[vision model](configuration.md#vision-model) is used only by `ViewImage`, never automatically.

## Sessions

<span class="marker">Your work is saved automatically</span> — the conversation, edits, and diffs
are tied to the project directory you started in, so an interrupted session picks up where it
stopped. Sessions untouched for seven days are removed by default, swept in the background when
wizolt starts; it reports how many it removed. Resuming a session resets its clock, so one you
keep returning to is never removed. Set `runtime.session_retention_days = 0` to keep them
indefinitely.

Resume from the command line:

```sh
wizolt -c              # resume the latest session in this project
wizolt --resume        # same, explicit
wizolt --resume UID    # resume a specific session by id, from any directory
wizolt --resume "fd leak"   # or by name, or by the first few characters of an id
```

Sessions are stored per project, so `-c` and a bare `--resume` never reach into another project's
history — even when your most recent session anywhere was somewhere else. A `UID` is looked up
across every project, so you can resume one by id from wherever you are.

A name or id prefix is searched in the current project first, then everywhere — you can resume a
session by name after moving directories. When a query matches more than one session, wizolt
<span class="marker">lists the candidates instead of guessing</span> between them.

Resuming replays the conversation into your scrollback, including the diff each edit made. Long
diffs are trimmed there; `/diff` always has the full text.

### Names

Every session has a name, so you have something to recognize it by later. It starts as the first
line you typed, becomes the agent's current goal once it has one, and stays whatever you set with
`/name`:

```text
/name                 # show the current name and where it came from
/name auth refactor   # set your own; nothing overwrites it afterwards
```

A name is a label, not an identity — sessions may share one, and the id is what makes each unique.
Names are decided once rather than re-read from the conversation, so a session you found under one
name yesterday is still under it today, even after its early messages have been compacted away.

### Switching sessions

`/sessions` — or `/resume`, the same command — lists what is saved, newest first, with how long ago
each was touched and how many rounds it ran. Type to filter across names and opening lines, and
press Enter to re-enter one:

<div class="term-shot" role="img" aria-label="The session picker: a searchable list of saved sessions, each showing its name, age, and round count, with the current session marked, above a preview of the highlighted session's id, opening message, and directory."><span class="fs-divider">──── Sessions ─────────────────────────────</span><span class="fs-sel">&gt; port the tool runner to asyncio<span class="fs-i fs-dim">  ·  2h ago · 14 rounds</span></span><span class="fs-dim">  split the large test modules<span class="fs-i fs-dim">  ·  yesterday · 31 rounds</span></span><span class="fs-dim">  fix the fd leak in MCPFileTokenStore<span class="fs-i fs-dim">  ·  3d ago · 1 round</span></span><span class="fs-dim">  what I am doing right now<span class="fs-i fs-dim">  ·  just now · 2 rounds · current</span></span><span> </span><span class="fs-dim">  uid   20260728074943-e22e69e8-070</span><span class="fs-dim">  start port the tool runner to asyncio, starting with Bash</span><span class="fs-dim">  where ~/dev/github/wizolt</span><span> </span><span class="fs-hint">  ↑/↓ or j/k move · / search · Enter open · Esc close</span></div>

`/sessions all` widens the list past the current project, adding each session's directory to its
row. Choosing a session ends the current one — it is saved first — and starts the next in its
place, exactly as if you had launched with `--resume`. Choosing the session you are already in, or
pressing `Esc`, changes nothing.

Run it from the prompt between turns. While the agent is working, wizolt says so and asks you to
press `Ctrl-C` first: switching sessions mid-turn would abandon a request already in flight.

Sessions saved before names existed list under their id until the next time they are saved.

### Reviewing changes

`/diff` opens an interactive, tabbed viewer with two views:

- **Latest** — what changed during the most recent round of your requests
- **Session** — the net diff for everything since the session began

Navigate with `j`/`k`, `g`/`G`, and `/` search; press `Esc` to close.

```{figure} ../snapshots/wizolt-diff-list.png
:alt: Interactive diff list showing changed files from the latest turn
:width: 600px
:align: center

Choosing a file to diff.
```

```{figure} ../snapshots/wizolt-diff-file-detail.png
:alt: Side-by-side file diff with syntax highlighting
:width: 600px
:align: center

Side-by-side detail view of a changed file.
```

### Long sessions

wizolt keeps long conversations within a working budget on its own, summarizing older
context as needed so a session can run indefinitely. Run `/compact` to trim it now, or
`/status` to see current context and token usage.
