# Commands

Type `/` commands at the prompt to inspect state, switch models, manage the
session, or configure runtime behavior on the fly. Run `/help` for the built-in
reference.

## Looking around

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
[Reviewing changes](usage.md#reviewing-changes).

<div class="term-shot" role="img" aria-label="The diff viewer: a Latest and Session tab above a list of changed files, each with added and removed line counts, and a key hint along the bottom."><span><span class="fs-i fs-tab-on"> Latest </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-off"> Session </span></span><span> </span><span class="fs-sel">&gt; <span class="fs-i fs-add">+45</span> <span class="fs-i fs-del">-12</span> docs/usage.md</span><span class="fs-dim">  <span class="fs-i fs-add">+12</span> <span class="fs-i fs-del">- 3</span> minacode.py</span><span class="fs-dim">  <span class="fs-i fs-add">+ 4</span> <span class="fs-i fs-del">- 0</span> tests/test_mcp.py</span><span> </span><span class="fs-dim">  [list] ↑/↓ or j/k move · ←/→ or h/l tab · Enter open · r refresh · Esc/q close [1/3]</span></div>

The two tabs pick the range; each row is one changed file with its added and removed line
counts. `Enter` opens the selected file's diff.

**`/ps`** — Lists active background jobs (see [Tools](tools.md#built-in-tools)).
Each row shows job id, state, command, and elapsed time.

**`/skills`** — Lists every installed [skill](skills.md) by name, source, and
description.

**`/config`** — Shows the active configuration: provider blocks, runtime settings,
and their resolved values.

## The code index

**`/index [force]`** — Build or rebuild the code symbol index that powers
`InspectCode`. The first build walks every source file; later syncs are fast. Add
`force` to rebuild from scratch. See [Code symbol index](tools.md#code-symbol-index)
for details.

## Switching models

**`/provider [NAME]`** — Show or switch the active provider. Without an argument it
lists every configured provider (see [Configuration](configuration.md#providers))
and lets you pick one interactively. With a name it switches immediately.

**`/model [MODEL]`** — Show or switch the model for the current provider. Without
an argument it opens an interactive picker with configured and discovered models.
Changing the model also prompts you to pick a reasoning effort.

**`/reason [EFFORT]`** — Show or set reasoning effort. Values: `off`, `minimal`,
`low`, `medium`, `high`, `xhigh`, `max`. Minacode maps known model families to
their nearest supported level; unrecognized providers and models keep the selected
value. Without an argument it opens a picker.

```{figure} ../snapshots/minacode-demo-switching-providers-models.gif
:alt: Switching providers and models interactively during a session
:width: 600px
:align: center

Switching providers and models mid-session.
```

## Managing the session

**`/name [TEXT]`** — Show or set this session's name. See [Names](usage.md#names).

**`/sessions [all]`** — Browse saved sessions and re-enter one; `/resume` is the same command.
See [Switching sessions](usage.md#switching-sessions).

**`/compact`** — Summarize and shrink the conversation immediately. minacode keeps
long sessions within budget on its own, but `/compact` trims on demand.

**`/compact log [seg.N]`** — Review what compaction kept: the stored segments newest
first, and the summary of the one you open. Naming a segment prints its summary
without the viewer. See
[Keeping context manageable](context.md#keeping-context-manageable).

**`/yolo`** — Toggle confirmation prompts. See [Safety](safety.md) before turning
this off permanently.

**`/hints`** — Toggle the model's suggested next-step chips at the idle prompt (the
`NextHints` tool). See [Tools](tools.md).

**`/strict`** — Toggle strict tool-call schemas (OpenAI / DeepSeek).

**`/worker [status|reset|on|off|provider|model|reason|api]`** — Inspect or control the worker session:
`status` (default) reports readable lines — provider/model, reasoning, state, rounds, context
percent — or `worker: no active session` plus the configured `[worker] provider` when no worker
exists; `on`/`off` toggle the `runtime.worker` setting, and `reset` clears the worker's context
(file changes and merged diffs survive). `provider` opens a picker over the provider entries —
with `off` at the end to clear — and choosing one flows on into the worker model and then the
reasoning pickers, mirroring `/provider`'s chain (backing out of any stage keeps the earlier
stages already set); `provider NAME` re-targets the worker immediately and for future spawns
without the cascade, while `provider off` clears the entry. `model`, `reason`, and `api` pick
from the worker's entry models, reasoning efforts, and wire protocols (`auto`, `chat`,
`responses`, `anthropic`), with `default` clearing an override back to inheriting the
provider entry's value; tab completion offers the subcommands and their values. The `Delegate`
tool block is fixed when the session starts — see [Worker
delegation](worker.md#worker-delegation).

A `Delegate send` is confirmed even under `yolo` — the approval brief is the one cheap check on a
spec the model wrote for itself; see [Worker delegation](worker.md#worker-delegation). Every
send asks, and a refused send feeds your reason back to the model.

**`/language [NAME]`** — Show or force the session's reply language. Without an argument it
prints the current value (`auto` by default). With a name like `Chinese` it appends one fixed
`LANGUAGE OVERRIDE` block to the tail of the system prompt, so the cacheable prompt prefix
stays unchanged; `auto` follows your messages and injects nothing. The setting is per-session —
it is not written back to the configuration file — and workers inherit it from the parent on
every `Delegate` send.

**`/api [API]`** — Select or set the request protocol (`auto`, `chat`, `responses`, `anthropic`)
used to reach the model. `/provider` and `/model` also confirm it as a step in their selection
chain, since the right protocol depends on the model you just picked.

An endpoint that serves several model families often exposes them on different protocols, and an
OpenAI-compatible `/models` listing says nothing about which protocol serves which model — so a
model offered by `/model` can still be rejected as unsupported. When that happens, pick a different
protocol with `/api` (or `auto` to re-infer from the URL and model). The reply names the wire that
took effect, and history is protocol-neutral, so switching mid-session is safe.

**`/set KEY VALUE`** — Set `provider.*` or `runtime.*` for the session; tab-completes both keys
and, where the values are a fixed set, the values. Example: `/set provider.response_timeout 900`.

**`/resend`** — Cancel and re-send the in-flight model request without restarting the
turn. Type it while a model request is waiting; it has no effect while the agent is
running a tool or otherwise between model calls. The divider briefly reports the retry,
keeps its elapsed timer and waiting pulse, then returns to `working` for the replacement
request. Automatic retries also show their attempt and concise reason, such as
`retrying 2/6 · timeout`, followed by `working · attempt 2/6` while that request continues:

<div class="term-shot" role="img" aria-label="The running divider briefly changes from working to retrying while preserving its green waiting pulse and elapsed timer, then returns to working as the replacement model request continues."><span class="fs-divider">──── <span class="fs-i fs-add">●</span> working (11s) ────────────────────</span><span class="fs-prompt">+&gt; /resend</span><span class="fs-divider">──── <span class="fs-i fs-add">●</span> retrying (12s) ──────────────────</span><span class="fs-divider">──── <span class="fs-i fs-add">●</span> working (14s) ────────────────────</span></div>

## MCP

**`/mcp`** — Manage [MCP](mcp.md) server connections. Sub-commands:

| Usage | Effect |
|---|---|
| `/mcp` | List servers and connection status |
| `/mcp connect <server> [server ...]` | Connect servers now |
| `/mcp disconnect <server>` | Disconnect a server |
| `/mcp tools [server]` | List tools from a connected server |

## Help and exit

**`/help`** — Show the built-in command and tool reference.

**`/exit`, `/quit`** — Leave minacode. Your session is saved automatically and can
be resumed with `-c` or `--resume`.
