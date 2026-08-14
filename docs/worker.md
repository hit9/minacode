(worker-delegation)=
# Worker delegation

A **worker** is a second minacode session running in the same process. When the model meets a
bounded task that deserves an independent look — a review, a refactor, a second opinion — it hands
the task over with the `Delegate` tool.

The worker has its own provider, its own system prompt, and a reduced tool set, and keeps its
context across delegations until you reset it. It never calls back into the parent: a delegation
is a detour that ends by returning its result.

The point is pairing models by cost. A large model orchestrates as the parent; the worker's tasks
arrive already bounded and spec'd, so a small cheap model handles them at a fraction of the
parent's rate.

## Quick start

The parent orchestrates, so it usually runs the larger model; the worker runs bounded tasks,
so a faster, cheaper entry is enough for it — ideally from a **different vendor** than
`provider.active`, so its reviews cross-validate the parent's. Point `[worker] provider` at
that entry and turn on `[runtime] worker`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.openai.com"
key = "sk-..."
model = "gpt-5.5"

# A faster, cheaper entry for the worker, from a different vendor.
[provider.deepseek]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"

[worker]
provider = "deepseek"    # worker provider key; unset disables delegation
model = ""               # optional: override the entry's model (inherit by default)
reasoning = ""           # optional: override the entry's reasoning effort (inherit by default)
api = ""                 # optional: override the entry's wire protocol (inherit by default)

[runtime]
worker = true            # or /worker on
```

The model decides when to delegate. When it does, it writes the `order` itself and asks you to
approve the send; the worker then takes over the terminal until it answers, and its answer goes
back to the parent, which decides what to do next.

| Command | Effect |
|---|---|
| `/worker` | Show its provider, model, state, and rounds |
| `/worker on` · `/worker off` | Allow or stop delegation for this session |
| `/worker reset` | Clear the worker's context |
| `/worker provider NAME` | Point it at another provider entry |
| `/worker model` · `reason` · `api` | Override one field of that entry; `default` restores it |

[Commands](commands.md) has the full syntax.

## What you see

### The delegation bracket

Two full-width rules bracket a delegation: a yellow `worker start` rule with the worker's live
provider and model plus the order's title (or first line), the worker's streamed lines, then a
yellow `worker done` rule with the step count, elapsed time, tokens in and out, and the files it
touched.

<div class="term-shot" role="img" aria-label="The delegation bracket: a full-width yellow worker start rule naming the worker's provider, model, and order title, a few worker tool lines beneath it, and a yellow worker done rule with step count, elapsed time, token counts, and touched files."><span class="fs-worker">──── worker start · deepseek/deepseek-v4-flash · Review the parser refactor ────</span><span class="fs-tool">  ├ Read minacode/loop.py</span><span class="fs-tool">  ├ Read tests/test_edit_tool.py</span><span class="fs-tool">  └ Bash uv run pytest tests/ -q</span><span class="fs-worker">──── worker done · steps 7 · 43.2s · 12.4K in / 1.1K out · minacode/loop.py, tests/ ────</span></div>

### Status bar while delegating

While the worker runs, the status bar swaps the parent's row for the worker's: a `[worker]`
marker, then the worker's provider and model, its reasoning effort, and its context fill with
cache ratio.

<div class="term-shot" role="img" aria-label="The status bar while a delegation runs: a yellow [worker] marker, then the worker's provider and model, its reasoning effort, and its context fill with cache ratio."><span><span class="fs-i fs-dim">delegating </span><span class="fs-i sb-worker">[worker]</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-warn">deepseek/deepseek-v4-flash</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-reason">medium</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-ctx">ctx 41% · cache 95%</span></span></div>

### The send approval brief

Every `Delegate send` asks for approval, even under `yolo` — the confirmation prints the title, a
one-line excerpt of the order, any explicit `language` or `max_steps`, and the effective worker
provider/model/effort/api, with `(inherit)` marking a field that inherits the provider entry's
value. A refused send feeds your reason back to the model.

The actions sit in a row above the input line, with `Approve` selected. `Tab` and the arrows move
along it, `Enter` fires what is selected:

| Key | Action |
| --- | --- |
| `Enter` | do the selected action — `Approve` unless you moved |
| `Tab` / `→` / `Shift-Tab` / `←` | move along the row |
| `Esc` | refuse without a reason |
| anything printable | start writing a refusal reason |

Typing always goes to the reason, so no letter is ever a shortcut and no reason is unwritable. While
you are typing, the row dims and `Enter` sends the reason instead; `Esc` takes the reason back and
returns to the row, and refuses only once there is nothing left to take back.

Every tool's approval works this way — non-`Delegate` ones simply offer `Approve` and `Refuse`.
Without a TUI (piped input, a headless run) the row is gone and the same actions are typed out:
`y`, `n`, `v`, `c`, each followed by Enter.

<div class="term-shot" role="img" aria-label="The Delegate send approval brief: FIELD rows for the title, an order excerpt, and the worker's effective provider, model, effort, and api with inherit markers, then a live action row with Approve selected, followed by the reason input line."><span class="fs-tool">Delegate send</span><span> </span><span><span class="fs-i fs-sel">title    </span>Review the parser refactor</span><span><span class="fs-i fs-sel">order    </span>Extract parser.py from loop.py, keep the CLI surface unchanged (… 3 more lines)</span><span><span class="fs-i fs-sel">provider </span>(inherit) deepseek</span><span><span class="fs-i fs-sel">model    </span>(inherit) deepseek-v4-flash</span><span><span class="fs-i fs-sel">effort   </span>(inherit) medium</span><span><span class="fs-i fs-sel">api      </span>(inherit) chat</span><span><span class="fs-i fs-sel"> Approve </span><span class="fs-dim">   View order    Worker config    Refuse     Tab to move</span></span><span class="fs-dim">  reason › </span></div>

`View order` opens the full order in a read-only viewer — the same field rows, then the order
itself rendered as markdown. Scroll it, then `Esc` back to the actions.

<div class="term-shot" role="img" aria-label="The order viewer opened from View order: a read-only header, the same field rows as the brief with cyan labels, a rule, then the full order rendered as markdown with an underlined heading, inline code, and a bullet list, above a line of scrolling keys."><span class="fs-dim">  Delegate order · read-only</span><span><span class="fs-i fs-sel">  title   </span>  Review the parser refactor</span><span><span class="fs-i fs-sel">  language</span>  python</span><span><span class="fs-i fs-sel">  provider</span>  (inherit) deepseek</span><span><span class="fs-i fs-sel">  model   </span>  (inherit) deepseek-v4-flash</span><span class="fs-dim">  ────────────────────────────────────────────────────────────────────────</span><span>  <span class="fs-i fs-md-h">Review the parser refactor</span></span><span>&nbsp;</span><span>  Extract <span class="fs-i fs-md-code">parser.py</span> from <span class="fs-i fs-md-code">loop.py</span> and keep the CLI surface unchanged.</span><span>&nbsp;</span><span>   • keep <span class="fs-i fs-md-code">tokenize()</span> public</span><span>   • add a test for empty input</span><span>&nbsp;</span><span class="fs-dim">  ↑/↓ · Ctrl-U/D · g/G · Esc/q close</span></div>

## When nothing delegates

Delegation is the model's call, not a command you run — there is no way to force one. If a session
never delegates, check in this order:

1. `/worker` — it reports the configured entry, or `worker: no active session` with the
   `[worker] provider` it would use.
2. `[worker] provider` and `[runtime] worker` in your config, or `/worker on` for this session.
3. **Whether delegation was available when the session started.** If `[worker] provider` was unset
   at startup, the model was never offered the tool, and turning it on now applies to the next
   session. Changing the provider mid-session is fine — that only re-targets a delegation the
   model already has.

Otherwise the model simply saw no task worth handing over, which is the normal case for short work.

## Good to know

- **Reset drops the conversation, not the work.** `/worker reset` clears the worker's context —
  along with any wrong beliefs it picked up — while file changes and merged diffs survive.
- **Every send asks, even under `yolo`.** The order is a spec the model wrote for itself, so the
  approval brief is the one cheap check on it.
- **The worker inherits your environment.** Same directory, skills, MCP servers, and language
  setting; the files and the repository are what both actually share.
- **Its tokens are its own.** The worker bills to its own entry and keeps its own context, and
  `/status` gives it separate rows — provider and model, context fill with round count, and cache
  ratio — so a delegation never blurs into the parent's numbers.
- **Its sessions are not yours to manage.** Worker snapshots ride along with the parent's and
  never appear in `/sessions`; when the parent's snapshot expires, the worker's goes with it.
