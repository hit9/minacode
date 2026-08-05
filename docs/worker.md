(worker-delegation)=
# Worker delegation

A **worker** is a second minacode session running **in the same process**. When the model
recognizes a bounded task that deserves an independent look — a review, a refactor, a second
opinion — it hands the task to the worker with the `Delegate` tool. The worker runs on its own
configured provider, with its own system prompt and a reduced tool set, and keeps its context
across delegations until you reset it. It never calls back into the parent: every delegation is
a serial detour whose result comes back to the parent model.

## Quick start

Give the worker its own provider entry — usually a **different vendor** than `provider.active`,
so its reviews cross-validate the parent's — then point `[worker] provider` at it and turn on
`[runtime] worker`:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"

# A second entry for the worker, from a different vendor.
[provider.anthropic]
url = "https://api.anthropic.com"
key = "sk-ant-..."
model = "claude-sonnet-4"

[worker]
provider = "anthropic"   # worker provider key; unset disables delegation
model = ""               # optional: override the entry's model (inherit by default)
reasoning = ""           # optional: override the entry's reasoning effort (inherit by default)
api = ""                 # optional: override the entry's wire protocol (inherit by default)

[runtime]
worker = true            # or /worker on
```

On the first run you will see the model call `Delegate` with an `order` it wrote itself, you
approve the send, and the worker takes over: a yellow bracket opens in the terminal, its
progress replaces the parent's row in the status bar, and its final answer comes back to the
parent model, which decides what to do next. The worker also inherits the session's language
setting (`[runtime] language` or `/language`) on every send.

`/worker` shows the worker's status, `/worker on` and `/worker off` toggle delegation for the
session, and `/worker reset` clears the worker's context. `/worker provider NAME`,
`/worker model MODEL`, `/worker reason EFFORT`, and `/worker api API` re-target a live worker
immediately and future spawns, mirroring the parent's `/provider` `/model` `/reason` `/api`
switches; `default` clears an override back to inheriting the provider entry. See
[Commands](commands.md) for the full syntax.

## What you see

### The delegation bracket

Two full-width rules bracket a delegation: a yellow `worker start` rule with the worker's live
provider and model plus the order's title (or first line), the worker's streamed lines, then a
yellow `worker done` rule with the step count, elapsed time, tokens in and out, and the files it
touched.

<div class="term-shot" role="img" aria-label="The delegation bracket: a full-width yellow worker start rule naming the worker's provider, model, and order title, a few worker tool lines beneath it, and a yellow worker done rule with step count, elapsed time, token counts, and touched files."><span class="fs-worker">──── worker start · anthropic/claude-sonnet-4 · Review the parser refactor ────</span><span class="fs-tool">  ├ Read minacode/loop.py</span><span class="fs-tool">  ├ Read tests/test_edit_tool.py</span><span class="fs-tool">  └ Bash uv run pytest tests/ -q</span><span class="fs-worker">──── worker done · steps 7 · 43.2s · 12.4K in / 1.1K out · minacode/loop.py, tests/ ────</span></div>

### Status bar while delegating

While the worker runs, the status bar swaps the parent's row for the worker's: a `[worker]`
marker, then the worker's provider and model, its reasoning effort, and its context fill with
cache ratio.

<div class="term-shot" role="img" aria-label="The status bar while a delegation runs: a yellow [worker] marker, then the worker's provider and model, its reasoning effort, and its context fill with cache ratio."><span><span class="fs-i fs-dim">delegating </span><span class="fs-i sb-worker">[worker]</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-warn">anthropic/claude-sonnet-4</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-reason">high</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-ctx">ctx 41% · cache 95%</span></span></div>

### The send approval brief

Every `Delegate send` asks for approval, even under `yolo` — the confirmation prints the title, a
one-line excerpt of the order, any explicit `language` or `max_steps`, and the effective worker
provider/model/effort/api, with `(inherit)` marking a field that inherits the provider entry's
value. `c` opens a small loop that adjusts those worker settings before you decide; a refused
send feeds your reason back to the model.

<div class="term-shot" role="img" aria-label="The Delegate send approval brief: FIELD rows for the title, an order excerpt, and the worker's effective provider, model, effort, and api with inherit markers, then a key legend row."><span class="fs-tool">Delegate send</span><span> </span><span><span class="fs-i fs-sel">title    </span>Review the parser refactor</span><span><span class="fs-i fs-sel">order    </span>Extract parser.py from loop.py, keep the CLI surface unchanged (… 3 more lines)</span><span><span class="fs-i fs-sel">provider </span>(inherit) anthropic</span><span><span class="fs-i fs-sel">model    </span>(inherit) claude-sonnet-4</span><span><span class="fs-i fs-sel">effort   </span>(inherit) high</span><span><span class="fs-i fs-sel">api      </span>(inherit) anthropic</span><span class="fs-dim">  Y/Enter approve · n refuse · c worker config · else reason</span></div>

## Semantics

- **Reset drops the conversation, not the work.** The worker owns only its conversation;
  `/worker reset` (or a `Delegate reset` call) clears that conversation — and the wrong beliefs
  it may have accumulated — on purpose, while file changes and merged diffs always survive.
- **Snapshots belong to the parent.** A worker's session id is the parent's with a `.w` suffix;
  worker snapshots are never listed in `/sessions`, and when the parent's snapshot expires its
  orphaned workers expire with it.
- **The tool block is frozen at session start.** The `Delegate` tool block is fixed from
  `[worker] provider` when the session begins, protecting the prompt-cache scope: changing the
  provider mid-session tunes an already-enabled delegation, but enabling delegation from scratch
  (provider unset at start) takes effect after a restart.
- **Sends are confirmed even under `yolo`.** The order is a spec the model wrote for itself, and
  the approval brief is the one cheap check on it — every send asks.
- **The worker inherits your environment.** It shares the parent's cwd, skills, MCP servers, and
  language setting; only files and the repository truly belong to both.
