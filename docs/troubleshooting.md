# Troubleshooting

What each symptom means, and what to do about it.

## It will not start

**`missing config: provider.url, provider.key, provider.model`** — the active provider entry is
incomplete. Open `~/.minacode/config.toml` and fill in the three required keys, or run
`minacode --init-config` to write a fresh starter file. `/config` shows which entry is active.

**Requests fail with an authentication error** — the key is wrong for the url, or the entry
belongs to a different vendor than the model name suggests. Check that `url`, `key`, and `model`
in the active entry come from the same provider.

**`compaction provider ... is missing key, model`** — the `[compaction]` entry is incomplete.
That entry needs its own url, key, and model, exactly like a provider you converse with. See
[Compaction model](configuration.md#compaction-model).

## The model call fails

**It failed instantly instead of retrying.** Rate limits, timeouts, and server errors are
retried automatically, with the attempt shown as `retrying 2/6` on the divider. Two kinds of
failure never retry, because a second attempt would fail the same way: <span class="marker">quota
and billing errors</span> — an exhausted balance, an expired plan — and a response that hit the
model's output cap. Read the error text: it comes from your provider and names the account
problem.

**`Model response exceeded provider.response_timeout=600s`** — one response took longer than the
total-generation limit. Raise `response_timeout` for that provider entry, or set it to `0` to
wait indefinitely. See [Optional provider settings](configuration.md#optional-provider-settings).

**The reply mentions a model or parameter the provider does not know.** Set `api` explicitly on
that entry rather than leaving it to be detected, and check the model name against the provider's
own list. `/config` shows the protocol and reasoning effort actually in use.

## Context and summaries

**The context was trimmed but no summary appeared.** The summary request failed and minacode made
room anyway. `/compact log` marks that pass `no summary`. See
[When a summary does not arrive](context.md#when-a-summary-does-not-arrive).

**`context is over budget and nothing is left to compact`** — the latest exchange alone exceeds
the window, so there is nothing older to summarize. It is usually one enormous tool result. Start
a fresh session for the next task, or raise `runtime.max_context_tokens` if the model's window
allows.

**`/compact` says there is nothing to compact.** Everything left is recent enough that compaction
may not drop it. Nothing is wrong — the context is already as small as it gets.

## Cache and cost

**The cache ratio sits near zero.** A provider reuses a request only up to its first difference,
so anything that changes the beginning of a request — switching models, connecting an MCP server,
installing a skill — starts a fresh prefix. It recovers over the following turns if you leave
those alone. Not every provider caches, and each reports it differently; see
[Prompt caching](context.md#prompt-caching).

**Tokens are climbing faster than expected.** `/status` breaks the session into conversation and
summary usage. [Spending less](context.md#spending-less) lists what each lever does.

## Editing and tools

**`stale anchor ...`** — the file changed after the agent read it, so the edit was refused rather
than applied to the wrong lines. This is the guardrail working. Ask the agent to re-read the file
and try again.

**A tool asks for confirmation every time.** That is the default for anything that writes files or
runs commands. `/yolo` turns confirmations off for the session; read [Safety](safety.md) first.

**The code index says `stale` or `?`.** Run `/index` to rebuild it. Search and navigation still
work without it — they fall back to scanning — but symbol lookups are slower and less precise.

## Sessions

**`Session snapshot not found`** — the id does not exist under this data directory, or it was
swept. Run `/sessions all` to list everything saved, across projects.

**A session I expected is gone.** Sessions untouched for seven days are removed at startup.
Resuming one resets its clock; `runtime.session_retention_days = 0` keeps them indefinitely. See
[Sessions](usage.md#sessions).

**Resuming shows blanks where I expect details.** A session started before a feature existed has
nothing recorded for it. See
[Sessions started before these features](context.md#sessions-started-before-these-features).

**`/sessions` refuses to switch.** Switching mid-turn would abandon a request already in flight.
Press `Ctrl-C` first, then run it again.
