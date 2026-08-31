# Troubleshooting

Find the symptom; the fix is beside it.

## Starting up

| Symptom | Fix |
|---|---|
| `missing config: provider.url, ...` | The active provider entry is incomplete. Fill the three keys, or run `wizolt --init-config` for a fresh file |
| Authentication errors | `url`, `key`, and `model` must come from the same provider |
| `compaction provider ... is missing` | The `[compaction]` entry needs its own url, key, and model. See [Compaction model](configuration.md#compaction-model) |

## Compatibility catalog

| Symptom | Fix |
|---|---|
| Provider behavior looks stale | Run `/catalog` to compare the active, bundled, and cached versions, then `/catalog sync` to check immediately. See [Compatibility catalog](catalog.md) |
| `/catalog sync` reports a network or validation error | The current catalog stays active. Check the connection or reported document error and retry later; no request policy was partially applied |
| `/catalog` reports an invalid cache or same-version conflict | The bundled copy remains active. Upgrade wizolt for a newer bundled copy, or wait for a higher valid catalog version |

## Model calls

| Symptom | Fix |
|---|---|
| Failed at once, no retry | Quota and billing errors never retry — a second try fails the same way. The error text comes from your provider |
| `Model response exceeded provider.response_timeout` | Raise `response_timeout` on that entry, or set `0` to wait indefinitely |
| Model or parameter rejected | Set `api` explicitly instead of leaving it to detection, and check the model name against the provider's list. `/config` shows what is in use |

Rate limits, timeouts, and server errors do retry, counting up on the divider as `retrying 2/6`.

## Context and summaries

| Symptom | Fix |
|---|---|
| Context trimmed, no summary | The summary request failed; room was made anyway. See [When a summary does not arrive](context.md#when-a-summary-does-not-arrive) |
| `nothing is left to compact` | The latest exchange alone fills the window, usually one huge tool result. Start a fresh session, or raise `runtime.max_context_tokens` |
| `/compact` says nothing to compact | Everything left is too recent to drop. Nothing is wrong |
| Cache ratio near zero | Something changed early in the request — a model switch, a new server, a new skill. It recovers over the next turns. See [Prompt caching](context.md#prompt-caching) |
| Tokens climbing fast | `/status` splits conversation from summary usage; [Spending less](context.md#spending-less) lists the levers |

## Tools and edits

| Symptom | Fix |
|---|---|
| `source target changed` | The file changed after it was read, so the edit was refused rather than misapplied. Ask for a re-read |
| Confirmation every time | The default for writing files and running commands. `/yolo` turns it off; read [Safety](safety.md) first |
| Code index `stale` or `?` | Run `/index`. Search still works without it, just slower and less precise |

## Sessions

| Symptom | Fix |
|---|---|
| `Session snapshot not found` | Gone or from another project. `/sessions all` lists everything saved |
| A session disappeared | Sessions idle for seven days are swept at startup. `runtime.session_retention_days = 0` keeps them |
| Blanks where details belong | The session predates the feature. See [Sessions started before these features](context.md#sessions-started-before-these-features) |
| `/sessions` refuses to switch | A request is in flight. Press `Ctrl-C`, then run it again |

## Delegation

| Symptom | Fix |
|---|---|
| Nothing ever delegates | The model decides, and there is no way to force one. See [When nothing delegates](worker.md#when-nothing-delegates) |
