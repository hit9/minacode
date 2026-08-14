# Context and caching

Each request contains more than the latest message. minacode puts stable session context first,
then sends one append-only conversation log. This keeps the agent informed while giving supported
providers exact earlier user and tool boundaries to reuse.

## What the model receives

<div class="term-shot" role="img" aria-label="The message context from first to last: system instructions, project environment with session start time, optional skills and MCP indexes, then an append-only conversation containing user messages, assistant replies, tool results, Note state changes, resume events, and occasional compaction checkpoints."><span class="fs-goal">─ stable session prefix ─────────────────────────────</span><span>  system instructions      <span class="fs-i fs-dim">how the agent should operate</span></span><span>  project environment      <span class="fs-i fs-dim">directory · local start time · OS · shell</span></span><span>  skills and MCP indexes   <span class="fs-i fs-dim">only when available</span></span><span class="fs-goal">─ append-only conversation ──────────────────────────</span><span>  user · assistant · tools <span class="fs-i fs-dim">normal turn history</span></span><span>  Note calls and results   <span class="fs-i fs-dim">goal · plan · facts · checks</span></span><span>  lifecycle events         <span class="fs-i fs-dim">resume time in the user's local zone</span></span><span>  current turn             <span class="fs-i fs-dim">always appended last</span></span></div>

The environment records the session's start time once as a local ISO timestamp with its numeric
offset, such as `2026-07-30T20:34:56+08:00`. Resuming appends a user-role lifecycle event with the
new local time. These timestamps are directly readable by both the user and model; no UTC
conversion is required, and there is no changing date block inserted into later requests.

Tool definitions are sent beside this message stack: built-in tools, `Skill` when skills are
installed, and MCP tools and resources from <span class="marker">currently connected servers</span>.
Configured but disconnected servers do not consume context.

When a model exposes reasoning, minacode keeps the returned protocol data in the session but only
replays what the active provider expects. Some APIs require preserved reasoning across turns;
others ignore old reasoning unless a preserved-thinking option is enabled, while still needing it
inside a multi-step tool call. Opaque signatures and encrypted reasoning are returned unchanged
when their protocol requires them, but are not shown as model-readable text.

When [provider-side search](tools.md#provider-side-tools) is enabled, the pages it reads are added
to the context by the provider rather than by minacode, so they are not shortened and the context
fill cannot predict them: a turn that searches arrives larger than one that does not, and most
providers charge for the search on top of the tokens.

## Keeping context manageable

Large tool results are shortened before they enter the conversation; the agent can retrieve the
complete result later with `Recall`. Repeated skill instructions and MCP descriptions are replaced
with references to their first full copy instead of being sent in full again.

### Compaction

As the estimated request approaches `runtime.max_context_tokens`, minacode **compacts**: the older
part of the conversation is replaced by a short summary, and the most recent messages are kept as
they are. The estimate uses the effective Chat, Responses, or Anthropic request, so reasoning that
the provider will discard does not cause early compaction. The trigger reserves the configured
provider output cap (16K when unspecified), tool schemas, and a safety margin of at least 4K. The
session continues in the same turn, so a long task does not have to stop.

The context fill shown in the status bar and `/status` is the provider-reported token count of the
last request; compaction still triggers on the estimate, which projects the next request.

The checkpoint summary in the active context is lossy, but each compaction also captures a bounded verbatim
excerpt of the evicted messages as a **history segment**. Earlier snapshots in the append-only
session log remain the cold source of truth, so compaction does not rewrite them.

<div class="term-shot" role="img" aria-label="Compaction replaces older active conversation with one checkpoint containing the summary, full working state, and a segment pointer. RecallContext can list, search, and retrieve bounded verbatim excerpts, while the append-only session log retains earlier snapshots as the cold source of truth."><span class="fs-goal">─ active context (hot) ────────────────</span><span>  checkpoint       <span class="fs-i fs-dim">summary · goal · plan · facts · checks · seg.N</span></span><span>  recent messages  <span class="fs-i fs-dim">kept as they are</span></span><span class="fs-dim">─ recallable segments (warm) ──────────</span><span>  seg.1 · seg.2    <span class="fs-i fs-dim">listed/searched only when needed</span></span><span class="fs-dim">─ append-only session log (cold) ──────</span><span>  earlier snapshots<span class="fs-i fs-dim"> original messages</span></span><span> </span><span class="fs-dim"><span class="fs-i fs-goal">RecallContext(list/search/get)</span> finds an excerpt</span></div>

Each compaction names the span it evicted, in the same reply that produces the summary, so the
title describes the work rather than whichever message happened to start the window.

Segment titles do not occupy every request. The agent uses `RecallContext` to list them newest
first, regex-search stored titles and text, or retrieve selected `seg.N` excerpts. `Note` can view
the current goal, plan, facts, and checks; updates remain visible in their original tool-call
history until a compaction checkpoint consolidates the complete current state.

Run `/compact` to compact immediately rather than waiting for the threshold, for example before
starting a large refactor. `/status` reports how many compactions a session has done.

`/compact log` reviews them one by one: each stored segment shows when it was compacted, whether
the pass was automatic or manual, whether it covered prior conversation or the running turn, and
how much it evicted. Opening one shows the summary that compaction produced at that point, which
the active context no longer keeps once a later compaction replaces it. `/compact log seg.N`
prints that summary without the viewer. A pass that found nothing to evict stores no segment, so
the compaction count can exceed the number of segments.

Neither form prints the stored excerpt of the evicted messages; that is the agent's to retrieve
with `RecallContext`.

While a summary is running, the status bar names the entry doing it — `[compaction]`, then that
entry's model and effort — the same way it names an in-flight [worker](worker.md). The row goes
back to the conversation's own model when the summary lands, and does not change at all when
compaction runs on the model already shown.

<div class="term-shot" role="img" aria-label="The status bar while a summary runs: a yellow [compaction] marker, then the entry and model serving the summary, its reasoning effort, and the conversation's context fill with cache ratio."><span><span class="fs-i fs-dim">compacting </span><span class="fs-i sb-ctx">[compaction]</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-warn">deepseek/deepseek-v4-flash</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-reason">off</span><span class="fs-i sb-sep"> | </span><span class="fs-i sb-ctx">ctx 41% · cache 95%</span></span></div>

### When a summary does not arrive

Compaction always makes room, even when the summary request fails: the same messages leave the
context, with no summary written in their place. Work is not lost — the segment is still stored
and the agent can still recall it — but the checkpoint carries less, so tell minacode what matters
if a long task continues past one.

`/compact log` marks such a pass `no summary`, and `/compact` reports the reason on the spot. The
usual causes are a summarizer that cannot fit the span, one slower than its `response_timeout`, and
a `[compaction]` entry missing its url, key, or model — that last one is refused by name before
the request, so the message tells you which entry to fix. See
[Compaction model](configuration.md#compaction-model) for choosing an entry that avoids all three.

### Sessions started before these features

Resuming a session older than a given feature shows blanks rather than errors. Segments compacted
by an earlier version have no recorded time, scope, or model, so `/compact log` lists them with
`—` and says the summary predates the log; summary tokens spent back then stayed in the overall
`usage` total, so `/status` shows no `compaction usage` row until this session compacts again.
Everything recorded from now on fills in normally.

## Prompt caching

Prompt caching lets a provider reuse work for an unchanged beginning of a request. The next request
usually begins with the same instructions, environment, tools, and earlier conversation, so only
the new tail needs to be processed. Note updates and resume events append to the conversation and
therefore do not move an earlier breakpoint. A compaction intentionally starts one new cache epoch.
Connecting an MCP server, changing installed skills, switching models, or otherwise changing an
early section can reduce the next request's cache hit.

<div class="term-shot" role="img" aria-label="Two request bars. Both start with the same long shaded prefix, which the provider reuses; only the shorter tail of each request is processed again."><span>previous  <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░</span></span><span>next      <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░░░░░</span></span><span> </span><span class="fs-dim">          <span class="fs-i fs-goal">█</span> reused prefix    ░ processed again</span></div>

A request is reused only up to its first difference, so a change near the beginning — connecting
an MCP server, installing a skill, switching models, enabling a provider-side tool — shortens the
reusable prefix. That is why
the stable sections are placed first.

OpenAI-compatible providers may reuse matching prefixes automatically; minacode supplies a stable
cache key where the provider supports one. For Anthropic, minacode explicitly marks the tools and
system instructions as an ephemeral cacheable prefix. Provider support and accounting differ.

### Checking the hit rate

`/status` reports cache-read tokens for the whole session and latest request, plus cache-write
tokens when the provider exposes them:

<div class="term-shot" role="img" aria-label="The usage row of /status, showing call count, total tokens, cached tokens for the whole session, and cached tokens for the latest request."><span><span class="fs-i fs-dim">usage</span>  calls 14; total 182304; cached <span class="fs-i fs-add">148992/162880</span> (<span class="fs-i fs-add">91.5%</span>); last <span class="fs-i fs-sel">21120/22016</span> (<span class="fs-i fs-sel">95.9%</span>)</span><span class="fs-dim">                                  └─ whole session ─┘        └─ latest request ─┘</span></div>

The latest-request ratio also shows live in the status bar, beside the context fill,
updating with each response.

The ratio varies with the provider, model, prompt length, and conversation. When request prefixes
line up, it <span class="marker">can reach 90–99%</span>. This is an observation, not a guaranteed
rate.
