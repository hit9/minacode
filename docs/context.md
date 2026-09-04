# Context and caching

Each request contains more than the latest message. wizolt puts stable session context first,
then sends one append-only conversation log. This keeps the agent informed while giving supported
providers exact earlier user and tool boundaries to reuse.

## What the model receives

<div class="term-shot" role="img" aria-label="The message context from first to last: system instructions, project environment with session start time, optional skills and MCP indexes, then an append-only conversation containing user messages, assistant replies, tool results, Note state changes, resume events, and occasional compaction checkpoints."><span class="fs-goal">─ stable session prefix ─────────────────────────────</span><span>  system instructions      <span class="fs-i fs-dim">how the agent should operate</span></span><span>  project environment      <span class="fs-i fs-dim">directory · local start time · OS · shell</span></span><span>  skills and MCP indexes   <span class="fs-i fs-dim">only when available</span></span><span class="fs-goal">─ append-only conversation ──────────────────────────</span><span>  user · assistant · tools <span class="fs-i fs-dim">normal turn history</span></span><span>  Note calls and results   <span class="fs-i fs-dim">goal · plan · facts · checks</span></span><span>  lifecycle events         <span class="fs-i fs-dim">resume time in the user's local zone</span></span><span>  current turn             <span class="fs-i fs-dim">always appended last</span></span></div>

Times are local, written once when the session starts and again when it resumes, so both you and
the model read them without converting anything.

Tool definitions are sent beside this stack: built-in tools, `Skill` when skills are installed,
and the tools of <span class="marker">currently connected</span> MCP servers. Configured but
disconnected servers cost nothing. Reasoning the model returns is kept and replayed only where the
active provider expects it.

[Provider-side search](tools.md#provider-side-tools) is the exception to all of this: the pages it
reads are added by the provider, so they are neither shortened nor counted in the context fill, and
most providers bill the search on top of the tokens.

## Keeping context manageable

Large tool results are shortened before they enter the conversation; the agent can retrieve the
complete result later with `Recall`. Repeated skill instructions and MCP descriptions are replaced
with references to their first full copy instead of being sent in full again.

### Compaction

As a request approaches `runtime.max_context_tokens`, wizolt **compacts**: the older part of the
conversation is replaced by a short summary, and the most recent messages — about eight of them —
are kept as they are. The session continues in the same turn, so a long task does not have to stop.

The threshold leaves room for what the next request carries besides the conversation — the reply
the model may write, and the tool definitions — so compaction happens before the window is full.
The fill shown in the status bar measures the last request; compaction looks ahead to the next.

The summary is lossy, so each compaction also stores a verbatim excerpt of the messages it
evicted, as a **history segment**. The session log keeps the originals either way.

<div class="term-shot" role="img" aria-label="Compaction replaces older active conversation with one checkpoint containing the summary, full working state, and a segment pointer. RecallContext can list, search, and retrieve bounded verbatim excerpts, while the append-only session log retains earlier snapshots as the cold source of truth."><span class="fs-goal">─ active context (hot) ────────────────</span><span>  checkpoint       <span class="fs-i fs-dim">summary · goal · plan · facts · checks · seg.N</span></span><span>  recent messages  <span class="fs-i fs-dim">kept as they are</span></span><span class="fs-dim">─ recallable segments (warm) ──────────</span><span>  seg.1 · seg.2    <span class="fs-i fs-dim">listed/searched only when needed</span></span><span class="fs-dim">─ append-only session log (cold) ──────</span><span>  earlier snapshots<span class="fs-i fs-dim"> original messages</span></span><span> </span><span class="fs-dim"><span class="fs-i fs-goal">RecallContext(list/search/get)</span> finds an excerpt</span></div>

Each compaction names the span it evicted, in the same reply that writes the summary, so the title
describes the work rather than whichever message happened to start the window. The agent reaches
segments through `RecallContext` — listing, searching, or retrieving one — and none of them take
up room in a request until it does.

Only the newest 50 segments stay recallable; a session that compacts more often drops its oldest
spans. `seg.N` keys keep counting, so the agent is told a segment is gone rather than handed a
different one.

Run `/compact` to compact immediately rather than waiting for the threshold, for example before
starting a large refactor. `/status` reports how many compactions a session has done.

`/compact log` reviews them one by one. Each row shows when the pass ran, whether it was
automatic or manual, whether it covered prior conversation or the running turn, and how much it
evicted. Opening one shows the summary written at that point — the active context keeps only the
newest — and `/compact log seg.N` prints that summary without the viewer.

Neither form prints the stored excerpt — that is the agent's to retrieve — and a pass that finds
nothing to evict stores no segment at all, so the compaction count can exceed the segment count.

### When a summary does not arrive

Compaction always makes room, even when the summary request fails: the same messages leave the
context, with no summary written in their place. Work is not lost — the segment is still stored
and the agent can still recall it — but the checkpoint carries less, so tell wizolt what matters
if a long task continues past one.

`/compact log` marks such a pass `no summary`, and `/compact` reports the reason on the spot. The
usual causes are a summarizer that cannot fit the span, one slower than its `response_timeout`, one
that answers with something other than a summary, and a `[compaction]` entry missing its url, key,
or model. The message names the entry, so you know where to look. See
[Compaction model](configuration.md#compaction-model) for choosing one that avoids all four.

### Sessions started before these features

Resuming a session older than a given feature shows blanks rather than errors. Segments compacted
by an earlier version have no recorded time, scope, or model, so `/compact log` lists them with
`—` and says the summary predates the log; summary tokens spent back then stayed in the overall
`usage` total, so `/status` shows no `compaction usage` row until this session compacts again.
Everything recorded from now on fills in normally.

## Spending less

Four levers, in the order they usually pay off:

- **Keep the cache prefix stable.** Switching models or connecting a server mid-session restarts
  the reusable prefix; leaving them alone lets it grow. See [Prompt caching](#prompt-caching).
- **Give summaries a cheap model — or leave them where they are.** A cheaper entry costs less per
  token, but a summary that stays put reuses the cache the turn just filled. The `compaction
  cache` row of `/status` says which is winning. See
  [Compaction model](configuration.md#compaction-model).
- **Delegate bounded work to a cheap [worker](worker.md).** Its tokens are billed at the worker
  entry's rate, and its context never enters the parent's.
- **Size `runtime.max_context_tokens` to the work.** A larger window means fewer compactions but a
  more expensive request every turn.

`/status` shows what each is doing: cache ratio, conversation usage, and summary usage and cache
on their own rows.

## Prompt caching

Prompt caching lets a provider reuse work for an unchanged beginning of a request. Each request
usually starts with the same instructions, environment, tools, and earlier conversation, so only
the new tail needs processing.

<div class="term-shot" role="img" aria-label="Two request bars. Both start with the same long shaded prefix, which the provider reuses; only the shorter tail of each request is processed again."><span>previous  <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░</span></span><span>next      <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░░░░░</span></span><span> </span><span class="fs-dim">          <span class="fs-i fs-goal">█</span> reused prefix    ░ processed again</span></div>

A request is reused only up to its first difference, which is why the stable sections come first.
Anything that changes an early one — connecting an MCP server, installing a skill, switching
models, enabling a provider-side tool — shortens the reusable prefix; a compaction deliberately
starts a new one. Appending to the conversation, including Note updates and resume events, does
not.

Support differs by provider. OpenAI-compatible endpoints may match prefixes on their own, all the
way through the conversation. Anthropic caches only what it is told to, and wizolt marks the
instructions and tools — so there the conversation itself is processed again each turn.

### Checking the hit rate

`/status` reports the ratio for the latest request and for the whole session, with cache-write
tokens when the provider exposes them:

<div class="term-shot" role="img" aria-label="Two rows of /status: a cache row with a fill meter, the latest request's hit ratio and the session's, then a usage row with the call count and total tokens."><span><span class="fs-i fs-dim">cache</span>  <span class="fs-i fs-add">[████████████▊░]</span> last <span class="fs-i fs-sel">95.9%</span>; session <span class="fs-i fs-add">91.5%</span></span><span><span class="fs-i fs-dim">usage</span>  calls 14; total 182.3K</span></div>

The latest-request ratio also shows live in the status bar, beside the context fill,
updating with each response.

The ratio varies with the provider, model, prompt length, and conversation. When request prefixes
line up, it <span class="marker">can reach 90–99%</span>. This is an observation, not a guaranteed
rate.
