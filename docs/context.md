# Context and caching

Each request contains more than the latest message. nanocode arranges the model's context so
the stable parts come first, the active conversation follows, and frequently changing task
state comes last. This keeps the agent informed while giving supported providers a long,
reusable prompt prefix.

## What the model receives

<div class="term-shot" role="img" aria-label="The message context from first to last: system instructions, project environment, skills, MCP servers, history index, and conversation history form the reused prefix; task memory follows the conversation, and the current turn is last."><span class="fs-goal">─ reused prefix ─────────────────────────────────────</span><span>  system instructions      <span class="fs-i fs-dim">how the agent should operate</span></span><span>  project environment      <span class="fs-i fs-dim">directory · OS · shell · commands</span></span><span>  skills index             <span class="fs-i fs-dim">only when skills are installed</span></span><span>  MCP servers              <span class="fs-i fs-dim">only when a server is connected</span></span><span>  history index            <span class="fs-i fs-dim">grows when compaction creates seg.N</span></span><span>  conversation history     <span class="fs-i fs-dim">reused until compaction</span></span><span class="fs-dim">─ dynamic tail ──────────────────────────────────────</span><span>  task memory              <span class="fs-i fs-dim">goal · plan · facts · checks</span></span><span>  current turn             <span class="fs-i fs-dim">the latest messages and results</span></span></div>

Tool definitions are sent beside this message stack: built-in tools, `Skill` when skills are
installed, and MCP tools and resources from <span class="marker">currently connected servers</span>.
Configured but disconnected servers do not consume context.

## Keeping context manageable

Large tool results are shortened before they enter the conversation; the agent can retrieve the
complete result later with `Recall`. Repeated skill instructions and MCP descriptions are replaced
with references to their first full copy instead of being sent in full again.

### Compaction

When the estimated context reaches `runtime.max_context_tokens`, nanocode **compacts**: the older
part of the conversation is replaced by a short summary, and the most recent messages are kept
as they are. The session continues in the same turn, so a long task does not have to stop.

The summary in the active context is lossy, but each compaction also captures a bounded verbatim
excerpt of the evicted messages as a **history segment**. Earlier snapshots in the append-only
session log remain the cold source of truth, so compaction does not rewrite them.

<div class="term-shot" role="img" aria-label="Compaction moves older conversation out of the active context. The active context keeps a short summary and bounded history index; RecallContext retrieves bounded verbatim excerpts, while the append-only session log retains earlier snapshots as the cold source of truth."><span class="fs-goal">─ active context (hot) ────────────────</span><span>  history index    <span class="fs-i fs-dim">bounded seg.N titles</span></span><span>  summary          <span class="fs-i fs-dim">short rewrite of older talk</span></span><span>  recent messages  <span class="fs-i fs-dim">kept as they are</span></span><span class="fs-dim">─ recallable segments (warm) ──────────</span><span>  seg.1 · seg.2    <span class="fs-i fs-dim">bounded verbatim excerpts</span></span><span class="fs-dim">─ append-only session log (cold) ──────</span><span>  earlier snapshots<span class="fs-i fs-dim"> original messages</span></span><span> </span><span class="fs-dim"><span class="fs-i fs-goal">RecallContext(seg.N)</span> pulls an excerpt back</span></div>

The bounded history index is a separate context section before conversation history. The agent
calls `RecallContext` with a `seg.N` key to retrieve that segment's stored excerpt when it needs
earlier detail. Task memory (goal, plan, facts, checks) follows conversation history and is carried
across untouched, which is why decisions worth keeping belong there.

Run `/compact` to compact immediately rather than waiting for the threshold, for example before
starting a large refactor. `/status` reports how many compactions a session has done.

## Prompt caching

Prompt caching lets a provider reuse work for an unchanged beginning of a request. The next request
usually begins with the same instructions, environment, tools, history index, and earlier
conversation, so only the new tail needs to be processed. Task memory follows conversation history
so updating it does not invalidate that larger prefix. Connecting an MCP server, changing installed
skills, switching models, or otherwise changing an early section can reduce the next request's
cache hit.

<div class="term-shot" role="img" aria-label="Two request bars. Both start with the same long shaded prefix, which the provider reuses; only the shorter tail of each request is processed again."><span>previous  <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░</span></span><span>next      <span class="fs-i fs-goal">████████████████████████</span><span class="fs-i fs-dim">░░░░░░░░░░</span></span><span> </span><span class="fs-dim">          <span class="fs-i fs-goal">█</span> reused prefix    ░ processed again</span></div>

A request is reused only up to its first difference, so a change near the beginning — connecting
an MCP server, installing a skill, switching models — shortens the reusable prefix. That is why
the stable sections are placed first.

OpenAI-compatible providers may reuse matching prefixes automatically; nanocode supplies a stable
cache key where the provider supports one. For Anthropic, nanocode explicitly marks the tools and
system instructions as an ephemeral cacheable prefix. Provider support and accounting differ.

### Checking the hit rate

`/status` reports the cached prompt tokens the provider counted — once for the whole session, and
again for the most recent request:

<div class="term-shot" role="img" aria-label="The usage row of /status, showing call count, total tokens, cached tokens for the whole session, and cached tokens for the latest request."><span><span class="fs-i fs-dim">usage</span>  calls 14; total 182304; cached <span class="fs-i fs-add">148992/162880</span> (<span class="fs-i fs-add">91.5%</span>); last <span class="fs-i fs-sel">21120/22016</span> (<span class="fs-i fs-sel">95.9%</span>)</span><span class="fs-dim">                                  └─ whole session ─┘        └─ latest request ─┘</span></div>

The ratio varies with the provider, model, prompt length, and conversation. When request prefixes
line up, it <span class="marker">can reach 90–99%</span>. This is an observation, not a guaranteed
rate.
