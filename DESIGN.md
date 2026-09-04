# Design notes

This file records decisions whose rationale is easy to lose and costly to rediscover. Keep it
short: document durable conclusions, not implementation diaries or complete investigation logs.

## Orientation

wizolt turns one user request into a bounded loop of model and tool calls in one local process.
Four objectives, often in tension, explain most decisions below:

1. **Resumable.** A session survives a crash, an interrupt, or a quit at any point.
2. **Protocol-neutral.** History is stored in one model; Chat, Responses, and Anthropic formats
   exist only at the send boundary.
3. **Bounded.** Context, retained output, and previews all have ceilings; nothing grows forever.
4. **Truthful.** The screen reports real state, and the terminal's own scrollback stays intact.

Modules (dependencies point downward only):

```
              __main__                     entry, startup ordering
                  |
    cli/ -- tui/ -- render.py              commands (cli/commands.py, cli/modals.py, cli/worker.py),
        |                                    TUI runtime (cli/runtime.py), view (cli/view.py),
        |                                    app (tui/app.py), view state (tui/views.py)
    engine.py                              the turn loop: commit or roll back
        |
        +-- context.py                     request projection, compaction
        +-- runner.py                      tool batch execution
                  |
              model/                       wire protocols (chat.py, responses.py,
                  |                          anthropic.py), streaming, retry
   tools/   mcp/   skill.py               vertical features
                  |
             session/                      durable semantic state (__init__.py)
                  |                         and snapshot persistence (store.py)
                  |
              image.py                     image storage and model projection
                  |
   base.py   config.py   providers/compat.py  value types, settings, policy
                  |
            providers/catalog.py           evidence-backed compatibility data
```

Three import rings stay runtime-only so the module graph above stays a DAG. Every upward edge is a
deferred import commented at its call site; lifting one to module scope makes the cycle part of
startup.

- **Orchestration ring (`tools/` ↔ engine).** `Delegate` spawns a worker by constructing
  `engine.Agent` (`tools/delegate.py`). The downward direction — engine/runner/model importing
  `tools` — is module scope. `ToolScript` needs no edge of its own: it uses the `ToolRunner` it was
  handed (`TYPE_CHECKING` only) and gets edit planning from `tools/editplan.py`. Separately, inside
  `tools/` a submodule that needs `TOOL_REGISTRY` or `tool_payload` imports it locally, because the
  registry in `__init__.py` is built on top of every tool module.
- **Session features (`session/` ↔ mcp/skill/mentions).** `Session` itself is feature-free:
  `__post_init__` never reaches upward. `bootstrap_features()` (deferred imports inside) attaches
  `MCPManager`/`SkillLibrary`/`FileMentions` when needed, called by `Session.from_config_file` and
  `Session.load_snapshot`; the delegate worker handoff injects the parent's `skills`/`mcp` fields
  explicitly instead, and `session/store.py` still imports its parent package at load time.
- **Assets (`image.py` ↔ session/).** `session/` imports the image value types at module scope;
  `ImageInputs.assets_dir` reaches back for `SessionSnapshotStore.session_path`, the one place the
  asset directory's layout is owned (`image.py`).

A turn, and its three endings:

```
  user input
      |
      v
  +-- Agent.run -------------------------------------------------+
  |                                                              |
  |   claim queued input                                         |
  |         |                                                    |
  |         v                                                    |
  |   context.prepare_messages --> model.request                 |
  |         ^                            |                       |
  |         |                     tool calls?  -- no --> answer  |
  |         |                            | yes                   |
  |         |                            v                       |
  |         +--------------------- runner.run(batch)             |
  |                                                              |
  |   checkpoint: turn start, each tool batch, each follow-up    |
  +--------------------------------------------------------------+
      |                    |                      |
   commit               interrupt               error
      v                    v                      v
  append to           retract, or keep       flush partial turn,
  session.messages    partial + marker       re-raise
```

The turn is a transaction: messages accumulate outside durable history until one of those three
endings, and nothing else may append mid-turn.

## Common pitfalls

Each looks like a cleanup and breaks something the code depends on; the section naming the rule
is in parentheses.

- **Lifting a deferred import to module scope.** Startup latency is a feature; the SDKs cost ~0.8s
  and are not needed until the first request (Startup path).
- **Rewriting stored history in a request transform.** Replay rules, image expansion, and schema
  dedup are send-time only; a resumed session must equal the saved one (Context is a projection).
- **Returning fewer tool results than the model emitted calls.** Refused, failed, skipped, and
  interrupted calls each need a matching result, or replay is invalid (Tool-call lifecycle).
- **Inserting context between the stable layers.** Saving tokens mid-prompt invalidates the cached
  prefix for every later turn (Context is a projection).
- **Persisting a live preview row, or reading state back off the screen.** Rendered text is never
  the source of truth (Three forms of state, Terminal boundary).
- **Expecting compaction to rescue an oversized fixed prefix.** It cannot; bound the source at its
  owner or fail clearly (Compaction).
- **Retrying a failure that is not transient.** Cancellation, capability rejection, and validation
  errors are decisions, not glitches (Failure boundaries).
- **Mocking the behavior under test instead of the external boundary** (Test design).

## Maintenance

- Docstrings describe interfaces and contracts, not development history.
- Comments protect local, non-obvious invariants and may link to primary evidence.
- Add a note here only for a cross-cutting decision maintainers may otherwise reopen; keep old
  conclusions visible and mark them superseded when a decision changes.

### Engineering posture

- **Abstract:** deep, local, earned abstractions behind small interfaces; remove dead code and
  pass-through wrappers.
- **Layer:** dependencies point from orchestration toward stable lower-level concepts; lower
  modules never import presentation or orchestration.
- **Keep it simple:** smallest cohesive, behavior-preserving change; no speculative specialization;
  keep the changelog aligned.
- Prefer generic standards; specialize only for a necessary, documented incompatibility, with
  primary evidence beside the rule.
- Prefer explicit imports and pragmatic typing; suppress a type error only when runtime behavior is
  demonstrably safe.
- Test contracts and reproduced regressions at their boundaries; CI enforces formatting, lint,
  typing, and the full suite.
- Keep UI and user docs quiet and direct; keep compatibility machinery and investigation history
  out of the common user path.

### Test design

Tests protect observable contracts and reproduced regressions, not implementation shape.

- Prefer black-box tests at the narrowest stable public boundary observing complete behavior;
  white-box only when a pure algorithm or edge cannot be exercised clearly there.
- A bug fix reproduces the real failure, then covers the intended result and unsafe paths that must
  stay rejected.
- Assert semantic output, durable state, or protocol payloads; exact text, call order, or rendering
  only when those details are the contract.
- Mock external or nondeterministic boundaries (providers, clocks, processes, terminals); never the
  core behavior under test.
- Keep tests deterministic and fast; reserve PTY/tmux for behavior that truly needs a terminal.

## System shape

- `base.py` defines configuration, shared value types, and error categories, plus the log-line
  vocabulary and resource handles; `providers/compat.py` folds `providers/catalog.py` data into resolved
  request policy.
- `Session` owns protocol-neutral semantic state (messages, transcript, checkpoints, queued input,
  retained output, diffs, usage, session resources); its snapshot codec decides what is persistable.
- Agent semantics split by owner: `context.py` projects/compacts, `model/` owns adapters,
  streaming, retry, `runner.py` owns execution/cancellation, `engine.py` composes the turn loop.
  Dependencies point downward only, so no pair needs a deferred import.
- `CommandLoop`/`TuiRuntime` orchestrate commands and transitions; `TuiApp` owns input, keys,
  layout, modals; `render.py` owns presentation.
- `tools/`, `image.py`, `mcp/`, `skill.py` are vertical features that never leak storage or UI
  details; `tools/` splits built-ins by capability, with the registry in `__init__.py`.
- How a call *reads* is a tool concern, not a runner one: `tools/tooloutput.py` bounds and parses
  result text, `tools/toolblocks.py` assembles the approval/rejection/finish `LogBlock` trees. Both
  are pure over the call, the tool, and the session — `render.py` still owns turning a block into
  styled terminal output, so this is the log-line vocabulary from `base.py`, not UI leaking down.
  Keeping them out of `ToolRunner` is what lets transcript replay render a saved call without
  standing up a live runner.
- Inside `toolblocks`, `finish_display`/`approval_display` branch on `call.name` for the handful of
  tools with a shaped result (Note, Bash, MCP, ToolScript, Ask, Delegate, ViewImage). That switch
  stays deliberately: a rendering hook on the `Tool` base would put six special cases into the
  interface every tool implements, and one chain that reads top to bottom is easier to keep
  consistent than seven overrides. Push the branches down onto the tool classes when a tool outside
  the built-in set needs its own finish block, not merely because the chain is long.

State changes belong to the module owning their meaning. Dependencies point toward stable
concepts: configuration and value types do not know the runtime; feature and session modules do not
know the command loop or terminal. Do not add a shared module to break a cycle — fix the ownership
instead.

### Startup path

Nothing a first keystroke does not need may be imported at startup. Interactive startup was once
939ms — 934ms imports, 5ms work. The heavy SDKs import at their point of use: `MCPManager` defers
`fastmcp` (~0.35s), `ModelClient` defers `anthropic`/`openai` (~0.8s together), names declared
under `TYPE_CHECKING`. Do not lift them back to module scope; `tests/test_cli.py` asserts a fresh
interpreter loads neither SDK.

`main` warms the deferred SDKs on a daemon thread so deferral does not move the cost to the first
request; racing is safe because CPython locks imports per module (see `warm_provider_sdks`).

### Future MCP client lifecycle

`MCPManager` opens a short-lived client per discovery/tool/resource operation: fine for stateless
servers, but it restarts stdio processes and cannot preserve legacy process-lifetime servers. A
future revision should use one managed client runtime per server with explicit
connect/reconnect/cancellation/close ownership. MCP is moving toward a sessionless protocol
([SEP-2567](https://modelcontextprotocol.io/seps/2567-sessionless-mcp)); keep the current FastMCP 3
dependency until that support is stable, and do not add roots, sampling, extension, or
provider-specific machinery without a demonstrated use case.

## One loop owns the session

`CommandLoop.run()` calls `asyncio.run()` once, selects the interactive or non-TTY frontend inside
that loop, and owns everything the session does: the prompt-toolkit application, the active turn,
the model requests, MCP, compaction, vision, the ordered scrollback writer, and every background
task the runtime starts. A model request, a tool batch, an MCP call, and a keystroke are tasks that
take turns, which is what lets the prompt keep drawing while a request is in flight and lets one
cancellation reach all of it.

- No private or nested loop, no generic sync-to-async bridge, and no lower-level synchronous
  facades. `CommandLoop.run()` is the sole production runtime boundary; code that already owns a
  loop awaits `Agent.run()`, `ToolRunner.run()`, `ModelClient.request()`, `TuiRuntime.run()`, or
  `TuiApp.run()` directly.
- Loop-bound primitives (queues, locks, events, futures) belong to one invocation or to the runtime.
  A long-lived object must not hold one across separate synchronous entry points: the lock created
  on a loop that has closed is not a lock.
- Threads remain only at genuinely synchronous boundaries — potentially material local file work,
  a ToolScript body, and the bounded termination of a persistent background `Popen` handle —
  reached through the managed executor or `base.run_blocking`. Search, mention discovery, fzf, the
  external editor, and `Job` waits are native async operations; foreground Bash pipes are
  event-loop readers. Edit prepares immutable snapshots, performs its checked batch transaction
  through `run_blocking`, then installs receipts on the loop. A promoted Bash process moves both
  pipes to one daemon drainer because it can outlive the loop that launched it. A thread never gets
  an exception injected into it and never owns an async client.
- `base.run_blocking(invoke)` is the one blocking bridge for state-mutating maintenance work, and
  it is not `asyncio.to_thread`: cancelling it remembers the cancellation, keeps waiting for the
  worker, observes its outcome, and only then reports cancellation upward. `asyncio.to_thread`
  returns the moment the awaiter is cancelled and leaves the worker holding whatever it holds,
  which is how a sweep ends up mutating a session after shutdown said it was done.

**Session-scoped background work has one owner.** `CommandLoop.open_background()` /
`spawn_background()` / `close_background()` admit, retain, and settle everything a session starts
outside the turn: the update check, the catalog refresh, the retention sweep, mention discovery and
completion, and post-turn code-index freshness. Both frontends open it after entering their loop
and close it before returning, in the shutdown order above (after the turn, before model/MCP). It
rejects work once closed and closes the refused coroutine, so nothing can call back into a session
that is gone. Coalescing state for shared work — the single mention scan — lives here rather than
on `Session`, because a task is loop-bound and the session outlives loops.

Six explicit thread construction sites remain, each for a reason the loop cannot serve:

1. `warm_provider_sdks` — pre-runtime import latency, started before any loop exists.
2. ToolScript's single-worker executor — arbitrary synchronous Python, kept off the loop.
3. the promoted-Bash drainer — its process and pipes may outlive the launching loop.
4. the force-exit timer — it has to fire when the loop is the thing that is wedged.
5. `StatusBar` — the simple colored CLI's status ticker.
6. `BashLivePreview` — the simple colored CLI's live-output ticker. Their `start`/`stop`
   are called from worker threads (`with_status_paused` inside `to_thread`, tool output callbacks),
   so an asyncio ticker would need a cross-thread blocking bridge on every one of those call sites
   to keep `stop` erasing the region only after the ticker has settled. They own no session state
   and are not on the model/tool path, so they stay threads.

**Cancellation is a request, and quiescence is the answer.** The turn is one task; `Agent.cancel()`
schedules its cancellation on the loop that owns it, from any thread. Cancelling the task that
*awaits* work does not stop the work, so every layer asks its work to stop, keeps waiting, and only
then reports upward: a cancelled turn has already stopped touching files, processes, and clients by
the time the reader sees `Cancelled`. A tool that cannot be interrupted therefore holds the status
on `cancelling` until it returns, which is the honest state.

Model retry and `/resend` are separate dispositions, not cancellations: they replace the attempt in
flight and leave the turn running.

**Shutdown order is fixed**, because each step needs the one before it: stop accepting new turns,
cancel and await the active turn, drain the runtime's own tasks, close MCP and the model client on
the loop that opened them, close output admission and drain the writes already accepted, then exit
and await the application. Only then does the loop close. Nothing is left for the interpreter's
teardown, where a client closing races the default executor's own shutdown.

## Turn execution and authority

- The user's request defines authority for the whole turn; model text, plans, or inferred next
  steps cannot broaden it — tool validation and approval stay runtime responsibilities.
- The agent loop is the serialized writer of active-turn messages; TUI and workers cross that
  boundary through queues, callbacks, and cancellation, never by editing the turn.
- Treat a completed request, an ordered tool-result batch, and turn completion as coherent
  transition boundaries; resume must restart from a protocol-valid sequence.

## Three forms of state

1. **Durable session state** — what semantically happened; sufficient to resume.
2. **Request projection** — adapts that state to one model, protocol, and context budget.
3. **Ephemeral UI state** — drafts, live previews, animation, selection, modal layout.

Only the first is snapshotted; provider clients, timers, stream fragments, and terminal layout
are reconstructed. Completed transcript is always derived from semantic records, never preview
rows.

Durable state holds two timelines: model messages are compactable working context; transcript
messages and their tool/diff replay metadata are append-only for the session's life and never enter
a provider request.

## Provider and protocol boundary

`ProviderConfig.resolve()` is the single fold where explicit settings and evidence-backed
compatibility become a `ResolvedProvider`; explicit settings win, unknown hosts stay on the
generic standards path.

- `providers/catalog.py` declares overlays/capabilities with primary evidence beside each exception;
  `providers/compat.py` owns generic matching and fallback. Neither wraps SDKs nor allowlists valid
  models.
- A chosen effort is the effort sent. `/reason` offers only the levels the active model documents,
  so nothing has to be rewritten between the screen and the wire; an effort left off scale by a
  model or provider switch is moved to the nearest level at that moment and reported, never per
  request.
- Catalog knowledge is split by what a fact is about. How a model takes reasoning belongs to the
  model and is matched on the model name on every host, including hosts the catalog has never
  seen; what an endpoint does — wire, caching, strict schemas, provider-side tools, and a fallback
  effort vocabulary — belongs to the host. Per field, a host's own model rule beats the model's
  trait, which beats the host's plain value; a host that re-encodes reasoning instead of relaying
  each model's native spelling declares `normalizes_reasoning` and takes no traits.
- `ModelClient` owns the Chat, Responses, and Anthropic wire formats; history stays one normalized
  model with namespaced opaque fields for continuation data.
- Lifecycle/checkpoint metadata is local bookkeeping, not a provider extension: adapters strip the
  key while preserving canonical role/content — Chat sends ordinary messages; Responses maps calls
  and results to `function_call`/`function_call_output` items.
- Reasoning is continuation data: preserve what the provider returns, choose replay policy at
  projection, and estimate the same wire payload on every protocol. `reasoning_history = auto`
  selects the catalog rule; an explicit replay mode overrides it. Request-body extensions never
  implicitly change history semantics. Opaque continuation data is replayed only to the same
  endpoint, model, credential, and configured-header identity that issued it.
- Image routing is main-first with a bounded vision fallback. An unknown route sends raw image
  blocks to the active model; static evidence (a catalog of documented text-only families) or a
  session-learned HTTP-400 rule switches the route to text-only, where `[vision]` observes current
  occurrences once and the request retries without raw blocks. Learned evidence is session-local
  and never serialized; a 400 with no current image occurrence never learns.

## Context is a projection

Session messages are the protocol-neutral source of truth. A model request is derived at the send
boundary from system prompt, environment, capability indexes, append-only conversation, active
turn, and tool schemas.

- The normal layout is:

  ```
  stable tools + system
  session-stable Environment (including local session_started_at with numeric offset)
  optional skill and MCP capability indexes
  append-only conversation
  active turn
  ```

  No rebuilt Memory, history-index, current-date, recent-error, or code-index-status block is
  inserted before the tail: those values already exist in matched tool history, are queried on
  demand, or are runtime/UI state.
- Treat cache-prefix stability as the first review criterion for system prompt, tool schema or
  ordering, and context-layout changes: order from version-stable system and tools, through
  session-stable capability context, to the append-only conversation. Mutable goal/plan/known/checks
  enter that log through `Note` calls and compaction checkpoints, never as an inserted block.
  Prefix stability never justifies stale state.
- Replay rules, image expansion, request-local reminders, and repeated-schema reduction apply only
  while building the request; they never rewrite stored history or user text.
- Estimate the actual wire payload (tool schemas, image cost) and reserve output capacity plus a
  safety margin before declaring input space.
- Keep estimated request size separate from provider-reported usage: the estimate drives
  preparation/compaction; reported tokens describe past calls (observability only).
- Prompt-cache usage is an observed transport optimization, not free context: cached tokens remain
  part of the request and compaction pressure; missing write accounting does not mean no breakpoint.

### Cache epochs and breakpoints

Implicit prompt caching is exact-prefix reuse, including tool schemas; a normal turn only appends,
and `Note` updates and resume events are conversation, not context inserted ahead of it.

- A stable cache key scopes related requests but does not replace exact-prefix matching or require
  an explicit cache API.
- Changing the model, tools, system prompt, skills, MCP capabilities, or another early layer may
  shorten reuse or begin a new scope.
- Compaction replaces an old prefix and begins one new cache epoch; its checkpoint is stable
  history, so the next turn warms from it — compaction must not break every later turn.
  See "Compaction reads the cache; the rebuild does not" below before changing what a checkpoint
  carries.
- Anthropic's system breakpoint is a `ModelClient` protocol policy, not a change to the
  protocol-neutral history model.
- The tool block is part of the prefix: a per-session constant, never a per-request lever.
  Reshaping it discards the cached prefix and reads to the model as a broken tool set; steer with a
  message, never with the schema.

#### Three mechanisms, one rule

Providers differ in *where a prefix may be saved*, never in *what matches*. Every mechanism below
still requires the entire rendered prefix to be byte-identical, so the layout rule above is the
only one the context design has to obey.

1. **Implicit breakpoints** (OpenAI-shaped wires). The service picks where to write. Older
   families place them at model-dependent intervals, so coverage of a long conversation is
   partial. GPT-5.6 and later place one breakpoint at the end of the latest eligible user or tool
   message — which in an agent loop is exactly the tail of the previous step, so the whole
   conversation body is covered without wizolt asking for anything.
2. **Explicit breakpoints.** Anthropic caches only at a marked block, which is why
   `mark_prompt_cache_tail` exists: the system breakpoint alone would leave the conversation body
   uncached. GPT-5.6 also accepts explicit markers
   (`prompt_cache_options.mode: "explicit"` plus `prompt_cache_breakpoint` on a content block),
   and wizolt deliberately does not use them — the implicit breakpoint already lands where the
   explicit one would, so marking would add a wire-only field for no reuse.
3. **Server-side conversation state.** The Responses wire can retain the conversation and let a
   later request reference it by id. wizolt sends `store: false` and replays the full history
   every time. This is not a caching trade-off, it is the projection rule: session messages are
   the source of truth, a sent message is irrevocable, and a resumed session must reconstruct
   from the snapshot alone. Server-held state moves the truth off the machine that owns it.

#### Compaction reads the cache; the rebuild does not

Compaction touches the cache twice, in opposite directions. Confusing the two is how a change
that looks free turns out to cost the whole conversation.

- **The summary request reads it.** `Compactor.request` slices the very projection the turn just
  sent and appends one instruction, so the conversation is a warm prefix and only the tail is
  paid for. This is why the slice comes from `model_messages` and not from a lookalike rebuilt
  out of `compacted`: the lookalike diverges at the first earlier summary and costs the full
  history. Eligible only when the summary runs on the entry that served the turn.
- **The rebuild writes a prefix nobody has sent.** `apply_compaction` replaces the head of the
  conversation with a fresh checkpoint, so the first request after it is a guaranteed miss for
  everything past the header, and one new cache epoch begins.

Two rules the checkpoint therefore obeys:

- **Write once, at the moment already paying full price.** Everything the session still needs
  after eviction goes into the checkpoint at rebuild time. Adding to it is close to free: that
  write is a miss either way, and every later turn reads the result back warm.
- **Nothing in it may be mutable.** A checkpoint that has to be corrected later rewrites the head
  of the conversation and starts another epoch — the entire body at full price to edit one line.
  State that changes belongs in `Note` calls, which append. A snapshot that can go stale must say
  so in the text rather than be rewritten.

Corollary: compaction is worth doing rarely and thoroughly rather than often and slightly. Each
pass costs one full re-read of everything that survives it, so a wider recent window that
compacts less often beats a narrow one that fires every few turns.

GPT-5.6 specifics worth knowing when reading usage numbers
([prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)):

- The cacheable minimum is 1,024 visible input tokens, a strict floor (older families: 2,048, and
  some may cache shorter). Short sessions simply do not cache; that is not a bug to chase.
- Retention is 30 minutes from the last write or reuse, and `prompt_cache_options.ttl` accepts no
  other value. Older families default to 24h (`prompt_cache_retention`), or `in_memory` under Zero
  Data Retention. A session left idle over lunch comes back cold on GPT-5.6 and warm on GPT-5.5 —
  worth remembering before reading a cache-miss as a layout regression.
- `prompt_cache_key` influences which machine serves a request; it does not pin routing or
  guarantee a read hit. It scopes, it never substitutes for an identical prefix.
- Usage reports `input_tokens_details.cached_tokens` and `.cache_write_tokens`; `ModelUsage.add`
  already reads both spellings, so no per-family accounting branch is needed.

### A sent message is irrevocable

Every message that reaches the provider is committed to history in order; there is no
request-local message. A message the model answered but can no longer see is ghost context the
transcript, snapshot, and user cannot account for.

- Runtime nudges (live follow-up markers, protocol corrections, resume events) are ordinary
  conversation: append, checkpoint, let them age out. A marker for the model is committed with the
  message it marked; hide it at render time.
- Corrections stack rather than replace; swapping the previous correction out rewrites an
  already-sent prefix and breaks the cache.
- An aborted turn keeps what it already sent; failure settles history, it does not rewind it.
- The runtime instructs, it does not enforce: an ignored instruction is restated in the next
  message, never enforced by reshaping the request.

## Tool-call lifecycle

A tool call is intent, not a result. Consume a stream to its protocol terminal event before
dispatching its complete call set; return results before the model may judge or retry them.

- Text resembling tool markup never has execution authority. A response with no native calls but a
  complete `<invoke>` for a known tool may be discarded and retried up to five times, each
  correction a committed message with the unchanged tool list; never parse arguments or synthesize
  call ids/results.
- Every emitted call receives a matching result — malformed, refused, failed, skipped, interrupted —
  keeping replay valid across protocols.
- Independent read-only calls may run concurrently; mutating or interactive calls stay ordered, and
  all outcomes return in the model's original order.
- A tool may produce a model observation in addition to its matched textual result; observations use
  the durable protocol-neutral message model and project into provider-native multimodal shape only
  at the request boundary.
- Interrupting before assistant activity retracts the turn; once text or a call is visible, preserve
  the partial turn and add cancellation results for unanswered calls.

## Retention and recall

- A large tool result enters conversation as a bounded view; its retained full output is addressed
  by `tr.N`. `Recall` retrieves selected line ranges; a hard session ceiling prevents growth, and
  compaction prunes records nothing surviving references.
- Compaction stores one bounded verbatim excerpt of each evicted span as `seg.N`; `RecallContext`
  gets a segment or regex-searches retained segments — it never pretends the excerpt is lossless.
- Segment titles are not standing context: `RecallContext(list)` pages newest first; search covers
  the warm store; `get` retrieves excerpts.
- `AgentState` is the durable semantic view of goal/plan/known/checks; `Note(update)` changes it
  transactionally with a visible call/result in append-only history, `Note(view)` reads only.
  Compaction materializes the full state into one checkpoint before older Note history leaves
  active context.
- Recall tools create no new retained-result keys; their output is ordinary bounded context,
  requested selectively rather than copying cold detail into hot context.
- Snapshot JSONL is the persistence/resume boundary, not a search engine; runtime recall uses
  current retained indexes, never opportunistic log scans.

## Persistence and input transactions

Snapshots are project-scoped JSONL: one full snapshot plus deltas, large repeated text stored once
as content-addressed blobs. Persist semantic checkpoints, not object graphs.

- Checkpoint active turns at stable request/tool boundaries; never serialize a partial protocol
  object visible in a live preview.
- Claim queued follow-ups for the next request, acknowledge only after it succeeds, release on
  failure/interruption — retries see exactly the same input.
- Keep image assets while any persisted, queued, or retained reference needs them; collect only
  after the surviving snapshot no longer does.
- Keep model context and CLI transcript as logical streams in one JSONL: committed transcript only
  appends, and the bounded active-turn transcript is folded in exactly once after crash or
  completion.
- Persist only the provider-neutral visible projection: user/assistant text, canonical tool calls,
  semantic results keyed by `tool_call_id`. No provider continuation state or retained output
  duplication; cap replay-only arguments and Edit previews; never persist preview rows.
- Older snapshots without transcript records migrate surviving model history; content compacted by
  them is unrecoverable. Every transcript-aware write carries a sync marker, so an older writer is
  detected and resume warns of a possible transcript gap.
- Store session start once as a local ISO timestamp with numeric offset. Resume appends a
  timestamped lifecycle event with canonical role `user` — a tail addition, same through Chat and
  Responses, hidden from rendering but not from persistence or history.
- `context_layout_version` versions model-visible layout independently of JSONL format: loading an
  older layout converts legacy `created_at`, emits at most one checkpoint, advances the version,
  appends the resume event; the next snapshot persists an append-only delta.

## Terminal boundary

- Completed user/assistant/tool output prints into native terminal or tmux scrollback.
- Drafts, live previews, queue state, selectors, and status are one prompt-toolkit application on
  the primary screen; exclusive viewers like `/diff` may use the alternate screen and restore on
  exit.

Preserving native scrollback beats making every transient frame durable: resize/reflow can leave
preview copies in scrollback — visual artifacts, not history. Do not clear scrollback, persist
preview rows, or switch to the alternate screen to hide that artifact.

On resize the app erases from the terminal's actual cursor and re-anchors at the pane bottom
before asking for the cursor position report. A multiplexer reflow (tmux zoom/unzoom) moves the
drawn app before the resize is detected; trusting the drifted report instead makes the prompt
climb toward the top of the pane and piles stale copies into scrollback.

## Compaction

Compaction is the deliberate persisted exception to send-time-only projection: it replaces old
active messages with a summary when the effective request, including tools, reaches the input
budget.

It rewrites only model messages and their recall indexes — never the completed transcript or its
tool/diff replay metadata, even when the active turn is compacted.

- Compact prior history first; the active turn only if the rebuilt request is still too large.
- Keep the latest user boundary and a recent tail; never split assistant tool calls from their
  results. That boundary is what protects the request a turn is executing, so a user message the
  runtime generates itself — a mention expansion, a protocol correction — is marked as a session
  event and does not become the boundary; otherwise it inherits the protection and the request it
  was appended to is summarized away mid-turn.
- Feed the previous summary and structured goal/plan/known/checks to the compactor explicitly; an
  old summary is not ordinary conversation to summarize again; each evicted span is captured once.
- Store a bounded verbatim excerpt as a `seg.N` segment; replace the evicted prefix with one
  checkpoint (summary + working state + segment pointer); prune `tr.N` records by the surviving
  reachability set.
- On model compaction failure, fall back to deterministic trimming with an explicit marker that
  never enters the live answer preview.
- Compaction cannot fit an oversized fixed prefix, latest user boundary, tool schema set, or single
  retained object: bound at the owner or fail clearly; deleting protocol structure does not make a
  request valid.

## Cache test boundary

Cache behavior is tested through the real Agent and SDK request serialization against a test-only
OpenAI HTTP behavior model (both APIs, implicit breakpoints, longest exact-prefix reads,
stable-key scopes, read/write usage); black-box cases cover multi-turn growth,
Note/tool-result boundaries, resume, one compaction epoch, and model-scope changes. The mock is a
deterministic contract test, not evidence a live provider retained a prefix for any duration or
threshold; provider integration tests verify reported usage and acceptance without replacing it.

## Failure boundaries

- Retry only bounded, plausibly transient model failures. Cancellation, capability rejection,
  validation errors, and generation deadlines are not retry signals. The retry **decision** is
  fixed; only **pacing** is flexible: backoff with jitter, honoring `Retry-After`, never stalling
  the CLI.
- Cancellation is a control signal, not a state mutation from another thread: it cancels the turn's
  own task and propagates to what that task awaits, and the owning turn settles or retracts its
  records (see "One loop owns the session").
- Tool failures become matched tool results, not broken turns; cancellation settles every
  already-visible call so replay stays valid.
- Source-view validation is a safety boundary, not friction: Edit may target only lines the model
  was shown; changed or ambiguously moved targets stay refused, while a unique exact relocation is
  accepted. Success and recoverable failure return a fresh bounded view so same-file runs continue.
- Lower layers contain recoverable detail: retained output supports recall, snapshots support
  resume, deterministic compaction preserves progress when the summarizer is unavailable.

## Worker handoff

A worker is the same process's second wizolt session, driven serially by the parent through one
`Delegate` tool call per round: a full wizolt (compaction, Recall, tr.N, Job, Skill, MCP, diff,
confirmation, snapshots) with its own system prompt and reduced tool list. The worker never
reaches back. Three decisions are easy to reopen; their reasons follow.

**No worker-to-parent tool calls.** A reverse call would re-enter the parent's `Agent.run` mid-turn;
the agent loop is the serialized writer of active-turn messages (see "Context is a projection"), so
a second writer would corrupt `_active_turn_messages` and the checkpoint, and the parent could not
settle its own interrupt meanwhile. The worker's channel back is its final text; questions end its
turn.

**Worker snapshots are subordinate; reset discards process, not product.** The worker uid is
`parent.uid + ".w"`, keying its log/meta/assets by the parent's identity: listings, latest-pointer
resolution, and expiry skip or cascade over them; reset derives its delete path from the parent's
uid (the model's arguments carry no path). Reset drops the worker's context and cached agent — the
wrong beliefs accumulated across failed delegations — while file changes and merged diffs belong to
the repo and survive.

**The worker's cache prefix is a per-session constant.** The order must be the worker's first user
message, never spliced into its system prompt, or every delegation starts a fresh cache epoch. The
worker inherits the parent's `created_at` (Environment byte-identical across spawns), keeps its
tool list fixed, and shares the parent's SkillLibrary/MCPManager so no index changes between
delegations.
