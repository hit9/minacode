# Design notes

This file records decisions whose rationale is easy to lose and costly to rediscover. Keep it
short: document durable conclusions, not implementation diaries or complete investigation logs.

## Orientation

minacode turns one user request into a bounded loop of model calls and tool calls, in one local
process. Four objectives explain most of the decisions below, and they are frequently in tension:

1. **Resumable.** A session survives a crash, an interrupt, or a quit at any point.
2. **Protocol-neutral.** History is stored in one model; Chat, Responses, and Anthropic formats
   exist only at the send boundary.
3. **Bounded.** Context, retained output, and previews all have ceilings; nothing grows forever.
4. **Truthful.** The screen reports real state, and the terminal's own scrollback stays intact.

Modules, with dependencies pointing downward only:

```
              __main__                     entry, startup ordering
                  |
    loop.py -- tui.py -- render.py         commands, interaction, presentation
        |
    engine.py                              the turn loop: commit or roll back
        |
        +-- context.py                     request projection, compaction
        +-- runner.py                      tool batch execution
                  |
              model.py                     wire protocols, streaming, retry
                  |
   tools/  image.py  mcp.py  skill.py      vertical features
                  |
             session.py                    durable semantic state
                  |
   base.py   provider_compat.py            value types, config, policy
```

A turn, and the three ways it can end:

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
  |   checkpoint after every response and every batch            |
  +--------------------------------------------------------------+
      |                    |                      |
   commit               interrupt               error
      v                    v                      v
  append to           settle partial turn,   flush partial turn,
  session.messages    add interrupt marker   re-raise
```

The turn is a transaction: messages accumulate outside durable history until one of those three
endings. That is what makes resume safe, and why nothing else may append mid-turn.

## Common pitfalls

Each of these looks like a cleanup or a small improvement, and each breaks something the code
depends on. The section naming the rule is in parentheses.

- **Lifting a deferred import to module scope.** Startup latency is a feature; the SDKs cost ~0.8s
  and are not needed until the first request (Startup path).
- **Rewriting stored history in a request transform.** Replay rules, image expansion, and schema
  dedup are send-time only; a resumed session must equal the saved one (Context is a projection).
- **Returning fewer tool results than the model emitted calls.** Refused, failed, skipped, and
  interrupted calls each still need a matching result, or replay is invalid (Tool-call lifecycle).
- **Inserting context between the stable layers.** Saving a few tokens mid-prompt invalidates the
  cached prefix for every later turn (Context is a projection).
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
- Add a note here only for a cross-cutting decision that future maintainers may otherwise reopen.
  If a decision changes, keep the old conclusion visible and mark it as superseded.

### Engineering posture

- **Abstract:** Use deep, local, earned abstractions that hide real complexity behind small
  interfaces; remove dead code and pass-through wrappers, and keep a private helper with its owner.
- **Layer:** Keep dependencies directed from higher-level orchestration toward stable lower-level
  concepts; lower modules never import presentation or orchestration layers.
- **Keep it simple:** Choose the smallest cohesive, behavior-preserving implementation, avoid
  speculative specialization, separate unrelated changes, and keep the changelog aligned.
- Prefer a generic standards path. Specialize only for a necessary, documented incompatibility,
  and keep primary evidence beside the rule.
- Prefer explicit imports and pragmatic typing: model domain shapes precisely, keep JSON as
  `dict[str, Any]`, and suppress a type error only when runtime behavior is demonstrably safe.
- Test contracts and reproduced regressions at their boundaries, not private spelling or prompt
  literals. CI enforces formatting, lint, typing, and the full suite.
- Keep UI and user documentation quiet and direct. Show truthful state with progressive detail;
  keep compatibility machinery and investigation history out of the common user path.

### Test design

Tests protect observable contracts and reproduced regressions, not implementation shape.

- Prefer black-box tests through the narrowest stable public boundary that observes the complete
  behavior. Use white-box tests only when a pure algorithm or difficult edge condition cannot be
  exercised clearly there.
- A bug fix should reproduce the real failure, then cover the intended result and any unsafe path
  that must remain rejected.
- Assert semantic output, durable state, or protocol payloads. Assert exact text, call order, or
  rendering only when those details are themselves the contract.
- Mock external or nondeterministic boundaries such as providers, clocks, processes, and terminals;
  do not mock the core behavior under test.
- Keep tests deterministic and fast. Replace sleeps with events or controlled time, and reserve
  PTY or tmux tests for behavior that truly depends on a terminal.

## System shape

One local process with explicit owners for each kind of behavior. The layers are drawn in
[Orientation](#orientation); this section records what each owner is responsible for.

- `base.py` defines configuration, shared value types, and error categories, including the log-line
  vocabulary every presentation layer renders and the resource handles they cancel through;
  `provider_compat.py` folds documented compatibility into resolved request policy.
- `Session` owns protocol-neutral semantic state: messages, active-turn checkpoints, queued input,
  retained output, diffs, usage, and session-scoped resources such as jobs and images. Its snapshot
  codec decides which of that state is persistable.
- Agent semantics are split by owner: `context.py` builds and compacts the model-facing context,
  `model.py` owns provider protocol adapters, streaming, and retry policy, `runner.py` owns tool
  execution and cancellation, and `engine.py` composes them into the turn loop with its commit or
  rollback. They depend downward only: `engine.py` -> `context.py`/`runner.py` -> `model.py` ->
  `base.py`, so no pair needs a deferred import to break a cycle.
- `CommandLoop` and `TuiRuntime` orchestrate commands and runtime transitions. `TuiApp` owns input,
  key bindings, layout, and modals; `render.py` owns transcript and status presentation.
- `tools/`, `image.py`, `mcp.py`, and `skill.py` are vertical feature modules. They expose useful
  behavior to the engine without making the engine understand their storage or UI details. Inside
  `tools/`, each module owns one capability (files, search, shell, memory, ask, plugin) over the
  shared `Tool` base; `__init__.py` owns the registry and is the package's only import surface.

State changes belong to the module that owns their meaning. Higher layers may request a transition
or observe it through callbacks, but rendered text and widget state are never the source of truth.
Dependencies point toward stable concepts: configuration and value types do not know the runtime;
feature and session modules do not know the command loop or terminal; orchestration composes them at
the boundary. Do not introduce a shared module merely to break a cycle—fix the ownership instead.

### Startup path

Nothing a first keystroke does not need may be imported at startup. Interactive startup was once
939ms, of which 934ms was imports and 5ms was work; the prompt could not echo a character until it
finished. The heavy third-party SDKs are therefore imported at their point of use, not at module
scope: `MCPManager` defers `fastmcp` (~0.35s) and `ModelClient` defers `anthropic` and `openai`
(~0.8s together), each declaring the names under `TYPE_CHECKING` so annotations and type checking
stay complete. Do not lift these back to module scope for tidiness; that is the regression, and
`tests/test_cli.py` asserts a fresh interpreter loads neither SDK.

`main` then warms the deferred SDKs on a daemon thread so the deferral does not simply move the
cost onto the first request. Racing that thread is safe because CPython locks imports per module;
the reasoning and its limits are recorded on `warm_provider_sdks`, beside the code that depends on
them.

### Future MCP client lifecycle

`MCPManager` currently opens a short-lived client for each discovery, tool, or resource operation.
This is sufficient for stateless servers, but repeatedly starts stdio processes, prevents transport
reuse, and cannot preserve legacy servers that rely on process-lifetime state. A future MCP revision
should evaluate one managed client runtime per configured server, with explicit connect, reconnect,
cancellation, and close ownership.

That runtime would reuse transport resources without treating an MCP connection as durable semantic
state. MCP is moving toward a sessionless protocol with explicit state handles
([SEP-2567](https://modelcontextprotocol.io/seps/2567-sessionless-mcp)); protocol negotiation and
server-specific compatibility remain the client library's responsibility. Keep the current FastMCP
3 dependency until the modern protocol support is stable, and do not add roots, sampling, extension,
or provider-specific machinery without a demonstrated minacode use case.

## Turn execution and authority

One agent turn is a bounded state machine; see [Orientation](#orientation) for its shape.

- The user's request defines authority for the entire turn. A model may propose work, but model text,
  a plan, or an inferred next step cannot broaden that authority; tool validation and approval remain
  runtime responsibilities.
- The agent loop is the serialized writer of active-turn messages. TUI and background workers cross
  that boundary through queues, callbacks, and cancellation signals rather than editing the turn.
- Treat a completed request, an ordered tool-result batch, and turn completion as coherent transition
  boundaries. UI progress may lead them, but resume must restart from a protocol-valid sequence.

## Three forms of state

Keep these forms separate even when they contain similar data:

1. **Durable session state** records what semantically happened and is sufficient to resume.
2. **Request projection** adapts that state to one model, protocol, and context budget.
3. **Ephemeral UI state** covers drafts, live previews, animation, selection, and modal layout.

Only the first form is snapshotted. Provider clients, timers, stream fragments, and terminal layout
are reconstructed. A live preview may disappear without changing history; completed transcript is
always derived from semantic records rather than preview rows.

## Provider and protocol boundary

Configuration expresses user intent. `ProviderConfig.resolve()` is the single fold where explicit
settings and evidence-backed compatibility become a `ResolvedProvider`; explicit settings win and
unknown hosts stay on the generic standards path.

- Compatibility profiles describe policy differences, not provider SDK wrappers. Keep host and
  model checks there, with primary evidence beside the rule; do not scatter them through tools,
  commands, or rendering code.
- `ModelClient` owns the Chat, Responses, and Anthropic wire formats. Session history remains one
  normalized model with namespaced opaque fields for protocol continuation data.
- Reasoning is continuation data, not one universal text field. Preserve what the provider returns,
  choose replay policy while projecting a request, and estimate the same effective wire payload.
- Capability discovery must be conservative and session-local. A successful image request can
  establish support; only an explicit modality rejection can establish non-support.

## Context is a projection

Session messages are the protocol-neutral source of truth. A model request is derived at the send
boundary from the system prompt, environment, capability indexes, retained history, memory, the
active turn, and tool schemas.

- Treat cache-prefix stability as the first review criterion for every system prompt, tool schema or
  ordering, and context-layout change. Order requests from version-stable system and tools, through
  session-stable capability context and append-only conversation, to volatile task memory and the
  active turn at the tail. Prefer trigger-local tail additions over conditionally rewriting an
  earlier layer; saving a small number of tokens does not justify invalidating a larger reusable
  prefix. Prefix stability never justifies putting stale state into durable history.
- Apply replay rules, image expansion, request-local reminders, and repeated-schema reduction only
  while building the request. These transforms must not rewrite stored history or user text.
- Estimate the payload that will actually cross the selected protocol boundary, including tool
  schemas and image cost. Reserve output capacity and a safety margin before declaring input space
  available.
- Keep estimated request size separate from provider-reported usage. The estimate drives preparation
  and compaction for the next request; reported prompt, completion, and cached tokens describe calls
  that already happened and are observability data.
- Prompt-cache usage is an observed transport optimization, not free context. Cached tokens remain
  part of the request and compaction pressure.

## Tool-call lifecycle

A tool call is intent, not a result. Consume a stream to its protocol terminal event before
dispatching its complete call set, then return results before the model may judge or retry them.

- Text that resembles tool markup never has execution authority. When a response has no native
  calls but ends with a complete `<invoke>` for a known tool, the agent may discard it and retry
  up to five times with request-local corrections. Never parse its arguments or synthesize a call
  id or result.
- Every emitted call receives a matching result, including malformed, refused, failed, skipped,
  and interrupted calls. This keeps replay valid across protocols.
- Independent read-only calls may run concurrently. Mutating or interactive calls remain ordered;
  all outcomes are displayed, stored, and returned in the model's original order.
- A tool may produce a model observation in addition to its matched textual result. Observations
  follow every result in the proposed batch, use the durable protocol-neutral message model, and
  are projected into each provider's native multimodal shape only at the request boundary.
- Interrupting before assistant activity retracts the turn. Once text or a tool call is visible,
  preserve the partial turn and add cancellation results for unanswered calls.

## Retention and recall

Bounded active context and recoverable detail are separate concerns:

- A large tool result enters the conversation as a bounded view while its retained full output is
  addressed by `tr.N`. `Recall` can retrieve selected line ranges; a hard session ceiling prevents
  indefinite growth, and compaction prunes records no surviving message or summary references.
- Compaction stores one bounded verbatim excerpt of each evicted span as `seg.N`. `RecallContext`
  retrieves a segment or regex-searches all retained segments; it does not pretend the excerpt is a
  lossless copy of arbitrarily large conversation history.
- The active history index contains only bounded `seg.N` titles. Its truncation limits standing
  context, while search still covers the warm segment store, so omitted middle titles remain
  discoverable.
- Recall tools do not create new retained-result keys. Their output is ordinary, bounded turn
  context and should be requested selectively instead of recursively copying cold detail into hot
  context.
- Snapshot JSONL is the persistence and resume boundary, not a model-facing search engine. Runtime
  recall uses the current retained indexes and never scans historical log records opportunistically.

## Persistence and input transactions

Snapshots are project-scoped JSONL: one full snapshot followed by deltas, with large repeated text
stored once as content-addressed blobs. Persist semantic checkpoints, not object graphs.

- Checkpoint active turns at stable request and tool boundaries; never serialize a partial protocol
  object merely because it is visible in a live preview.
- Claim queued follow-ups for the next request, acknowledge them only after that request succeeds,
  and release them on failure or interruption. Retries therefore see exactly the same input.
- Keep image assets while any persisted, queued, or retained reference needs them; garbage collect
  only after the surviving snapshot no longer does.
- Reconstruct transcript and UI state from semantic records on resume. Never persist live preview
  rows as conversation messages.

## Terminal boundary

The terminal has two deliberately different output paths:

- Completed user, assistant, and tool output is printed into native terminal or tmux scrollback.
- Drafts, live model/tool previews, queue state, selectors, and status are one prompt-toolkit
  application on the primary screen. Exclusive viewers such as `/diff` may temporarily use the
  alternate screen and restore the transcript on exit.

Preserving native scrollback is more important than making every transient frame durable. Terminal
resize and reflow can leave copies of a live preview in scrollback; those copies are visual artifacts,
not session history. Do not clear scrollback, persist preview rows, or move the whole application to
the alternate screen to hide that artifact—the cure would discard more valuable behavior.

## Compaction

Compaction is the deliberate persisted exception to send-time-only projection: it replaces old
active messages with a summary when the effective request, including tools, reaches the input
budget.

- Compact prior history first and the active turn only if the rebuilt request remains too large.
- Keep the latest user boundary and a recent tail. Never split assistant tool calls from their
  following results.
- Feed the previous summary and structured goal, plan, known facts, and checks to the compactor
  explicitly. Do not treat an old summary as ordinary conversation to summarize again; each newly
  evicted message span is captured once before it leaves the active history.
- Store a bounded verbatim excerpt as a `seg.N` history segment for `RecallContext`, and use surviving
  messages and summaries as the reachability set when compaction prunes `tr.N` records.
- If model-generated compaction fails, fall back to deterministic trimming with an explicit marker.
  Compaction must remain a recovery path, and its output never enters the live answer preview.
- Compaction cannot make an oversized fixed prefix, latest user boundary, tool schema set, or single
  retained object fit. Bound such sources at their owner or fail clearly; never claim that deleting
  protocol structure made a request valid.

## Failure boundaries

- Retry only bounded, plausibly transient model failures. User cancellation, explicit capability
  rejection, validation errors, and total-generation deadlines are not automatic retry signals.
- Cancellation is a control signal, not a state mutation from another thread. Fan it out to the
  active model and tool resources, then let the owning turn settle or retract its semantic records.
- Tool failures become matched tool results rather than broken turns. Cancellation settles every
  already-visible call so later protocol replay remains valid.
- Lower layers contain recoverable detail: retained output supports recall, snapshots support
  resume, and deterministic compaction preserves progress when the summarizer is unavailable.
