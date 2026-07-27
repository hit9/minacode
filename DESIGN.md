# Design notes

This file records decisions whose rationale is easy to lose and costly to rediscover. Keep it
short: document durable conclusions, not implementation diaries or complete investigation logs.

## Maintenance

- Docstrings describe interfaces and contracts, not development history.
- Comments protect local, non-obvious invariants and may link to primary evidence.
- Add a note here only for a cross-cutting decision that future maintainers may otherwise reopen.
  If a decision changes, keep the old conclusion visible and mark it as superseded.

### Engineering posture

- Prefer a generic standards path. Specialize only for a necessary, documented incompatibility,
  and keep primary evidence beside the rule.
- Keep abstractions deep, local, and earned. Remove dead code and pass-through wrappers; keep a
  private helper with its sole owner instead of making it global or adding another module layer.
- Favor the smallest behavior-preserving change. Separate unrelated concerns into cohesive commits
  and keep the changelog aligned with user-visible behavior.
- Prefer explicit imports and pragmatic typing: model domain shapes precisely, keep JSON as
  `dict[str, Any]`, and suppress a type error only when runtime behavior is demonstrably safe.
- Test contracts and reproduced regressions at their boundaries, not private spelling or prompt
  literals. CI enforces formatting, lint, typing, and the full suite.
- Keep UI and user documentation quiet and direct. Show truthful state with progressive detail;
  keep compatibility machinery and investigation history out of the common user path.

## System shape

minacode is one local process with explicit owners for each kind of behavior:

- `base.py` defines configuration, shared value types, and error categories;
  `provider_compat.py` folds documented compatibility into resolved request policy.
- `Session` owns protocol-neutral semantic state: messages, active-turn checkpoints, queued input,
  retained output, diffs, usage, and session-scoped resources such as jobs and images. Its snapshot
  codec decides which of that state is persistable.
- `engine.py` owns agent semantics: context construction, model protocol adapters, compaction, tool
  execution, cancellation, and turn commit or rollback.
- `CommandLoop` and `TuiRuntime` orchestrate commands and runtime transitions. `TuiApp` owns input,
  key bindings, layout, and modals; `render.py` owns transcript and status presentation.
- `tools.py`, `image.py`, `mcp.py`, and `skill.py` are vertical feature modules. They expose useful
  behavior to the engine without making the engine understand their storage or UI details.

State changes belong to the module that owns their meaning. Higher layers may request a transition
or observe it through callbacks, but rendered text and widget state are never the source of truth.

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

- Apply replay rules, image expansion, request-local reminders, and repeated-schema reduction only
  while building the request. These transforms must not rewrite stored history or user text.
- Estimate the payload that will actually cross the selected protocol boundary, including tool
  schemas and image cost. Reserve output capacity and a safety margin before declaring input space
  available.
- Prompt-cache usage is an observed transport optimization, not free context. Cached tokens remain
  part of the request and compaction pressure.

## Tool-call lifecycle

A tool call is intent, not a result. Consume a stream to its protocol terminal event before
dispatching its complete call set, then return results before the model may judge or retry them.

- Every emitted call receives a matching result, including malformed, refused, failed, skipped,
  and interrupted calls. This keeps replay valid across protocols.
- Independent read-only calls may run concurrently. Mutating or interactive calls remain ordered;
  all outcomes are displayed, stored, and returned in the model's original order.
- Model-facing output is bounded while full storable output is retained under `tr.N`; `Recall`
  restores detail on demand.
- Interrupting before assistant activity retracts the turn. Once text or a tool call is visible,
  preserve the partial turn and add cancellation results for unanswered calls.

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

## Compaction

Compaction is the deliberate persisted exception to send-time-only projection: it replaces old
active messages with a summary when the effective request, including tools, reaches the input
budget.

- Compact prior history first and the active turn only if the rebuilt request remains too large.
- Keep the latest user boundary and a recent tail. Never split assistant tool calls from their
  following results.
- Store removed text as a `seg.N` history segment for `RecallContext`, and prune `tr.N` records only
  after surviving state no longer references them.
- If model-generated compaction fails, fall back to deterministic trimming with an explicit marker.
  Compaction must remain a recovery path, and its output never enters the live answer preview.

## Failure boundaries

- Retry only bounded, plausibly transient model failures. User cancellation, explicit capability
  rejection, validation errors, and total-generation deadlines are not automatic retry signals.
- Tool failures become matched tool results rather than broken turns. Cancellation settles every
  already-visible call so later protocol replay remains valid.
- Lower layers contain recoverable detail: retained output supports recall, snapshots support
  resume, and deterministic compaction preserves progress when the summarizer is unavailable.
