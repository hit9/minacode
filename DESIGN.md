# Design notes

This file records decisions whose rationale is easy to lose and costly to rediscover. Keep it
short: document durable conclusions, not implementation diaries or complete investigation logs.

## Maintenance

- Docstrings describe interfaces and contracts, not development history.
- Comments protect local, non-obvious invariants and may link to primary evidence.
- Add a note here only for a cross-cutting decision that future maintainers may otherwise reopen.
  If a decision changes, keep the old conclusion visible and mark it as superseded.

## Context is a projection

Session messages are the protocol-neutral source of truth. A model request is a derived view built
at the request boundary from the system prompt, environment, capability indexes, retained history,
memory, the active turn, and tool schemas.

- Apply provider replay rules, image expansion, and repeated-schema reduction only while building
  the request. Token-saving transforms must not rewrite stored history.
- Estimate the payload that will actually cross the selected protocol boundary, including tool
  schemas and image cost. Reserve output capacity and a safety margin before declaring input space
  available.
- Prompt-cache usage is an observed transport optimization, not free context. Cached tokens remain
  part of the request and must not be subtracted from compaction pressure.

## Tool-call lifecycle

A tool call is intent, not a result. Chat, Responses, and Anthropic streams are consumed to their
protocol terminal event before the agent dispatches the complete call set. A model must then wait
for the returned tool messages before judging or retrying those calls.

- Every emitted call must receive a matching result, including malformed, refused, failed, skipped,
  and interrupted calls. This keeps replay valid across protocols.
- Independent read-only calls may execute concurrently. Mutating or interactive calls remain
  ordered; parallel outcomes are displayed, stored, and returned in the model's original order.
- Storable output is retained under a `tr.N` key while the model-facing result is bounded. `Recall`
  restores retained detail on demand, so large output does not need to occupy every later request.
- Follow-ups are claimed transactionally for the next model request and acknowledged only after
  that request succeeds. A retry sees the same input; a failure or interrupt releases it.
- Interrupting before any assistant activity retracts the turn. Once text or a tool call is visible,
  the partial turn is preserved and unanswered calls receive cancellation results.

## Compaction

Compaction is the deliberate, persisted exception to send-time-only projection: it replaces old
active messages with a summary when the estimated request, including tools, reaches the input
budget.

- Compact prior history first. Compact the active turn only if the rebuilt request is still too
  large.
- Keep the latest user boundary and a recent tail. Never split an assistant tool-call message from
  its following tool results.
- Replace compacted messages with one structured summary and store the removed excerpt as a
  `seg.N` history segment for `RecallContext`.
- If model-generated compaction fails, fall back to deterministic trimming and leave an explicit
  marker. Compaction must remain a recovery path rather than a new failure mode.
- Prune retained `tr.N` records only after the surviving summary and messages no longer reference
  them. Manual compaction is persisted immediately, and compactor output never enters the live
  answer preview.

## Reasoning history and context accounting

**Status:** Current

**Introduced:** 2026-07-26, commit `cda2a81`

### Context

Provider reasoning is not one uniform kind of text. It may be readable output, structured or
encrypted continuation state, or state required to complete a tool-call turn. Always replaying it
wastes input tokens and context on providers that ignore old reasoning; never replaying it breaks
continuation for providers that require it.

### Decision

- Preserve all reasoning returned by a provider in session history. Do not mutate stored history
  to optimize one protocol.
- Apply replay policy while projecting normalized history into Chat, Responses, or Anthropic wire
  input.
- Keep reasoning required by an active tool loop. Preserve earlier turns only when the provider
  requires it or the user explicitly enables preserved thinking.
- Keep unknown Chat providers conservative and replay their reasoning rather than guessing that it
  is disposable.
- Estimate context from the same effective protocol input. Opaque signatures, ciphertext, and
  image base64 are transport data rather than text-token estimates; image tokens are added using
  the image model instead.
- Cached input still occupies the logical context window, so cached tokens are never subtracted
  from the compaction estimate.

### Documented differences

- DeepSeek requires reasoning attached to tool calls in later requests; ordinary final-answer
  reasoning may be omitted.
- Qwen ignores historical reasoning by default but needs it during multi-step tool calls;
  `preserve_thinking=true` opts into cross-turn replay and billing.
- Kimi K3 and K2.7 preserve all thinking. K2.6 defaults to the active tool turn unless
  `thinking.keep="all"` is configured.
- Z.AI clears historical thinking by default; `thinking.clear_thinking=false` enables preserved
  thinking.
- OpenRouter may return `reasoning`, `reasoning_content`, or structured `reasoning_details`; the
  returned structure must be replayed without reconstruction.
- Anthropic thinking blocks and signatures are replayed unchanged. Anthropic decides per model
  whether all prior thinking or only the latest turn occupies effective context.
- Stateless OpenAI Responses requests replay the returned reasoning output items, including
  encrypted items.

### Evidence

- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Qwen thinking mode](https://platform.qianwenai.com/docs/developer-guides/text-generation/thinking)
- [Kimi thinking models](https://platform.kimi.com/docs/guide/use-thinking-models)
- [Z.AI thinking mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [OpenRouter reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Anthropic thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)

### Code

- Compatibility policy: `minacode/provider_compat.py`
- Context projection, compaction, agent loop, and tool execution: `minacode/engine.py`
- Queue transactions and retained tool results: `minacode/session.py`
- Tool contracts and implementations: `minacode/tools.py`
- Regression coverage: `tests/test_agent_logic.py`, `tests/test_core_logic.py`,
  `tests/test_model_client.py`
