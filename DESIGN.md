# Design notes

This file records decisions whose rationale is easy to lose and costly to rediscover. Keep it
short: document durable conclusions, not implementation diaries or complete investigation logs.

## Maintenance

- Docstrings describe interfaces and contracts, not development history.
- Comments protect local, non-obvious invariants and may link to primary evidence.
- Add a note here only for a cross-cutting decision that future maintainers may otherwise reopen.
  If a decision changes, keep the old conclusion visible and mark it as superseded.

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
- Protocol projection and context estimation: `minacode/engine.py`
- Regression coverage: `tests/test_core_logic.py`, `tests/test_model_client.py`
