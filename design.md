# nanocode Design

## Agent Model

nanocode uses one agent. The agent is responsible for:

- understanding the user request
- maintaining goal, plan, hypotheses, and memory
- calling tools
- verifying work
- deciding when the task is complete

The agent has a work path and a cleanup path:

- `ACT`: normal work. It plans, investigates, edits, verifies, and answers.
- `OBSERVE`: tool-result reducer. It decides which recent raw results stay in context and which are compacted away.

Conversation compaction is a background maintenance path. It summarizes old conversation history when the conversation list grows too large.

## Task State

The main task state lives in the blackboard:

- latest user request
- current task code
- goal
- plan
- hypotheses
- known facts: settled facts for the current task
- stable knowledge: rare reusable codebase facts
- verification state
- recent edits

New user input keeps the previous task state available for follow-ups like "continue".

Old task state is cleared only when the model explicitly starts a different goal. When that happens, transient investigation state such as hypotheses and selected tool-result context is reset, while durable knowledge is kept.

## Context Construction

ACT mode receives a working context:

- goal, plan, hypotheses, verification
- kept tool results
- recent tool calls
- errors
- tool result store summary
- recent edits
- known and stable knowledge
- conversation history
- latest user request

OBSERVE receives a smaller cleanup context:

- latest user request
- goal, plan, hypotheses
- known and stable knowledge
- kept tool results
- observe errors
- latest raw tool results waiting for cleanup

OBSERVE reduces tool-result noise before ACT continues.

## Tool Result Storage

Every tool call gets a result key such as `tr.12`.

The full tool output is written to the session log directory. The model sees bounded output in context and can fetch full output later with `Recall tr.N`.

This separates storage from context:

- logs keep the full result
- context keeps only what is useful right now
- `Recall` restores detail on demand

## Tool Result Context

Tool-result context has three conceptual layers:

- `latest`: raw bounded output from the most recent tool batch
- `kept_results`: useful raw results selected by OBSERVE and retained for ACT
- `recent`: older visible results, usually compact summaries

Recent raw results that have not been reduced yet remain visible in `latest` or `recent` until OBSERVE covers them.

From the model's view:

1. every tool result gets a `tr.N` key and full log entry
2. ACT sees bounded raw output for `latest` and unreduced `recent` results
3. ACT also sees `kept_results`, which are raw bounded results selected by OBSERVE
4. OBSERVE sees unreduced raw results from `latest` and `recent`
5. OBSERVE must `keep` useful results or `forget` noisy ones
6. forgotten results leave active context, but full logs remain available through `Recall tr.N`

After each tool batch:

1. the previous `latest` moves into `recent`
2. the new batch becomes `latest`
3. unreduced raw results remain visible to ACT
4. OBSERVE later converts unreduced raw results into kept results, summaries, or forgotten context

This keeps tool results visible until the model has had a chance to decide whether they matter.

## Observe Policy

OBSERVE is triggered when unresolved pending results accumulate, or when a meaningful tool failure needs cleanup.

In OBSERVE, every unreduced result key must be covered by either:

- `keep`: retain this raw result in `kept_results`
- `forget`: remove this result from future active context

`forget` releases context pressure while preserving logs and Recall ability.

If a forgotten result contained an important conclusion, the model should preserve that conclusion first in plan, known, hypothesis, or verification state.

## Context Budgets

Context is bounded at several layers:

- tool output is bounded before it enters context
- recent tool summaries have count and character budgets
- kept results have their own character budget
- conversation history can be compacted
- old stored tool results are pruned unless protected by active state

The design favors keeping useful raw tool results visible, while aggressively compacting or forgetting noise.

## Completion and Verification

The agent should complete only when:

- the goal is achieved
- plan items are done or blocked with concrete context
- verification strength matches the task risk
- required verification has passed or is blocked by the user/environment/tool

Verification is ACT work using tool calls plus a `verify` state update.

Verification strength is intentionally lightweight:

- `none`: simple chat, explanation, or documentation-only answer
- `light`: static/read confirmation
- `tool`: test, lint, build, search, or executable check
- `user`: visual/manual confirmation

## Design Principle

The core idea is:

Keep full data outside context, keep useful evidence inside context, and let OBSERVE periodically remove noise.
