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
- `OBSERVE`: tool-result reducer. It decides which unreduced raw tool results stay in context and which are compacted away.

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

## New Goal Handling

New user input does not immediately clear the previous task. This keeps short
follow-ups such as "continue" usable.

When the model outputs `start` with a different goal:

- goal and plan are replaced
- hypotheses are cleared
- verification is reset
- kept tool results are cleared
- visible raw tool results are compacted into summaries
- full tool logs remain available through `Recall tr.N`
- known and stable knowledge remain available

## Context Construction

ACT mode receives a working context:

- goal, plan, hypotheses, verification
- Tool Result Index
- Kept Tool Results
- Unreduced Tool Results
- Latest Tool Results
- errors
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
- unreduced raw tool results selected from recent/latest storage

OBSERVE reduces tool-result noise before ACT continues.

Context layout:

Layout rules:

- Lower means closer to `YOUR OUTPUT`.
- Put stable background and lookup-only indexes higher.
- Put newer, authoritative, decision-driving context lower.
- Keep large evidence blocks above the final decision area.
- Apply the same ordering inside each section.

```text
ACT user prompt, top -> bottom
+--------------------------------------------------+------------------------------+
| Context section                                  | Budget / control             |
+--------------------------------------------------+------------------------------+
| Background                                       | compact_at                   |
|   - Environment                                  |                              |
|   - Stable Knowledge                             |                              |
|   - User Rules                                   |                              |
|   - Conversation History                         |                              |
+--------------------------------------------------+------------------------------+
| Tool Result Index                                | TOOL_RESULT_INDEX_ITEMS      |
|   - Archived Recall Index                        |                              |
|   - Current Task Timeline                        |                              |
+--------------------------------------------------+------------------------------+
| Kept Tool Results                                | KEPT_TOOL_RESULT_CHARS       |
|   - kept_results                                 |                              |
+--------------------------------------------------+------------------------------+
| Unreduced Tool Results                           | TOOL_RESULT_RAW_CHARS trigger|
|   - unreduced recent                             | OBSERVE_AFTER_PENDING...     |
+--------------------------------------------------+------------------------------+
| Latest Tool Results                              | TOOL_RESULT_RAW_CHARS trigger|
|   - latest                                       | MAX_TOOL_OUTPUT_CHARS/item   |
+--------------------------------------------------+------------------------------+
| Current Decision                                 | section-local limits         |
|   - Recent Edits                                 |                              |
|   - Known                                        |                              |
|   - Task Code / Work Mode                        |                              |
|   - Goal / Plan / Hypotheses / Verify            |                              |
|   - Errors                                       |                              |
|   - Latest User Request                          |                              |
|   - Output Instructions                          |                              |
+--------------------------------------------------+------------------------------+
```

Bounded raw output means the original tool output after per-result truncation.
Compact summaries keep only execution metadata, size, and `recall=tr.N`.

Raw tool result content is de-duplicated by `tr.N`. Timeline summaries may keep
duplicate keys as compact index entries, especially for kept results, so the
model can still see result ordering without rereading raw content.

Tool result context budgets:

- `MAX_TOOL_OUTPUT_CHARS` bounds each raw tool result before it enters context.
- `KEPT_TOOL_RESULT_CHARS` limits `Kept Tool Results`.
- `TOOL_RESULT_RAW_CHARS` triggers OBSERVE when `Unreduced Tool Results + Latest Tool Results` grow too large. It is not a pre-observe truncation limit.
- `TOOL_RESULT_INDEX_ITEMS` limits compact index/timeline entries; current-task timeline entries take priority over archived entries.

## Tool Result Context

Internal tool-result storage has three fields:

- `latest`: raw bounded output from the most recent tool batch
- `kept_results`: useful raw results selected by OBSERVE and retained for ACT
- `recent`: older visible results, usually compact summaries

Prompt layout renders those fields as Tool Result Index, Kept Tool Results,
Unreduced Tool Results, and Latest Tool Results. Recent raw results that have
not been reduced yet remain visible as Unreduced Tool Results until OBSERVE
covers them.

ACT should render tool context in this order:

1. Tool Result Index:
   - archived recallable summaries, separated from the current task timeline
   - current task timeline summaries
2. Kept Tool Results: kept raw results
3. Unreduced Tool Results: unreduced older raw results
4. Latest Tool Results: latest raw results

This keeps the newest and most actionable tool output closest to the model's
next decision while preserving a compact timeline above it.

## Tool Result Storage

Every tool call gets a result key such as `tr.12`.

The full tool output is written to the session log directory. The model sees
bounded output or compact summaries in context and can fetch full output later
with `Recall tr.N`.

This separates storage from context:

- logs keep the full result
- context keeps active raw evidence and compact recall indexes
- `Recall` restores detail on demand
- the active store keeps up to `MAX_COMPLETED_GOAL_TOOL_RESULTS` completed-goal
  results, inside the lower-level `MAX_TOOL_RESULT_STORE_ITEMS` cap

Tool result lifetime:

- full output is always stored under `tr.N` and can be restored with `Recall`
- active context starts with bounded raw output in `Latest Tool Results`
- after another tool batch, older raw output becomes `Unreduced Tool Results`
- OBSERVE either keeps raw output in `Kept Tool Results` or compacts it into
  `Tool Result Index`
- kept results may still have compact timeline entries in `Tool Result Index`
- old timeline summaries may move under `Archived Recall Index`

From the model's view:

1. every tool result gets a `tr.N` key and full log entry
2. ACT sees bounded raw output in Latest Tool Results and Unreduced Tool Results
3. ACT also sees Kept Tool Results selected by OBSERVE
4. OBSERVE sees unreduced raw results selected from `latest` and `recent`
5. OBSERVE must `keep` useful results or `forget` noisy ones
6. forgotten results leave active context, but full logs remain available through `Recall tr.N`

After each tool batch:

1. the previous `latest` moves into `recent`
2. the new batch becomes `latest`
3. unreduced raw results render as Unreduced Tool Results or Latest Tool Results
4. OBSERVE later converts unreduced raw results into Kept Tool Results, Tool Result Index summaries, or forgotten context

This keeps tool results visible until the model has had a chance to decide whether they matter.

## Observe Policy

OBSERVE is triggered when unresolved pending results accumulate by count or raw
context pressure. Tool failures stay visible to ACT first; very large failures
still trigger OBSERVE through raw-context pressure.

In OBSERVE, every unreduced result key must be covered by either:

- `keep`: retain this raw result in `kept_results`
- `forget`: remove this result from future active context

`forget` releases context pressure while preserving logs and Recall ability.

If a forgotten result contained an important conclusion, the model should preserve that conclusion first in plan, known, hypothesis, or verification state.

## Context Budgets

Context is bounded at several layers:

- tool output is bounded before it enters context
- Tool Result Index has an item budget
- Kept Tool Results have a character budget
- Unreduced Tool Results and Latest Tool Results share a raw character pressure threshold that triggers OBSERVE
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
