# nanocode Design

## Agent Model

nanocode uses one primary agent plus a maintenance compactor.

The primary agent is responsible for:

- understanding the latest user request
- maintaining Goal, Plan, Facts, Leads, Checks, and User Rules
- calling repository, shell, edit, and context tools
- verifying work before completion
- deciding when the task is complete

Runtime activities:

- `agent`: normal work. It plans, investigates, edits, verifies, and answers.
- `compact`: maintenance. It rebuilds a minimal Working Context Snapshot from
  old prompt conversation, blackboard state, and raw tool evidence when context
  pressure is high or `/compact` is requested.

There is no separate result-reducer mode and no manual result-retention tools.
Tool-result cleanup is handled by context compaction and by dynamic prompt projection.

## Model Output Protocol

Model decisions use function tools:

- state actions update Goal, Plan, Facts, Leads, Checks, and User Rules
- repository tools read, search, inspect symbols, edit, run commands, and inspect
  git state
- `Recall` retrieves stored tool results by `tr.N` key
- compaction uses a dedicated JSON response contract in the compact activity

Assistant text is user-facing. It must not replace the next useful function
tool when work remains. Completing tracked work still requires a goal action
with `complete=true` after checks are settled.

Function-tool arguments are structured JSON. The CLI display is a separate
human-readable rendering, so a JSON call such as `Read({"path":"a.py"})` can be
shown as `Read a.py`.

## Task State

The active task state lives in the blackboard:

- latest user request
- task code
- goal and completion flag
- plan
- leads: unconfirmed findings, usually source-backed
- facts: confirmed knowledge, usually source-backed
- memory checkpoint for compacted tool results
- check requirement and check result

Recent edits and feedback errors live in the agent runtime. User Rules are
durable session rules.

New user input keeps previous task state available so follow-ups like
`continue` can resume. The agent must realign the state when the latest request
changes the task. When the model sets a different non-complete goal, current
raw tool context is compacted, the memory checkpoint advances, and active Leads
are cleared. Facts remain available unless explicitly changed by state updates.

## Context Construction

Agent prompts are built from stable context toward volatile decision context:

```text
agent user prompt, top -> bottom
+--------------------------------------------------+------------------------------+
| Section                                          | Main control                 |
+--------------------------------------------------+------------------------------+
| Stable Context                                   | provider prefix cache        |
|   - Environment                                  |                              |
|   - User Rules                                   |                              |
|   - Conversation History                         | compact activity             |
+--------------------------------------------------+------------------------------+
| Task State                                       | blackboard                   |
|   - Goal / Facts / Leads / Plan / Focus / Checks |                              |
|   - Recent Edits                                 | RECENT_EDITS                 |
+--------------------------------------------------+------------------------------+
| Tool Context                                     | context budget               |
|   - Tool Result Index                            | index_items                  |
|   - Discovery Context                            | raw_chars / 3                |
|   - File Context                                 | raw_chars + kept_chars       |
|   - Unreduced Tool Results                       | compact checkpoint           |
|   - Latest Tool Results                          | latest batch                 |
+--------------------------------------------------+------------------------------+
| Current Input                                    | latest user request          |
|   - Blocking Feedback                            |                              |
|   - Pending User Feedback                        |                              |
|   - Latest User Request                          |                              |
+--------------------------------------------------+------------------------------+
| Output Guide                                     | final steering               |
+--------------------------------------------------+------------------------------+
```

Layout rules:

- Put stable context higher to preserve provider prefix cache hits.
- Put current user input, blocking feedback, and output rules closest to
  `YOUR OUTPUT`.
- Keep large evidence blocks in Tool Context, above the final decision area.
- Prefer dynamic projections over repeating raw tool outputs.

## Tool Result Storage

Every non-context tool call gets a result key such as `tr.12`.

For regular tools:

- full output is written to the session log directory
- bounded output is stored in the active `tool_result_store`
- prompt context receives bounded raw output, compact summaries, or projections
- full detail can be retrieved later with `Recall tr.N`

Conversation has the same split:

- `conversation_log` is append-only audit state for the session
- `conversation` is prompt context and may be replaced by Working Context
  Snapshot plus recent turns

`Recall` is a context tool. It does not receive a new ordinary result key and
does not add its own raw block to the normal tool-result index. On success, the
stored results it returns are reconstructed as their original result blocks and
reactivated in the current tool context.

Tool-result storage is bounded:

- the runner keeps at most `MAX_TOOL_RESULT_STORE_ITEMS` entries during normal
  storage pressure
- at the start of a user turn, completed-goal storage is pruned toward
  `MAX_COMPLETED_GOAL_TOOL_RESULTS`
- result keys referenced by active state are protected from this pruning

## Tool Result Context

`ToolResultContext` keeps only two active prompt lists:

- `latest`: bounded raw output from the most recent regular tool batch
- `recent`: older blocks, either still raw or already compacted

There is no `kept_results` bucket. Raw blocks remain visible until they are
covered by `compact_context()` or by the memory checkpoint.

After each regular tool batch:

1. previous `latest` moves to `recent`
2. the new batch becomes `latest`
3. `recent` is pruned so compact timeline entries fit the current budget
4. the next prompt renders timeline summaries plus active raw/projection blocks

The Tool Result Index has two parts:

- `Archived Recall Index`: recallable stored results not otherwise visible
- `Current Task Timeline`: compact summaries for current `recent + latest`

Raw content is de-duplicated by result key when rendering unreduced blocks and
timeline entries.

## File Context

File Context is a dynamic prompt projection built before each model request.
It is not separate persistent storage.

Inputs:

- active raw `Read` results
- active raw `Edit` results
- successful `Recall` results after they are reactivated into their original
  blocks

Projection policy:

- Read and Edit outputs carry `source=tr.N`.
- The rendered `Ranges` list and each `@@` content block show the nearest
  source key.
- Lines are merged by file path and line number.
- Newer active Read/Edit results overwrite older lines.
- Edit results invalidate stale old ranges and add the edited replacement
  ranges.
- `replace_all` invalidates the whole file projection for that source.

Freshness policy:

- Read/Edit outputs include file stat and `line:hash|content` anchors.
- If the current file stat still matches the tool result stat, projected lines
  are accepted without rereading the file.
- If file stat changed, only projected line numbers are reread and their hashes
  are checked.
- Stale or missing lines are omitted and reported under `Omitted stale content`.

This prevents Bash or other out-of-band file changes from silently keeping stale
File Context lines in prompt. The slow path only reads lines that are already
being projected.

## Discovery Context

Discovery Context is a dynamic prompt projection for source-discovery results.

Inputs:

- active raw `Search` results
- active raw `InspectCode` results
- successful `Recall` results after reactivation

Policy:

- Discovery Context is source-backed by `tr.N`, but it is treated as leads, not
  current source truth.
- It may include match snippets, symbol outlines, and line anchors.
- Before editing exact code, the agent should use File Context line anchors or
  run `Read` for the missing/current range.
- Discovery blocks are compacted in normal Tool Result Index entries with
  `content=discovery_context`, so the raw output is not repeated in Recent Tool
  Results.

## Read, Search, Edit, and Recall

`Read` accepts structured JSON:

- `Read({"path":"code.py","range":[0,80]})`
- `Read({"path":"code.py","ranges":[[0,80],[160,220]]})`
- `Read({"files":[{"path":"a.py"},{"path":"b.py","range":[10,40]}]})`
- `Read({"path":"a.py","range":[0,20]}, {"path":"b.py","range":[20,40]})`

`Search` accepts one or more structured query objects:

- `Search({"pattern":"class .*Tool","path":"nanocode.py"})`
- `Search({"pattern":"version","glob":"*.toml"}, {"pattern":"version","glob":"*.cfg"})`

`Edit` uses anchored line hashes from Read, Search, or InspectCode. Successful
Edit results record changed ranges and File Context update data, so modified
ranges can appear in File Context without a follow-up Read.

`Recall` retrieves stored results by key and optional line ranges. Recalled
Read/Edit results merge back into File Context. Recalled Search/InspectCode
results merge back into Discovery Context. Newer active Read/Edit blocks still
win over older recalled file lines.

## Compact Policy

Context compaction is the single cleanup path.

`/compact` means rebuilding the working prompt context, not deleting logs. It
reads old prompt conversation, current blackboard state, user rules, recent
edits, and selected tool evidence. It returns direct JSON, not a function tool
call, so reasoning/thinking modes stay available and provider `tool_choice`
quirks do not apply.

The compact JSON contract is:

- `snapshot`: required readable Working Context Snapshot
- `known`: required durable facts, preserving source keys where available
- `goal`, `plan`, `leads`, `checks`, and `user_rules`: optional blackboard/rule
  updates

Before each model request:

1. build the system prompt, user prompt, and tool schemas
2. estimate prompt tokens and record context percent
3. if activity is `agent` and `runtime.compact_at` is reached, run
   `compact_context()`
4. rebuild once after compaction before sending the model request

`compact_context()`:

- selects unreduced raw tool blocks after the memory checkpoint
- passes those blocks, bounded by the raw budget, to the compact model
- replaces old prompt conversation with Working Context Snapshot plus recent
  turns when enough history or tool evidence exists
- updates Goal, Plan, Facts, Leads, Checks, and User Rules from compact JSON
- converts observed raw tool blocks into compact timeline summaries
- advances the memory checkpoint
- reapplies index pruning

Tool failures stay visible to the agent at least once through Latest Tool Results and
blocking feedback. Invalid tool arguments are also remembered as feedback errors
so the model can correct the next call.

## Context Budgets

Context is bounded at several layers:

- each tool output is bounded before it enters active storage
- Tool Result Index is capped by `index_items`
- Discovery Context uses part of the raw character budget
- File Context uses `raw_chars + kept_chars`
- compact triggering is based on estimated or actual prompt tokens
- prompt conversation can be compacted into a Working Context Snapshot while
  full conversation audit state remains append-only
- old stored tool results are pruned unless referenced by active state

Budget presets:

```text
low:    raw_chars=36000   kept_chars=16000   index_items=20   prompt_tokens=64000
medium: raw_chars=72000   kept_chars=32000   index_items=30   prompt_tokens=128000
high:   raw_chars=120000  kept_chars=64000   index_items=60   prompt_tokens=256000
```

`runtime.compact_at` is a context percent from `1` to `100`, or `0` to disable
automatic compaction. The default is `80%`.

The prompt-size estimate is `ceil(chars / 4)` plus tool schema size. When the
provider returns usage, actual prompt/input tokens replace the estimate for
status reporting.

## Status and Commands

The status bar shows:

- model and reasoning label
- optional mode/status notice
- `ctx:NN%`
- current turn tool-call count
- token totals and optional streaming token rate
- current turn elapsed time as `Ns` or `NmNs`

It does not show a separate current model-call timer.

`/context` reports the active context budget, including `prompt_tokens`.
`/status` reports runtime settings, model usage, token usage, code-index status,
goal, and checks.

## Completion and Verification

The agent should complete only when:

- the goal is achieved
- every plan item is done or blocked with concrete context
- required checks are passed or blocked with a stated reason
- failed checks have been recorded and addressed
- the final answer can state what changed, how it was verified, and remaining
  risk

Verification is agent work using tools plus a `verify` state update.

Verification strength is intentionally lightweight:

- `none`: simple chat, explanation, or documentation-only answer
- `light`: static/read confirmation
- `tool`: test, lint, build, search, or executable check
- `user`: visual/manual confirmation

## Design Principle

Keep full logs outside prompt, project current evidence by source inside prompt,
and use compact as the single cleanup path when context pressure requires it.
