# Snapshot-backed source editing

Status: draft implementation specification

This document specifies a breaking replacement for wizolt's model-facing line anchors. It is an
implementation plan, not user documentation. The intended end state has no `line:hash` values in
`Read`, `Search`, `InspectCode`, `Edit`, prompts, or tool schemas. Existing files are edited only
against an immutable source view that records the exact text the model was shown.

## Decision

Replace content-hashed line anchors with session-scoped source views named `view.N`.

A source-producing tool shows ordinary 1-based line numbers and a view id once per file. `Edit`
names that view and ordinary line numbers. The runtime retrieves the exact old text from the view,
validates the complete target against the current file, and either applies the edit, relocates an
unchanged target unambiguously, or refuses the call.

The view is evidence, not authority. The path, visible range, current contents, operation, and
normal Edit confirmation policy are all still validated.

This change is deliberately atomic at release: there is no model-facing compatibility mode and no
schema that accepts both anchors and views. A dual protocol would preserve the exact mental burden
this work is intended to remove.

## Why change

The current protocol makes every editable source line carry a repeated `anchor=line:hash` prefix.
The model must copy opaque hashes, keep them associated with files, understand when they become
stale, and recover through a second read when a call fails. That costs prompt tokens and tool-call
accuracy without adding useful problem information.

Anchors also validate less content than their presentation suggests. A multi-line replacement
validates its start and end lines; a concurrent change strictly inside that range is not part of
either proof. A source view can validate the complete old range because the runtime, rather than
the model, retains it.

The desired model interaction is:

```text
Read/Search/InspectCode -> path + view.N + ordinary numbered source
Edit                    -> path + view.N + ordinary numbered ranges
```

The initial read-before-write dependency remains. The expected gains are fewer input tokens,
fewer malformed Edit calls, fewer stale recovery rounds, and a smaller editing mental model.

## Goals

1. Remove all model-visible line hashes and anchor terminology.
2. Make the common existing-file edit use one view id and ordinary 1-based line numbers.
3. Never edit text that differs from the exact source content shown to the model.
4. Validate every line in a replaced or deleted range, not only its endpoints.
5. Preserve safe relocation for unchanged text shifted by nearby edits.
6. Preserve batched Edit planning, previews, confirmations, diffs, interruption, and resume.
7. Let `Read`, `Search`, and `InspectCode` produce the same kind of edit evidence.
8. Keep read-only tool execution parallel: worker threads produce immutable drafts; only the main
   runner thread mutates Session state.
9. Bound retained view data and make its lifetime deterministic across compaction and resume.
10. Make failure self-correcting: a stale edit returns a small fresh view whenever the current file
    can be read safely.

## Non-goals

- Inferring whether the model selected the semantically correct function or feature. A view proves
  what text was selected, not why it was selected.
- Making arbitrary shell output editable. `Bash("rg ...")`, pipelines, generated output, and stdin
  do not create source views.
- Fuzzy patching, AST rewriting, language-specific editing, or symbol-addressed writes.
- Editing non-UTF-8 files or changing the current newline normalization policy.
- Allowing a view to bypass confirmation, workspace policy, path checks, or diff reporting.
- Preserving old sessions whose conversation teaches the removed anchor schema.
- Adding a second legacy Edit tool.

## Terminology

**Source view**
: An immutable, session-scoped record of source lines actually projected to the model. Its public
  id is `view.N`.

**Source span**
: One contiguous, 1-based inclusive line range inside a source view. A view may contain several
  disjoint spans from the same file.

**Draft**
: Immutable source evidence produced by a read-only tool before a public view id is assigned. Drafts
  can be created concurrently and have no Session side effects.

**Target**
: The exact old line sequence or insertion boundary selected by one Edit operation.

**Same-position validation**
: Comparing the target from the view with the current file at the original line coordinates.

**Relocation**
: Finding the exact unchanged target at a nearby line after same-position validation fails.

**Fresh view**
: A view of the file after a successful Edit, or a bounded current neighborhood returned with a
  stale-edit failure.

## System shape

```text
                         immutable, parallel-safe
  Read/Search/InspectCode ------------------------------+
          |                                             |
          v                                             v
    SourceOutput                                  SourceViewDrafts
          |                                             |
          +---------------- ToolRunner -----------------+
                                |
                                | main thread, call order
                                v
                       Session.source_views
                                |
                                v
  model: Edit(path, source=view.N, lines) -> EditBatchPlan
                                |
                       exact target validation
                                |
                 +--------------+---------------+
                 |                              |
              accepted                       refused
                 |                              |
           write + diff                  error + fresh view
                 |
            fresh view
```

Dependency direction remains downward:

```text
engine -> runner -> tools -> source value types
             |         |
             +------> session -> source value types
context ----------------------> session
session codec ----------------> source value types
```

`wizolt/source.py` owns pure source-view values, rendering primitives, range validation, and exact
relocation algorithms. It imports no runner, tool, context, or Session code. Session owns durable
view identity and retention. Tools construct drafts. Runner commits drafts. This keeps source tools
parallel-safe and prevents a feature module from becoming a second writer of active turn state.

## Model-facing protocol

### View ids

Public ids are monotonically allocated per Session:

```text
view.1
view.2
view.3
```

The counter is assigned only on the runner's main thread, in model tool-call order and then source
block order. A worker has its own Session and therefore its own namespace. A view id mentioned in a
worker's final prose is not valid in the parent Session.

Ids are opaque. The model copies them; it never derives or increments them.

### Line format

Every source producer uses ordinary 1-based lines:

```text
138 | request = self.prepare_request(turn_messages)
139 | failed_request = request
140 | assistant, tool_calls, content = self.model.request(...)
```

The delimiter is exactly ` | `. Line content follows it verbatim except for the existing display
handling of newline terminators. No hash, checksum, revision, or hidden suffix appears in model
text.

### Read output

A single-file read:

```text
<Read path="wizolt/engine.py" source="view.12" lines="138:155" total_lines="689">
138 | request = self.prepare_request(turn_messages)
139 | failed_request = request
...
</Read>
```

A batched Read emits one block and one view per file, not one view per requested range. Requested
ranges for the same path are normalized, sorted, and merged when they overlap or touch. Disjoint
ranges remain separate source spans under the same view.

```text
<Read path="wizolt/engine.py" source="view.12" lines="20:40,138:155" total_lines="689">
...
</Read>
<Read path="wizolt/runner.py" source="view.13" lines="200:280" total_lines="578">
...
</Read>
```

### Search output

Search remains an `rg`-backed tool when ripgrep is available, but its final source rows are grouped
by file and hydrated from one current file read. The `>` marker means a regex match; a space means
context.

```text
<Search pattern="parallel_safe" matches="2">
<file path="wizolt/runner.py" source="view.21" lines="282:305">
  282 | def parallel_segment_end(...):
> 288 | def parallel_safe(self, call: ToolCall) -> bool:
  289 |     ...
</file>
</Search>
```

For a Search call containing several queries, all visible spans for one path are unioned into one
view. Each query result may refer to the same view id. The source text shown in every block must be
byte-for-byte consistent with that view.

Ripgrep is a candidate finder, not the evidence source. After `rg --json` returns path and line
candidates, Search reads each candidate file once, reruns the requested regex against that captured
content, and renders matches and context from the capture. A file changed between `rg` and capture
therefore produces current results, not a view paired with stale `rg` text.

### InspectCode output

InspectCode stops requesting anchors from code-symbol-index. Navigation metadata remains in its
compact text form. Any source or reference context intended to be editable is hydrated from the
current file and rendered as a normal source block with a view id.

```text
definition:
  name: ToolRunner.parallel_safe
  file: wizolt/runner.py
  range: 288:305

<source path="wizolt/runner.py" source="view.30" lines="288:305">
288 | def parallel_safe(self, call: ToolCall) -> bool:
...
</source>
```

The code index may identify a stale path or range. It cannot mint evidence itself. Hydration reads
the current path and either emits current lines or omits the source block with an explicit stale
index note. Structural metadata may still be useful, but only a hydrated source block is editable.

### Bash and raw rg

Bash output never contains trusted `source=` attributes. The runtime does not parse arbitrary `rg`
text because flags, cwd, pipes, transformations, stdin, color, and generated content destroy a
reliable correspondence with a file snapshot.

After locating code through Bash, the model must use Read, Search, or InspectCode before Edit. The
system prompt and Bash/Edit schemas state this directly.

### Edit schema

One Edit call still targets one path. All existing-file operations in that call use one top-level
source view:

```json
{
  "path": "wizolt/engine.py",
  "source": "view.12",
  "edits": [
    {
      "op": "replace",
      "start": 138,
      "end": 140,
      "content": "replacement\n"
    },
    {
      "op": "insert_after",
      "line": 145,
      "content": "new_line()\n"
    }
  ]
}
```

The model-facing operation set becomes:

```text
create        content
replace       start, end, content
delete        start, end
insert_before line, content
insert_after  line, content
```

Rules:

- `source` is required for every existing-file call.
- `source` is forbidden for `create`; `create` remains the only operation in its call.
- `start`, `end`, and `line` are integers, 1-based.
- `start` and `end` are inclusive.
- A replace or delete range must lie wholly inside one contiguous visible span in the view.
- An insertion line must be visible in the view.
- For an existing empty file, `insert_after` with `line: 0` is accepted only when the source view
  explicitly represents that empty file. Line zero is invalid in every other case.
- Operations are ordered logically but resolved against the source view before splicing. Overlap or
  a shared insertion boundary is refused as today.
- `replace_all` and `replace_unique` are removed from the model schema for this experiment. Keeping
  a second exact-text editing protocol would obscure whether the source-view model is simpler. A
  global change uses Search plus batched range edits. It may be reconsidered from evaluation data.
- Multiple views for one path require multiple Edit calls. The model may issue those calls in the
  same assistant message; EditBatchPlan coordinates them without another LLM round trip.

### Successful Edit output

Edit retains the current unified diff and warnings, then returns a fresh view covering every changed
hunk plus up to three unchanged context lines on either side:

```text
<Edit path="wizolt/engine.py">
... diff ...
<source path="wizolt/engine.py" source="view.31" lines="136:149">
136 | ...
...
</source>
</Edit>
```

The old view is immutable and is not rewritten. A later operation may still use an unchanged,
non-overlapping span from it, subject to normal validation. The fresh view is the preferred source
for work near the changed hunk.

### Failed Edit output

A view-related failure returns one error classification and, when the path is readable, a bounded
fresh view centered on the requested coordinates:

```text
status: failed
error: source target changed
requested: view.12 lines 138:140
<source path="wizolt/engine.py" source="view.32" lines="135:143">
135 | ...
...
</source>
```

The model can retry directly against `view.32`; it does not need a separate Read merely to obtain
new edit evidence. The fresh block is factual current context, never an inferred target.

## Source-view data model

The following shapes are normative; exact Python names may vary only if the same ownership and
invariants remain explicit.

```python
@dataclass(frozen=True)
class SourceSpan:
    start: int                 # 1-based inclusive
    lines: tuple[str, ...]     # exact normalized line strings, newline included

    @property
    def end(self) -> int: ...  # 1-based inclusive


@dataclass(frozen=True)
class SourceViewDraft:
    path: str                  # canonical resolved path
    display_path: str          # stable model-facing path
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str              # Read, Search, InspectCode, or Edit


@dataclass(frozen=True)
class SourceView:
    key: str                   # view.N
    path: str
    display_path: str
    total_lines: int
    spans: tuple[SourceSpan, ...]
    producer: str
    round: int
    step: int
```

Invariants:

- A non-empty file view has non-empty spans. An empty file view has `total_lines == 0` and no
  spans; that explicit state supplies its one legal insertion boundary.
- Spans are sorted, non-overlapping, and non-touching after normalization.
- `lines` preserve the exact normalized text shown to the model.
- `total_lines` is the captured file's line count, not the current line count.
- A view contains one canonical path only.
- A public key names exactly one immutable view for the life of a Session.
- Registering a draft cannot overwrite a prior key.
- A loaded view is validated before use; malformed persisted data is dropped, never repaired by
  guessing.

## Source-aware tool output

The current `Tool.call() -> str` result cannot express both retained full output and the exact
source blocks that survived model-output bounding. Source-producing tools therefore return a
structured source output. Ordinary tools may continue returning strings, normalized by the runner
to the same result abstraction.

Conceptually:

```python
@dataclass(frozen=True)
class SourceBlock:
    draft: SourceViewDraft
    markers: tuple[str, ...]   # e.g. Search match/context markers


@dataclass(frozen=True)
class ToolOutput:
    retained_text: str
    parts: tuple[str | SourceBlock, ...]
```

The final naming and rendering sequence is:

1. A worker thread executes the tool and returns immutable ToolOutput and drafts.
2. Source-aware projection selects complete source spans within the normal tool-output budget.
3. The main runner thread allocates `view.N` keys for the selected drafts in call order.
4. It registers those views in Session.
5. It renders the model text containing the assigned keys.
6. It stores the full plain retained output under `tr.N` and emits the bounded model text.

Only model-projected spans become views. A line omitted from model output cannot be targeted by
guessing its number. Projection clips at source-span boundaries and may split a large span into a
visible head span and visible tail span. The omitted middle is not part of either registered span.

The retained `tr.N` output remains useful for Recall, but v1 Recall does not create edit evidence.
Recalled plain text has ordinary line numbers and no `source=` attribute. To edit recalled or
materialized omitted content, the model uses Read/Search, which creates a new current view. This is
an intentional safety boundary and avoids teaching Recall to reconstruct partial file authority.

Source-bearing model output must already fit the normal output budget; passing it through generic
character-based `ContextManager.bound_output` again is a bug because that function may cut a source
block in the middle. The generic bounding path remains for non-source tools.

## Validation and relocation

### Path validation

Before resolving any operation, Edit:

1. Resolves the requested path through the existing Session path policy.
2. Loads the named view from the same Session.
3. Requires the resolved requested path to equal the view's canonical path.
4. Applies the existing file existence, directory, external-path, and confirmation checks.

A view id is not an authorization token and does not broaden writable scope.

### Range validation

For replace and delete:

1. Require `1 <= start <= end <= view.total_lines`.
2. Require every selected line to belong to one contiguous SourceSpan.
3. Extract the complete old line sequence from the view.
4. Reject an empty target.

For insert-before and insert-after:

1. Require the named line to be visible in one SourceSpan.
2. Record the insertion boundary plus the immediately adjacent visible lines on both sides, up to
   two per side. These lines form the boundary witness.
3. An empty-file view uses one special boundary at line zero and is valid only while the current
   file remains empty.

### Same-position validation

Replace/delete succeeds at its original coordinates only when every current target line equals the
view target. Interior changes therefore invalidate the edit even when the first and last lines are
unchanged.

Insert succeeds at its original boundary only when all available boundary-witness lines still
exist, are equal, and remain adjacent. A change on either side of a witnessed boundary invalidates
the insertion.

### Relocation

Relocation runs only after same-position validation fails.

- Search within `MAX_VIEW_DRIFT = 50` lines of the original target start, preserving the current
  anchor behavior's bounded tolerance.
- Match exact normalized line sequences; never use similarity, syntax, embeddings, or partial
  hashes.
- A replace/delete target relocates only when the complete old target sequence has exactly one
  candidate in the window.
- An insertion boundary relocates only when its complete available witness has exactly one
  candidate in the window.
- Zero candidates means changed or removed. Multiple candidates means ambiguous. Both are refused.
- A successful relocation is reported explicitly in the tool result and transcript:

  ```text
  relocated view.12 lines 138:140 -> current lines 141:143
  ```

No relocation is silent.

### Multi-operation calls

Every operation in one Edit call is resolved against the same immutable view before any splice is
applied. Replacements are then applied in reverse current-index order, preserving the existing
anti-shift behavior. Overlap and shared insertion points are rejected before writing.

### Batched Edit calls

EditBatchPlan continues to construct one in-memory file state for a serial run of Edit calls. Each
line carries its origin as `(source_view, source_line)` when available. An insertion preserves
origins of existing lines; new lines have no source origin. A replacement or deletion consumes its
origins.

Consequences:

- Later calls in the same assistant batch can use a pre-edit view after earlier insertions shifted
  its untouched lines.
- A later call targeting a line consumed by an earlier call is refused.
- Calls using different views of the same path validate against the planned current state in model
  order.
- Confirmation previews the final planned diff while each individual tool call still receives one
  result.

### Check-before-write

Planning remains side-effect free. Immediately before writing, PlannedEdit rereads the file and
requires it to equal the exact `before` content used by the plan. A changed file is refused as
`planned edit is stale`; the failure returns a fresh bounded view. This preserves the current
check-before-write boundary.

The filesystem offers no general atomic compare-and-swap for an arbitrary path. The remaining
read/write race is the same narrow OS-level race as today and is not widened by source views.

## Errors and recovery

View errors use stable categories in their first line. Detailed text follows for the model and user.

```text
source missing          view.N is unknown or expired
source path mismatch    Edit path and view path differ
source range unseen     requested lines were not projected in this view
source target changed   exact target differs and cannot relocate
source target ambiguous exact target or boundary has multiple relocation candidates
source target consumed  an earlier edit in the same batch replaced/deleted it
planned edit stale      file changed after planning and before write
```

Rules:

- None of these failures write the file.
- Retrying an unchanged failed call is never suggested.
- When the current path is readable, the failure includes at most seven current lines around the
  requested coordinates and registers them as a fresh view.
- Missing/path-mismatch errors do not reveal content from the mismatched view.
- An expired view tells the model to Read/Search again; it cannot be reconstructed from its id.
- Malformed operations fail before any current file content is returned.
- A no-op remains an error and returns the current target as a fresh view.

Failed Edit source output requires the runner to preserve source drafts carried by a ToolError.
Implement this as structured failure data rather than parsing error strings. The runner registers
those drafts on the main thread exactly like successful output, but still records a failed tool
result and short-circuits only under the existing refusal rules.

## Lifetime, persistence, and compaction

### Session ownership

Session adds:

```python
source_view_counter: int
source_views: dict[str, SourceView]
```

Only Session allocates keys and owns live views. Source tools never mutate this mapping.

### Retention roots

A view remains live while its id occurs in any model-visible current state:

- committed model messages;
- the active turn;
- current goal, plan, known facts, or check text.

Transcript-only history is not a retention root. Compacted HistorySegments and RecallContext may
show an old `view.N` as historical text, but the view expires when it leaves active model context.
Attempting to use it returns `source missing` and requires a current read.

After compaction, source-view pruning scans the same kept model messages used to prune tool records,
plus AgentState text. Unreferenced views are removed. There is no independent time-based expiry
while a view remains active.

### Size

Only visible spans are persisted, never an implicit whole-file snapshot. Source-aware output
projection keeps any one turn's new view bytes within the existing tool-output budget. Compaction
then bounds the aggregate reachable set.

The implementation records view count and bytes for diagnostics. It may enforce a defensive hard
ceiling only by dropping oldest unreferenced views. It must never silently evict a view still
referenced by active model context; if referenced state alone exceeds a storage ceiling, checkpoint
saving fails clearly rather than leaving valid-looking ids behind.

### Session snapshot encoding

Source views are durable because an interrupted turn may already have shown one to the model, and a
resumed assistant must be able to use it.

SessionSnapshotCodec stores view metadata inline and span text as content-addressed blobs, reusing
the existing blob mechanism used by TurnDiff and retained history. The marker includes a digest of
ordered view keys. Deltas replace the bounded view sequence when pruning changes it and append when
only new views were added.

Loading validates key shape, path, positive ordered spans, line counts, and referenced blobs.
Missing or malformed view blobs drop that view. If an active message still references a dropped
view, Edit returns `source missing`; load never invents content.

### Breaking session format

The implementation bumps `SessionSnapshotStore.FORMAT_VERSION`. Older sessions are refused with the
existing unsupported-format error. This is intentional:

- old assistant messages contain anchor-shaped Edit calls;
- old tool results teach the removed syntax;
- silently resuming them against a new schema increases failure and mental burden;
- the project already treats session formats as non-migrated snapshots.

No anchor-to-view migration is attempted because the old records do not encode which bounded lines
were actually visible to the model after output clipping.

## Concurrency

Read, Search, and InspectCode remain eligible for `max_parallel_tools`.

Their worker-thread phase may:

- read files;
- execute ripgrep or code-symbol-index;
- create immutable SourceViewDraft and SourceBlock values;
- compute retained and projected output candidates.

It may not:

- allocate `view.N`;
- write Session state;
- persist blobs;
- emit terminal output;
- update tool records.

ToolRunner finalizes outcomes in model order as today. It performs view allocation, registration,
rendering, display, record storage, and checkpointing on the main thread. Parallel completion order
therefore cannot change ids or replay text.

## Per-module changes

### `wizolt/source.py` (new)

- SourceSpan, SourceViewDraft, SourceView, SourceBlock, and structured ToolOutput values.
- Span normalization and containment.
- Ordinary numbered-line rendering.
- Exact range and boundary-witness extraction.
- Pure same-position and bounded relocation algorithms.
- No imports from tools, runner, context, or Session.

### `wizolt/session/__init__.py`

- Own the source-view counter and mapping.
- Allocate/register/get/prune methods.
- Source ids remain separate from `tr.N`; a tool result and a view have different lifetimes and
  purposes.

### `wizolt/session/codec.py` and `store.py`

- Encode live views and their content-addressed span blobs.
- Include view state in snapshot/delta markers and load.
- Retain referenced blobs during log rewrite/garbage collection.
- Bump the session format version.

### `wizolt/tools/base.py`

- Permit a tool to return structured ToolOutput as well as ordinary text.
- Permit ToolError to carry structured recovery output without making every error source-aware.
- Keep the default tool implementation surface unchanged for non-source tools.

### `wizolt/runner.py`

- Carry structured output and source drafts through serial and parallel outcomes.
- Project source blocks, allocate ids, and register views on the main thread.
- Store full retained plain output while sending the source-aware bounded projection to the model.
- Register fresh views on Edit failures.
- Preserve original tool-result ordering and one result per emitted call.

### `wizolt/context.py`

- Do not character-clip an already projected source-bearing result.
- Prune source views after compaction using active model references and AgentState.
- Keep generic output bounding and materialization unchanged for ordinary tool output.

### `wizolt/tools/files.py`

- Delete line hash, anchor parsing, anchor matching, and anchor relocation helpers.
- Read returns SourceBlocks with ordinary line numbers.
- Edit schema uses source ids and integer lines.
- Remove replace_all and replace_unique from the model-facing operation set.
- Edit validates complete targets through source.py.
- Successful and recoverable failed edits produce fresh views.
- Current-target and no-op context uses ordinary lines and fresh view ids.

### `wizolt/tools/editplan.py`

- Replace anchor origins with source-view line origins.
- Resolve each call against its named view and planned file state.
- Preserve final-preview and check-before-write behavior.

### `wizolt/tools/search.py`

- Refactor Search backends to produce structured path/line matches before formatting.
- Hydrate ripgrep candidates from one captured current read per path.
- Group visible source by path and emit SourceBlocks.
- Invoke code-symbol-index without anchors and hydrate editable snippets from current files.

### Prompts, built-in skill, docs, changelog

- Replace all current guidance about copying, refreshing, or refunding anchors.
- Teach the one rule: edit existing text only with a source id and visible ordinary line numbers.
- Explain that Bash output is not a source.
- Rewrite user documentation rather than appending a second editing explanation.
- Redraw Edit term-shots because their displayed rows change.
- Add a breaking Unreleased changelog entry.
- Historical changelog entries remain historical and are not rewritten.

## Implementation sequence

The branch may be implemented in stages, but no intermediate stage is releasable with both public
protocols enabled.

### Stage 0: baseline

- Capture representative real-model sessions using anchors.
- Record first Edit validity, retries, stale failures, Read/Search input characters, tool batches,
  and time from first source read to successful write.
- Freeze a small A/B task corpus before changing prompts.

### Stage 1: pure source model

- Add source.py values, span normalization, rendering, target extraction, and relocation.
- Exercise them independently from tools and Session.

### Stage 2: structured runner output

- Add ToolOutput normalization for string-returning tools.
- Carry drafts through serial and parallel runner paths.
- Add source-aware projection and deterministic main-thread view allocation.
- Keep source producers disabled until persistence exists.

### Stage 3: durable Session views

- Add Session registry, codec blobs/deltas, load validation, pruning, and format bump.
- Verify interrupt/resume after a source-producing tool call.

### Stage 4: Read plus Edit

- Convert Read output.
- Replace the Edit schema and implementation.
- Convert EditBatchPlan, previews, success context, no-op context, and structured failure recovery.
- Delete anchor helpers once no production path references them.

### Stage 5: Search and InspectCode

- Convert Search to structured matches and per-file views.
- Remove code-symbol-index anchor requests.
- Hydrate current source blocks for InspectCode.
- Confirm both rg and Python fallback produce identical view semantics.

### Stage 6: remove old surface

- Remove replace_all/replace_unique and anchor-era compatibility parsing.
- Rewrite prompts, built-in help, user docs, examples, screenshots, and changelog.
- Delete or rewrite anchor-specific tests; do not retain assertions for removed syntax.

### Stage 7: evaluate

- Run the same model/task corpus against the snapshot branch.
- Compare convergence and safety metrics before deciding whether to merge or revise the protocol.

## Verification plan

### Pure range behavior

- One-line and multi-line replace/delete at original coordinates.
- Insert before/after, BOF, EOF, and existing empty file.
- CRLF normalization and final line without newline retain current semantics.
- Disjoint spans accept contained ranges and reject gap-crossing ranges.
- Invalid, reversed, zero, negative, and out-of-view coordinates reject without writes.
- Multi-line interior mutation rejects even when first and last lines are unchanged.

### Relocation

- Unique exact target shifted within 50 lines relocates and reports the move.
- Shift beyond 50 lines rejects.
- Changed target rejects.
- Duplicate target rejects.
- Boundary witness disambiguates an insertion next to repeated lines.
- Ambiguous or partially changed boundary rejects.
- Empty-file insertion rejects after another writer adds content.

### Batched edits

- Several operations from one view apply in reverse splice order.
- Multiple Edit calls from one assistant batch survive earlier insertions.
- A later call targeting a consumed origin rejects.
- Calls from different views of the same path validate in model order.
- A refused confirmation skips later calls exactly as today.
- Every emitted call receives one result on success, failure, refusal, or interruption.

### Source producers

- Read batches one view per path and merges visible ranges correctly.
- Search rg and Python backends produce the same path/line/span model.
- A file changing between rg candidate discovery and hydration cannot pair stale text with a view.
- InspectCode index metadata cannot authorize stale indexed content.
- Bash/rg output never mints a source.
- Successful and failed Edit results mint only the lines actually returned.

### Bounding and visibility

- A large Read registers head/tail visible spans only.
- An omitted middle range is rejected as unseen even when its line number is guessed.
- Source blocks are never cut mid-row by generic context bounding.
- Recall of retained plain output does not silently create edit authority.
- Reading a materialized result file creates a normal new view for that artifact path.

### Persistence and compaction

- A view survives checkpoint, interruption, process restart, and resume while referenced.
- Content blobs deduplicate equal span text.
- Compaction retains views still named in current messages or AgentState.
- Compaction drops unreferenced views and Edit reports `source missing` afterward.
- RecallContext showing an expired historical id does not reactivate it.
- Worker and parent view namespaces do not cross.
- Old session format is refused clearly.

### Runner concurrency

- Parallel Read/Search/InspectCode calls overlap execution.
- View ids are assigned in model call order, independent of completion order.
- Worker threads do not mutate Session source state.
- Parallel failures do not reserve or skip ids for absent views unless that behavior is explicitly
  chosen and pinned.

### Existing behavior

- Edit previews, yolo display, approval/refusal, `/diff`, stored results, and code-index updates
  remain intact.
- External path confirmation and secret-handling boundaries remain intact.
- Full targeted tests, full pytest, ruff check/format, pyright, and both required documentation
  builds pass before release.

## Model evaluation

Unit tests prove mechanical safety; they cannot prove that an LLM finds the protocol easier. The
experiment therefore requires real model evaluation with fixed prompts and repositories.

Measure per task:

- source-output input characters and estimated tokens;
- model requests before first successful Edit;
- first Edit schema-valid rate;
- first Edit mechanically accepted rate;
- stale/missing/unseen view failures;
- unchanged failed-call retries;
- total Edit calls and total tool batches;
- incorrect-target changes caught by tests or human review;
- task completion time and total model tokens.

Corpus categories:

- one-line replacement;
- insertion beside a named symbol;
- multi-line function replacement;
- two edits in one file;
- edits in several files from one batched investigation;
- target shifted between read and edit;
- target changed between read and edit;
- duplicate source text;
- Search-driven edit;
- InspectCode-driven edit;
- raw-rg discovery followed by Read;
- large output with the target in an omitted middle.

Primary success criteria:

1. No mechanically wrong write in adversarial mutation cases.
2. First Edit mechanically accepted rate is higher than the anchor baseline.
3. Median source-output tokens are lower.
4. Median model requests from first source inspection to successful write do not increase.
5. Snapshot failures recover in at most one following Edit call when the returned fresh view
   contains the intended target.

The protocol should not merge merely because unit tests pass. If real-model first-call acceptance
does not improve, revise or abandon the experiment.

## Rollout and rollback

This work develops on a branch and lands as one breaking release change after evaluation.

Rollout:

1. Complete implementation and deterministic tests.
2. Run anchor baseline and snapshot candidate evaluations with the same model settings.
3. Review safety failures before performance results.
4. Update English user docs, built-in help, term-shots, and Unreleased changelog.
5. Merge only the snapshot protocol; do not ship a runtime toggle.

Rollback before merge is deleting the branch. Rollback after release is a normal code revert plus
another session-format bump; sessions created by the snapshot release are not loaded by an
anchor-based rollback. There is no attempt to translate source ids into hashes.

## Rejected alternatives

### Raw line numbers

Rejected because they silently target whatever currently occupies a position after the file
changes. A view id without exact old-content validation is only this unsafe design with another
name.

### Whole-file revision plus line numbers

Rejected as the primary proof because an unrelated edit anywhere invalidates every target. Ignoring
the revision would be unsafe; enforcing it would cause unnecessary rereads. Target-content views
validate only what the model intends to change.

### Keep anchors but shorten their rendering

Rejected because it saves some tokens while retaining hash copying, stale-anchor reasoning, and
endpoint-only range validation.

### Accept anchors and views together

Rejected because the model then chooses between two addressing systems, old sessions continue
teaching the removed one, tests double, and evaluation cannot isolate the new protocol.

### Parse arbitrary Bash/rg output

Rejected because the runtime cannot reliably recover path, version, exact content, or line identity
after arbitrary shell transformations.

### Fuzzy matching

Rejected because a plausible target is not proof. Only exact full-target or exact boundary-witness
matching may relocate.

### Whole hidden-file snapshots

Rejected because they let guessed unseen line numbers authorize edits and retain more data than the
model received. Views contain only projected source spans.

### Make source tools write Session directly

Rejected because Read/Search/InspectCode run concurrently. Shared registration inside `call()`
would violate the runner's pure parallel phase and make view ids completion-order dependent.

## Completion criteria

The implementation is complete only when all of the following are true:

- No current model-facing Read, Search, InspectCode, Edit, prompt, schema, help, or user-doc text
  contains anchor/hashline instructions.
- Existing-file Edit requires a valid source view and integer lines.
- Complete selected content is validated before every replacement or deletion.
- Insertions validate an exact visible boundary witness.
- Relocation is exact, bounded, unique, and explicitly reported.
- Read-only source producers remain parallel-safe and deterministic.
- Source views survive resume while active and expire deterministically after compaction.
- Output bounding cannot authorize omitted source.
- Old sessions fail clearly rather than replaying anchor instructions.
- Mechanical safety tests and real-model evaluation meet the primary success criteria.
- User documentation describes only the snapshot protocol and its cost: source views expire after
  leaving active context, and raw shell output must be re-read through a source-producing tool.
