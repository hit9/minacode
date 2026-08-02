# Tools

minacode uses tools to inspect your project and act on it. You describe the outcome you want;
the agent chooses the tools. Tool calls are shown in the terminal as they run. They are separate
from the `/` commands you type yourself. Read-only tools may run concurrently; actions that can
change your system ask for confirmation unless `--yolo` or `/yolo` is active.

## Built-in tools

::::{list-table}
:header-rows: 1
:widths: 24 76
:class: tool-reference

* - Tool
  - What it does
* - **`Read`**
  - Opens selected line ranges from one or several UTF-8 files. Every returned line has an
    anchor that later edits can verify.

    A shortened result looks like this:

    ```html
    <Read path="minacode.py">
      <file_stat mtime_ns="..." size="222039"/>
      <total_lines>5031</total_lines>
      <range>684:687</range>
      <content hashline-numbered>
        anchor=684:234ew | class Tool:
        anchor=685:7xy0d |     NAME: ClassVar[str] = ""
        anchor=686:5exvk |     DESCRIPTION: ClassVar[str] = ""
      </content>
    </Read>
    ```

    In `684:234ew`, `684` is the zero-based line number and `234ew` is a short hash of that
    line's content. The line number locates the edit; the hash proves that the line has not
    changed since it was read.
* - **`ViewImage`**
  - Opens one local PNG, JPEG, WebP, or single-frame GIF as visual input for the active model.
    The agent can use it proactively for screenshots, mockups, diagrams, and generated assets.
    Images outside the workspace require confirmation, and the active provider/model must support
    image input.
* - **`Search`**
  - Finds text with case-insensitive regular expressions, optionally limited by path or filename
    pattern. It skips hidden, binary, and gitignored files and returns editable anchors.
* - **`InspectCode`**
  - Finds definitions, references, implementations, callers, callees, and file outlines through
    the [code index](#code-symbol-index). Use it for code structure rather than exact text.
* - **`Edit`**
  - Creates or changes one UTF-8 file by inserting, replacing, or deleting content.
    For an anchored change, `Edit` sends back the `line:hash` value returned by `Read`, `Search`,
    or `InspectCode`. minacode checks the current line immediately before writing and
    <span class="marker">refuses the edit if the hash no longer matches</span>. Successful edits
    appear in [`/diff`](usage.md#reviewing-changes).

    :::{figure} ../snapshots/minacode-edit-preview.png
    :alt: An Edit confirmation previewing the proposed diff
    :width: 100%
    :align: center

    An Edit confirmation previews the proposed change before approval.
    :::
* - **`Bash`**
  - Runs one shell command in the project with live output. Commands still running after
    `runtime.bash_wait_timeout` <span class="marker">become background jobs automatically</span>.

    :::{figure} ../snapshots/minacode-bash-live-preview.gif
    :alt: A Bash tool call streaming command output in minacode
    :width: 100%
    :align: center

    Bash output appears as the command runs.
    :::
* - **`Job`**
  - Starts or manages background commands: check output, wait, list, or stop. The same jobs are
    visible through `/ps`.
* - **`Recall`**
  - Retrieves a <span class="marker">complete earlier tool result</span>, or selected line ranges,
    when only a shortened result was placed in the conversation.
* - **`RecallContext`**
  - Lists stored compacted segments newest first, retrieves an excerpt by its `seg.N` key, or
    searches titles and text with a case-insensitive regex such as `cache prefix|task memory`.
    Listing supports pagination; search results are capped matching lines. Segment titles are
    loaded only on demand instead of occupying every request.
* - **`Note`**
  - Views or updates the task's goal, plan, success check, and learned facts. Updates are durable
    conversation history, so they preserve append-only prompt-cache prefixes and do not edit files.

    <div class="term-shot" role="img" aria-label="A Note update printed in the terminal: goal and check lines, a plan whose items are marked done, in progress, or waiting, and a list of learned facts."><span class="fs-goal">goal: ship the tokenizer fix</span><span class="fs-goal">check: pytest -q passes</span><span class="fs-sel">plan:</span><span class="fs-add">  - [x] reproduce the failing test</span><span class="fs-doing">  - [~] fix the tokenizer</span><span>  - [ ] update the changelog</span><span class="fs-sel">known:</span><span class="fs-add">  + tests run with pytest -q</span></div>

    Plan items are marked `[x]` done, `[~]` in progress, `[ ]` waiting, or `[-]` blocked.
* - **`Ask`**
  - Pauses for a decision that genuinely needs you. A question may include choices and a
    recommended option.

    <div class="term-shot" role="img" aria-label="An Ask prompt: the question, then a selector listing two choices with the recommended one pre-selected, and a preview line for the highlighted choice."><span class="fs-user">Which approach?</span><span> </span><span>Select:</span><span class="fs-dim">  j/k move, / search, Esc/q back/cancel</span><span class="fs-sel">&gt;  1. Refactor <span class="fs-i fs-add">(recommended)</span></span><span class="fs-dim">   2. Rewrite</span><span class="fs-dim">  │ Extract module +87 -12</span></div>

    Pressing `Esc` declines the question; typing instead of choosing answers in free text.
* - **`NextHints`**
  - Offers 2–3 short next-step prompts the model suggests after its answer. They appear as
    selectable chips at the idle prompt; `Tab` cycles focus, `Enter` submits, and `/hints`
    toggles them off. An all-`NextHints` batch ends the turn in a single model call.

    <div class="term-shot" role="img" aria-label="A terminal at the idle prompt after a NextHints call: the answer text above, then an empty prompt with a caret, a gap line, and one horizontal row of three cyan suggestion chips separated by grey bars with the middle chip highlighted in reverse."><span>Everything is ready to review.</span><span> </span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span> </span><span><span class="fs-i fs-sel"> run the tests </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-on"> show the diff </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-sel"> commit the work </span></span></div>

    `Tab` moves the highlight; `Enter` submits the focused chip as your next message.
* - **`Skill`**
  - Loads an installed skill's full instructions when needed. It appears only when skills are
    installed; see [Skills](skills.md).
* - **`MCP`**
  - Describes or calls tools and reads resources from a connected MCP server. It appears only
    after a server is connected; see [MCP](mcp.md).
::::

(provider-side-tools)=
## Provider-side tools

Some providers can run tools <span class="marker">inside the model request itself</span>, without
minacode executing anything. Web search is the main one: the model searches, reads the results,
and answers with sources, all before the response comes back. This is off by default and enabled
per provider with [`builtin_tools`](configuration.md#provider-side-tools).

They behave differently from the built-in tools above, and the differences matter:

| | Built-in tools | Provider-side tools |
|---|---|---|
| Runs where | Your machine | The provider's servers |
| Confirmation | File-changing calls ask first | Never — the call happens inside one model request |
| Result size | Shortened, with `Recall` for the rest | Whatever the provider injects, uncapped |
| Availability | Always | Only if the provider offers it, and you enabled it |

A search is visible in three places: a transcript line as it is reported, a `web search` phase in
the working status while it runs, and the sources listed under the answer.

```text
  ├ web search httpx timeout configuration
The client accepts a `timeout` argument taking either a float or a `Timeout` object.

**Sources**

1. [Timeouts — HTTPX](https://example.com/httpx/timeouts)
```

The transcript line is written from the model's response rather than from the stream, so it
appears whether or not streaming is enabled. Sources are shown only when the provider reports
them; some report full citations, others only the pages they visited, and a few report neither.

Search results are untrusted text placed directly in the model's context, and nothing pauses to
ask before a search runs — the only control is whether the tool is offered at all. Leave it off
when the agent runs unattended, or when the questions themselves are sensitive.

## Code symbol index

minacode includes a **code symbol index** for <span class="marker">structured navigation</span> —
finding definitions, callers, references, and implementations without relying on an external
language server. The index is <span class="marker">built separately for each project</span>.

### What it is

The index is a static database of symbols (functions, classes, methods, variables,
etc.) extracted from your project's source files. It is built by a library called
[code-symbol-index](https://github.com/hit9/code-symbol-index), which supports a
broad set of languages.

When the index is available, the `InspectCode` tool can:

- **Find symbols** by name with fuzzy matching
- **Inspect a symbol** — show its definition and members
- **List references** — call, read, write, and type references across the project
- **Walk call chains** — transitive callers and callees
- **File outlines** — symbol tree of a single file

Asking where `MCPManager` is defined returns the symbol itself, not every line that mentions
the word:

<div class="term-shot" role="img" aria-label="An InspectCode find query for MCPManager returning matching symbols with their kind, file, line range, and whether the match was exact or fuzzy."><span><span class="fs-i fs-dim">query:</span> MCPManager</span><span><span class="fs-i fs-dim">count:</span> 3</span><span> </span><span class="fs-dim">symbols:</span><span>  - <span class="fs-i fs-dim">name:</span> <span class="fs-i fs-sel">MCPManager</span></span><span>    <span class="fs-i fs-dim">kind:</span> class</span><span>    <span class="fs-i fs-dim">file:</span> minacode.py</span><span>    <span class="fs-i fs-dim">range:</span> 4271:5374</span><span>    <span class="fs-i fs-dim">score:</span> <span class="fs-i fs-add">exact</span></span><span>  - <span class="fs-i fs-dim">name:</span> <span class="fs-i fs-sel">TestMCPManagerDiscovery</span></span><span>    <span class="fs-i fs-dim">kind:</span> class</span><span>    <span class="fs-i fs-dim">file:</span> tests/test_mcp.py</span><span>    <span class="fs-i fs-dim">range:</span> 272:573</span><span>    <span class="fs-i fs-dim">score:</span> <span class="fs-i fs-dim">fuzzy</span></span></div>

Each hit carries its file and line range, so the agent can open exactly the right lines. The
same index answers "who calls this" and "what implements this" the same way.

```{note}
Without an index, `InspectCode` reports that the index is unavailable. Run `/index` once in a
project to build it.
```

### Building and syncing

<span class="marker">Run `/index` to build or rebuild the index.</span> The first build walks every
source file; subsequent builds sync from the previous snapshot and are much faster. Add
`force` to rebuild from scratch.

When an index already exists, minacode refreshes it in the background at startup. After an
agent turn, it <span class="marker">automatically updates small batches of changed source
files</span>; run `/index` when a large set of changes leaves it stale. `/status` shows the
current state:

| State | Meaning |
|---|---|
| **synced** | Index is current and ready |
| **stale** | Out of date; wait for background refresh or run `/index` |
| **syncing** | A background refresh is in progress |
| **missing** | No index exists yet; run `/index` |
| **error** | The index failed to build or sync; `/status` shows the details |

The project index is stored in `.code-symbol-index/index.sqlite`. It covers
Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, and more.
