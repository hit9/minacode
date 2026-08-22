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

    In `684:234ew`, `684` is the line number and `234ew` is a short hash of that line's content.
    The line number locates the edit; the hash proves that the line has not changed since it was
    read. Line numbers are one-based and ranges include both ends, matching what `grep -n`, your
    editor, tracebacks, and diffs show, so `Read`, `Search`, and `InspectCode` all agree on which
    line is which.
* - **`ViewImage`**
  - Opens one local PNG, JPEG, WebP, or single-frame GIF. The active model reads it when it
    accepts images; on a text-only route a configured [vision model](configuration.md#vision-model)
    returns its text observation instead. Images outside the workspace require confirmation.
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
    <span class="marker">refuses the edit if the hash no longer matches</span>. A successful edit
    also refunds the new anchors for the region it changed, so consecutive edits to the same file
    keep going without re-reading it first. Successful edits
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
    loaded only on demand instead of occupying every request. A key older than the
    [retained window](context.md#compaction) says so.
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
    selectable chips at the idle prompt, flowing left to right with up to three per line and
    wrapping when the terminal is too narrow so every suggestion stays visible; `Tab` cycles focus, `Enter` picks a chip into the
    input and returns to the prompt, so `Tab` to the next chip and
    `Enter` again combines several before sending. An
    all-`NextHints` batch ends the turn in a single model call.

    <div class="term-shot" role="img" aria-label="A terminal at the idle prompt after a NextHints call: the answer text above, then an empty prompt with a caret, a gap line, and one row of three suggestion chips separated by grey bars with the middle chip highlighted in reverse."><span>Everything is ready to review.</span><span> </span><span class="fs-prompt">&gt; <span class="fs-caret">▏</span></span><span> </span><span><span class="fs-i fs-sel"> run the tests </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-tab-on"> show the diff </span><span class="fs-i fs-dim"> │ </span><span class="fs-i fs-sel"> commit the work </span></span></div>

    `Tab` moves the highlight; `Enter` picks the focused chip into the input line and returns
    to the prompt, so `Tab` to the next chip and `Enter` again combines several; a final
    `Enter` sends. Focus a picked chip and press `Enter` to unpick it.
* - **`Skill`**
  - Loads an installed skill's full instructions when needed. It appears only when skills are
    installed; see [Skills](skills.md).
* - **`MCP`**
  - Describes or calls tools and reads resources from a connected MCP server. It appears only
    after a server is connected; see [MCP](mcp.md).
* - **`ToolScript`**
  - Runs a Python script the agent writes, so many tool calls happen in one call instead of
    one at a time. Only what the script prints returns to the conversation, which saves
    tokens; see [ToolScript](#toolscript).
* - **`Delegate`**
  - Hands a bounded task to a second in-process session (the worker) that runs on its own
    configured provider with a reduced tool set, keeping its context across delegations until
    reset. It appears only when [worker delegation](worker.md#worker-delegation) is
    enabled.
::::

(toolscript)=
## ToolScript

`ToolScript` is how the agent makes many tool calls at once: it writes one small Python
program, and inside it a `call()` is a tool call. Because the calls are code, the agent can
loop over a list, branch on a result, or feed one call's output into the next - then run the
whole thing as one call.

The payoff is tokens: the calls the script makes, and everything they print, never enter the
conversation - only the few lines it prints at the end come back. A run that would have taken
a dozen tool messages, each output filling the context, now costs one.

The log stays compact: the script's calls hang indented under it on one `│` rail down to the
result line, so the whole batch reads as work the script did - not calls the agent made one
by one.

<div class="term-shot" role="img" aria-label="A ToolScript call: its numbered script excerpt, then the two calls the script made indented beneath it on a continuous vertical rail, then the result line reporting the call count and what the script printed."><span class="fs-tool">  ToolScript  call 10 lines (334 chars)</span><span class="fs-dim">    ├ script</span><span class="fs-output">    │  1  <span style="color:#abb2bf;display:inline">path</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#56b6c2;display:inline">=</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">demo.txt</span><span style="color:#98c379;display:inline">"</span></span><span class="fs-output">    │  2  <span style="color:#abb2bf;display:inline">call</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">Edit</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">path</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">path</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">edits</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">[</span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">op</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">create</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">content</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">line 1</span><span style="color:#98c379;display:inline">\n</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">]</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-output">    │  3  <span style="color:#abb2bf;display:inline">out</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#56b6c2;display:inline">=</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">call</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">Bash</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">command</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">f</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">wc -l &lt; </span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">path</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-output">    │  4  <span style="color:#56b6c2;display:inline">print</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">f</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">path</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline">: </span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">out</span><span style="color:#56b6c2;display:inline">.</span><span style="color:#abb2bf;display:inline">strip</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#abb2bf;display:inline">)</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline"> lines</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-dim">    │ … +6 more lines · Ctrl-O for more</span><span class="fs-tool">    │ Edit demo.txt</span><span class="fs-dim">    │ ├ preview</span><span class="fs-add">    │ │         1 | +line 1</span><span class="fs-dim">    │ └ stored tr.1 [approved]</span><span class="fs-tool">    │ Bash wc -l &lt; demo.txt</span><span class="fs-dim">    │ ├ output · 0.0s Ctrl-O for more</span><span class="fs-output">    │ │   1</span><span class="fs-dim">    │ └ stored tr.2 [approved]</span><span class="fs-dim">    ├ calls 2 · 1.4s Ctrl-O for more</span><span class="fs-output">    │ demo.txt: 1 lines</span><span class="fs-dim">    └ stored tr.3 [approved]</span></div>

The log shows the first lines of a script, not all of it. To see the whole thing: pick
**View script** at the confirmation prompt (or type `v`), or press **`Ctrl-O`** after it ran
and pick the script from the list. Both open a read-only viewer with the full script and,
after the run, the result it returned. Under `--yolo` there is no confirmation prompt, so
`Ctrl-O` is the only way.

<div class="term-shot" role="img" aria-label="The read-only script viewer: the numbered script, then a labeled result rule with the script's printed output below it."><span class="fs-divider">  Script · tr.3 · read-only</span><span class="fs-divider">  ──────────────────────────────────────────</span><span class="fs-output">  1  <span style="color:#abb2bf;display:inline">path</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#56b6c2;display:inline">=</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">demo.txt</span><span style="color:#98c379;display:inline">"</span></span><span class="fs-output">  2  <span style="color:#abb2bf;display:inline">call</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">Edit</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">path</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">path</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">edits</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">[</span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">op</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">create</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">content</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">line 1</span><span style="color:#98c379;display:inline">\n</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">]</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-output">  3  <span style="color:#abb2bf;display:inline">out</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#56b6c2;display:inline">=</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">call</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">Bash</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">,</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#abb2bf;display:inline">{</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">command</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">:</span><span style="color:#abb2bf;display:inline"> </span><span style="color:#98c379;display:inline">f</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">wc -l &lt; </span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">path</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">}</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-output">  4  <span style="color:#56b6c2;display:inline">print</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#98c379;display:inline">f</span><span style="color:#98c379;display:inline">"</span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">path</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline">: </span><span style="color:#98c379;display:inline">{</span><span style="color:#abb2bf;display:inline">out</span><span style="color:#56b6c2;display:inline">.</span><span style="color:#abb2bf;display:inline">strip</span><span style="color:#abb2bf;display:inline">(</span><span style="color:#abb2bf;display:inline">)</span><span style="color:#98c379;display:inline">}</span><span style="color:#98c379;display:inline"> lines</span><span style="color:#98c379;display:inline">"</span><span style="color:#abb2bf;display:inline">)</span></span><span class="fs-divider">  ── result ─────────────────────────────────</span><span class="fs-output">  demo.txt: 1 lines</span><span class="fs-dim">  ↑/↓ scroll · g/G top/bottom · Esc/q close</span></div>

A script has the same privileges as `Bash` - there is no sandbox. Every run asks for
confirmation showing the script's opening lines; `--yolo` skips it, as with `Bash`.

(provider-side-tools)=
## Provider-side tools

Some providers can <span class="marker">search the web themselves</span>, as part of answering.
The model searches, reads the pages, and answers with sources, without minacode running anything.
This is off by default; turn it on with [`builtin_tools`](configuration.md#provider-side-tools).

You see a line for each search, `web search` as the working phase while it runs, and the pages it
read listed under the answer:

<div class="term-shot" role="img" aria-label="A provider-side search inside one answer: a green web search line naming the query, the working divider labelled web search, then the rendered answer with inline code, and a bold Sources heading above a numbered list of the pages the provider reported."><span><span class="fs-i fs-dim">  ├ </span><span class="fs-i fs-tool">web search</span><span class="fs-i"> httpx timeout configuration</span></span><span> </span><span><span class="fs-i fs-rule">--</span><span class="fs-i fs-glow">-</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-add">●</span><span class="fs-i fs-rule"> </span><span class="fs-i fs-working">web search (3s)</span><span class="fs-i fs-rule"> ------------------------------</span></span><span> </span><span>The client accepts a <span class="fs-i fs-md-code">timeout</span> argument taking either a float</span><span>or a <span class="fs-i fs-md-code">Timeout</span> object.</span><span> </span><span><span class="fs-i fs-md-b">Sources</span></span><span> </span><span>1. example.com/httpx/timeouts</span></div>

Sources appear when the provider reports them; not all of them do.

Unlike the tools above, a search is never confirmed — it happens inside the model's own reply, so
the only control is whether you enable it. What it reads is untrusted web text, and it makes the
turn larger than it would otherwise be. Leave it off when the agent runs unattended, or when the
questions themselves are sensitive.

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
