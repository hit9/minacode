# Tools

You don't run tools yourself — you describe a goal and nanocode does the work. It helps to
know what it's capable of.

- **Read and search** files across your project. Search skips hidden, binary, and gitignored
  files.
- **Navigate code** — jump to definitions, callers, and implementations using the
  [code symbol index](#code-symbol-index). Build it with `/index`.
- **Edit files** safely — <span class="marked">before applying a change it checks the file hasn't changed
  underneath, so it won't patch the wrong place.</span>
- **Run commands**, including long-running ones in the background (list them with `/ps`).
- **Keep working notes** — a goal, a plan, and facts it has learned — so it stays on track
  through a long task.
- **Ask you** when a decision is genuinely yours to make.

## Built-in tools

These are the tools nanocode exposes to the model. They are separate from the `/` commands
you type at the prompt. `Skill` appears only when skills are installed; `MCP` appears after
a connected server exposes tools or resources.

| Tool | Purpose | Confirmation |
|---|---|---|
| `Read` | Read anchored line ranges from UTF-8 files | Only outside the project |
| `Search` | Search text with regular expressions, skipping hidden, binary, and gitignored files | Only outside the project |
| `InspectCode` | Find symbols, references, implementations, callers, callees, and file outlines through the code index | No |
| `Edit` | Create or patch one UTF-8 file using anchored edits | Yes |
| `Bash` | <span class="marked">Run one shell command with live output</span> | Only when the command is not conservatively classified as read-only |
| `Job` | Start, inspect, wait for, list, or stop background shell jobs | Start, wait, and stop only |
| `Recall` | Retrieve complete stored tool output, optionally by line range | No |
| `Note` | Maintain the agent's durable goal, plan, checks, and learned facts | No |
| `Ask` | Pause and ask you one or more questions | No |
| `Skill` | Load an installed skill's full instructions | No |
| `MCP` | Describe or call connected MCP tools and list or read their resources | Calls prompt unless the server marks the tool read-only |
## Execution behavior

Read-only calls may run concurrently, up to `runtime.max_parallel_tools`. File edits run in
order and verify their anchors before changing content. Large tool results are stored outside
the conversation; the model receives a bounded result and can retrieve the full output with
`Recall`.

## Confirmations

Before anything that changes your system — editing a file, running a mutating shell command,
or calling an external [MCP](mcp.md) tool not marked read-only — nanocode asks you to
confirm. Read-only actions, including safe shell commands, never prompt.

Turn confirmations off once you trust a workflow with `--yolo` (at startup) or `/yolo`
(in a session). See [Safety](safety.md) before doing so.

## Code symbol index

<span class="marked">nanocode ships with a pre-built **code symbol index** that lets the agent navigate
your codebase</span> — jump to definitions, find callers, list implementations — without
relying on slow, real-time file scans or external language servers.

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

```{note}
<span class="marked">Without the index `InspectCode` returns nothing.</span> Run `/index` once after opening a
project to build it.
```

### Building and syncing

Run `/index` to build or rebuild the index. The first build walks every source file;
subsequent builds sync from the previous snapshot and are much faster. Add `force`
to rebuild from scratch.

When the index already exists, nanocode refreshes it automatically in the background
at startup and after source-file edits. The `/status` command shows the current
index state:

| State | Meaning |
|---|---|
| **available** | Index is current and ready |
| **stale** | Out of date; wait for background refresh or run `/index` |
| **syncing** | A background refresh is in progress |
| **missing** (or **error**) | No index exists yet; run `/index` |

The index lives in `.nanocode/code-index/` as a set of sqlite databases. It covers
Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, and more.
