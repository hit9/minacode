# Code symbol index

nanocode ships with a pre-built **code symbol index** that lets the agent navigate
your codebase — jump to definitions, find callers, list implementations — without
relying on slow, real-time file scans or external language servers.

## What it is

The index is a static database of symbols (functions, classes, methods, variables,
etc.) extracted from your project's source files. It is built by a library called
[code-symbol-index](https://github.com/hit9/code-symbol-index), which supports a
broad set of languages and is the same engine that powers nanocode's `InspectCode`
tool.

When the index is available, the agent can:

- **Find symbols** by name with fuzzy matching (`InspectCode mode="find"`)
- **Inspect a symbol** — show its definition and members (`InspectCode mode="inspect"`)
- **List references** — all call, read, write, and type references across the project (`InspectCode mode="refs"`)
- **Walk call chains** — transitive callers and callees (`InspectCode mode="callers"` / `"callees"`)
- **File outlines** — symbol tree of a single file (`InspectCode mode="outline"`)

```{note}
Without the index `InspectCode` returns nothing, so the agent loses code-navigation
ability. Run `/index` once after opening a project to build it.
```

## Building and syncing

### `/index`

Run `/index` at the prompt to build or rebuild the index for the current directory.
The first build walks every source file and writes a local database; subsequent
builds sync from the previous snapshot, so they are much faster.

| Usage | Effect |
|---|---|
| `/index` | Build or sync the index |
| `/index force` | Force a full rebuild from scratch |

### Auto-refresh

When the index already exists, nanocode refreshes it automatically in the background
at startup and after tool edits that change source files. You don't need to keep
running `/index` by hand — it stays current on its own.

The index in `/status` shows one of four states:

| State | Meaning |
|---|---|
| **available** | Index is current and ready |
| **stale** | Out of date; wait for the background refresh or run `/index` |
| **syncing** | A background refresh is in progress |
| **missing** (or **error**) | No index exists yet; run `/index` |

## Under the hood

The index lives in `.nanocode/code-index/` and is a set of sqlite databases tracking
symbol names, locations, and relationships. It covers Python, JavaScript, TypeScript,
Go, Rust, C, C++, Java, and more. See the
[code-symbol-index docs](https://github.com/hit9/code-symbol-index) for supported
languages and configuration options.
