# What the agent can do

You don't run tools yourself — you describe a goal and nanocode does the work. It helps to
know what it's capable of. See [Tool reference](tool-reference.md) for the exact model-facing
tool names and confirmation behavior.

- **Read and search** files across your project. Search skips hidden, binary, and gitignored
  files.
- **Navigate code** — jump to definitions, callers, and implementations using a built-in
  code index. Rebuild it any time with `/index`.
- **Edit files** safely — before applying a change it checks the file hasn't changed
  underneath, so it won't patch the wrong place.
- **Run commands**, including long-running ones in the background (list them with `/ps`).
- **Keep working notes** — a goal, a plan, and facts it has learned — so it stays on track
  through a long task.
- **Ask you** when a decision is genuinely yours to make.

## Confirmations

Before anything that changes your system — editing a file, running a mutating shell command,
or calling an external [MCP](mcp.md) tool not marked read-only — nanocode asks you to
confirm. Read-only actions, including safe shell commands, never prompt.

Turn confirmations off once you trust a workflow with `--yolo` (at startup) or `/yolo`
(in a session).
