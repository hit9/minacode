# Tool reference

These are the tools nanocode exposes to the model. They are separate from the `/` commands
you type at the prompt. `Skill` appears only when skills are installed; `MCP` appears after
a connected server exposes tools or resources.

## Built-in tools

| Tool | Purpose | Confirmation |
|---|---|---|
| `Read` | Read anchored line ranges from UTF-8 files | Only outside the project |
| `Search` | Search text with regular expressions, skipping hidden, binary, and gitignored files | Only outside the project |
| `InspectCode` | Find symbols, references, implementations, callers, callees, and file outlines through the code index | No |
| `Edit` | Create or patch one UTF-8 file using anchored edits | Yes |
| `Bash` | Run one shell command with live output | Only when the command is not conservatively classified as read-only |
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

Use `--yolo` or `/yolo` to skip confirmation prompts. See [Safety](safety.md) before doing so.
