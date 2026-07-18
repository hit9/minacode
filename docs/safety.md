# Safety

```{admonition} Use at your own risk
:class: warning
nanocode edits files and runs shell commands in the environment where you start it. <span class="marked">It does
**not** sandbox itself.</span>
```

nanocode acts directly in your environment. Through its [tools](tools.md) it can read and edit
files in (and beyond) the working directory and run any shell command. There is no built-in
isolation, so treat it with the same care as running those commands yourself.

## Built-in guardrails

- **Confirmations.** File-changing and command-running tools — Edit, Bash, Job, and MCP
  calls — ask before they act. This is on by default; `--yolo` and `/yolo` turn it off.
- **Anchored edits.** Edits are tied to `line:hash` anchors and rejected if the file changed
  underneath, so the agent can't silently patch the wrong lines. See
  [Tools](tools.md).
- **Reviewable changes.** `/diff` shows exactly what changed this round and across the
  session before you rely on it.

## Reducing risk

- Work inside a **git repository** so every edit is reviewable and reversible.
- Keep **confirmations on** until you trust a workflow; only reach for `--yolo` when you do.
- Only connect [MCP](mcp.md) servers you trust — local servers run programs on your machine.
- For untrusted repositories, or when running unattended, put nanocode inside a **container
  or VM**.
