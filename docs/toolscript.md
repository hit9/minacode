# ToolScript

`ToolScript` runs a Python script whose `call()` invokes minacode's own tools. It is for the case
where the agent would otherwise emit the same call several times over with different arguments:
only what the script prints returns to the conversation, so a batch of calls collapses into a
single result.

It is always available — built-in tools are scriptable on their own, and the MCP entries below
additionally need a connected server.

## When it is worth it

Reach for a script at <span class="marker">four or more same-shape calls whose individual results
you do not need</span> — the script keeps them out of the context and returns the summary it
prints. Below that, or when each result has to be read, or when a step needs the model's own
judgment, plain batching is better: a script runs to the end on its own, with no model in the loop.


## Describing shapes in bulk

`ToolScript(action="describe", tools=[...])` batch-reports each tool's call shape and a json
gate. A list entry is either a built-in tool name like `"Read"` or an MCP tool as
`"server.tool"`:

- `json: yes` — the server declared an `outputSchema`, so `format="json"` below can rely on it.
- `json: unknown` — no declared schema; `format="json"` will try to parse the returned text and
  may fail.
- `json: no` — built-in tools: scriptable with `format="text"`, but no structured return yet.

The MCP rendering matches `MCP(action="describe")` with the gate line added; built-in entries
are compact one-liners with their parameter essentials. Either way, a shape learned here is what
a scripted call will actually receive.

## Running a script

`ToolScript(action="call", code="...")` runs the code as Python. Inside it, `call()` invokes a
tool — built-in or MCP:

```python
rows = []
for lang in ("zh", "en", "ja"):
    d = call("MCP", {"server": "orionTemplate", "tool": "get_template_detail",
                     "arguments": {"template_key": KEY, "language": lang}},
             format="json")
    rows.append((lang, d["title"]))
print(rows)  # only this line returns to the conversation
```

- `call(name, args, format="text")` — `name` is a built-in tool like `"Read"`, `"Bash"`, or
  `"Edit"`, or `"MCP"` for a server tool. `Delegate`, `Job`, and `ToolScript` itself cannot be
  nested. `format="text"` returns the rendered result; MCP calls may use `format="json"` for a
  parsed dict, while built-ins refuse it until they gain structured results
  (`<name> does not support format="json"`).
- Nested calls go through the <span class="marker">same confirmation, live logging, and `tr.N`
  retention as top-level calls</span>, and they never add extra tool messages — the agent still
  sees exactly one result per call it emitted.
- The result envelope lists the nested `tr.N` keys, so a shortened output can still be recalled.
- Refusing a nested call aborts the script; the rest of the outer batch continues.
- A script that fails returns with a traceback (source lines included) so the agent can correct
  itself.
- Pure script execution is budgeted at about 60 seconds, excluding time spent inside nested
  calls.

## The json gate in practice

Built-in tools report `json: no` and take `format="text"` only: `format="json"` refuses with
`<name> does not support format="json"`. For MCP calls, `format="json"` prefers the server's
declared `structuredContent`. Two errors are possible:

- The tool declared an `outputSchema` but the call returned no `structuredContent` — reported as
  `server declared outputSchema but no structuredContent`.
- No schema was declared and the returned text is not JSON — reported as
  `MCP returned text that is not JSON`.

## Safety

The script is <span class="marker">not sandboxed</span>: it runs with the same privileges as
`Bash`. `ToolScript` asks for confirmation on every `action="call"` and the prompt shows the
opening lines of the script, syntax-highlighted; `--yolo` skips that confirmation exactly as it
does for `Bash`.

## Reading the log

A script's calls are logged indented beneath it on a `│` rail that runs unbroken from the script,
through every call it made — including whatever each of those logged below itself — down to the
result line. The batch therefore reads as work the script did, rather than as calls the model made
itself.

<div class="term-shot" role="img" aria-label="A ToolScript call: its numbered script excerpt, then the two calls the script made indented beneath it on a continuous vertical rail, then the result line reporting the call count and what the script printed."><span class="fs-tool">  ToolScript  call 10 lines (334 chars)</span><span class="fs-dim">    ├ script</span><span class="fs-output">    │  1  path = "demo.txt"</span><span class="fs-output">    │  2  call("Edit", {"path": path, "edits": [{"op": "create", "content": "line 1\n"}]})</span><span class="fs-output">    │  3  out = call("Bash", {"command": f"wc -l &lt; {path}"})</span><span class="fs-output">    │  4  print(f"{path}: {out.strip()} lines")</span><span class="fs-dim">    │ … +6 more lines · Ctrl-O for more</span><span class="fs-tool">    │ Edit demo.txt</span><span class="fs-dim">    │ ├ preview</span><span class="fs-add">    │ │         1 | +line 1</span><span class="fs-dim">    │ └ stored tr.1 [approved]</span><span class="fs-tool">    │ Bash wc -l &lt; demo.txt</span><span class="fs-dim">    │ ├ output · 0.0s Ctrl-O for more</span><span class="fs-output">    │ │   1</span><span class="fs-dim">    │ └ stored tr.2 [approved]</span><span class="fs-dim">    ├ calls 2 · 1.4s Ctrl-O for more</span><span class="fs-output">    │ demo.txt: 1 lines</span><span class="fs-dim">    └ stored tr.3 [approved]</span></div>

The result line reports how many nested calls ran and the first lines of what the script printed;
a script that failed says so there and carries the error.

## Reading the script

The log shows a bounded excerpt of the script, not all of it — a long script would bury everything
it then goes on to do. The whole body is one keypress away:

- **`v` at the confirmation prompt** opens a read-only, scrolling viewer with the complete script,
  numbered and highlighted. `Esc`/`q` returns to the prompt; nothing is approved by viewing.
- **`Ctrl-O` afterwards** lists recent `Bash` and `ToolScript` calls; selecting a script opens the
  same viewer, with the result it returned below the source — the printed output, or the whole
  traceback when the script failed. This is how a script is read under `--yolo`, where no prompt
  ever stops to offer `v`. A very large result is bounded there (head and tail, long lines
  clipped) and the header says when it was.

Line numbers in the viewer match the ones in a failed script's traceback
(`File "<toolscript>", line N`).

<div class="term-shot" role="img" aria-label="The read-only script viewer: header rows naming the result key and line count, the numbered script, then a labeled result rule with the traceback below it."><span class="fs-divider">  Script · tr.3 · read-only</span><span class="fs-goal">  key    tr.3</span><span class="fs-goal">  lines  4</span><span class="fs-goal">  calls  2</span><span class="fs-divider">  ──────────────────────────────────────────</span><span class="fs-output">  1  rows = []</span><span class="fs-output">  2  for p in ("a.py", "b.py"):</span><span class="fs-output">  3      rows.append(call("Read", {"path": p}))</span><span class="fs-output">  4  print(rows[2])</span><span class="fs-divider">  ── result ─────────────────────────────────</span><span class="fs-dim">  ToolScript failed</span><span class="fs-dim">  calls: 2 [tr.1-2]</span><span class="fs-dim">  error:</span><span class="fs-dim">    File "&lt;toolscript&gt;", line 4, in &lt;module&gt;</span><span class="fs-dim">  IndexError: list index out of range</span><span class="fs-dim">  ↑/↓ scroll · g/G top/bottom · Esc/q close</span></div>
