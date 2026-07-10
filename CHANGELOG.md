# Changelog


## Unreleased

### Added
- Add `/diff`, a read-only viewer for edits made by nanocode. It shows the latest user round and the overall net session diff in `Latest` and `Session` tabs, supports keyboard navigation and paging, and persists edit snapshots across `--resume`.
- Add a standalone animated progress row for manual `/compact` operations while keeping the normal status line visible.
- Show a bounded Bash stdout/stderr preview in finished tool summaries when no live preview was already kept visible.
- Automatically move Bash commands still running after five seconds to background jobs without restarting them, and return the job id and follow-up action to the model.

### Changed
- Replace blank Enter as the queued-input send-now shortcut with Ctrl-C. Enter queues a follow-up for the next model request; when follow-ups are queued, Ctrl-C interrupts the active request so they can be sent immediately. Queue guidance now appears as a contextual `+>` placeholder and disappears while typing.
- Remove the synthesized FILE STATE context projection. `Read` and `Edit` outputs now remain as bounded, recallable conversation messages, and `/context` shows only Environment and Memory.
- Show each Bash command once before execution, keep live output visible without repeating the command, and use a compact dimmed `stored tr.N` result marker after live runs.
- Render added and removed diff lines with subtle green/red backgrounds so Pygments foreground syntax colors remain readable and semantically separate from diff state.
- Render `/ps` as a Markdown table and print resume commands on their own line for easier copying.

### Fixed
- Prioritize queued follow-ups when building the next model request.
- Keep Bash preview colors consistent, preserve literal closing-tag text, show bounded output when no live frame is available, and avoid repeating commands or output around live runs and failures.
- Coalesce capped fallback edits into one `/diff` entry per file, count only real hunk changes, and report `No changes` when a round has no net effect.
- Reject directory paths passed to `Edit` with a tool error instead of crashing during edit planning.
- Persist the safe prefix of an active turn before each model request so completed messages and tool calls survive interruption and `--resume`.

## 0.9.1 - 2026-07-09

### Added
- Add immediate queued-input flushing while the agent is waiting on a model request. Pressing Enter in the `+>` queue still records text for the next LLM request; pressing blank Enter with newly queued text interrupts the active model request and retries it with that queued input included.

### Fixed
- Harden queued-input flushing so stale SIGINT delivery cannot cancel the whole turn after a model request already returned.
- Avoid needless duplicate retries when the active model request already contains the queued input, prevent repeated blank Enter from stacking retry counts, and clear whitespace-only queue input on Enter.

## 0.9.0 - 2026-07-07

### Added
- Syntax-highlight inline Edit diff previews with Pygments. Added and context lines (the "new" file version) are lexed together as one code block so multiline strings and indentation-sensitive languages highlight correctly; removed lines stay plain diff-red. Degrades gracefully to plain diff coloring when Pygments is unavailable, the file extension is unknown, or the lexer fails.
- Add background jobs via a `Job` tool (`start`/`status`/`wait`/`list`/`kill`). Long-running or non-blocking work — dev servers, watchers, long builds and test suites — runs in its own process group without blocking the agent; output is buffered (capped per stream), drained non-blockingly on `status`, and returned as a tail. Concurrency is capped (`MAX_JOBS`), running jobs surface in the status bar (`jobs N`), the `/status` row, and a new read-only `/ps` command; `kill` sends SIGTERM then SIGKILL to the group. Jobs run until they exit or are killed (no foreground `shell_timeout` cap), and `Job(action="wait")` with no/zero timeout blocks until the process exits.

### Changed

- Rename the interactive `Question` tool to `Ask`; the old `Question` tool name is no longer registered.
- Replace the status-bar `+N` queued-message counter with a live queue region. While the agent works, messages you type sit below the +> input under a dim "── queued" divider with a left→right sweep animation, instead of being echoed into the scrollback log; when the turn flushes them they move up into the log as `+ <text>` lines and the region shrinks (the divider disappears once nothing is queued).
- Cache the Anthropic request's stable `tools`+`system` prefix with a `cache_control` breakpoint on the system block, so each turn no longer reprocesses the system prompt and tool schemas from scratch (Anthropic prompt caching only activates at an explicit breakpoint).
- Strengthen the system prompt for faster convergence: batch independent reads/searches into one parallel request by default, and drive each Bash call to complete in a single pass (chain known steps, split only on genuinely unpredictable dependencies).
- Replace the dedicated `LineCount`, `List`, `Find`, and `Git` tools with Bash-driven equivalents. The model now sees available shell commands near the top of Environment, read-only Bash commands (including safe `git status`/`diff`/`log` style commands) auto-run without confirmation, and mutating shell/git commands still require approval.
- Bash output is no longer erased from the terminal after the command finishes; the live preview output stays in the scrollback history.
- Edit diff previews and approve messages remain visible in the CLI history instead of being transiently cleared.
- Remove the Ctrl-A expanded Edit preview and fixed-height transient preview window. Edit approvals still show the full inline diff preview in the CLI history, with Pygments highlighting preserved.
- Git branch is no longer shown in the environment context sent to the model.
- Removed branch-change detection protection that prevented `git commit` after an external branch switch.

### Fixed
- Close a hole in Bash read-only auto-approval: `&&`/`||` were flattened to spaces and only the first command in a chain was validated, so `git log && rm -rf x` auto-ran without confirmation. Every stage of a `&& || | ; newline` chain is now validated independently, a lone background `&` is rejected, and a `cd` prefix (a benign builtin the model routinely adds) no longer forces a prompt on an otherwise read-only command.
- Follow common sense in the Bash read-only classifier rather than blanket strictness: `sort`, `uniq`, `sed`, and `tree` auto-run (with guards only on their real file-writing forms — `sort -o`, `uniq IN OUT`, `sed -i`, `tree -o`), and the ubiquitous harmless redirections (`2>/dev/null`, `>/dev/null`, `< /dev/null`, `2>&1`, `>&2`) no longer trigger a prompt. Real writes to a file and command substitution still ask.
- Suppress prompt-toolkit's CPR warning in terminals that do not answer cursor position requests by disabling CPR probing for nanocode's prompt applications.
- Stop closed stdin from escaping as an unhandled exception in the queue-input background thread.
- Stable Edit anchors, eliminating spurious "stale anchor" errors. `line_hash` no longer folds the trailing newline into the hash, so a line's anchor stays valid when only its final newline changes (e.g. the last line gaining or losing its `\n`) and Read/Search/Edit anchors now agree with InspectCode's. Edit also numbers lines exactly like Read — both split on `\n` only via a shared `ReadTool.split_lines`, replacing `str.splitlines(True)`, which additionally breaks on `\r`, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, `\u2028`, `\u2029` and desynced line numbering (and thus anchors) for any file containing those characters.
- Bump `code-symbol-index` to `>=0.3.5`, fixing a tree-sitter range extraction crash that could terminate `nanocode --resume last` with SIGSEGV and leave a core dump during background index refresh, while using the package's validated tree-sitter dependency floor.
- Render Ask choice previews with escaped `\n` sequences expanded into real lines, allow the choice window to use available vertical space above the status bar, and stop repeating the full raw question text when switching to "Type freely...".
- Render intermediate assistant markdown correctly while the `+>` queue prompt is active; Rich ANSI output is now replayed through prompt-toolkit instead of leaking as `?[90m`-style text.
- Move the active-turn elapsed timer into the animated `working` divider; the sweep now scans only the horizontal rule so the label stays readable.
- Show saved tool summaries on `--resume` even when the compacted transcript no longer contains the original assistant tool-call messages.
- Add regression coverage for Ask preview/free-text rendering, whole-second working timers, and restored saved tool summaries.

## 0.8.2 - 2026-07-03

### Fixed
- Auto-submit queued input you already pressed Enter on. While the agent is working, typing in the `+>` queue and pressing Enter records the line into `pending_user_inputs`; if the turn ended before a step consumed it, that already-entered text was pre-filled back into the prompt and sat there waiting for a *second* Enter (and could feel like a very delayed send). Now, in interactive mode, Enter-committed queue input auto-submits as the next turn with no second keypress, while text you were still typing (never Enter'd) is still pre-filled into the prompt for review. Mid-turn injection and the headless combined-submit path are unchanged.

## 0.8.1 - 2026-07-02

### Added
- Add a `nanocode update` (alias `upgrade`) command that upgrades nanocode to the latest PyPI release. It first checks the latest version, reports and exits when already current, then detects how nanocode was installed and runs the matching upgrade command: `uv tool upgrade nanocode-cli` for uv-tool installs, `pipx upgrade nanocode-cli` for pipx, and `python -m pip install --upgrade nanocode-cli` otherwise. Editable/dev installs are detected (via the distribution's `direct_url.json`) and refused with a hint to update the source instead.

## 0.8.0 - 2026-07-01

### Added
- Add Skills: reusable instruction packs discovered from `.nanocode/skills/<name>/SKILL.md` (project) and `~/.nanocode/skills/<name>/SKILL.md` (user; project wins on name clash). Each skill is a Markdown file with `name`/`description` frontmatter and may bundle helper scripts in its folder. A compact `SKILLS` index (name + description) rides the cache-stable request prefix so the model knows what exists, and the full body is pulled into the conversation only on demand — either when the model calls the new `Skill(name)` tool or when the user references a skill inline with `$name` (Tab-completed). A `{skill_dir}` / `${SKILL_DIR}` placeholder in the body expands to the skill's absolute folder path when loaded, so bundled scripts are runnable via `Bash` without hardcoded paths. Repeated `Skill(name)` loads collapse to a one-line pointer (like MCP `describe`), keeping the first full body cached and not re-billing the instructions. A built-in `nanocode-help` skill ships by default (out of the box, no install) that answers questions about nanocode itself — how to use it, its features, and common problems — from a self-contained manual, so it does not read the source per question. The body is an authored manual (concepts, workflows, troubleshooting) plus lists assembled at load time from the same in-code constants the app uses (the `/help` text, each tool's description, the settable config keys), so the prose is rich while the lists cannot drift from the running version; the raw source is named only as a last-resort fallback. A project/user skill of the same name overrides it. Because a built-in is always present, the `SKILLS` section and `Skill` tool ride every request; only a session with its skills explicitly cleared drops back to the prior byte-identical prefix. Installed skills are surfaced in the status bar (`skills N`), the `/status` context row (`skills N`), a read-only `/skills` command, and `/help` now documents the `$skill` inline mention.
- Replace `/memory` with `/context`, a viewer for the model's synthesized context frame — the Environment, Memory (goal, plan, known facts, check, code-index status), and File State sections that wrap the conversation on every request. At an interactive prompt, bare `/context` opens a tabbed viewer (`←/→` or `h/l` switch Environment/Memory/File State, `↑/↓` or `j/k` scroll, `1`–`3` jump, `Esc`/`q` close) with the sections rendered as Markdown tables/headings and a scroll-position indicator; while the agent is working or without a TTY it prints the same content as a static Markdown dump. `/context <path>` shows one in-context file's current anchored lines (matched by exact path, basename, or suffix) inside a fenced block. Unlike the old `/memory`, the Memory section now renders exactly as the model receives it (so it includes `check` and code-index status and omits the transcript-only summary).
- Detect prompt-prefix drift via fingerprinting. The cache-stable request prefix (system prompt + environment + MCP tool index + sorted tool schemas) is fingerprinted (SHA-256) each turn against a baseline pinned to the first one seen; a healthy session keeps a single fingerprint start to finish, while a second one means every token from the change onward is a cache miss that previously failed silently and only surfaced on the bill. `/status` shows a `⚠ prefix churn` warning with the distinct-fingerprint count when more than one is seen, `--debug` writes a unified diff of exactly what changed (`cache-prefix-drift`), and the fingerprints persist across `--resume`.

### Changed
- Track and show the context compaction count in `/status`. `AgentState` gains a persisted `compaction_count` incremented on every compaction path (history, turn, and their fallbacks), surfaced in the `/status` context row and preserved across session save/load.

## 0.7.2 - 2026-06-26

### Added
- Show a short, context-aware tip on startup (a curated set covering sessions, context, model/reasoning, tools, and config). Tips whose feature is unavailable are filtered out (e.g. `/strict` only shows when the provider supports strict tools; MCP tips only when MCP is configured). The line is styled as a muted hint with highlighted `code` spans and can be disabled with `runtime.tips = false` (or `/set runtime.tips off`).
- Add per-property `description` fields to every built-in tool's JSON Schema (`Read`, `LineCount`, `List`, `Find`, `Search`, `InspectCode`, `Edit`, `Bash`, `Git`, `Recall`, `Note`). The argument contract now lives in the schema the model parses natively rather than only in the prose signature, improving tool-call argument accuracy across providers.
- Add a `/strict` command to toggle `provider.strict_tools` for the active provider, reporting when it is enabled but inactive (the provider does not support strict tool calling).
- Add a `provider.strict_tools` option (default `false`, also settable via `/set`) that constrains tool-call arguments to each tool's JSON Schema. When enabled it is only active on hosts whose profile supports strict mode (`api.openai.com`, `api.deepseek.com`) and on the chat path; every other provider is unaffected. Activating it emits a strict-compliant schema (all properties required, genuine optionals made nullable, `additionalProperties: false`, and the unsupported `minItems`/`maxItems`/`minLength`/`maxLength` keywords dropped) with `strict: true` per function, and on DeepSeek it routes the client to the `/beta` endpoint where the feature lives. Because optionals become nullable, the model may send explicit `null` for an omitted argument; nanocode now strips `null` values (recursively) before dispatching to a tool, where `null` has always meant "absent".

### Changed
- Add a `provider.max_tokens` option (also settable via `/set`) capping chat-completion output. It defaults to `0` (unset) for generic OpenAI-compatible providers, so their requests are unchanged, and resolves to a profile default of `32768` on `api.deepseek.com` so DeepSeek thinking mode has enough room for `reasoning_content` plus the answer.
- Map `reasoning = "high"` to DeepSeek's agent-recommended `max` effort in the thinking effort table (was `high`); `xhigh` still maps to `max` and the default `medium` still maps to `high`. Only the DeepSeek/Qwen `thinking` style is affected.
- Skip the OpenAI-only `prompt_cache_key` parameter for `api.deepseek.com`, which caches by prefix automatically and ignores the key. All other hosts keep emitting it as before.

### Fixed
- Keep the Bash live preview alive during blocking commands so the terminal no longer looks frozen. A command that buffers its output (e.g. a quiet long-runner, or `... | tail` that emits nothing until EOF) previously left the screen completely static — the status bar is stopped while a command runs and the preview only drew when output arrived. The preview now renders an immediate frame showing the command being executed (`$ <command>`) and a live elapsed timer (`running… 12.3s`), ticked by a daemon heartbeat so the timer advances even with no output, and switches to `output · 12.3s` once output streams. On finish the frame is now erased (it previously lingered as dimmed ghost lines), and a full-width line can no longer auto-wrap and desync the redraw cursor math.
- Set an explicit output ceiling for DeepSeek thinking mode so long `reasoning_content` no longer exhausts the server-side default and truncates the response or drops the tool call.
- Stop sending `temperature` on the chat path when a native thinking style (`thinking`/`enable_thinking`) is enabled, since DeepSeek and Qwen reject or ignore it in that mode. Other reasoning styles and providers are untouched.
- Drain the MCP background event loop on exit. MCP work runs on a daemon loop thread; at interpreter shutdown the `concurrent.futures` atexit hook tore down the default executors before that thread was joined, so an in-flight client cleanup (HTTP session termination, DNS via `run_in_executor`) raced the teardown and printed `cannot schedule new futures after shutdown` / `Session termination failed`. `MCPManager.close()` now cancels pending tasks, stops the loop, and joins the thread (5s timeout) from `main()`'s `finally` before the interpreter tears down its executors.

## 0.7.1 - 2026-06-25

### Changed
- Run the code-symbol-index working-tree check off the UI critical path. `/status` previously forced a synchronous `csi.status(check=True)` full-tree scan (~1.5s on large repos) before rendering, and the same scan ran inline at the end of every turn, delaying the answer. `/status` now reads the last-known status instantly (`check=False`) and triggers the scan asynchronously, and the per-turn `update_pending()` runs in a guarded one-shot daemon thread, so neither blocks the UI. A new transient `code_index_checking` guard prevents overlapping scans.

## 0.7.0 - 2026-06-24

### Added
- Execute a model's batch of read-only tool calls concurrently. Within one assistant turn, maximal runs of auto-approved, non-interactive read tools (`Read`, `Search`, `Find`, `List`, `Recall`, `InspectCode`, read-only `Git`, read-only `MCP`) now run in a bounded thread pool, while results are finalized on the main thread in the exact order the model issued them. Mutating tools, `Edit` batches, confirmations, `Bash` (live output), and `Question` (interactive) stay serial. Bounded by the new `runtime.max_parallel_tools` setting (default `4`; set to `1` to disable and restore fully serial execution).
- Store `Note` plan items as explicit objects with `status` (`todo`, `doing`, `done`, `blocked`) and `text`, while preserving old string plans as `todo` during load/compaction. Model-visible memory renders readable status text; CLI memory and Note previews keep compact `[ ]`, `[~]`, `[x]`, `[-]` symbols.
- Added a concise system-prompt `GUIDE` section covering `THINK BEFORE CODING`, `SIMPLICITY FIRST`, `SURGICAL CHANGES`, and `GOAL-DRIVEN EXECUTION`.

### Changed
- Stop auto-approved (yolo) tool calls from logging two near-identical CLI lines. The pre-execution `auto …` line duplicated the result line's header for tools without a preview (MCP, Bash, Git); it is now suppressed in that case and the auto-approval is recorded by an `[auto]` tag on the single result line. `Edit` still shows its `auto …` pre-line because it carries the diff preview the result line omits.
- Depend on `fastmcp-slim[client]` instead of the full `fastmcp` meta-package, since only the MCP client is used. This drops the unused server/CLI stack (cyclopts, griffelib, uvicorn-side bits, openapi/jsonschema tooling, the keyring/py-key-value cluster, etc.) for a much smaller install.
- Switched Read/Search/Edit/InspectCode anchors to the explicit `anchor=line:hash | text` format and documented the hash as `hash(line_content)` in FILE STATE.
- Bumped the `code-symbol-index` dependency floor to `>=0.3.1` and use its upstream explicit InspectCode anchor formatter instead of local normalization.
- Reduced `nanocode.py` by removing dead code, thin wrappers, and over-complex helper paths without changing behavior.
- Render Note success output with user-facing `goal:`, `check:`, `plan:`, and `known:` labels, plus status-aware colors in the CLI, instead of exposing internal field names like `set_goal` and `replace_plan`.

### Fixed
- Keep every MCP server visible in the tools index. It previously built full per-tool schemas and then hard-truncated the whole block at the size cap, so one verbose server could consume the entire budget and silently drop every later server, leaving the model blind to them. The index now degrades by shedding detail rather than entities — inline schemas, then schemas-dropped, then name-only — keeping all servers and tool names visible (the model can re-fetch a schema via `describe`); only at thousands of tools does it truncate, and that case now warns via a post-discovery notice and a `!` status-line marker.
- Declare `websockets` as a direct dependency. The MCP client transport imports it at runtime, but `fastmcp` only declared it under its server extra, so trimming that extra broke MCP connections (including OAuth login).
- Render structured plan items in `/memory` and Note previews as status symbols/text instead of leaking `PlanItem(...)` repr strings.
- Resume sessions whose transcript contains tool calls with multi-line argument strings (e.g. a multi-line `git commit` message). Strict `json.loads` rejected the literal newlines and dropped the args to `{}`, which then failed the tool's own validation (`Git requires a non-empty 'argv' list`) and aborted `--resume`. Tool-call arguments now parse with `strict=False`, and a malformed historical call renders without args instead of crashing.
- Stop malformed live tool-call arguments from aborting the whole turn. Argument validation that ran while parsing a model response (e.g. `Git` with an empty `argv`) raised outside the tool-execution boundary, ending the turn with an error the model never saw. The error is now captured on the call and surfaced as a normal failed tool result, so the model can self-correct.
- Decode `Bash` output with a per-stream incremental UTF-8 decoder. Each 4096-byte read was decoded independently, so a multibyte character split across a read boundary (common with CJK output) was mangled into replacement characters.

## 0.6.6 - 2026-06-23

### Changed
- Report the prompt-cache hit rate in `/status` against input/prompt tokens instead of total tokens (which folded in completion/reasoning output and structurally understated the rate), and show the `cached/prompt` split for both the cumulative and last-call figures.
- Land MCP `describe` results inline in the conversation history (bounded and recallable like any tool output) instead of stripping them into a recomputed `MCP TOOL DETAILS` tail block. The schema is now written once and cached as the conversation grows, rather than re-billed uncached on every request; the tools index tells the model not to describe a tool again once its schema is already shown.
- Collapse repeated MCP `describe` results for the same tool to a one-line pointer when building the request, keeping only the first full schema per `(server, tool)`. This is a stateless send-time transform that never mutates stored history and only ever collapses the newer occurrence, so it reclaims duplicate-schema tokens without disturbing the cached prefix.

- Tighten the final-answer style guidance: default to a few lines, scale length to the task, lead with the result, and drop preamble / request-restatement / step-by-step recaps, replacing the soft "be concise" wording that models tended to ignore.
- Add Tab/Shift-Tab completion (commands, paths, mentions) and a completion menu to the queue-input area, matching the main prompt.
- Run read-only slash commands (`/help`, `/status`, `/memory`, read-only `/mcp`) plus `/yolo` typed in the queue-input area while the agent is working, instead of sending them to the model as literal text. `/yolo` is allowed because it is a single atomic flag the agent reads at the next approval; other state-mutating or control commands are refused with a hint to interrupt first, since they would race the in-flight turn.
- Pre-fill leftover queued input into the prompt for review/edit when a turn finishes, instead of auto-submitting it as the next turn. Input typed while the agent is working is still injected into the running task; only input left over at the turn boundary now waits at an editable prompt (non-interactive/piped input keeps auto-submitting, since there is no one to confirm).

- Collapse argument/usage tool-call rejections to a quiet dim one-liner (`tool X · rejected: <reason>`) in non-debug mode, instead of the full red `[failed]` + error block. These are usually self-corrected on retry; the full error still goes to the model and shows in debug, and real execution failures stay fully visible.

### Fixed
- Moved the volatile `code_index` status out of the early `Environment` block (which sits ahead of the conversation history) into the late `Memory` section, so its `synced ↔ stale` churn no longer invalidates the cached conversation prefix every time files change or the indexer runs.

## 0.6.5 - 2026-06-22

### Added
- Surface each MCP tool's input JSON Schema to the model: the tools index now carries a compact (capped) schema per tool and `MCP(action="describe")` emits the full schema, so the model can build nested arguments correctly instead of guessing.
- Added MCP resource support: discover resources alongside tools (concurrently, best-effort), list them in the tools index, and read them via `MCP(action="list_resources")` and `MCP(action="read_resource", uri=...)`. Resource reads flow through the normal tool-result path so they land in cached conversation history.
- Surface resource-like URIs referenced in MCP tool descriptions on the tool's index line, so doc references survive description truncation and stay discoverable.
- List a server's resources in `@server` mention blocks, matching the tools index.
- Auto-read a resource referenced by an MCP tool's description on the first call to that tool (once per uri per session, best-effort, web links skipped), attaching it to the call result on success and to the error on failure so argument-grammar docs reach the model on the first attempt and land in cached history.

### Changed
- Default the `MCP` tool `action` to `"call"` when omitted but `tool`/`arguments` are present, and make the unknown-action error actionable (lists valid actions and shows how to invoke a remote tool by name).

### Fixed
- Render connected MCP servers that advertise resources but no tools in the tools index (previously gated on having ≥1 tool, so resource-only servers were mislabeled "not connected" and their resources hidden); `_pending_status` now distinguishes a connected-but-empty server from a disconnected one.

## 0.6.4 - 2026-06-21

### Added
- Added JSONL session persistence with append-only deltas for the normal save path and `nanocode --resume [UID]` for restoring saved sessions.
- Added a `latest` session pointer and a resume command hint printed when nanocode exits.
- Added `nanocode --resume last` as an alias for resuming the latest saved session.
- Render restored session history once on resume without re-running tools or commands.
- Show the active session id in `/status`.
- Added session persistence coverage for snapshot/delta save/load, `latest`, usage/state/tool record roundtrips, missing snapshots, and exit resume hints.

### Changed
- Kept persisted session snapshots focused on necessary recovery data only, deriving runtime tool-result lookup state from saved tool records instead of storing config, settings, timestamps, git branch, or other rebuildable runtime data.
- Skip first-save persistence for sessions with no recoverable content, avoiding empty snapshots and stale `latest` pointers.
- Delete session files older than the configurable `runtime.session_retention_days` retention window on startup; the default is `7` days.
- Treat retention and update-check interval settings as config-file policy rather than `/set` runtime mutations.
- Split session snapshot encoding/merging from JSONL file storage, keeping `Session` as a thin owner of runtime state.
- Resume now applies the current config/runtime flags (`--config`, `--yolo`, `--debug`, `--mcp`) while loading only conversation state from the snapshot.

### Fixed
- Fixed command dispatch so ordinary non-command input reaches the agent instead of being treated as an unknown slash command.
- Changed the resume marker to an internal system message so it does not affect latest-user compaction behavior.
- Report resume/load errors as CLI errors instead of uncaught exceptions.

- Auto-submit queued input at round end: `queue_input_text` (typed but not confirmed) and unconsumed `pending_user_inputs` both get drained and submitted as the next user input; extracted as `CommandLoop.drain_queued_input()`.
- Clear blank `queue_input_text` even when it contains only whitespace, preventing stale whitespace from lingering in the input area.

- Record auto-submitted queued input in CLI history via `FileHistory.append_string()`, so previously-queued text appears in prompt history like any manually typed input.

## 0.6.3 - 2026-06-21

### Added
- Added `Question` tool that pauses the agent to ask the user one or more questions (asked in sequence) when intent is genuinely ambiguous, a design choice affects the external shape (module structure, public API, naming), or prioritization is needed. Each question supports structured choices with optional dynamic previews and an optional `recommended` choice index (pre-selected and marked).
- Wired `Question` tool into the shared interactive selector with `j`/`k` navigation, dynamic per-choice preview, Rich markdown rendering of the question text, and a free-text fallback.
- Echo the selected choice in the CLI after the interactive selector closes, giving clear feedback on what was chosen.
- Prefix a position indicator (e.g. `(1/3)`) onto each question when several are asked in one `Question` call.

### Changed
- Refined `Question` tool DESCRIPTION, SYSTEM_PROMPT usage guidance, and README with clear principles: question only when intent is truly ambiguous, design affects external shape, or prioritization is needed. Skip internal details, contextually-determinable items, and already-specified matters.

### Fixed
- GitTool: validate non-empty `argv` in `payload_args` with a clear error message when the model sends an empty command list.
- Update the tracked commit-target branch (renamed `initial_git_branch` → `expected_git_branch`) after a tool-driven branch switch so commits on a branch nanocode switched to are not rejected, while still guarding against unexpected external switches.

## 0.6.2 - 2026-06-21

### Changed
- Simplified the agent system prompt while preserving the main `TOOLS`, `FLOW`, `FILE STATE`, and `FINAL` sections.
- Simplified MCP command handling and server config parsing by sharing command metadata and config field parsing helpers.

### Fixed
- Reused one manager-owned MCP event loop for async MCP operations, including calls made while another event loop is already running.
- Made MCP OAuth token storage reuse one store and shared path lock to avoid concurrent token file writes racing each other.
- Allowed `env_http_headers.Authorization` when it is the only configured authorization source.
- Rejected extra `/mcp` subcommand arguments instead of silently ignoring them.

## 0.6.1 - 2026-06-19

### Changed
- Renamed `Note` fields: `goal` → `set_goal`, `plan` → `replace_plan`, `known` → `append_known`, `check` → `set_check`; added `replace_known` for full replacement of known facts.
- Aligned `Note` schemas and prompt guidance so `replace_plan`, `append_known`, and `replace_known` are always arrays and can be empty when replacing state.
- Split FILE STATE into its own prompt section and clarified it as the working view for visible file content and Edit anchors, while noting it may be partial.
- MCP tool calls now confirm by default, auto-approving only tools explicitly marked read-only (`readOnlyHint`) or non-destructive (`destructiveHint: false`); undiscovered tools also require confirmation.

### Added
- New Note field `replace_known` that completely replaces the known list, with test coverage.

### Fixed
- Made `Note` updates transactional, so invalid fields no longer partially mutate session notes before returning a tool error.
- Guarded MCP `login`/discovery error paths with the manager lock so they no longer race background discovery while mutating server tool/error state.

## 0.6.0 - 2026-06-18

### Added
- Added MCP client router support through a single `MCP` tool, with URL-based server configuration, bearer-token environment variable support, OAuth login/logout with persistent tokens, asynchronous tool discovery, compact model-visible MCP tool indexes, on-demand tool details, and `/mcp` inspection/refresh commands.
- Added local stdio MCP servers via `command`/`args`/`env`, alongside the existing streamable-HTTP transport; each server is `url` or `command`, and stdio servers reject the HTTP-only auth options.
- Added `@server` and `@server.tool` mentions in user input that inject a server's tool list or a tool's details into the turn, force discovery of undiscovered servers, report login/errors inline, and tab-complete server and tool names.
- Added MCP coverage for result normalization, successful tool calls, context pruning, `/mcp tools NAME`, missing-server refresh handling, and stdio config parsing/validation.
- Added bounded MCP connection timeouts, concise MCP connection-failure logs, a `--debug` flag for starting with debug mode enabled, and a `--mcp` selector for choosing MCP servers by name glob.

### Changed
- Refined the status bar with lowercase `mcp`, right-side loading animation, and semantic per-section colors.
- Show MCP discovery progress in the status bar while servers are loading.
- Render `/mcp` server and tool listings as Markdown tables for clearer terminal display.
- Load configured MCP servers in parallel during discovery.
- Include the MCP endpoint URL when OAuth login fails before an authorization URL is available.
- Suppress duplicate FastMCP OAuth URL logs and omit OAuth-login-required notices from startup MCP error logs.
- Skip MCP server discovery without startup error logs when bearer-token environment variables are missing.
- List all enabled MCP servers in the model-visible index, adding a "not yet available" section for servers that are still discovering, need login, or errored, so the model never assumes a configured server is absent.
- Enriched MCP tool-call logging with compact `key=value` arguments in the header (also shown in the approval preview), a success result summary (shape and payload size), and round-trip latency.
- Steered the agent toward `InspectCode` for symbol navigation with a `SEARCH/NAV` prompt section, and surfaced the code-index status in the Environment context so it knows when the tool is usable.
- Separated each round with a blank line after the user input and a rule before the agent's answer.
- Clarified prompt guidance around FILE STATE snapshots, automatic Read/Edit refreshes, stale-anchor retries, and avoiding unchanged failed tool-call retries.
- Tightened the system prompt's tool-choice guidance to prefer `Edit` for file changes, `Read` for known file ranges, `Search` for text lookup, and `InspectCode` for symbol navigation.
- Added current git branch to Environment context while keeping branch-specific data out of the stable system prompt.

### Fixed
- Expanded `Edit` no-op errors with current target-range content when anchored edits produce no changes, so agents can distinguish already-applied edits from wrong replacement content.
- Guarded git branch safety so yolo mode cannot auto-approve branch-changing commands, and git commits refuse to run after the branch changes from the session start.

## 0.5.12 - 2026-06-15

### Fixed
- Preserved queued input typed during a running turn and prefilled it into the next prompt instead of dropping it at turn completion.

## 0.5.11 - 2026-06-12

### Added
- `InspectCode` gained `refs`, `impls`, `callers`, and `callees` modes, backed by `code-symbol-index` 0.3.0: behavior-classified references, implementor listing, and transitive call-chain walks (`depth`). `refs` hides import/attribute noise by default, with `ref_kind` to filter to an explicit subset or `all_kinds` to show everything; `callees` takes `loose` to include ambiguous cross-module matches; `refs`/`impls` page with `offset`.

### Changed
- Bumped `code-symbol-index` floor to `>=0.3.0`.

## 0.5.10 - 2026-06-02

### Added
- Cached gitignore patterns across tool calls with mtime-based invalidation.

### Changed
- Clarified Bash tool output limits in its description.
- Strengthened assistant language prompt to reduce unnecessary `cd` commands.
- Clarified Git tool description so the model sees it defaults to the cwd from Environment.

## 0.5.9 - 2026-06-01

### Changed
- Colorized `Ctrl-A` full edit previews when shown through `less`.

## 0.5.8 - 2026-06-01

### Added
- Added `Ctrl-A` full edit preview in an external pager during approval.

### Changed
- Compact oversized current turns instead of only prior history.
- Removed thin internal wrappers without changing behavior.

### Fixed
- Rejected broad `git add` commands unless explicit file paths are supplied.
- Stopped exposing the output-language sentinel inside model-visible file state.

## 0.5.7 - 2026-05-30

### Changed
- Encouraged early `Note` usage for multi-step work with goal and plan updates.
- Added lightweight empty-memory guidance when goal or plan has not been set.
- Ordered `FILE STATE` files by most recent visible Read/Edit source before stable path fallback.

## 0.5.6 - 2026-05-30

### Added
- Added a visible purple approval wait indicator.
- Expanded compact logic tests around latest-turn retention, recent-message windows, fallback trimming, and tool-result preservation.

### Changed
- Clarified prompt rules for user-visible interim output and final answers.
- Simplified core flow by removing thin wrappers and duplicate tool-schema name extraction.

### Fixed
- Preserved raw tool results referenced from compact summaries so `tr.N` keys remain recallable.
- Improved transient model error retry detection and final retry reporting.

## 0.5.5 - 2026-05-30

### Changed
- Tightened the system prompt around FILE STATE, anchored edits, and final-answer flow.

### Fixed
- Added limited automatic retries for transient model request failures such as 5xx, rate limits, and timeouts.

## 0.5.4 - 2026-05-29

### Changed
- Reworked running-turn context as a current turn conversation, preserving mid-turn assistant text and appended user input.
- Made running-turn appended input visible through a `+>` prompt and compact `+N` status indicator.

### Fixed
- Started Bash live preview before command output so the `+>` prompt cannot cover it.
- Avoided extra approval prompt line clearing after confirmation.

## 0.5.3 - 2026-05-29

### Added
- Added support for additional user input during running agent turns.
- Added multiline approval input for pasted refusal reasons.

### Changed
- Simplified approval handling so direct non-yes input is treated as a refusal reason.

### Fixed
- Fixed CreateFile escaped-newline handling so preview and written content stay multiline.
- Made CreateFile/Edit code-index updates use the tool call path as a fallback.

## 0.5.2 - 2026-05-29

### Changed
- Animated the statusbar code-index refresh indicator while keeping `/status` semantic.

### Fixed
- Switched startup code-index refresh to the `code-symbol-index` async refresh API to avoid parser thread ownership errors.

## 0.5.1 - 2026-05-29

### Added
- Added the current date to context immediately before the current user request.

### Fixed
- Sanitized context/debug/model-request text so surrogate characters from terminal input cannot break UTF-8 encoding.
- Made Search ignore hidden paths and `.gitignore` paths consistently across ripgrep and Python fallback paths.

## 0.5.0 - 2026-05-28

### Added
- Added cached system information to the top of model context: cwd, OS, arch, shell timeout, and detected commands.
- Added configurable `runtime.max_context_tokens`.
- Added key behavior tests for tools, agent loop, context management, provider adaptation, and code index integration.

### Changed
- Replaced the legacy implementation with the smaller v1 core in `nanocode.py`.
- Rebuilt README around the current command set, context design, and screenshot.
- Simplified tool schemas for broader OpenAI-compatible provider support.
- Improved tool-call display for Search and Recall, and surfaced intermediate assistant progress before tool calls.

### Fixed
- Fixed Moonshot/Kimi-compatible tool schemas by avoiding unsupported schema forms.
- Fixed repeated command probing in context rendering by caching system command detection per session.

## 0.4.11 - 2026-05-24

### Changed
- Bumped version from 0.4.10 to 0.4.11.
## 0.4.8 - 2026-05-23

### Changed
- Renamed the `EditFile` tool to `Edit` across the codebase and tests.

## 0.4.5 - 2026-05-21

### Changed
- Updated the built-in code index integration for `code-symbol-index` 0.1.7.
- Added indexed symbol filters for kind, path, and exact matching.
- Added file-local symbol outlines and bounded pending-index details in `/status`.

## 0.4.4 - 2026-05-20

### Added
- Added built-in indexed code navigation backed by project data and `/index` for manual init/sync.

### Changed
- Replaced the external code-navigation CLI integration with the bundled code index API.
- Hid code navigation tools until an index exists, while lightly updating existing indexes at startup.
- Updated status/docs to describe code index availability without exposing dependency-install wording.

## 0.4.3 - 2026-05-20

### Changed
- Removed stable knowledge state while keeping current-task known facts.
- Extracted shared numbered-content and line-range helpers for tool output/range handling.
- Trimmed thin helper wrappers in List and indexed code-inspection tools.

## 0.4.2 - 2026-05-19

### Added
- Added indexed code inspection tools for symbol lookup, symbol investigation, and file outlines when the local index is available.
- Added queued user feedback during long-running turns.
- Added `PatchFile` for multi-location file edits.

### Changed
- Moved model calls to the OpenAI SDK and function-tool protocol.
- Reworked task-shape prompts for chat, one-shot tasks, and tracked tasks.
- Prioritized indexed code inspection for structural lookup while keeping Search/Read for exact literals and edit ranges.
- Improved terminal UX with persistent status, queued-input handling, Bash live preview, and terminal-friendly assistant output rules.
- Renamed `ListDir` to `List`.
- Improved `Read`, `Edit`, `ReplaceRange`, `PatchFile`, `Bash`, and `Git` tool guidance.
- Simplified gate behavior so only deterministic, correctable model errors are refused.

### Fixed
- Fixed duplicate final replies for goal-only text answers.
- Fixed repeated recall loops and several format/tool-name compatibility issues.
- Fixed PatchFile diagnostics and empty-hunk handling.
- Fixed queued feedback delivery, Ctrl-C/Ctrl-D handling, and Bash interrupt reporting.

## 0.3.35 - 2026-05-16

### Added
- Added batched `ReplaceRange` edits for multiple independent ranges in the same file.
- Added a design document covering agent state, context construction, tool-result storage, observe policy, and verification.

### Changed
- Aligned tool-result context layout with the design document.
- Refined tool-result context reduction around unreduced raw results, retained results, and checkpoint-based pruning.
- Compressed ACT and OBSERVE system prompts.
- Reduced routine OBSERVE triggers by raising the pending-result threshold and keeping ordinary tool failures in ACT for repair.
- Simplified agent gate and feedback handling, including single active plan item normalization.
- Added soft feedback for state-update-only ACT turns so models continue with frontier tools, verification, or completion.
- Highlighted recognized slash commands and reported unknown slash commands directly.

### Fixed
- Accepted harmless model output variants including trailing progress text, action type casing, and `message` action aliases.
- Ignored pending verification requests instead of treating them as blocking model output.

## 0.3.34 - 2026-05-16

### Changed
- Trigger observe by unresolved pending tool-result count only, instead of consecutive tool batch count.

## 0.3.33 - 2026-05-16

### Fixed
- Keep unresolved pending tool results visible as raw ACT context until observe mode explicitly keeps or forgets them.

## 0.3.32 - 2026-05-16

### Changed
- Removed the pending tool-result character threshold for observe mode; observe now triggers from failures, pending result count, or consecutive tool turns.

## 0.3.31 - 2026-05-16

### Changed
- Require a new user turn with retained task context to align via `start`, `goal`, or `plan` before running more tools.

### Fixed
- Removed an unused pytest import from bash tool tests.

## 0.3.30 - 2026-05-16

### Changed
- Status bar now shows compact token totals and model stream rate, including `turn:` duration labeling.
- Stream rate uses live character-based estimation and completion-token usage when available.

## 0.3.29 - 2026-05-16

### Fixed
- Ignored code-fence-only text when converting interleaved model output into progress actions.

## 0.3.28 - 2026-05-16

### Changed
- Status bar now labels model calls as `working`, `observing`, or `compacting`.
- Removed the hard gate for multiple `doing` plan items while keeping prompt guidance to prefer a single active item.

### Fixed
- Preserved specific retry notices such as `err:format` instead of overwriting them with generic gate notices.
- Accepted unmarked action streams with interleaved progress text between JSON actions.
- Normalized tool-name action types such as `Search` or `ListDir` into tool actions.

## 0.3.27 - 2026-05-16

### Fixed
- Added a nanocode `User-Agent` header to provider requests so OpenAI-compatible gateways that reject Python urllib defaults can accept chat and model-list requests.

## 0.3.26 - 2026-05-16

### Changed
- Generated configs now leave `reasoning_payload` unset by default for broader provider compatibility.
- Documented when to enable `reasoning_payload`, including OpenRouter-style reasoning providers.

## 0.3.25 - 2026-05-16

### Changed
- Added Vim-style selector search with `/keyword`, `j`/`k` navigation, and step-back Esc behavior.
- Made `/model` reasoning selection transactional so Esc returns to model selection instead of applying a partial change.

## 0.3.24 - 2026-05-16

### Changed
- `/model` now groups configured `available_models` first and appends deduplicated models discovered from the provider.
- Default generated config now documents `available_models` without writing an empty setting.
- Split latest/recent and kept tool-result context budgets for steadier context growth.
- Compacted tool-result CLI output while keeping result keys visible.
- Removed `ApplyPatch`; editing now uses `Edit` for tiny literal changes and `ReplaceRange` for read-backed focused ranges.
- Refined editing prompts to prefer minimal new-file skeletons followed by focused `ReplaceRange` chunks.

### Fixed
- Stopped executing later tool calls after the first failed tool call in a batch.
- Reported Ctrl-C interrupted Bash runs as explicit interrupted tool results.

## 0.3.23 - 2026-05-16

### Changed
- Reworked tool-result context around latest, recent, pending, and kept results.
- Increased default provider and plan-mode response timeouts.
- Simplified result keep/forget handling and removed stale evidence naming from agent context.

### Fixed
- Kept non-argument tool failures visible to observe mode while treating argument errors as immediate feedback.
- Preserved Recall access to stored tool logs while allowing noisy context entries to be forgotten.

## 0.3.22 - 2026-05-16

### Fixed
- Preserved tool-result store entries referenced by Known, Hypotheses, and Evidence.
- Aligned plan-mode verify guidance with the implemented verify action shape.

### Changed
- Generated hypothesis status prompt schema from the enum to avoid prompt drift.

## 0.3.21 - 2026-05-16

### Added
- Added investigation hypotheses, including `dropped` for branches that are no longer worth tracking.
- Added evidence forgetting so ruled-out or dropped branches can release old tool results from context.

### Changed
- Tightened completion gates, verification blockers, and compact state update grouping.
- Simplified Search argument parsing and removed legacy knowledge-update behavior.
- Made provider reasoning payload shape configurable.

## 0.3.20 - 2026-05-15

### Changed
- Clarified Search tool guidance so models use at most one `glob=` per Search action and split multiple globs into multiple actions.

## 0.3.19 - 2026-05-15

### Changed
- Observe mode now requires every latest tool-result key to be covered by either `evidence` or `discard`.
- Verification pass/fail/block tool results are treated as decision-changing evidence until verification is recorded.

### Fixed
- Prevented partial observe checkpoints from silently dropping unhandled tool results.

## 0.3.18 - 2026-05-15

### Added
- Added `provider.<name>.available_models` and an interactive `/model` selector.
- Added an interactive `/provider` selector.
- Added reasoning effort selection after changing models.

### Changed
- Selection prompts now use a subtle selected-item background, support `j`/`k`, and clear after completion.
- Removed the redundant `keep current` option from selection prompts; current values are marked inline.
- Removed the startup status snapshot now that the persistent status bar is always shown.

### Fixed
- Disallowed no-value `/set provider.key` and `/set provider.url` queries.

## 0.3.17 - 2026-05-15

### Added
- Added plan mode with readonly tool limits, plan-specific timeouts, and stricter plan-mode completion formatting.
- Added persistent prompt status display while keeping active status updates during agent work.
- Added completion pressure gates so settled tasks finish by default unless the plan is reopened with context.
- Added global runtime data storage under `~/.nanocode`, with per-session debug/tool-result logs and per-project user rules.

### Changed
- Replaced project-local `[paths].nanocode_dir` with `[paths].data_dir`.
- Moved prompt history to global `~/.nanocode/history`.
- Replaced `/clean-logs` with `/clean`, which removes tool-result logs across all stored sessions.
- Compact observed tool-call results after they have been digested into agent state.

### Fixed
- Rejected chat actions in plan mode.
- Tightened blocked verification completion so it requires explicit user/manual confirmation context.
- Kept unbounded tool-result logs available on disk while bounded results stay in model context.
