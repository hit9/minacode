# Changelog

## Pending Version

### Added
- Added MCP client router support through a single `MCP` tool, with URL-based server configuration, bearer-token environment variable support, OAuth login/logout with persistent tokens, asynchronous tool discovery, compact model-visible MCP tool indexes, on-demand tool details, and `/mcp` inspection/refresh commands.
- Added MCP coverage for result normalization, successful tool calls, context pruning, `/mcp tools NAME`, and missing-server refresh handling.
- Added bounded MCP connection timeouts, concise MCP connection-failure logs, a `--debug` flag for starting with debug mode enabled, and a `--mcp` selector for choosing MCP servers by name glob.

### Changed
- Refined the status bar with lowercase `mcp`, right-side loading animation, and semantic per-section colors.
- Clarified prompt guidance around FILE STATE snapshots, automatic Read/Edit refreshes, stale-anchor retries, and avoiding unchanged failed tool-call retries.

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
