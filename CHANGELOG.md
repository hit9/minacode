# Changelog

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
- Simplified Search argument parsing and removed legacy `/knowledge update` behavior.
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
