# Changelog

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
