# Agent guide

Keep this file short. It is an entry point, not a second design document.

## Start here

- New to the codebase: read [Orientation](DESIGN.md#orientation) for the objectives, the module
  layers, and the shape of one turn. Then skim [Common pitfalls](DESIGN.md#common-pitfalls) — those
  are the changes that look like cleanups and are not.
- Read [DESIGN.md](DESIGN.md) before changing cross-cutting behavior or module ownership.
- Follow the nearest existing pattern before introducing a new abstraction or dependency.

## Project map

- `minacode/engine.py`: the agent turn loop that composes context, model, and tools.
- `minacode/context.py`, `minacode/model.py`, `minacode/runner.py`: context projection and
  compaction, provider request protocols, and the tool execution lifecycle.
- `minacode/update.py`: the background version check.
- `minacode/session/`: durable semantic state (`__init__.py`) and snapshot persistence
  (`store.py`); `store.py` never imports the package at module scope.
- `minacode/tools/`, `minacode/image.py`, `minacode/mcp.py`, `minacode/skill.py`: vertical feature modules.
  `tools/` splits the built-in tool set by capability and owns the registry in its `__init__.py`.
- `minacode/config.py`, `minacode/provider_compat.py`: config-file settings and evidence-backed
  provider compatibility policy.
- `minacode/cli/`, `minacode/tui.py`, `minacode/render.py`: commands (`cli/commands.py`, `cli/modals.py`),
  the TUI runtime (`cli/runtime.py`), view fragments (`cli/view.py`), interaction, and presentation.
- `tests/`: behavior-oriented tests grouped by subsystem and boundary.

## Project workflow

- **Tests:** run targeted tests while iterating and `uv run pytest` before completing behavior changes.
- **Quality:** run `uv run ruff check minacode`, `uv run ruff format --check minacode`, and `uv run pyright`.
- **Docs:** when user-facing documentation changes, update the English source, run
  `make -C docs locale-zh`, update the Chinese catalog, then build `html` and `html-zh`.
- **Changelog:** record user-visible changes under `Unreleased` in the appropriate category; omit
  internal-only refactors and documentation maintenance.
- **Release (only when requested):** bump `pyproject.toml` and `minacode/base.py`, move Unreleased
  entries under the dated version, run tests, quality checks, both doc builds, and `uv build`, commit
  `Release X.Y.Z`, and create the lightweight tag `vX.Y.Z`. Do not push or publish.

## Working rules

- Make the smallest cohesive change; avoid pass-through wrappers and speculative specialization.
- Prefer black-box tests at the narrowest stable public boundary. Bug fixes cover the reproduced
  failure, intended result, and important rejection paths; see `DESIGN.md` for the full test policy.
- Mock external uncertainty, not the core behavior under test. Keep tests deterministic and fast.
- Keep `CHANGELOG.md` aligned with user-visible behavior.
