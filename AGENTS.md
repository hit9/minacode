# Agent guide

Keep this file short. It is an entry point, not a second design document.

## Start here

- Read [DESIGN.md](DESIGN.md) before changing cross-cutting behavior or module ownership.
- Follow the nearest existing pattern before introducing a new abstraction or dependency.

## Project map

- `minacode/engine.py`: agent loop, context projection, protocols, compaction, and tool lifecycle.
- `minacode/session.py`: durable semantic state and persistence.
- `minacode/tools/`, `minacode/image.py`, `minacode/mcp.py`, `minacode/skill.py`: vertical feature modules.
  `tools/` splits the built-in tool set by capability and owns the registry in its `__init__.py`.
- `minacode/provider_compat.py`: evidence-backed provider compatibility policy.
- `minacode/loop.py`, `minacode/tui.py`, `minacode/render.py`: commands, interaction, and presentation.
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
