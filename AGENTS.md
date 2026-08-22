# Agent guide

Keep this file short. It is an entry point, not a second design document.

## Start here

- New: read [Orientation](DESIGN.md#orientation) (objectives, module layers, one turn's shape)
  and skim [Common pitfalls](DESIGN.md#common-pitfalls) — changes that look like cleanups and are
  not.
- Read [DESIGN.md](DESIGN.md) before changing cross-cutting behavior or module ownership; follow
  the nearest existing pattern before introducing a new abstraction or dependency.

## Project map

- `minacode/engine.py`: the agent turn loop composing context, model, and tools.
- `minacode/context.py`, `minacode/model/`, `minacode/runner.py`: context projection/compaction,
  provider request protocols (`model/client.py` with the per-API adapters beside it), and the tool
  execution lifecycle.
- `minacode/cli/update.py`, `minacode/cli/hints.py`: the background version check and quick hints.
- `minacode/session/`: durable semantic state (`__init__.py`) and snapshot persistence
  (`store.py`); `store.py` never imports the package at module scope.
- `minacode/tools/`, `minacode/image.py`, `minacode/mcp/`, `minacode/skill.py`: vertical
  features; `tools/` splits built-ins by capability, registry in `__init__.py`.
- `minacode/config.py`, `minacode/providers/`: config-file settings, the model capability catalog
  (`providers/catalog.py`), and evidence-backed compatibility policy (`providers/compat.py`).
- `minacode/cli/`, `minacode/tui/`, `minacode/render.py`: commands (`cli/commands.py`,
  `cli/modals.py`, `/worker`'s flow in `cli/worker.py`), TUI runtime (`cli/runtime.py`), view
  fragments (`cli/view.py`), interaction, and presentation.
- `tests/`: behavior-oriented tests grouped by subsystem and boundary.

## Project workflow

- **Tests:** run targeted tests while iterating and `uv run pytest` before completing behavior changes.
- **Quality:** run `uv run ruff check minacode`, `uv run ruff format --check minacode`, and `uv run pyright`.
- **Docs:** on user-facing doc changes, update the English source, run `make -C docs locale-zh`,
  update the Chinese catalog, then build `html` and `html-zh`. The Chinese pages are gettext
  catalogs keyed on the English text, so editing English without this step renders those
  paragraphs in English on a Chinese page.
- **Docs standard:** `docs/` is written for users, not for the people who changed the code.
  - Say what the reader gets and what it costs them. Leave out how it is implemented, why an
    alternative was rejected, and the taxonomy of ways it can fail.
  - Prefer a number to a mechanism: "about eight recent messages survive" beats a paragraph on
    how the window is bounded.
  - Add a trade-off only when the reader has to choose. Name the setting or the `/status` row
    that answers it, and stop.
  - Rewriting a paragraph is usually better than appending one. A behavior change means the old
    sentence is wrong, not incomplete.
  - Reasoning that is worth keeping but not worth showing a user goes in a code comment or a
    commit message.
  - A `term-shot` is a screenshot in HTML: when the colors or rows it depicts change, it is stale
    and has to be redrawn. Keep every `<span>` closed.
- **Changelog:** record user-visible changes under `Unreleased`; omit internal-only refactors and
  doc maintenance.
- **Release (only when requested):** bump `pyproject.toml` and `minacode/base.py`, move Unreleased
  entries under the dated version, run tests, quality checks, both doc builds, and `uv build`,
  commit `Release X.Y.Z`, and create the lightweight tag `vX.Y.Z`. Do not push or publish.

## Working rules

- Make the smallest cohesive change; no pass-through wrappers or speculative specialization.
- Prefer black-box tests at the narrowest stable public boundary; bug fixes cover the reproduced
  failure, intended result, and important rejection paths (see `DESIGN.md` for the full policy).
- Mock external uncertainty, not the core behavior under test; keep tests deterministic and fast.
- Keep `CHANGELOG.md` aligned with user-visible behavior.
- Never rebuild or re-sync the project `.venv` (no `uv run --python X` / `uv sync --python X`
  with a different interpreter): the developer's own `minacode` process runs out of it, and swapping
  it mid-session deletes modules under that process and crashes it. For cross-version testing,
  point `UV_PROJECT_ENVIRONMENT` at a throwaway directory instead.
