# Skills

A **skill** is a reusable pack of instructions the agent loads only when it's needed —
a release checklist, a project convention, a multi-step procedure. Keeping the full text out
of context until it's used means you can have many skills without bloating every request.

## What a skill looks like

A skill is a folder containing a `SKILL.md` file with `name` and `description` frontmatter:

```
~/.nanocode/skills/
  release-notes/
    SKILL.md
    generate.py        # optional bundled script
```

```markdown
---
name: release-notes
description: Draft release notes from the git log since the last tag.
---

1. Run `git log $(git describe --tags --abbrev=0)..HEAD --oneline`.
2. Group the commits by type and summarize each group.
3. If a bundled script is needed, run it with Bash — see paths below.
```

nanocode only sees the `name` and `description` until the skill is used — the full body
loads on demand.

## Where skills come from

nanocode discovers skills from two places:

- `.nanocode/skills/` — project-local, checked in with the repo
- `~/.nanocode/skills/` — your personal skills, available everywhere

If both define the same name, the **project** skill wins. List what's installed with
`/skills`.

## Using a skill

- **On demand** — nanocode loads a skill itself when it's relevant to your request.
- **Inline** — type `$name` in a message (Tab-completes) to load a skill yourself for that turn.

## Bundled scripts

A skill can ship helper scripts alongside `SKILL.md`. Inside the body, `{skill_dir}` (or
`${SKILL_DIR}`) expands to the skill's absolute path, so instructions can point the agent at
a script to run through Bash:

```markdown
Run the generator: `python {skill_dir}/generate.py`
```
