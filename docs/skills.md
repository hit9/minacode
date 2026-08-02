# Skills

A **skill** is a reusable pack of instructions the agent loads only when it's needed —
a release checklist, a project convention, a multi-step procedure. Keeping the
<span class="marker">full text out of context until it's used</span> means you can have many
skills without bloating every request.

## Creating and installing skills

### What a skill looks like

A skill is a folder containing a `SKILL.md` file with `name` and `description` frontmatter:

```
~/.minacode/skills/
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

minacode only sees the `name` and `description` until the skill is used — the full body loads
on demand.

### Where skills come from

minacode discovers skills from three sources:

- Builtin skills shipped with minacode
- `.minacode/skills/` — project-local, checked in with the repo
- `~/.minacode/skills/` — your personal skills, available everywhere (under
  `<data_dir>/skills/` when `paths.data_dir` is customized)

When names collide, project skills override user skills, and user skills override builtins.
List what's available and which source won with `/skills`.

Every installation includes **`minacode-help`**, a compact manual for installation,
configuration, providers, commands, sessions, tools, safety, and troubleshooting. The agent can
load it when a question concerns minacode, or you can request it explicitly with
`$minacode-help`. If the manual does not settle the question, it directs the agent to inspect the
matching version's source code and tests.

## Using skills

- **On demand** — minacode loads a skill itself when it's relevant to your request.
- **Inline** — type `$name` in a message (Tab-completes) to load a skill yourself
  <span class="marker">for that turn</span>.

```{figure} ../snapshots/minacode-skill-mention.png
:alt: Using $skill mention to load a skill's instructions inline
:width: 600px
:align: center

Loading a skill with $name inline.
```

### Bundled scripts

A skill can ship helper scripts alongside `SKILL.md`. Inside the body, `{skill_dir}` (or
`${SKILL_DIR}`) expands to the skill's absolute path, so instructions can point the agent at
a script to run through Bash:

```markdown
Run the generator: `python {skill_dir}/generate.py`
```
