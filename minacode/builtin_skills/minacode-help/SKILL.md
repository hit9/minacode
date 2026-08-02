---
name: minacode-help
description: Understand, configure, operate, and troubleshoot minacode. Use for questions about minacode installation, concepts, commands, configuration, providers, models, reasoning, sessions, context, tools, skills, MCP, permissions, errors, or unexpected behavior.
---

# Minacode manual and support

Help the user understand minacode or solve a minacode problem. Answer in the user's language.

## Support workflow

1. Start with the manual below. Give the shortest answer that solves the user's problem.
2. Establish the installed version with `minacode --version` when behavior may differ by version.
3. For configuration problems, ask for the relevant redacted TOML and `/config` output. For runtime problems, ask for the exact error and `/status` output.
4. Separate minacode behavior from provider behavior. Authentication, account access, rate limits, model availability, and endpoint-specific restrictions usually come from the provider.
5. Never request or reproduce a complete API key, token, credential, or private session log. Ask the user to redact secrets.
6. Do not edit configuration, delete sessions, toggle `/yolo`, or run diagnostic commands unless the user asks you to act. Explain the proposed change and how to verify it.
7. If this manual is insufficient, follow [Inspect the implementation](#inspect-the-implementation).

## What minacode is

Minacode is a terminal coding agent that runs in the user's local environment. A turn repeatedly asks the selected model what to do, runs requested tools, returns their results to the model, and stops when the model answers or reaches the configured step limit.

Important consequences:

- It is not a sandbox. File edits and shell commands affect the environment where minacode runs.
- Mutating tools ask for confirmation by default. `--yolo` and `/yolo` disable those confirmations.
- Sessions are saved automatically and scoped to the working directory.
- Conversation history is protocol-neutral, so the user can switch providers, models, and API protocols during a session.
- Large conversations are compacted automatically. Earlier compacted material remains available through `RecallContext`.

## Install and start

Minacode requires Python 3.11 or newer. The usual installation uses uv:

```sh
uv tool install minacode
minacode --init-config
minacode
```

Upgrade with:

```sh
uv tool upgrade minacode
```

Useful CLI checks:

```sh
minacode --version
minacode --help
minacode -c
minacode --resume
minacode --resume SESSION
```

`-c` and a bare `--resume` resume the latest session in the current project. A session id, name, or unambiguous id prefix can locate a specific session.

## Configure providers

The default configuration file is `~/.minacode/config.toml`. Generate a commented starter with `minacode --init-config`, use `--config PATH` for another file, and inspect resolved values with `/config`.

Only a provider is required:

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.example.com/v1"
key = "REDACTED"
model = "model-name"
```

Define more `[provider.NAME]` blocks and switch with `/provider`. Minacode supports OpenAI-compatible Chat Completions and Responses APIs, plus the Anthropic Messages API.

Leave `api = "auto"` unless automatic protocol selection is wrong:

- `auto`: infer the protocol where possible, otherwise use Chat Completions.
- `chat`: OpenAI-compatible Chat Completions.
- `responses`: OpenAI-compatible Responses.
- `anthropic`: Anthropic-compatible Messages.

An endpoint URL ending in `/chat/completions`, `/responses`, or `/messages` also selects that protocol.

Common optional provider settings:

- `stream = true`: stream responses. Set `false` for endpoints that reject streaming.
- `image_input = "auto"`: learn image support; use `"on"` or `"off"` to override it.
- `reasoning = "medium"`: choose `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`.
- `available_models = [...]`: add entries to the `/model` picker.
- `temperature`: omit it unless the provider or task needs a specific value.
- `max_tokens = 8192`: output-token cap; `0` uses the provider default.
- `timeout = 120`: transport inactivity timeout in seconds.
- `response_timeout = 600`: total generation timeout; `0` disables it.
- `prompt_cache_key = "auto"`: use `"off"` to omit the cache key.
- `strict_tools = false`: request strict tool schemas where supported.
- `extra_body = {}`: add provider-specific OpenAI-compatible request fields.
- `chat_reasoning = "auto"`: select the Chat reasoning wire format. Override only when the endpoint documents a different format.

For known provider/model combinations, minacode maps the selected reasoning effort to a supported value. Unknown compatible providers and model names remain supported on the generic path and keep the selected value. When a request is rejected, inspect `/config`, confirm the API protocol, and set `chat_reasoning` only when the endpoint's documentation requires it.

Common runtime settings live under `[runtime]`:

- `yolo = false`: keep mutating-tool confirmations enabled.
- `quick_hints = true`: allow next-step suggestion chips.
- `max_context_tokens = 245760`: context ceiling used for compaction budgeting.
- `max_agent_steps = 200`: maximum tool steps in one turn.
- `shell_timeout = 60`: maximum shell command lifetime.
- `bash_wait_timeout = 10`: foreground wait before a command becomes a background job.
- `max_parallel_tools = 4`: maximum concurrent read-only tool calls.
- `session_retention_days = 7`: remove untouched sessions after this many days; `0` keeps them.
- `theme = "auto"`: terminal theme; `light` and `dark` are also accepted.

Set `[paths] data_dir = "~/.minacode"` to change where sessions, history, OAuth tokens, user skills, and update metadata are stored. Selected provider and runtime values can be changed for the current session with `/set`.

## Commands

Use these commands at the interactive prompt:

- `/help`: show the built-in command reference.
- `/status`: show workspace, session, provider, model, context, cache, index, and jobs.
- `/config`: show active configuration and resolved values.
- `/diff`: inspect edits from the latest round or whole session.
- `/ps`: list background jobs.
- `/skills`: list skills by name, source, and description.
- `/index [force]`: sync or rebuild the code symbol index.
- `/provider [NAME]`, `/model [MODEL]`, `/reason [EFFORT]`, `/api [API]`: inspect or switch model request settings.
- `/set KEY VALUE`: set a supported `provider.*` or `runtime.*` value for this session.
- `/name [TEXT]`: show or set the session name.
- `/sessions [all]`: browse and switch saved sessions; `/resume` is an alias.
- `/compact`: compact context now.
- `/resend`: cancel and resend an in-flight model request.
- `/yolo`: toggle mutating-tool confirmations.
- `/hints`: toggle next-step suggestions.
- `/strict`: toggle strict tool schemas.
- `/mcp`: list and manage MCP connections.
- `/exit` or `/quit`: save and exit.

Use `@server` or `@server.tool` to point the agent at an MCP server or tool. Use `$skill-name` to load a skill explicitly for the current turn.

## Tools and permissions

Core tools include:

- `Read`, `Search`, and `InspectCode` for files, text, and indexed symbols.
- `ViewImage` for supported local image files.
- `Edit` for anchored file changes.
- `Bash` and `Job` for foreground and background commands.
- `Recall` for complete stored tool results and `RecallContext` for compacted conversation segments.
- `Note` for durable goal, plan, check, and learned facts.
- `Ask` for decisions that require the user.
- `Skill` for on-demand instruction packs and `MCP` for connected server capabilities.

Read-only tools can run concurrently. Edit, Bash, Job, and MCP actions that can change state ask for confirmation unless yolo mode is active. Anchored edits reject stale file locations. Recommend working in Git and reviewing `/diff`.

## Sessions and context

Sessions are append-only logs stored below `<data_dir>/projects/`, grouped by working directory. They are saved automatically. `/sessions` stays in the current project; `/sessions all` widens the picker. A specific session identifier or name can be resumed from another directory.

Automatic compaction summarizes older conversation when the context budget fills. It does not delete the session log. The model can use `RecallContext` to list, retrieve, or search compacted segments, and `Recall` to retrieve a full tool result that was shortened in the live context.

## Skills and MCP

Skills are ordinary directories containing `SKILL.md` with `name` and `description` frontmatter. They are discovered in this precedence order:

1. Builtin skills shipped with minacode.
2. User skills under `<data_dir>/skills/`, normally `~/.minacode/skills/`.
3. Project skills under `.minacode/skills/`.

Later sources override an earlier skill with the same name. This builtin skill uses the same parser, index, mention syntax, and `Skill` tool as every other skill. The full body stays out of the prompt until the agent selects the skill or the user mentions it with `$minacode-help`.

MCP servers are configured separately and connected as needed. Use `/mcp`, `/mcp connect SERVER`, `/mcp disconnect SERVER`, and `/mcp tools [SERVER]`. Only connect trusted servers because local servers can execute programs and remote tool calls can change external state.

## Troubleshoot methodically

Use the smallest relevant check:

- Missing configuration: run `minacode --init-config`, then verify the active provider with `/config`.
- Authentication or account error: redact the key, confirm its environment/account scope, and consult the provider. Do not treat this as a minacode parser bug without evidence.
- Model not found or unavailable: verify the model name and account access with the provider; `available_models` affects the picker, not server entitlement.
- Unsupported endpoint or request shape: compare `provider.url` and resolved `api`; try an explicit standard protocol only when the endpoint supports it.
- Rejected reasoning value or field: check resolved reasoning in `/config`, identify the provider/model family, and consult its API documentation before overriding `chat_reasoning`.
- Streaming failure: set `provider.stream = false` temporarily and retry.
- Stalled generation: distinguish `timeout` inactivity from total `response_timeout`; inspect the exact error before increasing either.
- Image rejected: confirm the model accepts images and inspect `image_input`; forcing `on` cannot add provider capability.
- Tool did not run: check whether approval was declined, whether yolo is off, and whether the tool reported a validation or stale-anchor error.
- Missing earlier detail: use `Recall` for shortened tool output or `RecallContext` for compacted conversation.
- Skill not found: run `/skills`, verify the directory and `SKILL.md` frontmatter, then check whether a higher-precedence source overrides the name.
- Session not found: compare the current project, try `/sessions all`, then search by full id or a more specific name.
- Code index missing or stale: run `/index`; use `/index force` only when a rebuild is needed.

After proposing a fix, give one concrete verification step. State uncertainty when the evidence is incomplete.

## Inspect the implementation

If the manual does not settle the question, inspect source code that matches the user's installed version:

1. Prefer the current workspace when it contains the minacode repository and its version matches `minacode --version`.
2. Otherwise use the source tag matching the installed release at `https://github.com/hit9/minacode`. State clearly if only a different version is available.
3. Read `DESIGN.md` first for ownership boundaries and invariants.
4. Inspect the narrowest owning module and its tests. Use structured symbol navigation for definitions, references, callers, and implementations; use text search for exact strings and configuration keys.
5. Base the answer on observed behavior and cite the file or symbol. Do not infer current behavior from unrelated versions without saying so.

Use this ownership map:

- `minacode/base.py`: configuration schema, defaults, validation, errors, and shared types.
- `minacode/provider_compat.py`: provider and model compatibility policy.
- `minacode/model.py`: Chat, Responses, and Anthropic request/response adapters.
- `minacode/engine.py`: the agent turn loop and context/tool composition.
- `minacode/context.py`: context projection, token budgeting, and compaction.
- `minacode/runner.py`: tool execution, approvals, parallelism, and result storage.
- `minacode/session.py`: durable session state and persistence.
- `minacode/skill.py`: skill discovery, parsing, precedence, mentions, and expansion.
- `minacode/mcp.py`: MCP configuration, connection lifecycle, tools, and resources.
- `minacode/tools/`: built-in tool implementations and registry.
- `minacode/loop.py`, `minacode/tui.py`, and `minacode/render.py`: commands, interaction, and presentation.
- `minacode/__main__.py`: CLI arguments and startup.
- `tests/`: executable behavior examples and regression coverage.

When source and documentation disagree, report the mismatch and treat matching-version code plus tests as the strongest evidence for actual behavior.
