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
- `[vision] provider = "<entry>"`: use that provider for explicit `ViewImage` calls. Attachments still go directly to the active model.
- `reasoning = "medium"`: choose `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`.
- `available_models = [...]`: add entries to the `/model` picker.
- `temperature`: omit it unless the provider or task needs a specific value.
- `max_tokens = 8192`: output-token cap; `0` uses the provider default.
- `timeout = 120`: transport inactivity timeout in seconds.
- `response_timeout = 600`: total generation timeout; `0` disables it.
- `prompt_cache_key = "auto"`: use `"off"` to omit the cache key.
- `strict_tools = false`: request strict tool schemas where supported.
- `extra_body = {}`: add provider-specific OpenAI-compatible request fields.
- `builtin_tools = [...]`: offer tools that the provider runs on its own servers; entries use the provider's documented wire shape.
- `chat_reasoning = "auto"`: select the Chat reasoning wire format. Override only when the endpoint documents a different format.
- `reasoning_history = "auto"`: replay reasoning according to the catalog; use `all`, `current_turn`, or `tool_calls` only as an explicit compatibility override.

For known provider/model combinations, minacode maps the selected reasoning effort to a supported value. Unknown compatible providers and model names remain supported on the generic path and keep the selected value. When a request is rejected, inspect `/config`, confirm the API protocol, and set `chat_reasoning` only when the endpoint's documentation requires it.

### Provider-side tools and web search

Provider-side tools run inside the model request on the provider's servers. They are different from minacode's local tools and are disabled by default. Enable only entries documented by the active provider:

```toml
[provider.default]
builtin_tools = [{ type = "web_search" }]
```

Common web-search configurations are:

- OpenAI Responses: `{ type = "web_search" }`, optionally with `search_context_size` or `filters`.
- Qwen Responses: `{ type = "web_search" }`; also `web_extractor`.
- Anthropic: `{ type = "web_search_20250305", name = "web_search", max_uses = 5 }`.
- Z.AI / BigModel: `{ type = "web_search", web_search = { enable = "True" } }`.
- Kimi / Moonshot: `{ type = "builtin_function", function = { name = "$web_search" } }`.
- OpenRouter: `{ type = "openrouter:web_search" }`; also `openrouter:web_fetch` and `openrouter:datetime`. The legacy `plugins` / `:online` search config is deprecated.
- Qwen Chat Completions: configure `extra_body.enable_search` instead of `builtin_tools`.
- DeepSeek API: web search is not available.

Minacode requires each `builtin_tools` entry to contain a non-empty `type`. Entries are protocol-specific. For a known provider, an entry belonging to another `/api` wire remains configured but is omitted from that request; switching back activates it again. `/api` reports that state, and `/config` shows both configured and active entries. On the entry's intended wire, an unsupported provider-native shape or lifecycle is still refused locally. Unknown providers retain generic pass-through behavior. Qwen Chat search, when wanted, uses `provider.extra_body.enable_search` instead of Responses `builtin_tools`.

A provider-side search does not ask for minacode tool confirmation because it happens inside the provider's response. Search text is untrusted content placed into the model context, can make the turn larger, and may expose the query to the provider's search service. Leave it disabled for sensitive questions or unattended work unless that behavior is acceptable.

When available, minacode shows each search in the transcript and lists provider-reported sources under the answer. Sources are display-only and do not replay to the provider. Not every provider reports sources.

Two longer-lived flows are automatic:

- Kimi may return `$web_search` as a normal-looking function call. When that name was declared by a `builtin_function` entry, minacode returns its arguments unchanged as the provider requires; it does not run a local tool or request confirmation.
- Anthropic may return `stop_reason: "pause_turn"` while a server tool is still running. Minacode sends the paused message back unchanged and continues. Each continuation consumes an agent step, so `runtime.max_agent_steps` remains the upper bound.

Common runtime settings live under `[runtime]`:

- `yolo = false`: keep mutating-tool confirmations enabled.
- `max_context_tokens = 245760`: context ceiling used for compaction budgeting.
- `max_agent_steps = 400`: maximum tool steps in one turn.
- `shell_timeout = 60`: maximum shell command lifetime.
- `bash_wait_timeout = 10`: foreground wait before a command becomes a background job.
- `max_parallel_tools = 4`: maximum concurrent read-only tool calls.
- `session_retention_days = 7`: remove untouched sessions after this many days; `0` keeps them.
- `theme = "auto"`: terminal theme; `light` and `dark` are also accepted.
- `language = "auto"`: reply language for the session; `auto` follows your messages and injects
  nothing, a name like `Chinese` appends a fixed `LANGUAGE OVERRIDE` block to the system prompt.
  Set it with `/language [NAME]`.

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
- `/strict`: toggle strict tool schemas.
- `/mcp`: list and manage MCP connections.
- `/exit` or `/quit`: save and exit.

Use `@server` or `@server.tool` to point the agent at an MCP server or tool. Use `$skill-name` to load a skill explicitly for the current turn.

## Tools and permissions

Core tools include:

- `Read`, `Search`, and `InspectCode` for files, text, and indexed symbols.
- `ViewImage` for supported local image files, with an optional `question` answered by the vision model when one is configured.
- `Edit` for anchored file changes.
- `Bash` and `Job` for foreground and background commands.
- `Recall` for complete stored tool results and `RecallContext` for compacted conversation segments.
- `Note` for durable goal, plan, check, and learned facts.
- `Ask` for decisions that require the user.
- `Skill` for on-demand instruction packs and `MCP` for connected server capabilities.

Read-only tools can run concurrently. Edit, Bash, Job, and MCP actions that can change state ask for confirmation unless yolo mode is active. Anchored edits reject stale file locations. Recommend working in Git and reviewing `/diff`.

Provider-side tools are not part of this local registry. They run at the provider without a confirmation prompt when enabled through `builtin_tools` or `extra_body`; apply the privacy and trust guidance above.

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
- Provider-side tool rejected: compare `api` and `builtin_tools` with the provider's documented wire shape. A wrong-wire entry is omitted and shown as inactive, not sent. A refusal before the request therefore means the active wire recognizes the tool family but minacode does not support that entry shape or lifecycle; use a supported entry. Qwen Chat search uses `provider.extra_body.enable_search`. A picker entry or compatible model name does not prove the account can use the tool.
- Web search did not run: confirm that it is enabled in the active provider block shown by `/config`, that the selected model supports it, and that the request uses the required protocol. Do not expect a sources footer when the provider reports no sources.
- Provider-side search appears stuck: preserve the exact error or stop reason. Anthropic `pause_turn` and configured Kimi `builtin_function` handshakes are continued automatically but remain bounded by `max_agent_steps` and `response_timeout`.
- Streaming failure: set `provider.stream = false` temporarily and retry.
- Stalled generation: distinguish `timeout` inactivity from total `response_timeout`; inspect the exact error before increasing either.
- Image rejected: the failed image remains available at its session-owned path. Use `ViewImage` explicitly; configure `[vision] provider = "<entry>"` when the active model cannot inspect it directly.
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

- `minacode/base.py`: errors, text and log primitives, and shared data types.
- `minacode/config.py`: provider entries, runtime settings, and the config file.
- `minacode/providers/`: model capability catalog (`catalog.py`) and protocol-compatibility policy (`compat.py`).
- `minacode/model/`: the model client (`client.py`) with the Chat, Responses, and Anthropic adapters beside it, plus retry policy in `resilience.py`.
- `minacode/engine.py`: the agent turn loop and context/tool composition.
- `minacode/context.py`: context projection, token budgeting, and compaction.
- `minacode/runner.py`: tool execution, approvals, parallelism, and result storage.
- `minacode/session/`: live session state (`__init__.py`) and snapshot persistence (`store.py`).
- `minacode/skill.py`: skill discovery, parsing, precedence, and expansion; `minacode/mentions.py`: mention grammar and @file expansion.
- `minacode/mcp/`: MCP server config (`config.py`), connection lifecycle (`manager.py`), display formatting (`rendering.py`), and OAuth token storage (`tokens.py`).
- `minacode/tools/`: built-in tool implementations, split by capability, with the registry in `__init__.py`.
- `minacode/cli/`: the command loop (`__init__.py`), commands (`commands.py`), modals (`modals.py`), `/worker`'s flow (`worker.py`), TUI runtime (`runtime.py`), and view fragments (`view.py`).
- `minacode/tui/`: the prompt-toolkit application (`app.py`) and interactive view state (`views.py`); `minacode/render.py`: terminal rendering, live output, and the status bar.
- `minacode/prompts.py`: the system and worker prompts and compaction templates; `minacode/image.py`: local image input.
- `minacode/__main__.py`: CLI arguments and startup.
- `tests/`: executable behavior examples and regression coverage.

When source and documentation disagree, report the mismatch and treat matching-version code plus tests as the strongest evidence for actual behavior.
