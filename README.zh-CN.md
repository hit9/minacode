# nanocode

一个用 Python 写的小型终端编程代理。

[English](README.md)

[中文博客](https://hit9.dev/post/nanocode)

nanocode 仍是 1.0 前的软件。稳定版发布前，命令、配置和工具行为都可能变化。

![nanocode screenshot](snapshots/nanocode-snapshot.png)

## 特性

- **实时回合控制**：代理还在工作时，也可以追加输入，不打断当前工具流程。
- **文件状态大脑**：`Read` 和 `Edit` 会构建当前重要文件的带行号视图。
- **过期编辑保护**：`line:hash` 锚点会在目标代码漂移后拒绝错误编辑。
- **项目级导航**：通过符号索引快速查看 outline、references、implementors、call chains 和变更文件。
- **可恢复上下文**：prompt 中的工具输出保持有界，原始 `tr.N` 结果仍可按需召回。
- **Session 恢复**：通过 `nanocode --resume` 恢复已保存工作，包括重新展示的会话历史。
- **缓存友好上下文**：稳定内容靠前，嘈杂的工作状态靠后，提高 prompt cache 复用率。
- **聚焦工作记忆**：`Note` 把 goal、plan、known facts 从嘈杂执行日志中拆出来。
- **MCP 集成**：连接远程（HTTP）或本地（stdio）的 Model Context Protocol 服务器并调用其工具。
- **终端优先工作流**：模型选择、历史搜索、确认、实时命令输出、追加输入和状态展示都在一个 CLI 内完成。

## 安装

使用 uv 安装：

```sh
uv tool install nanocode-cli
```

升级：

```sh
uv tool upgrade nanocode-cli
```

本地开发：

```sh
uv sync --extra dev
uv run nanocode
```

## 快速开始

创建配置文件：

```sh
nanocode --init-config
```

编辑 `~/.nanocode/config.toml`，然后启动：

```sh
nanocode
```

常用参数：

- `--config <path>`：使用指定 TOML 配置文件。
- `--init-config`：创建默认配置文件。
- `--resume [UID]`：恢复已保存的 session；不传 `UID` 时恢复 `latest`。
- `--yolo`：跳过会修改环境的工具确认。
- `--mcp <selector>`：选择启用哪些已配置的 MCP 服务器。
- `--debug`：写入模型 I/O debug trace。
- `-v`, `--version`：显示版本。
- `update` / `upgrade`：将 nanocode 更新到最新版本，并根据安装方式（uv tool、pipx 或 pip）选择正确的升级方式。

代理运行中，可以在 `+>` 提示符里输入追加内容，发送到下一次模型请求。

## Sessions

nanocode 会把有可恢复内容的非空 session 保存到 `[paths] data_dir` 下，并在退出时打印恢复命令：

```sh
Resume with: nanocode --resume <session-id>
```

按 id 恢复，或恢复最近保存的 session：

```sh
nanocode --resume <session-id>
nanocode --resume latest
nanocode --resume last
```

恢复后会在启动时重放一次可见会话历史；工具摘要会显示，但不会打印原始 tool result 正文。`/status` 会显示当前 session id。早于 `runtime.session_retention_days` 的 session 文件会在启动时清理，默认 `7` 天；设置为 `0` 可关闭清理。

## CLI

命令：

- `/help`：显示命令和工具。
- `/status`：显示运行状态，包括当前 session id。
- `/context`：显示模型的上下文帧——环境、记忆（goal、plan、known facts、check）。
- `/diff`：显示最新编辑 diff 和本 session 的整体净 diff。交互式 prompt 下打开 `Latest`、`Session` 两个标签；`←/→` 或 `h`/`l` 切换标签，`↑/↓` 或 `j`/`k` 在列表中移动，`Enter` 打开文件 diff，在 diff 视图中 `↑/↓` 滚动，`Ctrl-U`/`Ctrl-D` 滚动半页，`PgUp`/`PgDn` 滚动一页，`Esc`/`←` 返回文件列表，`r` 刷新，`Esc`/`q` 从列表关闭。
- `/skills`：列出已安装的 skills（用 `Skill(name)` 加载，或在消息中用 `$name` 引用）。
- `/config`：显示当前配置。
- `/api [auto|chat|anthropic]`：显示或设置 provider API 格式。
- `/debug [on|off]`：切换模型 I/O debug trace。
- `/compact`：立即压缩上下文。
- `/index [force]`：同步或重建代码符号索引。
- `/mcp [tools|login|logout|refresh] ...`：管理 MCP 服务器和工具。
- `/provider [NAME]`：显示或设置 provider。
- `/model [MODEL]`：显示或设置模型。
- `/reason`：选择 reasoning effort。
- `/strict`：切换严格工具调用 schema（仅 OpenAI / DeepSeek）。
- `/set KEY VALUE`：设置当前 session 支持的 provider/runtime 值。
- `/yolo`：切换工具确认。
- `/exit`, `/quit`：退出。

交互选择器支持 `j`/`k`、方向键、`/` 搜索、Enter 和 Esc。输入框支持历史、补全和 `Ctrl-R` 历史搜索。

工具：

- 文件：`Read`, `Search`。
- 代码索引：`InspectCode`。
- 编辑：`Edit` 创建或修改文件内容。
- Shell：`Bash`（包括 `ls`、`find`、`wc` 和 `git`）。
- 工具结果：`Recall`。
- 工作笔记：`Note`。
- 询问用户：`Ask`。
- MCP：`MCP`。
- Skills：`Skill` 按需加载某个 skill 的完整说明（只要存在任一 skill 即提供——内置的 `nanocode-help` 意味着它通常始终可用）。

`Read`、`Search` 和 `InspectCode` 会在合适时返回行锚点。`Edit` 使用当前 `line:hash` 锚点拒绝过期编辑。

## 配置

默认配置位置：

```text
~/.nanocode/config.toml
```

主要字段：

- `[provider] active = "name"`
- `[provider.<name>]`：`url`, `key`, `model`, `api`, `prompt_cache_key`, `available_models`, `reasoning`, `chat_reasoning`, `temperature`, `max_tokens`, `strict_tools`, `timeout`
- `[paths] data_dir`
- `[runtime] shell_timeout`, `max_agent_steps`, `max_context_tokens`, `max_parallel_tools`, `check_updates`, `update_check_interval_hours`, `session_retention_days`, `yolo`, `debug`, `tips`

`api = "auto"` 会根据 provider/model profile 在 Chat Completions 和 Anthropic Messages 之间选择。`prompt_cache_key = "auto"` 会根据 provider、model、workspace 和工具 schema 名称生成稳定 key。

`strict_tools = true`（用 `/strict` 切换）会将工具调用参数约束到每个工具的 JSON schema。它仅在支持该特性的 host（OpenAI 和 DeepSeek）以及 Chat Completions 路径上生效，其他情况下为空操作。对 DeepSeek，启用后请求会路由到 `/beta` 端点。无法在严格函数调用下表示的工具 schema 会自动回退到非严格模式。

`--yolo`、`--debug` 和 `--mcp` 等 runtime flags 对恢复的 session 同样生效。保存的 session 不会携带旧 runtime config。

## MCP

nanocode 可连接 [Model Context Protocol](https://modelcontextprotocol.io) 服务器，并通过 `MCP` 工具暴露其工具。每个服务器配置在 `[mcp.<name>]` 下，且只能二选一：`url`（远程）或 `command`（本地）。

通过 streamable HTTP 的远程服务器：

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # 可选；发送 Authorization: Bearer
enabled = true

[mcp.oauth_example]
url = "https://example.com/mcp"
auth = "oauth"                              # 通过 /mcp login <server> 在浏览器登录
enabled = true
```

通过 stdio 的本地服务器：

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { SOME_TOKEN = "value" }              # 可选；会合并到继承的环境变量之上
enabled = true
```

HTTP 鉴权选项（`auth`、`bearer_token_env_var`、`env_http_headers`）只对 `url` 服务器生效。`env_http_headers` 把 header 名映射到存放其值的环境变量。

运行时管理服务器：

- `/mcp`：列出已配置服务器及连接状态。
- `/mcp tools [server]`：列出已发现的工具。
- `/mcp refresh [server]`：重新发现服务器。
- `/mcp login <server>` / `/mcp logout <server>`：OAuth 登录和登出。

## Skills

Skills 是可复用的指令包，agent 可按需加载。每个 skill 是一个包含 `SKILL.md` 的文件夹：

```text
.nanocode/skills/                 # 项目 skills（随仓库一起提交）
  release-notes/
    SKILL.md
    scripts/
      collect_commits.py
~/.nanocode/skills/               # 个人 skills（对所有项目生效）
```

`SKILL.md` 带有 `name`/`description` 前置元数据，正文是 Markdown 指令：

```markdown
---
name: release-notes
description: Draft a CHANGELOG entry from commits since the last release.
---
运行 `python "{skill_dir}/scripts/collect_commits.py" <last-tag>` 收集提交，
再按类型分组并以项目既有风格撰写条目。
```

- **发现路径**：`.nanocode/skills/`（项目）和 `~/.nanocode/skills/`（用户）。同名时项目 skill 优先。
- **模型如何看到**：上下文中只放一个精简的 `SKILLS` 索引（name + description）；完整正文仅在模型调用 `Skill(name)` 时按需加载。对同一 skill 的重复加载会折叠为一行指针，避免重复计费。未安装任何 skill 时不会向 prompt 添加内容。
- **内联引用**：在消息中输入 `$name`（支持 Tab 补全）以提示模型使用该 skill；其指令会为该轮注入。
- **随附脚本**：正文中的 `{skill_dir}`（或 `${SKILL_DIR}`）会展开为该 skill 的绝对目录路径，模型可通过 `Bash` 运行随附脚本（除非 `/yolo`，否则仍需正常确认）。
- **查看**：`/skills` 列出已安装的 skills；状态栏和 `/status` 会显示数量。
- **内置**：默认自带 `nanocode-help` skill，其正文是一份自包含手册——包含关于如何使用 nanocode、其功能和常见问题的成文说明，外加由 nanocode 自身的 `/help` 文本、工具描述和配置项在加载时拼装而成的命令/工具/配置清单。因此“怎么用 / X 是什么 / 为什么 Y”这类问题可直接从手册回答，无需检索源码，且清单不会与运行版本脱节。放置同名 `nanocode-help` skill 即可覆盖它。

## Providers

以下 provider 已在 nanocode 中测试通过：

- **deepseek**：DeepSeek API
- **opencode**：OpenCode API
- **aliyun**：阿里云通义千问 API（Chat Completions）
- **llama.cpp**：通过 llama.cpp 服务端本地推理

## 上下文模型

每次模型请求都由 nanocode 手动构建成明确的 messages。稳定上下文在前，会话作为 messages 保留，工作记忆放在末尾。

```text
model request
+--------------------------------------------------+
| system                                           |
|   concise agent contract and tool rules          |
+--------------------------------------------------+
| user                                             |
|   Environment                                    |
+--------------------------------------------------+
| user/assistant                                   |
|   conversation, compacted summaries, tools       |
+--------------------------------------------------+
| user                                             |
|   Memory: goal, plan, known, date                |
+--------------------------------------------------+
```

核心规则：

- 回合中的 assistant 文本和用户追加输入都会作为 conversation 保留。
- 上下文过大时，较早 conversation 会压缩成明确的 summary。
- 包括 `Read` 和 `Edit` 在内的工具输出会保留在 conversation messages 中。
- 较大的工具输出会在 conversation messages 中保持有界，并可通过 `tr.N` 召回。

## 安全

nanocode 会在启动它的环境中编辑文件和执行 shell 命令。它不提供 sandbox 保护。需要隔离时，请在你自己的 sandbox、容器、虚拟机或其他隔离环境中运行。
