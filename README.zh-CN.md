# nanocode-cli

<img src="snapshots/nanocode1.gif" alt="nanocode demo" width="680">

一个文件，一个编程代理。描述任务——它读取、编辑、运行命令，然后汇报结果。

[English](README.md)

## 安装

```sh
uv tool install nanocode-cli
```

创建配置，开始使用：

```sh
nanocode --init-config
# 编辑 ~/.nanocode/config.toml → 填写 provider.url, provider.key, provider.model
nanocode
```

升级：`uv tool upgrade nanocode-cli`

常用参数：

- `--resume [UID]`：恢复已保存的 session，不传 UID 恢复最近一次
- `--yolo`：跳过工具确认，直接执行
- `--mcp <selector>`：选择启用哪些 MCP 服务器
- `--config <path>`：使用指定的 TOML 配置文件

## 为什么选 nanocode

**实时输入。** 代理还在工作时就能继续打字。下一条消息排队等待，不打断当前回合。

**编辑不乱漂。** 每次编辑都携带 `line:hash` 锚点。代码在底层变了，编辑就被拒绝——不会悄悄损坏文件。

**Session 随时恢复。** 随时退出，`nanocode --resume` 恢复。对话、工具结果、工作记忆全部还原。

**Prompt-cache 友好。** 稳定上下文（system prompt、环境、工具 schema）字节级一致，支持 prompt cache 的 provider 直接复用，节省费用和延迟。

**只有一个文件。** `nanocode.py` 就是整个 agent——可读、可改、可直接 vendoring。

<img src="snapshots/nanocode2.gif" alt="nanocode session" width="680">

## 概览

| | |
|---|---|
| 工具 | Read, Search, Edit, Bash, InspectCode, Job, Recall, Note, Ask, MCP, Skill |
| Provider | OpenAI, Anthropic, DeepSeek, OpenRouter, llama.cpp，以及任意 Chat-Completions 端点 |
| MCP | 远程（HTTP streamable）和本地（stdio）服务器，支持 OAuth |
| Skills | 可复用指令包（Markdown）；项目 `.nanocode/skills/` 和用户 `~/.nanocode/skills/` |
| 编辑 | 结构化 patch 操作（`replace`, `insert_before`, `insert_after`, …）配 `line:hash` 锚点 |
| Session | 自动保存 JSONL 快照，支持 `--resume latest` / `--resume <id>` |
| 索引 | 代码符号索引——outline、references、implementors、call chains（`InspectCode`） |
| 上下文 | cache-stable prefix + conversation + Memory（goal/plan/known/check）三段式结构 |
| 配置 | TOML —— `~/.nanocode/config.toml` |

## 命令

| 命令 | 用途 |
|---|---|
| `/help` | 显示命令和工具列表 |
| `/status` | 运行状态：token 用量、context 占比、缓存命中率 |
| `/context` | 模型上下文帧——环境、记忆（goal/plan/known/check） |
| `/diff` | 最新编辑 diff 和 session 整体净 diff（交互式，支持标签切换） |
| `/skills` | 列出已安装 skills |
| `/config` | 显示当前配置 |
| `/debug` | 最近三条内存诊断（cache-prefix 不匹配等） |
| `/compact` | 立即压缩上下文 |
| `/mcp` | 管理 MCP 服务器和工具 |
| `/provider [NAME]` | 显示或切换 provider |
| `/model [MODEL]` | 显示或切换模型 |
| `/reason` | 调整 reasoning effort |
| `/strict` | 切换严格工具调用 schema（OpenAI / DeepSeek） |
| `/set KEY VALUE` | 在当前 session 中设置配置项 |
| `/yolo` | 切换工具确认 |
| `/exit`, `/quit` | 退出 |

交互选择器支持 `j`/`k`、方向键、`/` 搜索、Enter、Esc。输入支持历史补全和 `Ctrl-R` 历史搜索。

## 配置

配置文件：`~/.nanocode/config.toml`

核心段：

- `[provider] active = "name"`
- `[provider.<name>]`：`url`, `key`, `model`, `api`, `prompt_cache_key`, `reasoning`, `temperature`, `max_tokens`, `strict_tools`, `timeout`
- `[paths] data_dir`
- `[runtime]`：`shell_timeout`, `max_agent_steps`, `max_context_tokens`, `max_parallel_tools`, `session_retention_days`, `yolo`, `tips`

`api = "auto"` 会根据 provider/model profile 自动选择 Chat Completions 或 Anthropic Messages。`prompt_cache_key = "auto"` 根据 provider、model、workspace 和工具 schema 自动生成稳定 key。

## MCP

连接 [Model Context Protocol](https://modelcontextprotocol.io) 服务器，通过 `MCP` 工具暴露其能力。

远程服务器（HTTP）：

```toml
[mcp.example]
url = "https://example.com/mcp"
bearer_token_env_var = "EXAMPLE_MCP_TOKEN"  # 可选
enabled = true
```

本地服务器（stdio）：

```toml
[mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
enabled = true
```

运行时管理：`/mcp` 查看状态，`/mcp tools [server]` 列出工具，`/mcp login/logout <server>` 管理 OAuth。

## Skills

Skills 是 agent 按需加载的指令包。每个 skill 是一个包含 `SKILL.md` 的文件夹。

- **发现**：`.nanocode/skills/`（项目级）和 `~/.nanocode/skills/`（用户级），同名时项目优先
- **模型视角**：上下文只放 name + description 索引，正文在 `Skill(name)` 调用时加载
- **内联引用**：消息中 `$name` 直接引用（Tab 补全）
- **随附脚本**：`{skill_dir}` 展开为 skill 目录绝对路径，可通过 `Bash` 运行
- **内置**：默认自带 `nanocode-help` skill，包含使用手册和自动生成的命令/工具/配置清单

## 安全

**使用风险自负。** nanocode 会在启动它的环境中编辑文件和执行 shell 命令。它不提供 sandbox 保护。需要隔离时，请在你自己的 sandbox、容器、虚拟机或其他隔离环境中运行。
