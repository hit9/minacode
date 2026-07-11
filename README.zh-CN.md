<h1 align="center">nanocode-cli</h1>

<p align="center">
  单文件终端编程代理：控制流明确，session 设计对 prompt cache 友好。
</p>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode demo" width="600">
</p>
<p align="center"><sub>在一个交互式 session 中编辑代码并运行工具。</sub></p>

<p align="center"><a href="README.md">English</a></p>

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv tool install nanocode-cli
nanocode --init-config
```

编辑 `~/.nanocode/config.toml`，填入 OpenAI-compatible endpoint：

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.openai.com/v1"
key = "YOUR_API_KEY"
model = "gpt-5"
```

然后启动：

```sh
nanocode
```

升级：`uv tool upgrade nanocode-cli`

常用参数：

- `--resume [UID]`：恢复已保存的 session，不传 UID 恢复最近一次
- `--yolo`：跳过工具确认，直接执行
- `--mcp <selector>`：选择启用哪些 MCP 服务器
- `--config <path>`：使用指定的 TOML 配置文件

## 个人软件

nanocode 是我为自己使用而构建的 coding agent。我不打算把它做成一个通用框架，也不追求覆盖所有工作流的完整产品。

单文件不等于代码量小。把 agent loop、runtime 状态、工具、持久化和 CLI 放在一起，让我能直接看到它们如何交互，也能自由修改实现，而不必维护 plugin API 或框架抽象。

只有真正改善我工作流的功能才会加入，我也不介意它们保持鲜明的个人取向。相应的取舍是：`nanocode.py` 文件较大、模块边界较少，也不承诺便于外部扩展。这是有意的选择：我更看重直接定制，而不是广泛复用。

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode session" width="600">
</p>
<p align="center"><sub>恢复保存的 session，包括对话和工具调用历史。</sub></p>

## 亮点

| | |
|---|---|
| 实时 follow-up | Agent 工作时仍可输入；排队消息进入下一次模型请求，也可以中断当前请求立即发送 |
| 锚点编辑 | 结构化编辑使用 `line:hash` 锚点，遇到过期文件内容时拒绝执行，而不是猜测 |
| 可恢复 session | 对话、已完成的工具调用、diff 和工作记忆在中断后仍可通过 `--resume` 恢复 |
| 内置 diff viewer | `/diff` 显示最新用户 round 的变更和整个 session 的净结果 |
| Prompt-cache 友好 | 稳定的指令、环境信息和工具 schema 保持可复用的请求前缀 |
| 开放集成 | 支持 OpenAI-compatible 或 Anthropic API、远程/本地 MCP server 和 Markdown skill |

## 常用命令

| 命令 | 用途 |
|---|---|
| `/help` | 显示完整命令和工具参考 |
| `/status` | 运行状态：token 用量、context 占比、缓存命中率 |
| `/context` | 模型上下文帧——环境、记忆（goal/plan/known/check） |
| `/diff` | 最新编辑 diff 和 session 整体净 diff（交互式，支持标签切换） |
| `/compact` | 立即压缩上下文 |
| `/mcp` | 管理 MCP 服务器和工具 |
| `/model [MODEL]` | 显示或切换模型 |
| `/yolo` | 切换工具确认 |

运行 `/help` 查看全部命令、工具和快捷键。交互选择器支持 `j`/`k`、方向键、`/` 搜索、Enter 和 Esc；输入支持历史补全和 `Ctrl-R` 历史搜索。

## 配置

配置文件：`~/.nanocode/config.toml`

生成的配置文件会说明常用 provider 和 runtime 选项。可以定义多个 `[provider.<name>]`，再通过 `[provider] active = "name"` 选择。使用 `/config` 查看当前配置，使用 `/help` 查看运行时命令。

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
