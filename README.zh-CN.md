<h1 align="center">nanocode-cli</h1>

<p align="center">
  <img src="snapshots/nanocode1.gif" alt="nanocode 编辑代码并运行工具" width="600">
</p>

<p align="center">
  一个我自己在用、维护并定制的 coding agent。
</p>

<p align="center"><a href="README.md">English</a></p>

## 安全

**使用风险自负。** nanocode 会在启动环境中编辑文件并执行 shell 命令，不提供 sandbox 隔离。需要隔离时，请使用容器或虚拟机。

## 它是什么

nanocode 并不想发明一种新的 coding agent。它只是把熟悉的能力 — 读取和编辑文件、运行命令、追加指令、session 恢复、diff、MCP 和 skill — 整合成我自己真正在用的工具。

它不仅用于真实项目，也用于自身开发：我用 nanocode 来构建和维护 nanocode。所有功能都在一个 Python 模块中，所以我可以随时直接修改行为，让工作流按我想要的方式运行。

<p align="center">
  <img src="snapshots/nanocode2.gif" alt="nanocode 恢复保存的 session" width="600">
</p>
<p align="center"><sub>恢复保存的 session，包括对话和工具调用历史。</sub></p>

## 亮点

- **Prompt-cache 友好：** 稳定的指令、环境和工具 schema 保留可复用请求前缀，缓存命中率常达 98–99%。
- **代码导航：** 通过可搜索的代码索引跳转到定义、调用者和实现。
- **实时追加指令：** agent 工作时仍可输入，排队内容会加入下一轮或打断当前请求。
- **锚点编辑：** 结构化编辑使用 `line:hash` 锚点，文件内容过期时会被拒绝。
- **可恢复 session：** 对话、工具调用、diff 和工作记忆可通过 `-c` 或 `--resume` 恢复。
- **内置 diff viewer：** `/diff` 展示最新一轮改动以及整个 session 的累计变更。
- **MCP 与 skills：** 按需连接 Model Context Protocol server，加载 Markdown 指令包。
- **Provider 兼容：** 支持 OpenAI-compatible API 与 Anthropic。

## 安装

需要 macOS 或 Linux、Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
uv tool install nanocode-cli
nanocode --init-config
```

在 `~/.nanocode/config.toml` 中填写 provider：

```toml
[provider]
active = "default"

[provider.default]
url = "https://api.deepseek.com"
key = "sk-..."
model = "deepseek-v4-flash"
```

然后运行：

```sh
nanocode
```

升级：`uv tool upgrade nanocode-cli`。

## 链接

- [文档](https://nanocode.readthedocs.io) — 完整的使用指南和参考。
- [博客](https://hit9.dev/post/nanocode) — 设计动机与实现过程。
- [code-symbol-index](https://github.com/hit9/code-symbol-index) — nanocode 使用的代码索引库。
