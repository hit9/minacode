# Anthropic 路径：会话主体从未进入缓存

状态：**已修，待实测**（2026-08-19，分支 `refactor/model-package`）。修法与结论见文末「落地」一节。
发现于 compaction 前缀复用工作（分支 `toolscript-mcp-names-and-script-phase`）的收尾复查。

## 事实

Anthropic 没有隐式缓存。不写 `cache_control` 就完全不缓存 —— 这与 OpenAI 系不同，
后者对 ≥1024 token 的前缀自动缓存，无需任何参数（minacode 的 `prompt_cache_key`
只是给缓存分域/路由，不是开关）。

minacode 当前用**显式**断点，打在 system 块上：

    # minacode/model.py，anthropic_params
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

官方文档：

> Cache prefixes are created in the following order: `tools`, `system`, then `messages`.
> ... up to and including the block designated with `cache_control`.
> **cache writes happen only at your breakpoint**, not at earlier positions.

断点在 system 结束处 ⇒ 写入只发生在那里 ⇒ **messages 从来没有被写进缓存**。

## 影响

不只是压缩请求，**每一个普通 turn 都是**：Anthropic 路径上对话主体每轮按全价重新
处理，缓存住的只有 tools + system（本仓库实测约 5.4k token 的工具 schema + 系统提示）。
一个 10 万 token 的会话，每轮为 9 万多 token 付全价。

连带后果：分支上那一整轮「压缩请求复用对话前缀缓存」的改动，**在 Anthropic 上收益为 0**，
因为它要复用的那段缓存根本不存在。收益只在 OpenAI 系兑现（deepseek / 阿里云 / kimi /
GLM / OpenRouter）。

`context.py` 的类 docstring 里那条设计原则 —— "Layer order exists for prompt-cache
stability … inserting a rebuilt block into the conversation prefix would invalidate later
cache reuse" —— 整个是按「会话主体在缓存里」写的，但在 Anthropic 上它从未成立。

## 可行的修法

SDK 0.104.1 已支持顶层字段（已验证 `anthropic.types.message_create_params`）：

    cache_control: Optional[CacheControlEphemeralParam]
    """Top-level cache control automatically applies a cache_control marker to the last
       cacheable block in the request."""

打在最后一个可缓存块上，也就是对话末尾 —— 正是 OpenAI 隐式缓存的行为。用它可以让
Anthropic 路径与其他 provider 行为一致，不再是特例。

## 开工前必须先确认的取舍

1. **写入溢价**：缓存写按 1.25x 计。每轮为新增段付写入，换旧段 0.1x 读取。对 10 万
   token 的 agent 循环几乎肯定大赚，但这是**所有 Anthropic 用户的计费形态变化**，
   不是纯优化。
2. **TTL 5 分钟**：turn 间隔超过 5 分钟就只付写不收读。当前"只缓存 tools+system"的
   做法在这点上是稳而少。
3. **断点能否并存（关键，未验证）**：理想形态可能是两个断点 —— system 长期命中 +
   末尾滚动，正是文档说的 "cache different sections that change at different
   frequencies"（上限 4 个）。若能并存，那才是正解，而不是把 system 断点换掉。

## 验证手段

`/status` 的 cache 行（`last` / `session` 命中率）是现成观测点。改前改后各跑一个
真实 session 对比即可。这件事影响每一个 turn，收益和风险都远大于分支上的压缩改动，
因此单独成一个分支做。

## 参考

- https://platform.claude.com/docs/en/build-with-claude/prompt-caching

## 落地（2026-08-19）

没有用顶层 `cache_control`，用的是**第二个显式断点**，打在请求最后一个块上
（`minacode/model/anthropic.py` 的 `mark_prompt_cache_tail`）。理由：顶层字段是 Anthropic
SDK 的新参数，而这条 Messages 路径不只发往 api.anthropic.com —— OpenCode Zen 的
`claude-*` / `qwen-*` 也走它（`providers/catalog.py` 的 `api_rules`）。块级 `cache_control`
是 Messages 线协议本身的一部分，system 块上已经在无条件发了；顶层字段则是网关可能整个不认的
未知参数，按仓库里 `json_response_format` 那条注释的判断标准，"未知即关"。块级不需要开关。

对三条取舍的回答：

1. **断点能否并存** —— 能，官方上限 4 个，现在用 2 个（system + 滚动尾部）。所以是**加**一个
   断点，不是把 system 断点换掉，正是文档里 "cache different sections that change at
   different frequencies" 的形态。
2. **写入溢价** —— 保留默认 5 分钟 TTL（写 1.25x）。`ttl: "1h"` 写要 2x，得三轮才回本，
   而 agent 循环的轮间隔基本都在 5 分钟内；真要改，先看实测。
3. **TTL 5 分钟** —— 未变。用户离开超过 5 分钟的那一轮只付写不收读，和改之前一样。

一处顺带的改动：`anthropic_messages` 现在把所有文本内容统一渲染成块列表，不再有裸字符串。
因为断点必须落在块上，而"当轮渲染成块（带标记）、下一轮变回字符串"会让同一段历史在两轮里
是两个前缀 —— 缓存正好在它写入的那一段上读不中。统一形状后这个问题不存在。

滚动断点只有一个，所以受官方那条 **20 个块的回溯窗口**约束：一轮如果新增超过 20 个块
（大量并行工具调用 + tool_result），这一轮的断点找不到上一轮的条目，会整轮全价。实测如果
看到命中率间歇性掉底，下一步就是在长轮次中间再插一个断点（还剩 2 个额度）。

待办：按原计划跑真实 session 对比 `/status` 的 cache 行。
