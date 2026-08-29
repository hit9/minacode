# Provider/Model Catalog JSON 化技术方案

状态：Proposed

方案日期：2026-08-28

## 1. 目标

把 minacode 中会随 provider、model 和模型代际变化的兼容性知识全部迁入一个完整的
JSON catalog。Python 只保留稳定的协议实现、通用匹配/优先级算法、受限规则解释器、
同步与错误处理，不再出现 provider host、model family、模型版本阈值或某家 API 的特判。

完成后应满足：

- 包内自带一份可离线工作的完整 catalog。
- GitHub 上维护同一份完整 catalog；本机至多每 72 小时自动检查一次。
- 两个来源分别完整解析和校验，只采用 `version` 更大的整份文件，不跨来源 merge。
- JSON 顶层必须有数字 `version` 和更新日期 `updated_at`。
- 现有代码中的 provider/model 注释、why、evidence 和边界说明成为 JSON 数据，而不是在
  Python 中遗留一份不可同步的知识。
- 可以通过 `/catalog sync` 强制同步，并能看到当前版本、来源、更新时间和上次同步结果。

## 2. 非目标与边界

以下内容仍属于 Python：

- Chat Completions、Responses、Anthropic Messages 三种 wire adapter 的稳定协议结构、SDK
  调用和流式响应解析。协议名可以存在于 Python；provider/model 的兼容性事实不能。
- 通用的 hostname label-boundary 匹配、model selector、优先级、nearest-effort、JSON patch、
  原子文件替换和错误分类。
- 用户配置的显式覆盖；显式配置始终高于 catalog。
- catalog schema 支持的有限操作集合。远端 JSON 不能导入 Python、执行表达式、访问文件、
  发起额外网络请求或扩展操作集合。

以下内容必须离开 Python：

- provider 名称、域名、model prefix/regex、canonical vendor slug。
- 模型代际阈值、family token、effort scale、off spelling、thinking budget。
- reasoning request 字段差异、历史 reasoning 回放差异、temperature/cache/strict tool/image
  能力差异。
- provider-side tool 白名单、request protocol 路由规则。
- 与上述事实有关的 why、evidence URL、排除项和维护说明。

未知 provider/model 继续走 catalog 的 generic defaults；catalog 不是 allowlist。

## 3. 当前需要迁移的知识

当前特化知识并不只在 `providers/catalog.py`：

| 位置 | 需要迁移的内容 |
| --- | --- |
| `minacode/providers/catalog.py` | `MODEL_TRAITS`、`PROVIDER_CATALOG`、text-only rules、canonical vendors、effort/budget/Anthropic generation 常量及全部 why/evidence/comments |
| `minacode/providers/compat.py` | Claude version/family 判断、thinking payload、prior-thinking 策略，以及由具体字段名驱动的兼容性分支 |
| `minacode/model/anthropic.py` | 手动 thinking budget 表及模型代际相关参数生成 |
| `minacode/model/client.py` | `reasoning`、`reasoning_effort`、`thinking`、`enable_thinking` 等具名模式分支 |
| `minacode/model/protocol.py` | Anthropic model family 的 prior-thinking 特判 |
| `minacode/config.py` | 静态 compatibility class variable、catalog 驱动的 resolve/choice/source 逻辑，以及 provider/model 事实型注释 |

迁移时还要对 `minacode/**/*.py` 做一次 provider/model literal 审计。协议 adapter 自身、用户可见
配置示例和测试用例可以出现协议名；生产 Python 不得保留会随 catalog 更新的事实。

## 4. 总体结构

```text
 GitHub raw catalog.json                     installed package catalog.json
             |                                           |
             v                                           v
       CatalogRepository  -- validate each source --> CatalogCodec
             |                                           |
             +------------- choose whole max(version) ---+
                                      |
                                      v
                              immutable CatalogSnapshot
                                      |
                        +-------------+-------------+
                        |                           |
                        v                           v
              CompatibilityResolver          RequestRuleEngine
                        |                           |
                        +-------------+-------------+
                                      |
                                      v
                               ProviderPolicy
                                      |
                        Config / CLI / model adapters
```

建议目录：

```text
minacode/providers/
├── __init__.py
├── catalog.json       # 包内完整快照，也是 GitHub raw 同步目标
├── schema.py          # Raw TypedDict + immutable dataclass
├── catalog.py         # CatalogCodec、CatalogSnapshot 编译
├── compat.py          # CompatibilityResolver、RequestRuleEngine、ProviderPolicy
└── sync.py            # CatalogRepository、CatalogRuntime、同步状态
```

依赖只能向下：`config/model/cli/session -> compat/sync -> catalog -> schema`。`catalog.py` 不再
定义任何 provider/model 常量。`sync.py` 不导入 CLI 或 Session；由启动层传入 `data_dir`。

## 5. JSON 快照契约

### 5.1 顶层字段

```json
{
  "schema_version": 1,
  "version": 2026082801,
  "updated_at": "2026-08-28",
  "maintenance_scope": "well_known_and_necessary_specializations_only",
  "defaults": {},
  "model_id_forms": [],
  "request_recipes": {},
  "model_rules": [],
  "providers": []
}
```

- `schema_version`：JSON 格式版本，正整数。第一版客户端只接受 `1`。
- `version`：快照发布版本，正整数，是来源选择的唯一排序依据。建议采用
  `YYYYMMDDNN`，例如当日第一次发布为 `2026082801`。同一 version 的内容必须不可变。
- `updated_at`：UTC 日期，严格 `YYYY-MM-DD`；用于展示和审计，不参与胜负判断。
- `maintenance_scope`：固定为 `well_known_and_necessary_specializations_only`。catalog 只维护
  well-known 且确有必要偏离 generic protocol 的 provider/model 事实；未知对象走 generic
  fallback，不能为了“覆盖更多”收录无行为差异的枚举数据。
- `defaults`：generic provider/model 默认策略、normalized effort 顺序、budget tables 和 wire
  默认值。它使每份快照自洽，不依赖 Python 中的第二份默认数据。
- `model_id_forms`：例如受 allowlist 约束的 `vendor/model` 规范化方式；vendor 列表也在此。
- `request_recipes`：通用 request patch recipe，详见 5.4。
- `model_rules`：与 host 无关的 model facts。
- `providers`：host policy、provider model overlays 和 provider-side tools。

“不 merge”指来源级别：先选择一份完整 document，再编译其中的 defaults/rules。document 内
按照 schema 定义做 policy precedence，不是把包内和缓存的 section 拼起来。

### 5.2 Rule 与来源信息

每个 provider 和每条非 generic rule 都有稳定 `id`，并携带知识来源：

```json
{
  "id": "model.deepseek-v4-flash.reasoning",
  "match": {
    "prefixes": ["deepseek-v4-flash"]
  },
  "set": {
    "reasoning.recipe": "chat.thinking-with-effort",
    "reasoning.levels": ["low", "high", "max"],
    "history.reasoning": "tool_calls"
  },
  "why": "DeepSeek serves medium and xhigh as high",
  "evidence": [
    "https://api-docs.deepseek.com/guides/thinking_mode/"
  ],
  "notes": [
    "A level that behaves exactly like its neighbour is not offered in /reason."
  ]
}
```

- `why` 是用户界面可展示的一行原因；延续当前约束：无换行、建议不超过 80 字符。
- `evidence` 是非空 HTTPS URL 数组，优先官方一手资料。
- `notes` 保存原代码注释中的维护语义、排除项、precedence 原因和未来改动风险；不直接显示给
  普通用户。
- 单纯 generic default 可以只有 `description`；任何缩窄能力、改变 wire、禁止字段或声明
  text-only 的 rule 必须有 `why` 和 `evidence`。

JSON 不支持 comment，因此迁移不是删掉注释。原注释应按语义进入 `description`、`why`、
`evidence` 或 `notes`，并在迁移 review 中逐段核对。

### 5.3 Selector 与优先级

model selector 支持固定且可验证的几种条件：

- `prefixes`: case-insensitive prefix，数组内为 OR。
- `pattern`: Python regex，使用 `re.match` 语义；有长度上限并在加载时预编译。
- `tokens_any` / `tokens_all`: 按数据声明的 separator 分词后匹配。
- `version`: 由 rule 自带 regex 提取命名组 `major`/`minor`，支持
  `min_inclusive`、`max_inclusive`、`max_exclusive`。

一个 selector 内的不同字段为 AND；`prefixes` 数组内部为 OR。空 selector 非法，只有明确标记
为 generic 的 default rule 例外。

每个 policy field 独立选择第一个有值的来源，顺序固定为：

1. 用户显式配置。
2. 当前 provider 内第一个匹配的 `model_rules`。
3. 顶层第一个匹配的 `model_rules`，除非 provider 对该 policy namespace 声明
   `model_rule_modes.<namespace>: "ignore"`（现有 normalized reasoning gateway 场景）。
4. 当前 provider 的 `defaults`。
5. catalog 的 generic `defaults.provider_policy`。

同一层数组按 JSON 顺序 first-match。validator 要拒绝重复 provider host、重复 rule id、未知
recipe 引用和不可达的明显重复 selector。host 仍按 DNS label boundary 匹配，多个命中时取最长
domain；该算法不含任何 provider 名。

### 5.4 受限 request recipe

不能把 Python 的特判改名为 `AnthropicStrategy`、`DeepSeekStrategy` 后继续留在代码里。
`request_recipes` 使用非图灵完备的、白名单化 patch 语言描述请求差异，Python 只有一个
`RequestRuleEngine`。

recipe 结构：

```json
{
  "steps": [
    {
      "when": {
        "reasoning_enabled": true
      },
      "set": [
        {
          "path": ["reasoning_effort"],
          "value": {"source": "resolved_effort"}
        }
      ],
      "remove": []
    }
  ]
}
```

第一版只允许：

- condition：`eq`、`in`、`present`，输入键限于 `wire`、`reasoning_enabled`、
  `resolved_effort`、`off_value`、`max_tokens` 及 selector 已解析的 version/tokens。
- value：JSON literal、`source`、boolean `case`、catalog table `lookup`、
  `bounded_budget`。
- action：对已构造 request body 执行 `set` 或 `remove`；path 是字段数组，不支持通配符。
- 写入路径 allowlist：request body 与 `extra_body`；禁止 header、URL、filesystem、SDK 参数和
  callable。

`bounded_budget` 是通用整数运算：从 catalog table 按 effort 取值，并限制在
`minimum <= value <= max_tokens - headroom`。因此当前 `manual_thinking_budget()` 可删除，
thinking budget 与其证据都留在 JSON。

下面的片段证明模型代际差异也能数据化，而不需要 Python 解析 Claude 名称：

```json
{
  "id": "model.claude-4.6.adaptive-thinking",
  "match": {
    "prefixes": ["claude-"],
    "version": {
      "pattern": "(?:^|[-.])(?P<major>[0-9]{1,2})(?:[-.](?P<minor>[0-9]{1,2}))?(?:[-.]|$)",
      "min_inclusive": [4, 6],
      "max_exclusive": [4, 7]
    }
  },
  "set": {
    "reasoning.recipe": "messages.adaptive-effort-xhigh-as-max",
    "reasoning.levels": ["low", "medium", "high", "xhigh", "max"],
    "history.reasoning": "all"
  },
  "why": "This generation uses adaptive thinking and maps xhigh to max",
  "evidence": [
    "https://platform.claude.com/docs/en/build-with-claude/extended-thinking",
    "https://platform.claude.com/docs/en/build-with-claude/effort"
  ],
  "notes": [
    "Unknown aliases receive no generation recipe; guessing can turn a valid alias into HTTP 400."
  ]
}
```

4.5 manual thinking、4.5 Opus output effort、4.7+ adaptive thinking、always-thinking families 和
prior-thinking retention 分别由有序 rule + recipe 表达。Python 的 Messages adapter 只组装稳定的
`model/system/messages/max_tokens/tools` envelope，然后让通用 engine 应用选中的 recipe。

### 5.5 Provider 数据

provider record 的核心结构：

```json
{
  "id": "provider.openrouter",
  "hosts": ["openrouter.ai"],
  "model_rule_modes": {
    "reasoning": "ignore"
  },
  "defaults": {
    "api": "chat",
    "reasoning.recipe": "chat.reasoning-object"
  },
  "model_rules": [],
  "builtin_tools_by_wire": {
    "chat": [
      {"type": "openrouter:web_search"}
    ]
  },
  "why": "The endpoint normalizes upstream reasoning behind its own object",
  "evidence": [
    "https://openrouter.ai/docs/guides/best-practices/reasoning-tokens"
  ],
  "notes": [
    "Native model reasoning fields must not leak through this normalized gateway."
  ]
}
```

`model_rule_modes` 按 namespace 设置 `inherit` 或 `ignore`，未列出的 namespace 默认
`inherit`。因此 normalized gateway 可以忽略模型自身 reasoning dialect，同时仍继承全局
text-only/image evidence。该开关是通用 precedence，不在 Python 中提及任何 gateway。
provider defaults 还可包含 cache key、JSON response、strict schema、temperature、history、
image input、Responses reasoning family、builtin tool policy 等现有字段。

text-only negative list 作为普通 model policy `image.input: "text_only"` 存储。对
`vendor/model` 的 suffix 匹配只能通过 `model_id_forms` 中的 vendor allowlist 开启，避免 custom
alias 被误判；allowlist 本身也不能留在 Python。

## 6. Python 建模与接口

Raw JSON 用 `TypedDict` 描述边界，校验后编译成 frozen dataclass。不要让业务层直接持有
`dict[str, object]`。

```python
class RawCatalog(TypedDict):
    schema_version: int
    version: int
    updated_at: str
    maintenance_scope: str
    defaults: RawDefaults
    model_id_forms: list[RawModelIdForm]
    request_recipes: dict[str, RawRequestRecipe]
    model_rules: list[RawPolicyRule]
    providers: list[RawProvider]


@dataclass(frozen=True)
class CatalogSnapshot:
    schema_version: int
    version: int
    updated_at: date
    maintenance_scope: str
    defaults: CatalogDefaults
    model_id_forms: tuple[ModelIdForm, ...]
    request_recipes: Mapping[str, RequestRecipe]
    model_rules: tuple[PolicyRule, ...]
    providers: tuple[ProviderRule, ...]
    source: Literal["bundled", "cached"]
    content_hash: str
```

主要 class：

```python
class CatalogCodec:
    def decode(self, payload: bytes, source: CatalogSource) -> CatalogSnapshot: ...
    def canonical_hash(self, raw: RawCatalog) -> str: ...


class CatalogRepository:
    def load(self) -> CatalogLoadResult: ...
    def sync(self, *, force: bool = False) -> SyncResult: ...


class ProviderPolicy:
    def resolve(self, config: ProviderConfig) -> ResolvedProvider: ...
    def reasoning_choices(self, config: ProviderConfig, model: str = "") -> tuple[str, ...]: ...
    def effort_source(self, config: ProviderConfig, model: str = "") -> Evidence: ...
    def apply_request(self, params: Json, context: RequestPolicyContext) -> Json: ...


class CatalogRuntime:
    @property
    def policy(self) -> ProviderPolicy: ...
    def start_background_sync(self) -> None: ...
    def sync_and_activate(self) -> SyncResult: ...
    def status(self) -> CatalogStatus: ...
```

职责说明：

- `CatalogCodec` 只做 decode、schema/semantic validation、regex 预编译和不可变建模，不做 IO。
- `CatalogRepository` 拥有 package/cache/metadata 路径、GitHub fetch、锁和原子写入。
- `ProviderPolicy` 是业务层唯一入口，内部组合 `CompatibilityResolver` 与
  `RequestRuleEngine`，不暴露数据布局。
- `CatalogRuntime` 持有当前不可变 policy、72 小时调度状态，并在安全命令边界做原子引用替换；
  它不是全局变量，由顶层 Session 持有，同一会话的 delegate worker 共享它。

`ProviderConfig` 回归为纯用户配置 value object，移除 `COMPATIBILITY` class variable 和
`resolve()` 中的 catalog 查询。现有调用迁移为 `session.catalog.policy.resolve(provider)`；CLI、
image route、tool schema 和 ModelClient 共享 Session 中的同一快照。

配置加载先从原始 TOML 只解析 `data_dir`，据此选择 `CatalogRuntime`，再把获胜 policy 注入
`Config.from_dict()`，一次完成 provider、worker 和 compaction 的 catalog-dependent 值校验。
整个过程仍在 Session 启动前 fail fast。这样 `config.py` 无需保留一份与 JSON 重复的 effort
order，也不会出现“缓存 catalog 已获胜，但配置仍按 bundled vocabulary 拒绝”的分裂。wire
protocol 的固定枚举仍由 Python adapter 定义，因为它表示客户端实现了哪些协议，不是可远端
更新的 provider 知识。

所有嵌套 dataclass 字段也必须不可变：数组编译为 tuple/set 编译为 frozenset，mapping 使用只读
view，避免手动同步或测试代码原地修改 active snapshot。

`ResolvedProvider` 增加 `catalog_version`。一次 request 在 prepare 时获得完整 resolved policy，
retry/resend 复用同一个对象，不能在一次 request 中途重新读取 active catalog。

## 7. 来源选择与同步协议

### 7.1 路径和远端

- 包内：通过 `importlib.resources` 读取 `minacode/providers/catalog.json`。
- 缓存：`<data_dir>/catalog/catalog.json`。
- 状态：`<data_dir>/catalog/sync.json`。
- 锁：`<data_dir>/catalog/catalog.lock`。
- GitHub：
  `https://raw.githubusercontent.com/hit9/minacode/master/minacode/providers/catalog.json`。

`pyproject.toml` 的 package data 必须包含 `providers/catalog.json`，并在 wheel 验证中确认文件
实际存在。

本地开发不提供 `MINACODE_CATALOG_PATH` 或任意文件路径覆盖。调试 catalog 时直接修改包内完整
JSON，同时提高数字 `version` 并更新 `updated_at`；版本选择规则会让它胜过较旧缓存。若缓存版本
已经更高，继续提高包内版本或清理该开发数据目录，不能引入第三种来源绕过发布语义。

### 7.2 启动选择

1. 读取并校验 bundled。bundled 无效表示安装损坏，启动失败并给出明确错误；Python 不提供一份
   隐藏的 provider/model fallback。
2. 若 cached 存在，独立读取和完整校验。无效 cached 被忽略并记录诊断，不能影响 bundled。
3. 只在有效且受当前 `schema_version`/operation set 支持的候选中比较整数 `version`。
4. cached version 大于 bundled 时，整份采用 cached；小于时采用 bundled。
5. version 相同且 canonical hash 相同，采用 bundled；相同 version 但内容不同是发布冲突，采用
   bundled 并报告 `CatalogVersionConflict`。

`updated_at`、文件 mtime、HTTP Date 和下载时间都不参与选择。

### 7.3 自动同步

- CLI 完成首屏启动后调用 `start_background_sync()`，不阻塞启动。
- 以共享 `data_dir` 的 `last_checked_at` 计算，未满 72 小时不发网络请求。
- 自动检查成功、304、远端旧版、校验失败或网络失败都写入本次 checked 时间；这样自动请求至多
  每 72 小时一次。手动同步忽略该间隔。
- 使用 5 秒 timeout、明确 User-Agent、`Accept: application/json`、ETag/
  `If-None-Match`，并限制响应体大小（建议 2 MiB）。只有当前 cached 仍有效时才发送其 ETag。
- 后台错误不能逃出线程或打断 turn；保留当前快照，并通过 `/catalog`/`/status` 可见。
- 自动同步成功只更新缓存，不热切换当前 Session，避免一个长 turn 的不同 request 使用不同知识；
  下次启动自然选择新版本。

### 7.4 手动同步与激活

新增命令：

- `/catalog` 或 `/catalog status`：显示 active version、source、`updated_at`、schema version、
  cached/bundled version、last checked 和 last error。
- `/catalog sync`：忽略 72 小时间隔，下载并校验；若得到更大版本，在命令边界原子替换当前
  Session 的 `ProviderPolicy`。该命令不是 queue-safe，不能在模型 request 运行中执行。

示例结果：

```text
catalog: v2026082801 (cached, updated 2026-08-28, schema 1)
sync: activated v2026082801; previous v2026082101
```

若远端版本不更大，命令报告 `current`/`remote older`，不得改写 active policy。

### 7.5 原子性和并发

- 同步先在同目录写唯一临时文件，flush + fsync 后 `os.replace`。
- fetch、重新读取当前 cache、version 比较和 replace 在跨进程 lock 内完成，避免多个 minacode
  进程把较新 cache 回退为较旧版本。
- 写入前再次比较当前 cache；任何候选都只能单调提高 cache version。
- `sync.json` 用同样的临时文件 + replace；它只是调度/诊断数据，损坏时按“从未检查”处理，
  不影响 catalog 选择。

## 8. 校验与错误策略

错误分类：

- `CatalogFormatError`：JSON、类型、日期、regex、引用或 semantic invariant 无效。
- `CatalogVersionConflict`：相同 version 对应不同 canonical content。
- `CatalogSourceError`：package/cache 读取失败。
- `CatalogSyncError`：timeout、HTTP、响应过大、ETag 或原子写入失败。

加载时至少校验：

- `bool` 不得冒充整数 version；version 必须为正且不超过约定的安全整数范围。
- `updated_at` 是有效 UTC 日期。
- 顶层必需字段完整，未知 schema/operation/value source 被拒绝。
- provider/rule/recipe id 唯一，host 唯一，所有引用存在。
- selector 非空、regex 可编译且长度受限。
- effort order 无重复；rule 中引用的普通 effort 在 order 中存在。
- policy enum、history mode、wire、JSON path 和 builtin tool shape 合法。
- 缩窄菜单、禁用能力或声明 text-only 的条目有 why/evidence。
- notes/evidence/why 的大小和总 document size 有上限。

cached/remote 无效时回退到 bundled/active；bundled 无效时 fail fast。只对网络错误做后续周期
重试，不对格式、版本冲突或不支持 schema 在同一次同步中重试。

## 9. 运行时时间线

### 启动

```text
Config 解析 data_dir
  -> CatalogRuntime.load()
  -> bundled/cached 各自校验
  -> max(version) 整份选中
  -> Session 持有 immutable ProviderPolicy（delegate worker 共享同一 CatalogRuntime）
  -> 首屏完成
  -> 如到期，后台同步 cache
```

### 一次请求

```text
ProviderConfig + model
  -> ProviderPolicy.resolve()
  -> 固定 catalog_version 的 ResolvedProvider
  -> wire adapter 组装通用 envelope
  -> RequestRuleEngine 应用 recipe
  -> omit_body / SDK send
```

用户配置的 `omit_body` 仍是最后一步；catalog recipe 不能绕过受保护的 request 主字段。

### 手动同步

```text
/catalog sync
  -> lock + fetch + validate
  -> candidate.version > installed/cache/active max
  -> atomic cache write
  -> command boundary atomic policy swap
  -> report exact old/new version
```

## 10. 迁移步骤

1. 定义 schema/codec，并把当前 catalog 数据逐条转成一份 `catalog.json`。每段现有 comment 都
   要在迁移 review 表中映射到 `description/why/evidence/notes`。
2. 先用 bundled-only `ProviderPolicy` 跑现有 provider matrix，确认所有 resolve、reasoning
   choice、request body、history 和 image route 行为等价。
3. 把 `ProviderConfig.resolve()` 及相关 helper 的调用迁到 Session 持有的 `ProviderPolicy`；在
   同一变更中删除 Python 静态 tables，运行时不得同时读取 Python data 和 JSON data。
4. 引入 `RequestRuleEngine`，迁移 `ModelClient.apply_provider_params()` 的具名模式分支、Claude
   generation/budget/prior-thinking 分支；删除对应函数和常量。
5. 加入 `CatalogRepository`、三日后台同步和 `/catalog` 命令，更新 package data。
6. 更新 `DESIGN.md` 的 provider boundary、用户文档和 `CHANGELOG.md` 的 Unreleased；用户文档
   只说明自动更新、手动命令、版本/来源可见性和离线回退。
7. 对生产 Python 做 literal/comment audit，确认无 provider/model 可变知识残留。

迁移开发期间可以让测试同时计算旧/新结果作差异诊断，但最终提交不能保留双运行时或 fallback
merge。切换完成后，JSON 是唯一知识源。

## 11. 验证方案

实现阶段应覆盖：

- codec：完整合法文件、字段类型、未知 op、regex、引用、why/evidence invariant。
- selection：bundled/cache 高低版本、equal-identical、equal-conflict、cached 损坏、unsupported
  schema；断言从未产生 section merge。
- sync：72 小时 gate、force、304、timeout、oversize、旧远端、无效远端、原子写、并发单调性。
- behavior parity：保留当前 provider catalog matrix 的独立预期，覆盖 unknown host/model、host
  boundary、gateway normalization、effort/off、strict/cache/temperature、builtin tools、text-only。
- request recipes：每种现有 request shape 与拒绝路径，特别是 Claude 4.5/4.6/4.7+、unknown
  alias、always-thinking、budget cap 和 prior-thinking。
- lifecycle：一个 request/retry 固定同一 `catalog_version`；后台同步不热切换；手动命令只在
  安全边界激活。
- packaging：构建 wheel 后直接检查 bundled JSON 存在，并在无网络环境启动。
- static audit：provider host、model family、evidence URL 和 catalog-only request field 的生产
  literal 只允许出现在 `catalog.json`；协议 adapter 必需的 wire 名单独 allowlist。

行为变更完成时按项目流程运行 targeted tests、`uv run pytest`、ruff check/format check 和
pyright。

## 12. 验收标准

- `minacode/providers/catalog.json` 是唯一 provider/model 兼容性知识源，含数字 `version`、
  `updated_at` 和窄化的 `maintenance_scope`。
- 包内和 GitHub/cache 均为完整快照；只选择有效候选中的最大 version，不 merge。
- 离线首次启动可用；cached/remote 损坏不会覆盖 bundled/active。
- 自动同步至多每 72 小时一次；`/catalog sync` 可强制同步并明确报告是否激活。
- Python 中不存在 provider host、model family/version、thinking budget 或 evidence-driven
  request 特判。
- 当前所有 provider/model 行为与 unknown fallback 保持等价，除非另有明确 changelog。
- 原代码中的 catalog comments、why 和 evidence 都能在 JSON 的对应记录中找到。
- wheel/sdist 均包含 bundled JSON，运行中的 request 固定其 `catalog_version`。

## 13. 发布与维护规则

更新 catalog 时必须：

1. 修改完整 `catalog.json`，不得发布 fragment/patch。
2. 把 `version` 增加到一个从未使用过的整数，同时更新 `updated_at`。
3. 不得修改已经发布过的同 version 内容。
4. 为新增或改变的事实写 `why`、官方 `evidence` 和必要 `notes`。
5. 运行 schema、behavior matrix、packaging 和 static audit。
6. 正常提交到 `master`；GitHub raw 文件随后成为远端快照，下一次包发布又把同一快照带入
   bundled source。

schema 或 recipe operation 的扩展必须先随 Python 代码发布。旧客户端遇到不支持的
`schema_version`/operation 会拒绝远端并继续使用自己的 bundled/cached 有效版本，不能猜测执行。
