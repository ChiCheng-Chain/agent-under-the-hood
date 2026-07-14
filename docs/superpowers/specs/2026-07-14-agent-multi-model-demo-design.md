# Agent 多模型工程化与观测 Demo 设计规格

> 来源：`设计文档/agent_multi_model_engineering_demo.md`
> 创建日期：2026-07-14

## 1. 目标

把设计文档提出的"企业级 Agent 系统三个工程化主题"落成可运行 demo,用代码证明一句话:

> AgentService 永远只依赖自己的内部协议;底层 provider、框架、观测平台都是可替换实现。

三个主题对应三种情况:

1. **情况一(无框架)**:系统自己实现 `ModelGateway`,上层 Agent 只调 `modelGateway.invoke(request)`,不直接碰任何模型 SDK。
2. **情况二(框架作为 adapter)**:框架塞进 adapter 内部,业务层不被框架类型污染,`AgentService` 一行不改就能切换底层实现。
3. **情况三(观测)**:自建最小 trace 体系,业务层只依赖 `TraceRecorder`,sink 可插拔。

## 2. 核心决策(已与用户确认)

| 决策点 | 选择 |
|---|---|
| 语言 | TS 为主 + Python 对比,两种都做 |
| 深度 | 三情况全覆盖,以 mock 为主,不需要任何 API key |
| trace 展示 | CLI + 本地 Web UI |
| 仓库布局 | `ts/` + `python/` 双独立项目 |
| Python 完整度 | 接真实 LangChain + fake LLM,做完整 adapter(含工具调用) |

**学习价值锚点**:TS 用 `FrameworkLikeAdapter` 证明"模拟框架层也能被关进 adapter",Python 用真实 LangChain 证明"真实框架也能被关进 adapter"。同一个 `ModelProvider` 接口,两种语言、两种框架策略,殊途同归。

## 3. 架构与边界

### 3.1 分层

```
上层  AgentService          — 只依赖内部接口
中层  ModelGateway           — 路由 / 重试 / fallback
      TraceRecorder          — trace 编排
下层  ProviderAdapter / Sink — 唯一允许接触外部 SDK 的地方
```

铁律:上层永远只调用 `modelGateway.invoke(request)` 和 `traceRecorder.startSpan(...)`,不 import 任何模型 SDK、框架类型、观测 SDK。

### 3.2 仓库布局

```
ts/                          # TS 主项目(三情况全覆盖)
  src/
    agent/AgentService.ts
    model/
      types.ts               # ModelRequest / ModelResponse / ModelError
      errors.ts
      ModelGateway.ts
      ModelRouter.ts
      RetryPolicy.ts
      FallbackRouter.ts
      adapters/
        MockOpenAIAdapter.ts
        MockAnthropicAdapter.ts
        MockGeminiAdapter.ts
        MockLocalAdapter.ts
        FrameworkLikeAdapter.ts   # 模拟框架层(情况二)
    observability/
      types.ts               # Trace / Span / TraceEvent
      TraceRecorder.ts
      sinks/
        JsonlTraceSink.ts
        ConsoleTraceSink.ts
        LangfuseTraceSink.ts      # 空壳,证明可插拔
        OpenTelemetryTraceSink.ts # 空壳,证明可插拔
    web/                     # 本地观测 UI
      server.ts
      public/index.html
      public/app.js
    demo/
      runNoFrameworkDemo.ts
      runFrameworkAdapterDemo.ts
      runObservabilityDemo.ts
  tests/
  storage/                   # traces.jsonl / spans.jsonl / events.jsonl(运行时生成)
  package.json
  tsconfig.json
  vitest.config.ts

python/                      # Python 对比项目(情况二)
  agent/AgentService.py
  model/
    types.py
    errors.py
    ModelGateway.py
    ModelRouter.py
    RetryPolicy.py
    FallbackRouter.py
    adapters/
      LangChainModelAdapter.py   # 接真实 LangChain + FakeListChatModel
  observability/
    types.py
    TraceRecorder.py
    sinks/JsonlTraceSink.py
  demo/run_framework_adapter_demo.py
  tests/
  storage/
  pyproject.toml
```

## 4. TS 主项目设计(三情况全覆盖)

### 4.1 内部协议(types.ts)

严格按设计文档 84-132 行定义:

- `ModelRequest`:`traceId` / `taskType`(chat|code|summary|rag)/ `model?` / `messages` / `tools?` / `temperature?` / `stream?` / `timeoutMs?` / `metadata?`
- `ModelResponse`:`provider` / `model` / `content` / `toolCalls?` / `finishReason`(stop|tool_calls|length|content_filter|error)/ `usage?`(inputTokens/outputTokens/totalTokens/estimatedCost)/ `latencyMs` / `raw?`
- `ModelError`:`type`(AUTH_ERROR|RATE_LIMIT|TIMEOUT|CONTEXT_LENGTH_EXCEEDED|CONTENT_FILTERED|BAD_REQUEST|PROVIDER_UNAVAILABLE|RESPONSE_PARSE_ERROR|UNKNOWN)/ `provider` / `model?` / `retryable` / `message` / `raw?`

**错误归一化核心**:`ModelGateway` 只信 `ModelError.retryable` 标志做重试决策,不信 provider 原始错误。adapter 负责把各家错误映射成内部 `ModelError` 并设好 `retryable`。

### 4.2 Mock Adapter 行为(情况一)

| Adapter | 行为 |
|---|---|
| `MockOpenAIAdapter` | 正常返回 |
| `MockAnthropicAdapter` | 第一次返回 RATE_LIMIT,第二次成功 |
| `MockGeminiAdapter` | 返回 CONTEXT_LENGTH_EXCEEDED |
| `MockLocalAdapter` | 高延迟,展示超时和 fallback |

### 4.3 重试与 fallback(RetryPolicy + FallbackRouter)

严格按设计文档 136 行表:

| 错误类型 | 重试 | 处理 |
|---|---|---|
| RATE_LIMIT | 是 | 指数退避,超阈值切备用 |
| TIMEOUT | 是 | 短重试,超阈值 fallback |
| PROVIDER_UNAVAILABLE | 是 | 熔断当前 provider,切备用 |
| CONTEXT_LENGTH_EXCEEDED | 否 | 记 event,最小版直接失败(不实现真裁剪) |
| AUTH_ERROR | 否 | 直接失败 |
| BAD_REQUEST | 否 | 直接失败 |
| CONTENT_FILTERED | 否 | 记录,不重试 |
| RESPONSE_PARSE_ERROR | 可选 | 最多重试一次 |

`RetryPolicy` 负责单 provider 内的指数退避(基线 300ms,上限 3 次);`FallbackRouter` 负责单 provider 重试耗尽后按 `taskType` 选备用模型。两者都通过 `ModelError.retryable` 触发,不硬编码错误类型。

### 4.4 框架 adapter(情况二)

`FrameworkLikeAdapter` 模拟一个"有自己类型体系的框架":内部定义 `FrameworkMessage`、`FrameworkRunnable`、`FrameworkResult` 等类型,`invoke` 内部把 `ModelRequest` 转成 `FrameworkMessage[]`,跑 runnable,再把 `FrameworkResult` 转回 `ModelResponse`、把框架异常转成 `ModelError`。

**关键演示**:`AgentService` 代码一行不改,把注入的 adapter 从 `MockOpenAIAdapter` 换成 `FrameworkLikeAdapter`,返回的 `ModelResponse` 结构完全一致,trace 照常记录。这就证明了"框架只是 adapter 的内部实现"。

### 4.5 观测(情况三)

数据模型严格按设计文档 292-340 行:`Trace` / `Span` / `TraceEvent`。

`TraceRecorder` 提供异步安全 API:

- `startTrace(metadata) → traceId`
- `startSpan(type, name, parentSpanId?) → spanId`
- `endSpan(spanId, status, summary?)`
- `addEvent(name, payload?)`
- `endTrace(traceId, status)`

每次请求记录:1 个 root trace → 1 个 agent span → 每次 LLM 调用 1 个 llm span → 每次工具调用 1 个 tool span → 每次 retry/fallback 1 个 span 或 event。

LLM span 至少记录:provider / model / prompt version / input token / output token / cost / latency / finish reason / retry count / fallback from→to。

sink 接口:`interface TraceSink { writeTrace(t); writeSpan(s); writeEvent(e); }`。实现 `JsonlTraceSink`(写 storage/*.jsonl)、`ConsoleTraceSink`(打印树状结构)、`LangfuseTraceSink` 与 `OpenTelemetryTraceSink`(空壳类,带 TODO 注释,证明接口可插拔)。

### 4.6 Web UI(本地观测平台)

用 Node 原生 `http` 模块起 server,故意不引 React/Vite,呼应情况一"可控、轻量"理念。

- `GET /` → 返回 `public/index.html`(列表页)
- `GET /api/traces` → 全部 trace 摘要列表
- `GET /api/traces/:id` → 单 trace 完整 span 树
- `GET /api/metrics` → 聚合指标(成功率、错误率、p95 延迟、总 token、总成本、fallback 次数)

前端纯原生 JS:trace 列表页 + 单 trace 树状图 + waterfall 时间线 + 聚合指标卡片。

### 4.7 CLI 入口

```
npm run demo:no-framework         # 情况一:多模型网关 + 重试 fallback
npm run demo:framework-adapter    # 情况二:切换到 FrameworkLikeAdapter
npm run demo:trace                # 情况三:跑一次完整链路,输出 trace 树
npm run trace:view <traceId>      # 查看单 trace
npm run web                       # 启动观测 UI(默认 http://localhost:4318)
```

## 5. Python 对比项目设计(情况二)

### 5.1 职责边界

Python 项目只证明一件事:真实 LangChain 也能被关进 adapter。不做情况一和情况三的完整复刻——TS 版已经覆盖,Python 重复无价值。

### 5.2 LangChainModelAdapter

接真实 `langchain_core` + `langchain_openai` 的 `FakeListChatModel`(不需要 API key)。

adapter 职责:

1. 内部 `ModelRequest` → LangChain `BaseMessage` 列表(`SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`)
2. 内部 `ToolDefinition` → LangChain `bind_tools` 格式
3. 调用 fake chat model(预设返回序列,模拟正常 / 限流 / 工具调用)
4. `AIMessage` + `tool_calls` → 内部 `ModelResponse.toolCalls`
5. LangChain 异常 → 内部 `ModelError`(设好 `retryable`)
6. usage/latency 填充

### 5.3 Python 协议复用

Python 的 `ModelRequest`/`ModelResponse`/`ModelError` 用 dataclass + pydantic 表达,字段与 TS 版一一对应,证明"内部协议是稳定边界,与语言无关"。

`AgentService` 与 TS 版逻辑一致:通过 `ModelGateway.invoke` 调用,不直接 import langchain。

### 5.4 轻量 trace

Python 项目复用一个极简 `TraceRecorder`(只写 JSONL,不做 Web UI),够在终端打印 trace 树证明链路可观测即可。

### 5.5 CLI 入口

```
python -m demo.run_framework_adapter_demo
```

## 6. 测试与验收(TDD)

按 CLAUDE.md 要求,测试先写。重点不是测 mock 返回值,而是测**边界真的成立**。

### 6.1 TS 测试(vitest)

- **边界不变性**:把注入的 adapter 从 `MockOpenAIAdapter` 换成 `FrameworkLikeAdapter`,断言 `AgentService` 返回的 `ModelResponse` 结构一致
- **重试**:喂 RATE_LIMIT,断言重试次数、退避间隔、最终成功
- **fallback**:喂 TIMEOUT 超阈值,断言切到备用模型,trace 里有 fallback span
- **不重试**:喂 AUTH_ERROR / BAD_REQUEST,断言零重试直接失败
- **上下文超限**:喂 CONTEXT_LENGTH_EXCEEDED,断言不重试、记 event、失败
- **trace 完整性**:每个请求有 traceId;通过 traceId 能查到完整 span 树;llm span 含 token/cost/latency
- **多 sink**:`JsonlTraceSink` 和 `ConsoleTraceSink` 同时写,断言两者都收到全部 span

### 6.2 Python 测试(pytest)

- `ModelRequest` → LangChain messages → `ModelResponse` 往返结构正确
- 工具调用转换:`tool_calls` 正确映射
- LangChain 异常 → `ModelError.retryable` 正确
- `AgentService` 不 import langchain(用 import 检查或边界测试)

### 6.3 验收标准(对照设计文档第 7 节八条)

1. 同一 Agent 请求可切不同模型后端 ✓(MockRouter 按 taskType)
2. provider 原始响应归一成内部响应 ✓(adapter 转换 + 测试)
3. provider 错误映射成统一错误码 ✓(ModelError + 测试)
4. 重试/fallback 按错误类型触发,不盲目重试 ✓(retryable 标志 + 测试)
5. 框架 adapter 时上层 Agent 不改 ✓(边界不变性测试)
6. 每次请求都有 traceId ✓
7. trace 含 LLM/工具/retry/fallback/耗时/错误/token/成本 ✓
8. CLI 或页面可查单次链路 ✓(trace:view + Web UI)

## 7. 实现顺序(对照设计文档第 5 节五步)

1. **无框架网关**:types/errors + ModelGateway + ModelRouter + 四个 MockAdapter
2. **重试 fallback**:RetryPolicy + FallbackRouter
3. **框架 adapter**:FrameworkLikeAdapter(TS)+ LangChainModelAdapter(Python)
4. **最小 trace**:TraceRecorder + JsonlTraceSink + ConsoleTraceSink
5. **多 sink + Web UI**:LangfuseTraceSink/OpenTelemetryTraceSink 空壳 + 本地观测 UI

每步完成后跑对应测试,绿了再进下一步。

## 8. 范围外(YAGNI)

明确不做:

- 真实模型 API 调用(用 mock / fake LLM 替代)
- 真实 Langfuse / OpenTelemetry 后端(sink 只做空壳接口)
- 上下文裁剪/压缩的真实实现(CONTEXT_LENGTH_EXCEEDED 只记 event)
- RAG retriever / embedding / reranker 的真实实现(数据模型保留,演示用 mock retriever span)
- 多租户配额、权限、脱敏、加密(企业能力,demo 不做)
- monorepo 工具链(turbo/nx 等)
- Python 的完整情况一/三(TS 已覆盖)
