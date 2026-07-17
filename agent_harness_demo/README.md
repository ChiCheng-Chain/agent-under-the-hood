# agent_harness_demo

最小 Agent Harness 闭环，第二阶段升级为「可治理的 Agent Runtime」。不依赖 LangChain / LangGraph / CrewAI 等 Agent 框架，只保留底层结构。

## 阶段说明

- **第一阶段**：最小 Agent Harness 闭环（用户输入 → AgentRunner → ModelGateway → ToolRuntime → TraceRecorder）。
- **第二阶段**：在第一阶段模块边界上叠加错误治理、权限/风险控制、trace tree、eval 评测。不删除第一阶段能力。

## 核心链路

```
用户输入 -> AgentRunner -> ModelGateway -> 模型输出动作
                |                              |
                |                              v
                |                     tool_call / final_answer
                |                              |
                v                              v
          TraceRecorder <----- ToolRuntime(执行工具) -> observation 回喂
```

## 目录结构

```
agent_harness_demo/
├── agent_harness/
│   ├── types.py      # 统一类型契约（含 ModelError/RiskLevel/SideEffectType/ExecutionContext/PermissionDecision/Span）
│   ├── config.py     # 配置加载（.env / 环境变量）
│   ├── gateway.py    # ModelGateway：MockModelClient + OpenAICompatibleClient + ResilientModelGateway(retry/fallback)
│   ├── tools.py      # ToolRegistry + ToolRuntime + 内置工具(echo/get_time/search_docs/create_ticket/delete_record)
│   ├── policy.py     # ToolPolicy：权限与风险判断（allowed/denied/requires_confirmation/blocked）
│   ├── audit.py      # AuditLogger：写 audit.jsonl
│   ├── trace.py      # TraceRecorder：trace tree（agent/model/tool/retry/fallback/policy/audit/final span）
│   ├── runner.py     # AgentRunner：主循环
│   ├── cli.py        # 命令行入口（run / eval）
│   ├── __main__.py   # 支持 python -m agent_harness
│   └── eval/         # 评测模块（runner/cases/types，5 个固定 case）
├── tests/            # 8 组测试共 40 个用例
├── pyproject.toml
└── .env.example
```

## 安装

```bash
cd agent_harness_demo
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

依赖：`pydantic` `httpx` `python-dotenv`，开发额外 `pytest`。

## 运行 demo

默认使用 MockModelClient，无需任何真实模型：

```bash
python -m agent_harness.cli "帮我搜索 refund policy"
python -m agent_harness.cli "现在几点"
python -m agent_harness.cli "你好"
```

使用真实 OpenAI-compatible 模型（如本地 vLLM / Ollama / LM Studio）：

```bash
cp .env.example .env
# 编辑 .env：设置 USE_REAL_MODEL=true、OPENAI_BASE_URL、OPENAI_API_KEY、OPENAI_MODEL
python -m agent_harness.cli "帮我搜索 refund policy"
```

每次执行会在当前目录生成 `traces.jsonl`（trace span）和 `audit.jsonl`（工具审计），每行一个 JSON。

## 运行测试

```bash
pytest -v
```

八组测试覆盖（共 40 个用例）：

- `test_agent_loop.py`：搜索输入会调用 `search_docs` 并最终返回 final_answer；时间输入调用 `get_time`；普通输入直接回答；max_steps 兜底。
- `test_tool_runtime.py`：工具不存在返回结构化错误；echo 正常返回；参数校验错误；search_docs 匹配。
- `test_trace_recorder.py`：生成 trace_id；JSONL 包含 model / tool span；父子关系；工具失败时 span 记录 error。
- `test_model_gateway.py`：timeout 可重试；auth_error / bad_request 不重试；rate_limit 触发 fallback。
- `test_tool_policy.py`：read_only 自动 allowed；write 无 scope denied；external_side_effect requires_confirmation；destructive blocked。
- `test_audit_logger.py`：工具成功 / 被拒绝 / 被阻断都写 audit log。
- `test_trace_tree.py`：trace 含 agent / model / tool / policy span；policy 是 tool 的子 span。
- `test_eval_runner.py`：5 个 eval case 都能运行，destructive_tool_blocked 必须 passed。

## 运行 eval

```bash
python -m agent_harness.cli eval
```

执行 5 个固定 case，输出每个 case 的 passed / reason / trace_id 和汇总。

## trace 字段

每行 JSONL 包含：`trace_id` `span_id` `parent_span_id` `span_type` `name` `status` `latency_ms` `input_summary` `output_summary` `error_type` `metadata` `started_at` `ended_at`。

`span_type` 取值：`agent` / `model` / `tool` / `retry` / `fallback` / `policy` / `audit` / `final`。

## 第二阶段能力

- **ModelGateway 错误治理**：统一 `ModelErrorType`（TIMEOUT/RATE_LIMIT/AUTH_ERROR/BAD_REQUEST/PROVIDER_UNAVAILABLE/RESPONSE_PARSE_ERROR/UNKNOWN），`ResilientModelGateway` 支持 timeout_ms / max_retries / exponential backoff / fallback client。AUTH_ERROR 和 BAD_REQUEST 不重试，主模型重试失败后 fallback。每次 retry / fallback 写 trace span。`MockModelClient` 支持 always_success / timeout_once_then_success / rate_limit_then_fallback / auth_error / bad_request 五种场景。
- **ToolRuntime 权限和风险控制**：`ToolSpec` 增加 version / risk_level / required_scopes / idempotent / side_effect_type。`ExecutionContext` 携带 user_id / tenant_id / scopes / trace_id / require_confirmation。执行前依次做 schema validation → permission check → risk policy check。`ToolPolicy` 返回 allowed / denied / requires_confirmation / blocked。
- **内置工具调整**：echo / get_time / search_docs（read_only），新增 create_ticket（write，需 ticket:write）和 delete_record（destructive，默认阻断）。
- **ToolResult 增强**：status（success/error/denied/requires_confirmation/blocked）+ content + error_type + retryable + metadata。
- **Audit Log**：`AuditLogger` 写 audit.jsonl，每次工具调用无论结果都记录，含 trace_id / user_id / tenant_id / agent_id / tool_name / tool_version / risk_level / permission_decision / input_summary / output_summary / status / error_type / latency_ms / timestamp。
- **TraceRecorder 链路增强**：trace tree，AgentRunner 是 root span，model / tool 是 agent 子 span，retry / fallback 是 model 子 span，policy / audit 是 tool 子 span。
- **EvalRunner 最小评测**：5 个固定 case（normal_search_success / model_timeout_retry_success / model_rate_limit_fallback_success / write_tool_permission_denied / destructive_tool_blocked），CLI `eval` 命令。

## 设计要点

- **类型契约集中**：`types.py` 定义所有跨模块数据结构，模块间只传类型不传 dict，便于替换实现。
- **模型客户端可替换**：`ModelClient` 是 Protocol，Mock 和 OpenAI-compatible 两个实现满足同一接口，AgentRunner 不感知具体实现。
- **工具错误结构化**：ToolRuntime 把「工具不存在 / 参数错误 / 执行异常 / 权限不足 / 高风险阻断」都收口成结构化 `ToolResult`，不抛异常给上层，模拟企业级平台对工具失败的统一处理。
- **Trace 贯穿全程**：每个 model call、tool call、retry、fallback、policy check、audit write 都是一个 span，带 latency 和 parent，形成可观测的调用树。
- **治理与执行分离**：ToolPolicy 只做判断不执行，ToolRuntime 依赖 Policy 结果决定是否真正调用 handler，便于审计和替换策略。
