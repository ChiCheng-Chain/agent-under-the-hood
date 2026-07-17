# agent_harness_demo

最小 Agent Harness 闭环，用于学习企业级 Agent 平台的基础结构。不依赖 LangChain / LangGraph / CrewAI 等 Agent 框架，只保留底层结构。

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
│   ├── types.py      # 统一类型契约（Message / ModelRequest / ModelResponse / ToolResult / Span）
│   ├── config.py     # 配置加载（.env / 环境变量）
│   ├── gateway.py    # ModelGateway：MockModelClient + OpenAICompatibleClient
│   ├── tools.py      # ToolRegistry + ToolRuntime + 内置工具(echo/get_time/search_docs)
│   ├── trace.py      # TraceRecorder：trace_id + span，写 JSONL
│   ├── runner.py     # AgentRunner：主循环
│   ├── cli.py        # 命令行入口
│   └── __main__.py   # 支持 python -m agent_harness
├── tests/
│   ├── test_agent_loop.py
│   ├── test_tool_runtime.py
│   └── test_trace_recorder.py
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

每次执行会在当前目录生成 `traces.jsonl`，每行一个 span。

## 运行测试

```bash
pytest -v
```

三组测试覆盖：

- `test_agent_loop.py`：搜索输入会调用 `search_docs` 并最终返回 final_answer；时间输入调用 `get_time`；普通输入直接回答；max_steps 兜底。
- `test_tool_runtime.py`：工具不存在返回结构化错误；echo 正常返回；参数校验错误；search_docs 匹配。
- `test_trace_recorder.py`：生成 trace_id；JSONL 包含 model / tool span；父子关系；工具失败时 span 记录 error。

## trace 字段

每行 JSONL 包含：`trace_id` `span_id` `parent_span_id` `span_type` `status` `latency_ms` `input_summary` `output_summary` `error_type`。

`span_type` 取值：`run` / `model` / `tool` / `final`。

## 设计要点

- **类型契约集中**：`types.py` 定义所有跨模块数据结构，模块间只传类型不传 dict，便于替换实现。
- **模型客户端可替换**：`ModelClient` 是 Protocol，Mock 和 OpenAI-compatible 两个实现满足同一接口，AgentRunner 不感知具体实现。
- **工具错误结构化**：ToolRuntime 把「工具不存在 / 参数错误 / 执行异常」都收口成 `ToolResult(ok=False, error_type=...)`，不抛异常给上层，模拟企业级平台对工具失败的统一处理。
- **Trace 贯穿全程**：每个 model call 和 tool call 都是一个 span，带 latency 和 parent，形成可观测的调用树。
