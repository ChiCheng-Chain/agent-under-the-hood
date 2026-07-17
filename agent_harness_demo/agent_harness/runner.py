"""Agent 主循环（第二阶段增强）。

链路：
    用户输入 -> AgentRunner -> ModelGateway(含 retry/fallback) -> 模型输出
        -> tool_call 则走 ToolRuntime(schema 校验 + policy + audit) -> observation 回喂
        -> final_answer 则结束
    全程 TraceRecorder 记 trace tree：agent(root) -> model / tool，tool -> policy / audit。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .gateway import ModelClient, ModelError
from .tools import ToolRuntime, build_default_registry
from .trace import TraceRecorder
from .types import (
    ExecutionContext,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    SpanStatus,
    SpanType,
    ToolCall,
    ToolSpec,
    ToolStatus,
)


@dataclass
class AgentResult:
    """一次 agent 任务的最终结果。"""

    final_answer: str
    trace_id: str
    steps: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: Optional[str] = None
    tool_results: list = field(default_factory=list)


class AgentRunner:
    """Agent 主循环。"""

    def __init__(
        self,
        model: ModelClient,
        *,
        tool_runtime: Optional[ToolRuntime] = None,
        trace: Optional[TraceRecorder] = None,
        max_steps: int = 5,
        system_prompt: str = "你是一个能调用工具的助手。必要时调用工具获取信息，然后给出最终答案。",
        context: Optional[ExecutionContext] = None,
    ) -> None:
        self._trace = trace or TraceRecorder()
        # 把 trace 注入 tool_runtime，让 tool/policy/audit span 挂到同一棵树
        if tool_runtime is None:
            tool_runtime = ToolRuntime(build_default_registry(), trace=self._trace)
        else:
            # 已有 runtime 也确保共享 trace
            tool_runtime._trace = self._trace
        self._model = model
        self._tools = tool_runtime
        self._max_steps = max_steps
        self._system_prompt = system_prompt
        self._context = context or ExecutionContext(trace_id=self._trace.trace_id)

    @property
    def trace(self) -> TraceRecorder:
        return self._trace

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return self._tools._registry_ref.list_tools()

    @property
    def context(self) -> ExecutionContext:
        return self._context

    def run(self, user_input: str) -> AgentResult:
        """执行一次 agent 任务。"""

        # root span：agent
        agent_span = self._trace.start_span(
            SpanType.AGENT, user_input, name="agent_run"
        )
        # 同步 context 的 trace_id
        self._context.trace_id = self._trace.trace_id

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=user_input),
        ]
        executed_tool_calls: list[ToolCall] = []
        tool_results: list = []

        for step in range(1, self._max_steps + 1):
            # 1. 调模型（model span 包裹，retry/fallback 成为它的子 span）
            request = ModelRequest(messages=messages, tools=self.tool_specs)
            model_input_summary = _summary_messages(messages)
            model_span = self._trace.start_span(
                SpanType.MODEL, model_input_summary, name=f"model_step_{step}"
            )
            try:
                response = self._model.invoke(request)
            except ModelError as exc:
                self._trace.end_span(
                    model_span, SpanStatus.ERROR, str(exc), exc.error_type.value
                )
                self._trace.end_span(
                    agent_span, SpanStatus.ERROR, str(exc), exc.error_type.value
                )
                return AgentResult(
                    final_answer="",
                    trace_id=self._trace.trace_id,
                    steps=step - 1,
                    tool_calls=executed_tool_calls,
                    tool_results=tool_results,
                    error=str(exc),
                )

            self._trace.end_span(
                model_span, SpanStatus.OK, _summary_response(response)
            )

            # 2. final_answer -> 结束
            if response.finish_reason == FinishReason.FINAL_ANSWER:
                final = response.content or "(空答案)"
                self._trace.record_final(final)
                self._trace.end_span(agent_span, SpanStatus.OK, final)
                return AgentResult(
                    final_answer=final,
                    trace_id=self._trace.trace_id,
                    steps=step,
                    tool_calls=executed_tool_calls,
                    tool_results=tool_results,
                )

            # 3. tool_call -> 执行工具，observation 回喂
            if response.finish_reason == FinishReason.TOOL_CALL:
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                for call in response.tool_calls:
                    # ToolRuntime 内部记 tool/policy/audit span（agent 的子 span）
                    result = self._tools.execute(
                        call.name, call.arguments,
                        ctx=self._context,
                        parent_span_id=agent_span,
                    )
                    executed_tool_calls.append(call)
                    tool_results.append(result)
                    output_str = _tool_result_to_observation(result)
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=output_str,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                continue

            # 未知 finish_reason
            self._trace.end_span(
                agent_span,
                SpanStatus.ERROR,
                f"未知 finish_reason: {response.finish_reason}",
                "UnknownFinishReason",
            )
            return AgentResult(
                final_answer="",
                trace_id=self._trace.trace_id,
                steps=step,
                tool_calls=executed_tool_calls,
                tool_results=tool_results,
                error=f"未知 finish_reason: {response.finish_reason}",
            )

        # 达到最大步数仍未结束
        msg = f"达到最大循环次数 {self._max_steps} 仍未得到最终答案"
        self._trace.end_span(
            agent_span, SpanStatus.ERROR, msg, "MaxStepsExceeded"
        )
        return AgentResult(
            final_answer="",
            trace_id=self._trace.trace_id,
            steps=self._max_steps,
            tool_calls=executed_tool_calls,
            tool_results=tool_results,
            error=msg,
        )


# --------------------------------------------------------------------------- #
# 摘要与转换工具
# --------------------------------------------------------------------------- #
def _summary_messages(messages: list[Message]) -> str:
    parts = []
    for m in messages[-3:]:
        role = m.role.value
        if m.role == Role.TOOL:
            parts.append(f"tool({m.name})={m.content[:80]}")
        elif m.tool_calls:
            parts.append(f"assistant[tool_calls={[tc.name for tc in m.tool_calls]}]")
        else:
            parts.append(f"{role}={m.content[:80]}")
    return " | ".join(parts)


def _summary_response(resp: ModelResponse) -> str:
    if resp.finish_reason == FinishReason.TOOL_CALL:
        names = [tc.name for tc in resp.tool_calls]
        return f"tool_call={names}"
    return f"final={resp.content[:80]}"


def _tool_result_to_observation(result) -> str:
    """把 ToolResult 转成回喂给模型的 observation 字符串。"""

    if result.is_success:
        return _stringify(result.content)
    # 失败/拒绝/阻断：把状态和原因告诉模型
    parts = [f"status={result.status.value}"]
    if result.error_type:
        parts.append(f"error={result.error_type.value}")
    if result.error_message:
        parts.append(f"message={result.error_message}")
    return "; ".join(parts)


def _stringify(obj: object) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(obj)


# 避免未使用警告
_ = ToolStatus
