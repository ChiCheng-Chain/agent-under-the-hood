"""Agent 主循环。

链路：
    用户输入 -> AgentRunner -> ModelGateway -> 模型输出 tool_call/final_answer
        -> tool_call 则走 ToolRuntime -> observation 回喂 -> 下一轮
        -> final_answer 则结束
    全程 TraceRecorder 记录每个 model span 与 tool span。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .gateway import ModelClient, ModelError
from .tools import ToolRuntime, build_default_registry
from .trace import TraceRecorder
from .types import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    SpanStatus,
    SpanType,
    ToolCall,
    ToolSpec,
)


@dataclass
class AgentResult:
    """一次 agent 任务的最终结果。"""

    final_answer: str
    trace_id: str
    steps: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: Optional[str] = None


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
    ) -> None:
        self._model = model
        self._tools = tool_runtime or ToolRuntime(build_default_registry())
        self._trace = trace or TraceRecorder()
        self._max_steps = max_steps
        self._system_prompt = system_prompt

    @property
    def trace(self) -> TraceRecorder:
        return self._trace

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return self._tools._registry.list_tools()

    def run(self, user_input: str) -> AgentResult:
        """执行一次 agent 任务。"""

        run_span = self._trace.start_span(SpanType.RUN, user_input)
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=user_input),
        ]
        executed_tool_calls: list[ToolCall] = []

        for step in range(1, self._max_steps + 1):
            # 1. 调模型
            request = ModelRequest(messages=messages, tools=self.tool_specs)
            model_input_summary = _summary_messages(messages)
            try:
                response = self._model.invoke(request)
            except ModelError as exc:
                self._trace.end_span(
                    run_span,
                    SpanStatus.ERROR,
                    output_summary=str(exc),
                    error_type=type(exc).__name__,
                )
                return AgentResult(
                    final_answer="",
                    trace_id=self._trace.trace_id,
                    steps=step - 1,
                    tool_calls=executed_tool_calls,
                    error=str(exc),
                )

            self._trace.record_model_call(
                input_summary=model_input_summary,
                output_summary=_summary_response(response),
            )

            # 2. final_answer -> 结束
            if response.finish_reason == FinishReason.FINAL_ANSWER:
                final = response.content or "(空答案)"
                self._trace.record_final(final)
                self._trace.end_span(run_span, SpanStatus.OK, final)
                return AgentResult(
                    final_answer=final,
                    trace_id=self._trace.trace_id,
                    steps=step,
                    tool_calls=executed_tool_calls,
                )

            # 3. tool_call -> 执行工具，observation 回喂
            if response.finish_reason == FinishReason.TOOL_CALL:
                # 先把 assistant 的 tool_call 消息加入历史
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                for call in response.tool_calls:
                    result = self._tools.execute(call.name, call.arguments)
                    executed_tool_calls.append(call)
                    output_str = (
                        _stringify(result.output) if result.ok else f"ERROR: {result.error_message}"
                    )
                    self._trace.record_tool_call(
                        tool_name=call.name,
                        input_summary=_stringify(call.arguments),
                        output_summary=output_str,
                        status=SpanStatus.OK if result.ok else SpanStatus.ERROR,
                        error_type=result.error_type.value if result.error_type else None,
                    )
                    # observation 作为 tool 角色消息回喂
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=output_str,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                # 继续下一轮
                continue

            # 未知 finish_reason，按错误处理
            self._trace.end_span(
                run_span,
                SpanStatus.ERROR,
                output_summary=f"未知 finish_reason: {response.finish_reason}",
                error_type="UnknownFinishReason",
            )
            return AgentResult(
                final_answer="",
                trace_id=self._trace.trace_id,
                steps=step,
                tool_calls=executed_tool_calls,
                error=f"未知 finish_reason: {response.finish_reason}",
            )

        # 达到最大步数仍未结束
        msg = f"达到最大循环次数 {self._max_steps} 仍未得到最终答案"
        self._trace.end_span(
            run_span,
            SpanStatus.ERROR,
            output_summary=msg,
            error_type="MaxStepsExceeded",
        )
        return AgentResult(
            final_answer="",
            trace_id=self._trace.trace_id,
            steps=self._max_steps,
            tool_calls=executed_tool_calls,
            error=msg,
        )


# --------------------------------------------------------------------------- #
# 摘要工具
# --------------------------------------------------------------------------- #
def _summary_messages(messages: list[Message]) -> str:
    parts = []
    for m in messages[-3:]:  # 只取最近 3 条，避免摘要过长
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


def _stringify(obj: object) -> str:
    if isinstance(obj, str):
        return obj
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(obj)


# 避免未使用警告
_ = time
