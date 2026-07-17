"""统一类型定义。

整个 Agent Harness 的数据契约集中在这里，模块之间只通过这些类型传递信息，
这样 ModelGateway / ToolRuntime / TraceRecorder 可以独立替换。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 模型层类型
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """对话消息。tool 角色的消息用于把工具 observation 回喂给模型。"""

    role: Role
    content: str = ""
    # 当 assistant 消息包含工具调用时记录在这里
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # 当 role=tool 时，记录这是哪个工具的返回
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolCall(BaseModel):
    """模型决定调用某个工具。"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    """发给模型客户端的统一请求。"""

    messages: list[Message]
    # 可用工具的 schema 列表，由 AgentRunner 注入
    tools: list[ToolSpec] = Field(default_factory=list)
    # 透传的调用参数（温度等），第一版只占位
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class FinishReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class ModelResponse(BaseModel):
    """模型客户端返回给 AgentRunner 的统一响应。

    finish_reason=FINAL_ANSWER 时 content 是最终答案；
    finish_reason=TOOL_CALL 时 tool_calls 非空。
    """

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.FINAL_ANSWER
    raw: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 工具层类型
# --------------------------------------------------------------------------- #
class RiskLevel(str, Enum):
    """工具风险等级，企业级平台常用来决定是否需要人工审批。"""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSpec(BaseModel):
    """工具规格，给模型用的能力描述。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel = RiskLevel.SAFE


class ToolErrorType(str, Enum):
    """工具执行错误的分类。"""

    NOT_FOUND = "tool_not_found"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"


class ToolResult(BaseModel):
    """工具执行结果。ok=False 时 error_type/error_message 描述失败原因。"""

    ok: bool
    output: Any = None
    error_type: Optional[ToolErrorType] = None
    error_message: Optional[str] = None
    latency_ms: int = 0


# --------------------------------------------------------------------------- #
# Trace 层类型
# --------------------------------------------------------------------------- #
class SpanType(str, Enum):
    """一次 trace 内的 span 类型。"""

    RUN = "run"           # 整个 agent 任务的根 span
    MODEL_CALL = "model"  # 一次模型调用
    TOOL_CALL = "tool"    # 一次工具调用
    FINAL = "final"       # 最终输出


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class Span(BaseModel):
    """一条 trace 记录。直接序列化成 JSONL 的一行。"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    span_type: SpanType
    status: SpanStatus
    latency_ms: int
    input_summary: str
    output_summary: str
    error_type: Optional[str] = None
