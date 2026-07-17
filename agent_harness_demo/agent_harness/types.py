"""统一类型定义。

整个 Agent Harness / Runtime 的数据契约集中在这里，模块之间只通过这些类型传递信息，
这样 ModelGateway / ToolRuntime / TraceRecorder / ToolPolicy / AuditLogger 可以独立替换。

第一阶段字段保留向后兼容，第二阶段在原有结构上扩展。
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
    tool_calls: list[ToolCall] = Field(default_factory=list)
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
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    # 第二阶段：调用超时（毫秒），由 Gateway 处理
    timeout_ms: Optional[int] = None


class FinishReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"
    ERROR = "error"


class ModelResponse(BaseModel):
    """模型客户端返回给 AgentRunner 的统一响应。"""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.FINAL_ANSWER
    raw: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ModelGateway 错误治理（第二阶段）
# --------------------------------------------------------------------------- #
class ModelErrorType(str, Enum):
    """模型错误的统一分类。决定是否重试。"""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    BAD_REQUEST = "bad_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RESPONSE_PARSE_ERROR = "response_parse_error"
    UNKNOWN = "unknown"


# 可重试的错误类型
RETRYABLE_ERRORS: frozenset[ModelErrorType] = frozenset(
    {
        ModelErrorType.TIMEOUT,
        ModelErrorType.RATE_LIMIT,
        ModelErrorType.PROVIDER_UNAVAILABLE,
    }
)


class ModelError(Exception):
    """模型调用错误。携带 error_type 供 Gateway 判断是否重试。"""

    def __init__(
        self,
        message: str,
        error_type: ModelErrorType = ModelErrorType.UNKNOWN,
        *,
        retryable: Optional[bool] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        # retryable 不传则按 RETRYABLE_ERRORS 推断
        self.retryable = retryable if retryable is not None else error_type in RETRYABLE_ERRORS
        self.status_code = status_code

    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


# 向后兼容：旧名保留为 ModelError 的别名
ModelConnectionError = ModelError
ModelAPIError = ModelError


# --------------------------------------------------------------------------- #
# 工具层类型
# --------------------------------------------------------------------------- #
class RiskLevel(str, Enum):
    """工具风险等级，决定执行前的策略检查。

    第二阶段取值与第一阶段不同：read_only / write / external_side_effect / destructive。
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    DESTRUCTIVE = "destructive"

    # 向后兼容旧值（safe/low/medium/high 仍能解析，统一映射为 read_only）
    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        legacy = {"safe", "low", "medium", "high"}
        if isinstance(value, str) and value in legacy:
            return cls.READ_ONLY
        return None


class SideEffectType(str, Enum):
    """副作用类型。"""

    NONE = "none"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_CALL = "external_call"
    DESTRUCTIVE = "destructive"


class ToolSpec(BaseModel):
    """工具规格，给模型用的能力描述。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    # 第二阶段新增字段
    version: str = "v1"
    required_scopes: list[str] = Field(default_factory=list)
    idempotent: bool = True
    side_effect_type: SideEffectType = SideEffectType.NONE


class ToolErrorType(str, Enum):
    """工具执行错误的分类。"""

    NOT_FOUND = "tool_not_found"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    # 第二阶段新增
    PERMISSION_DENIED = "permission_denied"
    BLOCKED = "blocked"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class ToolStatus(str, Enum):
    """工具执行状态。"""

    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    BLOCKED = "blocked"


class ToolResult(BaseModel):
    """工具执行结果。

    第二阶段字段：
    - status：统一状态
    - content：给 Agent 使用的内容
    - error_type：错误类型
    - retryable：是否可重试
    - metadata：附加信息

    第一阶段字段（ok/output/error_message/latency_ms）保留向后兼容。
    """

    status: ToolStatus = ToolStatus.SUCCESS
    content: Any = None
    error_type: Optional[ToolErrorType] = None
    error_message: Optional[str] = None
    retryable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0

    # 向后兼容字段
    ok: Optional[bool] = None
    output: Any = None

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        # 同步 ok/output 与 status/content
        if self.ok is None:
            self.ok = self.status == ToolStatus.SUCCESS
        if self.output is None:
            self.output = self.content
        if self.content is None:
            self.content = self.output

    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


class ExecutionContext(BaseModel):
    """工具执行上下文：用户/租户/权限/链路。"""

    user_id: str = "anonymous"
    tenant_id: str = "default"
    scopes: list[str] = Field(default_factory=list)
    trace_id: str = ""
    # 是否允许触发确认流程；为 False 时 requires_confirmation 直接转为 denied
    require_confirmation: bool = True
    agent_id: str = "default-agent"


class PermissionDecision(str, Enum):
    """ToolPolicy 的统一决策结果。"""

    ALLOWED = "allowed"
    DENIED = "denied"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    BLOCKED = "blocked"


class PolicyResult(BaseModel):
    """ToolPolicy 判断结果。"""

    decision: PermissionDecision
    reason: str = ""
    # 触发决策的工具/上下文摘要，便于审计
    missing_scopes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Trace 层类型
# --------------------------------------------------------------------------- #
class SpanType(str, Enum):
    """一次 trace 内的 span 类型。第二阶段扩展为完整 trace tree。"""

    AGENT = "agent"        # AgentRunner 根 span（第一阶段叫 run）
    RUN = "run"            # 向后兼容别名，序列化值仍是 run
    MODEL = "model"        # 一次模型调用
    TOOL = "tool"          # 一次工具调用
    RETRY = "retry"        # 模型重试
    FALLBACK = "fallback"  # 模型 fallback
    POLICY = "policy"      # 工具策略检查
    AUDIT = "audit"        # 审计写入
    EVAL = "eval"          # 评测
    FINAL = "final"        # 最终输出


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    # 第二阶段：策略类 span 可能是 info（如 requires_confirmation）
    INFO = "info"


class Span(BaseModel):
    """一条 trace 记录。直接序列化成 JSONL 的一行。"""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    span_type: SpanType
    name: str = ""
    status: SpanStatus
    latency_ms: int
    input_summary: str = ""
    output_summary: str = ""
    error_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


# --------------------------------------------------------------------------- #
# Audit 层类型
# --------------------------------------------------------------------------- #
class AuditRecord(BaseModel):
    """审计日志记录。写入 audit.jsonl 的一行。"""

    trace_id: str
    user_id: str
    tenant_id: str
    agent_id: str
    tool_name: str
    tool_version: str
    risk_level: str
    permission_decision: str
    input_summary: str
    output_summary: str
    status: str
    error_type: Optional[str] = None
    latency_ms: int
    timestamp: str
