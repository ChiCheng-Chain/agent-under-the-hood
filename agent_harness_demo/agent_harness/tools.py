"""工具注册表、运行时与内置工具（第二阶段增强）。

- ToolRegistry：注册 / 查找工具，产出 ToolSpec 给模型
- ToolRuntime：执行前做 schema validation + permission check + risk policy check，
  执行后写 audit log，返回统一 ToolResult
- 内置工具：echo / get_time / search_docs（read_only）+ create_ticket（write）
  + delete_record（destructive）
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError, create_model

from .audit import AuditLogger
from .policy import ToolPolicy
from .trace import TraceRecorder
from .types import (
    ExecutionContext,
    PermissionDecision,
    RiskLevel,
    SideEffectType,
    ToolErrorType,
    ToolResult,
    ToolSpec,
    ToolStatus,
)

ToolHandler = Callable[..., Any]


class RegisteredTool(BaseModel):
    """一个已注册工具的完整定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    handler: ToolHandler
    version: str = "v1"
    required_scopes: list[str] = []
    idempotent: bool = True
    side_effect_type: SideEffectType = SideEffectType.NONE

    model_config = {"arbitrary_types_allowed": True}

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            risk_level=self.risk_level,
            version=self.version,
            required_scopes=list(self.required_scopes),
            idempotent=self.idempotent,
            side_effect_type=self.side_effect_type,
        )


# --------------------------------------------------------------------------- #
# 内置 demo 文档库
# --------------------------------------------------------------------------- #
DEMO_DOCS: list[str] = [
    "退款政策（refund policy）：购买后 7 天内可全额退款，超过 7 天按 50% 退款。",
    "退货流程（return process）：请联系客服提交退货申请，并在 14 天内寄回商品。",
    "隐私政策（privacy policy）：我们不会向第三方出售您的个人数据。",
    "发货时间（shipping time）：现货商品 24 小时内发货，预售商品按页面标注时间。",
]

# 记录被实际执行的 handler（测试用来断言 delete_record 是否被调用）
_EXECUTED_HANDLERS: list[str] = []


def _reset_executed_handlers() -> None:
    _EXECUTED_HANDLERS.clear()


# --------------------------------------------------------------------------- #
# 内置工具 handler
# --------------------------------------------------------------------------- #
def _echo(text: str = "") -> str:
    return text


def _get_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _search_docs(query: str = "") -> list[str]:
    if not query:
        return []
    q = query.lower()
    return [doc for doc in DEMO_DOCS if q in doc.lower() or query in doc]


def _create_ticket(title: str = "未命名工单", priority: str = "normal") -> dict[str, Any]:
    """模拟创建工单，返回 mock ticket_id。"""
    ticket_id = f"TKT-{int(time.time() * 1000) % 100000:05d}"
    return {"ticket_id": ticket_id, "title": title, "priority": priority, "status": "created"}


def _delete_record(record_id: str = "") -> dict[str, Any]:
    """删除记录。默认策略下不应被调用。"""
    _EXECUTED_HANDLERS.append("delete_record")
    return {"record_id": record_id, "deleted": True}


# --------------------------------------------------------------------------- #
# ToolRegistry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
        risk_level: RiskLevel = RiskLevel.READ_ONLY,
        *,
        version: str = "v1",
        required_scopes: Optional[list[str]] = None,
        idempotent: bool = True,
        side_effect_type: SideEffectType = SideEffectType.NONE,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"工具已存在: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            risk_level=risk_level,
            handler=handler,
            version=version,
            required_scopes=list(required_scopes) if required_scopes else [],
            idempotent=idempotent,
            side_effect_type=side_effect_type,
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools


def build_default_registry() -> ToolRegistry:
    """构造包含五个内置工具的注册表。

    第一阶段三个工具调整为 read_only，第二阶段新增 create_ticket / delete_record。
    """

    reg = ToolRegistry()
    reg.register(
        name="echo",
        description="原样返回输入文本，用于测试工具链路是否通畅。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": [],
        },
        handler=_echo,
        risk_level=RiskLevel.READ_ONLY,
        side_effect_type=SideEffectType.NONE,
    )
    reg.register(
        name="get_time",
        description="返回当前时间（ISO 8601 字符串）。",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_time,
        risk_level=RiskLevel.READ_ONLY,
        side_effect_type=SideEffectType.NONE,
    )
    reg.register(
        name="search_docs",
        description="在 demo 文档库里做关键词搜索，返回匹配到的文档列表。",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        handler=_search_docs,
        risk_level=RiskLevel.READ_ONLY,
        side_effect_type=SideEffectType.NONE,
    )
    reg.register(
        name="create_ticket",
        description="创建一个工单，返回 ticket_id。",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "工单标题"},
                "priority": {"type": "string", "description": "优先级"},
            },
            "required": [],
        },
        handler=_create_ticket,
        risk_level=RiskLevel.WRITE,
        required_scopes=["ticket:write"],
        idempotent=False,
        side_effect_type=SideEffectType.INTERNAL_WRITE,
    )
    reg.register(
        name="delete_record",
        description="删除一条记录。高风险，默认阻断，不自动执行。",
        input_schema={
            "type": "object",
            "properties": {"record_id": {"type": "string", "description": "记录ID"}},
            "required": ["record_id"],
        },
        handler=_delete_record,
        risk_level=RiskLevel.DESTRUCTIVE,
        required_scopes=["record:delete"],
        idempotent=False,
        side_effect_type=SideEffectType.DESTRUCTIVE,
    )
    return reg


# --------------------------------------------------------------------------- #
# ToolRuntime
# --------------------------------------------------------------------------- #
class ToolRuntime:
    """执行工具调用。

    执行前依次做：
    1. schema validation
    2. permission check（ToolPolicy）
    3. risk policy check（ToolPolicy）

    执行后写 audit log（无论成功/失败/拒绝/确认/阻断）。
    所有失败路径返回结构化 ToolResult，不抛未处理异常。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: Optional[ToolPolicy] = None,
        audit: Optional[AuditLogger] = None,
        trace: Optional[TraceRecorder] = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or ToolPolicy()
        self._audit = audit
        self._trace = trace

    @property
    def _registry_ref(self) -> ToolRegistry:
        """供 AgentRunner 取 tool_specs 用。"""
        return self._registry

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        ctx: Optional[ExecutionContext] = None,
        parent_span_id: Optional[str] = None,
    ) -> ToolResult:
        arguments = arguments or {}
        ctx = ctx or ExecutionContext()
        start = time.perf_counter()

        # tool span（policy / audit 作为它的子 span）
        tool_span = self._start_tool_span(name, arguments, parent_span_id)

        tool = self._registry.get(name)
        if tool is None:
            result = ToolResult(
                status=ToolStatus.ERROR,
                error_type=ToolErrorType.NOT_FOUND,
                error_message=f"工具不存在: {name}",
                latency_ms=_elapsed_ms(start),
            )
            self._finish_tool_span(tool_span, name, result)
            self._audit_log(name, "v1", RiskLevel.READ_ONLY, PermissionDecision.DENIED,
                            arguments, result, ctx, start)
            return result

        spec = tool.to_spec()

        # 1. schema validation
        validated = self._validate(tool, arguments)
        if validated is None:
            result = ToolResult(
                status=ToolStatus.ERROR,
                error_type=ToolErrorType.VALIDATION_ERROR,
                error_message=f"参数校验失败: {name}",
                latency_ms=_elapsed_ms(start),
            )
            self._finish_tool_span(tool_span, name, result)
            self._audit_log(name, spec.version, spec.risk_level, PermissionDecision.DENIED,
                            arguments, result, ctx, start)
            return result

        # 2+3. permission & risk policy check
        policy_result = self._policy.check(spec, ctx)
        self._record_policy_span(name, policy_result, tool_span)

        decision = policy_result.decision
        if decision == PermissionDecision.DENIED:
            result = ToolResult(
                status=ToolStatus.DENIED,
                error_type=ToolErrorType.PERMISSION_DENIED,
                error_message=policy_result.reason,
                metadata={"missing_scopes": policy_result.missing_scopes},
                latency_ms=_elapsed_ms(start),
            )
        elif decision == PermissionDecision.REQUIRES_CONFIRMATION:
            result = ToolResult(
                status=ToolStatus.REQUIRES_CONFIRMATION,
                error_type=ToolErrorType.REQUIRES_CONFIRMATION,
                error_message=policy_result.reason,
                retryable=False,
                latency_ms=_elapsed_ms(start),
            )
        elif decision == PermissionDecision.BLOCKED:
            result = ToolResult(
                status=ToolStatus.BLOCKED,
                error_type=ToolErrorType.BLOCKED,
                error_message=policy_result.reason,
                latency_ms=_elapsed_ms(start),
            )
        else:
            # ALLOWED：真正执行 handler
            try:
                output = tool.handler(**validated)
            except Exception as exc:  # noqa: BLE001
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    error_type=ToolErrorType.EXECUTION_ERROR,
                    error_message=f"{type(exc).__name__}: {exc}",
                    retryable=False,
                    latency_ms=_elapsed_ms(start),
                )
            else:
                result = ToolResult(
                    status=ToolStatus.SUCCESS,
                    content=output,
                    latency_ms=_elapsed_ms(start),
                )

        self._finish_tool_span(tool_span, name, result)
        self._audit_log(name, spec.version, spec.risk_level, decision,
                        validated, result, ctx, start)
        return result

    # ----- span 辅助 -------------------------------------------------------- #
    def _start_tool_span(
        self, name: str, arguments: dict[str, Any], parent_span_id: Optional[str]
    ) -> Optional[str]:
        if self._trace is None:
            return None
        return self._trace.start_span(
            _span_type_tool(),
            f"{name} | {_stringify(arguments)}",
            name=f"tool:{name}",
            parent_span_id=parent_span_id,
        )

    def _finish_tool_span(self, span_id: Optional[str], name: str, result: ToolResult) -> None:
        if self._trace is None or span_id is None:
            return
        status = _tool_status_to_span(result.status)
        err = result.error_type.value if result.error_type else None
        self._trace.end_span(span_id, status, f"{name} | {result.status.value}", err)

    def _record_policy_span(
        self, name: str, policy_result, parent_span_id: Optional[str]
    ) -> None:
        if self._trace is None:
            return
        self._trace.record_policy(
            tool_name=name,
            decision=policy_result.decision.value,
            reason=policy_result.reason,
            parent_span_id=parent_span_id,
        )

    # ----- audit 辅助 ------------------------------------------------------- #
    def _audit_log(
        self,
        name: str,
        version: str,
        risk_level: RiskLevel,
        decision: PermissionDecision,
        arguments: dict[str, Any],
        result: ToolResult,
        ctx: ExecutionContext,
        start: float,
    ) -> None:
        if self._audit is None:
            return
        # audit 作为 tool span 的子 span
        if self._trace is not None:
            # 注意：audit span 的 parent 由调用时栈顶决定，这里手动记一条
            self._trace.record_audit(name, result.status.value)

        self._audit.log(
            trace_id=ctx.trace_id or (self._trace.trace_id if self._trace else ""),
            user_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
            agent_id=ctx.agent_id,
            tool_name=name,
            tool_version=version,
            risk_level=risk_level,
            permission_decision=decision,
            input_summary=_stringify(arguments),
            output_summary=_stringify(result.content if result.is_success else result.error_message),
            status=result.status,
            error_type=result.error_type.value if result.error_type else None,
            latency_ms=_elapsed_ms(start),
        )

    # ----- 校验 ------------------------------------------------------------- #
    def _validate(
        self, tool: RegisteredTool, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        schema = tool.input_schema
        props: dict[str, Any] = schema.get("properties", {}) or {}
        required: list[str] = schema.get("required", []) or []

        fields: dict[str, Any] = {}
        type_map = {
            "string": (str, ...),
            "integer": (int, ...),
            "number": (float, ...),
            "boolean": (bool, ...),
            "array": (list, ...),
            "object": (dict, ...),
        }
        for field_name, field_schema in props.items():
            py_type, _ = type_map.get(field_schema.get("type", "string"), (str, ...))
            if field_name in required:
                fields[field_name] = (py_type, ...)
            else:
                fields[field_name] = (py_type, None)

        try:
            Model = create_model(f"{tool.name}_args", **fields)  # noqa: N806
            # pydantic v2 默认严格模式，int 传给 str 字段会报错
            instance = Model(**arguments)
            return instance.model_dump(exclude_none=True)
        except ValidationError:
            return None


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def _span_type_tool():
    from .types import SpanType

    return SpanType.TOOL


def _tool_status_to_span(status: ToolStatus):
    from .types import SpanStatus

    if status == ToolStatus.SUCCESS:
        return SpanStatus.OK
    if status in (ToolStatus.DENIED, ToolStatus.REQUIRES_CONFIRMATION, ToolStatus.BLOCKED):
        return SpanStatus.INFO
    return SpanStatus.ERROR


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


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# 避免未使用警告
_ = BaseModel
