"""5 个固定 eval case。

每个 case 构造自己的 AgentRunner（含特定 mock model / context），由 EvalRunner 执行。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..gateway import MockModelClient, ResilientModelGateway
from ..runner import AgentRunner, AgentResult
from ..tools import ToolRuntime, build_default_registry
from ..trace import TraceRecorder
from ..audit import AuditLogger
from ..types import (
    ExecutionContext,
    FinishReason,
    ModelResponse,
    ToolCall,
    ToolStatus,
)
from .types import EvalCase


def _build_runner(
    *,
    model,
    tmp_dir: Path,
    context: ExecutionContext | None = None,
    max_steps: int = 5,
) -> AgentRunner:
    """构造一个带 trace + audit 的 runner。

    如果 model 是 ResilientModelGateway，把 trace 注入进去，让 retry/fallback span 落盘。
    """
    trace = TraceRecorder(trace_file=str(tmp_dir / "traces.jsonl"))
    audit = AuditLogger(audit_file=str(tmp_dir / "audit.jsonl"))
    # 注入 trace 到弹性网关，使 retry/fallback span 被记录
    if isinstance(model, ResilientModelGateway):
        model._trace = trace
    runtime = ToolRuntime(build_default_registry(), trace=trace, audit=audit)
    return AgentRunner(
        model,
        tool_runtime=runtime,
        trace=trace,
        max_steps=max_steps,
        context=context,
    )


# --------------------------------------------------------------------------- #
# 自定义 mock 客户端：按输入关键词返回指定 tool_call
# --------------------------------------------------------------------------- #
class KeywordToolCallClient:
    """根据用户输入关键词决定调用哪个工具，收到 observation 后返回 final。"""

    def __init__(self, keyword_map: dict[str, str], *, name: str = "kw"):
        self._map = keyword_map  # keyword -> tool_name
        self._name = name
        self._first = True

    def invoke(self, request) -> ModelResponse:
        # 收到 observation -> final
        if any(m.role.value == "tool" for m in request.messages):
            tool_msg = next(m for m in reversed(request.messages) if m.role.value == "tool")
            return ModelResponse(
                content=f"完成：{tool_msg.content[:80]}",
                finish_reason=FinishReason.FINAL_ANSWER,
            )

        user_text = ""
        for m in reversed(request.messages):
            if m.role.value == "user":
                user_text = m.content
                break

        for keyword, tool_name in self._map.items():
            if keyword in user_text:
                args: dict[str, Any] = {}
                if tool_name == "search_docs":
                    args = {"query": user_text}
                elif tool_name == "create_ticket":
                    args = {"title": user_text}
                elif tool_name == "delete_record":
                    args = {"record_id": "rec-001"}
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALL,
                    tool_calls=[ToolCall(id=f"call-{tool_name}", name=tool_name, arguments=args)],
                )

        return ModelResponse(
            content=f"已收到：{user_text}", finish_reason=FinishReason.FINAL_ANSWER
        )


# --------------------------------------------------------------------------- #
# Case 1: 正常搜索成功
# --------------------------------------------------------------------------- #
def _case1(tmp_dir: Path) -> AgentRunner:
    return _build_runner(model=MockModelClient(behavior="always_success"), tmp_dir=tmp_dir)


def _assert_called_search_docs(result: AgentResult, runner: AgentRunner) -> tuple[bool, str]:
    names = [tc.name for tc in result.tool_calls]
    if "search_docs" not in names:
        return False, f"未调用 search_docs，实际: {names}"
    if result.error:
        return False, f"返回了 error: {result.error}"
    return True, "调用 search_docs 并成功返回"


# --------------------------------------------------------------------------- #
# Case 2: 模型超时重试后成功
# --------------------------------------------------------------------------- #
def _case2(tmp_dir: Path) -> AgentRunner:
    primary = MockModelClient(behavior="timeout_once_then_success")
    # 用 ResilientModelGateway 包一层，让它重试
    gateway = ResilientModelGateway(primary, max_retries=2, sleep=False)
    return _build_runner(model=gateway, tmp_dir=tmp_dir)


def _assert_retry_success(result: AgentResult, runner: AgentRunner) -> tuple[bool, str]:
    if result.error:
        return False, f"重试后仍失败: {result.error}"
    # 检查 trace 里有没有 retry span
    retry_spans = [s for s in runner.trace.spans if s.span_type.value == "retry"]
    if not retry_spans:
        return False, "trace 中没有 retry span"
    return True, f"发生 {len(retry_spans)} 次 retry 后成功"


# --------------------------------------------------------------------------- #
# Case 3: 模型限流触发 fallback 后成功
# --------------------------------------------------------------------------- #
def _case3(tmp_dir: Path) -> AgentRunner:
    primary = MockModelClient(behavior="rate_limit_then_fallback", name="primary")
    fallback = MockModelClient(behavior="always_success", name="fallback")
    gateway = ResilientModelGateway(
        primary, fallback=fallback, max_retries=1, sleep=False
    )
    return _build_runner(model=gateway, tmp_dir=tmp_dir)


def _assert_fallback_success(result: AgentResult, runner: AgentRunner) -> tuple[bool, str]:
    if result.error:
        return False, f"fallback 后仍失败: {result.error}"
    fallback_spans = [s for s in runner.trace.spans if s.span_type.value == "fallback"]
    if not fallback_spans:
        return False, "trace 中没有 fallback span"
    return True, "触发 fallback 后成功"


# --------------------------------------------------------------------------- #
# Case 4: write 工具权限拒绝
# --------------------------------------------------------------------------- #
def _case4(tmp_dir: Path) -> AgentRunner:
    model = KeywordToolCallClient({"创建": "create_ticket", "ticket": "create_ticket"})
    # 用户没有 ticket:write scope
    ctx = ExecutionContext(
        user_id="user-no-write",
        tenant_id="tenant-a",
        scopes=[],  # 没有 ticket:write
        require_confirmation=True,
    )
    return _build_runner(model=model, tmp_dir=tmp_dir, context=ctx)


def _assert_permission_denied(result: AgentResult, runner: AgentRunner) -> tuple[bool, str]:
    if not result.tool_results:
        return False, "没有工具调用结果"
    tr = result.tool_results[0]
    if tr.status != ToolStatus.DENIED:
        return False, f"期望 denied，实际 {tr.status.value}"
    return True, "create_ticket 被正确拒绝 (denied)"


# --------------------------------------------------------------------------- #
# Case 5: destructive 工具被阻断
# --------------------------------------------------------------------------- #
def _case5(tmp_dir: Path) -> AgentRunner:
    model = KeywordToolCallClient({"删除": "delete_record", "record": "delete_record"})
    # 即使用户有 record:delete scope，也默认 blocked
    ctx = ExecutionContext(
        user_id="user-admin",
        tenant_id="tenant-a",
        scopes=["record:delete"],  # 有 scope
        require_confirmation=True,
    )
    return _build_runner(model=model, tmp_dir=tmp_dir, context=ctx)


def _assert_destructive_blocked(result: AgentResult, runner: AgentRunner) -> tuple[bool, str]:
    if not result.tool_results:
        return False, "没有工具调用结果"
    tr = result.tool_results[0]
    if tr.status != ToolStatus.BLOCKED:
        return False, f"期望 blocked，实际 {tr.status.value}"
    # 校验 handler 没被执行
    from ..tools import _EXECUTED_HANDLERS

    if "delete_record" in _EXECUTED_HANDLERS:
        return False, "delete_record handler 被执行了"
    return True, "delete_record 被阻断，handler 未执行"


# --------------------------------------------------------------------------- #
# 工厂：返回所有 case（tmp_dir 由 EvalRunner 注入）
# --------------------------------------------------------------------------- #
def build_cases(tmp_dir: Path) -> list[EvalCase]:
    return [
        EvalCase(
            case_id="normal_search_success",
            description="输入搜索 refund policy，期望调用 search_docs 并成功",
            setup=lambda: _case1(tmp_dir),
            user_input="搜索 refund policy",
            assert_fn=_assert_called_search_docs,
        ),
        EvalCase(
            case_id="model_timeout_retry_success",
            description="主模型超时一次后重试成功",
            setup=lambda: _case2(tmp_dir),
            user_input="搜索 refund policy",
            assert_fn=_assert_retry_success,
        ),
        EvalCase(
            case_id="model_rate_limit_fallback_success",
            description="主模型限流触发 fallback 后成功",
            setup=lambda: _case3(tmp_dir),
            user_input="搜索 refund policy",
            assert_fn=_assert_fallback_success,
        ),
        EvalCase(
            case_id="write_tool_permission_denied",
            description="用户无 ticket:write，create_ticket 被拒绝",
            setup=lambda: _case4(tmp_dir),
            user_input="帮我创建 ticket",
            assert_fn=_assert_permission_denied,
        ),
        EvalCase(
            case_id="destructive_tool_blocked",
            description="delete_record 即使有 scope 也被阻断",
            setup=lambda: _case5(tmp_dir),
            user_input="帮我删除 record",
            assert_fn=_assert_destructive_blocked,
        ),
    ]
