"""测试 ToolPolicy 权限与风险判断（第二阶段）。"""
from __future__ import annotations

from agent_harness.policy import ToolPolicy
from agent_harness.types import (
    ExecutionContext,
    PermissionDecision,
    RiskLevel,
    SideEffectType,
    ToolSpec,
)


def _spec(
    risk_level: RiskLevel,
    required_scopes: list[str] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name="t",
        description="d",
        input_schema={"type": "object", "properties": {}},
        risk_level=risk_level,
        required_scopes=required_scopes or [],
        side_effect_type=SideEffectType.NONE,
    )


def _ctx(scopes: list[str] | None = None, require_confirmation: bool = True) -> ExecutionContext:
    return ExecutionContext(
        user_id="u1",
        tenant_id="t1",
        scopes=scopes or [],
        require_confirmation=require_confirmation,
    )


def test_read_only_auto_allowed():
    """read_only 有 schema 即自动放行。"""
    policy = ToolPolicy()
    result = policy.check(_spec(RiskLevel.READ_ONLY), _ctx(scopes=[]))
    assert result.decision == PermissionDecision.ALLOWED


def test_write_without_scope_denied():
    """write 工具无 scope 则 denied。"""
    policy = ToolPolicy()
    spec = _spec(RiskLevel.WRITE, required_scopes=["ticket:write"])
    result = policy.check(spec, _ctx(scopes=[]))
    assert result.decision == PermissionDecision.DENIED
    assert "ticket:write" in result.missing_scopes


def test_write_with_scope_allowed():
    """write 工具有 scope 则 allowed。"""
    policy = ToolPolicy()
    spec = _spec(RiskLevel.WRITE, required_scopes=["ticket:write"])
    result = policy.check(spec, _ctx(scopes=["ticket:write"]))
    assert result.decision == PermissionDecision.ALLOWED


def test_external_side_effect_requires_confirmation():
    """external_side_effect 有 scope 但需要确认。"""
    policy = ToolPolicy()
    spec = _spec(
        RiskLevel.EXTERNAL_SIDE_EFFECT,
        required_scopes=["notify:send"],
    )
    result = policy.check(spec, _ctx(scopes=["notify:send"], require_confirmation=True))
    assert result.decision == PermissionDecision.REQUIRES_CONFIRMATION


def test_external_side_effect_no_scope_denied():
    """external_side_effect 无 scope 先被 denied。"""
    policy = ToolPolicy()
    spec = _spec(RiskLevel.EXTERNAL_SIDE_EFFECT, required_scopes=["notify:send"])
    result = policy.check(spec, _ctx(scopes=[]))
    assert result.decision == PermissionDecision.DENIED


def test_destructive_blocked_even_with_scope():
    """destructive 即使有 scope 也默认 blocked。"""
    policy = ToolPolicy()
    spec = _spec(RiskLevel.DESTRUCTIVE, required_scopes=["record:delete"])
    result = policy.check(spec, _ctx(scopes=["record:delete"]))
    assert result.decision == PermissionDecision.BLOCKED


def test_external_side_effect_confirmation_disabled_denied():
    """require_confirmation=False 时，external_side_effect 降级为 denied。"""
    policy = ToolPolicy()
    spec = _spec(RiskLevel.EXTERNAL_SIDE_EFFECT, required_scopes=["notify:send"])
    result = policy.check(spec, _ctx(scopes=["notify:send"], require_confirmation=False))
    assert result.decision == PermissionDecision.DENIED
