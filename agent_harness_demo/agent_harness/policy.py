"""工具策略与权限判断（第二阶段）。

ToolPolicy 在工具真正执行前做权限和风险判断，返回统一 PolicyResult：
- allowed：可自动执行
- denied：权限不足
- requires_confirmation：需要人工确认
- blocked：高风险，默认阻断

风险策略：
- read_only：有 schema 即可自动执行
- write：必须具备 required_scopes
- external_side_effect：必须具备 required_scopes，并且需要 confirmation
- destructive：默认阻断，不自动执行
"""
from __future__ import annotations

from .types import (
    ExecutionContext,
    PermissionDecision,
    PolicyResult,
    RiskLevel,
    ToolSpec,
)


class ToolPolicy:
    """根据 ToolSpec 风险等级与 ExecutionContext 权限做决策。"""

    def check(self, spec: ToolSpec, ctx: ExecutionContext) -> PolicyResult:
        risk = spec.risk_level

        if risk == RiskLevel.READ_ONLY:
            return PolicyResult(decision=PermissionDecision.ALLOWED, reason="read_only 自动放行")

        if risk == RiskLevel.WRITE:
            return self._check_scopes(spec, ctx)

        if risk == RiskLevel.EXTERNAL_SIDE_EFFECT:
            scope_result = self._check_scopes(spec, ctx)
            if scope_result.decision != PermissionDecision.ALLOWED:
                return scope_result
            # 权限通过，但仍需确认
            if ctx.require_confirmation:
                return PolicyResult(
                    decision=PermissionDecision.REQUIRES_CONFIRMATION,
                    reason="external_side_effect 需要人工确认",
                )
            # 上下文不允许确认流程 -> 降级为 denied
            return PolicyResult(
                decision=PermissionDecision.DENIED,
                reason="external_side_effect 需要确认但上下文不允许确认流程",
            )

        if risk == RiskLevel.DESTRUCTIVE:
            # 即使有 scope 也默认阻断
            return PolicyResult(
                decision=PermissionDecision.BLOCKED,
                reason="destructive 工具默认阻断，不自动执行",
            )

        # 未知风险等级，保守阻断
        return PolicyResult(
            decision=PermissionDecision.BLOCKED,
            reason=f"未知风险等级: {risk}",
        )

    def _check_scopes(
        self, spec: ToolSpec, ctx: ExecutionContext
    ) -> PolicyResult:
        required = spec.required_scopes
        if not required:
            return PolicyResult(decision=PermissionDecision.ALLOWED, reason="无 required_scopes")

        missing = [s for s in required if s not in ctx.scopes]
        if missing:
            return PolicyResult(
                decision=PermissionDecision.DENIED,
                reason=f"缺少权限范围: {missing}",
                missing_scopes=missing,
            )
        return PolicyResult(decision=PermissionDecision.ALLOWED, reason="权限校验通过")
