"""审计日志（第二阶段）。

每次工具调用，无论成功、失败、被拒绝、需要确认或被阻断，都写一条 audit log 到本地 JSONL。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import AuditRecord, PermissionDecision, RiskLevel, ToolStatus


class AuditLogger:
    """写入 audit.jsonl。"""

    def __init__(self, audit_file: str | Path = "audit.jsonl") -> None:
        self._path = Path(audit_file)

    @property
    def path(self) -> Path:
        return self._path

    def log(
        self,
        *,
        trace_id: str,
        user_id: str,
        tenant_id: str,
        agent_id: str,
        tool_name: str,
        tool_version: str,
        risk_level: RiskLevel,
        permission_decision: PermissionDecision,
        input_summary: str,
        output_summary: str,
        status: ToolStatus,
        error_type: Optional[str] = None,
        latency_ms: int = 0,
        timestamp: Optional[str] = None,
    ) -> AuditRecord:
        record = AuditRecord(
            trace_id=trace_id,
            user_id=user_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_version=tool_version,
            risk_level=risk_level.value,
            permission_decision=permission_decision.value,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status.value,
            error_type=error_type,
            latency_ms=latency_ms,
            timestamp=timestamp or _now_iso(),
        )
        self._write(record)
        return record

    def _write(self, record: AuditRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# 避免未使用警告（Any 在签名里被引用，保留）
_ = Any
