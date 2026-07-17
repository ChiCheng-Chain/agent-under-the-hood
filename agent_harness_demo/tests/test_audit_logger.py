"""测试 AuditLogger（第二阶段）。"""
from __future__ import annotations

import json
from pathlib import Path

from agent_harness.audit import AuditLogger
from agent_harness.gateway import MockModelClient
from agent_harness.runner import AgentRunner
from agent_harness.tools import ToolRuntime, build_default_registry
from agent_harness.trace import TraceRecorder
from agent_harness.types import (
    ExecutionContext,
    PermissionDecision,
    RiskLevel,
    ToolStatus,
)


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_audit_log_written_on_success(tmp_path: Path):
    """工具成功执行时写 audit log。"""
    audit_file = tmp_path / "audit.jsonl"
    trace_file = tmp_path / "traces.jsonl"
    audit = AuditLogger(audit_file=str(audit_file))
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), audit=audit, trace=trace)
    runner = AgentRunner(MockModelClient(), tool_runtime=runtime, trace=trace)
    runner.run("搜索 refund policy")

    records = _read_jsonl(audit_file)
    assert len(records) >= 1
    rec = records[0]
    for field in [
        "trace_id", "user_id", "tenant_id", "agent_id", "tool_name", "tool_version",
        "risk_level", "permission_decision", "input_summary", "output_summary",
        "status", "error_type", "latency_ms", "timestamp",
    ]:
        assert field in rec, f"缺失字段: {field}"
    assert rec["tool_name"] == "search_docs"
    assert rec["status"] == ToolStatus.SUCCESS.value
    assert rec["permission_decision"] == PermissionDecision.ALLOWED.value


def test_audit_log_written_on_denied(tmp_path: Path):
    """工具被拒绝也会写 audit log。"""
    audit_file = tmp_path / "audit.jsonl"
    trace_file = tmp_path / "traces.jsonl"
    audit = AuditLogger(audit_file=str(audit_file))
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), audit=audit, trace=trace)

    # 无 ticket:write scope 调 create_ticket -> denied
    ctx = ExecutionContext(
        user_id="u-no-write",
        tenant_id="t1",
        scopes=[],
        trace_id=trace.trace_id,
    )
    result = runtime.execute("create_ticket", {"title": "x"}, ctx=ctx)
    assert result.status == ToolStatus.DENIED

    records = _read_jsonl(audit_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["tool_name"] == "create_ticket"
    assert rec["status"] == ToolStatus.DENIED.value
    assert rec["permission_decision"] == PermissionDecision.DENIED.value
    assert rec["user_id"] == "u-no-write"
    assert rec["timestamp"]  # 非空


def test_audit_log_written_on_blocked(tmp_path: Path):
    """destructive 工具被阻断也写 audit log。"""
    audit_file = tmp_path / "audit.jsonl"
    audit = AuditLogger(audit_file=str(audit_file))
    runtime = ToolRuntime(build_default_registry(), audit=audit)

    ctx = ExecutionContext(
        user_id="u-admin",
        tenant_id="t1",
        scopes=["record:delete"],
    )
    result = runtime.execute("delete_record", {"record_id": "r1"}, ctx=ctx)
    assert result.status == ToolStatus.BLOCKED

    records = _read_jsonl(audit_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["tool_name"] == "delete_record"
    assert rec["risk_level"] == RiskLevel.DESTRUCTIVE.value
    assert rec["permission_decision"] == PermissionDecision.BLOCKED.value
    assert rec["status"] == ToolStatus.BLOCKED.value
