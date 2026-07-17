"""测试 TraceRecorder。"""
from __future__ import annotations

import json
from pathlib import Path

from agent_harness.gateway import MockModelClient
from agent_harness.runner import AgentRunner
from agent_harness.tools import ToolRuntime, build_default_registry
from agent_harness.trace import TraceRecorder
from agent_harness.types import SpanStatus, SpanType


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_trace_id_generated():
    """每次任务生成唯一 trace_id。"""
    t1 = TraceRecorder(trace_file="traces.jsonl")
    t2 = TraceRecorder(trace_file="traces.jsonl")
    assert t1.trace_id != t2.trace_id
    assert t1.trace_id.startswith("trace-")


def test_task_writes_jsonl_with_model_and_tool_spans(tmp_path: Path):
    """执行一次搜索任务后，JSONL 包含 model span 和 tool span。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runner = AgentRunner(
        MockModelClient(),
        tool_runtime=ToolRuntime(build_default_registry()),
        trace=trace,
        max_steps=5,
    )
    runner.run("搜索 refund policy")

    assert trace_file.exists()
    records = _read_jsonl(trace_file)
    assert len(records) >= 3  # 至少：run + model + tool

    span_types = [r["span_type"] for r in records]
    assert SpanType.RUN.value in span_types
    assert SpanType.MODEL_CALL.value in span_types
    assert SpanType.TOOL_CALL.value in span_types

    # 每个 span 必须包含规定字段
    required = {
        "trace_id", "span_id", "span_type", "status", "latency_ms",
        "input_summary", "output_summary", "error_type",
    }
    for rec in records:
        assert required.issubset(rec.keys()), f"缺失字段: {required - set(rec.keys())}"
        assert rec["trace_id"] == trace.trace_id
        assert rec["span_id"]
        assert rec["status"] in (SpanStatus.OK.value, SpanStatus.ERROR.value)


def test_span_parent_child_relationship(tmp_path: Path):
    """run span 是根，model/tool span 的 parent 指向 run。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runner = AgentRunner(
        MockModelClient(),
        tool_runtime=ToolRuntime(build_default_registry()),
        trace=trace,
        max_steps=5,
    )
    runner.run("搜索 refund policy")

    records = _read_jsonl(trace_file)
    run_spans = [r for r in records if r["span_type"] == SpanType.RUN.value]
    assert len(run_spans) == 1
    run_id = run_spans[0]["span_id"]

    # run span 自己没有 parent
    assert run_spans[0]["parent_span_id"] is None

    # model / tool span 的 parent 应是 run_id
    children = [
        r for r in records
        if r["span_type"] in (SpanType.MODEL_CALL.value, SpanType.TOOL_CALL.value, SpanType.FINAL.value)
    ]
    assert len(children) >= 2
    for c in children:
        assert c["parent_span_id"] == run_id


def test_tool_span_records_error_status(tmp_path: Path):
    """工具调用失败时 span 状态为 error，并记录 error_type。"""

    class BadToolClient:
        """总是调用不存在的工具。"""
        def __init__(self):
            self._first = True

        def invoke(self, request):
            from agent_harness.types import ModelResponse, ToolCall, FinishReason
            if self._first:
                self._first = False
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALL,
                    tool_calls=[ToolCall(id="c1", name="no_such_tool", arguments={})],
                )
            from agent_harness.types import Role
            # 收到 observation 后返回 final
            return ModelResponse(content="完成", finish_reason=FinishReason.FINAL_ANSWER)

    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runner = AgentRunner(
        BadToolClient(),
        tool_runtime=ToolRuntime(build_default_registry()),
        trace=trace,
        max_steps=5,
    )
    result = runner.run("触发错误工具")
    assert result.final_answer == "完成"

    records = _read_jsonl(trace_file)
    tool_spans = [r for r in records if r["span_type"] == SpanType.TOOL_CALL.value]
    assert len(tool_spans) == 1
    assert tool_spans[0]["status"] == SpanStatus.ERROR.value
    assert tool_spans[0]["error_type"] is not None
