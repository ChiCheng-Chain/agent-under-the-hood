"""测试 trace tree 结构（第二阶段）。"""
from __future__ import annotations

from pathlib import Path

from agent_harness.gateway import MockModelClient
from agent_harness.runner import AgentRunner
from agent_harness.tools import ToolRuntime, build_default_registry
from agent_harness.trace import TraceRecorder
from agent_harness.types import SpanType


def test_trace_has_agent_model_tool_policy_spans(tmp_path: Path):
    """执行一次工具调用后，trace 里有 agent / model / tool / policy span。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), trace=trace)
    runner = AgentRunner(MockModelClient(), tool_runtime=runtime, trace=trace)
    runner.run("搜索 refund policy")

    spans = trace.spans
    span_types = {s.span_type.value for s in spans}

    # 必须含 agent / model / tool / policy
    assert SpanType.AGENT.value in span_types
    assert SpanType.MODEL.value in span_types
    assert SpanType.TOOL.value in span_types
    assert SpanType.POLICY.value in span_types


def test_policy_span_is_child_of_tool_span(tmp_path: Path):
    """policy span 是 tool span 的子 span。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), trace=trace)
    runner = AgentRunner(MockModelClient(), tool_runtime=runtime, trace=trace)
    runner.run("搜索 refund policy")

    spans = trace.spans
    tool_spans = [s for s in spans if s.span_type.value == SpanType.TOOL.value]
    policy_spans = [s for s in spans if s.span_type.value == SpanType.POLICY.value]

    assert len(tool_spans) >= 1
    assert len(policy_spans) >= 1
    tool_ids = {s.span_id for s in tool_spans}
    # 至少有一个 policy span 的 parent 是某个 tool span
    assert any(p.parent_span_id in tool_ids for p in policy_spans)


def test_model_and_tool_are_children_of_agent(tmp_path: Path):
    """model 和 tool span 的 parent 指向 agent root。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), trace=trace)
    runner = AgentRunner(MockModelClient(), tool_runtime=runtime, trace=trace)
    runner.run("搜索 refund policy")

    spans = trace.spans
    agent_spans = [s for s in spans if s.span_type.value == SpanType.AGENT.value]
    assert len(agent_spans) == 1
    agent_id = agent_spans[0].span_id

    model_spans = [s for s in spans if s.span_type.value == SpanType.MODEL.value]
    tool_spans = [s for s in spans if s.span_type.value == SpanType.TOOL.value]

    for m in model_spans:
        assert m.parent_span_id == agent_id
    for t in tool_spans:
        assert t.parent_span_id == agent_id


def test_span_has_name_and_metadata_and_timestamps(tmp_path: Path):
    """第二阶段 span 含 name / metadata / started_at / ended_at。"""
    trace_file = tmp_path / "traces.jsonl"
    trace = TraceRecorder(trace_file=str(trace_file))
    runtime = ToolRuntime(build_default_registry(), trace=trace)
    runner = AgentRunner(MockModelClient(), tool_runtime=runtime, trace=trace)
    runner.run("搜索 refund policy")

    for s in trace.spans:
        assert s.name  # 非空
        assert isinstance(s.metadata, dict)
        assert s.started_at  # 非空
        assert s.ended_at  # 非空
