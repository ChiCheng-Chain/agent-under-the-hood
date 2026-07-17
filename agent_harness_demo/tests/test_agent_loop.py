"""测试 Agent 主循环。"""
from __future__ import annotations

from agent_harness.gateway import MockModelClient
from agent_harness.runner import AgentRunner
from agent_harness.trace import TraceRecorder
from agent_harness.tools import ToolRuntime, build_default_registry
from agent_harness.types import FinishReason


def test_search_input_calls_search_docs_then_final_answer(runner):
    """输入含“搜索”时应调用 search_docs 并最终返回 final_answer。"""
    result = runner.run("帮我搜索 refund policy")

    assert result.error is None
    assert result.final_answer  # 非空
    # 至少调用过一次 search_docs 工具
    tool_names = [tc.name for tc in result.tool_calls]
    assert "search_docs" in tool_names
    # 最终答案里应包含退款政策相关内容
    assert "refund" in result.final_answer.lower() or "退款" in result.final_answer


def test_time_input_calls_get_time(runner):
    """输入含“时间”时应调用 get_time。"""
    result = runner.run("现在时间")
    assert result.error is None
    tool_names = [tc.name for tc in result.tool_calls]
    assert "get_time" in tool_names


def test_non_tool_input_returns_direct_final_answer(runner):
    """不含触发词的输入应直接返回 final_answer，不调用任何工具。"""
    result = runner.run("你好呀")
    assert result.error is None
    assert result.tool_calls == []
    assert result.final_answer


def test_loop_terminates_within_max_steps():
    """Mock 客户端保证闭环在 max_steps 内结束。"""
    trace = TraceRecorder(trace_file="traces.jsonl")
    runner = AgentRunner(
        MockModelClient(),
        tool_runtime=ToolRuntime(build_default_registry()),
        trace=trace,
        max_steps=5,
    )
    result = runner.run("搜索 return process")
    assert result.steps <= 5
    assert result.error is None


def test_max_steps_enforced():
    """构造一个永不返回 final_answer 的假客户端，验证 max_steps 兜底。"""
    import copy
    from agent_harness.types import ModelResponse, ToolCall

    class NeverFinalClient:
        def invoke(self, request):
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALL,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "x"})],
            )

    trace = TraceRecorder(trace_file="traces.jsonl")
    runner = AgentRunner(
        NeverFinalClient(),
        tool_runtime=ToolRuntime(build_default_registry()),
        trace=trace,
        max_steps=3,
    )
    result = runner.run("无限循环")
    assert result.error is not None
    assert "最大循环次数" in result.error
    assert result.steps == 3

    # 避免未使用
    _ = copy
