"""测试工具运行时。"""
from __future__ import annotations

from agent_harness.tools import ToolRuntime, build_default_registry
from agent_harness.types import ToolErrorType


def test_echo_returns_input(tool_runtime):
    """echo 工具正常返回输入文本。"""
    result = tool_runtime.execute("echo", {"text": "hello"})
    assert result.ok is True
    assert result.output == "hello"
    assert result.error_type is None


def test_unknown_tool_returns_structured_error(tool_runtime):
    """工具不存在时返回结构化错误而非抛异常。"""
    result = tool_runtime.execute("not_a_tool", {"x": 1})
    assert result.ok is False
    assert result.error_type == ToolErrorType.NOT_FOUND
    assert "not_a_tool" in (result.error_message or "")
    # 结构化错误必须带 latency
    assert result.latency_ms >= 0


def test_search_docs_returns_matches(tool_runtime):
    """search_docs 能从 demo 文档库匹配到内容。"""
    result = tool_runtime.execute("search_docs", {"query": "refund"})
    assert result.ok is True
    assert isinstance(result.output, list)
    assert len(result.output) > 0
    assert any("refund" in doc.lower() or "退款" in doc for doc in result.output)


def test_search_docs_empty_query(tool_runtime):
    """空查询返回空列表，不报错。"""
    result = tool_runtime.execute("search_docs", {"query": ""})
    assert result.ok is True
    assert result.output == []


def test_validation_error_missing_required(tool_runtime):
    """search_docs 缺少必填 query 时返回校验错误。"""
    result = tool_runtime.execute("search_docs", {})
    assert result.ok is False
    assert result.error_type == ToolErrorType.VALIDATION_ERROR


def test_validation_error_wrong_type(tool_runtime):
    """参数类型错误时返回校验错误。"""
    result = tool_runtime.execute("echo", {"text": 123})
    # text 期望 string，传 int 应被校验拦截
    assert result.ok is False
    assert result.error_type == ToolErrorType.VALIDATION_ERROR


def test_get_time_returns_iso(tool_runtime):
    """get_time 返回 ISO 时间字符串。"""
    result = tool_runtime.execute("get_time", {})
    assert result.ok is True
    assert isinstance(result.output, str)
    assert "T" in result.output  # ISO 8601


def test_execute_no_arguments(tool_runtime):
    """不传 arguments 等价于空字典。"""
    result = tool_runtime.execute("get_time")
    assert result.ok is True
