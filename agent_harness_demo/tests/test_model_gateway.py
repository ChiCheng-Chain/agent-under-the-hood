"""测试 ModelGateway 错误治理（第二阶段）。"""
from __future__ import annotations

import pytest

from agent_harness.gateway import (
    MockModelClient,
    ResilientModelGateway,
)
from agent_harness.trace import TraceRecorder
from agent_harness.types import (
    ModelError,
    ModelErrorType,
    ModelRequest,
    Message,
    Role,
)


def _request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content="搜索 refund policy")])


def test_timeout_is_retried():
    """timeout_once_then_success：第一次 TIMEOUT，重试后成功。"""
    primary = MockModelClient(behavior="timeout_once_then_success")
    trace = TraceRecorder(trace_file="traces.jsonl")
    gw = ResilientModelGateway(primary, max_retries=2, sleep=False, trace=trace)

    resp = gw.invoke(_request())
    # 重试后拿到正常响应（可能是 tool_call 或 final_answer，取决于输入）
    assert resp.finish_reason.value in ("final_answer", "tool_call")
    # 主 client 被调用了 2 次（1 次失败 + 1 次成功）
    assert primary.invoke_count == 2

    # trace 里有 retry span
    retry_spans = [s for s in trace.spans if s.span_type.value == "retry"]
    assert len(retry_spans) == 1
    assert retry_spans[0].metadata["error_type"] == ModelErrorType.TIMEOUT.value


def test_auth_error_not_retried():
    """auth_error 不重试，立即抛出。"""
    primary = MockModelClient(behavior="auth_error")
    trace = TraceRecorder(trace_file="traces.jsonl")
    gw = ResilientModelGateway(primary, max_retries=3, sleep=False, trace=trace)

    with pytest.raises(ModelError) as exc_info:
        gw.invoke(_request())
    assert exc_info.value.error_type == ModelErrorType.AUTH_ERROR
    # 只调用 1 次，没有重试
    assert primary.invoke_count == 1
    # 没有 retry span
    retry_spans = [s for s in trace.spans if s.span_type.value == "retry"]
    assert len(retry_spans) == 0


def test_bad_request_not_retried():
    """bad_request 不重试。"""
    primary = MockModelClient(behavior="bad_request")
    gw = ResilientModelGateway(primary, max_retries=3, sleep=False)

    with pytest.raises(ModelError) as exc_info:
        gw.invoke(_request())
    assert exc_info.value.error_type == ModelErrorType.BAD_REQUEST
    assert primary.invoke_count == 1


def test_rate_limit_triggers_fallback():
    """主模型 RATE_LIMIT，重试耗尽后 fallback 到备用模型成功。"""
    primary = MockModelClient(behavior="rate_limit_then_fallback", name="primary")
    fallback = MockModelClient(behavior="always_success", name="fallback")
    trace = TraceRecorder(trace_file="traces.jsonl")
    gw = ResilientModelGateway(
        primary, fallback=fallback, max_retries=1, sleep=False, trace=trace
    )

    resp = gw.invoke(_request())
    # fallback 成功返回（tool_call 或 final_answer）
    assert resp.finish_reason.value in ("final_answer", "tool_call")
    # 主 client 被调用（1 次正常 + 1 次重试 = 2）
    assert primary.invoke_count == 2
    # fallback 被调用
    assert fallback.invoke_count == 1

    # trace 里有 fallback span 和 retry span
    fallback_spans = [s for s in trace.spans if s.span_type.value == "fallback"]
    assert len(fallback_spans) == 1
    retry_spans = [s for s in trace.spans if s.span_type.value == "retry"]
    assert len(retry_spans) >= 1


def test_rate_limit_no_fallback_raises():
    """主模型限流且无 fallback 时，重试耗尽后抛 ModelError。"""
    primary = MockModelClient(behavior="rate_limit_then_fallback")
    gw = ResilientModelGateway(primary, max_retries=1, sleep=False)

    with pytest.raises(ModelError) as exc_info:
        gw.invoke(_request())
    assert exc_info.value.error_type == ModelErrorType.RATE_LIMIT
    assert primary.invoke_count == 2  # 1 + 1 重试


def test_always_success_no_retry():
    """always_success 第一次就成功，不重试。"""
    primary = MockModelClient(behavior="always_success")
    gw = ResilientModelGateway(primary, max_retries=3, sleep=False)

    resp = gw.invoke(_request())
    assert primary.invoke_count == 1
    assert resp.finish_reason.value in ("final_answer", "tool_call")
