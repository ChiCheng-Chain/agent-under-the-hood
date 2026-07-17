"""Trace 记录器。

一次 agent 任务 = 一个 trace_id，下面挂多个 span。
所有 span 追加写入本地 JSONL 文件，每行一个 JSON 对象。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .types import Span, SpanStatus, SpanType


def _new_id(prefix: str = "") -> str:
    # uuid4 的前 8 位足够 demo 使用，加前缀便于人眼区分
    return f"{prefix}{uuid.uuid4().hex[:8]}"


class TraceRecorder:
    """写入 JSONL trace。

    使用方式：record_span(...) 直接落盘。keep_in_memory 用于测试断言。
    """

    def __init__(
        self,
        trace_file: str | Path = "traces.jsonl",
        *,
        trace_id: Optional[str] = None,
        keep_in_memory: bool = True,
    ) -> None:
        self._path = Path(trace_file)
        self.trace_id = trace_id or _new_id("trace-")
        self._parent_stack: list[str] = []
        self._spans: list[Span] = []
        self._keep = keep_in_memory

    # ----- span 生命周期 ---------------------------------------------------- #
    def start_span(
        self,
        span_type: SpanType,
        input_summary: str,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """开始一个 span，返回 span_id。后续用 end_span 落盘。"""

        span_id = _new_id("span-")
        parent = parent_span_id or (self._parent_stack[-1] if self._parent_stack else None)
        # 把当前 span 压栈，后续子 span 默认挂到它下面
        self._parent_stack.append(span_id)
        # 暂存上下文，end_span 时取出来
        self._pending[span_id] = {
            "span_id": span_id,
            "parent_span_id": parent,
            "span_type": span_type,
            "input_summary": _truncate(input_summary),
            "start": time.perf_counter(),
        }
        return span_id

    def end_span(
        self,
        span_id: str,
        status: SpanStatus,
        output_summary: str = "",
        error_type: Optional[str] = None,
    ) -> Span:
        ctx = self._pending.pop(span_id, None)
        if ctx is None:
            # 容错：span 没有对应 start，直接构造一个
            ctx = {
                "span_id": span_id,
                "parent_span_id": None,
                "span_type": SpanType.RUN,
                "input_summary": "",
                "start": time.perf_counter(),
            }
        latency_ms = int((time.perf_counter() - ctx["start"]) * 1000)

        span = Span(
            trace_id=self.trace_id,
            span_id=ctx["span_id"],
            parent_span_id=ctx["parent_span_id"],
            span_type=ctx["span_type"],
            status=status,
            latency_ms=latency_ms,
            input_summary=ctx["input_summary"],
            output_summary=_truncate(output_summary),
            error_type=error_type,
        )
        self._write(span)
        if self._keep:
            self._spans.append(span)
        # 弹栈：只移除栈顶等于该 id 的位置
        while self._parent_stack and self._parent_stack[-1] == span_id:
            self._parent_stack.pop()
        # 如果栈顶不是该 id（理论上不该发生），清理到最后一次出现
        if span_id in self._parent_stack:
            idx = len(self._parent_stack) - 1 - self._parent_stack[::-1].index(span_id)
            self._parent_stack = self._parent_stack[:idx]
        return span

    # ----- 便捷封装 --------------------------------------------------------- #
    def record_model_call(
        self,
        input_summary: str,
        output_summary: str,
        status: SpanStatus = SpanStatus.OK,
        error_type: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(SpanType.MODEL_CALL, input_summary)
        return self.end_span(sid, status, output_summary, error_type)

    def record_tool_call(
        self,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        status: SpanStatus = SpanStatus.OK,
        error_type: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(SpanType.TOOL_CALL, f"{tool_name} | {input_summary}")
        return self.end_span(sid, status, f"{tool_name} | {output_summary}", error_type)

    def record_final(self, output_summary: str, status: SpanStatus = SpanStatus.OK) -> Span:
        """记录最终输出 span。"""
        sid = self.start_span(SpanType.FINAL, output_summary)
        return self.end_span(sid, status, output_summary)

    # ----- 查询 ------------------------------------------------------------- #
    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def spans_of_type(self, span_type: SpanType) -> list[Span]:
        return [s for s in self._spans if s.span_type == span_type]

    # ----- 落盘 ------------------------------------------------------------- #
    def _write(self, span: Span) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(span.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # start_span 用到的暂存表
    @property
    def _pending(self) -> dict[str, dict[str, Any]]:
        # 延迟初始化，避免 __init__ 之外构造
        if not hasattr(self, "_pending_map"):
            self._pending_map: dict[str, dict[str, Any]] = {}
        return self._pending_map


def _truncate(s: str, limit: int = 500) -> str:
    s = str(s)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"
