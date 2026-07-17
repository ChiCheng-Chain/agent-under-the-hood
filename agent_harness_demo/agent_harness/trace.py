"""Trace 记录器（第二阶段增强为 trace tree）。

一次 agent 任务 = 一个 trace_id，下面挂多个 span，span 之间通过 parent_span_id
形成调用树。所有 span 追加写入本地 JSONL，每行一个 JSON 对象。

第二阶段增强：
- 支持 name / metadata / started_at / ended_at
- 支持 retry / fallback / policy / audit / eval 等 span 类型
- 提供 context manager 简化父子 span 嵌套
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .types import Span, SpanStatus, SpanType


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceRecorder:
    """写入 JSONL trace。

    使用方式：
    - 直接调用 record_model_call / record_tool_call / record_retry 等便捷方法
    - 或用 span(...) context manager 管理父子嵌套
    keep_in_memory=True 时保留内存副本供测试断言。
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
        self._pending: dict[str, dict[str, Any]] = {}

    # ----- span 生命周期 ---------------------------------------------------- #
    def start_span(
        self,
        span_type: SpanType,
        input_summary: str = "",
        *,
        name: str = "",
        parent_span_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """开始一个 span，返回 span_id。后续用 end_span 落盘。"""

        span_id = _new_id("span-")
        parent = parent_span_id or (self._parent_stack[-1] if self._parent_stack else None)
        self._parent_stack.append(span_id)
        self._pending[span_id] = {
            "span_id": span_id,
            "parent_span_id": parent,
            "span_type": span_type,
            "name": name or span_type.value,
            "input_summary": _truncate(input_summary),
            "metadata": dict(metadata) if metadata else {},
            "started_at": _now_iso(),
            "start": time.perf_counter(),
        }
        return span_id

    def end_span(
        self,
        span_id: str,
        status: SpanStatus,
        output_summary: str = "",
        error_type: Optional[str] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        ctx = self._pending.pop(span_id, None)
        if ctx is None:
            ctx = {
                "span_id": span_id,
                "parent_span_id": None,
                "span_type": SpanType.AGENT,
                "name": span_id,
                "input_summary": "",
                "metadata": {},
                "started_at": _now_iso(),
                "start": time.perf_counter(),
            }
        latency_ms = int((time.perf_counter() - ctx["start"]) * 1000)
        if metadata:
            ctx["metadata"].update(metadata)

        span = Span(
            trace_id=self.trace_id,
            span_id=ctx["span_id"],
            parent_span_id=ctx["parent_span_id"],
            span_type=ctx["span_type"],
            name=ctx["name"],
            status=status,
            latency_ms=latency_ms,
            input_summary=ctx["input_summary"],
            output_summary=_truncate(output_summary),
            error_type=error_type,
            metadata=ctx["metadata"],
            started_at=ctx["started_at"],
            ended_at=_now_iso(),
        )
        self._write(span)
        if self._keep:
            self._spans.append(span)
        # 弹栈
        while self._parent_stack and self._parent_stack[-1] == span_id:
            self._parent_stack.pop()
        if span_id in self._parent_stack:
            idx = len(self._parent_stack) - 1 - self._parent_stack[::-1].index(span_id)
            self._parent_stack = self._parent_stack[:idx]
        return span

    @contextmanager
    def span(
        self,
        span_type: SpanType,
        input_summary: str = "",
        *,
        name: str = "",
        parent_span_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[str]:
        """context manager：自动 start/end。失败时 status=error。"""
        sid = self.start_span(
            span_type, input_summary, name=name, parent_span_id=parent_span_id, metadata=metadata
        )
        try:
            yield sid
        except Exception as exc:
            self.end_span(sid, SpanStatus.ERROR, str(exc), type(exc).__name__)
            raise
        else:
            self.end_span(sid, SpanStatus.OK, "")

    # ----- 便捷封装 --------------------------------------------------------- #
    def record_model_call(
        self,
        input_summary: str,
        output_summary: str,
        status: SpanStatus = SpanStatus.OK,
        error_type: Optional[str] = None,
        *,
        name: str = "model_call",
        parent_span_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.MODEL, input_summary, name=name, parent_span_id=parent_span_id, metadata=metadata
        )
        return self.end_span(sid, status, output_summary, error_type)

    def record_tool_call(
        self,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        status: SpanStatus = SpanStatus.OK,
        error_type: Optional[str] = None,
        *,
        parent_span_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.TOOL,
            f"{tool_name} | {input_summary}",
            name=f"tool:{tool_name}",
            parent_span_id=parent_span_id,
            metadata=metadata,
        )
        return self.end_span(sid, status, f"{tool_name} | {output_summary}", error_type)

    def record_retry(
        self,
        attempt: int,
        error_type: str,
        output_summary: str = "",
        *,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.RETRY,
            f"attempt={attempt}",
            name=f"retry#{attempt}",
            parent_span_id=parent_span_id,
            metadata={"attempt": attempt, "error_type": error_type},
        )
        return self.end_span(sid, SpanStatus.INFO, output_summary, error_type)

    def record_fallback(
        self,
        reason: str,
        output_summary: str,
        status: SpanStatus = SpanStatus.OK,
        *,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.FALLBACK,
            reason,
            name="fallback",
            parent_span_id=parent_span_id,
            metadata={"reason": reason},
        )
        return self.end_span(sid, status, output_summary)

    def record_policy(
        self,
        tool_name: str,
        decision: str,
        reason: str,
        *,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.POLICY,
            f"{tool_name} | {decision}",
            name=f"policy:{tool_name}",
            parent_span_id=parent_span_id,
            metadata={"decision": decision, "reason": reason},
        )
        status = SpanStatus.OK if decision == "allowed" else SpanStatus.INFO
        return self.end_span(sid, status, reason)

    def record_audit(
        self,
        tool_name: str,
        status: str,
        *,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        sid = self.start_span(
            SpanType.AUDIT,
            tool_name,
            name=f"audit:{tool_name}",
            parent_span_id=parent_span_id,
        )
        return self.end_span(sid, SpanStatus.OK, status)

    def record_final(self, output_summary: str, status: SpanStatus = SpanStatus.OK) -> Span:
        sid = self.start_span(SpanType.FINAL, output_summary, name="final")
        return self.end_span(sid, status, output_summary)

    # ----- 查询 ------------------------------------------------------------- #
    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def spans_of_type(self, span_type: SpanType) -> list[Span]:
        # SpanType.RUN 是 AGENT 的别名值，统一按 value 比较
        target = span_type.value
        return [s for s in self._spans if s.span_type.value == target]

    def root_span(self) -> Optional[Span]:
        for s in self._spans:
            if s.parent_span_id is None and s.span_type in (SpanType.AGENT, SpanType.RUN):
                return s
        return None

    # ----- 落盘 ------------------------------------------------------------- #
    def _write(self, span: Span) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(span.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _truncate(s: str, limit: int = 500) -> str:
    s = str(s)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"
