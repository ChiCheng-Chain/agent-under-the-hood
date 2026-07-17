"""eval 类型定义。"""
from __future__ import annotations

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """一个评测用例。"""

    case_id: str
    description: str
    # 构造 AgentRunner 的工厂：返回 (runner, 期望校验用的额外信息)
    setup: Callable[[], Any]
    # 用户输入
    user_input: str
    # 断言函数：(runner 执行后的结果, runner) -> (passed, reason)
    assert_fn: Callable[[Any, Any], tuple[bool, str]]

    model_config = {"arbitrary_types_allowed": True}


class EvalCaseResult(BaseModel):
    """单个 case 的运行结果。"""

    case_id: str
    passed: bool
    reason: str
    trace_id: str = ""


class EvalResult(BaseModel):
    """整批 eval 的汇总结果。"""

    results: list[EvalCaseResult] = Field(default_factory=list)
    total: int = 0
    passed_count: int = 0

    @property
    def all_passed(self) -> bool:
        return self.passed_count == self.total and self.total > 0
