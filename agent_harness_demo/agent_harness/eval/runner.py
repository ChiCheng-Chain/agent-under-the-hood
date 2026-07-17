"""EvalRunner：执行一批 eval case 并汇总结果。"""
from __future__ import annotations

from pathlib import Path

from ..tools import _reset_executed_handlers
from .cases import build_cases
from .types import EvalCase, EvalCaseResult, EvalResult


class EvalRunner:
    """运行所有 eval case。"""

    def __init__(self, tmp_dir: str | Path | None = None) -> None:
        self._tmp_dir = Path(tmp_dir) if tmp_dir else Path("./eval_runs")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> EvalResult:
        cases = build_cases(self._tmp_dir)
        results: list[EvalCaseResult] = []

        for case in cases:
            results.append(self._run_one(case))

        passed = sum(1 for r in results if r.passed)
        return EvalResult(results=results, total=len(results), passed_count=passed)

    def _run_one(self, case: EvalCase) -> EvalCaseResult:
        # 每个 case 前清空全局 handler 执行记录，避免跨 case 污染
        _reset_executed_handlers()

        try:
            runner = case.setup()
            result = runner.run(case.user_input)
            passed, reason = case.assert_fn(result, runner)
            return EvalCaseResult(
                case_id=case.case_id,
                passed=passed,
                reason=reason,
                trace_id=runner.trace.trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            return EvalCaseResult(
                case_id=case.case_id,
                passed=False,
                reason=f"case 执行抛异常: {type(exc).__name__}: {exc}",
                trace_id="",
            )
