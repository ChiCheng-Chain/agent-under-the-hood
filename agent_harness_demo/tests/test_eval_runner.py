"""测试 EvalRunner（第二阶段）。"""
from __future__ import annotations

from agent_harness.eval.runner import EvalRunner


def test_eval_runner_runs_all_five_cases():
    """5 个 eval case 都能运行。"""
    runner = EvalRunner(tmp_dir="./eval_runs_test")
    result = runner.run_all()

    assert result.total == 5
    case_ids = {r.case_id for r in result.results}
    assert case_ids == {
        "normal_search_success",
        "model_timeout_retry_success",
        "model_rate_limit_fallback_success",
        "write_tool_permission_denied",
        "destructive_tool_blocked",
    }
    # 每个 case 都有 trace_id（即使失败也应有，除非抛异常）
    for r in result.results:
        assert r.reason  # 有原因说明


def test_destructive_tool_blocked_case_passes():
    """destructive_tool_blocked 必须 passed。"""
    runner = EvalRunner(tmp_dir="./eval_runs_test")
    result = runner.run_all()
    destructive = next(r for r in result.results if r.case_id == "destructive_tool_blocked")
    assert destructive.passed is True
    assert "blocked" in destructive.reason.lower() or "阻断" in destructive.reason


def test_all_cases_pass():
    """5 个 case 全部通过。"""
    runner = EvalRunner(tmp_dir="./eval_runs_test")
    result = runner.run_all()
    assert result.all_passed, (
        "未全部通过："
        + "; ".join(f"{r.case_id}={r.passed}({r.reason})" for r in result.results if not r.passed)
    )
    assert result.passed_count == 5
