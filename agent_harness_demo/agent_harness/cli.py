"""命令行入口。

用法：
    python -m agent_harness.cli "帮我搜索 refund policy"
    python -m agent_harness.cli eval
    USE_REAL_MODEL=true python -m agent_harness.cli "现在几点"

默认使用 MockModelClient；USE_REAL_MODEL=true 时使用 OpenAICompatibleClient。
"""
from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .gateway import MockModelClient, OpenAICompatibleClient
from .runner import AgentRunner
from .trace import TraceRecorder


def build_model(settings):
    """根据配置选择模型客户端。"""
    if settings.use_real_model:
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    return MockModelClient()


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else list(sys.argv[1:])

    # 子命令分发：eval / run，其余按 run 处理（兼容旧用法）
    if raw and raw[0] == "eval":
        return _run_eval()

    if raw and raw[0] == "run":
        rest = raw[1:]
    else:
        rest = raw

    # 解析 run 的参数
    parser = argparse.ArgumentParser(prog="agent-harness run", add_help=True)
    parser.add_argument("input", nargs="?", help="用户输入")
    parser.add_argument("--trace-file", default=None, help="trace 输出路径")
    parser.add_argument("--max-steps", type=int, default=None, help="最大循环次数")
    args = parser.parse_args(rest)

    user_input = args.input
    if not user_input:
        text = _read_input()
        if not text:
            print("输入为空，退出。")
            return 1
        user_input = text

    return _run_one(user_input, args.trace_file, args.max_steps)


def _read_input() -> str | None:
    try:
        text = input("请输入: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return None
    return text or None


def _run_one(user_input: str, trace_file_arg, max_steps_arg) -> int:
    settings = load_settings()
    trace_file = trace_file_arg or str(settings.trace_path)
    max_steps = max_steps_arg or settings.max_steps

    model = build_model(settings)
    trace = TraceRecorder(trace_file=trace_file)
    runner = AgentRunner(model, trace=trace, max_steps=max_steps)

    print(f"[trace_id={trace.trace_id}] 开始执行...")
    result = runner.run(user_input)

    print("\n--- 最终输出 ---")
    print(result.final_answer or "(无输出)")
    if result.error:
        print(f"[error] {result.error}")
    print(f"\n[steps={result.steps}] [tool_calls={len(result.tool_calls)}] [trace_file={trace_file}]")
    return 0 if not result.error else 1


def _run_eval() -> int:
    from .eval.runner import EvalRunner

    runner = EvalRunner()
    result = runner.run_all()

    print("=" * 60)
    print("Eval 结果")
    print("=" * 60)
    for r in result.results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"[{flag}] {r.case_id}")
        print(f"       reason: {r.reason}")
        print(f"       trace_id: {r.trace_id}")
    print("-" * 60)
    print(f"通过 {result.passed_count}/{result.total}")
    print("=" * 60)
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
