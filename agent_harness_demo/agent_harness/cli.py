"""命令行入口。

用法：
    python -m agent_harness.cli "帮我搜索 refund policy"
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
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="最小 Agent Harness 闭环 demo",
    )
    parser.add_argument("input", nargs="?", help="用户输入")
    parser.add_argument(
        "--trace-file",
        default=None,
        help="trace 输出路径，默认读 TRACE_FILE 环境变量",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="最大循环次数，默认读 AGENT_MAX_STEPS 环境变量",
    )
    args = parser.parse_args(argv)

    settings = load_settings()

    user_input = args.input
    if not user_input:
        # 没给参数时用交互式读取一行
        try:
            user_input = input("请输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            return 130
        if not user_input:
            print("输入为空，退出。")
            return 1

    trace_file = args.trace_file or str(settings.trace_path)
    max_steps = args.max_steps or settings.max_steps

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


if __name__ == "__main__":
    sys.exit(main())
