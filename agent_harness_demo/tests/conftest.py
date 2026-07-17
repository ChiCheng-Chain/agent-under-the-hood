"""共享 fixture。"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.gateway import MockModelClient
from agent_harness.runner import AgentRunner
from agent_harness.tools import build_default_registry
from agent_harness.tools import ToolRuntime
from agent_harness.trace import TraceRecorder


@pytest.fixture
def tmp_trace_file(tmp_path: Path) -> Path:
    return tmp_path / "traces.jsonl"


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def tool_runtime(registry):
    return ToolRuntime(registry)


@pytest.fixture
def trace(tmp_trace_file):
    return TraceRecorder(trace_file=str(tmp_trace_file))


@pytest.fixture
def runner(trace, tool_runtime):
    return AgentRunner(
        MockModelClient(),
        tool_runtime=tool_runtime,
        trace=trace,
        max_steps=5,
    )
