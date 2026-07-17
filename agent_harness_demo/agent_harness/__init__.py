"""agent_harness 包入口。"""

from .config import Settings, load_settings
from .gateway import (
    MockModelClient,
    ModelAPIError,
    ModelClient,
    ModelConnectionError,
    ModelError,
    OpenAICompatibleClient,
)
from .runner import AgentResult, AgentRunner
from .tools import ToolRegistry, ToolRuntime, build_default_registry
from .trace import TraceRecorder
from .types import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    Span,
    SpanStatus,
    SpanType,
    ToolCall,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "Settings",
    "load_settings",
    "MockModelClient",
    "OpenAICompatibleClient",
    "ModelClient",
    "ModelError",
    "ModelConnectionError",
    "ModelAPIError",
    "AgentRunner",
    "AgentResult",
    "ToolRegistry",
    "ToolRuntime",
    "build_default_registry",
    "TraceRecorder",
    "Message",
    "Role",
    "ToolCall",
    "ToolSpec",
    "ToolResult",
    "ModelRequest",
    "ModelResponse",
    "FinishReason",
    "Span",
    "SpanType",
    "SpanStatus",
]

__version__ = "0.1.0"
