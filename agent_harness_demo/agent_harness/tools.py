"""工具注册表、运行时与内置工具。

- ToolRegistry：注册 / 查找工具，产出 ToolSpec 给模型
- ToolRuntime：执行工具调用，做参数校验，返回统一 ToolResult
- 内置工具：echo / get_time / search_docs
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, ValidationError, create_model

from .types import (
    RiskLevel,
    ToolErrorType,
    ToolResult,
    ToolSpec,
)

# 工具 handler 的统一签名：接收关键字参数，返回任意可 JSON 序列化结果
ToolHandler = Callable[..., Any]


class RegisteredTool(BaseModel):
    """一个已注册工具的完整定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel
    handler: ToolHandler

    # pydantic v2 默认不接收任意 callable，需要放开
    model_config = {"arbitrary_types_allowed": True}

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            risk_level=self.risk_level,
        )


# --------------------------------------------------------------------------- #
# 内置 demo 文档库
# --------------------------------------------------------------------------- #
DEMO_DOCS: list[str] = [
    "退款政策（refund policy）：购买后 7 天内可全额退款，超过 7 天按 50% 退款。",
    "退货流程（return process）：请联系客服提交退货申请，并在 14 天内寄回商品。",
    "隐私政策（privacy policy）：我们不会向第三方出售您的个人数据。",
    "发货时间（shipping time）：现货商品 24 小时内发货，预售商品按页面标注时间。",
]


# --------------------------------------------------------------------------- #
# 内置工具 handler
# --------------------------------------------------------------------------- #
def _echo(text: str = "") -> str:
    return text


def _get_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _search_docs(query: str = "") -> list[str]:
    """对内存里的 DEMO_DOCS 做简单的关键词包含匹配。"""

    if not query:
        return []
    q = query.lower()
    return [doc for doc in DEMO_DOCS if q in doc.lower() or query in doc]


# --------------------------------------------------------------------------- #
# ToolRegistry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
        risk_level: RiskLevel = RiskLevel.SAFE,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"工具已存在: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            risk_level=risk_level,
            handler=handler,
        )

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [t.to_spec() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools


def build_default_registry() -> ToolRegistry:
    """构造包含三个内置工具的注册表。"""

    reg = ToolRegistry()
    reg.register(
        name="echo",
        description="原样返回输入文本，用于测试工具链路是否通畅。",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要回显的文本"},
            },
            "required": [],
        },
        handler=_echo,
        risk_level=RiskLevel.SAFE,
    )
    reg.register(
        name="get_time",
        description="返回当前时间（ISO 8601 字符串）。",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_time,
        risk_level=RiskLevel.SAFE,
    )
    reg.register(
        name="search_docs",
        description="在 demo 文档库里做关键词搜索，返回匹配到的文档列表。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        handler=_search_docs,
        risk_level=RiskLevel.LOW,
    )
    return reg


# --------------------------------------------------------------------------- #
# ToolRuntime
# --------------------------------------------------------------------------- #
class ToolRuntime:
    """执行工具调用。

    职责：按名字找工具 -> 校验参数 -> 执行 handler -> 包成 ToolResult。
    工具不存在或参数错误返回结构化错误，不抛异常给上层。
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = arguments or {}
        start = time.perf_counter()

        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error_type=ToolErrorType.NOT_FOUND,
                error_message=f"工具不存在: {name}",
                latency_ms=_elapsed_ms(start),
            )

        # 参数校验：根据 input_schema 动态建 pydantic 模型
        validated = self._validate(tool, arguments)
        if validated is None:
            return ToolResult(
                ok=False,
                error_type=ToolErrorType.VALIDATION_ERROR,
                error_message=f"参数校验失败: {name}",
                latency_ms=_elapsed_ms(start),
            )

        try:
            output = tool.handler(**validated)
        except Exception as exc:  # noqa: BLE001 - 工具内部任意异常都收口成结构化错误
            return ToolResult(
                ok=False,
                error_type=ToolErrorType.EXECUTION_ERROR,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(start),
            )

        return ToolResult(
            ok=True,
            output=output,
            latency_ms=_elapsed_ms(start),
        )

    def _validate(
        self, tool: RegisteredTool, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        schema = tool.input_schema
        props: dict[str, Any] = schema.get("properties", {}) or {}
        required: list[str] = schema.get("required", []) or []

        # 动态构造一个 pydantic 模型做类型校验
        fields: dict[str, Any] = {}
        type_map = {
            "string": (str, ...),
            "integer": (int, ...),
            "number": (float, ...),
            "boolean": (bool, ...),
            "array": (list, ...),
            "object": (dict, ...),
        }
        for field_name, field_schema in props.items():
            py_type, _ = type_map.get(field_schema.get("type", "string"), (str, ...))
            if field_name in required:
                fields[field_name] = (py_type, ...)
            else:
                fields[field_name] = (py_type, None)

        try:
            Model = create_model(f"{tool.name}_args", **fields)  # noqa: N806
            instance = Model(**arguments)
            return instance.model_dump(exclude_none=True)
        except ValidationError:
            return None


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
