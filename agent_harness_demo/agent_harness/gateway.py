"""模型网关。

定义统一 ModelClient 协议，并提供两个实现：
- MockModelClient：按规则返回 tool_call / final_answer，用于无模型环境跑通闭环
- OpenAICompatibleClient：调用 OpenAI-compatible API

第一版不做复杂 retry，只定义错误类型占位。
"""
from __future__ import annotations

import json
from typing import Protocol

import httpx

from .types import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolSpec,
)


class ModelError(Exception):
    """模型调用错误的基类。"""


class ModelConnectionError(ModelError):
    """网络层错误（连接失败、超时）。"""


class ModelAPIError(ModelError):
    """API 返回了非 2xx 或响应不可解析。"""


class ModelClient(Protocol):
    """模型客户端协议。"""

    def invoke(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Mock 客户端
# --------------------------------------------------------------------------- #
class MockModelClient:
    """规则驱动的假模型，用来在无模型环境验证 harness 闭环。

    行为约定：
    - 输入含“搜索”或 “search” -> tool_call: search_docs
    - 输入含“时间”或 “time”    -> tool_call: get_time
    - 收到工具 observation（对话历史里出现 role=tool 的消息）-> final_answer
    - 其他情况直接 final_answer
    """

    def __init__(self, *, call_counter: dict[str, int] | None = None) -> None:
        # 仅测试观察用：记录每种 finish_reason 触发次数
        self._counter = call_counter if call_counter is not None else {}

    def invoke(self, request: ModelRequest) -> ModelResponse:
        # 判定是否已经收到工具 observation：检查最后一条非 assistant 消息
        has_observation = any(m.role == Role.TOOL for m in request.messages)

        # 取最近一条 user 消息作为意图判定依据；没有就用历史拼接
        user_text = ""
        for m in reversed(request.messages):
            if m.role == Role.USER:
                user_text = m.content
                break
            if m.role == Role.TOOL:
                # 工具结果作为最终答案素材
                user_text = m.content
                break

        if has_observation:
            # 工具已经跑完，直接基于 observation 生成最终答案
            self._counter["final_answer"] = self._counter.get("final_answer", 0) + 1
            return ModelResponse(
                content=f"根据工具结果：{user_text}",
                finish_reason=FinishReason.FINAL_ANSWER,
            )

        lowered = user_text.lower()
        if "搜索" in user_text or "search" in lowered:
            self._counter["search_docs"] = self._counter.get("search_docs", 0) + 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALL,
                tool_calls=[
                    ToolCall(
                        id="call-mock-search",
                        name="search_docs",
                        arguments={"query": _extract_query(user_text)},
                    )
                ],
            )

        if "时间" in user_text or "time" in lowered:
            self._counter["get_time"] = self._counter.get("get_time", 0) + 1
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALL,
                tool_calls=[
                    ToolCall(
                        id="call-mock-time",
                        name="get_time",
                        arguments={},
                    )
                ],
            )

        self._counter["final_answer"] = self._counter.get("final_answer", 0) + 1
        return ModelResponse(
            content=f"已收到请求：{user_text}",
            finish_reason=FinishReason.FINAL_ANSWER,
        )


def _extract_query(text: str) -> str:
    """从 “帮我搜索 refund policy” 这类输入里抠出查询词。"""

    for sep in ("搜索", "search"):
        if sep in text.lower() or sep in text:
            idx = text.lower().find(sep) if sep == "search" else text.find(sep)
            if idx >= 0:
                rest = text[idx + len(sep) :].strip(" :：")
                if rest:
                    return rest
    return text


# --------------------------------------------------------------------------- #
# OpenAI-compatible 客户端
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient:
    """调用 OpenAI /v1/chat/completions 接口。

    通过 tools 字段传入工具 schema，解析 choices[0].message.tool_calls。
    如果模型没有调用工具，则把 content 作为 final_answer 返回。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def invoke(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_payload(request)
        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ModelConnectionError(str(exc)) from exc

        if resp.status_code >= 400:
            raise ModelAPIError(f"HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelAPIError(f"无法解析响应 JSON: {exc}") from exc

        return self._parse_response(data)

    def _build_payload(self, request: ModelRequest) -> dict:
        messages = []
        for m in request.messages:
            item: dict = {"role": m.role.value, "content": m.content}
            if m.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.role == Role.TOOL:
                item["tool_call_id"] = m.tool_call_id
            messages.append(item)

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [self._spec_to_openai(t) for t in request.tools]
        return payload

    @staticmethod
    def _spec_to_openai(spec: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

    def _parse_response(self, data: dict) -> ModelResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ModelAPIError(f"响应缺少 choices: {data}")
        msg = choices[0].get("message", {})

        raw_calls = msg.get("tool_calls") or []
        tool_calls = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            arg_str = fn.get("arguments", "{}")
            try:
                arguments = json.loads(arg_str) if arg_str else {}
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": arg_str}
            tool_calls.append(
                ToolCall(id=tc.get("id", "call"), name=fn.get("name", ""), arguments=arguments)
            )

        if tool_calls:
            return ModelResponse(
                content=msg.get("content", "") or "",
                tool_calls=tool_calls,
                finish_reason=FinishReason.TOOL_CALL,
                raw=data,
            )

        return ModelResponse(
            content=msg.get("content", "") or "",
            finish_reason=FinishReason.FINAL_ANSWER,
            raw=data,
        )
