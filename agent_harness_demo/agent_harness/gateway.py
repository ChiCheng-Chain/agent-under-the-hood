"""模型网关（第二阶段增强错误治理）。

定义统一的 ModelClient 协议，并提供：
- MockModelClient：规则驱动，支持场景模拟（always_success / timeout_once_then_success /
  rate_limit_then_fallback / auth_error / bad_request）
- OpenAICompatibleClient：调用 OpenAI-compatible API，把 HTTP 错误映射成 ModelErrorType
- ResilientModelGateway：统一处理 timeout_ms / max_retries / exponential backoff /
  fallback client，每次 retry 和 fallback 都写 trace span

错误重试策略：
- AUTH_ERROR、BAD_REQUEST 不重试
- TIMEOUT、RATE_LIMIT、PROVIDER_UNAVAILABLE 可重试
- 主模型重试失败后 fallback 到备用模型
"""
from __future__ import annotations

import time
from typing import Optional, Protocol

import httpx

from .trace import TraceRecorder
from .types import (
    FinishReason,
    Message,
    ModelError,
    ModelErrorType,
    ModelRequest,
    ModelResponse,
    Role,
    RETRYABLE_ERRORS,
    ToolCall,
    ToolSpec,
)


class ModelClient(Protocol):
    """模型客户端协议。invoke 失败时抛 ModelError。"""

    def invoke(self, request: ModelRequest) -> ModelResponse:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# Mock 客户端
# --------------------------------------------------------------------------- #
class MockModelClient:
    """规则驱动的假模型。

    behavior 取值：
    - always_success：按第一阶段规则返回 tool_call / final_answer
    - timeout_once_then_success：第一次 TIMEOUT，第二次成功
    - rate_limit_then_fallback：始终 RATE_LIMIT（交给 gateway fallback）
    - auth_error：始终 AUTH_ERROR，不重试
    - bad_request：始终 BAD_REQUEST，不重试

    场景 always_success 的规则（与第一阶段一致）：
    - 输入含“搜索”或 “search” -> tool_call: search_docs
    - 输入含“时间”或 “time”    -> tool_call: get_time
    - 收到工具 observation -> final_answer
    - 其他情况直接 final_answer
    """

    def __init__(
        self,
        *,
        behavior: str = "always_success",
        call_counter: dict[str, int] | None = None,
        name: str = "mock",
    ) -> None:
        self.behavior = behavior
        self._counter = call_counter if call_counter is not None else {}
        self.name = name
        self._invoke_count = 0

    def invoke(self, request: ModelRequest) -> ModelResponse:
        self._invoke_count += 1
        self._counter["invoke"] = self._counter.get("invoke", 0) + 1

        if self.behavior == "auth_error":
            raise ModelError("模拟鉴权失败", ModelErrorType.AUTH_ERROR, status_code=401)
        if self.behavior == "bad_request":
            raise ModelError("模拟请求参数错误", ModelErrorType.BAD_REQUEST, status_code=400)
        if self.behavior == "rate_limit_then_fallback":
            raise ModelError("模拟限流", ModelErrorType.RATE_LIMIT, status_code=429)
        if self.behavior == "timeout_once_then_success":
            if self._invoke_count == 1:
                raise ModelError("模拟超时", ModelErrorType.TIMEOUT)
            return self._rule_based_response(request)

        # always_success
        return self._rule_based_response(request)

    def _rule_based_response(self, request: ModelRequest) -> ModelResponse:
        has_observation = any(m.role == Role.TOOL for m in request.messages)

        user_text = ""
        for m in reversed(request.messages):
            if m.role == Role.USER:
                user_text = m.content
                break
            if m.role == Role.TOOL:
                user_text = m.content
                break

        if has_observation:
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
                tool_calls=[ToolCall(id="call-mock-time", name="get_time", arguments={})],
            )

        self._counter["final_answer"] = self._counter.get("final_answer", 0) + 1
        return ModelResponse(
            content=f"已收到请求：{user_text}",
            finish_reason=FinishReason.FINAL_ANSWER,
        )

    @property
    def invoke_count(self) -> int:
        return self._invoke_count


def _extract_query(text: str) -> str:
    for sep in ("搜索", "search"):
        needle = sep
        idx = text.find(needle)
        if idx < 0:
            idx = text.lower().find(sep)
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

    把 HTTP 错误映射成 ModelErrorType，让上层 ResilientModelGateway 决定重试。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        *,
        name: str = "openai",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self.name = name

    def invoke(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_payload(request)
        # 优先用 request 上的 timeout_ms，否则用构造默认值
        timeout = (request.timeout_ms / 1000.0) if request.timeout_ms else self._timeout
        try:
            resp = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ModelError(str(exc), ModelErrorType.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise ModelError(str(exc), ModelErrorType.PROVIDER_UNAVAILABLE) from exc

        if resp.status_code == 401 or resp.status_code == 403:
            raise ModelError(f"HTTP {resp.status_code}", ModelErrorType.AUTH_ERROR, status_code=resp.status_code)
        if resp.status_code == 400:
            raise ModelError(f"HTTP 400: {resp.text}", ModelErrorType.BAD_REQUEST, status_code=400)
        if resp.status_code == 429:
            raise ModelError("HTTP 429 限流", ModelErrorType.RATE_LIMIT, status_code=429)
        if resp.status_code >= 500:
            raise ModelError(
                f"HTTP {resp.status_code}", ModelErrorType.PROVIDER_UNAVAILABLE, status_code=resp.status_code
            )
        if resp.status_code >= 400:
            raise ModelError(
                f"HTTP {resp.status_code}: {resp.text}", ModelErrorType.BAD_REQUEST, status_code=resp.status_code
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelError(f"无法解析响应 JSON: {exc}", ModelErrorType.RESPONSE_PARSE_ERROR) from exc

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
                            "arguments": _json_dumps(tc.arguments),
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
        import json

        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"响应缺少 choices: {data}", ModelErrorType.RESPONSE_PARSE_ERROR)
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


def _json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 弹性网关：retry + backoff + fallback
# --------------------------------------------------------------------------- #
class ResilientModelGateway:
    """在 ModelClient 之上叠加重试、退避、fallback，并写 trace。

    - 先对主 client 重试 max_retries 次（指数退避）
    - 仍失败且有 fallback client，则切换到 fallback 再试
    - 每次 retry / fallback 都通过 TraceRecorder 记 span
    - AUTH_ERROR / BAD_REQUEST 立即抛出，不重试
    """

    def __init__(
        self,
        primary: ModelClient,
        *,
        fallback: Optional[ModelClient] = None,
        max_retries: int = 2,
        base_backoff_ms: int = 10,
        max_backoff_ms: int = 1000,
        trace: Optional[TraceRecorder] = None,
        sleep: bool = True,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max_retries
        self._base_backoff_ms = base_backoff_ms
        self._max_backoff_ms = max_backoff_ms
        self._trace = trace
        # sleep=False 时退避不真正 sleep（测试用，加快速度）
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "resilient-gateway"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        """按主->重试->fallback 顺序调用。失败抛最后一个 ModelError。"""

        last_error: Optional[ModelError] = None

        # 主 client：1 次正常 + max_retries 次重试
        for attempt in range(0, self._max_retries + 1):
            try:
                return self._primary.invoke(request)
            except ModelError as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    # 不可重试错误（auth/bad_request）立即抛出，不记 retry span
                    raise
                # 可重试错误才记 retry span
                self._maybe_record_retry(attempt, exc)
                if attempt < self._max_retries:
                    self._backoff(attempt)
                # 否则继续下一次重试

        # 主 client 耗尽重试，尝试 fallback
        if self._fallback is not None:
            self._maybe_record_fallback(last_error)
            try:
                return self._fallback.invoke(request)
            except ModelError as exc:
                last_error = exc
                self._maybe_record_retry(self._max_retries + 1, exc)
                if not self._is_retryable(exc):
                    raise
                # fallback 也失败，抛出
                raise

        # 没有 fallback，抛出最后的错误
        assert last_error is not None
        raise last_error

    def _is_retryable(self, exc: ModelError) -> bool:
        return exc.error_type in RETRYABLE_ERRORS

    def _backoff(self, attempt: int) -> None:
        delay_ms = min(self._base_backoff_ms * (2**attempt), self._max_backoff_ms)
        if self._sleep and delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    def _maybe_record_retry(self, attempt: int, exc: ModelError) -> None:
        if self._trace is None:
            return
        # retry 作为 model span 的子 span（若有当前 model span）或独立 span
        self._trace.record_retry(
            attempt=attempt + 1,
            error_type=exc.error_type.value,
            output_summary=str(exc)[:200],
        )

    def _maybe_record_fallback(self, exc: Optional[ModelError]) -> None:
        if self._trace is None:
            return
        reason = exc.error_type.value if exc else "exhausted_retries"
        self._trace.record_fallback(
            reason=reason,
            output_summary=f"切换到 fallback client",
        )
