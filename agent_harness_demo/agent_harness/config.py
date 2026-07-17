"""集中读取配置。

从环境变量 / .env 读取，并提供默认值。单独抽出来是为了让其它模块不直接
接触 os.environ，方便测试时注入配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    use_real_model: bool
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    max_steps: int
    trace_file: str

    @property
    def trace_path(self) -> Path:
        return Path(self.trace_file)


def load_settings(env_file: str | None = ".env") -> Settings:
    """读取配置。env_file=None 时跳过 .env 加载（测试用）。"""

    if env_file is not None:
        # load_dotenv 找不到文件会静默跳过，不会报错
        load_dotenv(env_file)

    def _bool(value: str | None) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        use_real_model=_bool(os.getenv("USE_REAL_MODEL")),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        openai_model=os.getenv("OPENAI_MODEL", "default-model"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
        trace_file=os.getenv("TRACE_FILE", "traces.jsonl"),
    )
