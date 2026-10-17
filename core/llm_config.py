"""Centralized, reloadable configuration for OpenAI-compatible LLMs.

Configuration is read on every call so editing the machine-local dotenv file
and rerunning Streamlit takes effect without mutating ``os.environ``.  The
process environment wins over the machine-local file, which wins over the
project ``.env`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 2
PROJECT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
LOCAL_ENV_PATH = Path.home() / ".config" / "meal-parliament" / ".env"


class LLMConfigError(RuntimeError):
    """An actionable configuration error that never includes a secret."""


class LLMConnectionError(RuntimeError):
    """A configured client returned an unusable connection-check response."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    parliament_mode: str = "auto"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def request_options(self) -> dict[str, Any]:
        """Options shared by every chat-completions call for this provider."""
        if self.provider == "deepseek":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {}


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def _setting(name: str, local_values: dict[str, str], project_values: dict[str, str], default: str = "", *, allow_blank: bool = False) -> str:
    # Explicit process values are the highest priority. Empty values normally
    # fall through, while secret settings pass allow_blank=True so a deliberate
    # blank can clear a lower-priority key.
    for values in (os.environ, local_values, project_values):
        if name not in values:
            continue
        value = values[name]
        if value is None:
            continue
        if allow_blank or str(value).strip():
            return str(value).strip()
    return default


def get_llm_config() -> LLMConfig:
    """Return current provider settings without exporting dotenv values."""
    project_values = _read_dotenv(PROJECT_ENV_PATH)
    local_values = _read_dotenv(LOCAL_ENV_PATH)
    provider = _setting("LLM_PROVIDER", local_values, project_values, "openai").lower()
    parliament_mode = _setting("PARLIAMENT_MODE", local_values, project_values, "auto").lower()
    if provider == "deepseek":
        api_key = _setting("DEEPSEEK_API_KEY", local_values, project_values, allow_blank=True) or None
        model = _setting("DEEPSEEK_MODEL", local_values, project_values, "deepseek-flash")
        return LLMConfig(provider=provider, model=model, api_key=api_key, base_url=DEEPSEEK_BASE_URL, parliament_mode=parliament_mode)
    if provider == "openai":
        api_key = _setting("OPENAI_API_KEY", local_values, project_values, allow_blank=True) or None
        model = _setting("OPENAI_MODEL", local_values, project_values, "gpt-4o-mini")
        return LLMConfig(provider=provider, model=model, api_key=api_key, base_url=None, parliament_mode=parliament_mode)
    # Preserve the invalid value for a clear UI/error message; never silently
    # route a typo or unsupported provider to OpenAI.
    return LLMConfig(provider=provider, model="", api_key=None, base_url=None, parliament_mode=parliament_mode)


def create_client(config: LLMConfig | None = None) -> Any:
    """Build the configured OpenAI-compatible client with bounded retries."""
    config = config or get_llm_config()
    if config.provider not in {"openai", "deepseek"}:
        raise LLMConfigError("LLM_PROVIDER 必须是 openai 或 deepseek")
    if not config.configured:
        raise LLMConfigError(f"未配置 {config.provider} API key，请编辑 ~/.config/meal-parliament/.env")
    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def check_connection() -> str:
    """Make the explicit, tiny connection check used by the UI.

    The return value is deliberately the configured model only.  Callers must
    report exception classes rather than exception text, which can contain
    provider or request details.
    """
    config = get_llm_config()
    if config.provider not in {"openai", "deepseek"}:
        raise LLMConfigError("LLM_PROVIDER 必须是 openai 或 deepseek")
    if not config.configured:
        raise LLMConfigError(f"未配置 {config.provider} API key")
    client = create_client(config)
    options: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": "请只回复：已连接"}],
        "max_tokens": 64,
        "temperature": 0,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        **config.request_options,
    }
    response = client.chat.completions.create(**options)
    try:
        content = (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMConnectionError("连接检查返回格式异常") from exc
    if not content:
        raise LLMConnectionError("连接检查返回为空")
    return config.model
