"""按 OpenAI 兼容协议组装 ChatOpenAI（langchain 侧唯一的模型入口）。

内部网关的怪癖集中在这里，避免每个 agent 各写一遍：
- `base_url` 要去掉 `/chat/completions` 后缀（openai 客户端会自己补上）；
- `api_key` 给占位符即可，真实鉴权头放在 `default_headers`（如 `accessKey`）；
  Bearer 与 accessKey 两种头一起发，网关读哪个都行。
"""

from langchain_openai import ChatOpenAI

from app.core.config import Settings


def build_chat_model(settings: Settings, *, url: str, model: str) -> ChatOpenAI:
    base_url = settings.openai_base_url or url.removesuffix("/chat/completions")
    api_key = settings.openai_api_key or settings.model_access_key or "EMPTY"
    headers = {"accessKey": settings.model_access_key} if settings.model_access_key else None
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=1,
        default_headers=headers,
    )
