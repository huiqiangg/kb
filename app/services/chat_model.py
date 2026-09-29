"""按 OpenAI 兼容协议组装 ChatOpenAI（langchain 侧唯一的模型入口）。

内部网关的怪癖集中在这里，避免每个 agent 各写一遍：
- `base_url` 要去掉 `/chat/completions` 后缀（openai 客户端会自己补上）；
  地址**每个模型各配一条完整 URL**（`REWRITE_MODEL_URL` / `ANSWER_MODEL_URL`），
  不做「全局 base + 局部覆盖」那种双层配置 —— 一层就够，也少一处优先级要记；
- `api_key` 位置给 `MODEL_ACCESS_KEY`（没有则占位串），真实鉴权头是 `accessKey`；
  Bearer 与 accessKey 两种头一起发，网关读哪个都行。
"""

from langchain_openai import ChatOpenAI

from app.core.config import Settings


def build_chat_model(settings: Settings, *, url: str, model: str) -> ChatOpenAI:
    headers = {"accessKey": settings.model_access_key} if settings.model_access_key else None
    return ChatOpenAI(
        model=model,
        base_url=url.removesuffix("/chat/completions"),
        api_key=settings.model_access_key or "EMPTY",
        timeout=settings.request_timeout_seconds,
        max_retries=1,
        default_headers=headers,
    )
