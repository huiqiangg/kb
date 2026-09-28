"""把 chat messages 交给模型并逐字流式返回文本：langchain create_agent + OpenAI 兼容协议。

FAQ 直返的润色与 RAG 链路的最终答案生成共用这一个 agent —— 两者都是「喂 messages、
要一段流式文本」，用的是同一个回答模型，差别只在提示词。拆成两个类只会重复实现，
因此这里只保留一个流式 agent，由调用方决定装配哪条提示词。

与 rewrite_agent 的差别：改写要的是完整 JSON，必须非流式一次取回；这里要的就是最终答案，
用户要逐字看到，所以走 `astream` 增量下发。
"""

import logging
from collections.abc import AsyncIterator

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.services.chat_model import build_chat_model

logger = logging.getLogger(__name__)


class ChatStreamAgent:
    """流式问答 agent：create_agent 不带工具，只借它的图结构做统一编排。"""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._agent = create_agent(model=chat_model, tools=[])

    @classmethod
    def build(cls, settings: Settings) -> "ChatStreamAgent":
        """用回答模型构建；未配置 ANSWER_MODEL_URL 时抛 RuntimeError，由调用方降级。"""
        if not settings.answer_model_url:
            raise RuntimeError("未配置 ANSWER_MODEL_URL")
        chat_model = build_chat_model(
            settings, url=settings.answer_model_url, model=settings.answer_model_name
        )
        return cls(chat_model)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """增量产出文本。

        调用失败只记一条 warning 就结束——把「降级成什么」留给调用方决定
        （FAQ 直返回落库中原文，RAG 链路回落一句提示语），不在这里替它做主。
        """
        try:
            async for chunk in self._agent.astream(
                {"messages": messages}, stream_mode="messages"
            ):
                token, _metadata = chunk
                content = token.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:  # noqa: BLE001 —— 生成是最后一环，失败不该掀掉整条流
            logger.warning("模型流式生成失败: %s", exc)


_agent: ChatStreamAgent | None = None


def get_chat_stream_agent() -> ChatStreamAgent:
    """惰性单例：ChatOpenAI 是无状态客户端，全进程复用一个即可。"""
    global _agent
    if _agent is None:
        _agent = ChatStreamAgent.build(get_settings())
    return _agent
