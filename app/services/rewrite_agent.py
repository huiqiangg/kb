"""query 改写：langchain create_agent + OpenAI 兼容协议。

设计取舍：
- 模型调用走 OpenAI chat/completions 格式（ChatOpenAI），网关无需感知 langchain；
- create_agent 不带工具，只借它的图结构做统一编排；调用是非流式的（ainvoke，
  HTTP 层 stream=false）——改写结果必须拿到完整 JSON 才能用，流式增量没有意义；
- 模型按提示词约定输出**纯 JSON 文本**（不经 tool calling，网关无需支持 tools），
  由本模块做容错解析，兼容新旧字段（rewritten_query/query、keywords/tags）；
- keywords 与改写同一次调用产出，替代了原来单独的「标签提取」模型调用。
"""

import json
import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.services.chat_model import build_chat_model
from app.services.text_normalizer import normalize_query

logger = logging.getLogger(__name__)

KEYWORD_LIMIT = 8


class RewriteResult(BaseModel):
    """改写环节的约定输出（与 DB 提示词 query_understanding 的输出格式一致）。"""

    rewritten_query: str = ""
    keywords: list[str] = []


def parse_rewrite_json(text: str) -> RewriteResult:
    """容错解析模型输出的 JSON。

    只接受结构化结果——模型的解释性文字不能带进后续检索；解析失败返回空结果，
    由调用方决定用原始 query 继续检索。
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return RewriteResult()
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return RewriteResult()
    if not isinstance(value, dict):
        return RewriteResult()

    rewritten = ""
    for key in ("rewritten_query", "query"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            rewritten = normalize_query(candidate)
            break

    raw_keywords = value.get("keywords") or value.get("tags") or []
    keywords = (
        [str(item).strip() for item in raw_keywords if str(item).strip()][:KEYWORD_LIMIT]
        if isinstance(raw_keywords, list)
        else []
    )
    return RewriteResult(rewritten_query=rewritten, keywords=keywords)


class QueryRewriteAgent:
    """对术语映射后的 query 做意图识别、改写与关键词提取。"""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._agent = create_agent(model=chat_model, tools=[])

    @classmethod
    def build(cls, settings: Settings) -> "QueryRewriteAgent":
        chat_model = build_chat_model(
            settings, url=settings.rewrite_model_url, model=settings.rewrite_model_name
        )
        return cls(chat_model)

    async def rewrite(self, messages: list[dict[str, str]]) -> RewriteResult:
        """一次性非流式调用，返回结构化改写结果。

        调用失败或输出不可解析都不抛异常：返回空 RewriteResult，
        让调用方用原始 query 继续检索，问答链路不中断。
        """
        try:
            state = await self._agent.ainvoke({"messages": messages})
        except Exception as exc:  # noqa: BLE001 —— 改写是增强环节，失败不该中断问答
            logger.warning("query 改写调用失败: %s", exc)
            return RewriteResult()

        final_text = ""
        if state and state.get("messages"):
            last_content = state["messages"][-1].content
            final_text = last_content if isinstance(last_content, str) else ""
        result = parse_rewrite_json(final_text)
        if final_text and not result.rewritten_query:
            logger.warning("query 改写输出无法解析为结构化结果: %.120s", final_text)
        return result


_agent: QueryRewriteAgent | None = None


def get_query_rewrite_agent() -> QueryRewriteAgent:
    """惰性单例：ChatOpenAI 是无状态客户端，全进程复用一个即可。"""
    global _agent
    if _agent is None:
        _agent = QueryRewriteAgent.build(get_settings())
    return _agent
