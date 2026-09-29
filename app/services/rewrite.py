"""改写环节的输出约定：模型按提示词输出**纯 JSON 文本**，这里做容错解析。

为什么不用 tool calling / `response_format`：内部网关不认 tools（也不保证支持
`response_format`），所以让模型照提示词直接吐 JSON，解析不出来就当作没改写。
只接受结构化结果 —— 模型的解释性文字不能带进后续检索。

模型客户端与调用**不在这里**：改写模型和回答模型一样，由 `RagAgent` 就地组装、
就地调用（见 `RagAgent._ensure_rewrite_model` / `_rewrite`），这里只留纯函数，
没有可以把玩的客户端对象。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.services.text_normalizer import normalize_query

KEYWORD_LIMIT = 8


@dataclass(slots=True)
class RewriteResult:
    """改写环节的约定输出（与 DB 提示词 `query_understanding` 的输出格式一致）。"""

    rewritten_query: str = ""
    keywords: list[str] = field(default_factory=list)


def parse_rewrite_json(text: str) -> RewriteResult:
    """容错解析模型输出的 JSON，兼容新旧字段名（`rewritten_query`/`query`、`keywords`/`tags`）。

    解析不出来（模型答了人话、JSON 被截断、网关回了个错误页）就返回空结果，
    由调用方用原 query 继续检索，不让改写这一环掀掉整条问答。
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
