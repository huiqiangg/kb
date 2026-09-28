"""LLMOps 知识库召回客户端（单库 FAQ 检索 + 跨库混合检索）。

两条通道对应接口文档的两个不同端点：
- `kbs:retrieve`     单库检索，用 target_range 指定召回范围（本项目只用来取标准问答 FAQ）；
- `kbs:mix-retrieve` 跨库检索，一次请求覆盖多个知识库，由平台统一重排——
  跨库时各库相似度度量体系不一致，接口要求重排必须开启，因此固定
  `disable_rerank=false` 并携带 `rerank_params`。

网关 IP 与路径尚未最终确定，两条地址都支持整条覆盖（`KB_FAQ_RETRIEVE_URL` /
`KB_MIX_RETRIEVE_URL`），留空才按 `KB_PLATFORM_BASE_URL` 推导默认路径 —— 定下地址后
只改环境变量，不必动代码。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.models.chat import KnowledgeRange

logger = logging.getLogger(__name__)

# 召回范围枚举：0 分段，1 文档元信息，2 标准问答，3 实体
FAQ_TARGET_RANGE = 2


@dataclass(slots=True)
class RetrievalHit:
    content: str
    score: float
    knowledge_base_id: str
    doc_id: str | None
    doc_name: str | None
    chunk_id: str | None
    raw: dict[str, Any]


class KnowledgeBaseClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def faq_retrieve(self, query: str, ranges: list[KnowledgeRange]) -> list[RetrievalHit]:
        """按知识库分别召回标准问答，高置信命中可绕过模型直接返回标准答案。"""
        results = await asyncio.gather(
            *(self._retrieve_faq(query, item) for item in ranges), return_exceptions=True
        )
        hits: list[RetrievalHit] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("FAQ 召回失败: %s", result)
            else:
                hits.extend(result)
        return sorted(hits, key=lambda item: item.score, reverse=True)

    async def mix_retrieve(
        self, query: str, keywords: list[str], ranges: list[KnowledgeRange]
    ) -> list[RetrievalHit]:
        """跨库混合检索：一次请求覆盖 ranges 内全部知识库，返回重排后的候选分段。

        失败（含知识库不可达）只记日志并返回空列表，由调用方走「无检索结果」分支，
        不让检索故障整条链路报错。
        """
        payload = {
            "query": query,
            "keywords": keywords,
            "ranges": [
                {"knowledge_base_id": item.knowledge_base_id, "doc_range": item.doc_range}
                for item in ranges
            ],
            "disable_rerank": False,
            "rerank_params": {
                "top_k": self.settings.retrieval_top_k,
                "score_threshold": 0,
                # 1 = 动态权重 WRRF，按向量/全文命中名次加权，无需传 rerank 模型对象
                "weight_type": 1,
                "full_text_rerank_weight": 0.4,
            },
        }
        try:
            body = await self._post(self.settings.mix_retrieve_url, payload)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("跨库混合检索失败: %s", exc)
            return []
        hits = [self._to_hit(value) for value in body.get("result", [])]
        return sorted(hits, key=lambda item: item.score, reverse=True)

    async def _retrieve_faq(self, query: str, range_item: KnowledgeRange) -> list[RetrievalHit]:
        payload = {
            "query": query,
            "knowledge_base_id": range_item.knowledge_base_id,
            "doc_range": range_item.doc_range,
            "target_range": [FAQ_TARGET_RANGE],
            "disable_rerank": True,
            "retrieval_config": {
                "retrieval_strategy_origin": 1,
                "strategy": 2,
                "recall_params": {"top_k": self.settings.retrieval_top_k, "score_threshold": 0},
                "rerank_params": {
                    "top_k": self.settings.retrieval_top_k,
                    "score_threshold": 0,
                    "weight_type": 1,
                    "full_text_rerank_weight": 0.4,
                },
            },
        }
        body = await self._post(self.settings.faq_retrieve_url, payload)
        return [
            self._to_hit(value, range_item.knowledge_base_id)
            for value in body.get("result", [])
        ]

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.kb_authorization:
            headers["Authorization"] = self.settings.kb_authorization
        params = {"project_id": self.settings.kb_project_id}
        if self.settings.kb_tenant_id:
            params["tenantId"] = self.settings.kb_tenant_id
        response = await self.client.post(
            url, params=params, headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _to_hit(value: dict[str, Any], default_kb_id: str = "") -> RetrievalHit:
        chunk = value.get("chunk") or {}
        content = value.get("merged_content", {}).get("text") or chunk.get("content", "")
        # 跨库召回按知识库分别返回，命中项自带 knowledge_base_id，无需默认值
        hit = RetrievalHit(
            content=content,
            score=float(value.get("score") or 0),
            knowledge_base_id=value.get("knowledge_base_id") or default_kb_id,
            doc_id=value.get("doc_id"),
            doc_name=value.get("doc_name"),
            chunk_id=chunk.get("id"),
            raw=value,
        )
        # FAQ 目标可能承载在 qa_pairs 中，优先保留可直接回答的答案。
        qa_pairs = chunk.get("qa_pairs") or []
        if qa_pairs:
            answer = qa_pairs[0].get("answer") or qa_pairs[0].get("a")
            if answer:
                hit.content = answer
        return hit
