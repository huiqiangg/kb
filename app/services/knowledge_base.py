import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.models.chat import KnowledgeRange

logger = logging.getLogger(__name__)


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

    async def retrieve(
        self, query: str, ranges: list[KnowledgeRange], target_range: int
    ) -> list[RetrievalHit]:
        """按知识库分别调用检索 API，兼容每个 range 的文档范围。"""
        results = await asyncio.gather(
            *(self._retrieve_one(query, item, target_range) for item in ranges),
            return_exceptions=True,
        )
        hits: list[RetrievalHit] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("知识库检索失败: %s", result)
            else:
                hits.extend(result)
        return sorted(hits, key=lambda item: item.score, reverse=True)

    async def _retrieve_one(
        self, query: str, range_item: KnowledgeRange, target_range: int
    ) -> list[RetrievalHit]:
        payload = {
            "query": query,
            "knowledge_base_id": range_item.knowledge_base_id,
            "doc_range": range_item.doc_range,
            "target_range": [target_range],
            "disable_rerank": target_range == 2,
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
        headers = {"Content-Type": "application/json"}
        if self.settings.kb_authorization:
            headers["Authorization"] = self.settings.kb_authorization
        response = await self.client.post(
            f"{self.settings.kb_platform_base_url.rstrip('/')}/applet/api/v1/knowlhub/kbs:retrieve",
            params={"project_id": self.settings.kb_project_id},
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        return [self._to_hit(value, range_item.knowledge_base_id) for value in body.get("result", [])]

    @staticmethod
    def _to_hit(value: dict[str, Any], default_kb_id: str) -> RetrievalHit:
        chunk = value.get("chunk") or {}
        content = value.get("merged_content", {}).get("text") or chunk.get("content", "")
        # FAQ 目标可能承载在 qa_pairs 中，优先保留可直接回答的答案。
        qa_pairs = chunk.get("qa_pairs") or []
        if qa_pairs:
            answer = qa_pairs[0].get("answer") or qa_pairs[0].get("a")
            if answer:
                content = answer
        return RetrievalHit(
            content=content,
            score=float(value.get("score") or 0),
            knowledge_base_id=value.get("knowledge_base_id", default_kb_id),
            doc_id=value.get("doc_id"), doc_name=value.get("doc_name"),
            chunk_id=chunk.get("id"), raw=value,
        )
