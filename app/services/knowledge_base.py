"""LLMOps 知识库召回客户端。

FAQ 探测与最终答案检索走的是**同一个接口**（接口文档第 8 节跨库检索 `kbs:mix-retrieve`），
请求体与响应结构完全一致，差别只有两点：探哪些库（FAQ 只探门户标了 `knowledgeType=2` 的
标准问答库，由 `RagAgent._faq_probe` 收窄）、以及**打到哪个地址**。所以这里只有一个
`_retrieve()`，两个入口各传一条地址 —— 两条通道可能是同一接口的两处部署（FAQ 库不保证
挂在同一个网关上），因此地址各自可整条覆盖：`KB_FAQ_RETRIEVE_URL`（留空复用 mix）。
网关 IP 与路径尚未最终确定，定下来只改环境变量，不必动代码。

跨库检索一次请求可覆盖多个知识库，由平台统一重排 —— 跨库时各库相似度度量体系不一致，
接口要求重排必须开启，因此固定 `disable_rerank=false` 并携带 `rerank_params`。
但**知识库不支持跨空间检索**：平台层级是「租户 -> 空间(project_id) -> 知识库」，而
`project_id` / `tenantId` 是请求级查询参数，所以 ranges 跨空间时必须按 (project_id, tenantId)
分组，每组各发一次（见 `_group_by_space`）。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.models.chat import KnowledgeRange

logger = logging.getLogger(__name__)

# 知识库类型（门户 knowledge_type 字段）：1 切片库，2 标准问答（QA）库
KNOWLEDGE_TYPE_QA = 2


@dataclass(slots=True)
class RetrievalHit:
    content: str
    score: float
    knowledge_base_id: str
    doc_id: str | None
    doc_name: str | None
    chunk_id: str | None
    # 标准问答库命中时该 QA 对的问题原文（切片库命中为 None），供「其他相近问题」用
    question: str | None = None


class KnowledgeBaseClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def faq_retrieve(self, query: str, ranges: list[KnowledgeRange]) -> list[RetrievalHit]:
        """FAQ 探测：在标准问答库上召回，`keywords` 恒为空（没有改写，也就没有关键词）。"""
        return await self._retrieve(query, [], ranges, self.settings.faq_retrieve_url)

    async def mix_retrieve(
        self, query: str, keywords: list[str], ranges: list[KnowledgeRange]
    ) -> list[RetrievalHit]:
        """最终答案的跨库混合检索：`query` 负责语义召回，`keywords` 负责业务词命中。"""
        return await self._retrieve(query, keywords, ranges, self.settings.mix_retrieve_url)

    async def _retrieve(
        self,
        query: str,
        keywords: list[str],
        ranges: list[KnowledgeRange],
        url: str,
    ) -> list[RetrievalHit]:
        """按 (project_id, tenantId) 把 ranges 分组，各组并发召回后合并排序。

        组内多个知识库共享一次请求与一次重排。单组失败只记 warning 并跳过 ——
        召回是增强环节，命不命中 FAQ、有没有参考资料都不该让问答链路报错，
        降级成空列表后由调用方决定怎么表达（见 `RagAgent.stream`）。
        """
        groups = self._group_by_space(ranges)
        requests = [
            self._request(url, self._payload(query, keywords, items), scope)
            for scope, items in groups.items()
        ]
        outcomes = await asyncio.gather(*requests, return_exceptions=True)
        hits: list[RetrievalHit] = []
        for (project_id, tenant_id), outcome in zip(groups, outcomes):
            if isinstance(outcome, Exception):
                logger.warning(
                    "跨库召回失败（project_id=%s tenantId=%s）: %s", project_id, tenant_id, outcome
                )
            else:
                hits.extend(outcome)
        return sorted(hits, key=lambda item: item.score, reverse=True)

    def _payload(
        self, query: str, keywords: list[str], ranges: list[KnowledgeRange]
    ) -> dict[str, Any]:
        """跨库检索请求体。ranges 只带平台认识的字段，空间信息走查询参数。"""
        return {
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

    def _group_by_space(
        self, ranges: list[KnowledgeRange]
    ) -> dict[tuple[str, str], list[KnowledgeRange]]:
        """按 (project_id, tenantId) 分组，一次请求只落一个空间。

        range 自带空间信息就用自己的；缺省才回落 settings —— 老门户不传这两个字段时
        仍按「全局单一空间」的旧行为走，不至于直接不可用。
        """
        groups: dict[tuple[str, str], list[KnowledgeRange]] = {}
        for item in ranges:
            scope = (
                item.project_id or self.settings.kb_project_id,
                item.tenant_id or self.settings.kb_tenant_id,
            )
            groups.setdefault(scope, []).append(item)
        return groups

    async def _request(
        self, url: str, payload: dict[str, Any], scope: tuple[str, str]
    ) -> list[RetrievalHit]:
        """发一次召回请求并按分数降序解析结果。"""
        headers = {"Content-Type": "application/json"}
        if self.settings.kb_authorization:
            headers["Authorization"] = self.settings.kb_authorization
        # project_id / tenantId 是接口必填的**空间级**查询参数
        project_id, tenant_id = scope
        params = {"project_id": project_id}
        if tenant_id:
            params["tenantId"] = tenant_id
        response = await self.client.post(url, params=params, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        hits = [self._to_hit(value) for value in body.get("result", [])]
        return sorted(hits, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _to_hit(value: dict[str, Any]) -> RetrievalHit:
        chunk = value.get("chunk") or {}
        content = value.get("merged_content", {}).get("text") or chunk.get("content", "")
        # 跨库召回按知识库分别返回，命中项自带 knowledge_base_id，无需默认值
        hit = RetrievalHit(
            content=content,
            score=float(value.get("score") or 0),
            knowledge_base_id=value.get("knowledge_base_id") or "",
            doc_id=value.get("doc_id"),
            doc_name=value.get("doc_name"),
            chunk_id=chunk.get("id"),
        )
        # 标准问答库的命中可能承载在 qa_pairs 中，优先保留可直接回答的答案。
        qa_pairs = chunk.get("qa_pairs") or []
        if qa_pairs:
            pair = qa_pairs[0]
            answer = pair.get("answer") or pair.get("a")
            if answer:
                hit.content = answer
            hit.question = pair.get("question") or pair.get("q") or None
        return hit
