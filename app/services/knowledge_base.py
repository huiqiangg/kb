"""LLMOps 知识库召回客户端。

两条召回通道（FAQ 探测 / 最终答案检索）都走接口文档第 8 节的**跨库检索**
`kbs:mix-retrieve`：

- 跨库检索一次请求可覆盖多个知识库，由平台统一重排 —— 跨库时各库相似度度量体系
  不一致，接口要求重排必须开启，因此固定 `disable_rerank=false` 并携带 `rerank_params`；
- 但**知识库不支持跨空间检索**。平台层级是「租户 -> 空间(project_id) -> 知识库」，
  而 `project_id` / `tenantId` 是请求级查询参数，所以 ranges 跨空间时不能塞进一次请求，
  必须按 (project_id, tenantId) 分组，每组各发一次（见 `_group_by_space`）。

FAQ 探测（`faq_retrieve`）与最终答案检索（`mix_retrieve`）**请求体与响应结构完全相同**，
区别只在两处：探测哪些库（前者只探门户标了 `knowledgeType=2` 的标准问答库，由
`RagAgent._faq_probe` 过滤）、以及**打到哪个地址**。两者可能是同一接口的不同部署
（FAQ 库不挂在同一个网关上），所以地址各自可整条覆盖：
`KB_MIX_RETRIEVE_URL`（最终检索）/ `KB_FAQ_RETRIEVE_URL`（FAQ 探测，留空复用前者）。
网关 IP 与路径尚未最终确定，留空才按 `KB_PLATFORM_BASE_URL` 推导默认路径 ——
定下地址后只改环境变量，不必动代码。
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

# 空间作用域 (project_id, tenantId)：一次召回请求只能落在一个空间里
SpaceScope = tuple[str, str]


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
        """FAQ 探测：在标准问答库上做跨库召回，高置信命中可绕过模型直接返回标准答案。

        调用方已把 ranges 收窄到 `knowledgeType=2`，这里只负责按空间分组发请求：
        组内多个 QA 库共享一次请求与一次重排，组与组之间并发。单组失败只记日志并跳过，
        不让 FAQ 探测的故障影响后续链路。请求打到 `faq_retrieve_url`（可与最终检索不同）。
        """
        groups = self._group_by_space(ranges)
        if not groups:
            return []
        outcomes = await asyncio.gather(
            *(
                self._retrieve(
                    self.settings.faq_retrieve_url, self._payload(query, [], items), scope
                )
                for scope, items in groups.items()
            ),
            return_exceptions=True,
        )
        hits: list[RetrievalHit] = []
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.warning("FAQ 跨库召回失败: %s", outcome)
            else:
                hits.extend(outcome)
        return sorted(hits, key=lambda item: item.score, reverse=True)

    async def mix_retrieve(
        self, query: str, keywords: list[str], ranges: list[KnowledgeRange]
    ) -> list[RetrievalHit]:
        """最终答案的跨库混合检索：返回重排后的候选分段。

        失败（含知识库不可达）只记日志并返回空列表，由调用方走「无检索结果」分支，
        不让检索故障整条链路报错。

        注意：当前仍按全局 `KB_PROJECT_ID` / `KB_TENANT_ID` 定位空间，
        ranges 自带的空间信息尚未生效（跨空间 ranges 需要像 FAQ 那样分组）。
        """
        payload = self._payload(query, keywords, ranges)
        scope: SpaceScope = (self.settings.kb_project_id, self.settings.kb_tenant_id)
        try:
            return await self._retrieve(self.settings.mix_retrieve_url, payload, scope)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("跨库混合检索失败: %s", exc)
            return []

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

    async def _retrieve(
        self, url: str, payload: dict[str, Any], scope: SpaceScope
    ) -> list[RetrievalHit]:
        """打到指定地址做一次跨库检索并解析结果。

        地址由调用方传入而不是内部固定用 `mix_retrieve_url` —— FAQ 探测与最终答案检索
        可能是同一接口的不同部署（两个 env 各自可覆盖）。
        """
        body = await self._post(url, payload, scope)
        hits = [self._to_hit(value) for value in body.get("result", [])]
        return sorted(hits, key=lambda item: item.score, reverse=True)

    def _group_by_space(
        self, ranges: list[KnowledgeRange]
    ) -> dict[SpaceScope, list[KnowledgeRange]]:
        """按 (project_id, tenantId) 把 ranges 分组，一次请求只落一个空间。

        range 自带空间信息就用自己的；缺省才回落 settings —— 老门户不传这两个字段时
        仍按「全局单一空间」的旧行为走，不至于直接不可用。
        """
        groups: dict[SpaceScope, list[KnowledgeRange]] = {}
        for item in ranges:
            scope: SpaceScope = (
                item.project_id or self.settings.kb_project_id,
                item.tenant_id or self.settings.kb_tenant_id,
            )
            groups.setdefault(scope, []).append(item)
        return groups

    async def _post(
        self, url: str, payload: dict[str, Any], scope: SpaceScope
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.kb_authorization:
            headers["Authorization"] = self.settings.kb_authorization
        # project_id / tenantId 是接口必填的**空间级**查询参数
        project_id, tenant_id = scope
        params = {"project_id": project_id}
        if tenant_id:
            params["tenantId"] = tenant_id
        response = await self.client.post(
            url, params=params, headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()

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
            raw=value,
        )
        # 标准问答库的命中可能承载在 qa_pairs 中，优先保留可直接回答的答案。
        qa_pairs = chunk.get("qa_pairs") or []
        if qa_pairs:
            answer = qa_pairs[0].get("answer") or qa_pairs[0].get("a")
            if answer:
                hit.content = answer
        return hit
