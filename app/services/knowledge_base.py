"""LLMOps 知识库召回客户端。

这里有两个**各自独立**的召回入口，都属于跨库检索接口 `kbs:mix-retrieve` 那一族：

- `faq_retrieve()` —— FAQ 探测：`keywords` 恒为空（没有改写就没有关键词），
  探哪些库由 `RagAgent._faq_probe` 先按 `knowledgeType=2` 收窄；
- `mix_retrieve()` —— 最终答案的跨库混合检索：`query` 负责语义召回、
  `keywords` 负责业务词命中。

两个接口由不同方提供、地址各自可整条覆盖（`KB_MIX_RETRIEVE_URL` / `KB_FAQ_RETRIEVE_URL`），
**请求流程与请求体各写一份**（同形的流程就地各写一遍，改一条不会误伤另一条）；但**响应结构
两边相同** —— 都是 `result[]` 里同一套命中字段（含 QA 库的 `chunk.qa_pairs`），所以解析
只有一份：`_parse()` / `_to_hit()`，两条通道都调它。**请求形状各写、响应解析共用**就是这个
文件的骨架。

两条通道的请求形状**完全不同**，别互相抄：
- `mix_retrieve()` —— 外部方给的接口，**一个普通的 POST**：地址 + JSON body，body 只有
  `query` / `keywords` / `ranges` 三项，`ranges` 直接用本次请求的 ranges（门户传什么发什么）、
  不带查询参数、不带任何请求头，一次请求把全部知识库发出去；
- `faq_retrieve()` —— 平台接口：`project_id` / `tenantId` 查询参数，
  body 里除前三项外还要 `disable_rerank=false` 与 `rerank_params`（跨库时各库相似度度量体系
  不一致，接口要求重排必须开启），ranges 只带平台认识的字段，跨空间时按
  (project_id, tenantId) 分组、每组各发一次（见 `_group_by_space()`）。
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
        """FAQ 探测：在标准问答库上召回，`keywords` 恒为空（没有改写，也就没有关键词）。

        平台接口，请求体见 `_faq_body()`。按 (project_id, tenantId) 把 ranges 分组并发召回后
        合并排序（知识库不支持跨空间检索）。单组失败只记 warning 并跳过 —— 召回是增强环节，
        命不命中 FAQ 都不该让问答链路报错，降级成空列表后由调用方决定怎么表达
        （见 `RagAgent.stream`）。
        """
        groups = self._group_by_space(ranges)
        outcomes = await asyncio.gather(
            *[
                self._faq_request(self._faq_body(query, items), scope)
                for scope, items in groups.items()
            ],
            return_exceptions=True,
        )
        hits: list[RetrievalHit] = []
        for (project_id, tenant_id), outcome in zip(groups, outcomes):
            if isinstance(outcome, Exception):
                logger.warning(
                    "FAQ 召回失败（project_id=%s tenantId=%s）: %s", project_id, tenant_id, outcome
                )
            else:
                hits.extend(outcome)
        return sorted(hits, key=lambda item: item.score, reverse=True)

    async def mix_retrieve(
        self, query: str, keywords: list[str], ranges: list[KnowledgeRange]
    ) -> list[RetrievalHit]:
        """最终答案的跨库混合检索，地址取环境变量 `KB_MIX_RETRIEVE_URL`。

        外部方给的接口，**一个普通的 POST**：body 只有下面三项，`ranges` **直接用本次请求的
        ranges**（原样带上门户传的 `project_id` / `tenantId` / `knowledgeType`，对方按它定位
        知识库），不带查询参数、不加任何请求头。一次请求把全部知识库发出去，结果由平台统一
        重排，按分数降序返回。

        失败只记 warning 并降级成空列表 —— 召回是增强环节，不该让问答链路报错
        （见 `RagAgent.stream` 对空结果的处理）。
        """
        payload = {
            "query": query,
            "keywords": keywords,
            # 本次请求的 ranges 原样出网：键名还原成门户的写法，没传的字段不补空值
            "ranges": [item.model_dump(by_alias=True, exclude_unset=True) for item in ranges],
        }
        try:
            response = await self.client.post(self.settings.mix_retrieve_url, json=payload)
            response.raise_for_status()
            return self._parse(response.json())
        except Exception as error:  # noqa: BLE001 —— 外部接口，任何失败都只降级不抛
            logger.warning("跨库混合检索失败: %s", error)
            return []

    def _faq_body(self, query: str, ranges: list[KnowledgeRange]) -> dict[str, Any]:
        """FAQ 探测的请求体（**平台接口**形状，与外部接口的 `mix_retrieve()` 不同）。

        ranges 只带平台认识的 `knowledge_base_id` / `doc_range`（空间走查询参数）——
        跨库时各库相似度度量体系不一致，接口要求重排必须开启，故固定 `disable_rerank=false`
        并携带 `rerank_params`。
        """
        return {
            "query": query,
            "keywords": [],
            "ranges": self._kb_ranges(ranges),
            "disable_rerank": False,
            "rerank_params": {
                "top_k": self.settings.retrieval_top_k,
                "score_threshold": 0,
                # 1 = 动态权重 WRRF，按向量/全文命中名次加权，无需传 rerank 模型对象
                "weight_type": 1,
                "full_text_rerank_weight": 0.4,
            },
        }

    @staticmethod
    def _kb_ranges(ranges: list[KnowledgeRange]) -> list[dict[str, Any]]:
        """平台接口只认识的 range 字段：知识库 + 可选文档收窄（空间走查询参数）。"""
        return [
            {"knowledge_base_id": item.knowledge_base_id, "doc_range": item.doc_range}
            for item in ranges
        ]

    def _group_by_space(
        self, ranges: list[KnowledgeRange]
    ) -> dict[tuple[str, str], list[KnowledgeRange]]:
        """按 (project_id, tenantId) 分组，一次请求只落一个空间。

        `project_id` 取 range 自带的，缺省才回落 settings（老门户不传这个字段时仍按
        「全局单一空间」的旧行为走，不至于直接不可用）；`tenantId` 只认 range 自带的，
        没有全局兜底。两个都取不到时得到一个空串键，等价于「不分组」。
        """
        groups: dict[tuple[str, str], list[KnowledgeRange]] = {}
        for item in ranges:
            scope = (item.project_id or self.settings.kb_project_id, item.tenant_id)
            groups.setdefault(scope, []).append(item)
        return groups

    async def _faq_request(
        self, payload: dict[str, Any], scope: tuple[str, str]
    ) -> list[RetrievalHit]:
        """FAQ 探测的一次请求：`project_id` / `tenantId` 查询参数。

        地址固定取 `faq_retrieve_url`。这是**平台接口**的请求形状，只服务 `faq_retrieve()`
        —— `mix_retrieve()` 是外部接口的普通 POST，不拼查询参数。
        """
        # project_id / tenantId 是平台接口的**空间级**查询参数
        project_id, tenant_id = scope
        params = {"project_id": project_id}
        if tenant_id:
            params["tenantId"] = tenant_id
        response = await self.client.post(
            self.settings.faq_retrieve_url, params=params, json=payload
        )
        response.raise_for_status()
        return self._parse(response.json())

    @staticmethod
    def _parse(body: dict[str, Any]) -> list[RetrievalHit]:
        """把响应体里的 `result[]` 解析成命中列表，按分数降序。"""
        hits = [KnowledgeBaseClient._to_hit(value) for value in body.get("result", [])]
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
