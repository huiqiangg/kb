import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.models.chat import ChatCompletionRequest
from app.services.knowledge_base import KnowledgeBaseClient, RetrievalHit
from app.services.model_client import ModelClient
from app.services.sse import sse
from app.services.text_normalizer import map_business_terms, normalize_query

logger = logging.getLogger(__name__)


class RagAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        query = request.messages[-1].content
        normalized = normalize_query(query)
        mapped = map_business_terms(normalized)
        yield sse("status", {"stage": "normalized", "query": mapped})

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            kb = KnowledgeBaseClient(self.settings, client)
            model = ModelClient(self.settings, client)
            # 原问题与术语映射问题，各自并行查询知识切片和 FAQ。
            initial = await self._parallel_search(kb, query, mapped, request)
            direct = self._faq_match(initial["faq"])
            if direct:
                yield sse("sources", self._sources([direct]))
                yield sse("token", {"content": direct.content})
                yield sse("done", {"answer_type": "faq", "query": mapped})
                return

            yield sse("status", {"stage": "rewrite", "message": "正在理解问题并提取业务标签"})
            rewrite_messages = self._rewrite_messages(request.messages, mapped)
            tag_task = asyncio.create_task(model.extract_tags(self._tag_messages(mapped)))
            try:
                rewritten_parts: list[str] = []
                async for token in model.stream_rewrite(rewrite_messages):
                    rewritten_parts.append(token)
                    yield sse("rewrite_token", {"content": token})
                rewritten = self._parse_rewrite("".join(rewritten_parts))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning("query 改写失败: %s", exc)
                rewritten = ""
            try:
                tags = await tag_task
                if tags:
                    yield sse("status", {"stage": "tags_extracted", "tags": tags})
            except httpx.HTTPError as exc:
                logger.warning("标签提取失败: %s", exc)

            if rewritten:
                yield sse("status", {"stage": "rewritten", "query": rewritten})
                retried = await self._parallel_search(kb, query, rewritten, request)
                direct = self._faq_match(retried["faq"])
                if direct:
                    yield sse("sources", self._sources([direct]))
                    yield sse("token", {"content": direct.content})
                    yield sse("done", {"answer_type": "faq_rewrite", "query": rewritten})
                    return
                hits = retried["kb"] or initial["kb"]
            else:
                yield sse("status", {"stage": "rewrite_failed", "message": "改写失败，使用原问题检索"})
                hits = initial["kb"]

            yield sse("status", {"stage": "hybrid_retrieval", "message": "正在进行多路混合检索"})
            # 检索请求采用向量+全文混合与动态权重重排，结果即多路召回后的候选。
            sources = hits[: self.settings.final_context_top_k]
            if not sources:
                yield sse("token", {"content": "未在已授权知识库中检索到可用于回答的内容。"})
                yield sse("done", {"answer_type": "no_context"})
                return
            yield sse("sources", self._sources(sources))
            async for token in model.stream_answer(self._answer_messages(request.messages, sources)):
                yield sse("token", {"content": token})
            yield sse("done", {"answer_type": "rag", "query": rewritten or mapped})

    async def _parallel_search(self, kb: KnowledgeBaseClient, raw: str, mapped: str, request: ChatCompletionRequest) -> dict[str, list[RetrievalHit]]:
        tasks = (
            kb.retrieve(raw, request.ranges, 0), kb.retrieve(mapped, request.ranges, 0),
            kb.retrieve(raw, request.ranges, 2), kb.retrieve(mapped, request.ranges, 2),
        )
        raw_kb, mapped_kb, raw_faq, mapped_faq = await asyncio.gather(*tasks)
        return {"kb": self._dedupe(raw_kb + mapped_kb), "faq": self._dedupe(raw_faq + mapped_faq)}

    def _faq_match(self, hits: list[RetrievalHit]) -> RetrievalHit | None:
        return next((item for item in hits if item.score >= self.settings.faq_similarity_threshold and item.content), None)

    @staticmethod
    def _dedupe(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        seen: set[str] = set()
        result = []
        for item in sorted(hits, key=lambda value: value.score, reverse=True):
            key = item.chunk_id or f"{item.knowledge_base_id}:{item.content}"
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _rewrite_messages(history: list, query: str) -> list[dict[str, str]]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是农商行业务查询改写器。结合上下文，将最后一个问题改写为可检索的银行标准术语问题，并提取关键业务标签。严格只返回 JSON：{{\"query\": \"改写问题\", \"tags\": [\"标签\"]}}。"),
            ("user", "当前标准化问题：{query}"),
        ])
        messages = [{"role": item.role, "content": item.content} for item in history[-8:]]
        messages.extend({"role": m.type, "content": m.content} for m in prompt.format_messages(query=query))
        return messages

    @staticmethod
    def _parse_rewrite(output: str) -> str:
        """只接受有效的结构化改写，避免把模型解释文字带入第二轮检索。"""
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end <= start:
            return ""
        try:
            value = json.loads(output[start : end + 1])
        except json.JSONDecodeError:
            return ""
        query = value.get("query")
        return normalize_query(query) if isinstance(query, str) and query.strip() else ""

    @staticmethod
    def _tag_messages(query: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "你是农商行业务标签提取器。严格只返回 JSON：{\"tags\":[\"产品\",\"业务动作\",\"客群\"]}。标签不超过 8 个。"},
            {"role": "user", "content": query},
        ]

    @staticmethod
    def _answer_messages(history: list, hits: list[RetrievalHit]) -> list[dict[str, str]]:
        context = "\n\n".join(f"[资料{i + 1}] {hit.content}" for i, hit in enumerate(hits))
        system = (
            "你是农商行知识问答助手。仅根据给出的资料回答；资料不足时明确说明。"
            "回答准确、简明，不编造制度、利率、期限或办理要求。\n\n" + context
        )
        return [{"role": "system", "content": system}] + [
            {"role": item.role, "content": item.content} for item in history[-10:]
        ]

    @staticmethod
    def _sources(hits: list[RetrievalHit]) -> dict:
        return {"items": [{"knowledge_base_id": item.knowledge_base_id, "doc_id": item.doc_id, "doc_name": item.doc_name, "chunk_id": item.chunk_id, "score": item.score} for item in hits]}
