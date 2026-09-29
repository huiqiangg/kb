import asyncio
import logging
from collections.abc import AsyncIterator, Iterator

import httpx

from app.core.config import Settings
from app.models.chat import ChatCompletionRequest, KnowledgeRange
from app.models.prompt import Prompt
from app.services.knowledge_base import KNOWLEDGE_TYPE_QA, KnowledgeBaseClient, RetrievalHit
from app.services.prompting import (
    DEFAULT_POLISH_PROMPT,
    DEFAULT_REWRITE_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
    build_answer_messages,
    build_polish_messages,
    build_rewrite_messages,
)
from app.services.repository import (
    PromptRepository,
    TermMappingRepository,
    get_prompt_repository,
    get_term_mapping_repository,
)
from app.services.rewrite_agent import QueryRewriteAgent, get_query_rewrite_agent
from app.services.sse import sse
from app.services.stream_agent import ChatStreamAgent, get_chat_stream_agent
from app.services.text_normalizer import normalize_query

logger = logging.getLogger(__name__)

# 最终答案按该长度切块下发，前端保持逐字输出观感
TOKEN_CHUNK_SIZE = 12


def chunk_text(text: str, size: int = TOKEN_CHUNK_SIZE) -> Iterator[str]:
    """把完整答案切块逐段下发。"""
    return (text[index : index + size] for index in range(0, len(text), size))


class RagAgent:
    def __init__(
        self,
        settings: Settings,
        term_repository: TermMappingRepository | None = None,
        prompt_repository: PromptRepository | None = None,
        rewrite_agent: QueryRewriteAgent | None = None,
        stream_agent: ChatStreamAgent | None = None,
    ) -> None:
        self.settings = settings
        self._terms = term_repository or get_term_mapping_repository()
        self._prompts = prompt_repository or get_prompt_repository()
        # 构造期不建任何模型 agent——FAQ 首轮直返走不到改写、改写失败的请求也用不到
        # 流式生成，各自首次需要时才创建（见下面两个 _ensure_*），None 表示尚未创建
        self._rewriter = rewrite_agent
        self._streamer = stream_agent

    def _ensure_rewriter(self) -> QueryRewriteAgent:
        """取改写 agent，首次调用时才创建（构造期建它属于白付初始化代价）。"""
        if self._rewriter is None:
            self._rewriter = get_query_rewrite_agent()
        return self._rewriter

    def _ensure_streamer(self) -> ChatStreamAgent:
        """取流式问答 agent（FAQ 直返润色与最终答案生成共用），首次调用时才创建。"""
        if self._streamer is None:
            self._streamer = get_chat_stream_agent()
        return self._streamer

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        query = request.messages[-1].content
        history = [{"role": item.role, "content": item.content} for item in request.messages]
        # 历史轮次不含当前问题，当前问题由消息装配统一放在最后一条 user 消息
        previous_turns = history[:-1]

        normalized = normalize_query(query)
        # 库中关键词（术语）替换；库不可用时映射表为空，替换结果与原问题相同
        replace_keyword_query = (await self._terms.mapper()).apply(normalized)
        yield sse("status", {"stage": "normalized", "query": replace_keyword_query})

        # 三条提示词各自在**用到的那条分支**里现取，不提前一次性取好：
        # FAQ 首轮直返走不到改写提示词，生成失败的请求也没用到润色提示词。
        # 两个易错点：取库是异步的，必须 await；也不要把结果写成带尾逗号的元组。
        polish_prompt = await self._prompts.get(self.settings.prompt_key_answer_polish)
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            kb = KnowledgeBaseClient(self.settings, client)
            # 原问题与关键词替换后的问题各探一次 FAQ，高置信命中直接返回库中标准答案。
            direct = await self._faq_probe(kb, query, replace_keyword_query, request.ranges)
            if direct:
                yield sse("sources", self._sources([direct]))
                async for event in self._stream_faq_answer(polish_prompt, query, direct):
                    yield event
                yield sse("done", {"answer_type": "faq", "query": replace_keyword_query})
                return

            yield sse("status", {"stage": "rewrite", "message": "正在进行问题改写"})
            understanding_prompt = await self._prompts.get(self.settings.prompt_key_query_understanding)
            rewrite_messages = build_rewrite_messages(
                understanding_prompt or DEFAULT_REWRITE_PROMPT,
                replace_keyword_query,
                previous_turns,
            )
            # create_agent（OpenAI 兼容协议）非流式一次调用，同时产出改写与关键词，
            # 替代原来单独的「标签提取」模型调用；失败时回落原始 query 检索
            result = await self._ensure_rewriter().rewrite(rewrite_messages)
            if result.keywords:
                yield sse("status", {"stage": "tags_extracted", "tags": result.keywords})
            rewritten = result.rewritten_query

            if rewritten:
                yield sse("status", {"stage": "rewritten", "query": rewritten})
                # 改写后 FAQ 再探一次：命中即可省掉跨库检索与生成
                direct = await self._faq_probe(kb, query, rewritten, request.ranges)
                if direct:
                    yield sse("sources", self._sources([direct]))
                    async for event in self._stream_faq_answer(polish_prompt, query, direct):
                        yield event
                    yield sse("done", {"answer_type": "faq_rewrite", "query": rewritten})
                    return
            else:
                yield sse("status", {"stage": "rewrite_failed", "message": "改写失败，使用原问题检索"})

            yield sse("status", {"stage": "hybrid_retrieval", "message": "正在进行多路混合检索"})
            # 跨库混合检索：改写后的 query 负责语义召回，改写同一次调用产出的 keywords
            # 负责业务词命中；一次请求覆盖 ranges 内全部知识库，结果已由平台重排。
            hits = await kb.mix_retrieve(
                rewritten or replace_keyword_query, result.keywords, request.ranges
            )
            sources = hits[: self.settings.final_context_top_k]
            if not sources:
                yield sse("token", {"content": "未在已授权知识库中检索到可用于回答的内容。"})
                yield sse("done", {"answer_type": "no_context"})
                return
            yield sse("sources", self._sources(sources))
            context = "\n\n".join(f"[资料{index + 1}] {hit.content}" for index, hit in enumerate(sources))

            # 最终答案一步到位：提示词取 answer_summary（库里没有则回落内置），
            # 检索到的资料以 {ragkm} 注入该提示词的 user 段。单轮调用，不带对话历史；
            # 生成即最终答案，不再润色；走 langchain 的流式 agent，token 直接下发。
            yield sse("status", {"stage": "answer_generate", "message": "正在生成答案"})
            summary_prompt = await self._prompts.get(self.settings.prompt_key_answer_summary)
            streamed = False
            try:
                async for token in self._ensure_streamer().stream(
                    build_answer_messages(
                        summary_prompt or DEFAULT_SUMMARY_PROMPT,
                        query,
                        context=context,
                    )
                ):
                    streamed = True
                    yield sse("token", {"content": token})
            except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as exc:
                logger.warning("最终答案生成失败: %s", exc)
            if not streamed:
                yield sse("token", {"content": "未能生成答案，请稍后重试。"})
            yield sse("done", {"answer_type": "rag", "query": rewritten or replace_keyword_query})

    async def _stream_faq_answer(
        self, prompt: Prompt | None, query: str, hit: RetrievalHit
    ) -> AsyncIterator[str]:
        """FAQ 直返的答案润色（create_agent 流式下发）。

        命中 FAQ 说明库里已有人工维护的标准答案，润色只做表达层整理，没有检索知识，
        也不带对话历史，因此只装配 `answer_polish` 的 system 段与含 `{query}` + `{answer}`
        的 user 段。
        润色是可选步骤：调用失败（含未配置回答模型）时原样回落库中答案，
        直返链路不会因为润色而整体失败。
        """
        yield sse("status", {"stage": "answer_polish", "message": "正在润色答案"})
        messages = build_polish_messages(
            prompt or DEFAULT_POLISH_PROMPT, query, hit.content
        )
        streamed = False
        try:
            async for token in self._ensure_streamer().stream(messages):
                streamed = True
                yield sse("token", {"content": token})
        except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as exc:
            logger.warning("FAQ 答案润色失败，回落库中原文: %s", exc)
        # 已经吐过内容就不重复下发，否则把库中原文切块补上
        if not streamed:
            for piece in chunk_text(hit.content):
                yield sse("token", {"content": piece})

    async def _faq_probe(
        self, kb: KnowledgeBaseClient, first: str, second: str, ranges: list[KnowledgeRange]
    ) -> RetrievalHit | None:
        """两个 query 变体各召回一次标准问答，返回高置信命中（无命中则为 None）。

        **只探门户显式标注的标准问答库（`knowledgeType == 2`）**：FAQ 直返靠的是库里
        人工维护的 QA 对，切片库（1）里没有这种东西，拿它去探只会白跑一次跨库检索。
        未传 `knowledge_type` 的 range **不视为 QA 库**，「没标类型」和「标了 QA」是两回事 ——
        按未知保留会让切片库也被拿去当 FAQ 探，等于凭空多一次跨库检索而永远不可能命中。
        召回本身是跨库检索，由 `KnowledgeBaseClient` 再按 (project_id, tenantId) 分组发请求。
        """
        candidates = [item for item in ranges if item.knowledge_type == KNOWLEDGE_TYPE_QA]
        if not candidates:
            return None
        first_hits, second_hits = await asyncio.gather(
            kb.faq_retrieve(first, candidates), kb.faq_retrieve(second, candidates)
        )
        return self._faq_match(self._dedupe(first_hits + second_hits))

    def _faq_match(self, hits: list[RetrievalHit]) -> RetrievalHit | None:
        """取最高分命中；未达阈值时不直返，把最高分打进日志便于回调阈值。

        阈值 `FAQ_SIMILARITY_THRESHOLD` 原本是按**单库相似度**调的，现在 FAQ 也走跨库
        重排（`weight_type=1` 的动态权重 WRRF），分数量纲换了 —— 不记这一行的话，
        阈值不合适会表现为「FAQ 永远不直返」且没有任何线索。
        """
        best = next((item for item in hits if item.content), None)
        if best is None or best.score >= self.settings.faq_similarity_threshold:
            return best
        logger.info(
            "FAQ 最高分 %.4f 未达阈值 %.4f，本次不直返", best.score, self.settings.faq_similarity_threshold
        )
        return None

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
    def _sources(hits: list[RetrievalHit]) -> dict:
        return {"items": [
            {
                "knowledge_base_id": item.knowledge_base_id,
                "doc_id": item.doc_id,
                "doc_name": item.doc_name,
                "chunk_id": item.chunk_id,
                "score": item.score,
            }
            for item in hits
        ]}
