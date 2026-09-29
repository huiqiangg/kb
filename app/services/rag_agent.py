import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass, field

import httpx
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.logging import elapsed_ms
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
from app.services.rewrite import RewriteResult, parse_rewrite_json
from app.services.sse import sse
from app.services.text_normalizer import normalize_query

logger = logging.getLogger(__name__)

# 最终答案按该长度切块下发，前端保持逐字输出观感
TOKEN_CHUNK_SIZE = 12

# FAQ 直返后附带「其他相近问题」的条数：**固定 3 条，不够就一条都不给**。
# 需求就是「凑不满三条不返回」，所以不是可调上限，不做成配置项。
FAQ_RELATED_QUERY_COUNT = 3


def chunk_text(text: str) -> Iterator[str]:
    """把完整答案切块逐段下发。"""
    return (
        text[index : index + TOKEN_CHUNK_SIZE]
        for index in range(0, len(text), TOKEN_CHUNK_SIZE)
    )


@dataclass(slots=True)
class FaqProbeResult:
    """FAQ 探测结果：命中的标准答案 + 库里其他相近问题。

    `hit` 为 None 表示未命中（不直返）。`related_queries` **要么是 3 条、要么是空列表** ——
    「凑不满三条就不返回」在构造时就已经收敛好，调用方直接 `if probe.related_queries`
    判断即可，不必再数一遍条数。
    """

    hit: RetrievalHit | None = None
    related_queries: list[str] = field(default_factory=list)


class RagAgent:
    def __init__(
        self,
        settings: Settings,
        term_repository: TermMappingRepository | None = None,
        prompt_repository: PromptRepository | None = None,
        rewrite_model: ChatOpenAI | None = None,
        answer_model: ChatOpenAI | None = None,
    ) -> None:
        self.settings = settings
        # 仓库有模块级默认实例（`get_*_repository()`），传参是为了测试能塞不达库的假仓库
        self._terms = term_repository or get_term_mapping_repository()
        self._prompts = prompt_repository or get_prompt_repository()
        # 构造期不建任何模型客户端——FAQ 首轮直返走不到改写、改写失败的请求也用不到
        # 生成，各自首次需要时才创建（见下面两个 _ensure_*），None 表示尚未创建
        self._rewrite_model = rewrite_model
        self._answer_model = answer_model

    def _ensure_rewrite_model(self) -> ChatOpenAI:
        """取改写模型，首次用到才建（构造期建它属于白付初始化代价）。

        地址取 `REWRITE_MODEL_URL`（去掉 `/chat/completions`：openai 客户端会自己补），
        鉴权用 `REWRITE_MODEL_KEY`：`api_key` 位置与真实鉴权头 `accessKey` 都用它。
        **未配地址时抛 RuntimeError**，理由同 `_ensure_answer_model` —— 空 base_url 会静默
        把问题发到 api.openai.com，而改写失败又只吞成一条 warning，很难发现。
        """
        if self._rewrite_model is None:
            if not self.settings.rewrite_model_url:
                raise RuntimeError("未配置 REWRITE_MODEL_URL")
            access_key = self.settings.rewrite_model_key
            self._rewrite_model = ChatOpenAI(
                model=self.settings.rewrite_model_name,
                base_url=self.settings.rewrite_model_url.removesuffix("/chat/completions"),
                api_key=access_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=1,
                default_headers={"accessKey": access_key} if access_key else None,
            )
        return self._rewrite_model

    def _ensure_answer_model(self) -> ChatOpenAI:
        """取回答模型（FAQ 直返润色与最终答案生成共用），首次用到才建。

        就是 `ChatOpenAI(...)` 本身，不再包一层 agent —— 无工具的 agent 除了多一层
        graph 之外什么都没做，直接用模型的 `astream` 下发增量更直白。
        地址取 `ANSWER_MODEL_URL`（同样去掉 `/chat/completions`），鉴权用 `ANSWER_MODEL_KEY`
        （与改写模型各配各的，互不覆盖）。**未配地址时抛 RuntimeError** —— 否则 base_url
        为空会静默打到 api.openai.com。
        """
        if self._answer_model is None:
            if not self.settings.answer_model_url:
                raise RuntimeError("未配置 ANSWER_MODEL_URL")
            access_key = self.settings.answer_model_key
            self._answer_model = ChatOpenAI(
                model=self.settings.answer_model_name,
                base_url=self.settings.answer_model_url.removesuffix("/chat/completions"),
                api_key=access_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=1,
                default_headers={"accessKey": access_key} if access_key else None,
            )
        return self._answer_model

    async def _rewrite(self, messages: list[dict[str, str]]) -> RewriteResult:
        """改写 + 关键词提取：**非流式**一次调用，输出约定为纯 JSON 文本。

        改写结果必须拿到完整 JSON 才有用，所以不流式；一次调用同时产出改写与关键词，
        替代了原来单独的「标签提取」调用。调用失败或输出不可解析都不抛异常 ——
        返回空结果让调用方用原 query 继续检索，问答链路不中断。
        """
        try:
            message = await self._ensure_rewrite_model().ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 —— 改写是增强环节，失败不该中断问答
            logger.warning("query 改写调用失败: %s", exc)
            return RewriteResult()
        text = message.content if isinstance(message.content, str) else ""
        result = parse_rewrite_json(text)
        if text and not result.rewritten_query:
            logger.warning("query 改写输出无法解析为结构化结果: %.120s", text)
        return result

    async def _stream_tokens(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """把 messages 交给回答模型，逐段产出文本。

        与对外的 `stream(request)` 不是一回事：那个是整条问答链路的 SSE 事件流，
        这里只负责**一次模型调用**。
        调用失败（含未配地址、网关不可达）只记一条 warning 就结束 —— 降级成什么
        留给调用方决定（FAQ 直返回落库中原文，RAG 链路回落一句提示语），不在这里替它做主。
        """
        try:
            async for chunk in self._ensure_answer_model().astream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:  # noqa: BLE001 —— 生成是最后一环，失败不该掀掉整条流
            logger.warning("模型流式生成失败: %s", exc)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """整条问答链路的 SSE 事件流。

        每个关键节点都在工作做完后打一条日志（耗时 + 关键数据），标识由中间件绑定的
        trace_id 统一带上，不必在消息里重复。日志只记录、不影响事件序列。
        """
        query = request.messages[-1].content
        history = [{"role": item.role, "content": item.content} for item in request.messages]
        # 历史轮次不含当前问题，当前问题由消息装配统一放在最后一条 user 消息
        previous_turns = history[:-1]

        started = time.perf_counter()
        normalized = normalize_query(query)
        # 库中关键词（术语）替换；库不可用时映射表为空，替换结果与原问题相同
        replace_keyword_query = (await self._terms.mapper()).apply(normalized)
        logger.info(
            "关键词替换 耗时=%.0fms 替换后=%s",
            elapsed_ms(started),
            replace_keyword_query,
        )
        yield sse("status", {"stage": "normalized", "message": "正在进行关键词替换"})

        # 三条提示词各自在**用到的那条分支**里现取，不提前一次性取好：
        # FAQ 首轮直返走不到改写提示词，生成失败的请求也没用到润色提示词。
        # 两个易错点：取库是异步的，必须 await；也不要把结果写成带尾逗号的元组。
        polish_prompt = await self._prompts.get(self.settings.prompt_key_answer_polish)
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            kb = KnowledgeBaseClient(self.settings, client)
            yield sse("status", {"stage": "faq", "message": "正在进行FAQ检索"})
            # 原问题与关键词替换后的问题各探一次 FAQ，高置信命中直接返回库中标准答案。
            probe = await self._faq_probe(kb, query, replace_keyword_query, request.ranges)
            if probe.hit:
                async for event in self._faq_direct(
                    probe, polish_prompt, query, "faq", replace_keyword_query
                ):
                    yield event
                return

            yield sse("status", {"stage": "rewrite", "message": "正在进行问题改写"})
            understanding_prompt = await self._prompts.get(
                self.settings.prompt_key_query_understanding
            )
            started = time.perf_counter()
            # 改写失败时 result 是空结果，回落术语映射后的 query 检索
            result = await self._rewrite(
                build_rewrite_messages(
                    understanding_prompt or DEFAULT_REWRITE_PROMPT,
                    replace_keyword_query,
                    previous_turns,
                )
            )
            rewritten = result.rewritten_query
            logger.info(
                "问题改写 耗时=%.0fms 改写后=%s 关键词=%s",
                elapsed_ms(started),
                rewritten or "（无，仍用关键词替换后的问题检索）",
                result.keywords,
            )

            if rewritten:
                yield sse("status", {"stage": "faq_second", "query": "二次FAQ检索"})
                # 改写后 FAQ 再探一次：命中即可省掉跨库检索与生成
                probe = await self._faq_probe(kb, query, rewritten, request.ranges)
                if probe.hit:
                    async for event in self._faq_direct(
                        probe, polish_prompt, query, "faq_rewrite", rewritten
                    ):
                        yield event
                    return
            else:
                yield sse("status", {"stage": "rewrite_failed", "message": "改写失败，使用原问题检索"})

            yield sse("status", {"stage": "hybrid_retrieval", "message": "正在进行多路混合检索"})
            # 跨库混合检索：改写后的 query 负责语义召回，改写同一次调用产出的 keywords
            # 负责业务词命中；一次请求覆盖 ranges 内全部知识库，结果已由平台重排。
            started = time.perf_counter()
            hits = await kb.mix_retrieve(
                rewritten or replace_keyword_query, result.keywords, request.ranges
            )
            sources = hits[: self.settings.final_context_top_k]
            # 明细整包 JSON：`sources` 是 `RetrievalHit` 列表，逐条 asdict 后一起序列化，
            # 分数、归属库、文档、切片正文全在 JSON 里，排查「为什么这些资料被采用」够用。
            logger.info(
                "跨库混合检索 耗时=%.0fms 命中=%d 采用=%d 明细=%s",
                elapsed_ms(started),
                len(hits),
                len(sources),
                json.dumps(
                    [asdict(item) for item in sources],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            if not sources:
                yield sse("token", {"content": "未在知识库中检索到可用于回答的内容。"})
                yield sse("done", {})
                return
            yield sse("sources", self._sources(sources))
            context = "\n\n".join(
                f"[资料{index + 1}] {hit.content}" for index, hit in enumerate(sources)
            )

            # 最终答案一步到位：提示词取 answer_summary（库里没有则回落内置），
            # 检索到的资料以 {ragkm} 注入该提示词的 user 段。单轮调用，不带对话历史；
            # 生成即最终答案，不再润色；直接走 ChatOpenAI 的流式调用，token 直接下发。
            yield sse("status", {"stage": "answer_generate", "message": "正在生成答案"})
            summary_prompt = await self._prompts.get(self.settings.prompt_key_answer_summary)
            started = time.perf_counter()
            pieces: list[str] = []
            async for token in self._stream_tokens(
                build_answer_messages(
                    summary_prompt or DEFAULT_SUMMARY_PROMPT,
                    query,
                    context=context,
                )
            ):
                pieces.append(token)
                yield sse("token", {"content": token})
            answer = "".join(pieces)
            if not answer:
                answer = "未能生成答案，请稍后重试。"
                yield sse("token", {"content": answer})
            logger.info(
                "answer_summary 耗时=%.0fms 最终结果=%s",
                elapsed_ms(started),
                answer,
            )
            yield sse("done", {"answer_type": "rag", "query": rewritten or replace_keyword_query})

    async def _faq_direct(
        self,
        probe: FaqProbeResult,
        prompt: Prompt | None,
        query: str,
        answer_type: str,
        done_query: str,
    ) -> AsyncIterator[str]:
        """FAQ 直返的完整事件序列：sources → 润色流式 → 相近问题 → done。

        命中 FAQ 说明库里已有人工维护的标准答案，润色只做表达层整理（没有检索知识，
        也不带对话历史，因此只装配 `answer_polish` 的 `{query}` + `{answer}`）。
        润色是可选步骤：`_stream_tokens` 内部已经把调用失败吞成空流（含未配置回答模型），
        这里只需发现「一个字都没吐」就原样回落库中答案，直返链路不会因为润色而整体失败。

        相近问题在**探测阶段就已经拿到**（`probe.related_queries`），但要等大模型把答案
        输出完才能下发 —— 所以它排在 token 流之后，作为**一条** `related_queries` 事件把
        三条一次带走（前端拿到就能整块渲染引导项，不必逐条拼接）；凑不满 3 条时列表为空，
        一条都不发。两个直返分支（首轮命中 / 改写后命中）只差 `answer_type` 与 `done` 里的
        query，共用这里，免得顺序规则在两处各写一遍、日后漏改一处。
        """
        hit = probe.hit
        if hit is None:  # 调用方已判过，这里只是收窄类型
            return
        yield sse("status", {"stage": "answer_polish", "message": "正在润色答案"})
        messages = build_polish_messages(prompt or DEFAULT_POLISH_PROMPT, query, hit.content)
        started = time.perf_counter()
        pieces: list[str] = []
        async for token in self._stream_tokens(messages):
            pieces.append(token)
            yield sse("token", {"content": token})
        polished = "".join(pieces)
        # 已经吐过内容就不重复下发，否则把库中原文切块补上
        if not polished:
            for piece in chunk_text(hit.content):
                yield sse("token", {"content": piece})
        logger.info(
            "答案润色 耗时=%.0fms 已润色=%s 结果=%s",
            elapsed_ms(started),
            bool(polished),
            polished or hit.content,
        )
        yield sse("end", {})
        if probe.related_queries:
            yield sse("related_queries", {"queries": probe.related_queries})
        yield sse("done", {})

    async def _faq_probe(
        self, kb: KnowledgeBaseClient, first: str, second: str, ranges: list[KnowledgeRange]
    ) -> FaqProbeResult:
        """两个 query 变体各召回一次标准问答，返回高置信命中 + 其他相近问题。

        **只探门户显式标注的标准问答库（`knowledgeType == 2`）**：FAQ 直返靠的是库里
        人工维护的 QA 对，切片库（1）里没有这种东西，拿它去探只会白跑一次跨库检索。
        未传 `knowledge_type` 的 range **不视为 QA 库**，「没标类型」和「标了 QA」是两回事 ——
        按未知保留会让切片库也被拿去当 FAQ 探，等于凭空多一次跨库检索而永远不可能命中。
        召回本身是跨库检索，由 `KnowledgeBaseClient` 再按 (project_id, tenantId) 分组发请求。

        「其他相近问题」直接复用这一次召回的结果（QA 命中自带 `question`），**不额外发请求**：
        排除选中作答的那条、按分数序去重取 `FAQ_RELATED_QUERY_COUNT` 条，凑不满就给空列表。
        未命中（不直返）时不给相近问题 —— 没有可信答案，光推几个问题没有意义。
        """
        started = time.perf_counter()
        candidates = [item for item in ranges if item.knowledge_type == KNOWLEDGE_TYPE_QA]
        hits: list[RetrievalHit] = []
        result = FaqProbeResult()
        if candidates:
            first_hits, second_hits = await asyncio.gather(
                kb.faq_retrieve(first, candidates), kb.faq_retrieve(second, candidates)
            )
            hits = self._dedupe(first_hits + second_hits)
            matched = self._faq_match(hits)
            if matched is not None:
                result = FaqProbeResult(
                    hit=matched, related_queries=self._related_queries(hits, matched)
                )
        # 耗时 + **返回结果整包 JSON**（`FaqProbeResult` 是 dataclass，`asdict` 递归展开，
        # `hit` 与 `related_queries` 全在 JSON 里，不再逐字段摘要）。
        # 两个计数 JSON 里没有、但排查「为什么不直返」时正要看：候选库=0 即 ranges 里
        # 没有 knowledgeType=2 的库（跳过探测），召回=0 即平台一条都没给。
        logger.info(
            "FAQ 检索 耗时=%.0fms 候选库=%d 召回=%d 返回=%s",
            elapsed_ms(started),
            len(candidates),
            len(hits),
            json.dumps(asdict(result), ensure_ascii=False, separators=(",", ":")),
        )
        return result

    @staticmethod
    def _related_queries(hits: list[RetrievalHit], matched: RetrievalHit) -> list[str]:
        """从召回命中里挑出「其他相近问题」，凑不满 `FAQ_RELATED_QUERY_COUNT` 条则返回空列表。

        命中项已按分数降序（`_dedupe` 排过序），这里顺着取就是「除答案外最相近的几条」。
        没有 `question` 的命中（切片库、或平台没给 qa_pairs）直接跳过 —— 宁缺毋滥，
        反正凑不满就不下发。
        """
        seen = {matched.question} if matched.question else set()
        related: list[str] = []
        for item in hits:
            question = (item.question or "").strip()
            if not question or question in seen:
                continue
            seen.add(question)
            related.append(question)
            if len(related) == FAQ_RELATED_QUERY_COUNT:
                return related
        return []

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
            "FAQ 最高分 %.4f 未达阈值 %.4f，本次不直返",
            best.score,
            self.settings.faq_similarity_threshold,
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
        return {
            "items": [
                {
                    "knowledge_base_id": item.knowledge_base_id,
                    "doc_id": item.doc_id,
                    "doc_name": item.doc_name,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                }
                for item in hits
            ]
        }
