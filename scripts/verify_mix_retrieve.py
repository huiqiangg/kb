"""跨库检索 + 单步答案生成链路回归。

用 httpx.MockTransport 在进程内拦掉全部出网请求。FAQ 探测与最终答案检索走的是同一个
接口（`kbs:mix-retrieve`，请求体与响应结构一致）但**地址各自可覆盖**，脚本按
「请求体里的 keywords 是否为空」区分二者：空 = FAQ 探测，非空 = 最终答案检索。
覆盖十三个场景：

1. 正常链路：ranges 不带空间信息 -> 回落到全局配置；请求地址/查询参数/请求体形状、
   result[] 解析、SSE 事件顺序、生成阶段按「system 固定指令 + user 含 {query}/{ragkm}」装配消息；
2. 生成提示词取 `answer_summary`（并确认不再请求已删除的旧 key）；
3. FAQ 高置信直返：命中即秒回，不再做最终跨库检索，也不调改写模型；
   FAQ 答案优先取 `chunk.qa_pairs[].answer`；
4. FAQ 分数未达阈值 -> 不直返，继续走完整链路；
5. **range 自带 project_id/tenantId/knowledgeType=2**：FAQ 探测用 range 自带的空间，
   最终跨库检索仍用全局配置（本轮只改了 `_faq_probe`）；
6. **只有 knowledgeType=1 的切片库**：一次 FAQ 都不探，直接进改写与检索；
7. **range 没带 knowledge_type**：同样一次 FAQ 都不探 —— 未标类型不等于 QA 库；
8. **两个 QA 库分属两个空间**：按空间分组，每个 query 变体各发 2 次请求，组内不混库；
9. 混合检索失败（5xx）：降级为「未检索到内容」，不抛异常；
10. 生成模型失败：回落一句提示语，done 事件照常发出；
11. 未配置 KB_TENANT_ID：请求不带 tenantId 参数；
12. 整条覆盖 KB_MIX_RETRIEVE_URL：自定义地址生效；
13. **FAQ 地址与最终检索地址各自覆盖**：FAQ 打 FAQ 地址、最终检索打 mix 地址，互不影响；
    未单独配 FAQ 地址时复用 mix 地址（正常链路场景里校验）。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_mix_retrieve.py
"""

import asyncio
import json
import sys
from typing import Any

import httpx

import app.services.rag_agent as rag_module
from app.core.config import Settings
from app.models.chat import ChatCompletionRequest
from app.models.prompt import Prompt
from app.services.rag_agent import RagAgent
from app.services.rewrite_agent import RewriteResult

KB_ID = "37c5dwtg4ufbw332"
KB_ID_2 = "98ab12cd34ef567890abcdef"
DOC_ID = "ds3523"
TENANT = "tenant-001"
SPACE_A = "space-a"
SPACE_B = "space-b"
RAW_QUERY = "房贷能提前还还款吗"
REWRITTEN = "个人住房贷款可以提前还款吗"
KEYWORDS = ["房贷", "提前还款"]
CHUNK_TEXT = "借款人可申请提前归还个人住房贷款。"
FAQ_ANSWER = "可通过手机银行或经办行柜面办理提前还款。"
MIX_RESULT = [
    {
        "chunk": {"id": "chunk-1", "content": CHUNK_TEXT},
        "score": 0.82,
        "knowledge_base_id": KB_ID,
        "doc_id": DOC_ID,
        "doc_name": "个人贷款管理办法.pdf",
    },
    {
        "chunk": {"id": "chunk-2", "content": "提前还款需提前 30 天向经办行提出申请。"},
        "score": 0.61,
        "knowledge_base_id": KB_ID,
        "doc_id": DOC_ID,
        "doc_name": "个人贷款管理办法.pdf",
    },
]
ANSWER_TOKENS = ("提前还款", "可向经办行申请。")

FAILURES: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        FAILURES.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")


class _Mapper:
    def apply(self, text: str) -> str:
        return text


class _TermRepository:
    async def mapper(self) -> _Mapper:
        return _Mapper()


class _PromptRepository:
    """替掉提示词仓库：记录被请求的 key，返回注入的 Prompt（也可以是 None 走内置兜底）。"""

    def __init__(
        self, captured: dict[str, Any], prompts: dict[str, Prompt] | None = None
    ) -> None:
        self.captured = captured
        self.prompts = prompts or {}

    async def get(self, key: str | None) -> Prompt | None:
        self.captured.setdefault("prompt_keys", []).append(key)
        return self.prompts.get(key or "")


class _Rewriter:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.captured = captured

    async def rewrite(self, messages: list[dict[str, str]]) -> RewriteResult:
        self.captured["rewrite_calls"] = self.captured.get("rewrite_calls", 0) + 1
        return RewriteResult(rewritten_query=REWRITTEN, keywords=list(KEYWORDS))


class _Streamer:
    """替掉流式问答 agent（FAQ 润色与最终答案生成共用）：只记录 messages 并吐固定 token。"""

    def __init__(self, captured: dict[str, Any], error: Exception | None = None) -> None:
        self.captured = captured
        self.error = error

    async def stream(self, messages: list[dict[str, str]]):
        self.captured.setdefault("answer_messages", []).append(messages)
        if self.error is not None:
            raise self.error
        for token in ANSWER_TOKENS:
            yield token


def qa_result(score: float) -> list[dict[str, Any]]:
    """标准问答库的召回结果：正文放在 qa_pairs 里，chunk.content 只是陪衬。"""
    return [
        {
            "chunk": {
                "id": "qa-1",
                "content": "（切片正文，命中 qa_pairs 时不该被使用）",
                "qa_pairs": [{"question": "个人住房贷款怎么提前还款", "answer": FAQ_ANSWER}],
            },
            "score": score,
            "knowledge_base_id": KB_ID,
        }
    ]


def make_handler(captured: dict[str, Any], *, qa_score: float, mix_status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("kbs:mix-retrieve"):
            raise AssertionError(f"不该出现的请求：{request.url}")
        params = dict(request.url.params)
        body = json.loads(request.content)
        url = str(request.url.copy_with(query=None))
        # 带 keywords 的是最终答案的跨库检索；不带的是 FAQ 探测（改写前/后各一次）
        if body.get("keywords"):
            captured.setdefault("mix_calls", []).append(
                {"url": url, "params": params, "body": body}
            )
            captured["mix_url"] = url
            captured["mix_params"] = params
            captured["mix_body"] = body
            if mix_status != 200:
                return httpx.Response(mix_status, json={"message": "boom"})
            return httpx.Response(200, json={"result": MIX_RESULT})
        captured.setdefault("faq_calls", []).append({"url": url, "params": params, "body": body})
        return httpx.Response(200, json={"result": qa_result(qa_score)})

    return handler


def parse(events: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for raw in events:
        head, _, data = raw.partition("\n")
        parsed.append(
            (
                head.removeprefix("event: ").strip(),
                json.loads(data.removeprefix("data: ").strip()),
            )
        )
    return parsed


async def run_case(
    label: str,
    *,
    qa_score: float = 0.42,
    mix_status: int = 200,
    tenant_id: str = TENANT,
    ranges: list[dict[str, Any]] | None = None,
    prompts: dict[str, Prompt] | None = None,
    stream_error: Exception | None = None,
    **overrides: Any,
) -> tuple[list[tuple[str, dict]], dict[str, Any], Settings]:
    captured: dict[str, Any] = {}
    settings = Settings(
        kb_tenant_id=tenant_id,
        answer_model_url="http://model.test/answer/chat/completions",
        **overrides,
    )
    agent = RagAgent(
        settings,
        term_repository=_TermRepository(),
        prompt_repository=_PromptRepository(captured, prompts),
        rewrite_agent=_Rewriter(captured),
        stream_agent=_Streamer(captured, stream_error),
    )
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": RAW_QUERY}],
            # 默认 ranges 标成标准问答库（knowledgeType=2），否则连 FAQ 探测都不会发生；
            # 空间字段仍留空，用来验证回落全局 KB_PROJECT_ID / KB_TENANT_ID。
            "ranges": ranges
            or [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID], "knowledgeType": 2}],
        }
    )
    real_client = rag_module.httpx.AsyncClient
    rag_module.httpx.AsyncClient = lambda **kwargs: real_client(
        transport=httpx.MockTransport(
            make_handler(captured, qa_score=qa_score, mix_status=mix_status)
        ),
        **kwargs,
    )
    try:
        events = [chunk async for chunk in agent.stream(request)]
    finally:
        rag_module.httpx.AsyncClient = real_client
    print(f"--- 场景：{label}")
    return parse(events), captured, settings


def names_of(parsed: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in parsed]


def last_user_content(captured: dict[str, Any]) -> str:
    """最后一次生成调用里的 user 消息内容（含问题与参考上下文）。"""
    messages = captured["answer_messages"][-1]
    return next(item["content"] for item in reversed(messages) if item["role"] == "user")


def messages_text(captured: dict[str, Any]) -> str:
    """最后一次生成调用里全部消息拼起来的文本。"""
    messages = captured["answer_messages"][-1]
    return "\n".join(item["content"] for item in messages)


def roles_of_answer_call(captured: dict[str, Any]) -> list[str]:
    messages = captured["answer_messages"][-1]
    return [item["role"] for item in messages]


def system_content(captured: dict[str, Any]) -> str | None:
    messages = captured["answer_messages"][-1]
    return next((item["content"] for item in messages if item["role"] == "system"), None)


def scope_of(call: dict[str, Any]) -> str:
    """把一次请求的查询参数压成可比较的字符串。"""
    return json.dumps(call["params"], sort_keys=True)


async def case_normal() -> None:
    parsed, captured, settings = await run_case("正常链路")
    names = names_of(parsed)

    check(
        "召回地址形状",
        settings.mix_retrieve_url,
        f"{settings.kb_platform_base_url.rstrip('/')}/applet/api/v1/knowlhub/kbs:mix-retrieve",
    )
    check("最终检索地址", captured.get("mix_url"), settings.mix_retrieve_url)
    check(
        "查询参数",
        captured.get("mix_params"),
        {"project_id": settings.kb_project_id, "tenantId": TENANT},
    )
    body = captured.get("mix_body") or {}
    check("body.query", body.get("query"), REWRITTEN)
    check("body.keywords", body.get("keywords"), KEYWORDS)
    check(
        "body.ranges",
        body.get("ranges"),
        [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID]}],
    )
    check("body.disable_rerank", body.get("disable_rerank"), False)
    check("body.rerank_params.top_k", (body.get("rerank_params") or {}).get("top_k"), settings.retrieval_top_k)
    check("body.rerank_params.weight_type", (body.get("rerank_params") or {}).get("weight_type"), 1)
    check(
        "空间信息只走查询参数，不进 range",
        [key for key in ("project_id", "tenantId", "knowledgeType", "knowledge_type") if key in body["ranges"][0]],
        [],
    )

    # FAQ 探测：两轮（改写前 / 改写后），每轮并发两个 query 变体
    # （原问题与术语映射后的问题，本脚本映射是恒等函数故两者相同）-> 共 4 次请求。
    # ranges 带了 knowledgeType=2（否则根本不探），但未带空间信息 -> 回落全局配置。
    faq_calls = captured.get("faq_calls") or []
    check("FAQ 探测请求数", len(faq_calls), 4)
    check(
        "FAQ 探测过的 query",
        sorted({call["body"]["query"] for call in faq_calls}),
        sorted({RAW_QUERY, REWRITTEN}),
    )
    check(
        "FAQ 探测都回落全局空间",
        {scope_of(call) for call in faq_calls},
        {json.dumps({"project_id": settings.kb_project_id, "tenantId": TENANT}, sort_keys=True)},
    )
    check("FAQ 探测不带 keywords", {tuple(call["body"]["keywords"]) for call in faq_calls}, {()})
    check(
        "FAQ 探测也只是跨库检索的 body",
        all("disable_rerank" in call["body"] and "rerank_params" in call["body"] for call in faq_calls),
        True,
    )
    # 未单独配 KB_FAQ_RETRIEVE_URL -> FAQ 地址复用最终检索地址（两通道同端点）
    check("FAQ 地址复用 mix 地址", settings.faq_retrieve_url, settings.mix_retrieve_url)
    check(
        "FAQ 探测打的也是该地址",
        {call["url"] for call in faq_calls},
        {settings.mix_retrieve_url},
    )

    sources = next((data for name, data in parsed if name == "sources"), None)
    if not sources:
        FAILURES.append("正常链路 sources: 未收到 sources 事件")
    else:
        check("sources 条数", len(sources["items"]), 2)
        first = sources["items"][0]
        check("sources[0].chunk_id", first.get("chunk_id"), "chunk-1")
        check("sources[0].doc_id", first.get("doc_id"), DOC_ID)
        check("sources[0].doc_name", first.get("doc_name"), "个人贷款管理办法.pdf")
        check("sources[0].score", first.get("score"), 0.82)

    check(
        "事件顺序",
        names,
        ["status"] * 5 + ["sources", "status"] + ["token", "token", "done"],
    )
    stages = [data.get("stage") for _, data in parsed if data.get("stage")]
    check(
        "阶段序列",
        stages,
        ["normalized", "rewrite", "tags_extracted", "rewritten", "hybrid_retrieval", "answer_generate"],
    )
    check("done.answer_type", parsed[-1][1].get("answer_type"), "rag")
    check("done.query", parsed[-1][1].get("query"), REWRITTEN)

    check("生成阶段消息形状", roles_of_answer_call(captured), ["system", "user"])
    check("内置兜底 system 非空", bool((system_content(captured) or "").strip()), True)
    check("内置兜底注入资料", CHUNK_TEXT in messages_text(captured), True)
    check("内置兜底问题只出现一次", messages_text(captured).count(RAW_QUERY), 1)
    check(
        "token 内容",
        [data["content"] for name, data in parsed if name == "token"],
        list(ANSWER_TOKENS),
    )
    print(f"    最终检索请求体={json.dumps(body, ensure_ascii=False)}")
    print(f"    FAQ 探测参数={[call['params'] for call in faq_calls]}")
    print(f"    事件序列={names}")
    print(f"    取用提示词 key={captured.get('prompt_keys')}")


async def case_prompt_key() -> None:
    parsed, captured, _ = await run_case(
        "生成提示词取 answer_summary",
        prompts={
            "answer_summary": Prompt(
                system="你是银行客服助手。仅依据参考上下文回答。",
                user="检索资料如下：\n{ragkm}\n请依据资料作答。",
            ),
            "answer_polish": Prompt(system="你是润色助手。", user="润色：{answer}"),
        },
    )
    check(
        "提示词请求 key 集合",
        sorted(captured.get("prompt_keys") or []),
        ["answer_polish", "answer_summary", "query_understanding"],
    )
    check("生成阶段消息形状", roles_of_answer_call(captured), ["system", "user"])
    check("system 取库里的固定指令", system_content(captured), "你是银行客服助手。仅依据参考上下文回答。")
    check("system 里没有占位符残留", "{" in (system_content(captured) or ""), False)
    check("user 注入检索资料", CHUNK_TEXT in last_user_content(captured), True)
    check("user 取库里的正文", last_user_content(captured).startswith("检索资料如下："), True)
    check(
        "生成阶段无润色事件",
        [d.get("stage") for _, d in parsed if d.get("stage", "").startswith("answer_")],
        ["answer_generate"],
    )


async def case_faq_direct() -> None:
    parsed, captured, _ = await run_case("FAQ 高置信直返", qa_score=0.99)
    check("FAQ 直返 answer_type", parsed[-1][1].get("answer_type"), "faq")
    check("FAQ 直返未做最终跨库检索", "mix_body" in captured, False)
    check("FAQ 直返只探一轮", len(captured.get("faq_calls") or []), 2)
    check("FAQ 直返未调改写模型", captured.get("rewrite_calls", 0), 0)
    # 命中标准问答库时答案取 chunk.qa_pairs[].answer，而不是 chunk.content
    check("FAQ 答案取 qa_pairs", FAQ_ANSWER in messages_text(captured), True)
    check("未误用 chunk.content", "（切片正文" in messages_text(captured), False)
    check(
        "FAQ 直返有答案输出",
        [data["content"] for name, data in parsed if name == "token"],
        list(ANSWER_TOKENS),
    )
    print(f"    事件序列={names_of(parsed)}")


async def case_faq_threshold() -> None:
    """FAQ 分数没过阈值：不直返，走完整链路（RagAgent 会把最高分打进日志便于回调阈值）。"""
    threshold = Settings().faq_similarity_threshold
    parsed, _, _ = await run_case("FAQ 分数未达阈值", qa_score=threshold - 0.01)
    check("未达阈值不直返", parsed[-1][1].get("answer_type"), "rag")


async def case_range_carries_space() -> None:
    """range 自带空间与类型：FAQ 探测用 range 的空间，最终跨库检索仍用全局配置。"""
    ranges = [
        {
            "knowledge_base_id": KB_ID,
            "doc_range": [DOC_ID],
            "project_id": SPACE_A,
            "tenantId": "tenant-9",
            "knowledgeType": 2,
        }
    ]
    _, captured, settings = await run_case("QA 库自带空间", ranges=ranges)

    check(
        "FAQ 探测用 range 自带空间",
        {scope_of(call) for call in captured["faq_calls"]},
        {json.dumps({"project_id": SPACE_A, "tenantId": "tenant-9"}, sort_keys=True)},
    )
    check(
        "最终跨库检索仍用全局空间",
        captured.get("mix_params"),
        {"project_id": settings.kb_project_id, "tenantId": TENANT},
    )
    print(f"    FAQ 参数={[call['params'] for call in captured['faq_calls']]}")
    print(f"    最终检索参数={captured.get('mix_params')}")


async def case_slice_only() -> None:
    """只有切片库（knowledgeType=1）：一次 FAQ 都不探，直接进改写与检索。"""
    ranges = [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID], "knowledge_type": 1}]
    parsed, captured, _ = await run_case("只有切片库时不探 FAQ", ranges=ranges)

    check("切片库不探 FAQ", len(captured.get("faq_calls") or []), 0)
    check("仍走最终跨库检索", "mix_body" in captured, True)
    check("answer_type", parsed[-1][1].get("answer_type"), "rag")
    check("未触发 rewrite_failed", "rewrite_failed" in [d.get("stage") for _, d in parsed], False)
    print(f"    事件序列={names_of(parsed)}")


async def case_missing_type() -> None:
    """range 没带 knowledge_type：同样不探 FAQ —— 未标类型不等于标准问答库。

    如果这里放宽成「未知按候选保留」，切片库会被拿去当 FAQ 探，白跑一次跨库检索
    而永远不可能命中；所以两种写法（驼峰/下划线）都不带时必须一次都不探。
    """
    ranges = [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID]}]
    parsed, captured, _ = await run_case("未标 knowledge_type 不探 FAQ", ranges=ranges)

    check("未标类型不探 FAQ", len(captured.get("faq_calls") or []), 0)
    check("仍走最终跨库检索", "mix_body" in captured, True)
    check("answer_type", parsed[-1][1].get("answer_type"), "rag")
    print(f"    事件序列={names_of(parsed)}")


async def case_multi_space() -> None:
    """两个 QA 库分属两个空间：按空间分组，每组各发一次，组内不混库。"""
    ranges = [
        {
            "knowledge_base_id": KB_ID,
            "doc_range": [DOC_ID],
            "project_id": SPACE_A,
            "tenantId": TENANT,
            "knowledgeType": 2,
        },
        {
            "knowledge_base_id": KB_ID_2,
            "project_id": SPACE_B,
            "tenantId": TENANT,
            "knowledgeType": 2,
        },
    ]
    _, captured, _ = await run_case("跨空间按空间分组", ranges=ranges)

    calls = captured.get("faq_calls") or []
    # 2 个空间 × 2 个 query 变体 × 2 轮探测
    check("两空间 × 两变体 × 两轮", len(calls), 8)
    check(
        "每个空间各自覆盖两个 query 变体",
        {(call["params"]["project_id"], call["body"]["query"]) for call in calls},
        {(SPACE_A, RAW_QUERY), (SPACE_B, RAW_QUERY), (SPACE_A, REWRITTEN), (SPACE_B, REWRITTEN)},
    )
    by_space: dict[str, set[str]] = {}
    for call in calls:
        by_space.setdefault(call["params"]["project_id"], set()).update(
            item["knowledge_base_id"] for item in call["body"]["ranges"]
        )
    check("空间 -> 知识库 归属", by_space, {SPACE_A: {KB_ID}, SPACE_B: {KB_ID_2}})
    print(f"    FAQ 参数={[call['params'] for call in calls]}")
    print(f"    FAQ 分组={ {key: sorted(value) for key, value in by_space.items()} }")


async def case_mix_down() -> None:
    parsed, captured, _ = await run_case("混合检索 500", mix_status=500)
    check("检索失败仍调用混合检索", "mix_body" in captured, True)
    check("检索失败降级 answer_type", parsed[-1][1].get("answer_type"), "no_context")
    check("检索失败无 sources", names_of(parsed).count("sources"), 0)
    token = next(data["content"] for name, data in parsed if name == "token")
    check("检索失败提示语", token, "未在已授权知识库中检索到可用于回答的内容。")
    check("检索失败未调生成模型", "answer_messages" in captured, False)
    print(f"    事件序列={names_of(parsed)}")


async def case_generate_down() -> None:
    parsed, captured, _ = await run_case(
        "生成模型失败", stream_error=RuntimeError("未配置 ANSWER_MODEL_URL")
    )
    check("生成失败仍发出 done", parsed[-1][0], "done")
    check("生成失败 answer_type", parsed[-1][1].get("answer_type"), "rag")
    token = next(data["content"] for name, data in parsed if name == "token")
    check("生成失败回落提示语", token, "未能生成答案，请稍后重试。")
    print(f"    事件序列={names_of(parsed)}")


async def case_no_tenant() -> None:
    _, captured, settings = await run_case("未配置租户 id", tenant_id="")
    check("无 tenantId 时查询参数", captured.get("mix_params"), {"project_id": settings.kb_project_id})
    check(
        "FAQ 探测同样不带 tenantId",
        {scope_of(call) for call in captured["faq_calls"]},
        {json.dumps({"project_id": settings.kb_project_id}, sort_keys=True)},
    )


async def case_custom_url() -> None:
    custom = (
        "https://10.0.0.9:30443/llm/llmops/tenants/t1/gateway"
        "/applet/api/v1/knowlhub/kbs:mix-retrieve"
    )
    _, captured, settings = await run_case("整条覆盖接口地址", kb_mix_retrieve_url=custom)
    check("自定义地址生效", captured.get("mix_url"), custom)
    check(
        "自定义地址仍拼 project_id",
        (captured.get("mix_params") or {}).get("project_id"),
        settings.kb_project_id,
    )


async def case_custom_faq_url() -> None:
    """FAQ 地址与最终检索地址各自独立覆盖，互不影响。

    两者是同一个接口的两处部署（请求体与响应结构一致），FAQ 库可能不挂在同一个网关上。
    这里让 FAQ 不直返（分数不够），好让两条通道都真的发出去，一次校验两个地址。
    """
    faq_url = (
        "https://10.0.0.7:30443/llm/llmops/tenants/t1/faq-hub"
        "/applet/api/v1/knowlhub/kbs:mix-retrieve"
    )
    mix_url = (
        "https://10.0.0.8:30443/llm/llmops/tenants/t1/gateway"
        "/applet/api/v1/knowlhub/kbs:mix-retrieve"
    )
    parsed, captured, settings = await run_case(
        "FAQ 与检索地址各自覆盖",
        kb_faq_retrieve_url=faq_url,
        kb_mix_retrieve_url=mix_url,
    )

    check("FAQ 地址覆盖生效", settings.faq_retrieve_url, faq_url)
    check("最终检索地址覆盖生效", settings.mix_retrieve_url, mix_url)
    check("FAQ 探测只打 FAQ 地址", {call["url"] for call in captured["faq_calls"]}, {faq_url})
    check("最终检索只打 mix 地址", captured.get("mix_url"), mix_url)
    check("answer_type", parsed[-1][1].get("answer_type"), "rag")
    print(f"    FAQ 地址={ {call['url'] for call in captured['faq_calls']} }")
    print(f"    最终检索地址={captured.get('mix_url')}")


async def main() -> int:
    await case_normal()
    await case_prompt_key()
    await case_faq_direct()
    await case_faq_threshold()
    await case_range_carries_space()
    await case_slice_only()
    await case_missing_type()
    await case_multi_space()
    await case_mix_down()
    await case_generate_down()
    await case_no_tenant()
    await case_custom_url()
    await case_custom_faq_url()

    if FAILURES:
        print("\nFAIL")
        for item in FAILURES:
            print(" -", item)
        return 1
    print("\nOK  跨库检索 + 单步生成链路十三组场景全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
