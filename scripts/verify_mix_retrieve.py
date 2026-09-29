"""跨库检索 + 单步答案生成链路回归。

用 httpx.MockTransport 在进程内拦掉全部出网请求。FAQ 探测与最终答案检索打的是同一个接口
（`kbs:mix-retrieve`，响应结构一致）但**地址与请求体各自独立**，脚本按「body 里有没有
`disable_rerank`」区分二者：有 = FAQ 探测（平台形状），没有 = 最终答案检索（外部接口的
普通 POST，只有 query/keywords/ranges 三项）。用形态而不是 keywords 判，改写失败
（keywords 恰好为空）时也不会误判。
覆盖十六个场景：

1. 正常链路：ranges 不带空间信息 -> FAQ 那条回落到全局配置；请求地址/请求体形状、
   result[] 解析、SSE 事件顺序、生成阶段按「system 固定指令 + user 含 {query}/{ragkm}」装配消息；
   同时钉住**两条通道的请求形状不同**：最终检索是普通 POST（无查询参数、无请求头、
   body 只有 query/keywords/ranges 三项、ranges 原样含门户字段），FAQ 探测带
   project_id/tenantId 查询参数、body 里还要 disable_rerank/rerank_params；
2. 生成提示词取 `answer_summary`（并确认不再请求已删除的旧 key）；
3. FAQ 高置信直返：命中即秒回，不再做最终跨库检索，也不调改写模型；
   FAQ 答案优先取 `chunk.qa_pairs[].answer`；库里没有其他相近问题时**不发** `related_queries`；
4. **FAQ 直返附带三条相近问题**：token 流结束后、`done` 之前发**一条** `related_queries`
   （`{"queries": [...]}`，库中其他 QA 的问题原文，已排除选中作答的那条；数组顺序即相似度序）；
5. **相近问题去重去空后凑不满三条**：一条都不发（不是发两条）；
6. FAQ 分数未达阈值 -> 不直返，继续走完整链路；
7. **range 自带 project_id/tenantId/knowledgeType=2**：两条通道都**不回落全局配置** ——
   FAQ 拿它做查询参数，最终检索把整条 range（含 knowledgeType）原样放进 body 的 ranges（见 10）；
8. **只有 knowledgeType=1 的切片库**：一次 FAQ 都不探，直接进改写与检索；
9. **range 没带 knowledge_type**：同样一次 FAQ 都不探 —— 未标类型不等于 QA 库；
10. **两个 QA 库分属两个空间**：FAQ 探测按空间分组、每边各发各的、组内不混库；最终检索是
    外部接口的普通 POST，**只发一次**、不分组、不带查询参数，两个库连同各自空间原样进 body；
11. 混合检索失败（5xx）：降级为「未检索到内容」，不抛异常；
12. 生成模型失败：回落一句提示语，done 事件照常发出；
13. **range 未带 tenantId**：FAQ 查询参数里只有 project_id（tenantId 没有全局兜底），
    最终检索也不往 body 塞 tenant（门户没传的字段不会被补成空值）；
14. 整条覆盖 KB_MIX_RETRIEVE_URL：自定义地址生效，且仍不带查询参数；
15. **FAQ 地址与最终检索地址各自覆盖**：FAQ 打 FAQ 地址、最终检索打 mix 地址，互不影响；
    未单独配 FAQ 地址时复用 mix 地址（正常链路场景里校验）；
16. **回答模型用真的 `ChatOpenAI`**（只 mock 它的 HTTP 出口）：验证「直接用 ChatOpenAI」
    这条路真的能流式出 token —— 消息 dict 被正确转换、SSE 被正确解析。其余场景都把模型
    换成假对象，只有这一组守着 langchain 侧的契约。

两条通道的**请求体各写一份**（`mix_retrieve()` 是外部接口、只有三项；`faq_retrieve()` 走
`_faq_body()`，多出 rerank 参数），但响应解析 `_parse()` / `_to_hit()` 共用，脚本两边都断言，
避免以后只改了一边。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_mix_retrieve.py
"""

import asyncio
import json
import sys
import types
from typing import Any

import httpx

import app.services.rag_agent as rag_module
from app.core.config import Settings
from app.models.chat import ChatCompletionRequest
from app.models.prompt import Prompt
from app.services.rag_agent import RagAgent

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
FAQ_QUESTION = "个人住房贷款怎么提前还款"
# 「其他相近问题」用的候选：分数依次低于答案那条，故三条依次被取用
RELATED_QUESTIONS = ["提前还款需要预约吗", "提前还款收违约金吗", "线上能办提前还款吗"]
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


class _RewriteModel:
    """替掉改写模型（`ChatOpenAI`）：记录调用次数，回一段固定的 JSON 文本。

    故意回**原始 JSON 字符串**而不是构造好的结果对象 —— 这样脚本会真跑一遍生产代码里的
    `parse_rewrite_json`，「模型输出纯 JSON 文本 + 容错解析」这条契约不至于没人守。
    """

    def __init__(self, captured: dict[str, Any]) -> None:
        self.captured = captured

    async def ainvoke(self, messages: list[dict[str, str]]) -> "_TextReply":
        self.captured["rewrite_calls"] = self.captured.get("rewrite_calls", 0) + 1
        return _TextReply(
            json.dumps({"rewritten_query": REWRITTEN, "keywords": KEYWORDS}, ensure_ascii=False)
        )


class _TextReply:
    """AIMessage 的最小同形替身：生产代码只读 `.content`。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _AnswerModel:
    """替掉回答模型（FAQ 润色与最终答案生成共用一个 ChatOpenAI）：记录 messages、吐固定 token。

    直接实现 `ChatOpenAI.astream` 那一个方法即可 —— 生产代码已经不再包 agent，
    `RagAgent._stream_tokens()` 拿到的就是模型本身。
    """

    def __init__(self, captured: dict[str, Any], error: Exception | None = None) -> None:
        self.captured = captured
        self.error = error

    async def astream(self, messages: list[dict[str, str]]):
        self.captured.setdefault("answer_messages", []).append(messages)
        if self.error is not None:
            raise self.error
        for token in ANSWER_TOKENS:
            yield _Chunk(token)


class _Chunk:
    """AIMessageChunk 的最小同形替身：生产代码只读 `.content`。"""

    def __init__(self, content: str) -> None:
        self.content = content


def qa_result(score: float, questions: list[str] | None = None) -> list[dict[str, Any]]:
    """标准问答库的召回结果：正文放在 qa_pairs 里，chunk.content 只是陪衬。

    给多条 `questions` 就能造出多条 QA 命中（分数按序递减，首条即最高分），
    用来验证「其他相近问题」的提取与「凑不满三条就不返回」。
    """
    questions = questions or [FAQ_QUESTION]
    return [
        {
            "chunk": {
                "id": f"qa-{index}",
                "content": "（切片正文，命中 qa_pairs 时不该被使用）",
                "qa_pairs": [{"question": question, "answer": FAQ_ANSWER}],
            },
            "score": score - index * 0.01,
            "knowledge_base_id": KB_ID,
        }
        for index, question in enumerate(questions)
    ]


def make_handler(
    captured: dict[str, Any], *, qa_score: float, mix_status: int, qa_questions: list[str] | None
):
    def handler(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("kbs:mix-retrieve"):
            raise AssertionError(f"不该出现的请求：{request.url}")
        params = dict(request.url.params)
        body = json.loads(request.content)
        url = str(request.url.copy_with(query=None))
        headers = dict(request.headers)
        # 带 disable_rerank 的是 FAQ 探测（改写前/后各一次）；只有三项的是最终答案的跨库检索
        if "disable_rerank" not in body:
            captured.setdefault("mix_calls", []).append(
                {"url": url, "params": params, "body": body, "headers": headers}
            )
            captured["mix_url"] = url
            captured["mix_params"] = params
            captured["mix_body"] = body
            if mix_status != 200:
                return httpx.Response(mix_status, json={"message": "boom"})
            return httpx.Response(200, json={"result": MIX_RESULT})
        captured.setdefault("faq_calls", []).append(
            {"url": url, "params": params, "body": body, "headers": headers}
        )
        return httpx.Response(200, json={"result": qa_result(qa_score, qa_questions)})

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


def sse_chunk(delta: str) -> str:
    """一段 OpenAI 兼容的流式响应（`data: {...}\\n\\n`）。"""
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "mock",
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


def make_model_handler(captured: dict[str, Any]):
    """真 ChatOpenAI 的 HTTP 出口：记录请求体，回一段流式 SSE。"""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("model_requests", []).append(json.loads(request.content))
        text = "".join(sse_chunk(token) for token in ANSWER_TOKENS) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"})

    return handler


async def run_case(
    label: str,
    *,
    qa_score: float = 0.42,
    mix_status: int = 200,
    ranges: list[dict[str, Any]] | None = None,
    prompts: dict[str, Prompt] | None = None,
    stream_error: Exception | None = None,
    qa_questions: list[str] | None = None,
    real_model: bool = False,
    **overrides: Any,
) -> tuple[list[tuple[str, dict]], dict[str, Any], Settings]:
    captured: dict[str, Any] = {}
    settings = Settings(
        answer_model_url="http://model.test/answer/chat/completions",
        **overrides,
    )
    # real_model：不注入假模型，而是把模块里的 ChatOpenAI 换成「真类 + MockTransport」，
    # 让 langchain 真的走一遍消息转换与 SSE 解析（其余场景走假模型，快且与 langchain 解耦）
    model_client: httpx.AsyncClient | None = None
    real_openai = rag_module.ChatOpenAI
    answer_model: Any = _AnswerModel(captured, stream_error)
    if real_model:
        model_client = httpx.AsyncClient(transport=httpx.MockTransport(make_model_handler(captured)))
        rag_module.ChatOpenAI = lambda **kwargs: real_openai(  # type: ignore[assignment]
            **kwargs, http_async_client=model_client
        )
        answer_model = None
    agent = RagAgent(
        settings,
        term_repository=_TermRepository(),
        prompt_repository=_PromptRepository(captured, prompts),
        rewrite_model=_RewriteModel(captured),
        answer_model=answer_model,
    )
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": RAW_QUERY}],
            # 默认 ranges 标成标准问答库（knowledgeType=2），否则连 FAQ 探测都不会发生；
            # 空间字段仍留空：project_id 回落全局 KB_PROJECT_ID，tenantId 直接不带。
            "ranges": ranges
            or [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID], "knowledgeType": 2}],
        }
    )
    real_client = httpx.AsyncClient
    real_httpx = rag_module.httpx
    # 给 rag_agent 换一个「httpx 副本」而不是直接改 httpx 模块的属性：openai SDK 内部会做
    # `isinstance(client, httpx.AsyncClient)`，把模块属性换成函数会让它抛
    # `TypeError: isinstance() arg 2 must be a type ...`（真模型场景踩过这个坑）。
    shim = types.SimpleNamespace(**vars(real_httpx))
    shim.AsyncClient = lambda **kwargs: real_client(
        transport=httpx.MockTransport(
            make_handler(
                captured, qa_score=qa_score, mix_status=mix_status, qa_questions=qa_questions
            )
        ),
        **kwargs,
    )
    rag_module.httpx = shim
    try:
        events = [chunk async for chunk in agent.stream(request)]
    finally:
        rag_module.httpx = real_httpx
        rag_module.ChatOpenAI = real_openai
        if model_client is not None:
            await model_client.aclose()
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
    check("最终检索不带查询参数（普通 POST）", captured.get("mix_params"), {})
    body = captured.get("mix_body") or {}
    check("body.query", body.get("query"), REWRITTEN)
    check("body.keywords", body.get("keywords"), KEYWORDS)
    # body 只有三项：外部接口不认识 rerank 相关字段，别照着 FAQ 那条抄
    check("body 只有 query/keywords/ranges", sorted(body), ["keywords", "query", "ranges"])
    check(
        "body.ranges 原样用请求里的 ranges（含门户的 knowledgeType）",
        body.get("ranges"),
        [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID], "knowledgeType": 2}],
    )

    # FAQ 探测：两轮（改写前 / 改写后），每轮并发两个 query 变体
    # （原问题与术语映射后的问题，本脚本映射是恒等函数故两者相同）-> 共 4 次请求。
    # ranges 带了 knowledgeType=2（否则根本不探），但未带空间信息 -> 回落全局配置。
    faq_calls = captured.get("faq_calls") or []
    check("FAQ 探测请求数", len(faq_calls), 4)
    check(
        "两条通道都不带 Authorization（召回接口不拼鉴权头）",
        {
            "authorization" in call["headers"]
            for call in [*captured["mix_calls"], *faq_calls]
        },
        {False},
    )
    check(
        "FAQ 探测过的 query",
        sorted({call["body"]["query"] for call in faq_calls}),
        sorted({RAW_QUERY, REWRITTEN}),
    )
    check(
        "FAQ 探测回落全局 project_id、不带 tenantId",
        {scope_of(call) for call in faq_calls},
        {json.dumps({"project_id": settings.kb_project_id}, sort_keys=True)},
    )
    check("FAQ 探测不带 keywords", {tuple(call["body"]["keywords"]) for call in faq_calls}, {()})
    check(
        "FAQ 探测（平台接口）body 仍带 disable_rerank/rerank_params",
        all(
            call["body"].get("disable_rerank") is False and "rerank_params" in call["body"]
            for call in faq_calls
        ),
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
    check("只有一条候选时不给相近问题", "related_queries" in names_of(parsed), False)
    print(f"    事件序列={names_of(parsed)}")


async def case_related_queries() -> None:
    """FAQ 直返 + 库里另有 3 条相近问题：token 流结束后一条事件带三条。"""
    parsed, _, _ = await run_case(
        "FAQ 直返附带三条相近问题", qa_score=0.99, qa_questions=[FAQ_QUESTION, *RELATED_QUESTIONS]
    )
    names = names_of(parsed)
    related = next((data for name, data in parsed if name == "related_queries"), None)

    check("answer_type", parsed[-1][1].get("answer_type"), "faq")
    check("相近问题三条一次给出、顺序即相似度序", (related or {}).get("queries"), RELATED_QUESTIONS)
    check(
        "答案那条不重复出现在相近问题里",
        FAQ_QUESTION in ((related or {}).get("queries") or []),
        False,
    )
    # 顺序即需求：流式输出全部结束后才发，且必须在 done 之前
    check(
        "事件顺序",
        names,
        ["status", "sources", "status", "token", "token", "related_queries", "done"],
    )
    print(f"    事件序列={names}")
    print(f"    相近问题={(related or {}).get('queries')}")


async def case_related_queries_short() -> None:
    """候选去重/去空之后凑不满三条：一条都不发（不是发两条）。"""
    parsed, _, _ = await run_case(
        "相近问题不足三条不发",
        qa_score=0.99,
        # 答案那条 + 一条有效 + 一条重复 + 一条空问题
        qa_questions=[FAQ_QUESTION, RELATED_QUESTIONS[0], RELATED_QUESTIONS[0], ""],
    )

    check("不足三条不发相近问题", "related_queries" in names_of(parsed), False)
    check("answer_type", parsed[-1][1].get("answer_type"), "faq")
    print(f"    事件序列={names_of(parsed)}")


async def case_faq_threshold() -> None:
    """FAQ 分数没过阈值：不直返，走完整链路（RagAgent 会把最高分打进日志便于回调阈值）。"""
    threshold = Settings().faq_similarity_threshold
    parsed, _, _ = await run_case("FAQ 分数未达阈值", qa_score=threshold - 0.01)
    check("未达阈值不直返", parsed[-1][1].get("answer_type"), "rag")


async def case_range_carries_space() -> None:
    """range 自带空间与类型：两条通道都用 range 自带的空间，不回落全局配置。

    FAQ 那条按 (project_id, tenantId) 分组，所以这里断言 FAQ 请求带的空间就是 range
    自带的；最终检索是外部接口，整条 range（空间 + knowledgeType）原样进 body。
    """
    ranges = [
        {
            "knowledge_base_id": KB_ID,
            "doc_range": [DOC_ID],
            "project_id": SPACE_A,
            "tenantId": "tenant-9",
            "knowledgeType": 2,
        }
    ]
    expect = {"project_id": SPACE_A, "tenantId": "tenant-9"}
    _, captured, _ = await run_case("QA 库自带空间", ranges=ranges)

    check(
        "FAQ 探测用 range 自带空间",
        {scope_of(call) for call in captured["faq_calls"]},
        {json.dumps(expect, sort_keys=True)},
    )
    # 最终检索不取查询参数，整条 range 原样进 body（空间 + 类型都在）
    check("最终检索不带查询参数", captured.get("mix_params"), {})
    check(
        "最终检索把整条 range 原样带在 body 里",
        captured["mix_calls"][0]["body"]["ranges"],
        [
            {
                "knowledge_base_id": KB_ID,
                "doc_range": [DOC_ID],
                "project_id": SPACE_A,
                "tenantId": "tenant-9",
                "knowledgeType": 2,
            }
        ],
    )
    print(f"    FAQ 参数={[call['params'] for call in captured['faq_calls']]}")
    print(f"    最终检索参数={captured.get('mix_params')}")
    print(f"    最终检索 ranges={captured['mix_calls'][0]['body']['ranges']}")


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
    """两个 QA 库分属两个空间：FAQ 按空间分组、每组各发一次、组内不混库；最终检索只发一次。"""
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
    check("FAQ 空间 -> 知识库 归属", by_space, {SPACE_A: {KB_ID}, SPACE_B: {KB_ID_2}})

    # 最终检索是外部接口的普通 POST：不分组、**只发一次**、不带查询参数，
    # ranges 里两个库（连同各自的空间）一起原样发。分组是 FAQ 那条通道自己的事，别抄过来。
    mix_calls = captured.get("mix_calls") or []
    check("最终检索只发一次", len(mix_calls), 1)
    only = mix_calls[0]
    check("最终检索不带查询参数", only["params"], {})
    check("最终检索 body 只有三项", sorted(only["body"]), ["keywords", "query", "ranges"])
    check(
        "最终检索一次带上全部知识库（含各自空间）",
        only["body"]["ranges"],
        [
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
        ],
    )
    print(f"    FAQ 参数={[call['params'] for call in calls]}")
    print(f"    FAQ 分组={ {key: sorted(value) for key, value in by_space.items()} }")
    print(f"    最终检索参数={[call['params'] for call in mix_calls]}")
    print(f"    最终检索 ranges={only['body']['ranges']}")


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
    """range 不带 tenantId：查询参数只剩 project_id，body 里也不凭空补 tenant 字段。"""
    _, captured, settings = await run_case("range 未带 tenantId")
    check("最终检索不带查询参数", captured.get("mix_params"), {})
    check(
        "最终检索也不往 body 里塞 tenant",
        [key for key in ("tenantId", "tenant_id") if key in captured["mix_body"]["ranges"][0]],
        [],
    )
    check(
        "FAQ 探测不带 tenantId 查询参数",
        {scope_of(call) for call in captured["faq_calls"]},
        {json.dumps({"project_id": settings.kb_project_id}, sort_keys=True)},
    )


async def case_custom_url() -> None:
    custom = (
        "https://10.0.0.9:30443/llm/llmops/tenants/t1/gateway"
        "/applet/api/v1/knowlhub/kbs:mix-retrieve"
    )
    _, captured, _ = await run_case("整条覆盖接口地址", kb_mix_retrieve_url=custom)
    check("自定义地址生效", captured.get("mix_url"), custom)
    check("自定义地址也不带查询参数", captured.get("mix_params"), {})


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


async def case_real_model_stream() -> None:
    """回答模型用**真的** `ChatOpenAI` 流式出 token（只在进程内 mock 掉它的 HTTP 出口）。

    其余场景都把模型换成假对象（跑得快、与 langchain 解耦），代价是没人验证
    「`ChatOpenAI.astream(messages)` 真的认我们装配的 dict 消息、`chunk.content` 真的是文本」
    —— 这恰恰是「拆掉包装 agent、直接用 ChatOpenAI」时唯一换掉的契约。
    这里把真类装回模块里、注入 MockTransport 客户端，端到端跑一遍非 FAQ 链路。
    """
    parsed, captured, _ = await run_case(
        "真 ChatOpenAI 流式出 token",
        real_model=True,
        ranges=[{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID], "knowledgeType": 1}],
    )

    check(
        "token 事件来自真模型的流式响应",
        [data["content"] for name, data in parsed if name == "token"],
        list(ANSWER_TOKENS),
    )
    requests = captured.get("model_requests") or []
    check("模型确实被调用一次", len(requests), 1)
    body = requests[0] if requests else {}
    check("HTTP 层开了流式", body.get("stream"), True)
    check(
        "消息角色：system（固定指令）+ user（含问题与资料）",
        [item["role"] for item in body.get("messages", [])],
        ["system", "user"],
    )
    check(
        "user 段带上了检索资料",
        CHUNK_TEXT in (body.get("messages", [{}, {}])[1].get("content") or ""),
        True,
    )
    check("answer_type", parsed[-1][1].get("answer_type"), "rag")
    print(f"    事件序列={names_of(parsed)}")


async def main() -> int:
    await case_normal()
    await case_prompt_key()
    await case_faq_direct()
    await case_related_queries()
    await case_related_queries_short()
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
    await case_real_model_stream()

    if FAILURES:
        print("\nFAIL")
        for item in FAILURES:
            print(" -", item)
        return 1
    print("\nOK  跨库检索 + 单步生成链路十六组场景全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
