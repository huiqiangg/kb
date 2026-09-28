"""跨库混合检索 + 单步答案生成链路回归。

用 httpx.MockTransport 在进程内拦掉全部出网请求，覆盖七个场景：
1. 正常链路：请求地址/查询参数/请求体形状、result[] 解析、SSE 事件顺序、
   生成阶段按「system 固定指令 + user 含 {query}/{ragkm}」装配消息；
2. 生成提示词取 `answer_summary`（并确认不再请求已删除的旧 key）；
3. FAQ 高置信直返：命中即秒回，不再调混合检索，也不调改写模型；
4. 混合检索失败（5xx）：降级为「未检索到内容」，不抛异常；
5. 生成模型失败：回落一句提示语，done 事件照常发出；
6. 未配置 KB_TENANT_ID：请求不带 tenantId 参数；
7. 整条覆盖 KB_MIX_RETRIEVE_URL：自定义地址生效。

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
DOC_ID = "ds3523"
REWRITTEN = "个人住房贷款可以提前还款吗"
KEYWORDS = ["房贷", "提前还款"]
CHUNK_TEXT = "借款人可申请提前归还个人住房贷款。"
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


def make_handler(captured: dict[str, Any], *, faq_score: float, mix_status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("kbs:mix-retrieve"):
            captured["mix_url"] = str(request.url.copy_with(query=None))
            captured["mix_params"] = dict(request.url.params)
            captured["mix_body"] = json.loads(request.content)
            if mix_status != 200:
                return httpx.Response(mix_status, json={"message": "boom"})
            return httpx.Response(200, json={"result": MIX_RESULT})
        if path.endswith("kbs:retrieve"):
            captured["faq_calls"] = captured.get("faq_calls", 0) + 1
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "chunk": {
                                "id": "faq-1",
                                "content": "可通过手机银行或经办行柜面办理提前还款。",
                            },
                            "score": faq_score,
                            "knowledge_base_id": KB_ID,
                        }
                    ]
                },
            )
        raise AssertionError(f"不该出现的请求：{request.url}")

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
    faq_score: float = 0.42,
    mix_status: int = 200,
    tenant_id: str = "tenant-001",
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
            "messages": [{"role": "user", "content": "房贷能提前还还款吗"}],
            "ranges": [{"knowledge_base_id": KB_ID, "doc_range": [DOC_ID]}],
        }
    )
    real_client = rag_module.httpx.AsyncClient
    rag_module.httpx.AsyncClient = lambda **kwargs: real_client(
        transport=httpx.MockTransport(
            make_handler(captured, faq_score=faq_score, mix_status=mix_status)
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


async def case_normal() -> None:
    parsed, captured, settings = await run_case("正常链路")
    names = names_of(parsed)

    check("mix-retrieve 默认推导地址", captured.get("mix_url"), settings.mix_retrieve_url)
    check(
        "默认推导地址形状",
        settings.mix_retrieve_url,
        f"{settings.kb_platform_base_url.rstrip('/')}/applet/api/v1/knowlhub/kbs:mix-retrieve",
    )
    check(
        "faq-retrieve 默认推导地址",
        settings.faq_retrieve_url,
        f"{settings.kb_platform_base_url.rstrip('/')}/applet/api/v1/knowlhub/kbs:retrieve",
    )
    check(
        "查询参数",
        captured.get("mix_params"),
        {"project_id": settings.kb_project_id, "tenantId": "tenant-001"},
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

    # 生成阶段：不再有 answer_polish，只有一次 answer_generate
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

    # 生成阶段：system 是固定指令，user 里是问题 + 参考上下文
    check("生成阶段消息形状", roles_of_answer_call(captured), ["system", "user"])
    check("内置兜底 system 非空", bool((system_content(captured) or "").strip()), True)
    # 回落内置 answer_summary 兜底提示词（DB 未配置）时，资料仍应注入
    check("内置兜底注入资料", CHUNK_TEXT in messages_text(captured), True)
    check("内置兜底问题只出现一次", messages_text(captured).count("房贷能提前还还款吗"), 1)
    check(
        "token 内容",
        [data["content"] for name, data in parsed if name == "token"],
        list(ANSWER_TOKENS),
    )
    print(f"    请求体={json.dumps(body, ensure_ascii=False)}")
    print(f"    事件序列={names}")
    print(f"    取用提示词 key={captured.get('prompt_keys')}")


async def case_prompt_key() -> None:
    # answer_summary 的 user 段用 {ragkm} 接参考上下文（system 段是固定指令，不含变量）
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
    # 生成场景：system 取库里的固定指令，user 取库里的正文并注入检索资料
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
    parsed, captured, _ = await run_case("FAQ 高置信直返", faq_score=0.99)
    check("FAQ 直返 answer_type", parsed[-1][1].get("answer_type"), "faq")
    check("FAQ 直返未调混合检索", "mix_body" in captured, False)
    check("FAQ 直返未调改写模型", captured.get("rewrite_calls", 0), 0)
    check(
        "FAQ 直返有答案输出",
        [data["content"] for name, data in parsed if name == "token"],
        list(ANSWER_TOKENS),
    )
    print(f"    事件序列={names_of(parsed)}")


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


async def main() -> int:
    await case_normal()
    await case_prompt_key()
    await case_faq_direct()
    await case_mix_down()
    await case_generate_down()
    await case_no_tenant()
    await case_custom_url()

    if FAILURES:
        print("\nFAIL")
        for item in FAILURES:
            print(" -", item)
        return 1
    print("\nOK  跨库检索 + 单步生成链路七个场景全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
