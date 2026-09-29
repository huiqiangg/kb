"""模型客户端的组装规则回归。

组装点**就在 `RagAgent` 里，两个方法各写一遍**（没有公共的 `chat_model` 模块，
也不包 agent）：
- `RagAgent._ensure_rewrite_model()` —— 改写模型（非流式，出 JSON）；
- `RagAgent._ensure_answer_model()` —— 回答模型，FAQ 直返润色与最终答案生成共用。

两处规则相同，因此都用「探针 + 真类」的方式验证：把模块里的 `ChatOpenAI` 换成
记录 kwargs 再交回真类的函数，断言**真实调用点传了哪些参数**，同时真类照常实例化
（参数名写错会立刻暴露）。规则：
- 每个模型各配一条完整地址（`REWRITE_MODEL_URL` / `ANSWER_MODEL_URL`）与一个密钥
  （`REWRITE_MODEL_KEY` / `ANSWER_MODEL_KEY`），组装时只去掉 `/chat/completions` 后缀，
  两条地址、两个密钥都互不覆盖；
- 鉴权用各自的 `*_MODEL_KEY`（Bearer 与 `accessKey` 头都用它），留空时 `api_key`
  位置给占位串；
- 两个模型**都**在地址留空时抛 RuntimeError（否则 base_url 为空会静默打到 api.openai.com），
  且必须是**懒建**——构造 `RagAgent` 本身不该抛。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_model_config.py
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import app.services.rag_agent as rag_module
from app.core.config import Settings
from app.services.rag_agent import RagAgent

REWRITE_URL = "http://gw.test:8099/api/model/chat/completions"
ANSWER_URL = "http://gw.test:8099/api/model/75e3041c-7b05-4d8d-a51c-317c470dfc26/chat/completions"

FAILURES: list[str] = []


def check(label: str, actual: Any, expected: Any) -> None:
    ok = actual == expected
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")


@contextmanager
def spy_chat_openai(module: Any) -> Iterator[dict[str, Any]]:
    """把模块里的 `ChatOpenAI` 换成探针：记录 kwargs 后照常实例化真类。"""
    real = module.ChatOpenAI
    captured: dict[str, Any] = {}

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real(**kwargs)

    module.ChatOpenAI = spy
    try:
        yield captured
    finally:
        module.ChatOpenAI = real


def case_answer_model() -> None:
    """回答模型：地址去掉 `/chat/completions`，模型名/超时/重试照传。"""
    settings = Settings(answer_model_url=ANSWER_URL, answer_model_name="Qwen3-32B")
    with spy_chat_openai(rag_module) as kwargs:
        model = RagAgent(settings)._ensure_answer_model()

    check(
        "base_url 去掉 /chat/completions",
        kwargs.get("base_url"),
        ANSWER_URL.removesuffix("/chat/completions"),
    )
    check("模型名照传", kwargs.get("model"), "Qwen3-32B")
    check("max_retries", kwargs.get("max_retries"), 1)
    check(
        "timeout 取 REQUEST_TIMEOUT_SECONDS",
        kwargs.get("timeout"),
        settings.request_timeout_seconds,
    )
    check("真类实例化后地址一致", model.openai_api_base, kwargs.get("base_url"))


def case_two_models_independent() -> None:
    """改写与回答模型各用各的地址与密钥，互不影响（含带模型 uuid 的路径）。"""
    settings = Settings(
        rewrite_model_url=REWRITE_URL,
        rewrite_model_key="rw-key",
        answer_model_url=ANSWER_URL,
        answer_model_key="ans-key",
    )

    with spy_chat_openai(rag_module) as rewrite_kwargs:
        RagAgent(settings)._ensure_rewrite_model()
    with spy_chat_openai(rag_module) as answer_kwargs:
        RagAgent(settings)._ensure_answer_model()

    check(
        "改写模型地址",
        rewrite_kwargs.get("base_url"),
        REWRITE_URL.removesuffix("/chat/completions"),
    )
    check(
        "回答模型地址保留 uuid 段",
        answer_kwargs.get("base_url"),
        ANSWER_URL.removesuffix("/chat/completions"),
    )
    check(
        "两者地址不同",
        rewrite_kwargs.get("base_url") == answer_kwargs.get("base_url"),
        False,
    )
    check("改写模型名", rewrite_kwargs.get("model"), settings.rewrite_model_name)
    check("改写模型 api_key", rewrite_kwargs.get("api_key"), "rw-key")
    check("回答模型 api_key", answer_kwargs.get("api_key"), "ans-key")
    check(
        "两者密钥不同",
        rewrite_kwargs.get("api_key") == answer_kwargs.get("api_key"),
        False,
    )


def case_missing_model_url() -> None:
    """模型未配地址：懒建，取用时抛 RuntimeError 并点名是哪个变量。

    不抛的话 `base_url` 为空会让 openai 客户端静默打到 api.openai.com ——
    而两侧的调用失败都只吞成一条 warning，问题会一路跑到生产才发现。
    """
    for label, key, call in (
        ("回答模型", "ANSWER_MODEL_URL", lambda agent: agent._ensure_answer_model()),
        ("改写模型", "REWRITE_MODEL_URL", lambda agent: agent._ensure_rewrite_model()),
    ):
        agent = RagAgent(Settings(**{key.lower(): ""}))  # 构造期不该抛
        try:
            call(agent)
            message = "<没抛异常>"
        except RuntimeError as exc:
            message = str(exc)

        check(f"{label}未配地址时抛 RuntimeError", message != "<没抛异常>", True)
        check(f"{label}错误信息点名 {key}", key in message, True)
        print(f"    {label}错误信息={message}")


def case_keys() -> None:
    """密钥与鉴权头：各自的 `*_MODEL_KEY` 有值则两处都用它，留空则占位串且不带 accessKey。"""
    settings = Settings(answer_model_key="ans-key", answer_model_url=ANSWER_URL)
    with spy_chat_openai(rag_module) as kwargs:
        RagAgent(settings)._ensure_answer_model()
    check("回答模型 api_key 用 ANSWER_MODEL_KEY", kwargs.get("api_key"), "ans-key")
    check("accessKey 头带上", kwargs.get("default_headers"), {"accessKey": "ans-key"})

    with spy_chat_openai(rag_module) as empty:
        RagAgent(Settings(answer_model_key="", answer_model_url=ANSWER_URL))._ensure_answer_model()
    check("留空时 api_key 给占位串", empty.get("api_key"), "EMPTY")
    check("留空时不带 accessKey 头", empty.get("default_headers"), None)

    with spy_chat_openai(rag_module) as rewrite_kwargs:
        settings = Settings(rewrite_model_key="rw-key", rewrite_model_url=REWRITE_URL)
        RagAgent(settings)._ensure_rewrite_model()
    check("改写模型 api_key 用 REWRITE_MODEL_KEY", rewrite_kwargs.get("api_key"), "rw-key")


def main() -> int:
    for name, case in (
        ("回答模型组装", case_answer_model),
        ("两个模型地址与密钥互不覆盖", case_two_models_independent),
        ("模型未配地址", case_missing_model_url),
        ("密钥与鉴权头", case_keys),
    ):
        print(f"--- {name}")
        case()

    if FAILURES:
        print("\n✗ 失败：")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("\nOK  模型客户端组装四组用例全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
