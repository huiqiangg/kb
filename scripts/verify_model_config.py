"""模型地址与密钥的组装规则回归。

规则（`app/services/chat_model.py`）：
- **每个模型各配一条完整地址**（`REWRITE_MODEL_URL` / `ANSWER_MODEL_URL`），
  组装时只去掉 `/chat/completions` 后缀，两条地址互不覆盖 —— 不设「全局 base + 局部覆盖」
  的双层配置，所以不存在「全局变量把某个模型的专用路径顶掉」这类静默故障；
- 鉴权用 `MODEL_ACCESS_KEY`（Bearer 与 `accessKey` 头都用它），留空时 `api_key`
  位置给占位串；
- 答案模型未配地址时 `ChatStreamAgent.build` 抛 RuntimeError，由调用方降级。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_model_config.py
"""

import sys
from typing import Any

from app.core.config import Settings
from app.services.chat_model import build_chat_model
from app.services.stream_agent import ChatStreamAgent

REWRITE_URL = "http://gw.test:8099/api/model/chat/completions"
ANSWER_URL = "http://gw.test:8099/api/model/75e3041c-7b05-4d8d-a51c-317c470dfc26/chat/completions"

FAILURES: list[str] = []


def check(label: str, actual: Any, expected: Any) -> None:
    ok = actual == expected
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")


def case_url_suffix() -> None:
    """`/chat/completions` 后缀被去掉（openai 客户端会自己补上）。"""
    settings = Settings(answer_model_url=ANSWER_URL, answer_model_name="Qwen3-32B")
    chat_model = build_chat_model(
        settings, url=settings.answer_model_url, model=settings.answer_model_name
    )

    check(
        "答案模型地址去掉 /chat/completions",
        chat_model.openai_api_base,
        ANSWER_URL.removesuffix("/chat/completions"),
    )
    check("模型名照传", chat_model.model_name, "Qwen3-32B")


def case_two_models_independent() -> None:
    """改写与答案模型各用各的地址，互不影响（含带模型 uuid 的路径）。"""
    settings = Settings(rewrite_model_url=REWRITE_URL, answer_model_url=ANSWER_URL)
    rewrite = build_chat_model(
        settings, url=settings.rewrite_model_url, model=settings.rewrite_model_name
    )
    answer = build_chat_model(
        settings, url=settings.answer_model_url, model=settings.answer_model_name
    )

    check("改写模型地址", rewrite.openai_api_base, REWRITE_URL.removesuffix("/chat/completions"))
    check(
        "答案模型地址保留 uuid 段",
        answer.openai_api_base,
        ANSWER_URL.removesuffix("/chat/completions"),
    )
    check("两者地址不同", rewrite.openai_api_base == answer.openai_api_base, False)


def case_missing_answer_url() -> None:
    """答案模型未配地址：抛 RuntimeError，错误信息点名 ANSWER_MODEL_URL。"""
    settings = Settings(answer_model_url="")
    try:
        ChatStreamAgent.build(settings)
        message = "<没抛异常>"
    except RuntimeError as exc:
        message = str(exc)

    check("未配地址时抛 RuntimeError", message != "<没抛异常>", True)
    check("错误信息点名 ANSWER_MODEL_URL", "ANSWER_MODEL_URL" in message, True)
    print(f"    错误信息={message}")


def case_keys() -> None:
    """密钥与鉴权头：MODEL_ACCESS_KEY 有值则两处都用它，留空则占位串且不带 accessKey。"""
    settings = Settings(model_access_key="ak-key")
    chat_model = build_chat_model(settings, url=REWRITE_URL, model="m")
    check("api_key 用 MODEL_ACCESS_KEY", chat_model.openai_api_key.get_secret_value(), "ak-key")
    check("accessKey 头带上", chat_model.default_headers, {"accessKey": "ak-key"})

    empty = Settings(model_access_key="")
    without = build_chat_model(empty, url=REWRITE_URL, model="m")
    check("留空时 api_key 给占位串", without.openai_api_key.get_secret_value(), "EMPTY")
    check("留空时不带 accessKey 头", without.default_headers or {}, {})


def main() -> int:
    for name, case in (
        ("地址去掉 /chat/completions", case_url_suffix),
        ("两个模型地址互不影响", case_two_models_independent),
        ("答案模型未配地址", case_missing_answer_url),
        ("密钥与鉴权头", case_keys),
    ):
        print(f"--- {name}")
        case()

    if FAILURES:
        print("\n✗ 失败：")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("\nOK  模型地址与密钥组装四组用例全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
