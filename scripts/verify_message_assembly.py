"""消息装配回归：`app/services/prompting.py` 的三个场景装配函数。

一条提示词（`Prompt`）拆成两段：`system` 是固定指令（不含变量，原样下发），
`user` 是含 `{query}` / `{ragkm}` / `{answer}` 占位符的正文（渲染成 user 消息）。

| 场景 | 函数 | user 段占位符 | 对话历史 |
| --- | --- | --- | --- |
| 改写 | `build_rewrite_messages` | `{query}` | **带**（提示词要求结合上下文改写） |
| 最终答案生成 | `build_answer_messages` | `{query}` + `{ragkm}` | 不带（单轮） |
| FAQ 直返润色 | `build_polish_messages` | `{query}` + `{answer}` | 不带（单轮） |

共同的不变量：system 只出现一次且在最前、当前问题**只出现一次**（由各场景 user 段的
`{query}` 带出来）、且落在最后一条 user 消息里。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_message_assembly.py
"""

import sys

from app.models.prompt import Prompt
from app.services.prompting import (
    build_answer_messages,
    build_polish_messages,
    build_rewrite_messages,
    fill,
)

QUERY = "房贷能提前还款吗"
HISTORY = [
    {"role": "user", "content": "上一轮问题"},
    {"role": "assistant", "content": "上一轮回答"},
]
CONTEXT = "[资料1] 借款人可申请提前归还个人住房贷款。\n\n[资料2] 需提前 30 天提出申请。"
FAQ_ANSWER = "可通过手机银行或经办行柜面办理提前还款。"
FAILURES: list[str] = []

REWRITE = Prompt(
    system="你是改写器。只输出 JSON 格式：{\"rewritten_query\": \"...\"}",
    user="用户问题：\n{query}",
)
ANSWER = Prompt(
    system="你是客服助手。\n\n【处理原则】\n1. 仅依据参考上下文回答。",
    user="用户问题：\n{{query}}\n\n参考上下文：\n{{ragkm}}\n\n请输出最终答案",
)
POLISH = Prompt(
    system="你是银行客服润色助手。\n\n要求：1. 不得改变事实。",
    user="用户问题：\n{{query}}\n\n原始答案：\n{{answer}}\n\n请输出最终答案",
)


def check(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        FAILURES.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")


def roles(messages: list[dict[str, str]]) -> list[str]:
    return [item["role"] for item in messages]


def joined(messages: list[dict[str, str]]) -> str:
    return "\n".join(item["content"] for item in messages)


# ---- 场景 1：改写（带上下文）----


def case_rewrite() -> None:
    """改写：system 指令在最前，历史轮次居中，含问题的 user 收尾。"""
    messages = build_rewrite_messages(REWRITE, QUERY, HISTORY)
    check("改写：角色序列", roles(messages), ["system", "user", "assistant", "user"])
    check("改写：system 取固定指令", messages[0], {"role": "system", "content": REWRITE.system})
    check("改写：历史原样在中间", messages[1:3], HISTORY)
    check(
        "改写：user 注入问题",
        messages[-1],
        {"role": "user", "content": f"用户问题：\n{QUERY}"},
    )
    print(f"  改写 -> {roles(messages)}")


def case_system_has_no_placeholder() -> None:
    """system 段是固定指令：原样下发，JSON 花括号与本该由 user 承担的占位符都不动。"""
    prompt = Prompt(system="只输出 JSON：{\"rewritten_query\": \"x\"}，问题在 {query} 里", user="{query}")
    messages = build_rewrite_messages(prompt, QUERY, [])
    check("system 原样保留 JSON 花括号", '{"rewritten_query": "x"}' in messages[0]["content"], True)
    check("system 不做占位符替换", "{query}" in messages[0]["content"], True)
    check("user 段正常替换", messages[-1]["content"], QUERY)
    print("  system -> 固定指令原样下发（不做占位符替换）")


def case_system_empty() -> None:
    """system 段为空时不产生空消息，只发 user。"""
    for label, messages in (
        ("改写", build_rewrite_messages(Prompt("", "{query}"), QUERY, HISTORY)),
        ("生成", build_answer_messages(Prompt("", "{query}\n{ragkm}"), QUERY, CONTEXT)),
        ("润色", build_polish_messages(Prompt("", "{query}\n{answer}"), QUERY, FAQ_ANSWER)),
    ):
        check(f"空 system 不产生 system 消息（{label}）", "system" in roles(messages), False)
        check(f"空 system 仍有 user（{label}）", messages[-1]["role"], "user")
    print("  空 system -> 不产生 system 消息")


# ---- 场景 2：最终答案生成（单轮）----


def case_answer() -> None:
    """生成：system + 一条 user（`{query}` + `{ragkm}`），单轮。"""
    messages = build_answer_messages(ANSWER, QUERY, CONTEXT)
    check("生成：角色序列", roles(messages), ["system", "user"])
    check("生成：system 取固定指令", messages[0]["content"], ANSWER.system)
    check(
        "生成：user 注入问题与参考上下文",
        messages[1],
        {
            "role": "user",
            "content": f"用户问题：\n{QUERY}\n\n参考上下文：\n{CONTEXT}\n\n请输出最终答案",
        },
    )
    check("生成：{ragkm} 不留字面量", "ragkm" in joined(messages), False)
    check("生成：参考上下文确实到位", CONTEXT in joined(messages), True)
    print(f"  生成 -> {roles(messages)}")


# ---- 场景 3：FAQ 直返润色（单轮）----


def case_polish() -> None:
    """润色：system + 一条 user（`{query}` + `{answer}`），单轮且**不带历史**。"""
    messages = build_polish_messages(POLISH, QUERY, FAQ_ANSWER)
    check("润色：角色序列", roles(messages), ["system", "user"])
    check("润色：system 取固定指令", messages[0]["content"], POLISH.system)
    check(
        "润色：库里原文注入 {answer}",
        messages[1],
        {
            "role": "user",
            "content": f"用户问题：\n{QUERY}\n\n原始答案：\n{FAQ_ANSWER}\n\n请输出最终答案",
        },
    )
    check("润色：不带对话历史", "上一轮问题" in joined(messages), False)
    print(f"  润色 -> {roles(messages)}")


# ---- 跨场景不变量 ----


def case_invariants() -> None:
    """三个场景共同的不变量：问题只出现一次，且落在最后一条 user 消息里。"""
    cases = {
        "改写": build_rewrite_messages(REWRITE, QUERY, HISTORY),
        "生成": build_answer_messages(ANSWER, QUERY, CONTEXT),
        "润色": build_polish_messages(POLISH, QUERY, FAQ_ANSWER),
    }
    for label, messages in cases.items():
        check(f"当前问题只出现一次（{label}）", joined(messages).count(QUERY), 1)
        check(f"最后一条是 user（{label}）", messages[-1]["role"], "user")
        check(f"system 至多一条且在最前（{label}）", roles(messages).count("system") <= 1, True)
    print("  不变量 -> 问题只出现一次、落在最后一条 user")


def case_missing_query_placeholder() -> None:
    """user 段漏写 `{query}`：问题不会出现在消息里。

    这是「配置漏写」的可见信号 —— 装配层不自动把问题补到末尾，免得提示词和实际输入对不上。
    db.sql 的三条记录与三个内置兜底都带 `{query}`，正常配置下问题必然会被带出来。
    """
    for label, messages in (
        ("改写", build_rewrite_messages(Prompt("指令。", "请改写。"), QUERY, [])),
        ("生成", build_answer_messages(Prompt("指令。", "参考：{ragkm}"), QUERY, CONTEXT)),
        ("润色", build_polish_messages(Prompt("指令。", "原文：{answer}"), QUERY, FAQ_ANSWER)),
    ):
        check(f"漏写 {{{{query}}}} 时问题不出现（{label}）", joined(messages).count(QUERY), 0)
        check(f"漏写 {{{{query}}}} 时其它占位符照填（{label}）", "{" in joined(messages), False)
    print("  user 段漏写 {query} -> 问题不出现（可见信号，不自动补齐）")


def case_scenario_isolation() -> None:
    """各场景只填自己声明的占位符，不串台、也不静默剔除漏传的那个。"""
    generated = build_answer_messages(Prompt("", "指令：{answer}"), QUERY, CONTEXT)
    check("生成场景不填 {answer}", "{answer}" in joined(generated), True)
    check("生成场景的 context 未注入", CONTEXT in joined(generated), False)
    polished = build_polish_messages(Prompt("", "指令：{ragkm}"), QUERY, FAQ_ANSWER)
    check("润色场景不填 {ragkm}", "{ragkm}" in joined(polished), True)
    print("  场景隔离 -> 占位符名对不上时原样保留（可见信号，不静默剔除）")


def case_fill_both_brace_styles() -> None:
    """`fill()` 同时认双/单花括号，且 JSON 花括号原样保留。"""
    template = '{"rewritten_query": "x"} 单：{query} 双：{{query}}'
    result = fill(template, query=QUERY)
    check("fill 双花括号被替换", "{{query}}" in result, False)
    check("fill 单花括号被替换", f"单：{QUERY}" in result, True)
    check("fill 双花括号替换为纯值", result.endswith(f"双：{QUERY}"), True)
    check("fill JSON 花括号保留", '{"rewritten_query": "x"}' in result, True)
    print(f"  fill -> {result!r}")


def main() -> int:
    cases = (
        ("改写（带上下文）", case_rewrite),
        ("system 段不含变量", case_system_has_no_placeholder),
        ("system 段为空", case_system_empty),
        ("最终答案生成（单轮）", case_answer),
        ("FAQ 直返润色（单轮）", case_polish),
        ("跨场景不变量", case_invariants),
        ("user 段漏写 {query}", case_missing_query_placeholder),
        ("场景隔离", case_scenario_isolation),
        ("占位符替换基础", case_fill_both_brace_styles),
    )
    for label, case in cases:
        print(f"--- {label}")
        case()

    if FAILURES:
        print("\nFAIL")
        for item in FAILURES:
            print(" -", item)
        return 1
    print(f"\nOK  三个场景的消息装配共 {len(cases)} 组用例全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
