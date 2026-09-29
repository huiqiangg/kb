"""配置表读取层回归：把「行 -> 领域对象」的映射规则钉死。

为什么要单独守这一层：`_load()` 的错误是**静默**的。
用 `for key, system, user in rows` 这类位置解包时，只要 SELECT 的列顺序被调整过
（比如把 `user_content` 挪到 `system_content` 前面），system 与 user 就**悄悄对调**，
不报错、不 warning，直到模型拿到的提示词少了变量才发现。

这里用假的 session 造出「列顺序被打乱、还多带了列」的行，断言仍按列名取到正确内容。
另外守住缓存的**并发脾气**：冷启动 + 并发取用只许查一次库，库故障时也不许并发重试。
没有真实出网请求。

跑法：PYTHONPATH=. .venv/bin/python scripts/verify_repository_loading.py
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.repository import PromptRepository, TermMappingRepository  # noqa: E402

FAILURES: list[str] = []


def check(label: str, actual: Any, expected: Any) -> None:
    ok = actual == expected
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {actual!r}")
    if not ok:
        FAILURES.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")


class _Result:
    """只实现被调用到的那部分：`.mappings().all()` / `.mappings().one()`。"""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[Mapping[str, Any]]:
        return list(self._rows)

    def one(self) -> Mapping[str, Any]:
        return self._rows[0]


class _Session:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = rows

    async def execute(self, _statement: Any) -> _Result:
        return _Result(self._rows)


class _Database:
    """`enabled` + 异步上下文 `session()`，与 `app.core.db.Database` 的最小同形替身。

    `delay` 让 `<session>` 真正让出一次事件循环 —— 否则 loader 全程不让出，
    并发的 `get()` 会退化成「第一个跑完第二个才开始」，测不出抢锁的行为。
    `sessions` 统计开了几次会话，即「查了几次库」。
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], delay: float = 0.0) -> None:
        self.enabled = True
        self._rows = rows
        self.delay = delay
        self.sessions = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_Session]:
        self.sessions += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        yield _Session(self._rows)


class _BrokenDatabase:
    """一开会话就炸，模拟库不可用（`_load` 抛错 -> store 降级）。"""

    def __init__(self, delay: float = 0.0) -> None:
        self.enabled = True
        self.delay = delay
        self.sessions = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_Session]:
        self.sessions += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        raise RuntimeError("模拟数据库不可用")
        yield  # pragma: no cover - 仅为让函数成为生成器


async def case_prompt_column_order() -> None:
    """列顺序被打乱、并多出一个未选取的列时，仍按列名取到正确内容。"""
    # 键顺序刻意写成 user / id / key / system，模拟有人调整了 SELECT 的列顺序
    rows = [
        {
            "user_content": "用户问题：\n{{query}}",
            "id": 1,
            "prompt_key": "answer_polish",
            "system_content": "你是银行助手。",
            "prompt_name": "答案润色",
            "status": 1,
        },
        {
            "user_content": "用户问题：\n{{query}}\n\n参考上下文：\n{{ragkm}}",
            "id": 2,
            "prompt_key": "answer_summary",
            "system_content": "你是答案生成助手。",
            "prompt_name": "答案总结",
            "status": 1,
        },
    ]
    prompts = await PromptRepository(_Database(rows), 60)._load()

    check("载入条数", len(prompts), 2)
    check("润色 system", prompts["answer_polish"].system, "你是银行助手。")
    check("润色 user", prompts["answer_polish"].user, "用户问题：\n{{query}}")
    check("生成 system", prompts["answer_summary"].system, "你是答案生成助手。")
    check("生成 user 含 ragkm", "{{ragkm}}" in prompts["answer_summary"].user, True)


async def case_prompt_skip_rule() -> None:
    """key 为空 / user 段为空的行被跳过（宁可回落内置，也不放残缺提示词过去）。"""
    rows = [
        {"prompt_key": "  ", "system_content": "s", "user_content": "u"},
        {"prompt_key": "empty_user", "system_content": "s", "user_content": "   "},
        {"prompt_key": "null_user", "system_content": "s", "user_content": None},
        {"prompt_key": "ok", "system_content": "  固定指令  ", "user_content": "  正文  "},
    ]
    prompts = await PromptRepository(_Database(rows), 60)._load()

    check("只保留合法的一条", sorted(prompts), ["ok"])
    check("system / user 去首尾空白", (prompts["ok"].system, prompts["ok"].user), ("固定指令", "正文"))


async def case_term_mapping() -> None:
    """术语表：空值行丢弃、重复源术语保留首条、顺序不依赖列位置。"""
    rows = [
        {"standard_term": "个人住房贷款", "source_term": "房贷"},
        {"standard_term": "网上银行", "source_term": "网银"},
        {"standard_term": "个人住房贷款", "source_term": "住房贷款"},
        {"standard_term": "  ", "source_term": "空标准术语"},
        {"standard_term": "无源术语", "source_term": None},
        {"standard_term": "重复取首条", "source_term": "网银"},
    ]
    mapper = await TermMappingRepository(_Database(rows), 60)._load()

    check("装入条数", len(mapper.mapping), 3)
    check("重复源术语保留首条", mapper.apply("网银"), "网上银行")
    check("正常替换", mapper.apply("房贷怎么还"), "个人住房贷款怎么还")
    check("空白项未被装入", mapper.mapping.get("空标准术语"), None)


async def case_concurrent_get_single_query() -> None:
    """冷启动时 `gather` 三个 key：只查一次库，TTL 内再来也不再查。"""
    rows = [
        {"prompt_key": "query_understanding", "system_content": "s1", "user_content": "u1"},
        {"prompt_key": "answer_summary", "system_content": "s2", "user_content": "u2"},
        {"prompt_key": "answer_polish", "system_content": "s3", "user_content": "u3"},
    ]
    database = _Database(rows, delay=0.01)
    repository = PromptRepository(database, 60)

    got = await asyncio.gather(
        repository.get("query_understanding"),
        repository.get("answer_summary"),
        repository.get("answer_polish"),
    )

    check("并发取用只查库一次", database.sessions, 1)
    check("各自取到自己 key 的 user 段", [p.user if p else None for p in got], ["u1", "u2", "u3"])

    await asyncio.gather(repository.get("answer_polish"), repository.get("answer_summary"))
    check("TTL 内重复取用仍为一次", database.sessions, 1)


async def case_failure_retry_window() -> None:
    """冷启动 + 库故障：并发取用只尝试一次，静默窗口内不再连库。"""
    database = _BrokenDatabase(delay=0.01)
    repository = PromptRepository(database, 60)

    got = await asyncio.gather(
        repository.get("query_understanding"),
        repository.get("answer_summary"),
        repository.get("answer_polish"),
    )

    check("并发失败不叠加重试", database.sessions, 1)
    check("全部降级为 None", got, [None, None, None])

    await repository.get("answer_summary")
    check("静默窗口内不再打库", database.sessions, 1)

    repository.invalidate()
    await repository.get("answer_summary")
    check("显式失效后立即重试", database.sessions, 2)


async def main() -> int:
    for name, case in (
        ("按列名取值（列顺序被调整）", case_prompt_column_order),
        ("跳过残缺行", case_prompt_skip_rule),
        ("术语映射装配", case_term_mapping),
        ("并发取用只查一次库", case_concurrent_get_single_query),
        ("库故障时不并发重试", case_failure_retry_window),
    ):
        print(f"--- {name}")
        await case()

    if FAILURES:
        print("\n✗ 失败：")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("\nOK  配置表读取层五组用例全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
