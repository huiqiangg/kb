"""术语映射与提示词的读取层。

两张表都是低频变更的配置表，因此按 TTL 缓存在进程内，避免每次问答都查库。
任何一次加载失败都不向上抛：术语映射降级为空表（不替换），提示词降级为 None（回落内置）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from functools import lru_cache
from typing import Any, Generic, TypeVar

from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import Database, get_database
from app.models.prompt import Prompt
from app.services.text_normalizer import EMPTY_TERM_MAPPER, TermMapper

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 加载失败后的重试间隔，避免库故障时每个请求都打一次库
RETRY_INTERVAL_SECONDS = 30.0


def _text_column(row: Mapping[str, Any], name: str) -> str:
    """按**列名**取一个可空文本列，规整成去掉首尾空白的字符串。

    不要用 `for a, b, c in rows` 这类位置解包：SELECT 的列顺序一调整（或中间插一列），
    位置解包轻则静默把两列读串、重则 `ValueError`，而报错点离真正的原因很远。
    按列名取则与 SELECT 的书写顺序解耦 —— 列名写错会直接 `KeyError`，即刻可见。
    """
    value = row.get(name)
    return str(value).strip() if value is not None else ""


class _TtlStore(Generic[T]):
    """带 TTL 与失败降级的进程内缓存，并用锁避免并发重复加载。

    同一个 store 的多个并发 `get()` **只会触发一次 loader**（即只查一次库）：
    缓存未命中时先抢锁，拿到锁后再判一次缓存，只有第一个进来的协程真正加载。
    所以 `asyncio.gather(prompts.get(a), prompts.get(b), prompts.get(c))`
    不会打出三条 SQL —— 它们共用同一个 store，而 `_load()` 本来就把整张表读进来，
    最终只有一条 `SELECT ... FROM rag_prompt`。注意前提是**同一个 repository**：
    术语表和提示词表是两个独立 store，跨表 gather 会各开一个连接。

    失败静默窗口必须在锁内判定，否则「冷启动 + 库故障」时并发的几个调用会绕过它，
    挨个去重试 —— 每个都要等满一次 connect_timeout，把首个请求卡到超时叠加。
    """

    def __init__(self, name: str, ttl_seconds: int, empty: T) -> None:
        self._name = name
        self._ttl = max(ttl_seconds, 0)
        self._empty = empty
        self._value: T | None = None
        self._expires_at = 0.0  # 成功值的过期时刻
        self._retry_after = 0.0  # 加载失败后的静默窗口截止时刻
        self._lock = asyncio.Lock()

    def _lookup(self, now: float) -> tuple[bool, T]:
        """命中缓存时返回 `(True, 缓存值)`；需要调 loader 时返回 `(False, 占位值)`。"""
        if self._value is not None and now < self._expires_at:
            return True, self._value
        # 从未成功加载过、且正处在失败静默窗口内：直接降级，不再打库
        if self._value is None and now < self._retry_after:
            return True, self._empty
        return False, self._empty

    async def get(self, loader: Callable[[], Awaitable[T]]) -> T:
        hit, value = self._lookup(time.monotonic())
        if hit:
            return value
        async with self._lock:
            # 等锁期间可能已被别的协程填好，锁内再判一次
            hit, value = self._lookup(time.monotonic())
            if hit:
                return value
            try:
                loaded = await loader()
            except Exception as exc:  # 配置表不可用不应打断问答链路
                logger.warning("%s 加载失败，本次走降级：%s", self._name, exc)
                self._retry_after = time.monotonic() + RETRY_INTERVAL_SECONDS
                # 有过成功结果就继续沿用，避免库抖动导致配置瞬间清空
                return self._value if self._value is not None else self._empty
            self._value = loaded
            self._expires_at = time.monotonic() + self._ttl
            self._retry_after = 0.0
            return loaded

    def invalidate(self) -> None:
        """显式失效：下次访问立即重新加载，不受失败静默窗口约束。"""
        self._value = None
        self._expires_at = 0.0
        self._retry_after = 0.0


class TermMappingRepository:
    """rag_keywords_mapping：用户口语 -> 银行标准术语。"""

    _SQL = text(
        "SELECT source_term, standard_term FROM rag_keywords_mapping "
        "WHERE enabled = 1 ORDER BY id ASC"
    )

    def __init__(self, database: Database, ttl_seconds: int) -> None:
        self._database = database
        self._store = _TtlStore[TermMapper]("术语映射表", ttl_seconds, EMPTY_TERM_MAPPER)

    async def mapper(self) -> TermMapper:
        if not self._database.enabled:
            return EMPTY_TERM_MAPPER
        return await self._store.get(self._load)

    async def _load(self) -> TermMapper:
        async with self._database.session() as session:
            rows = (await session.execute(self._SQL)).mappings().all()
        # 同名术语保留排序中的首条（现库无重复行，这里只作兜底）
        mapping: dict[str, str] = {}
        for row in rows:
            source_term = _text_column(row, "source_term")
            standard_term = _text_column(row, "standard_term")
            if source_term and standard_term:
                mapping.setdefault(source_term, standard_term)
        logger.info("术语映射表已加载：%d 条", len(mapping))
        return TermMapper.build(mapping)

    def invalidate(self) -> None:
        self._store.invalidate()


class PromptRepository:
    """rag_prompt：提示词配置（固定 system 指令 + 含变量的 user 正文），按 prompt_key 取用。"""

    _SQL = text(
        "SELECT prompt_key, system_content, user_content FROM rag_prompt "
        "WHERE status = 1 ORDER BY modify_time asc"
    )

    def __init__(self, database: Database, ttl_seconds: int) -> None:
        self._database = database
        self._store = _TtlStore[dict[str, Prompt]]("提示词配置表", ttl_seconds, {})

    async def get(self, key: str | None) -> Prompt | None:
        """取指定 key 的提示词；key 为空、无记录或未启用时返回 None（由调用方回落内置）。"""
        if not key or not self._database.enabled:
            return None
        prompts = await self._store.get(self._load)
        return prompts.get(key)

    async def _load(self) -> dict[str, Prompt]:
        async with self._database.session() as session:
            rows = (await session.execute(self._SQL)).mappings().all()
        prompts: dict[str, Prompt] = {}
        for row in rows:
            name = _text_column(row, "prompt_key")
            if not name:
                continue
            user_content = _text_column(row, "user_content")
            # user 段为空等于「问题一个字都没发给模型」，宁可回落内置也不放残缺提示词过去
            if not user_content:
                logger.warning("提示词 %s 的 user_content 为空，跳过并回落内置", name)
                continue
            prompts[name] = Prompt(
                system=_text_column(row, "system_content"), user=user_content
            )
        logger.info("提示词配置表已加载：%s", ", ".join(sorted(prompts)) or "无")
        return prompts

    def invalidate(self) -> None:
        self._store.invalidate()


@lru_cache
def get_term_mapping_repository() -> TermMappingRepository:
    settings = get_settings()
    return TermMappingRepository(get_database(), settings.db_cache_ttl_seconds)


@lru_cache
def get_prompt_repository() -> PromptRepository:
    settings = get_settings()
    return PromptRepository(get_database(), settings.db_cache_ttl_seconds)
