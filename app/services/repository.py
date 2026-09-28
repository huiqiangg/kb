"""术语映射与提示词的读取层。

两张表都是低频变更的配置表，因此按 TTL 缓存在进程内，避免每次问答都查库。
任何一次加载失败都不向上抛：术语映射降级为空表（不替换），提示词降级为 None（回落内置）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Generic, TypeVar

from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import Database, get_database
from app.models.prompt import Prompt
from app.services.text_normalizer import EMPTY_TERM_MAPPER, TermMapper

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 加载失败后的重试间隔，避免库故障时每个请求都打一次库
RETRY_INTERVAL_SECONDS = 30.0


class _TtlStore(Generic[T]):
    """带 TTL 与失败降级的进程内缓存，并用锁避免并发重复加载。"""

    def __init__(self, name: str, ttl_seconds: int, empty: T) -> None:
        self._name = name
        self._ttl = max(ttl_seconds, 0)
        self._empty = empty
        self._value: T | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self, loader: Callable[[], Awaitable[T]]) -> T:
        if self._value is not None and time.monotonic() < self._expires_at:
            return self._value
        async with self._lock:
            if self._value is not None and time.monotonic() < self._expires_at:
                return self._value
            try:
                value = await loader()
            except Exception as exc:  # 配置表不可用不应打断问答链路
                logger.warning("%s 加载失败，本次走降级：%s", self._name, exc)
                self._expires_at = time.monotonic() + RETRY_INTERVAL_SECONDS
                # 有过成功结果就继续沿用，避免库抖动导致配置瞬间清空
                return self._value if self._value is not None else self._empty
            self._value = value
            self._expires_at = time.monotonic() + self._ttl
            return value

    def invalidate(self) -> None:
        self._value = None
        self._expires_at = 0.0


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
            rows = (await session.execute(self._SQL)).all()
        # 同名术语保留排序中的首条（库中「网银」存在重复行）
        mapping: dict[str, str] = {}
        for source, standard in rows:
            source_term, standard_term = (source or "").strip(), (standard or "").strip()
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
        "WHERE status = 1 ORDER BY id ASC"
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
            rows = (await session.execute(self._SQL)).all()
        prompts: dict[str, Prompt] = {}
        for key, system, user in rows:
            name = (str(key) if key else "").strip()
            body = (str(user) if user else "").strip()
            if not name:
                continue
            # user 段为空等于「问题一个字都没发给模型」，宁可回落内置也不放残缺提示词过去
            if not body:
                logger.warning("提示词 %s 的 user_content 为空，跳过并回落内置", name)
                continue
            prompts[name] = Prompt(
                system=(str(system) if system else "").strip(), user=body
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
