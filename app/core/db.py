"""异步数据库引擎与连接池。

设计原则：启动不主动连库、连不上不影响服务可用。引擎惰性创建，
调用方（见 app/services/repository.py）负责捕获异常并走降级逻辑。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Database:
    """MySQL 异步引擎封装。db_enabled 为 false 时退化为空实现。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.db_enabled

    def _ensure_engine(self) -> AsyncEngine | None:
        if not self.enabled:
            return None
        if self._engine is None:
            self._engine = create_async_engine(
                self.settings.db_url,
                pool_size=self.settings.db_pool_size,
                max_overflow=self.settings.db_max_overflow,
                pool_recycle=self.settings.db_pool_recycle_seconds,
                # 取连接前先探活，避免 MySQL 主动断连后拿到失效连接
                pool_pre_ping=True,
                connect_args={"connect_timeout": self.settings.db_connect_timeout},
            )
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        self._ensure_engine()
        if self._session_factory is None:
            raise RuntimeError("数据库未启用（DB_ENABLED=false）")
        async with self._session_factory() as session:
            yield session

    async def dispose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


@lru_cache
def get_database() -> Database:
    return Database(get_settings())
