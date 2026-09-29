"""按 `db.sql` 重建术语映射表 `rag_keywords_mapping`。

为什么不是「整份 db.sql 重跑」：初始化脚本里只有 `CREATE TABLE` 没有 `DROP`，
表已存在时整份跑会在 1050 上失败；本机也没装 `mysql` CLI（MySQL 跑在 OrbStack 容器里）。
所以这里借项目自己的 async 引擎，从 `db.sql` 里**原样抽出**建表与 INSERT 两条语句，
DROP + 重建 + 灌数据，做完按行数核对，顺便用生产解析路径跑一遍替换自检。

只动 `rag_keywords_mapping` 一张表；`rag_prompt` 的同步走 `scripts/sync_rag_prompt.py`。

跑法：`PYTHONPATH=. .venv/bin/python scripts/reload_term_mapping.py`
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import get_database
from app.services.text_normalizer import TermMapper, normalize_query

TABLE = "rag_keywords_mapping"
SQL_FILE = Path(__file__).resolve().parent.parent / "db.sql"


def statement(prefix: str) -> str:
    """从 db.sql 里取以 `prefix` 开头的那条语句（原样，含中间的行注释）。"""
    for chunk in SQL_FILE.read_text(encoding="utf-8").split(";"):
        if chunk.strip().startswith(prefix):
            return chunk.strip()
    raise SystemExit(f"db.sql 里找不到以 {prefix!r} 开头的语句")


async def main() -> None:
    settings = get_settings()
    if not settings.db_enabled:
        raise SystemExit("DB_ENABLED=false，不重建（这是离线/单测模式）")

    create = statement(f"CREATE TABLE {TABLE}")
    insert = statement(f"INSERT INTO {TABLE}")

    database = get_database()
    async with database.session() as session:
        await session.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await session.execute(text(create))
        await session.execute(text(insert))
        await session.commit()

        rows = (
            await session.execute(
                text(f"SELECT source_term, standard_term FROM {TABLE} ORDER BY id ASC")
            )
        ).mappings().all()
    # 脚本入口用完就释放连接池，否则退出时 aiomysql 会在已关闭的 event loop 里 __del__
    await database.dispose()

    mapping = {
        # 读回校验同样按列名取值，不做位置解包
        str(row.get("source_term") or "").strip(): str(row.get("standard_term") or "").strip()
        for row in rows
    }
    print(f"{TABLE} 已重建：{len(mapping)} 条")
    for index, (source, standard) in enumerate(mapping.items(), start=1):
        print(f"  {index:>2}. {source} -> {standard}")

    mapper = TermMapper.build(mapping)
    print("\n替换自检：")
    for query in ("房贷能提前还款吗", "提前还贷怎么操作", "lpr 是多少"):
        normalized = normalize_query(query)
        print(f"  {normalized} -> {mapper.apply(normalized)}")


if __name__ == "__main__":
    if sys.path[0] != str(SQL_FILE.parent):
        sys.path.insert(0, str(SQL_FILE.parent))
    asyncio.run(main())
