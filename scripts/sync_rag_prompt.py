"""把 db.sql 中的 rag_prompt 记录同步进 MySQL。

为什么不用 mysql CLI：SQL 初始化脚本里的提示词正文含大量 `\\n` / `\\"` 转义，
CLI 的逐行解析会在长字符串处直接断开（报 `ERROR 1064 ... near '' at line N`），
而且每条记录只是 `( ... )` 的 values 元组，单独截出来跑还缺 `INSERT INTO ... VALUES` 前缀。
这里改成：正则抽元组 → 按 MySQL 转义表反转义 → 绑定参数写入 → 读回逐字符比对。

老库（单列 `prompt_content`）会自动就地迁移成 `system_content` + `user_content`。

用法：
    python scripts/sync_rag_prompt.py            # 先预览差异，确认后写入
    python scripts/sync_rag_prompt.py --yes      # 跳过确认
    python scripts/sync_rag_prompt.py --dry-run  # 只看差异，不写库
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.core.db import get_database  # noqa: E402

DEFAULT_SQL = Path(__file__).resolve().parent.parent / "db.sql"

# MySQL 字符串转义表（NO_BACKSLASH_ESCAPES 关闭时的默认行为）
ESCAPES = {
    "0": "\0",
    "'": "'",
    '"': '"',
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "\\": "\\",
    "%": "\\%",
    "_": "\\_",
}

# 只匹配六列的 rag_prompt 记录（key、name、system、user、status、UNIX_TIMESTAMP()），
# 术语映射的十四列元组不会被误抓
ENTRY_RE = re.compile(
    r"\(\s*'(?P<key>[A-Za-z0-9_]+)'\s*,\s*"
    r"'(?P<name>(?:[^'\\]|\\.)*)'\s*,\s*"
    r"'(?P<system>(?:[^'\\]|\\.)*)'\s*,\s*"
    r"'(?P<user>(?:[^'\\]|\\.)*)'\s*,\s*"
    r"(?P<status>\d+)\s*,\s*UNIX_TIMESTAMP\(\)\s*\)",
    re.S,
)


def unescape(raw: str) -> str:
    out, index = [], 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            out.append(ESCAPES.get(raw[index + 1], raw[index + 1]))
            index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def parse_entries(sql_path: Path) -> list[dict]:
    sql = sql_path.read_text(encoding="utf-8")
    return [
        {
            "key": match.group("key"),
            "name": unescape(match.group("name")),
            "system": unescape(match.group("system")),
            "user": unescape(match.group("user")),
            "status": int(match.group("status")),
        }
        for match in ENTRY_RE.finditer(sql)
    ]


def preview(label: str, item: dict) -> None:
    system, user = item["system"], item["user"]
    print(f"  {item['key']:<20} {item['name']:<14} system {len(system):>5} 字 / user {len(user):>4} 字")
    print(f"    {label}·system 尾部: ...{system[-34:].replace(chr(10), ' / ')}")
    print(f"    {label}·user  正文: {user.replace(chr(10), ' / ')}")


async def ensure_schema(session) -> str | None:
    """老库的 rag_prompt 只有单列 prompt_content，就地迁移成 system + user 两列。

    迁移后老内容会先落到 system_content，紧接着被 db.sql 的记录整行覆写（见 changed 判定）。
    """
    columns = {
        row["Field"]
        for row in (await session.execute(text("SHOW COLUMNS FROM rag_prompt"))).mappings().all()
    }
    if {"system_content", "user_content"} <= columns:
        return None
    if "prompt_content" not in columns:
        raise RuntimeError(f"rag_prompt 表结构无法识别，现有列：{sorted(columns)}")
    await session.execute(
        text(
            "ALTER TABLE rag_prompt "
            "CHANGE COLUMN prompt_content system_content TEXT NOT NULL "
            "COMMENT '系统提示词：固定指令，不含变量', "
            "ADD COLUMN user_content TEXT NOT NULL "
            "COMMENT '用户提示词：含变量占位符' AFTER system_content"
        )
    )
    await session.commit()
    return "rag_prompt：prompt_content → system_content + user_content"


async def run(sql_path: Path, *, dry_run: bool, assume_yes: bool) -> int:
    entries = parse_entries(sql_path)
    if not entries:
        print(f"✗ 未能从 {sql_path} 解析出任何 rag_prompt 记录")
        return 1

    database = get_database()
    if not database.enabled:
        print("✗ DB_ENABLED=false，无法同步。请在 .env 中开启后重试。")
        return 1

    try:
        async with database.session() as session:
            migration = await ensure_schema(session)
            if migration:
                print(f"! 已迁移表结构：{migration}\n")

            rows = (
                await session.execute(
                    text(
                        "SELECT prompt_key, prompt_name, system_content, user_content "
                        "FROM rag_prompt ORDER BY id"
                    )
                )
            ).mappings().all()
            # 一律按列名取值，不用位置下标 —— SELECT 的列顺序调整后下标会静默错位
            current = {
                row["prompt_key"]: (
                    row["prompt_name"],
                    row["system_content"],
                    row["user_content"],
                )
                for row in rows
            }

            print(f"db.sql: {sql_path}")
            print(f"库中现有 {len(current)} 条；db.sql 中 {len(entries)} 条\n")

            changed, unchanged = [], []
            for item in entries:
                existing = current.get(item["key"])
                target = (item["name"], item["system"], item["user"])
                match = existing == target
                (unchanged if match else changed).append(item)
                preview("库中" if match else ("当前" if existing else "新增"), item)

            print()
            for item in changed:
                existing = current.get(item["key"])
                if existing is None:
                    print(f"  + {item['key']}：库中不存在，将新增")
                else:
                    print(
                        f"  ~ {item['key']}：库中 system {len(existing[1])} 字 / user {len(existing[2])} 字"
                        f" → 将覆写为 system {len(item['system'])} 字 / user {len(item['user'])} 字"
                    )
            for item in unchanged:
                print(f"  = {item['key']}：与库中一致，无需改动")

            extra = sorted(set(current) - {item["key"] for item in entries})
            if extra:
                print(f"\n  ! 库中另有 {extra} 未出现在 db.sql 中，保持原样（如需清理请手动处理）")

            if not changed:
                print("\n✓ 已是最新，无需写入")
                return 0
            if dry_run:
                print("\n(dry-run，未写入)")
                return 0

            if not assume_yes:
                answer = input("\n确认写入？[y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("已取消")
                    return 1

            for item in changed:
                await session.execute(
                    text("DELETE FROM rag_prompt WHERE prompt_key = :key"), {"key": item["key"]}
                )
                await session.execute(
                    text(
                        "INSERT INTO rag_prompt "
                        "(prompt_key, prompt_name, system_content, user_content, status, modify_time) "
                        "VALUES (:key, :name, :system, :user, :status, UNIX_TIMESTAMP())"
                    ),
                    item,
                )
            await session.commit()

            # 读回逐字符比对：确认写入过程没有二次转义
            for item in changed:
                stored = (
                    await session.execute(
                        text(
                            "SELECT system_content, user_content FROM rag_prompt "
                            "WHERE prompt_key = :key"
                        ),
                        {"key": item["key"]},
                    )
                ).mappings().one()
                if (stored["system_content"], stored["user_content"]) != (
                    item["system"],
                    item["user"],
                ):
                    print(f"✗ {item['key']} 入库后内容与 db.sql 不一致，请检查转义处理")
                    return 1
        print(f"\n✓ 已同步 {len(changed)} 条，且与 db.sql 逐字符一致")
        print("  提示：服务端有 60s 进程内缓存（DB_CACHE_TTL_SECONDS），重启可立即生效。")
        return 0
    finally:
        await database.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 db.sql 中的 rag_prompt 到 MySQL")
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL, help="SQL 初始化脚本路径")
    parser.add_argument("--dry-run", action="store_true", help="只预览差异，不写库")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()
    return asyncio.run(run(args.sql, dry_run=args.dry_run, assume_yes=args.yes))


if __name__ == "__main__":
    raise SystemExit(main())
