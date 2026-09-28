"""查询归一化与银行术语映射。

术语映射来自数据库 rag_keywords_mapping，由 TermMappingRepository 装配成 TermMapper，
在 TTL 缓存周期内只编译一次正则（不再硬编码在代码里）。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass


def normalize_query(query: str) -> str:
    """NFKC 全半角归一化并折叠空白。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query)).strip()


@dataclass(frozen=True, slots=True)
class TermMapper:
    """口语术语 -> 银行标准术语 替换器。

    采用「最长优先 + 单次扫描」：re.sub 不会重扫替换产生的文本，因此不会出现
    「房贷利率」先被替成「个人住房贷款执行利率」、其中「住房贷款」又被二次替换成
    「个人个人住房贷款执行利率」这类级联问题。
    """

    mapping: Mapping[str, str]
    pattern: re.Pattern[str] | None
    lookup: Mapping[str, str]

    @classmethod
    def build(cls, mapping: Mapping[str, str]) -> "TermMapper":
        cleaned: dict[str, str] = {}
        for source, standard in mapping.items():
            source_term = str(source or "").strip()
            standard_term = str(standard or "").strip()
            if source_term and standard_term:
                cleaned[source_term] = standard_term
        if not cleaned:
            return cls(mapping={}, pattern=None, lookup={})
        # 长术语优先，避免短术语在重叠位置抢先匹配
        terms = sorted(cleaned, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
        # 大小写不敏感匹配后需回查标准术语，保证 LPR / lpr 都能命中
        lookup = {term.lower(): standard for term, standard in cleaned.items()}
        return cls(mapping=cleaned, pattern=pattern, lookup=lookup)

    def apply(self, query: str) -> str:
        if not query or self.pattern is None:
            return query
        return self.pattern.sub(
            lambda match: self.lookup.get(match.group(0).lower(), match.group(0)), query
        )


EMPTY_TERM_MAPPER = TermMapper.build({})
