import re
import unicodedata


# 可由业务方持续扩充；key 使用标准化后的口语表达。
BUSINESS_TERMS = {
    "存钱": "存款",
    "取钱": "取款",
    "办卡": "银行卡开立",
    "贷款还不上": "贷款逾期还款",
    "提前还贷款": "贷款提前还款",
}


def normalize_query(query: str) -> str:
    """NFKC 全半角归一化并折叠空白。"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query)).strip()


def map_business_terms(query: str) -> str:
    mapped = query
    for spoken, standard in BUSINESS_TERMS.items():
        mapped = mapped.replace(spoken, standard)
    return mapped
