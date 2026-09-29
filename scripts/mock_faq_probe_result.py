"""直接可用的 `FaqProbeResult` 假数据（个人住房贷款）—— 拿返回值直接用，不走网络。

与 `scripts/mock_kb_gateway.py` 是**同一套房贷数据**（6 条 QA），只是交付形态不同：
网关那条路要起服务、把两个召回地址指过去、让 `_faq_probe` 真发一次请求；这里把
`FaqProbeResult` 直接摆出来，替掉返回值就能跑通整条 FAQ 直返链路。

三种取用方式：

1) 替掉探测（推荐，路由与 SSE 序列跟真实链路完全一致，零网络）：

    from scripts.mock_faq_probe_result import fake_faq_probe

    RagAgent._faq_probe = fake_faq_probe        # 签名与真实方法一致（async）

2) 直接取现成返回值（key 是意图名）：

    from scripts.mock_faq_probe_result import FAQ_PROBE_RESULTS

    result = FAQ_PROBE_RESULTS["提前还款"]

3) 按需拼一条（三个入参对应三种要测的场景）：

    build("提前还款")                       # 命中 + 3 条相近问题
    build("提前还款", related_count=2)       # 凑不满 3 条 → related_queries 为空列表
    build("提前还款", score=0.5)             # 未达阈值 → 不直返
    build(None)                             # 未命中 → 不直返

字段口径与生产链路一致：`hit.content` 是库里的标准答案（`_faq_direct` 拿它去润色、
失败回落原文），`hit.question` 是命中的那条问题原文，`related_queries` 是**其他**
相近问题（排除作答那条、去重、取满 3 条否则给空列表）。

跑一下看全貌，含可直接粘贴的 Python 字面量：

    PYTHONPATH=. .venv/bin/python scripts/mock_faq_probe_result.py
"""

from app.core.config import Settings
from app.services.knowledge_base import RetrievalHit
from app.services.rag_agent import FAQ_RELATED_QUERY_COUNT, FaqProbeResult

FAQ_KB_ID = "kb-qa-001"
FAQ_DOC_NAME = "个人住房贷款标准问答库"
# 作答那条的分数：要 ≥ FAQ_SIMILARITY_THRESHOLD（默认 0.98）才会被 `_faq_match` 采纳
FAQ_SCORE = 0.99
# 相近问题的分数从作答那条往下递减，保证排序上不会盖过作答项
RELATED_STEP = 0.02
FAQ_THRESHOLD = Settings().faq_similarity_threshold

# 房贷标准问答库：（意图名, 触发词, 问题原文, 标准答案）。
# 触发词按「是否出现在 query 里」匹配，**列表顺序即优先级**（宽泛的词往后放）；
# 一条都不命中时用第 0 条（提前还款）兜底。
FAQ_LIBRARY: list[tuple[str, list[str], str, str]] = [
    (
        "提前还款",
        ["提前还款", "提前还", "提前结清", "提前还贷"],
        "个人住房贷款可以提前还款吗",
        "可以。可通过手机银行「我的贷款」或经办行柜面提交提前还款申请，一般需提前 3 个工作日"
        "预约；每年可免费办理的次数与违约金计收规则以贷款合同约定为准。",
    ),
    (
        "还款方式",
        ["等额本息", "等额本金", "还款方式"],
        "等额本息和等额本金有什么区别",
        "等额本息每月还款额固定，前期利息占比高、总利息较多；等额本金每月归还本金固定、月供"
        "逐月递减，总利息更少但前期还款压力较大。合同期内一般可申请变更一次还款方式。",
    ),
    (
        "公积金额度",
        ["公积金", "贷款额度", "能贷多少"],
        "公积金贷款额度怎么计算",
        "额度与缴存年限、账户余额、房屋总价及当地公积金政策相关，一般取「账户余额的若干倍」"
        "与「房价规定比例」中的较低者，最终以公积金中心核定为准。",
    ),
    (
        "逾期征信",
        ["逾期", "征信", "断供", "晚还"],
        "房贷逾期还款会影响征信吗",
        "会产生逾期记录并报送金融信用信息基础数据库。是否影响后续贷款审批取决于逾期天数与"
        "累计次数，发现后应尽快补足欠款并联系经办行说明情况。",
    ),
    (
        "放款时间",
        ["放款", "审批", "到账"],
        "房贷审批通过后多久放款",
        "抵押登记办妥后一般 3 至 5 个工作日放款；若遇放款额度紧张或资料需补充，时间会相应"
        "延长，具体以经办行通知为准。",
    ),
    (
        "利率调整",
        ["利率", "lpr", "重定价", "月供变了"],
        "房贷利率多久调整一次",
        "采用 LPR 浮动利率的个人住房贷款，通常每年 1 月 1 日按最新一期 LPR 重新定价，也可按"
        "合同约定的贷款发放日对月对日调整；固定利率贷款在合同期内不调整。",
    ),
]


def _index_of(text: str) -> int:
    """按触发词选中一条 QA 的下标；一条都不命中就用第 0 条兜底。"""
    lowered = text.lower()
    for index, (_, triggers, _, _) in enumerate(FAQ_LIBRARY):
        if any(trigger.lower() in lowered for trigger in triggers):
            return index
    return 0


def _index_of_name(name: str) -> int:
    for index, (intent, _, _, _) in enumerate(FAQ_LIBRARY):
        if name == intent:
            return index
    raise KeyError(f"没有叫 {name!r} 的意图，可选：{[item[0] for item in FAQ_LIBRARY]}")


def _qa_hit(index: int, score: float) -> RetrievalHit:
    """按 `_to_hit` 解析后的形状拼一条 QA 命中（字段与平台返回的一致）。"""
    _, _, question, answer = FAQ_LIBRARY[index]
    return RetrievalHit(
        content=answer,
        score=score,
        knowledge_base_id=FAQ_KB_ID,
        doc_id=f"doc-qa-{index + 1:03d}",
        doc_name=FAQ_DOC_NAME,
        chunk_id=f"qa-{index + 1:03d}",
        question=question,
    )


def build(
    name: str | None = "提前还款",
    score: float = FAQ_SCORE,
    related_count: int = FAQ_RELATED_QUERY_COUNT,
) -> FaqProbeResult:
    """按意图名拼一个 `FaqProbeResult`；`name=None` 或未达阈值都返回「不直返」的空结果。

    阈值判据与 `_faq_match` 相同（`score >= faq_similarity_threshold`），所以
    `build(score=0.5)` 拿到的空结果就是生产里真实的「没命中」形状。
    """
    if name is None or score < FAQ_THRESHOLD:
        return FaqProbeResult()
    matched_index = _index_of_name(name)
    others = [index for index in range(len(FAQ_LIBRARY)) if index != matched_index]
    related = [FAQ_LIBRARY[index][2] for index in others[:related_count]]
    if len(related) < FAQ_RELATED_QUERY_COUNT:  # 凑不满三条就一条都不给
        related = []
    return FaqProbeResult(hit=_qa_hit(matched_index, score), related_queries=related)


def faq_probe_result(
    query: str, score: float = FAQ_SCORE, related_count: int = FAQ_RELATED_QUERY_COUNT
) -> FaqProbeResult:
    """按 query 里的触发词拼结果 —— 端到端调试与 `fake_faq_probe` 都走这里。"""
    return build(FAQ_LIBRARY[_index_of(query)][0], score=score, related_count=related_count)


async def fake_faq_probe(self, kb, first: str, second: str, ranges) -> FaqProbeResult:
    """替掉 `RagAgent._faq_probe`：签名一致（async、5 个参数），按原问题命中。

        RagAgent._faq_probe = fake_faq_probe

    `kb` / `second` / `ranges` 都收下但不用 —— 假数据没必要真发请求，也不看
    `knowledgeType`（真实实现只探 `knowledgeType=2` 的标准问答库）。
    """
    return faq_probe_result(first)


# 六条现成返回值，key 是意图名；import 时就已经拼好，取来即可用
FAQ_PROBE_RESULTS: dict[str, FaqProbeResult] = {item[0]: build(item[0]) for item in FAQ_LIBRARY}


def _as_source(result: FaqProbeResult) -> str:
    """把结果打印成可直接粘贴的 Python 字面量。"""
    if result.hit is None:
        return "FaqProbeResult()"
    hit = result.hit
    fields = (
        ("content", hit.content),
        ("score", hit.score),
        ("knowledge_base_id", hit.knowledge_base_id),
        ("doc_id", hit.doc_id),
        ("doc_name", hit.doc_name),
        ("chunk_id", hit.chunk_id),
        ("question", hit.question),
    )
    lines = ["FaqProbeResult(", "    hit=RetrievalHit("]
    lines += [f"        {name}={value!r}," for name, value in fields]
    lines += ["    ),", "    related_queries=["]
    lines += [f"        {question!r}," for question in result.related_queries]
    return "\n".join([*lines, "    ],", ")"])


def main() -> None:
    threshold = FAQ_THRESHOLD
    step = RELATED_STEP
    print(f"房贷标准问答库 {len(FAQ_LIBRARY)} 条，阈值 {threshold}，作答分 {FAQ_SCORE}（差 {step}）\n")
    for name, result in FAQ_PROBE_RESULTS.items():
        hit = result.hit
        print(f"[{name}] {hit.question}  score={hit.score}")
        print(f"  命中答案：{hit.content}")
        print(f"  相近问题：{result.related_queries}")
        print()
    print("=== 不直返的三种情形（都是空 FaqProbeResult）===")
    print(f"  未命中            build(None)                    -> {build(None)}")
    print(f"  未达阈值          build('提前还款', score=0.5)     -> {build('提前还款', score=0.5)}")
    empty = build("提前还款", related_count=2)
    print(f"  凑不满三条        build('提前还款', related_count=2) -> 相近问题 {empty.related_queries}")
    print()
    print("=== 可直接复制的返回值（以「提前还款」为例）===")
    print(_as_source(FAQ_PROBE_RESULTS["提前还款"]))
    print()
    print("=== 按 query 取（触发词命中哪条就给哪条）===")
    for query in ("房贷能提前还款吗", "公积金能贷多少", "房贷逾期会影响征信吗"):
        hit = faq_probe_result(query).hit
        print(f"  {query}  ->  {hit.question}")


if __name__ == "__main__":
    main()
