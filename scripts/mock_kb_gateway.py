"""本地 mock 召回网关：给 `RagAgent._faq_probe` / `KnowledgeBaseClient.mix_retrieve`
喂假数据，用来联调 `POST /api/kb/chat/completions`，**不需要改一行生产代码**。

零依赖（标准库 http.server）。启动后把召回地址指过来即可，`_faq_probe` 收到的就是这份数据：

    KB_FAQ_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve
    KB_MIX_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve

两条通道按请求体的 `keywords` 区分：**空 = FAQ 探测**（返回 QA 命中）、
**非空 = 最终答案检索**（返回切片命中）。FAQ 作答那条分数默认 0.99
（≥ `FAQ_SIMILARITY_THRESHOLD` 的 0.98）→ 触发直返 + 三条相近问题。

假数据是一套**个人住房贷款**标准问答库（`FAQ_LIBRARY`，6 条），按 query / keywords 里
出现的**触发词**选中一条作答，其余条目按分数递减充当「其他相近问题」候选 ——
换个问法就能命中不同条目，不必改代码：

    房贷能提前还款吗          -> 个人住房贷款可以提前还款吗
    房贷利率多久调一次        -> 房贷利率多久调整一次
    等额本息和等额本金区别    -> 等额本息和等额本金有什么区别
    公积金能贷多少            -> 公积金贷款额度怎么计算
    房贷逾期会影响征信吗      -> 房贷逾期还款会影响征信吗
    审批通过多久放款          -> 房贷审批通过后多久放款

可调环境变量：
    MOCK_PORT=8099          监听端口
    MOCK_CHANNEL=auto       返回哪条通道的数据：auto（默认，按 keywords 判断）/ faq / chunk
    MOCK_FAQ_SCORE=0.99     作答那条的分数；设成 0.5 可测「未达阈值 → 走完整链路」
    MOCK_RELATED_COUNT=3    QA 库里候选相近问题的条数；设成 2 可测「凑不满三条 → 一条都不发」

> `auto` 的判据是「请求体 `keywords` 是否为空」，两种通道的请求体形状本来完全一样，
> 只有 keywords 可能不同。所以**改写失败**（模型不可达 → keywords 为空）时，
> 最终检索也会被认成 FAQ 探测。要精确验证某一条通道就显式设 `MOCK_CHANNEL`。

跑法：.venv/bin/python scripts/mock_kb_gateway.py
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = os.getenv("MOCK_HOST", "127.0.0.1")
PORT = int(os.getenv("MOCK_PORT", "8099"))
CHANNEL = os.getenv("MOCK_CHANNEL", "auto").strip().lower()
FAQ_SCORE = float(os.getenv("MOCK_FAQ_SCORE", "0.99"))
RELATED_COUNT = int(os.getenv("MOCK_RELATED_COUNT", "3"))

FAQ_KB_ID = "kb-qa-001"
FAQ_DOC_NAME = "个人住房贷款标准问答库"

# 房贷标准问答库：(触发词, 问题, 答案)。`_pick_qa` 按「触发词是否出现在 query/keywords 里」
# 选中一条作答，**列表顺序即匹配优先级**（宽泛的词往后放）；第 0 条兼作兜底。
FAQ_LIBRARY: list[tuple[list[str], str, str]] = [
    (
        ["提前还款", "提前还", "提前结清", "提前还贷"],
        "个人住房贷款可以提前还款吗",
        "可以。可通过手机银行「我的贷款」或经办行柜面提交提前还款申请，一般需提前 3 个工作日预约；"
        "每年可免费办理的次数与违约金计收规则以贷款合同约定为准。",
    ),
    (
        ["等额本息", "等额本金", "还款方式"],
        "等额本息和等额本金有什么区别",
        "等额本息每月还款额固定，前期利息占比高、总利息较多；等额本金每月归还本金固定、月供逐月递减，"
        "总利息更少但前期还款压力较大。合同期内一般可申请变更一次还款方式。",
    ),
    (
        ["公积金", "贷款额度", "能贷多少"],
        "公积金贷款额度怎么计算",
        "额度与缴存年限、账户余额、房屋总价及当地公积金政策相关，一般取「账户余额的若干倍」"
        "与「房价规定比例」中的较低者，最终以公积金中心核定为准。",
    ),
    (
        ["逾期", "征信", "断供", "晚还"],
        "房贷逾期还款会影响征信吗",
        "会产生逾期记录并报送金融信用信息基础数据库。是否影响后续贷款审批取决于逾期天数与累计次数，"
        "发现后应尽快补足欠款并联系经办行说明情况。",
    ),
    (
        ["放款", "审批", "到账"],
        "房贷审批通过后多久放款",
        "抵押登记办妥后一般 3 至 5 个工作日放款；若遇放款额度紧张或资料需补充，时间会相应延长，"
        "具体以经办行通知为准。",
    ),
    (
        ["利率", "lpr", "重定价", "月供变了"],
        "房贷利率多久调整一次",
        "采用 LPR 浮动利率的个人住房贷款，通常每年 1 月 1 日按最新一期 LPR 重新定价，"
        "也可按合同约定的贷款发放日对月对日调整；固定利率贷款在合同期内不调整。",
    ),
]

# 最终答案检索用的切片命中（房贷主题）
CHUNKS = [
    (
        "个人住房贷款提前还款办理说明",
        "借款人可申请提前归还个人住房贷款本金，需提前 3 个工作日向经办行预约；"
        "贷款发放满一年后提前还款的免收违约金，不满一年的按剩余本金 1% 计收。",
        0.72,
    ),
    (
        "存量个人住房贷款利率调整规则",
        "浮动利率贷款每年重定价一次，重定价日为每年 1 月 1 日或贷款发放日对月对日，"
        "按重定价日前最新一期 LPR 加（减）合同约定基点执行。",
        0.68,
    ),
    (
        "个人住房贷款还款方式说明",
        "等额本息与等额本金可在合同期内申请变更一次，变更后剩余期限按新方式重新计算月供。",
        0.61,
    ),
]


def _pick_qa(query: str, keywords: list[str]) -> int:
    """按触发词选中一条 QA 的下标；一条都不命中时用第 0 条兜底。"""
    text = f"{query} {' '.join(keywords)}".lower()
    for index, (triggers, _, _) in enumerate(FAQ_LIBRARY):
        if any(trigger.lower() in text for trigger in triggers):
            return index
    return 0


def _qa_doc(index: int, doc_id: str, score: float) -> dict:
    """把 `FAQ_LIBRARY` 的一条包成平台响应形状。

    `_to_hit` 会优先取 `chunk.qa_pairs[0].answer` 作为正文、`question` 作为 `RetrievalHit.question`，
    所以 `chunk.content` 只是占位，不会被用到。
    """
    _, question, answer = FAQ_LIBRARY[index]
    return {
        "knowledge_base_id": FAQ_KB_ID,
        "doc_id": doc_id,
        "doc_name": FAQ_DOC_NAME,
        "score": score,
        "chunk": {
            "id": f"qa-{index + 1:03d}",
            "content": "（切片正文占位，命中 qa_pairs 时不会被使用）",
            "qa_pairs": [{"question": question, "answer": answer}],
        },
    }


def faq_result(query: str, keywords: list[str]) -> list[dict]:
    """FAQ 探测响应：1 条高分作答 + `RELATED_COUNT` 条相近问题候选。

    候选分数从作答那条往下递减（0.97 / 0.95 / …），保证 `_related_queries` 取到的
    「其他相近问题」不会盖过作答项；把 `MOCK_FAQ_SCORE` 调低时它们也一起低下去。
    """
    matched_index = _pick_qa(query, keywords)
    others = [index for index in range(len(FAQ_LIBRARY)) if index != matched_index]
    related = [
        _qa_doc(other, f"doc-qa-{order + 2:03d}", round(FAQ_SCORE - 0.02 * (order + 1), 2))
        for order, other in enumerate(others[:RELATED_COUNT])
    ]
    return [_qa_doc(matched_index, "doc-qa-001", FAQ_SCORE), *related]


def chunk_result() -> list[dict]:
    """最终答案检索响应：切片命中。"""
    return [
        {
            "knowledge_base_id": "kb-chunk-001",
            "doc_id": f"doc-chunk-{index + 1:03d}",
            "doc_name": title,
            "score": score,
            "chunk": {"id": f"chunk-{index + 1:03d}", "content": content},
        }
        for index, (title, content, score) in enumerate(CHUNKS)
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # 探活用
        self._send({"status": "ok", "service": "mock-kb-gateway"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        parsed = urlparse(self.path)
        query = body.get("query") or ""
        keywords = body.get("keywords") or []
        use_chunks = CHANNEL == "chunk" or (CHANNEL == "auto" and bool(keywords))
        channel = "最终检索" if use_chunks else "FAQ 探测"
        result = chunk_result() if use_chunks else faq_result(query, keywords)

        summary = f"返回 {len(result)} 条（首条 score={result[0]['score']}"
        if not use_chunks:
            summary += f"，作答={result[0]['chunk']['qa_pairs'][0]['question']}"
        print(
            f"\n[{channel}] POST {parsed.path}"
            f"?{parsed.query}\n"
            f"    query={query!r} keywords={keywords}\n"
            f"    ranges={body.get('ranges')}\n"
            f"    -> {summary}）",
            flush=True,
        )
        self._send({"result": result})

    def _send(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:  # 默认会往 stderr 打，用不上
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    path = "/applet/api/v1/knowlhub/kbs:mix-retrieve"
    print(f"mock 召回网关已启动：http://{HOST}:{PORT}{path}")
    print(f"  通道={CHANNEL}  FAQ 作答分={FAQ_SCORE}  相近问题候选={RELATED_COUNT} 条")
    print(f"  房贷问答库 {len(FAQ_LIBRARY)} 条，按触发词命中：")
    for _, question, _ in FAQ_LIBRARY:
        print(f"    - {question}")
    print("  把下面两行作为环境变量启动主服务即可：")
    print(f"    KB_FAQ_RETRIEVE_URL=http://{HOST}:{PORT}{path}")
    print(f"    KB_MIX_RETRIEVE_URL=http://{HOST}:{PORT}{path}")
    print("Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
