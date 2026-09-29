"""本地 mock 召回网关：给 `RagAgent._faq_probe` / `KnowledgeBaseClient.mix_retrieve`
喂假数据，用来联调 `POST /api/kb/chat/completions`，**不需要改一行生产代码**。

零依赖（标准库 http.server）。启动后把召回地址指过来即可，`_faq_probe` 收到的就是这份数据：

    KB_FAQ_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve
    KB_MIX_RETRIEVE_URL=http://127.0.0.1:8099/applet/api/v1/knowlhub/kbs:mix-retrieve

两条通道按请求体的 `keywords` 区分：**空 = FAQ 探测**（返回 QA 命中）、
**非空 = 最终答案检索**（返回切片命中）。FAQ 作答那条分数默认 0.99
（≥ `FAQ_SIMILARITY_THRESHOLD` 的 0.98）→ 触发直返 + 三条相近问题。

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

# 作答的那条 QA（`_to_hit` 会优先取 qa_pairs[0].answer 作为 content，
# qa_pairs[0].question 作为 `RetrievalHit.question`）
FAQ_ANSWER = (
    "个人住房贷款可以提前还款。可通过手机银行「我的贷款」或经办行柜面提交提前还款申请，"
    "一般需提前 3 个工作日预约，具体以贷款合同约定为准。"
)
FAQ_MATCHED_QUESTION = "个人住房贷款怎么提前还款"

# QA 库里其他候选：分数低于作答那条，故会被 `_related_queries` 依次取用
FAQ_RELATED = [
    ("提前还款需要预约吗", "需要，一般提前 3 个工作日预约。"),
    ("提前还款要收违约金吗", "以合同约定为准，部分产品满一年后免收。"),
    ("线上能办提前还款吗", "可以在手机银行「我的贷款」里办理。"),
]

# 最终答案检索用的切片命中
CHUNKS = [
    (
        "个人住房贷款提前还款办理说明",
        "借款人可申请提前归还个人住房贷款本金，需提前 3 个工作日向经办行预约办理。",
        0.72,
    ),
    (
        "提前还款违约金计收规则",
        "贷款发放满一年后提前还款的免收违约金；不满一年的按剩余本金 1% 计收。",
        0.66,
    ),
]


def faq_result() -> list[dict]:
    """FAQ 探测响应：一条高分作答 + 若干条相近问题候选。"""
    matched = {
        "knowledge_base_id": "kb-qa-001",
        "doc_id": "doc-qa-001",
        "doc_name": "个人贷款 FAQ",
        "score": FAQ_SCORE,
        "chunk": {
            "id": "qa-001",
            "content": "（切片正文占位，命中 qa_pairs 时不会被使用）",
            "qa_pairs": [
                {"question": FAQ_MATCHED_QUESTION, "answer": FAQ_ANSWER},
            ],
        },
    }
    related = [
        {
            "knowledge_base_id": "kb-qa-001",
            "doc_id": f"doc-qa-{index + 2:03d}",
            "doc_name": "个人贷款 FAQ",
            "score": round(0.87 - index * 0.02, 2),
            "chunk": {
                "id": f"qa-{index + 2:03d}",
                "content": answer,
                "qa_pairs": [{"question": question, "answer": answer}],
            },
        }
        for index, (question, answer) in enumerate(FAQ_RELATED[:RELATED_COUNT])
    ]
    return [matched, *related]


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
        keywords = body.get("keywords") or []
        use_chunks = CHANNEL == "chunk" or (CHANNEL == "auto" and bool(keywords))
        channel = "最终检索" if use_chunks else "FAQ 探测"
        result = chunk_result() if use_chunks else faq_result()

        print(
            f"\n[{channel}] POST {parsed.path}"
            f"?{parsed.query}\n"
            f"    query={body.get('query')!r} keywords={keywords}\n"
            f"    ranges={body.get('ranges')}\n"
            f"    -> 返回 {len(result)} 条命中（首条 score={result[0]['score']}）",
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
