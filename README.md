# 农商行知识问答 Agent

基于 FastAPI、LangChain 1.2 和 LLMOps 知识库的流式 RAG 服务。接口为 `POST /api/kb/chat/completions`，根路径提供简单的调试网页，Swagger 位于 `/docs`。

## 启动

```bash
cp .env.example .env
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn app.main:app --reload
```

必须配置 `KB_PLATFORM_BASE_URL`、`KB_PROJECT_ID` 和银行环境需要的鉴权信息。模型密钥请放入 `MODEL_ACCESS_KEY`，不要提交 `.env`。

## 请求示例

```json
{
  "messages": [{"role": "user", "content": "个人贷款提前还款怎么办理？"}],
  "ranges": [{"knowledge_base_id": "37c5dwtg4ufbw332", "doc_range": ["ds3523"]}]
}
```

响应为 SSE：`status` 表示处理阶段、`token` 为最终答案增量、`sources` 为检索来源、`done` 表示结束。FAQ 高置信命中会直接返回答案。
