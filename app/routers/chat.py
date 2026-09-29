import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.logging import elapsed_ms
from app.models.chat import ChatCompletionRequest
from app.services.rag_agent import RagAgent
from app.services.sse import sse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["知识问答"])


@router.post("/chat/completions", summary="流式知识问答")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> StreamingResponse:
    # 请求参数整包记一条 JSON（`ensure_ascii=False` 保中文可读）。模型序列化必然是单行，
    # 消息里带换行也不会把日志撑成两行 —— 比手工拼字段更全，拿去 jq 也能直接解析
    logger.info("问答请求 %s", payload.model_dump_json(ensure_ascii=False))

    async def event_generator() -> AsyncIterator[str]:
        started = time.perf_counter()
        try:
            async for event in RagAgent(get_settings()).stream(payload):
                if await request.is_disconnected():
                    logger.info("调用方已断开连接")
                    return
                yield event
        except Exception as exc:
            logger.exception("知识问答失败")
            yield sse("error", {"message": "知识问答服务暂时不可用", "detail": str(exc)})
            yield sse("done", {"answer_type": "error"})
        finally:
            # 流式响应在中间件那层只等到「响应头就绪」，整条问答的真实总耗时记在这里
            logger.info("问答结束 总耗时=%.0fms", elapsed_ms(started))

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })
