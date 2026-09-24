import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.models.chat import ChatCompletionRequest
from app.services.rag_agent import RagAgent
from app.services.sse import sse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["知识问答"])


@router.post("/chat/completions", summary="流式知识问答")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
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

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })
