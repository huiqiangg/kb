from pathlib import Path
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.db import get_database
from app.core.logging import bind_trace_id, configure_logging, elapsed_ms, new_trace_id
from app.routers.chat import router as chat_router

settings = get_settings()
configure_logging(settings.log_dir, settings.log_level, settings.log_backup_days)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    # 仅在用到过连接池时才会真正释放
    await get_database().dispose()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    # 请求一进来就定下唯一标识：同一请求内所有日志（各关键节点、其他模块的 warning）
    # 都自动带上它；上游带了 X-Request-ID 就沿用，便于跨服务串成一条链路
    trace_id = bind_trace_id(request.headers.get("X-Request-ID") or new_trace_id())
    client = request.client.host if request.client else "-"
    started = time.perf_counter()
    logger.info("收到请求 method=%s path=%s client=%s", request.method, request.url.path, client)
    response = await call_next(request)
    response.headers["X-Request-ID"] = trace_id
    logger.info("请求处理完成 status=%s 耗时=%.0fms", response.status_code, elapsed_ms(started))
    return response


@app.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
