"""日志：按天切分落文件，并给每个请求一个贯穿全链路的唯一标识。

`configure_logging()` 在 `app/main.py` 启动时调用一次；每个请求进来时由中间件
`bind_trace_id()` 定下标识（沿用上游的 `X-Request-ID`，没有就新生成），此后同一请求内的
**所有**日志都自动带上它 —— 各关键节点的日志因此不必自己拼标识，直接拿它 grep 就能把
一次问答的完整链路捞出来。

跑起来后日志落在 `<LOG_DIR>/app.log`，跨零点自动改名归档成 `app.log.<昨天日期>`，
`LOG_BACKUP_DAYS` 天前的归档自动清理。
"""

import logging
import time
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# 非请求上下文（启动、脚本、单测）没有标识，统一显示 "-"
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(trace_id)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _TraceIdFilter(logging.Filter):
    """把当前请求的标识塞进每条日志：Formatter 里的 `%(trace_id)s` 靠它取值。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


def new_trace_id() -> str:
    """新生成一个请求唯一标识。"""
    return uuid.uuid4().hex[:16]


def bind_trace_id(trace_id: str) -> str:
    """把标识绑到当前请求上，返回它本身（省得调用方再赋一次值）。"""
    _trace_id.set(trace_id)
    return trace_id


def elapsed_ms(started: float) -> float:
    """距 `started`（`time.perf_counter()` 的返回值）的毫秒数。

    关键节点日志都要记耗时，换算写在这里，免得每个节点各写一遍 `(perf_counter() - started) * 1000`。
    """
    return (time.perf_counter() - started) * 1000


def configure_logging(log_dir: str, level: str, backup_days: int) -> None:
    """按天切分的文件日志 + 控制台输出；重复调用不会叠加 handler。"""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler = TimedRotatingFileHandler(
        directory / "app.log", when="midnight", backupCount=backup_days, encoding="utf-8"
    )
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        handler.addFilter(_TraceIdFilter())

    root = logging.getLogger()
    root.handlers = [file_handler, stream_handler]
    root.setLevel(level.upper())
    # uvicorn 默认给这几个 logger 各挂了 handler，会截住日志不进 root；清掉交回 root，
    # 这样访问日志与启动日志也一起按天落文件
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
