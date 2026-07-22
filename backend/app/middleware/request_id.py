from collections.abc import Awaitable, Callable
from contextvars import ContextVar
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response

from app.observability.tracing import (
    SPAN_ID_HEADER,
    TRACE_ID_HEADER,
    TRACEPARENT_HEADER,
    current_span_id,
    current_trace_id,
    current_traceparent,
    reset_trace_context,
    set_trace_context,
)


REQUEST_ID_HEADER = "X-Request-ID"
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("app.request")


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = _normalize_request_id(request.headers.get(REQUEST_ID_HEADER)) or uuid4().hex
    token = _request_id.set(request_id)
    trace_tokens = set_trace_context(request.headers.get(TRACEPARENT_HEADER))
    request.state.request_id = request_id
    request.state.trace_id = current_trace_id()
    request.state.span_id = current_span_id()
    started_at = perf_counter()
    status_code = 500
    response: Response | None = None
    error: BaseException | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except BaseException as exc:
        error = exc
    finally:
        latency_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "request_id": request_id,
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
            },
        )
    if response is None:
        reset_trace_context(trace_tokens)
        _request_id.reset(token)
        if error:
            raise error
        raise RuntimeError("Request middleware did not receive a response.")

    response.headers[REQUEST_ID_HEADER] = request_id
    if traceparent := current_traceparent():
        response.headers[TRACEPARENT_HEADER] = traceparent
    if trace_id := current_trace_id():
        response.headers[TRACE_ID_HEADER] = trace_id
    if span_id := current_span_id():
        response.headers[SPAN_ID_HEADER] = span_id
    reset_trace_context(trace_tokens)
    _request_id.reset(token)
    return response


def current_request_id() -> str | None:
    return _request_id.get()


def _normalize_request_id(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized[:128]
