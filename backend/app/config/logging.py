import json
import logging
from datetime import UTC, datetime
from typing import Any

from app.config.settings import settings
from app.middleware.request_id import current_request_id
from app.observability.tracing import current_span_id, current_trace_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or current_request_id()
        if request_id:
            payload["request_id"] = request_id
        trace_id = getattr(record, "trace_id", None) or current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        span_id = getattr(record, "span_id", None) or current_span_id()
        if span_id:
            payload["span_id"] = span_id

        for field in ("method", "path", "status_code", "latency_ms", "error_code"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        if not hasattr(record, "trace_id"):
            record.trace_id = current_trace_id()
        if not hasattr(record, "span_id"):
            record.span_id = current_span_id()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    logging.basicConfig(level=settings.LOG_LEVEL.upper(), handlers=[handler])
