from __future__ import annotations

from contextvars import ContextVar
import logging
import re
from secrets import token_hex

from fastapi import FastAPI

from app.config.settings import settings


TRACEPARENT_HEADER = "traceparent"
TRACE_ID_HEADER = "X-Trace-ID"
SPAN_ID_HEADER = "X-Span-ID"
_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_traceparent: ContextVar[str | None] = ContextVar("traceparent", default=None)
logger = logging.getLogger("app.observability")


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_span_id() -> str | None:
    return _span_id.get()


def current_traceparent() -> str | None:
    return _traceparent.get()


def current_trace_fields() -> dict[str, str]:
    fields = {}
    if trace_id := current_trace_id():
        fields["trace_id"] = trace_id
    if span_id := current_span_id():
        fields["span_id"] = span_id
    return fields


def set_trace_context(traceparent_header: str | None) -> tuple[object, object, object]:
    traceparent = normalize_traceparent(traceparent_header) or new_traceparent()
    parsed = parse_traceparent(traceparent)
    trace_id = parsed["trace_id"]
    span_id = parsed["span_id"]
    return (
        _trace_id.set(trace_id),
        _span_id.set(span_id),
        _traceparent.set(traceparent),
    )


def reset_trace_context(tokens: tuple[object, object, object]) -> None:
    trace_token, span_token, traceparent_token = tokens
    _trace_id.reset(trace_token)
    _span_id.reset(span_token)
    _traceparent.reset(traceparent_token)


def normalize_traceparent(value: str | None) -> str | None:
    if not value:
        return None
    traceparent = value.strip().lower()
    parsed = parse_traceparent(traceparent)
    if not parsed:
        return None
    if parsed["trace_id"] == "0" * 32 or parsed["span_id"] == "0" * 16:
        return None
    return traceparent


def parse_traceparent(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    match = _TRACEPARENT_RE.match(value)
    if not match:
        return {}
    return match.groupdict()


def new_traceparent() -> str:
    return f"00-{token_hex(16)}-{token_hex(8)}-01"


def configure_opentelemetry(app: FastAPI) -> None:
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("opentelemetry_not_available", extra={"error_code": type(exc).__name__})
        return

    resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    exporter_kwargs = {}
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        exporter_kwargs["endpoint"] = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
