import json
import logging

from app.config.logging import JsonFormatter


def test_json_formatter_includes_request_fields() -> None:
    record = logging.LogRecord(
        name="app.request",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-1"
    record.method = "POST"
    record.path = "/api/v1/chat/stream"
    record.status_code = 200
    record.latency_ms = 12.5
    record.trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    record.span_id = "00f067aa0ba902b7"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.request"
    assert payload["message"] == "request_completed"
    assert payload["request_id"] == "req-1"
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/v1/chat/stream"
    assert payload["status_code"] == 200
    assert payload["latency_ms"] == 12.5
    assert payload["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert payload["span_id"] == "00f067aa0ba902b7"
    assert "timestamp" in payload
