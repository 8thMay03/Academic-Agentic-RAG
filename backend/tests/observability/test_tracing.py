from app.observability.tracing import normalize_traceparent, parse_traceparent


def test_normalize_traceparent_accepts_valid_w3c_header() -> None:
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    assert normalize_traceparent(traceparent.upper()) == traceparent
    assert parse_traceparent(traceparent)["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert parse_traceparent(traceparent)["span_id"] == "00f067aa0ba902b7"


def test_normalize_traceparent_rejects_invalid_or_zero_ids() -> None:
    assert normalize_traceparent("not-a-traceparent") is None
    assert normalize_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01") is None
    assert normalize_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01") is None
