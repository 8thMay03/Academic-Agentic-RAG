from fastapi.testclient import TestClient

from app.api.security import _RATE_BUCKETS
from app.config import settings as settings_module
from app.main import app


def test_api_key_is_not_required_when_unset(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    client = TestClient(app)

    response = client.get("/api/v1/papers")

    assert response.status_code == 200


def test_api_key_rejects_protected_endpoint_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", "secret-key")
    client = TestClient(app)

    response = client.get("/api/v1/papers", headers={"X-Request-ID": "auth-req-1"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
    assert response.headers["X-Request-ID"] == "auth-req-1"


def test_api_key_accepts_x_api_key_header(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", "secret-key")
    client = TestClient(app)

    response = client.get("/api/v1/papers", headers={"X-API-Key": "secret-key"})

    assert response.status_code == 200


def test_api_key_accepts_bearer_token(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", "secret-key")
    client = TestClient(app)

    response = client.get("/api/v1/papers", headers={"Authorization": "Bearer secret-key"})

    assert response.status_code == 200


def test_health_stays_public_when_api_key_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", "secret-key")
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_rate_limit_rejects_excess_requests(monkeypatch) -> None:
    _RATE_BUCKETS.clear()
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    monkeypatch.setattr(settings_module.settings, "API_RATE_LIMIT_PER_MINUTE", 1)
    client = TestClient(app)

    first_response = client.get("/api/v1/papers")
    second_response = client.get("/api/v1/papers")

    _RATE_BUCKETS.clear()
    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json() == {"detail": "Rate limit exceeded."}
    assert first_response.headers["X-RateLimit-Limit"] == "1"
    assert first_response.headers["X-RateLimit-Remaining"] == "0"
    assert second_response.headers["X-RateLimit-Limit"] == "1"
    assert second_response.headers["X-RateLimit-Remaining"] == "0"
    assert int(second_response.headers["X-RateLimit-Reset"]) > 0
    assert int(second_response.headers["Retry-After"]) > 0


def test_tenant_id_is_required_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    monkeypatch.setattr(settings_module.settings, "REQUIRE_TENANT_ID", True)
    client = TestClient(app)

    response = client.get("/api/v1/papers")

    assert response.status_code == 400
    assert response.json() == {"detail": "Missing or invalid tenant id."}


def test_valid_tenant_id_is_echoed_on_protected_responses(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    monkeypatch.setattr(settings_module.settings, "REQUIRE_TENANT_ID", True)
    client = TestClient(app)

    response = client.get("/api/v1/papers", headers={"X-Tenant-ID": "tenant-a"})

    assert response.status_code == 200
    assert response.headers["X-Tenant-ID"] == "tenant-a"


def test_rate_limit_is_scoped_by_tenant(monkeypatch) -> None:
    _RATE_BUCKETS.clear()
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    monkeypatch.setattr(settings_module.settings, "REQUIRE_TENANT_ID", True)
    monkeypatch.setattr(settings_module.settings, "API_RATE_LIMIT_PER_MINUTE", 1)
    client = TestClient(app)

    tenant_a_first = client.get("/api/v1/papers", headers={"X-Tenant-ID": "tenant-a"})
    tenant_a_second = client.get("/api/v1/papers", headers={"X-Tenant-ID": "tenant-a"})
    tenant_b_first = client.get("/api/v1/papers", headers={"X-Tenant-ID": "tenant-b"})

    _RATE_BUCKETS.clear()
    assert tenant_a_first.status_code == 200
    assert tenant_a_second.status_code == 429
    assert tenant_b_first.status_code == 200


def test_tenant_header_scopes_local_chat_storage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_module.settings, "API_KEY", None)
    monkeypatch.setattr(settings_module.settings, "REQUIRE_TENANT_ID", True)
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/sessions",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"title": "Tenant chat"},
    )

    assert response.status_code == 200
    chat_id = response.json()["chat_id"]
    assert (tmp_path / "tenants" / "tenant-a" / "chat_history" / f"{chat_id}.json").is_file()
    assert not (tmp_path / "chat_history" / f"{chat_id}.json").exists()
