from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert payload["data_dir_configured"] is True
    assert payload["chroma_dir_configured"] is True
    assert isinstance(payload["data_dir_exists"], bool)
    assert isinstance(payload["chroma_dir_exists"], bool)
