import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_pdf_index_service, get_pdf_service
from app.config import settings as settings_module
from app.main import app
from app.services.pdf_service import PDFDownloadError, PDFDownloadResult
from app.services.pdf_index_service import PDFIndexResult


class FakePDFService:
    async def download_pdf_result(
        self,
        pdf_url: str,
        destination_dir: Path,
    ) -> PDFDownloadResult:
        assert destination_dir == Path("data") / "pdfs"
        if pdf_url == "https://arxiv.org/pdf/2606.12345v1":
            return PDFDownloadResult(Path("data/pdfs/2606.12345v1.pdf"), cached=False)
        if pdf_url == "https://arxiv.org/pdf/2606.67890v1":
            return PDFDownloadResult(Path("data/pdfs/2606.67890v1.pdf"), cached=True)
        raise AssertionError(f"Unexpected PDF URL: {pdf_url}")


class PartiallyFailingPDFService:
    async def download_pdf_result(
        self,
        pdf_url: str,
        destination_dir: Path,
    ) -> PDFDownloadResult:
        if pdf_url == "https://arxiv.org/pdf/2606.12345v1":
            return PDFDownloadResult(Path("data/pdfs/2606.12345v1.pdf"), cached=False)
        raise PDFDownloadError(f"Failed to download PDF: {pdf_url} returned HTTP 404")


class FailingPDFService:
    async def download_pdf_result(
        self,
        pdf_url: str,
        destination_dir: Path,
    ) -> PDFDownloadResult:
        raise PDFDownloadError(f"Failed to download PDF: {pdf_url} returned HTTP 404")


def test_download_papers_downloads_pdf_urls() -> None:
    app.dependency_overrides[get_pdf_service] = lambda: FakePDFService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/download",
        json={
            "pdf_urls": [
                "https://arxiv.org/pdf/2606.12345v1",
                "https://arxiv.org/pdf/2606.67890v1",
            ]
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "files": [
            "data/pdfs/2606.12345v1.pdf",
            "data/pdfs/2606.67890v1.pdf",
        ],
        "cached_files": ["data/pdfs/2606.67890v1.pdf"],
        "errors": [],
    }


def test_download_papers_returns_successes_and_errors_for_partial_failure() -> None:
    app.dependency_overrides[get_pdf_service] = lambda: PartiallyFailingPDFService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/download",
        json={
            "pdf_urls": [
                "https://arxiv.org/pdf/2606.12345v1",
                "https://arxiv.org/pdf/2606.67890v1",
            ]
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "files": ["data/pdfs/2606.12345v1.pdf"],
        "cached_files": [],
        "errors": [
            {
                "pdf_url": "https://arxiv.org/pdf/2606.67890v1",
                "error": (
                    "Failed to download PDF: "
                    "https://arxiv.org/pdf/2606.67890v1 returned HTTP 404"
                ),
            }
        ],
    }


def test_download_papers_returns_bad_gateway_on_download_error() -> None:
    app.dependency_overrides[get_pdf_service] = lambda: FailingPDFService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/download",
        json={"pdf_urls": ["https://example.com/file.pdf"]},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {
        "detail": [
            {
                "pdf_url": "https://example.com/file.pdf",
                "error": "Failed to download PDF: https://example.com/file.pdf returned HTTP 404",
            }
        ]
    }


def test_list_downloaded_pdfs_returns_existing_pdf_files(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    first_pdf = pdf_dir / "paper-1.pdf"
    second_pdf = pdf_dir / "paper-2.pdf"
    first_pdf.write_bytes(b"%PDF first")
    second_pdf.write_bytes(b"%PDF second")
    os.utime(first_pdf, (1000, 1000))
    os.utime(second_pdf, (2000, 2000))
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.get("/api/v1/papers/pdfs")

    assert response.status_code == 200
    payload = response.json()
    assert [item["filename"] for item in payload] == ["paper-2.pdf", "paper-1.pdf"]
    assert payload[0]["path"].endswith("paper-2.pdf")
    assert payload[0]["size_bytes"] == len(b"%PDF second")
    assert payload[0]["modified_at"]


def test_upload_local_pdfs_saves_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/pdfs/upload",
        files=[
            ("files", ("paper-1.pdf", b"%PDF first", "application/pdf")),
            ("files", ("paper-2.pdf", b"%PDF second", "application/pdf")),
        ],
    )

    assert response.status_code == 201
    payload = response.json()
    assert [item["filename"] for item in payload] == ["paper-1.pdf", "paper-2.pdf"]
    assert (tmp_path / "pdfs" / "paper-1.pdf").read_bytes() == b"%PDF first"
    assert (tmp_path / "pdfs" / "paper-2.pdf").read_bytes() == b"%PDF second"


def test_upload_local_pdfs_uses_unique_filename(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF existing")
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/pdfs/upload",
        files=[("files", ("paper.pdf", b"%PDF uploaded", "application/pdf"))],
    )

    assert response.status_code == 201
    assert response.json()[0]["filename"] == "paper-2.pdf"
    assert (pdf_dir / "paper.pdf").read_bytes() == b"%PDF existing"
    assert (pdf_dir / "paper-2.pdf").read_bytes() == b"%PDF uploaded"


def test_upload_local_pdfs_rejects_non_pdf(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/api/v1/papers/pdfs/upload",
        files=[("files", ("notes.txt", b"not a pdf", "text/plain"))],
    )

    assert response.status_code == 422
    assert not (tmp_path / "pdfs" / "notes.txt").exists()


def test_get_downloaded_pdf_content_returns_inline_pdf(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_file = pdf_dir / "paper-1.pdf"
    pdf_file.write_bytes(b"%PDF paper content")
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.get("/api/v1/papers/pdfs/paper-1.pdf/content")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF paper content"


def test_get_downloaded_pdf_content_returns_404_for_missing_pdf(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.get("/api/v1/papers/pdfs/missing.pdf/content")

    assert response.status_code == 404


def test_index_downloaded_pdf_indexes_local_pdf(monkeypatch, tmp_path) -> None:
    class FakePDFIndexService:
        async def index_downloaded_pdf(self, filename: str):
            assert filename == "paper-1.pdf"
            return PDFIndexResult("paper-1", "paper-1.pdf", 2)

    app.dependency_overrides[get_pdf_index_service] = lambda: FakePDFIndexService()
    client = TestClient(app)

    response = client.post("/api/v1/papers/pdfs/index", json={"filename": "paper-1.pdf"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "paper_id": "paper-1",
        "filename": "paper-1.pdf",
        "chunks_indexed": 2,
    }


def test_index_downloaded_pdf_returns_404_for_missing_file(monkeypatch, tmp_path) -> None:
    class MissingPDFIndexService:
        async def index_downloaded_pdf(self, filename: str):
            raise FileNotFoundError(filename)

    app.dependency_overrides[get_pdf_index_service] = lambda: MissingPDFIndexService()
    client = TestClient(app)

    response = client.post("/api/v1/papers/pdfs/index", json={"filename": "missing.pdf"})

    app.dependency_overrides.clear()

    assert response.status_code == 404
