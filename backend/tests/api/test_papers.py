from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_pdf_service
from app.main import app
from app.services.pdf_service import PDFDownloadError, PDFDownloadResult


class FakePDFService:
    async def download_pdf_results(
        self,
        pdf_urls: list[str],
        destination_dir: Path,
    ) -> list[PDFDownloadResult]:
        assert pdf_urls == [
            "https://arxiv.org/pdf/2606.12345v1",
            "https://arxiv.org/pdf/2606.67890v1",
        ]
        assert destination_dir == Path("data") / "pdfs"
        return [
            PDFDownloadResult(Path("data/pdfs/2606.12345v1.pdf"), cached=False),
            PDFDownloadResult(Path("data/pdfs/2606.67890v1.pdf"), cached=True),
        ]


class FailingPDFService:
    async def download_pdf_results(
        self,
        pdf_urls: list[str],
        destination_dir: Path,
    ) -> list[PDFDownloadResult]:
        raise PDFDownloadError("Failed to download PDF: https://example.com/file.pdf")


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
    assert response.json() == {"detail": "Failed to download PDF: https://example.com/file.pdf"}
