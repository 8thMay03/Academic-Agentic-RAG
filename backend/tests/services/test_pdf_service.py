import httpx
import pytest

from app.services.pdf_service import PDFDownloadError, PDFService


PDF_BYTES = b"%PDF-1.7\nfake pdf content\n%%EOF"


def _client_for(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, follow_redirects=True)


@pytest.mark.asyncio
async def test_download_pdf_writes_file_to_destination_directory(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://arxiv.org/pdf/2606.12345v1"
        return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})

    async with _client_for(httpx.MockTransport(handler)) as client:
        service = PDFService(client=client)

        downloaded_path = await service.download_pdf(
            "https://arxiv.org/pdf/2606.12345v1",
            tmp_path,
        )

    assert downloaded_path == tmp_path / "2606.12345v1.pdf"
    assert downloaded_path.read_bytes() == PDF_BYTES


@pytest.mark.asyncio
async def test_download_pdfs_downloads_each_url(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})

    async with _client_for(httpx.MockTransport(handler)) as client:
        service = PDFService(client=client)

        downloaded_paths = await service.download_pdfs(
            [
                "https://arxiv.org/pdf/2606.12345v1",
                "https://arxiv.org/pdf/2606.67890v1",
            ],
            tmp_path,
        )

    assert downloaded_paths == [
        tmp_path / "2606.12345v1.pdf",
        tmp_path / "2606.67890v1.pdf",
    ]
    assert all(path.read_bytes() == PDF_BYTES for path in downloaded_paths)


@pytest.mark.asyncio
async def test_download_pdf_rejects_non_pdf_content(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not pdf</html>", headers={"content-type": "text/html"})

    async with _client_for(httpx.MockTransport(handler)) as client:
        service = PDFService(client=client)

        with pytest.raises(PDFDownloadError, match="not a PDF"):
            await service.download_pdf("https://example.com/not-pdf", tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_download_pdf_uses_cached_file_without_http_request(tmp_path) -> None:
    cached_path = tmp_path / "2606.12345v1.pdf"
    cached_path.write_bytes(PDF_BYTES)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request should not be made for cached PDFs.")

    async with _client_for(httpx.MockTransport(handler)) as client:
        service = PDFService(client=client)

        result = await service.download_pdf_result(
            "https://arxiv.org/pdf/2606.12345v1",
            tmp_path,
        )

    assert result.path == cached_path
    assert result.cached is True
    assert cached_path.read_bytes() == PDF_BYTES
