from pathlib import Path

import pytest

from app.tools import pdf_dowload_tool, pdf_download_many_tool, pdf_download_tool


@pytest.mark.asyncio
async def test_pdf_download_tool_downloads_one_pdf(monkeypatch, tmp_path) -> None:
    expected_path = tmp_path / "paper.pdf"

    async def fake_download_pdf(self, pdf_url: str, destination: Path) -> Path:
        assert pdf_url == "https://arxiv.org/pdf/2606.12345v1"
        assert destination == tmp_path
        return expected_path

    monkeypatch.setattr("app.services.pdf_service.PDFService.download_pdf", fake_download_pdf)

    assert await pdf_download_tool("https://arxiv.org/pdf/2606.12345v1", tmp_path) == expected_path


@pytest.mark.asyncio
async def test_pdf_download_many_tool_downloads_multiple_pdfs(monkeypatch, tmp_path) -> None:
    expected_paths = [tmp_path / "paper-1.pdf", tmp_path / "paper-2.pdf"]

    async def fake_download_pdfs(self, pdf_urls: list[str], destination_dir: Path) -> list[Path]:
        assert pdf_urls == [
            "https://arxiv.org/pdf/2606.12345v1",
            "https://arxiv.org/pdf/2606.67890v1",
        ]
        assert destination_dir == tmp_path
        return expected_paths

    monkeypatch.setattr("app.services.pdf_service.PDFService.download_pdfs", fake_download_pdfs)

    assert (
        await pdf_download_many_tool(
            [
                "https://arxiv.org/pdf/2606.12345v1",
                "https://arxiv.org/pdf/2606.67890v1",
            ],
            tmp_path,
        )
        == expected_paths
    )


@pytest.mark.asyncio
async def test_pdf_dowload_tool_alias_supports_multiple_pdfs(monkeypatch, tmp_path) -> None:
    expected_paths = [tmp_path / "paper-1.pdf"]

    async def fake_download_pdfs(self, pdf_urls: list[str], destination_dir: Path) -> list[Path]:
        assert pdf_urls == ["https://arxiv.org/pdf/2606.12345v1"]
        assert destination_dir == tmp_path
        return expected_paths

    monkeypatch.setattr("app.services.pdf_service.PDFService.download_pdfs", fake_download_pdfs)

    assert (
        await pdf_dowload_tool(
            pdf_urls=["https://arxiv.org/pdf/2606.12345v1"],
            destination_dir=tmp_path,
        )
        == expected_paths
    )
