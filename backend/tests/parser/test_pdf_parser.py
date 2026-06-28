from pathlib import Path

import fitz
import pytest

from app.parser.pdf_parser import extract_text_from_pdf
from app.services.parser_service import ParserService


def _create_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extract_text_from_pdf_reads_all_pages_in_order(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _create_pdf(
        pdf_path,
        [
            "Page one title\nAgentic RAG introduction",
            "Page two methods\nRetriever and planner details",
            "Page three conclusion\nFuture work",
        ],
    )

    text = extract_text_from_pdf(pdf_path)

    assert "Page one title" in text
    assert "Agentic RAG introduction" in text
    assert "Page two methods" in text
    assert "Retriever and planner details" in text
    assert "Page three conclusion" in text
    assert text.index("Page one title") < text.index("Page two methods")
    assert text.index("Page two methods") < text.index("Page three conclusion")


def test_extract_text_from_pdf_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(tmp_path / "missing.pdf")


@pytest.mark.asyncio
async def test_parser_service_parses_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _create_pdf(pdf_path, ["First page text", "Second page text"])

    text = await ParserService().parse_pdf(pdf_path)

    assert "First page text" in text
    assert "Second page text" in text
