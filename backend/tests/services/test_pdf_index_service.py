from app.services.pdf_index_service import PDFIndexService


async def test_pdf_index_service_indexes_downloaded_pdf(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper-1.pdf"
    pdf_path.write_bytes(b"%PDF")
    indexed_chunks = []

    def fake_extract_text_from_pdf(path):
        assert path == pdf_path
        return "Page one text\fPage two text"

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.services.pdf_index_service.extract_text_from_pdf", fake_extract_text_from_pdf)
    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", fake_index_chunks)

    result = await PDFIndexService(data_dir=tmp_path).index_downloaded_pdf("paper-1.pdf")

    assert result.paper_id == "paper-1"
    assert result.filename == "paper-1.pdf"
    assert result.chunks_indexed == 2
    assert result.cached is False
    assert [chunk.page_number for chunk in indexed_chunks] == [1, 2]
    assert indexed_chunks[0].metadata["title"] == "paper-1.pdf"
    assert indexed_chunks[0].metadata["source_path"].endswith("paper-1.pdf")


async def test_pdf_index_service_skips_unchanged_pdf_from_manifest(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper-1.pdf"
    pdf_path.write_bytes(b"%PDF")
    service = PDFIndexService(data_dir=tmp_path)

    async def fake_index_chunks(chunks):
        return None

    monkeypatch.setattr(
        "app.services.pdf_index_service.extract_text_from_pdf",
        lambda path: "Page one text",
    )
    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", fake_index_chunks)

    first_result = await service.index_downloaded_pdf("paper-1.pdf")

    def fail_extract_text_from_pdf(path):
        raise AssertionError("unchanged PDFs should not be parsed again")

    monkeypatch.setattr("app.services.pdf_index_service.extract_text_from_pdf", fail_extract_text_from_pdf)
    second_result = await service.index_downloaded_pdf("paper-1.pdf")

    assert first_result.cached is False
    assert second_result.cached is True
    assert second_result.chunks_indexed == first_result.chunks_indexed


async def test_pdf_index_service_persists_source_metadata(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "paper-1.pdf"
    pdf_path.write_bytes(b"%PDF")
    indexed_chunks = []
    source_metadata = {
        "source_type": "arxiv",
        "source_url": "https://arxiv.org/abs/2601.12345",
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "discovered_by_query": "agentic rag",
        "trust_level": "high",
    }

    monkeypatch.setattr(
        "app.services.pdf_index_service.extract_text_from_pdf",
        lambda path: "Page one text",
    )

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", fake_index_chunks)

    service = PDFIndexService(data_dir=tmp_path)
    first_result = await service.index_downloaded_pdf("paper-1.pdf", source_metadata=source_metadata)
    second_result = await service.index_downloaded_pdf("paper-1.pdf")

    assert first_result.source_metadata == source_metadata
    assert second_result.cached is True
    assert second_result.source_metadata == source_metadata
    assert indexed_chunks[0].metadata["source_type"] == "arxiv"
    assert indexed_chunks[0].metadata["source_url"] == "https://arxiv.org/abs/2601.12345"
    assert indexed_chunks[0].metadata["pdf_url"] == "https://arxiv.org/pdf/2601.12345"


async def test_pdf_index_service_indexes_all_downloaded_pdfs(monkeypatch, tmp_path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper-1.pdf").write_bytes(b"%PDF")
    (pdf_dir / "paper-2.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(
        "app.services.pdf_index_service.extract_text_from_pdf",
        lambda path: "Page text",
    )
    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", lambda chunks: None)

    async def fake_index_chunks(chunks):
        return None

    monkeypatch.setattr("app.services.pdf_index_service.index_chunks", fake_index_chunks)

    results = await PDFIndexService(data_dir=tmp_path).index_all_downloaded_pdfs()

    assert [result.filename for result in results] == ["paper-1.pdf", "paper-2.pdf"]
