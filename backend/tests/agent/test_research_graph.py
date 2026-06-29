from pathlib import Path

import pytest

from app.agent.graph import run_research_workflow
from app.config import settings as settings_module
from app.models.paper import Paper
from app.models.research import ResearchRequest
from app.services.llm_service import LLMService
from app.services.parser_service import ParserService
from app.services.pdf_index_service import PDFIndexResult, PDFIndexService
from app.services.pdf_service import PDFDownloadResult, PDFService
from app.services.search_service import SearchService


@pytest.mark.asyncio
async def test_research_workflow_expands_search_and_builds_report(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    pdf_path = tmp_path / "pdfs" / "2606.12345v1.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF test")

    async def fake_search(self, query: str, max_results: int, sort_by: str = "submittedDate"):
        calls.append(query)
        if query == "agentic rag":
            return []
        return [
            Paper(
                paper_id="2606.12345v1",
                title="Agentic RAG: A Survey",
                authors=["Ada Lovelace"],
                published="2026-06-20",
                abstract="Agentic RAG combines planning, retrieval, and reflection.",
                arxiv_url="https://arxiv.org/abs/2606.12345v1",
                url="https://arxiv.org/abs/2606.12345v1",
                pdf_url="https://arxiv.org/pdf/2606.12345v1",
            )
        ]

    async def fake_download(self, pdf_url: str, destination: Path):
        return PDFDownloadResult(pdf_path, cached=False)

    async def fake_parse(self, path: Path):
        return "Agentic RAG uses planning to decide when to retrieve and reflect."

    async def fake_index(self, path: Path, force: bool = False):
        return PDFIndexResult("2606.12345v1", path.name, 1)

    async def fake_complete(self, prompt: str):
        if "Compare these papers" in prompt:
            return "| Paper | Method |\n| --- | --- |\n| Agentic RAG | Planning + retrieval |"
        return "## Problem\n- Research agents need grounded retrieval."

    monkeypatch.setattr(settings_module.settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(SearchService, "search", fake_search)
    monkeypatch.setattr(PDFService, "download_pdf_result", fake_download)
    monkeypatch.setattr(ParserService, "parse_pdf", fake_parse)
    monkeypatch.setattr(PDFIndexService, "index_pdf", fake_index)
    monkeypatch.setattr(LLMService, "complete", fake_complete)

    response = await run_research_workflow(ResearchRequest(query="agentic rag", max_results=1))

    assert calls == ["agentic rag", "agentic rag survey"]
    assert response.papers[0].paper_id == "2606.12345v1"
    assert "Research agents need grounded retrieval" in response.summary
    assert "Planning + retrieval" in response.comparison
    assert response.report.startswith("# Research Report: agentic rag")
