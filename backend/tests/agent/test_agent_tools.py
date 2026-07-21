from pathlib import Path

import pytest

from app.agent.models import ToolResult
from app.agent.tools.arxiv_search_tool import ArxivSearchTool
from app.agent.tools.local_retrieve_tool import LocalRetrieveTool, local_retrieve_input
from app.agent.tools.pdf_download_tool import PDFDownloadTool
from app.agent.tools.pdf_index_tool import PDFIndexTool
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.tools.web_snippet_ingest_tool import WebSnippetIngestTool
from app.models.chat import ChatHistoryMessage
from app.models.paper import Paper
from app.services.pdf_index_service import PDFIndexResult
from app.services.pdf_service import PDFDownloadResult
from app.services.web_search_service import WebSearchResult


class FakeRAGService:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve_context(
        self,
        question,
        chat_id=None,
        paper_ids=None,
        top_k=5,
        score_threshold=0.65,
        chat_history=None,
    ):
        self.calls.append(
            {
                "question": question,
                "chat_id": chat_id,
                "paper_ids": paper_ids,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "chat_history": chat_history,
            }
        )
        return [{"id": "paper-1:p1:c0", "text": "Planning retrieves evidence."}]


class FakeWebSearchService:
    async def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        assert query == "agentic rag"
        assert max_results == 2
        return WebSearchResult(
            sources=[
                {
                    "title": "Agentic RAG",
                    "url": "https://example.com/agentic-rag",
                    "content": "Agentic RAG plans retrieval before answering.",
                    "raw_content": "",
                    "score": "0.82",
                },
                {
                    "title": "Empty",
                    "url": "https://example.com/empty",
                    "content": "",
                },
            ]
        )


class FakeArxivService:
    async def search(self, query: str, max_results: int, sort_by: str = "submittedDate"):
        assert query == "agentic rag"
        assert max_results == 2
        assert sort_by == "relevance"
        return [
            Paper(
                paper_id="2601.12345",
                title="Agentic RAG",
                authors=["A. Researcher"],
                published="2026-01-01",
                abstract="A paper about agentic retrieval.",
                arxiv_url="https://arxiv.org/abs/2601.12345",
                url="https://arxiv.org/abs/2601.12345",
                pdf_url="https://arxiv.org/pdf/2601.12345",
            )
        ]


class FakePDFService:
    async def download_pdf_result(self, pdf_url: str, destination: Path):
        assert pdf_url == "https://arxiv.org/pdf/2601.12345"
        assert destination == Path("data/pdfs")
        return PDFDownloadResult(path=Path("data/pdfs/2601.12345.pdf"), cached=False)


class FakePDFIndexService:
    async def index_downloaded_pdf(self, filename: str, force: bool = False, source_metadata: dict | None = None):
        assert filename == "2601.12345.pdf"
        assert force is True
        return PDFIndexResult(
            paper_id="2601.12345",
            filename="2601.12345.pdf",
            chunks_indexed=12,
            cached=False,
            source_metadata=source_metadata or {},
        )

    async def index_pdf(self, pdf_path: Path, force: bool = False, source_metadata: dict | None = None):
        assert pdf_path == Path("data/pdfs/2601.12345.pdf")
        assert force is False
        return PDFIndexResult(
            paper_id="2601.12345",
            filename="2601.12345.pdf",
            chunks_indexed=12,
            cached=True,
            source_metadata=source_metadata or {},
        )


class EchoTool:
    name = "echo"

    async def run(self, input: dict) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, metadata=input)


@pytest.mark.asyncio
async def test_local_retrieve_tool_delegates_to_rag_service() -> None:
    history = [
        ChatHistoryMessage(
            role="user",
            content="Previous question?",
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]
    rag = FakeRAGService()
    tool = LocalRetrieveTool(rag)

    result = await tool.run(
        local_retrieve_input(
            question="What is the method?",
            chat_id="chat-1",
            paper_ids=["paper-1"],
            top_k=3,
            score_threshold=0.7,
            chat_history=history,
        )
    )

    assert result.success is True
    assert result.tool_name == "local_retrieve"
    assert result.chunks == [
        {
            "id": "paper-1:p1:c0",
            "text": "Planning retrieves evidence.",
            "metadata": {"chunk_id": "paper-1:p1:c0"},
            "citation": {
                "chunk_id": "paper-1:p1:c0",
                "text": "Planning retrieves evidence.",
            },
            "retrieval_sources": [],
        }
    ]
    assert result.metadata == {"chunk_count": 1, "paper_ids": ["paper-1"]}
    assert rag.calls == [
        {
            "question": "What is the method?",
            "chat_id": "chat-1",
            "paper_ids": ["paper-1"],
            "top_k": 3,
            "score_threshold": 0.7,
            "chat_history": history,
        }
    ]


@pytest.mark.asyncio
async def test_web_search_tool_normalizes_sources_to_chunks() -> None:
    tool = WebSearchTool(FakeWebSearchService())

    result = await tool.run({"query": "agentic rag", "max_results": 2})

    assert result.success is True
    assert result.tool_name == "web_search"
    assert result.metadata == {"chunk_count": 1, "skipped_reason": None, "raw_content_count": 0}
    assert result.chunks[0]["id"] == "web:1"
    assert result.chunks[0]["score"] == 0.82
    assert result.chunks[0]["citation"]["url"] == "https://example.com/agentic-rag"
    assert result.chunks[0]["retrieval_sources"] == ["web"]
    assert result.chunks[0]["metadata"]["content_source"] == "snippet"


@pytest.mark.asyncio
async def test_web_search_tool_uses_and_chunks_raw_content() -> None:
    class RawContentWebSearchService:
        async def search(self, query: str, max_results: int = 5) -> WebSearchResult:
            return WebSearchResult(
                sources=[
                    {
                        "title": "GRU vs LSTM",
                        "url": "https://example.com/gru-lstm",
                        "content": "Short snippet.",
                        "raw_content": " ".join(["GRU and LSTM use recurrent gates."] * 120),
                        "score": 0.91,
                    }
                ]
            )

    tool = WebSearchTool(RawContentWebSearchService())

    result = await tool.run({"query": "GRU vs LSTM", "max_results": 1})

    assert result.success is True
    assert result.metadata["raw_content_count"] == 1
    assert result.metadata["chunk_count"] > 1
    assert result.chunks[0]["id"] == "web:1:c0"
    assert result.chunks[0]["metadata"]["content_source"] == "raw_content"
    assert result.chunks[0]["metadata"]["source_chunk_count"] == result.metadata["chunk_count"]
    assert "Short snippet" not in result.chunks[0]["text"]


@pytest.mark.asyncio
async def test_web_snippet_ingest_tool_indexes_valid_snippets(monkeypatch) -> None:
    indexed_chunks = []

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.agent.tools.web_snippet_ingest_tool.index_chunks", fake_index_chunks)
    tool = WebSnippetIngestTool()

    result = await tool.run(
        {
            "web_chunks": [
                {
                    "text": "This is a long enough web snippet about agentic retrieval and planning.",
                    "metadata": {
                        "url": "https://example.com/agentic-rag",
                        "title": "Agentic RAG",
                    },
                },
                {
                    "text": "Too short.",
                    "metadata": {"url": "https://example.com/short"},
                },
            ]
        }
    )

    assert result.success is True
    assert result.metadata == {"snippets_ingested": 1}
    assert len(indexed_chunks) == 1
    assert indexed_chunks[0].chunk_id == "web-ingest:https://example.com/agentic-rag"
    assert indexed_chunks[0].metadata["source"] == "web"


@pytest.mark.asyncio
async def test_web_snippet_ingest_tool_uses_source_chunk_ids_for_full_page_chunks(monkeypatch) -> None:
    indexed_chunks = []

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.agent.tools.web_snippet_ingest_tool.index_chunks", fake_index_chunks)
    tool = WebSnippetIngestTool()

    result = await tool.run(
        {
            "web_chunks": [
                {
                    "id": "web:1:c0",
                    "text": "This is long enough raw page chunk zero about GRU and LSTM gates.",
                    "metadata": {
                        "chunk_id": "web:1:c0",
                        "url": "https://example.com/gru-lstm",
                        "title": "GRU vs LSTM",
                        "content_source": "raw_content",
                    },
                },
                {
                    "id": "web:1:c1",
                    "text": "This is long enough raw page chunk one about GRU and LSTM memory.",
                    "metadata": {
                        "chunk_id": "web:1:c1",
                        "url": "https://example.com/gru-lstm",
                        "title": "GRU vs LSTM",
                        "content_source": "raw_content",
                    },
                },
            ]
        }
    )

    assert result.metadata == {"snippets_ingested": 2}
    assert [chunk.chunk_id for chunk in indexed_chunks] == ["web-ingest:web:1:c0", "web-ingest:web:1:c1"]
    assert indexed_chunks[0].metadata["content_source"] == "raw_content"


@pytest.mark.asyncio
async def test_web_snippet_ingest_tool_scopes_chunks_to_chat(monkeypatch) -> None:
    indexed_chunks = []

    async def fake_index_chunks(chunks):
        indexed_chunks.extend(chunks)

    monkeypatch.setattr("app.agent.tools.web_snippet_ingest_tool.index_chunks", fake_index_chunks)
    tool = WebSnippetIngestTool()

    result = await tool.run(
        {
            "chat_id": "chat-1",
            "web_chunks": [
                {
                    "id": "web:1",
                    "text": "This is long enough web content about CNN convolution filters.",
                    "metadata": {
                        "chunk_id": "web:1",
                        "url": "https://example.com/cnn",
                        "title": "CNN",
                    },
                }
            ],
        }
    )

    assert result.metadata == {"snippets_ingested": 1, "chat_id": "chat-1"}
    assert indexed_chunks[0].metadata["chat_id"] == "chat-1"


@pytest.mark.asyncio
async def test_tool_registry_runs_registered_tools_by_name() -> None:
    registry = ToolRegistry([EchoTool()])

    result = await registry.run("echo", {"value": 1})

    assert registry.names() == ["echo"]
    assert result == ToolResult(tool_name="echo", success=True, metadata={"value": 1})


@pytest.mark.asyncio
async def test_arxiv_search_tool_returns_paper_artifacts_and_pdf_urls() -> None:
    tool = ArxivSearchTool(FakeArxivService())

    result = await tool.run({"query": "agentic rag", "max_results": 2, "sort_by": "relevance"})

    assert result.success is True
    assert result.tool_name == "arxiv_search"
    assert result.artifacts[0]["paper_id"] == "2601.12345"
    assert result.artifacts[0]["pdf_url"] == "https://arxiv.org/pdf/2601.12345"
    assert result.artifacts[0]["source_type"] == "arxiv"
    assert result.artifacts[0]["source_url"] == "https://arxiv.org/abs/2601.12345"
    assert result.artifacts[0]["discovered_by_query"] == "agentic rag"
    assert result.artifacts[0]["trust_level"] == "high"
    assert result.artifacts[0]["ingestion_status"] == "discovered"
    assert result.metadata == {
        "paper_count": 1,
        "pdf_urls": ["https://arxiv.org/pdf/2601.12345"],
    }


@pytest.mark.asyncio
async def test_pdf_download_tool_returns_download_artifact() -> None:
    tool = PDFDownloadTool(FakePDFService(), default_destination_dir="data")

    result = await tool.run({"pdf_url": "https://arxiv.org/pdf/2601.12345"})

    assert result.success is True
    assert result.tool_name == "pdf_download"
    assert result.metadata == {
        "path": "data/pdfs/2601.12345.pdf",
        "filename": "2601.12345.pdf",
        "cached": False,
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "source_url": "https://arxiv.org/pdf/2601.12345",
        "source_type": "web_pdf",
        "discovered_by_query": None,
        "trust_level": "unknown",
        "ingestion_status": "downloaded",
    }
    assert result.artifacts == [result.metadata]


@pytest.mark.asyncio
async def test_pdf_download_tool_preserves_discovered_source_metadata() -> None:
    tool = PDFDownloadTool(FakePDFService(), default_destination_dir="data")

    result = await tool.run(
        {
            "pdf_url": "https://arxiv.org/pdf/2601.12345",
            "source_metadata": {
                "source_type": "arxiv",
                "source_url": "https://arxiv.org/abs/2601.12345",
                "discovered_by_query": "agentic rag",
                "trust_level": "high",
            },
        }
    )

    assert result.metadata["source_type"] == "arxiv"
    assert result.metadata["source_url"] == "https://arxiv.org/abs/2601.12345"
    assert result.metadata["discovered_by_query"] == "agentic rag"
    assert result.metadata["trust_level"] == "high"


@pytest.mark.asyncio
async def test_pdf_index_tool_indexes_downloaded_filename() -> None:
    tool = PDFIndexTool(FakePDFIndexService())

    result = await tool.run({"filename": "2601.12345.pdf", "force": True})

    assert result.success is True
    assert result.tool_name == "pdf_index"
    assert result.metadata == {
        "paper_id": "2601.12345",
        "filename": "2601.12345.pdf",
        "chunks_indexed": 12,
        "cached": False,
    }


@pytest.mark.asyncio
async def test_pdf_index_tool_can_index_by_path() -> None:
    tool = PDFIndexTool(FakePDFIndexService())

    result = await tool.run(
        {
            "path": "data/pdfs/2601.12345.pdf",
            "source_metadata": {
                "source_type": "arxiv",
                "source_url": "https://arxiv.org/abs/2601.12345",
            },
        }
    )

    assert result.success is True
    assert result.metadata["cached"] is True
    assert result.metadata["source_type"] == "arxiv"
    assert result.metadata["source_url"] == "https://arxiv.org/abs/2601.12345"
