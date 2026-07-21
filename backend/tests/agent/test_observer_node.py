from app.agent.models import ToolResult
from app.agent.nodes.observer_node import observer_node


async def test_observer_node_records_source_metadata_in_trace() -> None:
    state = await observer_node(
        {
            "current_step_index": 1,
            "current_tool_result": ToolResult(
                tool_name="pdf_download",
                success=True,
                artifacts=[{"path": "data/pdfs/2601.12345.pdf"}],
                metadata={
                    "source_type": "arxiv",
                    "source_url": "https://arxiv.org/abs/2601.12345",
                    "pdf_url": "https://arxiv.org/pdf/2601.12345",
                    "discovered_by_query": "agentic rag",
                    "trust_level": "high",
                    "ingestion_status": "downloaded",
                },
            ),
            "trace": [],
            "tool_results": [],
        }
    )

    assert state["current_step_index"] == 2
    assert state["tool_results"][0].tool_name == "pdf_download"
    assert state["trace"] == [
        {
            "stage": "observe",
            "tool_name": "pdf_download",
            "step_index": 1,
            "success": True,
            "artifact_count": 1,
            "source_type": "arxiv",
            "source_url": "https://arxiv.org/abs/2601.12345",
            "pdf_url": "https://arxiv.org/pdf/2601.12345",
            "discovered_by_query": "agentic rag",
            "trust_level": "high",
            "ingestion_status": "downloaded",
            "tool_result": {
                "tool_name": "pdf_download",
                "success": True,
                "metadata": {
                    "source_type": "arxiv",
                    "source_url": "https://arxiv.org/abs/2601.12345",
                    "pdf_url": "https://arxiv.org/pdf/2601.12345",
                    "discovered_by_query": "agentic rag",
                    "trust_level": "high",
                    "ingestion_status": "downloaded",
                },
                "artifacts": [{"path": "data/pdfs/2601.12345.pdf"}],
                "artifact_count": 1,
            },
        }
    ]


async def test_observer_node_compacts_chunk_results_in_trace() -> None:
    state = await observer_node(
        {
            "current_step_index": 0,
            "current_tool_result": ToolResult(
                tool_name="web_search",
                success=True,
                chunks=[
                    {
                        "id": "web:1",
                        "text": " ".join(["retrieved"] * 80),
                        "score": 0.82,
                        "citation": {
                            "title": "Retrieved source",
                            "url": "https://example.com/source",
                            "chunk_id": "web:1",
                        },
                    }
                ],
            ),
            "trace": [],
            "tool_results": [],
        }
    )

    tool_result = state["trace"][0]["tool_result"]

    assert tool_result["tool_name"] == "web_search"
    assert tool_result["chunk_count"] == 1
    assert tool_result["chunks"][0]["id"] == "web:1"
    assert tool_result["chunks"][0]["title"] == "Retrieved source"
    assert tool_result["chunks"][0]["url"] == "https://example.com/source"
    assert tool_result["chunks"][0]["text"].endswith("...")
