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
        }
    ]
