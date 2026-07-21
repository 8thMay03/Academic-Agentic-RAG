from app.agent.models import (
    ToolCall,
    append_trace,
    normalize_retrieved_chunk,
    normalize_retrieved_chunks,
    retrieved_chunk_id,
    retrieved_chunk_page_number,
    retrieved_chunk_ranking_score,
    retrieved_chunk_source_id,
    retrieved_chunk_text,
)


def test_normalize_retrieved_chunk_canonicalizes_legacy_metadata() -> None:
    chunk = normalize_retrieved_chunk(
        {
            "id": "paper-1:p3:c0",
            "text": "Agentic RAG plans retrieval.",
            "metadata": {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "page_number": "3",
            },
            "score": "0.82",
        }
    )

    assert chunk["id"] == "paper-1:p3:c0"
    assert chunk["metadata"]["chunk_id"] == "paper-1:p3:c0"
    assert chunk["citation"]["chunk_id"] == "paper-1:p3:c0"
    assert chunk["citation"]["paper_id"] == "paper-1"
    assert chunk["citation"]["title"] == "Agentic RAG"
    assert chunk["citation"]["text"] == "Agentic RAG plans retrieval."
    assert chunk["retrieval_sources"] == []
    assert retrieved_chunk_id(chunk) == "paper-1:p3:c0"
    assert retrieved_chunk_text(chunk) == "Agentic RAG plans retrieval."
    assert retrieved_chunk_source_id(chunk) == "paper-1"
    assert retrieved_chunk_page_number(chunk) == 3
    assert retrieved_chunk_ranking_score(chunk) == 0.82


def test_normalize_retrieved_chunks_preserves_existing_citation_fields() -> None:
    chunks = normalize_retrieved_chunks(
        [
            {
                "text": "Web evidence.",
                "metadata": {"url": "https://example.com"},
                "citation": {
                    "paper_id": "web-source",
                    "title": "Web Source",
                    "chunk_id": "web:1",
                    "url": "https://example.com/source",
                },
                "retrieval_sources": ["web"],
            }
        ]
    )

    assert chunks[0]["id"] == "web:1"
    assert chunks[0]["metadata"]["url"] == "https://example.com"
    assert chunks[0]["citation"]["url"] == "https://example.com/source"
    assert chunks[0]["retrieval_sources"] == ["web"]


def test_append_trace_adds_typed_agent_event() -> None:
    trace = append_trace(
        [{"stage": "local_retrieve", "chunk_count": 1}],
        "quality_gate",
        sufficient=True,
        reason="strong_context",
    )

    assert trace == [
        {"stage": "local_retrieve", "chunk_count": 1},
        {
            "stage": "quality_gate",
            "sufficient": True,
            "reason": "strong_context",
        },
    ]


def test_tool_call_records_schema_tool_invocation() -> None:
    call = ToolCall(
        tool_name="web_search",
        input={"query": "agentic rag", "max_results": 5},
        reason="Local context was insufficient.",
        step_index=1,
    )

    assert call.tool_name == "web_search"
    assert call.input == {"query": "agentic rag", "max_results": 5}
    assert call.reason == "Local context was insufficient."
    assert call.step_index == 1
