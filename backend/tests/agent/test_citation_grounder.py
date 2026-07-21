from app.agent.citations import CitationGrounder
from app.models.citation import Citation


def test_citation_grounder_preserves_research_source_provenance() -> None:
    citations = CitationGrounder().build_citations(
        [
            {
                "id": "2601.12345:p2:c0",
                "text": "Fresh research discusses verifier loops.",
                "metadata": {
                    "paper_id": "2601.12345",
                    "title": "Fresh Agentic RAG",
                    "chunk_id": "2601.12345:p2:c0",
                    "source_type": "arxiv",
                    "source_url": "https://arxiv.org/abs/2601.12345",
                    "pdf_url": "https://arxiv.org/pdf/2601.12345",
                    "trust_level": "high",
                    "ingestion_status": "downloaded",
                },
                "score": 0.91,
                "retrieval_sources": ["vector"],
            }
        ],
        question="What does fresh Agentic RAG research discuss?",
    )

    assert len(citations) == 1
    assert citations[0].url == "https://arxiv.org/abs/2601.12345"
    assert citations[0].source_type == "arxiv"
    assert citations[0].source_url == "https://arxiv.org/abs/2601.12345"
    assert citations[0].pdf_url == "https://arxiv.org/pdf/2601.12345"
    assert citations[0].trust_level == "high"
    assert citations[0].ingestion_status == "downloaded"
    assert citations[0].evidence_quality == "high"


def test_citation_grounder_formats_display_answer_with_numbered_citations() -> None:
    citations = [
        Citation(
            paper_id="https://example.com/rag",
            title="RAG Guide",
            chunk_id="web-ingest:https://example.com/rag",
            url="https://example.com/rag",
        ),
        Citation(
            paper_id="paper-1",
            title="Agentic RAG",
            chunk_id="paper-1:p3:c0",
        ),
    ]

    answer = (
        "RAG reduces retraining needs "
        "[web-ingest:https://example.com/rag, paper-1:p3:c0]."
    )

    assert CitationGrounder().display_answer_with_numbered_citations(answer, citations) == (
        "RAG reduces retraining needs [1][2]."
    )
