from app.models.chunk import Chunk
from app.vectorstore.indexing import chunks_to_documents


def test_chunks_to_documents_includes_chunk_metadata() -> None:
    chunk = Chunk(
        chunk_id="paper-1:p3:c0",
        paper_id="paper-1",
        text="Chunk text",
        page_number=3,
        metadata={"source": "arxiv"},
    )

    documents, metadatas = chunks_to_documents([chunk])

    assert documents == ["Chunk text"]
    assert metadatas == [
        {
            "chunk_id": "paper-1:p3:c0",
            "paper_id": "paper-1",
            "page_number": "3",
            "page": "3",
            "source": "arxiv",
        }
    ]
