from app.models.chunk import Chunk


def chunks_to_documents(chunks: list[Chunk]) -> tuple[list[str], list[dict]]:
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "page_number": str(chunk.page_number or chunk.page or ""),
            "page": str(chunk.page_number or chunk.page or ""),
            **chunk.metadata,
        }
        for chunk in chunks
    ]
    return documents, metadatas
