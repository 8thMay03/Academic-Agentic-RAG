from app.models.chunk import Chunk
from app.vectorstore.chroma import ChromaVectorStore


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


async def index_chunks(
    chunks: list[Chunk],
    vector_store: ChromaVectorStore | None = None,
) -> None:
    documents, metadatas = chunks_to_documents(chunks)
    store = vector_store or ChromaVectorStore()
    await store.add_documents(documents, metadatas)
