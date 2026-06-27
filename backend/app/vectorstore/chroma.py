class ChromaVectorStore:
    def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        # TODO: Persist documents and embeddings to ChromaDB.
        raise NotImplementedError("Chroma indexing is not implemented yet.")

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        # TODO: Query ChromaDB.
        return []

