from app.vectorstore.bm25 import tokenize


class RerankerService:
    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return chunks

        reranked_chunks = []
        for chunk in chunks:
            reranked_chunk = dict(chunk)
            text_terms = set(tokenize(reranked_chunk.get("text", "")))
            lexical_overlap = len(query_terms & text_terms) / len(query_terms)
            base_score = float(reranked_chunk.get("score", 0.0))
            reranked_chunk["rerank_score"] = (0.85 * base_score) + (0.15 * lexical_overlap)
            reranked_chunks.append(reranked_chunk)

        return sorted(
            reranked_chunks,
            key=lambda chunk: (chunk.get("rerank_score", 0.0), chunk.get("score", 0.0)),
            reverse=True,
        )
