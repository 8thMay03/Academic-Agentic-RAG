from app.agent.models import ToolResult
from app.models.chunk import Chunk
from app.vectorstore.indexing import index_chunks


class WebSnippetIngestTool:
    name = "web_snippet_ingest"
    description = "Persist web search chunks into the vector store scoped to the current chat when available."
    input_schema = {"web_chunks": "list[RetrievedChunk]", "chat_id": "string|null"}
    when_to_use = "Use after web_search returns useful chunks that should be available for follow-up retrieval."
    failure_modes = ["no_web_chunks", "embedding_failure", "vector_store_write_failure"]

    async def run(self, input: dict) -> ToolResult:
        web_chunks = input.get("web_chunks") or []
        chat_id = str(input.get("chat_id") or "")
        chunks_to_index: list[Chunk] = []
        for chunk in web_chunks:
            text = str(chunk.get("text") or "")
            if not text or len(text) < 50:
                continue
            metadata = chunk.get("metadata") or {}
            url = str(metadata.get("url") or "")
            title = str(metadata.get("title") or "")
            source_chunk_id = str(metadata.get("chunk_id") or chunk.get("id") or "")
            chunk_id = f"web-ingest:{source_chunk_id or url}"
            chunk_metadata = {
                "chunk_id": chunk_id,
                "title": title,
                "url": url,
                "source": "web",
                "source_type": metadata.get("source_type") or "web_page",
                "content_source": metadata.get("content_source") or "snippet",
                "source_chunk_id": source_chunk_id,
            }
            if chat_id:
                chunk_metadata["chat_id"] = chat_id
            chunks_to_index.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=url or chunk_id,
                    text=text,
                    metadata=chunk_metadata,
                )
            )

        embedding_usage = None
        if chunks_to_index:
            embedding_usage = await index_chunks(chunks_to_index)

        result_metadata = {"snippets_ingested": len(chunks_to_index)}
        if chat_id:
            result_metadata["chat_id"] = chat_id
        if embedding_usage:
            result_metadata.update(
                {
                    "embedding_model": embedding_usage.model,
                    "embedding_input_count": embedding_usage.input_count,
                    "embedding_tokens": embedding_usage.total_tokens,
                    "embedding_estimated_cost_usd": embedding_usage.estimated_cost_usd,
                }
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            metadata=result_metadata,
        )
