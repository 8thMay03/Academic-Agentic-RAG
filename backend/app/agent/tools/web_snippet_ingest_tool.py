from app.agent.models import ToolResult
from app.models.chunk import Chunk
from app.vectorstore.indexing import index_chunks


class WebSnippetIngestTool:
    name = "web_snippet_ingest"

    async def run(self, input: dict) -> ToolResult:
        web_chunks = input.get("web_chunks") or []
        chunks_to_index: list[Chunk] = []
        for chunk in web_chunks:
            text = str(chunk.get("text") or "")
            if not text or len(text) < 50:
                continue
            metadata = chunk.get("metadata") or {}
            url = str(metadata.get("url") or "")
            title = str(metadata.get("title") or "")
            chunk_id = f"web-ingest:{url or metadata.get('chunk_id', '')}"
            chunks_to_index.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=url or chunk_id,
                    text=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "title": title,
                        "url": url,
                        "source": "web",
                    },
                )
            )

        if chunks_to_index:
            await index_chunks(chunks_to_index)

        return ToolResult(
            tool_name=self.name,
            success=True,
            metadata={"snippets_ingested": len(chunks_to_index)},
        )
