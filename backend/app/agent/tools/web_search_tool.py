from app.agent.models import ToolResult, normalize_retrieved_chunks
from app.services.web_search_service import WebSearchService


class WebSearchTool:
    name = "web_search"

    def __init__(self, web_search_service: WebSearchService) -> None:
        self._web_search_service = web_search_service

    async def run(self, input: dict) -> ToolResult:
        query = str(input["query"])
        max_results = int(input.get("max_results", 5))
        result = await self._web_search_service.search(query, max_results=max_results)
        chunks = []
        for index, source in enumerate(result.sources[:max_results], start=1):
            text = " ".join(str(source.get("content") or "").split())
            if not text:
                continue
            title = str(source.get("title") or source.get("url") or f"Web source {index}")
            url = str(source.get("url") or "")
            chunk_id = f"web:{index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "paper_id": url or chunk_id,
                        "title": title,
                        "chunk_id": chunk_id,
                        "url": url,
                    },
                    "score": self._optional_float(source.get("score")),
                    "retrieval_sources": ["web"],
                    "citation": {
                        "paper_id": url or chunk_id,
                        "title": title,
                        "chunk_id": chunk_id,
                        "url": url,
                        "text": text,
                    },
                }
            )

        chunks = normalize_retrieved_chunks(chunks)
        return ToolResult(
            tool_name=self.name,
            success=True,
            chunks=chunks,
            metadata={
                "chunk_count": len(chunks),
                "skipped_reason": result.skipped_reason,
            },
        )

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
