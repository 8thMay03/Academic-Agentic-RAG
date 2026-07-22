from app.agent.models import ToolResult, normalize_retrieved_chunks
from app.parser.chunker import chunk_text
from app.services.web_search_service import WebSearchService


WEB_CHUNK_SIZE = 1800
WEB_CHUNK_OVERLAP = 180


class WebSearchTool:
    name = "web_search"
    description = "Search the web and convert result snippets or raw content into temporary cited chunks."
    input_schema = {"query": "string", "max_results": "integer"}
    when_to_use = "Use when local context is insufficient or current web information is required."
    failure_modes = ["missing_tavily_api_key", "empty_results", "tool_timeout", "untrusted_web_content"]

    def __init__(self, web_search_service: WebSearchService) -> None:
        self._web_search_service = web_search_service

    async def run(self, input: dict) -> ToolResult:
        query = str(input["query"])
        max_results = int(input.get("max_results", 5))
        result = await self._web_search_service.search(query, max_results=max_results)
        chunks = []
        for index, source in enumerate(result.sources[:max_results], start=1):
            raw_text = str(source.get("raw_content") or "")
            snippet_text = str(source.get("content") or "")
            text = " ".join((raw_text or snippet_text).split())
            if not text:
                continue
            title = str(source.get("title") or source.get("url") or f"Web source {index}")
            url = str(source.get("url") or "")
            content_source = "raw_content" if raw_text else "snippet"
            source_chunks = self._split_source_text(text)
            for chunk_index, chunk_text_value in enumerate(source_chunks):
                chunk_id = f"web:{index}" if len(source_chunks) == 1 else f"web:{index}:c{chunk_index}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text_value,
                        "metadata": {
                            "paper_id": url or chunk_id,
                            "title": title,
                            "chunk_id": chunk_id,
                            "url": url,
                            "source_type": "web_page",
                            "content_source": content_source,
                            "source_result_index": index,
                            "source_chunk_index": chunk_index,
                            "source_chunk_count": len(source_chunks),
                            "raw_content_chars": len(raw_text),
                        },
                        "score": self._optional_float(source.get("score")),
                        "retrieval_sources": ["web"],
                        "citation": {
                            "paper_id": url or chunk_id,
                            "title": title,
                            "chunk_id": chunk_id,
                            "url": url,
                            "source_type": "web_page",
                            "text": chunk_text_value,
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
                "raw_content_count": sum(1 for source in result.sources if source.get("raw_content")),
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

    @staticmethod
    def _split_source_text(text: str) -> list[str]:
        if len(text) <= WEB_CHUNK_SIZE:
            return [text]
        return chunk_text(text, chunk_size=WEB_CHUNK_SIZE, overlap=WEB_CHUNK_OVERLAP)
