from app.agent.models import ToolResult
from app.services.search_arxiv_service import SearchArxivService


class ArxivSearchTool:
    name = "arxiv_search"

    def __init__(self, search_arxiv_service: SearchArxivService) -> None:
        self._search_arxiv_service = search_arxiv_service

    async def run(self, input: dict) -> ToolResult:
        query = str(input["query"])
        papers = await self._search_arxiv_service.search(
            query=query,
            max_results=int(input.get("max_results", 5)),
            sort_by=str(input.get("sort_by", "submittedDate")),
        )
        artifacts = []
        for paper in papers:
            artifact = paper.model_dump(mode="json", exclude_none=True)
            artifact.update(
                {
                    "source_type": "arxiv",
                    "source_url": artifact.get("arxiv_url") or artifact.get("url"),
                    "discovered_by_query": query,
                    "trust_level": "high",
                    "ingestion_status": "discovered",
                }
            )
            artifacts.append(artifact)
        return ToolResult(
            tool_name=self.name,
            success=True,
            artifacts=artifacts,
            metadata={
                "paper_count": len(artifacts),
                "pdf_urls": [artifact["pdf_url"] for artifact in artifacts if artifact.get("pdf_url")],
            },
        )
