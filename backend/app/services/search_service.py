from app.models.paper import Paper


class SearchService:
    async def search(self, query: str, max_results: int) -> list[Paper]:
        # TODO: Replace with arXiv API integration.
        return []

