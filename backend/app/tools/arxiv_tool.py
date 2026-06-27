from app.services.search_service import SearchService


async def arxiv_search_tool(query: str, max_results: int = 5):
    return await SearchService().search(query, max_results)

