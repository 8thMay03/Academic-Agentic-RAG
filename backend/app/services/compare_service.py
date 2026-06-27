from app.models.paper import Paper
from app.services.llm_service import LLMService


class CompareService:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def compare(self, papers: list[Paper]) -> str:
        titles = "\n".join(f"- {paper.title}" for paper in papers)
        prompt = f"Compare these papers in a concise markdown table:\n{titles}"
        return await self.llm_service.complete(prompt)

