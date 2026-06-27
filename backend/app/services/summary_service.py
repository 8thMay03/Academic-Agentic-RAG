from app.services.llm_service import LLMService


class SummaryService:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def summarize(self, title: str, text: str) -> str:
        prompt = (
            "Summarize this research paper using Problem, Method, "
            "Contributions, Experiments, Limitations, and Relevance.\n\n"
            f"Title: {title}\n\n{text}"
        )
        return await self.llm_service.complete(prompt)

