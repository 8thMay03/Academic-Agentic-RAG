from app.services.llm_service import LLMService
from app.services.summary_service import SummaryService


async def summary_tool(title: str, text: str) -> str:
    return await SummaryService(LLMService()).summarize(title, text)

