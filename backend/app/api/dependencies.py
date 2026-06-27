from app.services.compare_service import CompareService
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.services.summary_service import SummaryService


def get_search_service() -> SearchService:
    return SearchService()


def get_llm_service() -> LLMService:
    return LLMService()


def get_summary_service() -> SummaryService:
    return SummaryService(get_llm_service())


def get_compare_service() -> CompareService:
    return CompareService(get_llm_service())

