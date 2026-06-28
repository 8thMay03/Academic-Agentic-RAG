from app.services.chat_service import ChatService
from app.services.compare_service import CompareService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.pdf_index_service import PDFIndexService
from app.services.retriever_service import RetrieverService
from app.services.search_service import SearchService
from app.services.summary_service import SummaryService
from app.storage.chat_history_store import ChatHistoryStore


def get_search_service() -> SearchService:
    return SearchService()


def get_llm_service() -> LLMService:
    return LLMService()


def get_summary_service() -> SummaryService:
    return SummaryService(get_llm_service())


def get_compare_service() -> CompareService:
    return CompareService(get_llm_service())


def get_pdf_service() -> PDFService:
    return PDFService()


def get_pdf_index_service() -> PDFIndexService:
    return PDFIndexService()


def get_retriever_service() -> RetrieverService:
    return RetrieverService()


def get_chat_service() -> ChatService:
    return ChatService(get_retriever_service(), get_llm_service())


def get_chat_history_store() -> ChatHistoryStore:
    return ChatHistoryStore()
