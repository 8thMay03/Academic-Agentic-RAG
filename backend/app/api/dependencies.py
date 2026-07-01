from app.services.agentic_chat_workflow import AgenticChatWorkflow
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.pdf_index_service import PDFIndexService
from app.services.rag_service import RAGService
from app.services.retriever_service import RetrieverService
from app.services.web_search_service import WebSearchService
from app.storage.chat_history_store import ChatHistoryStore


def get_llm_service() -> LLMService:
    return LLMService()


def get_pdf_service() -> PDFService:
    return PDFService()


def get_pdf_index_service() -> PDFIndexService:
    return PDFIndexService()


def get_retriever_service() -> RetrieverService:
    return RetrieverService()


def get_rag_service() -> RAGService:
    return RAGService(get_retriever_service(), get_llm_service())


def get_chat_service() -> ChatService:
    llm_service = get_llm_service()
    workflow = AgenticChatWorkflow(
        RAGService(get_retriever_service(), llm_service),
        llm_service,
        WebSearchService(),
    )
    return ChatService(workflow)


def get_chat_history_store() -> ChatHistoryStore:
    return ChatHistoryStore()
