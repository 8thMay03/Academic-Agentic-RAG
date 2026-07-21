from collections.abc import AsyncIterator

from app.agent.citations import CitationGrounder
from app.agent.evaluators.answer_verifier import AnswerVerifier
from app.agent.evaluators.context_quality import ContextQualityEvaluator
from app.agent.models import (
    AgentTraceEvent,
    ChatWorkflowRequest,
    ChatWorkflowResult,
)
from app.agent.state import AgenticRAGState
from app.agent.prompts.answer_prompt import AnswerPromptBuilder
from app.agent.tools.arxiv_search_tool import ArxivSearchTool
from app.agent.tools.local_retrieve_tool import LocalRetrieveTool
from app.agent.tools.pdf_download_tool import PDFDownloadTool
from app.agent.tools.pdf_index_tool import PDFIndexTool
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.tools.web_snippet_ingest_tool import WebSnippetIngestTool
from app.models.citation import Citation
from app.services.llm_service import LLMService
from app.services.pdf_index_service import PDFIndexService
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService
from app.services.search_arxiv_service import SearchArxivService
from app.services.web_search_service import WebSearchService

UNKNOWN_ANSWER = "I don't know"


class AgenticChatWorkflow:
    def __init__(
        self,
        rag_service: RAGService,
        llm_service: LLMService,
        web_search_service: WebSearchService | None = None,
        citation_grounder: CitationGrounder | None = None,
        quality_evaluator: ContextQualityEvaluator | None = None,
        answer_verifier: AnswerVerifier | None = None,
        prompt_builder: AnswerPromptBuilder | None = None,
        search_arxiv_service: SearchArxivService | None = None,
        pdf_service: PDFService | None = None,
        pdf_index_service: PDFIndexService | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._llm_service = llm_service
        resolved_web_search_service = web_search_service or WebSearchService()
        self._citation_grounder = citation_grounder or CitationGrounder()
        self._quality_evaluator = quality_evaluator or ContextQualityEvaluator(llm_service)
        self._answer_verifier = answer_verifier or AnswerVerifier(self._citation_grounder)
        self._prompt_builder = prompt_builder or AnswerPromptBuilder()
        self._tool_registry = tool_registry or ToolRegistry(
            [
                LocalRetrieveTool(rag_service),
                WebSearchTool(resolved_web_search_service),
                WebSnippetIngestTool(),
                ArxivSearchTool(search_arxiv_service or SearchArxivService()),
                PDFDownloadTool(pdf_service or PDFService()),
                PDFIndexTool(pdf_index_service or PDFIndexService()),
            ]
        )

    async def run(self, request: ChatWorkflowRequest) -> ChatWorkflowResult:
        from app.agent.graph import run_verified_agentic_rag_workflow

        return await run_verified_agentic_rag_workflow(self, request)

    def initial_state(self, request: ChatWorkflowRequest) -> AgenticRAGState:
        return {
            "request": request,
            "llm_service": self._llm_service,
            "answer_verifier": self._answer_verifier,
            "citation_grounder": self._citation_grounder,
            "prompt_builder": self._prompt_builder,
            "quality_evaluator": self._quality_evaluator,
            "tool_registry": self._tool_registry,
            "trace": [],
        }

    async def stream(self, request: ChatWorkflowRequest) -> tuple[AsyncIterator[str], list[Citation], list[AgentTraceEvent]]:
        result = await self.run(request)
        return (
            self._token_stream(result.answer),
            result.citations,
            result.trace,
        )

    @staticmethod
    async def _token_stream(answer: str) -> AsyncIterator[str]:
        words = answer.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else f"{word} "


