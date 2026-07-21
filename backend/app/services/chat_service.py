from collections.abc import AsyncIterator

from app.agent.models import AgentTraceEvent
from app.agent.workflow import (
    AgenticChatWorkflow,
    ChatWorkflowRequest,
    ChatWorkflowResult,
)
from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation


class ChatService:
    def __init__(self, workflow: AgenticChatWorkflow) -> None:
        self._workflow = workflow

    async def answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
        max_agent_steps: int = 6,
        enable_web_search: bool = True,
        enable_research_ingest: bool = True,
        auto_download_pdfs: bool = True,
    ) -> ChatWorkflowResult:
        return await self._workflow.run(
            ChatWorkflowRequest(
                question=question,
                paper_ids=paper_ids,
                top_k=top_k,
                score_threshold=score_threshold,
                chat_history=chat_history,
                max_agent_steps=max_agent_steps,
                enable_web_search=enable_web_search,
                enable_research_ingest=enable_research_ingest,
                auto_download_pdfs=auto_download_pdfs,
            )
        )

    async def stream_answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
        max_agent_steps: int = 6,
        enable_web_search: bool = True,
        enable_research_ingest: bool = True,
        auto_download_pdfs: bool = True,
    ) -> tuple[AsyncIterator[str], list[Citation], list[AgentTraceEvent]]:
        return await self._workflow.stream(
            ChatWorkflowRequest(
                question=question,
                paper_ids=paper_ids,
                top_k=top_k,
                score_threshold=score_threshold,
                chat_history=chat_history,
                max_agent_steps=max_agent_steps,
                enable_web_search=enable_web_search,
                enable_research_ingest=enable_research_ingest,
                auto_download_pdfs=auto_download_pdfs,
            )
        )
