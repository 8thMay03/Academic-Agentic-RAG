from collections.abc import AsyncIterator

from app.models.chat import ChatHistoryMessage
from app.models.citation import Citation
from app.services.agentic_chat_workflow import (
    AgenticChatWorkflow,
    ChatWorkflowRequest,
)


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
    ) -> tuple[str, list[Citation]]:
        result = await self._workflow.run(
            ChatWorkflowRequest(
                question=question,
                paper_ids=paper_ids,
                top_k=top_k,
                score_threshold=score_threshold,
                chat_history=chat_history,
            )
        )
        return result.answer, result.citations

    async def stream_answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[AsyncIterator[str], list[Citation]]:
        token_stream, citations, _trace = await self._workflow.stream(
            ChatWorkflowRequest(
                question=question,
                paper_ids=paper_ids,
                top_k=top_k,
                score_threshold=score_threshold,
                chat_history=chat_history,
            )
        )
        return token_stream, citations
