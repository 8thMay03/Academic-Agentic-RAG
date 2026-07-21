from app.agent.models import ToolResult, normalize_retrieved_chunks
from app.models.chat import ChatHistoryMessage
from app.services.rag_service import RAGService


class LocalRetrieveTool:
    name = "local_retrieve"

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service

    async def run(self, input: dict) -> ToolResult:
        chunks = normalize_retrieved_chunks(
            await self._rag_service.retrieve_context(
                question=str(input["question"]),
                chat_id=input.get("chat_id"),
                paper_ids=input.get("paper_ids"),
                top_k=int(input.get("top_k", 5)),
                score_threshold=input.get("score_threshold", 0.65),
                chat_history=input.get("chat_history"),
            )
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            chunks=chunks,
            metadata={
                "chunk_count": len(chunks),
                "paper_ids": input.get("paper_ids"),
            },
        )


def local_retrieve_input(
    question: str,
    chat_id: str | None,
    paper_ids: list[str] | None,
    top_k: int,
    score_threshold: float | None,
    chat_history: list[ChatHistoryMessage] | None,
) -> dict:
    return {
        "question": question,
        "chat_id": chat_id,
        "paper_ids": paper_ids,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "chat_history": chat_history,
    }
