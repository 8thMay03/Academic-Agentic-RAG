from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_history_store, get_chat_service
from app.models.chat import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.storage.chat_history_store import ChatHistoryStore

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatResponse:
    answer, citations = await chat_service.answer(
        question=request.question,
        paper_ids=request.paper_ids,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    if request.paper_ids:
        await history_store.append_exchange(
            paper_id=request.paper_ids[0],
            question=request.question,
            answer=answer,
            citations=citations,
        )
    return ChatResponse(answer=answer, citations=citations)


@router.get("/history/{paper_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    paper_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatHistoryResponse:
    return ChatHistoryResponse(
        paper_id=paper_id,
        messages=await history_store.get_messages(paper_id),
    )


@router.delete("/history/{paper_id}", response_model=ChatHistoryResponse)
async def clear_chat_history(
    paper_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatHistoryResponse:
    await history_store.clear(paper_id)
    return ChatHistoryResponse(paper_id=paper_id, messages=[])
