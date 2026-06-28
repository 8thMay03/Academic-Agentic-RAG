from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    answer, citations = await chat_service.answer(
        question=request.question,
        paper_ids=request.paper_ids,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    return ChatResponse(answer=answer, citations=citations)
