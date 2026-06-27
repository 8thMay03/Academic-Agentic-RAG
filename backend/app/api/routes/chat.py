from fastapi import APIRouter

from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_with_papers(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer="RAG chat is not implemented yet.",
        citations=[],
    )

