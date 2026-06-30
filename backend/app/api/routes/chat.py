import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_history_store, get_chat_service
from app.models.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreateRequest,
    ChatSessionUpdateRequest,
    ChatSourceAddRequest,
    ChatThreadListResponse,
)
from app.services.chat_service import ChatService
from app.storage.chat_history_store import ChatHistoryStore

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatResponse:
    paper_ids = request.paper_ids
    history_key = request.chat_id or (request.paper_ids[0] if request.paper_ids else None)
    chat_history = []
    if request.chat_id:
        session = await history_store.get_session(request.chat_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {request.chat_id}")
        paper_ids = paper_ids or [source.paper_id for source in session.sources]
        chat_history = session.messages
    elif history_key:
        chat_history = await history_store.get_messages(history_key)

    answer, citations = await chat_service.answer(
        question=request.question,
        paper_ids=paper_ids,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        chat_history=chat_history,
    )
    if history_key:
        await history_store.append_exchange(
            paper_id=history_key,
            question=request.question,
            answer=answer,
            citations=citations,
        )
    return ChatResponse(answer=answer, citations=citations)


@router.post("/stream")
async def stream_chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> StreamingResponse:
    paper_ids = request.paper_ids
    history_key = request.chat_id or (request.paper_ids[0] if request.paper_ids else None)
    chat_history = []
    if request.chat_id:
        session = await history_store.get_session(request.chat_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {request.chat_id}")
        paper_ids = paper_ids or [source.paper_id for source in session.sources]
        chat_history = session.messages
    elif history_key:
        chat_history = await history_store.get_messages(history_key)

    async def event_stream() -> AsyncIterator[str]:
        answer_parts: list[str] = []
        try:
            token_stream, citations = await chat_service.stream_answer(
                question=request.question,
                paper_ids=paper_ids,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                chat_history=chat_history,
            )
            async for token in token_stream:
                answer_parts.append(token)
                yield _stream_event("token", content=token)

            answer = "".join(answer_parts)
            if history_key:
                await history_store.append_exchange(
                    paper_id=history_key,
                    question=request.question,
                    answer=answer,
                    citations=citations,
                )
            yield _stream_event("citations", citations=[citation.model_dump(mode="json") for citation in citations])
            yield _stream_event("done")
        except Exception as exc:
            yield _stream_event("error", message=str(exc))

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", response_model=ChatThreadListResponse)
async def list_chat_history(
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatThreadListResponse:
    return ChatThreadListResponse(chats=await history_store.list_threads())


@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatSession:
    return await history_store.create_session(request.title)


@router.get("/sessions/{chat_id}", response_model=ChatSession)
async def get_chat_session(
    chat_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatSession:
    session = await history_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")
    return session


@router.delete("/sessions/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    chat_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> None:
    deleted = await history_store.delete_session(chat_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")


@router.patch("/sessions/{chat_id}", response_model=ChatSession)
async def update_chat_session(
    chat_id: str,
    request: ChatSessionUpdateRequest,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatSession:
    try:
        session = await history_store.update_session_title(chat_id, request.title)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")
    return session


@router.post("/sessions/{chat_id}/sources", response_model=ChatSession)
async def add_chat_source(
    chat_id: str,
    request: ChatSourceAddRequest,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatSession:
    return await history_store.add_source(chat_id, request)


@router.delete("/sessions/{chat_id}/sources/{paper_id}", response_model=ChatSession)
async def remove_chat_source(
    chat_id: str,
    paper_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> ChatSession:
    session = await history_store.remove_source(chat_id, paper_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")
    return session


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


def _stream_event(event_type: str, **payload: object) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"
