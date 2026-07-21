import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_agent_run_store, get_chat_history_store, get_chat_service
from app.models.chat import (
    AgentRunListResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreateRequest,
    ChatSessionUpdateRequest,
    ChatSourceAddRequest,
    ChatThreadListResponse,
    ResearchFindingListResponse,
    agent_trace_event_payload,
    agent_trace_payload,
)
from app.services.chat_service import ChatService
from app.storage.agent_run_store import AgentRunStore
from app.storage.chat_history_store import ChatHistoryStore

router = APIRouter()


@router.post("", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
    agent_run_store: AgentRunStore = Depends(get_agent_run_store),
) -> ChatResponse:
    paper_ids = request.paper_ids
    history_key = request.chat_id or (request.paper_ids[0] if request.paper_ids else None)
    chat_history = []
    if request.chat_id:
        session = await history_store.get_session(request.chat_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {request.chat_id}")
        chat_history = session.messages
    elif history_key:
        chat_history = await history_store.get_messages(history_key)

    result = await chat_service.answer(
        question=request.question,
        paper_ids=paper_ids,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        chat_history=chat_history,
        max_agent_steps=request.max_agent_steps,
        enable_web_search=request.enable_web_search,
        enable_research_ingest=request.enable_research_ingest,
        auto_download_pdfs=request.auto_download_pdfs,
    )
    trace = agent_trace_payload(result.trace)
    if history_key:
        await history_store.append_exchange(
            paper_id=history_key,
            question=request.question,
            answer=result.answer,
            citations=result.citations,
            trace=trace,
        )
        await agent_run_store.append_run(
            chat_id=history_key,
            question=request.question,
            answer=result.answer,
            citations=result.citations,
            trace=trace,
        )
    return ChatResponse(answer=result.answer, citations=result.citations, trace=trace)


@router.post("/stream")
async def stream_chat_with_papers(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
    agent_run_store: AgentRunStore = Depends(get_agent_run_store),
) -> StreamingResponse:
    paper_ids = request.paper_ids
    history_key = request.chat_id or (request.paper_ids[0] if request.paper_ids else None)
    chat_history = []
    if request.chat_id:
        session = await history_store.get_session(request.chat_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {request.chat_id}")
        chat_history = session.messages
    elif history_key:
        chat_history = await history_store.get_messages(history_key)

    async def event_stream() -> AsyncIterator[str]:
        answer_parts: list[str] = []
        citations = []
        trace = []
        try:
            async for event in chat_service.stream_events(
                question=request.question,
                paper_ids=paper_ids,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                chat_history=chat_history,
                max_agent_steps=request.max_agent_steps,
                enable_web_search=request.enable_web_search,
                enable_research_ingest=request.enable_research_ingest,
                auto_download_pdfs=request.auto_download_pdfs,
            ):
                if event["type"] == "agent_step":
                    step = agent_trace_event_payload(event["step"])
                    trace.append(step)
                    yield _stream_event("agent_step", **step)
                elif event["type"] == "token":
                    token = event["content"]
                    answer_parts.append(token)
                    yield _stream_event("token", content=token)
                elif event["type"] == "citations":
                    citations = event["citations"]
                    yield _stream_event(
                        "citations",
                        citations=[citation.model_dump(mode="json", exclude_none=True) for citation in citations],
                    )
                elif event["type"] == "result":
                    result = event["result"]
                    citations = result.citations

            answer = "".join(answer_parts)
            if history_key:
                await history_store.append_exchange(
                    paper_id=history_key,
                    question=request.question,
                    answer=answer,
                    citations=citations,
                    trace=trace,
                )
                await agent_run_store.append_run(
                    chat_id=history_key,
                    question=request.question,
                    answer=answer,
                    citations=citations,
                    trace=trace,
                )
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


@router.get("/sessions/{chat_id}/runs", response_model=AgentRunListResponse, response_model_exclude_none=True)
async def list_agent_runs(
    chat_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
    agent_run_store: AgentRunStore = Depends(get_agent_run_store),
) -> AgentRunListResponse:
    session = await history_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")
    return AgentRunListResponse(runs=await agent_run_store.list_runs(chat_id))


@router.get("/sessions/{chat_id}/findings", response_model=ResearchFindingListResponse)
async def list_research_findings(
    chat_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
    agent_run_store: AgentRunStore = Depends(get_agent_run_store),
) -> ResearchFindingListResponse:
    session = await history_store.get_session(chat_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat not found: {chat_id}")
    return ResearchFindingListResponse(findings=await agent_run_store.list_findings(chat_id))


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
