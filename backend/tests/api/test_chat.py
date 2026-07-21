from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_run_store, get_chat_history_store, get_chat_service
from app.agent.workflow import ChatWorkflowResult
from app.main import app
from app.models.chat import (
    AgentRunRecord,
    AgentTraceEventResponse,
    ChatHistoryMessage,
    ChatSession,
    ChatSource,
    ChatThread,
    ResearchFinding,
)
from app.models.citation import Citation


FAKE_TRACE = [
    {
        "stage": "local_retrieve",
        "chunk_count": 2,
        "paper_ids": ["paper-1"],
    },
    {
        "stage": "quality_gate",
        "sufficient": True,
        "reason": "strong_context",
    },
    {
        "stage": "generate_answer",
        "status": "completed",
        "success": True,
        "answer_chars": 47,
    },
    {
        "stage": "verify_answer",
        "status": "passed",
        "success": True,
    },
]
RAW_FAKE_TRACE = [
    *FAKE_TRACE[:2],
    {**FAKE_TRACE[2], "debug_prompt": "ignored at API boundary"},
    *FAKE_TRACE[3:],
]


class FakeChatService:
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
        assert question == "What is the method?"
        assert paper_ids == ["paper-1"]
        assert top_k == 3
        assert score_threshold == 0.7
        assert max_agent_steps == 4
        assert enable_web_search is False
        assert enable_research_ingest is True
        assert auto_download_pdfs is False
        assert chat_history == [
            ChatHistoryMessage(
                role="user",
                content="Previous question?",
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
        return ChatWorkflowResult(
            answer="It uses planning for retrieval decisions (p. 3).",
            citations=[
                Citation(
                    paper_id="paper-1",
                    title="Agentic RAG",
                    page_number=3,
                    page=3,
                    chunk_id="paper-1:p3:c0",
                    text="Agentic RAG uses planning.",
                    score=0.91,
                    rerank_score=0.93,
                    cross_encoder_score=2.6,
                    vector_score=0.9,
                    keyword_score=1.0,
                    reranker="fake-cross-encoder",
                    retrieval_sources=["keyword", "vector"],
                    evidence_quality="high",
                    matched_terms=["planning"],
                )
            ],
            trace=RAW_FAKE_TRACE,
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
    ):
        assert question == "What is the method?"
        assert paper_ids == ["paper-1"]
        assert top_k == 3
        assert score_threshold == 0.7
        assert max_agent_steps == 4
        assert enable_web_search is False
        assert enable_research_ingest is True
        assert auto_download_pdfs is False
        assert chat_history == [
            ChatHistoryMessage(
                role="user",
                content="Previous question?",
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]

        async def token_stream():
            for token in ["It ", "uses ", "planning."]:
                yield token

        return (
            token_stream(),
            [
                Citation(
                    paper_id="paper-1",
                    title="Agentic RAG",
                    page_number=3,
                    page=3,
                    chunk_id="paper-1:p3:c0",
                    text="Agentic RAG uses planning.",
                    score=0.91,
                    rerank_score=0.93,
                    cross_encoder_score=2.6,
                    vector_score=0.9,
                    keyword_score=1.0,
                    reranker="fake-cross-encoder",
                    retrieval_sources=["keyword", "vector"],
                    evidence_quality="high",
                    matched_terms=["planning"],
                )
            ],
            RAW_FAKE_TRACE,
        )

    async def stream_events(
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
    ):
        token_stream, citations, trace = await self.stream_answer(
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
        for step in trace:
            yield {"type": "agent_step", "step": step}
        async for token in token_stream:
            yield {"type": "token", "content": token}
        yield {"type": "citations", "citations": citations}
        yield {
            "type": "result",
            "result": ChatWorkflowResult(
                answer="It uses planning.",
                citations=citations,
                trace=trace,
            ),
        }


class FakeChatHistoryStore:
    def __init__(self) -> None:
        self.appended = None
        self.messages = [
            ChatHistoryMessage(
                role="user",
                content="Previous question?",
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
        self.session = ChatSession(
            chat_id="chat-1",
            title="Agentic RAG chat",
            sources=[ChatSource(paper_id="paper-1", title="Agentic RAG.pdf")],
            messages=self.messages,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    async def append_exchange(self, paper_id, question, answer, citations, trace=None):
        self.appended = {
            "paper_id": paper_id,
            "question": question,
            "answer": answer,
            "citations": citations,
            "trace": trace,
        }
        return self.messages

    async def get_messages(self, paper_id):
        assert paper_id == "paper-1"
        return self.messages

    async def clear(self, paper_id):
        assert paper_id == "paper-1"
        self.messages = []

    async def list_threads(self):
        return [
            ChatThread(
                chat_id="paper-1",
                title="Previous question?",
                last_message="Previous question?",
                updated_at="2026-01-01T00:00:00+00:00",
                message_count=1,
                source_count=1,
            )
        ]

    async def create_session(self, title=None):
        return self.session

    async def get_session(self, chat_id):
        assert chat_id == "chat-1"
        return self.session

    async def add_source(self, chat_id, source):
        assert chat_id == "chat-1"
        self.session.sources.append(source)
        return self.session

    async def remove_source(self, chat_id, paper_id):
        assert chat_id == "chat-1"
        self.session.sources = [source for source in self.session.sources if source.paper_id != paper_id]
        return self.session

    async def delete_session(self, chat_id):
        assert chat_id == "chat-1"
        return True

    async def update_session_title(self, chat_id, title):
        assert chat_id == "chat-1"
        self.session.title = title.strip()
        return self.session


class FakeAgentRunStore:
    def __init__(self) -> None:
        self.runs = []

    async def append_run(self, chat_id, question, answer, citations, trace):
        self.runs.append(
            {
                "chat_id": chat_id,
                "question": question,
                "answer": answer,
                "citations": citations,
                "trace": trace,
            }
        )

    async def list_runs(self, chat_id):
        return [
            run
            for run in self.runs
            if (run.chat_id if hasattr(run, "chat_id") else run["chat_id"]) == chat_id
        ]

    async def list_findings(self, chat_id):
        findings = []
        for run in await self.list_runs(chat_id):
            findings.extend(run.findings if hasattr(run, "findings") else run.get("findings", []))
        return findings


def test_chat_with_papers_returns_answer_and_citations() -> None:
    history_store = FakeChatHistoryStore()
    agent_run_store = FakeAgentRunStore()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    app.dependency_overrides[get_agent_run_store] = lambda: agent_run_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the method?",
            "paper_ids": ["paper-1"],
            "top_k": 3,
            "score_threshold": 0.7,
            "max_agent_steps": 4,
            "enable_web_search": False,
            "enable_research_ingest": True,
            "auto_download_pdfs": False,
        },
    )

    app.dependency_overrides.clear()

    assert history_store.appended["paper_id"] == "paper-1"
    assert history_store.appended["question"] == "What is the method?"
    assert history_store.appended["trace"] == FAKE_TRACE
    assert agent_run_store.runs[0]["chat_id"] == "paper-1"
    assert agent_run_store.runs[0]["trace"] == FAKE_TRACE
    assert response.status_code == 200
    assert response.json() == {
        "answer": "It uses planning for retrieval decisions (p. 3).",
        "citations": [
            {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "page_number": 3,
                "page": 3,
                "chunk_id": "paper-1:p3:c0",
                "text": "Agentic RAG uses planning.",
                "score": 0.91,
                "rerank_score": 0.93,
                "cross_encoder_score": 2.6,
                "vector_score": 0.9,
                "keyword_score": 1.0,
                "reranker": "fake-cross-encoder",
                "retrieval_sources": ["keyword", "vector"],
                "evidence_quality": "high",
                "matched_terms": ["planning"],
            }
        ],
        "trace": FAKE_TRACE,
    }


def test_chat_session_without_sources_searches_all_local_documents() -> None:
    class NoSourceChatService:
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
            assert question == "What is CRAG?"
            assert paper_ids is None
            assert chat_history == []
            assert max_agent_steps == 6
            assert enable_web_search is True
            assert enable_research_ingest is True
            assert auto_download_pdfs is True
            return ChatWorkflowResult(
                answer="CRAG corrects retrieved context.",
                citations=[],
                trace=[{"stage": "local_retrieve", "chunk_count": 0}],
            )

    class NoSourceHistoryStore(FakeChatHistoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.session.sources = []
            self.session.messages = []
            self.messages = []

    history_store = NoSourceHistoryStore()
    agent_run_store = FakeAgentRunStore()
    app.dependency_overrides[get_chat_service] = lambda: NoSourceChatService()
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    app.dependency_overrides[get_agent_run_store] = lambda: agent_run_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is CRAG?",
            "chat_id": "chat-1",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "CRAG corrects retrieved context.",
        "citations": [],
        "trace": [{"stage": "local_retrieve", "chunk_count": 0}],
    }
    assert history_store.appended["paper_id"] == "chat-1"
    assert history_store.appended["trace"] == [{"stage": "local_retrieve", "chunk_count": 0}]
    assert agent_run_store.runs[0]["chat_id"] == "chat-1"


def test_agent_trace_event_response_filters_unknown_fields() -> None:
    event = AgentTraceEventResponse(
        stage="generate_answer",
        step_count=2,
        answer_chars=47,
        source_type="arxiv",
        source_url="https://arxiv.org/abs/2601.12345",
        pdf_url="https://arxiv.org/pdf/2601.12345",
        trust_level="high",
        ingestion_status="downloaded",
        unknown="ignored",
    )

    assert event.model_dump(exclude_none=True) == {
        "stage": "generate_answer",
        "step_count": 2,
        "answer_chars": 47,
        "source_type": "arxiv",
        "source_url": "https://arxiv.org/abs/2601.12345",
        "pdf_url": "https://arxiv.org/pdf/2601.12345",
        "trust_level": "high",
        "ingestion_status": "downloaded",
    }


def test_get_chat_history_returns_messages() -> None:
    app.dependency_overrides[get_chat_history_store] = lambda: FakeChatHistoryStore()
    client = TestClient(app)

    response = client.get("/api/v1/chat/history/paper-1")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "paper_id": "paper-1",
        "messages": [
                {
                    "role": "user",
                    "content": "Previous question?",
                    "citations": [],
                    "trace": [],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
        ],
    }


def test_list_agent_runs_returns_persisted_runs() -> None:
    history_store = FakeChatHistoryStore()
    agent_run_store = FakeAgentRunStore()
    agent_run_store.runs.append(
        AgentRunRecord(
            run_id="run-1",
            chat_id="chat-1",
            question="What is the method?",
            answer="It uses planning.",
            citations=[],
            trace=FAKE_TRACE,
            findings=[
                ResearchFinding(
                    finding_id="run-1:f0",
                    chat_id="chat-1",
                    run_id="run-1",
                    question="What is the method?",
                    summary="It uses planning.",
                    source_ids=["paper-1"],
                    citation_ids=["paper-1:p3:c0"],
                    confidence="high",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    app.dependency_overrides[get_agent_run_store] = lambda: agent_run_store
    client = TestClient(app)

    response = client.get("/api/v1/chat/sessions/chat-1/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "runs": [
            {
                "run_id": "run-1",
                "chat_id": "chat-1",
                "question": "What is the method?",
                "answer": "It uses planning.",
                "citations": [],
                "trace": FAKE_TRACE,
                "findings": [
                    {
                        "finding_id": "run-1:f0",
                        "chat_id": "chat-1",
                        "run_id": "run-1",
                        "question": "What is the method?",
                        "summary": "It uses planning.",
                        "source_ids": ["paper-1"],
                        "citation_ids": ["paper-1:p3:c0"],
                        "confidence": "high",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    }


def test_list_agent_runs_returns_not_found_for_missing_chat() -> None:
    class MissingChatHistoryStore(FakeChatHistoryStore):
        async def get_session(self, chat_id):
            return None

    app.dependency_overrides[get_chat_history_store] = lambda: MissingChatHistoryStore()
    app.dependency_overrides[get_agent_run_store] = lambda: FakeAgentRunStore()
    client = TestClient(app)

    response = client.get("/api/v1/chat/sessions/missing/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_list_research_findings_returns_run_findings() -> None:
    history_store = FakeChatHistoryStore()
    agent_run_store = FakeAgentRunStore()
    agent_run_store.runs.append(
        AgentRunRecord(
            run_id="run-1",
            chat_id="chat-1",
            question="What is the method?",
            answer="It uses planning.",
            citations=[],
            trace=FAKE_TRACE,
            findings=[
                ResearchFinding(
                    finding_id="run-1:f0",
                    chat_id="chat-1",
                    run_id="run-1",
                    question="What is the method?",
                    summary="It uses planning.",
                    source_ids=["paper-1"],
                    citation_ids=["paper-1:p3:c0"],
                    confidence="high",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    app.dependency_overrides[get_agent_run_store] = lambda: agent_run_store
    client = TestClient(app)

    response = client.get("/api/v1/chat/sessions/chat-1/findings")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "findings": [
            {
                "finding_id": "run-1:f0",
                "chat_id": "chat-1",
                "run_id": "run-1",
                "question": "What is the method?",
                "summary": "It uses planning.",
                "source_ids": ["paper-1"],
                "citation_ids": ["paper-1:p3:c0"],
                "confidence": "high",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    }


def test_stream_chat_with_papers_returns_token_events_and_persists_history() -> None:
    history_store = FakeChatHistoryStore()
    agent_run_store = FakeAgentRunStore()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    app.dependency_overrides[get_agent_run_store] = lambda: agent_run_store
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "question": "What is the method?",
            "paper_ids": ["paper-1"],
            "top_k": 3,
            "score_threshold": 0.7,
            "max_agent_steps": 4,
            "enable_web_search": False,
            "enable_research_ingest": True,
            "auto_download_pdfs": False,
        },
    ) as response:
        body = response.read().decode("utf-8")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert '"type": "agent_step", "stage": "local_retrieve"' in body
    assert '"type": "agent_step", "stage": "verify_answer"' in body
    assert body.index('"type": "agent_step"') < body.index('"type": "token"')
    assert '"answer_chars": 47' in body
    assert "debug_prompt" not in body
    assert '"type": "token", "content": "It "' in body
    assert '"type": "citations"' in body
    assert '"type": "done"' in body
    assert history_store.appended["answer"] == "It uses planning."
    assert history_store.appended["trace"] == FAKE_TRACE
    assert agent_run_store.runs[0]["answer"] == "It uses planning."
    assert agent_run_store.runs[0]["trace"] == FAKE_TRACE


def test_list_chat_history_returns_threads() -> None:
    app.dependency_overrides[get_chat_history_store] = lambda: FakeChatHistoryStore()
    client = TestClient(app)

    response = client.get("/api/v1/chat/history")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "chats": [
            {
                "chat_id": "paper-1",
                "title": "Previous question?",
                "last_message": "Previous question?",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "message_count": 1,
                "source_count": 1,
            }
        ]
    }


def test_clear_chat_history_deletes_messages() -> None:
    app.dependency_overrides[get_chat_history_store] = lambda: FakeChatHistoryStore()
    client = TestClient(app)

    response = client.delete("/api/v1/chat/history/paper-1")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"paper_id": "paper-1", "messages": []}


def test_delete_chat_session_removes_chat() -> None:
    app.dependency_overrides[get_chat_history_store] = lambda: FakeChatHistoryStore()
    client = TestClient(app)

    response = client.delete("/api/v1/chat/sessions/chat-1")

    app.dependency_overrides.clear()

    assert response.status_code == 204


def test_update_chat_session_renames_chat() -> None:
    app.dependency_overrides[get_chat_history_store] = lambda: FakeChatHistoryStore()
    client = TestClient(app)

    response = client.patch("/api/v1/chat/sessions/chat-1", json={"title": "Updated title"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["title"] == "Updated title"
