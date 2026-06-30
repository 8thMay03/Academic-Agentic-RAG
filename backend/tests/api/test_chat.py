from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_history_store, get_chat_service
from app.main import app
from app.models.chat import ChatHistoryMessage, ChatSession, ChatSource, ChatThread
from app.models.citation import Citation


class FakeChatService:
    async def answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ) -> tuple[str, list[Citation]]:
        assert question == "What is the method?"
        assert paper_ids == ["paper-1"]
        assert top_k == 3
        assert score_threshold == 0.7
        assert chat_history == [
            ChatHistoryMessage(
                role="user",
                content="Previous question?",
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
        return (
            "It uses planning for retrieval decisions (p. 3).",
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
        )

    async def stream_answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
        chat_history: list[ChatHistoryMessage] | None = None,
    ):
        assert question == "What is the method?"
        assert paper_ids == ["paper-1"]
        assert top_k == 3
        assert score_threshold == 0.7
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
        )


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
            title="Research chat",
            sources=[ChatSource(paper_id="paper-1", title="Agentic RAG.pdf")],
            messages=self.messages,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    async def append_exchange(self, paper_id, question, answer, citations):
        self.appended = {
            "paper_id": paper_id,
            "question": question,
            "answer": answer,
            "citations": citations,
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


def test_chat_with_papers_returns_answer_and_citations() -> None:
    history_store = FakeChatHistoryStore()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the method?",
            "paper_ids": ["paper-1"],
            "top_k": 3,
            "score_threshold": 0.7,
        },
    )

    app.dependency_overrides.clear()

    assert history_store.appended["paper_id"] == "paper-1"
    assert history_store.appended["question"] == "What is the method?"
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
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }


def test_stream_chat_with_papers_returns_token_events_and_persists_history() -> None:
    history_store = FakeChatHistoryStore()
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    app.dependency_overrides[get_chat_history_store] = lambda: history_store
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "question": "What is the method?",
            "paper_ids": ["paper-1"],
            "top_k": 3,
            "score_threshold": 0.7,
        },
    ) as response:
        body = response.read().decode("utf-8")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert '"type": "token", "content": "It "' in body
    assert '"type": "citations"' in body
    assert '"type": "done"' in body
    assert history_store.appended["answer"] == "It uses planning."


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
