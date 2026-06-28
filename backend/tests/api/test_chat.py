from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.main import app
from app.models.citation import Citation


class FakeChatService:
    async def answer(
        self,
        question: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.65,
    ) -> tuple[str, list[Citation]]:
        assert question == "What is the method?"
        assert paper_ids == ["paper-1"]
        assert top_k == 3
        assert score_threshold == 0.7
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
                )
            ],
        )


def test_chat_with_papers_returns_answer_and_citations() -> None:
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
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
            }
        ],
    }
