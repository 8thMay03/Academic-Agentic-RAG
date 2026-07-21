import pytest

from app.agent.models import ContextQuality
from app.agent.nodes.quality_gate_node import quality_gate_node
from app.agent.workflow import ChatWorkflowRequest


class FakeQualityEvaluator:
    def __init__(self, quality: ContextQuality) -> None:
        self.quality = quality
        self.calls = []

    async def evaluate(self, question: str, chunks: list[dict]) -> ContextQuality:
        self.calls.append({"question": question, "chunks": chunks})
        return self.quality


@pytest.mark.asyncio
async def test_quality_gate_node_uses_explicit_evaluator_dependency():
    chunks = [{"text": "Agentic RAG plans retrieval.", "score": 0.9}]
    evaluator = FakeQualityEvaluator(
        ContextQuality(
            sufficient=True,
            chunk_count=1,
            context_chars=29,
            reason="strong_context",
            top_score=0.9,
            average_score=0.9,
            source_count=1,
            query_coverage=0.75,
        )
    )

    state = await quality_gate_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG retrieve?"),
            "local_chunks": chunks,
            "quality_evaluator": evaluator,
            "trace": [],
        }
    )

    assert evaluator.calls == [
        {"question": "How does Agentic RAG retrieve?", "chunks": chunks}
    ]
    assert state["quality"].reason == "strong_context"
    assert state["trace"][-1] == {
        "stage": "quality_gate",
        "sufficient": True,
        "reason": "strong_context",
        "chunk_count": 1,
        "context_chars": 29,
        "top_score": 0.9,
        "average_score": 0.9,
        "source_count": 1,
        "query_coverage": 0.75,
        "self_check_used": False,
        "self_check_passed": None,
    }
