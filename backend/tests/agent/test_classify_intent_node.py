from app.agent.models import ChatWorkflowRequest
from app.agent.nodes.classify_intent_node import classify_intent_node
from app.models.chat import ChatHistoryMessage


async def test_classify_intent_node_detects_fresh_research_requests() -> None:
    state = await classify_intent_node(
        {"request": ChatWorkflowRequest("What is the latest Agentic RAG approach?"), "trace": []}
    )

    assert state["intent"] == "fresh_research"
    assert state["trace"] == [
        {
            "stage": "classify_intent",
            "intent": "fresh_research",
            "reason": "question_requests_current_or_recent_information",
        }
    ]


async def test_classify_intent_node_detects_follow_up_questions() -> None:
    state = await classify_intent_node(
        {
            "request": ChatWorkflowRequest(
                "How does it compare?",
                chat_history=[
                    ChatHistoryMessage(
                        role="user",
                        content="Explain Agentic RAG.",
                        created_at="2026-01-01T00:00:00+00:00",
                    )
                ],
            ),
            "trace": [],
        }
    )

    assert state["intent"] == "follow_up_research"


async def test_classify_intent_node_defaults_to_research_qa() -> None:
    state = await classify_intent_node(
        {"request": ChatWorkflowRequest("How does planning retrieve evidence?"), "trace": []}
    )

    assert state["intent"] == "research_qa"
