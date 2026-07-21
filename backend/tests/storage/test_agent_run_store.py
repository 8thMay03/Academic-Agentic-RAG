import json

import pytest

from app.models.citation import Citation
from app.storage.agent_run_store import AgentRunStore


@pytest.mark.asyncio
async def test_agent_run_store_appends_and_lists_runs(tmp_path) -> None:
    store = AgentRunStore(tmp_path)

    record = await store.append_run(
        chat_id="chat-1",
        question="What is the method?",
        answer="It uses planning.",
        citations=[
            {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "chunk_id": "paper-1:p3:c0",
                "text": "Planning evidence.",
                "evidence_quality": "high",
            }
        ],
        trace=[
            {"stage": "generate_answer", "answer_chars": 17, "debug_prompt": "internal"},
            {"stage": "verify_answer", "success": True},
        ],
    )

    runs = await store.list_runs("chat-1")
    findings = await store.list_findings("chat-1")

    assert runs == [record]
    assert findings == record.findings
    assert record.findings[0].summary == "It uses planning."
    assert record.findings[0].source_ids == ["paper-1"]
    assert record.findings[0].citation_ids == ["paper-1:p3:c0"]
    assert record.findings[0].confidence == "high"
    assert record.citations[0] == Citation(
        paper_id="paper-1",
        title="Agentic RAG",
        chunk_id="paper-1:p3:c0",
        text="Planning evidence.",
        evidence_quality="high",
    )
    assert [event.model_dump(exclude_none=True) for event in record.trace] == [
        {"stage": "generate_answer", "answer_chars": 17},
        {"stage": "verify_answer", "success": True},
    ]
    path = tmp_path / "agent_runs" / "chat-1" / f"{record.run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["question"] == "What is the method?"
    assert payload["trace"] == [
        {"stage": "generate_answer", "answer_chars": 17},
        {"stage": "verify_answer", "success": True},
    ]
    assert payload["findings"][0]["summary"] == "It uses planning."


@pytest.mark.asyncio
async def test_agent_run_store_returns_empty_list_for_unknown_chat(tmp_path) -> None:
    store = AgentRunStore(tmp_path)

    assert await store.list_runs("missing-chat") == []
    assert await store.list_findings("missing-chat") == []


@pytest.mark.asyncio
async def test_agent_run_store_does_not_create_findings_for_unknown_answers(tmp_path) -> None:
    store = AgentRunStore(tmp_path)

    record = await store.append_run(
        chat_id="chat-1",
        question="What is missing?",
        answer="I don't know",
        citations=[],
        trace=[{"stage": "draft_answer", "status": "no_context"}],
    )

    assert record.findings == []
    assert await store.list_findings("chat-1") == []


@pytest.mark.asyncio
async def test_agent_run_store_lowers_confidence_from_verification_action(tmp_path) -> None:
    store = AgentRunStore(tmp_path)

    record = await store.append_run(
        chat_id="chat-1",
        question="What claims are unsupported?",
        answer="This needs more evidence.",
        citations=[
            {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "chunk_id": "paper-1:p3:c0",
                "text": "Planning evidence.",
                "evidence_quality": "high",
            }
        ],
        trace=[
            {
                "stage": "verify_answer",
                "status": "failed",
                "suggested_action": "retrieve_more",
                "unsupported_claim_count": 1,
            }
        ],
    )

    assert record.findings[0].confidence == "low"


@pytest.mark.asyncio
async def test_agent_run_store_marks_revised_answers_medium_confidence(tmp_path) -> None:
    store = AgentRunStore(tmp_path)

    record = await store.append_run(
        chat_id="chat-1",
        question="What was revised?",
        answer="The answer was grounded.",
        citations=[
            {
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "chunk_id": "paper-1:p3:c0",
                "text": "Planning evidence.",
                "evidence_quality": "high",
            }
        ],
        trace=[{"stage": "verify_answer", "suggested_action": "revise_answer"}],
    )

    assert record.findings[0].confidence == "medium"
