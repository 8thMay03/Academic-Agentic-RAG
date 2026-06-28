from app.models.citation import Citation
from app.storage.chat_history_store import ChatHistoryStore


async def test_chat_history_store_persists_history_per_paper(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)

    await store.append_exchange(
        paper_id="paper-1",
        question="What is the method?",
        answer="It uses planning.",
        citations=[
            Citation(
                paper_id="paper-1",
                title="Agentic RAG",
                page_number=3,
                page=3,
                chunk_id="paper-1:p3:c0",
                text="Planning text.",
            )
        ],
    )
    await store.append_exchange(
        paper_id="paper-2",
        question="What is the limitation?",
        answer="Not specified.",
        citations=[],
    )

    paper_1_messages = await store.get_messages("paper-1")
    paper_2_messages = await store.get_messages("paper-2")

    assert [message.role for message in paper_1_messages] == ["user", "assistant"]
    assert paper_1_messages[0].content == "What is the method?"
    assert paper_1_messages[1].citations[0].page_number == 3
    assert paper_2_messages[0].content == "What is the limitation?"


async def test_chat_history_store_clears_paper_history(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)
    await store.append_exchange("paper-1", "Question?", "Answer.", [])

    await store.clear("paper-1")

    assert await store.get_messages("paper-1") == []
