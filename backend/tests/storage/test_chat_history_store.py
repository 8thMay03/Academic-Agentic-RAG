from app.models.citation import Citation
from app.models.chat import ChatSource
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


async def test_chat_history_store_lists_threads(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)
    await store.append_exchange("paper-1", "What is the method?", "It uses planning.", [])
    await store.append_exchange("paper-2", "What is the limitation?", "Not specified.", [])

    threads = await store.list_threads()

    assert {thread.chat_id for thread in threads} == {"paper-1", "paper-2"}
    paper_1_thread = next(thread for thread in threads if thread.chat_id == "paper-1")
    assert paper_1_thread.title == "What is the method?"
    assert paper_1_thread.last_message == "It uses planning."
    assert paper_1_thread.message_count == 2


async def test_chat_history_store_adds_sources_to_session(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)
    session = await store.create_session("Literature review")

    updated_session = await store.add_source(
        session.chat_id,
        ChatSource(
            paper_id="paper-1",
            title="Agentic RAG.pdf",
            filename="Agentic RAG.pdf",
            path="data/pdfs/Agentic RAG.pdf",
        ),
    )

    assert updated_session.sources[0].paper_id == "paper-1"
    assert updated_session.sources[0].filename == "Agentic RAG.pdf"
    assert (await store.get_session(session.chat_id)).sources[0].title == "Agentic RAG.pdf"


async def test_chat_history_store_updates_session_title(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)
    session = await store.create_session("Draft title")

    updated_session = await store.update_session_title(session.chat_id, "  Better title  ")
    threads = await store.list_threads()

    assert updated_session.title == "Better title"
    assert (await store.get_session(session.chat_id)).title == "Better title"
    assert threads[0].title == "Better title"


async def test_chat_history_store_deletes_session(tmp_path) -> None:
    store = ChatHistoryStore(base_dir=tmp_path)
    session = await store.create_session("Disposable chat")

    deleted = await store.delete_session(session.chat_id)

    assert deleted is True
    assert await store.get_session(session.chat_id) is None
    assert await store.delete_session(session.chat_id) is False
