from app.vectorstore.keyword_index import PersistentKeywordIndex


def test_persistent_keyword_index_upserts_and_searches_documents(tmp_path) -> None:
    index = PersistentKeywordIndex(tmp_path / "keyword_index.json")

    index.upsert(
        documents=[
            "Agentic RAG planning retrieves evidence.",
            "Vision transformer benchmark results.",
        ],
        metadatas=[
            {
                "chunk_id": "paper-1:p1:c0",
                "paper_id": "paper-1",
                "title": "Agentic RAG",
                "page_number": "1",
            },
            {
                "chunk_id": "paper-2:p1:c0",
                "paper_id": "paper-2",
                "title": "Vision Transformer",
                "page_number": "2",
            },
        ],
        ids=["paper-1:p1:c0", "paper-2:p1:c0"],
    )

    results = index.search("agentic rag planning", top_k=2)

    assert [result["id"] for result in results] == ["paper-1:p1:c0"]
    assert results[0]["keyword_score"] == 1.0
    assert results[0]["citation"]["page_number"] == 1


def test_persistent_keyword_index_filters_by_paper_id(tmp_path) -> None:
    index = PersistentKeywordIndex(tmp_path / "keyword_index.json")
    index.upsert(
        documents=[
            "Agentic RAG planning retrieves evidence.",
            "Agentic RAG benchmark results.",
        ],
        metadatas=[
            {"chunk_id": "paper-1:p1:c0", "paper_id": "paper-1"},
            {"chunk_id": "paper-2:p1:c0", "paper_id": "paper-2"},
        ],
        ids=["paper-1:p1:c0", "paper-2:p1:c0"],
    )

    results = index.search("agentic rag", top_k=5, paper_ids=["paper-2"])

    assert [result["id"] for result in results] == ["paper-2:p1:c0"]


def test_persistent_keyword_index_replaces_existing_documents(tmp_path) -> None:
    index = PersistentKeywordIndex(tmp_path / "keyword_index.json")
    index.upsert(["old content"], [{"chunk_id": "chunk-1", "paper_id": "paper-1"}], ["chunk-1"])
    index.upsert(["new agentic content"], [{"chunk_id": "chunk-1", "paper_id": "paper-1"}], ["chunk-1"])

    results = index.search("agentic", top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "new agentic content"
