from app.vectorstore.bm25 import BM25Scorer, tokenize


def test_tokenize_normalizes_terms_and_removes_common_stop_words() -> None:
    assert tokenize("The Agentic-RAG method retrieves evidence.") == [
        "agentic",
        "rag",
        "method",
        "retrieves",
        "evidence",
    ]


def test_bm25_scorer_ranks_exact_keyword_matches_first() -> None:
    scorer = BM25Scorer(
        [
            "planning agents retrieve paper evidence before answering",
            "vision transformer benchmark results",
            "agentic rag planning retrieval",
        ]
    )

    ranked_documents = scorer.rank("agentic rag planning", top_k=2)

    assert ranked_documents[0][0] == 2
    assert ranked_documents[0][1] == 1.0
    assert len(ranked_documents) == 2
