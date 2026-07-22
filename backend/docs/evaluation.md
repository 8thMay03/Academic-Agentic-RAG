# Evaluation Methodology

This project now includes an offline evaluation harness in `backend/evals/` for comparing the Agentic RAG workflow against simpler RAG baselines.

## Goals

The evaluation should answer four questions:

1. Does the agentic workflow improve answer quality over simpler RAG?
2. Does it improve citation grounding and retrieval recall?
3. How often does it correctly abstain when the corpus has no answer?
4. What latency and cost trade-off does the agentic workflow introduce?

## Baselines

Run all baselines with:

```bash
cd backend
python evals/run_eval.py --dataset evals/datasets/agentic_rag_eval.jsonl --mode all --report-output evals/results/latest_report.md
```

For a deterministic benchmark that does not require external services:

```bash
cd backend
python evals/run_eval.py --profile offline_fixture --mode all --output evals/results/offline_fixture_results.json --report-output evals/results/offline_fixture_report.md
```

`offline_fixture` is intentionally synthetic. It is useful for CI, portfolio demos, and proving the evaluation methodology, but the final project report should also include `live` results on an indexed PDF corpus.

For a stronger local benchmark that builds a temporary Chroma corpus from fixture chunks and runs the real retrieval/agent code without external APIs:

```bash
cd backend
python evals/run_eval.py --dataset tests/fixtures/eval_cases.jsonl --profile local_fixture --mode all --output evals/results/local_fixture_results.json --report-output evals/results/local_fixture_report.md
```

Latest `local_fixture` summary:

| Mode | Cases | Answer recall | Citation precision | Citation recall | Retrieval recall | Abstention acc. | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vector_only_rag` | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 4.062 ms | 5.893 ms |
| `hybrid_rag` | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 3.064 ms | 3.739 ms |
| `hybrid_rerank_rag` | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | 2.494 ms | 2.612 ms |
| `full_agentic_rag` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 357.817 ms | 667.555 ms |

Interpret this as a pipeline/instrumentation check, not a broad quality claim: the corpus has only two fixture cases and uses deterministic fake embeddings/LLM outputs.

Latest full 110-case live benchmark summary (`backend/evals/results/live_report.md`):

| Mode | Cases | Answer recall | Citation precision | Retrieval recall | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vector_only_rag` | 110 | 0.545 | 0.727 | 0.591 | 0 |
| `hybrid_rag` | 110 | 0.727 | 0.909 | 0.909 | 0 |
| `hybrid_rerank_rag` | 110 | 0.864 | 0.864 | 0.909 | 0 |
| `full_agentic_rag` | 110 | 0.982 | 0.991 | 0.991 | 0 |

Generated markdown reports:

- `backend/evals/results/offline_fixture_report.md`
- `backend/evals/results/local_fixture_report.md`
- `backend/evals/results/live_report.md`

Each report includes the baseline summary, `full_agentic_rag` delta against `hybrid_rerank_rag`, latency/cost trade-off, and top failing cases.

Available modes:

| Mode | Description |
| --- | --- |
| `vector_only_rag` | Chroma vector search only, no keyword retrieval, reranker, web, arXiv, or agent recovery. |
| `hybrid_rag` | Vector + BM25-style keyword retrieval with reranking disabled, no agent workflow. |
| `hybrid_rerank_rag` | Current local RAG stack with query rewrite, hybrid retrieval, and reranking, but no agent recovery. |
| `full_agentic_rag` | Full LangGraph workflow through `AgenticChatWorkflow`. |

For smoke tests, limit the dataset:

```bash
cd backend
python evals/run_eval.py --mode all --limit 3 --case-timeout-seconds 10
```

## Dataset

The dataset lives at `backend/evals/datasets/agentic_rag_eval.jsonl` and currently contains 110 cases.

Each case uses JSONL:

```json
{
  "id": "q001",
  "question": "How does CRAG differ from standard RAG?",
  "language": "en",
  "paper_ids": ["paper-crag"],
  "answer_type": "comparison",
  "expected_answer_points": [
    "CRAG evaluates retrieved documents before generation",
    "CRAG can trigger corrective retrieval"
  ],
  "expected_citation_chunk_ids": ["paper-crag:p2:c1", "paper-crag:p3:c0"],
  "is_answerable": true,
  "requires_fresh_context": false,
  "requires_multi_hop": true
}
```

The distribution is:

- 20 factual lookup questions.
- 20 comparison questions.
- 20 multi-hop or multi-aspect questions.
- 15 follow-up questions with chat history.
- 15 unanswerable questions.
- 10 latest/current questions that require web or arXiv.
- 10 adversarial questions with prompt injection or citation traps.

Regenerate the dataset with:

```bash
cd backend
python evals/generate_dataset.py
```

## Metrics

Retrieval:

- `retrieval_recall`
- `mrr`
- `ndcg`
- `context_precision`

Citation:

- `citation_precision`
- `citation_recall`
- `invalid_citation_rate`

Answer:

- `answer_point_recall`
- `abstention_accuracy`
- `unsupported_claim_rate`
- `claim_citation_coverage`
- `verifier_availability_rate`

Agent:

- `avg_tool_calls`
- `tool_success_rate`
- `web_arxiv_usage_rate`
- `recovery_usage_rate`

System:

- `p50_latency_ms`
- `p95_latency_ms`
- `avg_input_tokens`
- `avg_output_tokens`
- `avg_estimated_cost_usd`

Token and cost fields are populated when provider usage is captured by the LLM/embedding services or when per-tool cost estimates are configured with the `*_COST_USD` environment variables.

Verifier-only metrics such as `unsupported_claim_rate` and `claim_citation_coverage` should be interpreted together with `verifier_availability_rate`. If availability is `0.0`, the mode did not emit verifier trace events, so grounding metrics are not directly measured for that mode.

## Interpreting Results

The important comparison is not whether `full_agentic_rag` wins every row. The useful evidence is:

- It should improve fresh/current and low-local-context questions.
- It should improve abstention on unanswerable cases.
- It should reduce invalid citations or unsupported answers after semantic verification is added.
- It will likely cost more latency and tokens than non-agent baselines.

If `full_agentic_rag` does not beat `hybrid_rerank_rag` on the cases where agent behavior is supposed to matter, the workflow is probably agentic in structure but not yet useful in behavior.

## Current Limitations

- `offline_fixture` scores are deterministic harness scores, not live LLM quality scores.
- `local_fixture` scores exercise local indexing/retrieval/agent flow, but the fixture corpus is too small for final portfolio claims.
- The generated dataset should be supplemented with hand-authored paper-specific cases before final portfolio use.
- Answer scoring uses expected point lexical coverage, not a semantic judge.
- Faithfulness and unsupported claim metrics are still approximated outside the runtime claim verifier.
- Cost metrics are estimates, not billing records. Keep model and tool pricing env vars current before using `avg_estimated_cost_usd` in portfolio claims.
