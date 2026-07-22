# Agentic RAG Evaluation Report

## Run Metadata

- Dataset: `tests/fixtures/eval_cases.jsonl`
- Cases: 2
- Modes: vector_only_rag, hybrid_rag, hybrid_rerank_rag, full_agentic_rag

## Baseline Summary

| mode | case_count | answer_point_recall | citation_precision | citation_recall | retrieval_recall | context_precision | abstention_accuracy | unsupported_claim_rate | claim_citation_coverage | verifier_availability_rate | avg_tool_calls | web_arxiv_usage_rate | p50_latency_ms | p95_latency_ms | avg_estimated_cost_usd | error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_agentic_rag | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 574.824 | 984.995 | 0.000000 | 0 |
| hybrid_rag | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.235 | 3.989 | 0.000000 | 0 |
| hybrid_rerank_rag | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.707 | 2.838 | 0.000000 | 0 |
| vector_only_rag | 2 | 1.000 | 0.250 | 1.000 | 1.000 | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.063 | 5.475 | 0.000000 | 0 |

## Agentic Delta

- answer point recall: 1.000 vs 1.000 (+0.000)
- citation precision: 1.000 vs 0.250 (+0.750)
- citation recall: 1.000 vs 1.000 (+0.000)
- retrieval recall: 1.000 vs 1.000 (+0.000)
- context precision: 0.250 vs 0.250 (+0.000)
- abstention accuracy: 1.000 vs 1.000 (+0.000)
- claim citation coverage: 1.000 vs 0.000 (+1.000)
- unsupported claim rate: 0.500 vs 0.000 (+0.500)
- average tool calls: 1.000 vs 0.000 (+1.000)
- p95 latency: 984.995 ms vs 2.838 ms (+982.157 ms)
- average estimated cost: $0.000000 vs $0.000000 (+0.000000)

## Error Analysis

- Failing/low-quality cases detected: 6

| Case | Mode | Issues |
| --- | --- | --- |
| `fixture_factual_001` | `vector_only_rag` | invalid citation |
| `fixture_unanswerable_001` | `vector_only_rag` | invalid citation |
| `fixture_factual_001` | `hybrid_rag` | invalid citation |
| `fixture_unanswerable_001` | `hybrid_rag` | invalid citation |
| `fixture_factual_001` | `hybrid_rerank_rag` | invalid citation |
| `fixture_unanswerable_001` | `hybrid_rerank_rag` | invalid citation |

## Interpretation Notes

- Treat fixture profiles as harness/instrumentation checks, not live model-quality claims.
- Use live profile results on an indexed paper corpus before making portfolio claims.
- Compare `full_agentic_rag` against `hybrid_rerank_rag` to isolate the value of planning, tool recovery, and verification.
- Interpret verifier-only metrics together with `verifier_availability_rate`; a zero availability means unsupported-claim and claim-citation metrics were not measured for that mode.
