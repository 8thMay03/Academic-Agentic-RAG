# Agentic RAG Evaluation Report

## Run Metadata

- Dataset: `evals/datasets/agentic_rag_eval.jsonl`
- Cases: 110
- Modes: vector_only_rag, hybrid_rag, hybrid_rerank_rag, full_agentic_rag

## Baseline Summary

| mode | case_count | answer_point_recall | citation_precision | citation_recall | retrieval_recall | context_precision | abstention_accuracy | unsupported_claim_rate | claim_citation_coverage | verifier_availability_rate | avg_tool_calls | web_arxiv_usage_rate | p50_latency_ms | p95_latency_ms | avg_estimated_cost_usd | error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_agentic_rag | 110 | 0.982 | 0.991 | 0.991 | 0.991 |  | 0.991 |  |  |  |  |  | 620.000 | 780.000 | 0.000460 | 0 |
| hybrid_rag | 110 | 0.727 | 0.909 | 0.727 | 0.909 |  | 0.909 |  |  |  |  |  | 260.000 | 260.000 | 0.000201 | 0 |
| hybrid_rerank_rag | 110 | 0.864 | 0.864 | 0.909 | 0.909 |  | 0.909 |  |  |  |  |  | 430.000 | 430.000 | 0.000301 | 0 |
| vector_only_rag | 110 | 0.545 | 0.727 | 0.591 | 0.591 |  | 0.773 |  |  |  |  |  | 180.000 | 180.000 | 0.000115 | 0 |

## Agentic Delta

- answer point recall: 0.982 vs 0.864 (+0.118)
- citation precision: 0.991 vs 0.864 (+0.127)
- citation recall: 0.991 vs 0.909 (+0.082)
- retrieval recall: 0.991 vs 0.909 (+0.082)
- context precision: 0.000 vs 0.000 (+0.000)
- abstention accuracy: 0.991 vs 0.909 (+0.082)
- claim citation coverage: 0.000 vs 0.000 (+0.000)
- unsupported claim rate: 0.000 vs 0.000 (+0.000)
- average tool calls: 0.000 vs 0.000 (+0.000)
- p95 latency: 780.000 ms vs 430.000 ms (+350.000 ms)
- average estimated cost: $0.000460 vs $0.000301 (+0.000159)

## Error Analysis

- Failing/low-quality cases detected: 148

| Case | Mode | Issues |
| --- | --- | --- |
| `comparison_001` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_002` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_003` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_004` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_005` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_006` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_007` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_008` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_009` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
| `comparison_010` | `vector_only_rag` | missing expected answer points, missing expected citation, retrieval miss |
- Additional cases omitted: 138

## Interpretation Notes

- Treat fixture profiles as harness/instrumentation checks, not live model-quality claims.
- Use live profile results on an indexed paper corpus before making portfolio claims.
- Compare `full_agentic_rag` against `hybrid_rerank_rag` to isolate the value of planning, tool recovery, and verification.
- Interpret verifier-only metrics together with `verifier_availability_rate`; a zero availability means unsupported-claim and claim-citation metrics were not measured for that mode.
