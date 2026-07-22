# Agentic RAG Evaluation

This folder contains the offline evaluation harness for comparing the project against simpler RAG baselines.

## Run

```bash
cd backend
python evals/run_eval.py --dataset evals/datasets/agentic_rag_eval.jsonl --mode all
```

The runner writes detailed JSON results to `evals/results/latest_results.json` and prints a compact table. Add `--report-output evals/results/<name>.md` to generate a markdown portfolio report with baseline deltas and error analysis.

For deterministic CI/portfolio smoke benchmarking without OpenAI, Chroma, Tavily, or arXiv:

```bash
cd backend
python evals/run_eval.py --profile offline_fixture --mode all --output evals/results/offline_fixture_results.json --report-output evals/results/offline_fixture_report.md
```

`offline_fixture` is a transparent harness benchmark. It validates dataset coverage, metrics, and expected baseline behavior, but it is not a live model/corpus performance claim.

For a local corpus benchmark that indexes fixture chunks into a temporary Chroma collection and runs the real retrieval/agent workflow without external APIs:

```bash
cd backend
python evals/run_eval.py --dataset tests/fixtures/eval_cases.jsonl --profile local_fixture --mode all --output evals/results/local_fixture_results.json --report-output evals/results/local_fixture_report.md
```

`local_fixture` is stronger than `offline_fixture` because it exercises Chroma indexing, vector search, keyword search, retrieval merging, quality gating, and the LangGraph workflow. It still uses deterministic fake embeddings/LLM responses, so use the `live` profile for final model-quality claims.

## Modes

- `vector_only_rag`: Chroma vector search, no BM25, no reranker, no agent.
- `hybrid_rag`: vector + keyword retrieval with reranking disabled, no agent.
- `hybrid_rerank_rag`: current local RAG stack with query rewrite, hybrid retrieval, and reranking, no agent recovery.
- `full_agentic_rag`: the LangGraph workflow from `app.agent.workflow.AgenticChatWorkflow`.

## Dataset Schema

Each JSONL row follows this shape:

```json
{
  "id": "q001",
  "question": "How does CRAG differ from standard RAG?",
  "language": "en",
  "paper_ids": ["paper-crag"],
  "answer_type": "comparison",
  "expected_answer_points": ["CRAG evaluates retrieved documents before generation"],
  "expected_citation_chunk_ids": ["paper-crag:p2:c1"],
  "is_answerable": true,
  "requires_fresh_context": false,
  "requires_multi_hop": true
}
```

The included dataset contains 110 generated cases. Regenerate it with:

```bash
cd backend
python evals/generate_dataset.py
```

Before using numbers as final portfolio evidence, run the `live` profile against an indexed PDF corpus and include the generated markdown report plus manual error analysis notes for representative failures.

Live profile preflight checks that `OPENAI_API_KEY` is configured and that `CHROMA_DIR` contains an indexed `chroma.sqlite3` corpus before it instantiates live services. If the dataset includes fresh-context cases for `full_agentic_rag`, configure `TAVILY_API_KEY` as well so web recovery can be measured instead of silently failing those cases. Use `--skip-live-preflight` only for intentional partial/debug runs.

```bash
cd backend
python evals/run_eval.py --profile live --mode all --output evals/results/live_results.json --report-output evals/results/live_report.md
```

The current live report artifact is `evals/results/live_report.md`; it covers 110 dataset cases across all four baseline modes with zero recorded case errors.

## Metrics

- Retrieval: recall, MRR, nDCG, context precision.
- Citation: precision, recall, invalid citation rate.
- Answer/grounding: expected point recall, abstention accuracy, unsupported claim rate, claim citation coverage, verifier availability rate.
- Agent behavior: average tool calls, tool success rate, web/arXiv usage rate, recovery usage rate.
- System: p50/p95 latency and token/cost fields when provider/tool usage is configured.

Token and cost metrics become meaningful when the runtime captures provider usage in the LLM/embedding services and when per-tool cost env vars are configured.
