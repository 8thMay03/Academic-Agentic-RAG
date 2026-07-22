# Evaluation Results

`run_eval.py` writes detailed results here. Keep small summary reports in git, but avoid committing large raw traces or private datasets.

Current committed result artifacts:

- `offline_fixture_results.json`: deterministic 110-case harness benchmark generated with `python evals/run_eval.py --profile offline_fixture --mode all --output evals/results/offline_fixture_results.json --report-output evals/results/offline_fixture_report.md`.
- `offline_fixture_report.md`: markdown summary with baseline deltas and top failing cases for the deterministic harness benchmark.
- `local_fixture_results.json`: temp-indexed local corpus benchmark generated with `python evals/run_eval.py --dataset tests/fixtures/eval_cases.jsonl --profile local_fixture --mode all --output evals/results/local_fixture_results.json --report-output evals/results/local_fixture_report.md`.
- `local_fixture_report.md`: markdown summary with baseline deltas and top failing cases for the temp-indexed local corpus benchmark.
- `latest_results.json`: copy of the most recent benchmark artifact for quick inspection.

The offline fixture benchmark is intentionally synthetic. The local fixture benchmark exercises more of the retrieval stack, but still uses deterministic fake embeddings and fake LLM responses. Use live runs on indexed papers for final portfolio claims.
