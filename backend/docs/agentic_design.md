# Agentic Design

This project is a semi-agentic RAG system with explicit state, graph routing, tool execution, recovery, verification, and stop reasons. It supports an optional LLM planner, while the default mode keeps a deterministic heuristic planner for repeatable local development.

## Agent State

`AgenticRAGState` stores:

- request and chat history
- intent and query plan
- local/web/evidence chunks
- context quality
- planner decision and research plan
- tool calls and tool results
- prompt, answer, citations, verification
- limits and stop reason
- trace events

## Planner Decision

`PlannerDecision` contains:

- `goal`
- `intent`
- `needs_fresh_context`
- `can_answer_from_local_context`
- `selected_tools`
- `steps`
- `stop_condition`
- `risk_notes`

The default planner chooses deterministic strategies:

- sufficient local context: no tool steps
- latest/fresh query: arXiv search, PDF download, PDF index, local retrieve
- insufficient local context: web search and snippet ingest

When `ENABLE_LLM_PLANNER=true`, `planner_node` asks the LLM for the same structured JSON decision using current state, context quality, limits, and tool descriptions. The planner output is parsed, capped by `AgentLimits`, and validated against `ToolRegistry.names()`. Invalid JSON or planner failures fall back to the deterministic planner and add `llm_planner_fallback:*` to `risk_notes`.

Trace events include `planner_source`:

- `heuristic`
- `llm`
- `heuristic_fallback`

## Tool Affordances

Each tool exposes:

- name
- description
- input schema
- when-to-use guidance
- failure modes

This gives the LLM planner explicit affordances instead of asking it to guess hidden APIs.

## Verification

Verification has three layers:

1. Citation grounding removes invalid citation IDs.
2. Uncited trailing claims are removed when the answer uses explicit citations.
3. Claim-level support checks run through an injectable `ClaimSupportJudge` that labels each cited claim as supported, contradicted, or insufficient.
4. `verify_answer` trace records `claim_citation_map`, so the run history can show which chunk IDs were used to support or reject each claim.

The default `HeuristicClaimSupportJudge` combines term coverage with a negation-conflict check, so simple contradictions like "X does Y" versus cited evidence saying "X does not do Y" are rejected. When `ENABLE_LLM_VERIFIER=true`, `AnswerVerifier.verify_async` uses `LLMClaimSupportJudge` to ask the configured LLM for a JSON label: supported, contradicted, or insufficient. Provider failures fall back to the heuristic judge. This is still not a substitute for a calibrated NLI benchmark, but the production provider hook is now present.

## Recovery

If verification requests `retrieve_more`, `plan_recovery` creates a validated recovery decision. The graph only executes recovery when web search is enabled and retrieval limits allow more tool calls.

Evidence-producing observations route back through `quality_gate` before `draft_answer`.

## Agentic Vs Fixed

Agentic:

- graph state and conditional routing
- tool registry and tool execution with limits/timeouts
- quality gate before answering
- recovery after verification
- stop reasons and trace

Still fixed or limited:

- heuristic planner remains the default unless `ENABLE_LLM_PLANNER=true`
- query planning is rule-based
- default verifier is deterministic claim support; LLM verifier is optional and still needs live calibration
