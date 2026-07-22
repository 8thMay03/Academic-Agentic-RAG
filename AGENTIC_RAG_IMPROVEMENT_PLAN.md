# Agentic RAG Improvement Plan

## 1. Mục Tiêu

Mục tiêu của kế hoạch này là nâng project từ một hệ thống RAG có workflow agentic sang một Agentic RAG có thể bảo vệ được trong phỏng vấn Junior AI Engineer / LLM Engineer.

Ưu tiên cao nhất:

1. Chứng minh hệ thống agentic bằng planning, routing, tool selection và stopping condition rõ ràng.
2. Có evaluation methodology và baseline comparison thay vì chỉ có unit tests.
3. Tăng độ tin cậy của grounding, citation và semantic verification.
4. Làm rõ latency, token usage, cost, observability và security.
5. Chuẩn hóa tài liệu để nhà tuyển dụng hiểu được kiến trúc, trade-off và kết quả thực nghiệm.

## 2. Chẩn Đoán Hiện Trạng

### Điểm Mạnh Hiện Có

- Có LangGraph workflow trong `backend/app/agent/graph.py`.
- Có agent state typed trong `backend/app/agent/state.py`.
- Có tool registry và tool execution layer trong `backend/app/agent/tools/`.
- Có retrieval pipeline tương đối tốt: vector search, BM25, reranking, query rewrite, multi-query.
- Có citation grounding trong `backend/app/agent/citations.py`.
- Có tests tốt: backend hiện có hơn 230 tests pass sau các cải tiến.
- Có frontend hiển thị agent trace, giúp demo dễ hiểu.

### Điểm Yếu Chính

- `planner_node.py` đã có structured planner decision, tool metadata, validation và optional LLM planner với heuristic fallback.
- `retrieval_planning_node.py` đã lập retrieval strategy V1 gồm retrieval mode, per-query top_k, score threshold và max chunk budget.
- Planner/recovery policy đã có local low-threshold retry trước external search và đã có live benchmark để quan sát tác động trên corpus hiện tại.
- `AnswerVerifier` đã có claim judge interface, contradiction heuristic V1, optional LLM claim judge provider, claim-status trace và live benchmark artifact để theo dõi grounding.
- Đã có evaluation framework, 110-case dataset, offline fixture benchmark, local fixture benchmark có temp-indexed Chroma corpus và full live benchmark report.
- Đã đo latency/token/cost ở generation node, capture embedding usage cho local retrieval + ingest/index operations, đo latency cho `execute_tool`, thêm configurable external tool-provider cost attribution và tổng hợp vào agent run usage.
- BM25 đã chuyển sang persistent keyword index, không còn full scan Chroma collection.
- Security đã có SSRF/download limit/domain allowlist/prompt injection detection, optional API key auth, tenant enforcement, tenant-aware rate-limit headers, tenant-scoped local storage và upload size policy.
- Documentation đã được cập nhật, README/docs có benchmark offline, local fixture, live report và demo trace mẫu.

## 3. Thứ Tự Ưu Tiên

| Priority | Workstream | Lý do |
| --- | --- | --- |
| P0 | Evaluation + Baseline | Đây là bằng chứng quan trọng nhất cho portfolio LLM/RAG. |
| P0 | Real Planner + Tool Selection | Đây là phần quyết định project có agentic thật hay không. |
| P0 | Semantic Verification | Giảm hallucination và làm citation đáng tin hơn. |
| P1 | Observability + Cost | Nhà tuyển dụng sẽ hỏi latency/token/cost. |
| P1 | Re-grade Context + Retry Policy | Làm control flow chặt hơn và tránh recovery mù. |
| P1 | Security Hardening | Bắt buộc nếu có web/PDF download. |
| P2 | BM25 Scalability | Quan trọng khi scale corpus. |
| P2 | Chunking Improvements | Tăng chất lượng retrieval cho academic papers. |
| P2 | Documentation + Demo Report | Biến repo thành portfolio dễ review. |

## 3.1. Trạng Thái Triển Khai Hiện Tại

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Evaluation skeleton | Done | `backend/evals/`, `backend/evals/run_eval.py`, 110-case dataset trong `backend/evals/datasets/agentic_rag_eval.jsonl`; metrics gồm retrieval/citation/answer/agent/system và report markdown |
| Baseline modes | Done | `backend/evals/baselines.py` có vector/hybrid/hybrid_rerank/full_agentic modes, offline/local fixture benchmark và report delta |
| Structured planner | Done | `backend/app/agent/nodes/planner_node.py`, `PlannerDecision`, tool descriptions, optional `ENABLE_LLM_PLANNER`, heuristic fallback, validation tool registry và local-retry-before-web policy |
| Retrieval strategy | Done | `backend/app/agent/nodes/query_planning_node.py`, `QueryPlan.retrieval_mode`, `per_query_top_k`, `max_total_chunks` |
| Tool metadata | Done | `backend/app/agent/tools/base.py`, `ToolRegistry.descriptions()` và metadata trong từng tool |
| Re-grade context sau tool | Done | `backend/app/agent/graph.py`, `backend/app/agent/nodes/quality_gate_node.py` |
| Stop reason | Done | `StopReason`, `ChatWorkflowResult.stop_reason`, streaming `done.stop_reason` |
| Semantic verification V1 | Done | `backend/app/agent/evaluators/answer_verifier.py`, injectable `ClaimSupportJudge`, optional `LLMClaimSupportJudge`, contradiction detection, claim-status trace và `claim_citation_map` |
| Grounding/citation trace | Done | `backend/app/agent/nodes/verify_answer_node.py`, `backend/app/models/chat.py`, `frontend/src/utils/format.js`, `frontend/src/components/ClaimCitationMap.jsx`; trace giải thích claim nào được support bởi chunk nào và frontend hiển thị claim-level citation map |
| Observability/cost V1 | Done | `backend/app/services/llm_service.py`, `backend/app/services/embedding_service.py`, `backend/app/agent/nodes/generate_answer_node.py`, `backend/app/agent/nodes/tool_executor_node.py`, `AgentRunRecord.usage`, configurable `*_COST_USD` env vars |
| PDF security | Done | `backend/app/services/pdf_service.py`, upload limit/header checks trong `backend/app/api/routes/papers.py`, optional `PDF_DOWNLOAD_ALLOWED_DOMAINS` |
| Prompt injection defense | Done | `backend/app/agent/security.py`, `backend/app/agent/nodes/draft_answer_node.py`, `backend/app/agent/prompts/answer_prompt.py`, `backend/tests/agent/test_security.py`, `backend/tests/agent/test_answer_prompt.py`; suspicious context được flag trong metadata/citation/trace và prompt coi context là untrusted data |
| BM25 scalability | Done | `backend/app/vectorstore/keyword_index.py`, `backend/app/vectorstore/chroma.py` |
| Section-aware chunking | Done | `backend/app/parser/chunker.py` |
| Frontend trace polish | Done | `frontend/src/components/AgentActivity.jsx`, `frontend/src/components/ClaimCitationMap.jsx`, `frontend/src/utils/format.js`, route-level code splitting trong `frontend/src/App.jsx` |
| Integration fixture tests | Done | `backend/tests/integration/test_agentic_rag_e2e.py` có 10 case fixture covering grounded answer, unanswerable/no-web, follow-up history, retrieval strategies, web recovery, step limit, fabricated citation rejection và streaming events |
| Full benchmark report | Done | Offline fixture, local fixture và full 110-case live benchmark có JSON + markdown reports trong `backend/evals/results/`; `backend/evals/run_eval.py` có live preflight cho API key + Chroma corpus |
| API auth/rate limit/upload policy | Done | Optional API key auth, tenant header enforcement (`REQUIRE_TENANT_ID`, `X-Tenant-ID`), tenant-scoped local chat/run storage, tenant-aware in-memory rate limit, rate-limit headers và upload limit đã có; production OAuth/Redis/DB vẫn là deployment hardening ngoài scope local repo |
| Deployment hardening | Done | Docker chạy non-root, không bind-mount source code ở root compose, có healthcheck, runtime data volume, `.dockerignore` và `.gitignore` secret/runtime hygiene, `*_FILE` secret loading cho Docker/Kubernetes secrets, tenant-scoped local storage và deployment docs; managed DB/Redis/secret manager/OTel collector là hạ tầng production ngoài repo |
| Observability/request ID | Done | `backend/app/middleware/request_id.py`, `backend/app/observability/tracing.py`, `backend/app/config/logging.py`, `backend/app/api/routes/chat.py`; có `X-Request-ID`, W3C `traceparent`, `X-Trace-ID`, `X-Span-ID`, request-scoped stream events, JSON request logs, structured stream error và optional FastAPI OpenTelemetry instrumentation |

## 4. Workstream 1: Evaluation Và Baseline

### Mục Tiêu

Tạo một evaluation framework có thể so sánh các cấu hình RAG khác nhau và chứng minh Agentic RAG cải thiện ở đâu.

### Files Nên Tạo

- `backend/evals/README.md`
- `backend/evals/datasets/agentic_rag_eval.jsonl`
- `backend/evals/run_eval.py`
- `backend/evals/baselines.py`
- `backend/evals/metrics.py`
- `backend/evals/results/README.md`

### Baseline Cần Có

1. `vector_only_rag`
   - Chroma similarity search.
   - Không BM25.
   - Không rerank.
   - Không query rewrite.
   - Không web/arXiv.

2. `hybrid_rag`
   - Vector + BM25.
   - Không rerank.
   - Không agent workflow.

3. `hybrid_rerank_rag`
   - Vector + BM25 + reranker.
   - Không quality gate.
   - Không web/arXiv/recovery.

4. `full_agentic_rag`
   - Dùng graph hiện tại hoặc graph sau khi cải tiến.

### Dataset Format

Mỗi dòng JSONL:

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

### Bộ Câu Hỏi Tối Thiểu

- 20 factual lookup.
- 20 comparison.
- 20 multi-hop / multi-aspect.
- 15 follow-up questions dùng chat history.
- 15 unanswerable questions.
- 10 latest/current questions cần web/arXiv.
- 10 adversarial questions có prompt injection hoặc citation trap.

Tổng tối thiểu: 110 câu.

### Metrics

Retrieval:

- Recall@k.
- MRR.
- nDCG@k.
- Context precision.
- Query rewrite drift rate.

Answer:

- Answer correctness.
- Faithfulness.
- Answer relevance.
- Unsupported claim rate.
- Abstention accuracy cho câu không answerable.

Citation:

- Citation precision.
- Citation recall.
- Invalid citation rate.
- Claim citation coverage.

Agent:

- Tool success rate.
- Average tool calls per query.
- Recovery success rate.
- Stop reason distribution.
- Web/arXiv usage rate.

System:

- p50/p95 latency.
- Input/output tokens.
- Estimated cost per query.

### Acceptance Criteria

- Có thể chạy:

```bash
cd backend
python evals/run_eval.py --dataset evals/datasets/agentic_rag_eval.jsonl --mode all
```

- Output có bảng so sánh ít nhất 4 baseline.
- README có summary kiểu:

```text
full_agentic_rag improves answer faithfulness by X% over hybrid_rerank_rag
but costs Y% more tokens and Z seconds more p95 latency.
```

## 5. Workstream 2: Real Planner Và Tool Selection

### Mục Tiêu

Thay planner rule-based bằng planner có structured output, có khả năng chọn tool dựa trên state, evidence, history và failure reason.

### Files Cần Sửa

- `backend/app/agent/nodes/planner_node.py`
- `backend/app/agent/nodes/recovery_planner_node.py`
- `backend/app/agent/models.py`
- `backend/app/agent/tools/registry.py`
- `backend/tests/agent/test_planner_node.py`
- `backend/tests/agent/test_recovery_planner_node.py`

### Model Cần Thêm

Thêm structured models:

```python
@dataclass(frozen=True)
class PlannerDecision:
    goal: str
    intent: str
    needs_fresh_context: bool
    can_answer_from_local_context: bool
    selected_tools: list[str]
    steps: list[ResearchPlanStep]
    stop_condition: str
    risk_notes: list[str]
```

Nếu dùng Pydantic thì tốt hơn vì parse/validate JSON từ LLM dễ hơn.

### Planner Prompt Cần Có

Planner phải thấy:

- User question.
- Intent.
- Query plan.
- Context quality result.
- Available tools từ `ToolRegistry.names()`.
- Tool descriptions.
- Previous tool results.
- Current limits.
- Stop conditions.

Planner phải trả JSON only.

### Tool Description

Mỗi tool nên expose:

- `name`
- `description`
- `input_schema`
- `when_to_use`
- `failure_modes`

Ví dụ:

```python
class AgentTool(Protocol):
    name: str
    description: str
    input_schema: dict
    async def run(self, input: dict) -> ToolResult: ...
```

### Guardrails

Planner output phải được validate:

- Tool name phải tồn tại trong registry.
- Không vượt quá `AgentLimits`.
- Không được gọi `pdf_download` nếu chưa có PDF URL từ previous artifact.
- Không được gọi `pdf_index` nếu chưa có downloaded PDF.
- Nếu không có evidence và web disabled thì stop với answer_unknown.

### Acceptance Criteria

- Planner có thể chọn khác nhau cho:
  - Local context đủ.
  - Local context thiếu nhưng web enabled.
  - Latest query cần arXiv.
  - Web disabled.
  - Prior web search failed.
- Tests chứng minh planner không chỉ sinh fixed recipe.
- Trace ghi rõ `planner_decision`, `selected_tools`, `stop_condition`.

## 6. Workstream 3: Retrieval Planning Thật

### Mục Tiêu

Biến `retrieval_planning_node.py` từ trace-only thành node quyết định retrieval strategy.

### Files Cần Sửa

- `backend/app/agent/nodes/query_planning_node.py`
- `backend/app/agent/nodes/local_retrieve_node.py`
- `backend/app/agent/models.py`
- `backend/tests/agent/test_query_planning_node.py`

### Retrieval Strategy Cần Có

Thêm model:

```python
@dataclass(frozen=True)
class RetrievalStrategy:
    queries: list[str]
    top_k: int
    score_threshold: float | None
    search_mode: Literal["vector", "keyword", "hybrid"]
    rerank: bool
    reason: str
```

### Logic Gợi Ý

- Simple lookup: 1-2 queries, top_k thấp.
- Comparison: query riêng từng entity + query comparison.
- Paper review: query theo method/contribution/result/limitation.
- Follow-up: standalone rewrite bằng chat history.
- Low recall retry: lower threshold hoặc tăng top_k.

### Acceptance Criteria

- Trace có `retrieval_strategy`.
- Local retrieve nhận strategy thay vì chỉ question/top_k mặc định.
- Eval đo được strategy nào giúp tăng recall.

## 7. Workstream 4: Re-grade Context Sau Recovery

### Vấn Đề Hiện Tại

Sau khi recovery hoặc tool execution lấy thêm evidence, graph có thể quay lại `quality_gate` để đánh giá lại context trước khi trả lời.

### Files Cần Sửa

- `backend/app/agent/graph.py`
- `backend/app/agent/nodes/quality_gate_node.py`
- `backend/tests/agent/test_graph_routing.py`
- `backend/tests/agent/test_agentic_rag_graph.py`

### Thay Đổi Đề Xuất

Sau `observe`, nếu tool tạo thêm chunks thì route về `quality_gate` thay vì trực tiếp `draft_answer`.

Flow mới:

```text
execute_tool -> observe -> quality_gate -> plan | draft_answer
```

Cần tránh lặp bằng:

- `max_retrieval_rounds`.
- `max_agent_steps`.
- `quality_gate_attempts`.
- `same_evidence_no_improvement_count`.

### Acceptance Criteria

- Nếu web search trả chunks nhưng score/coverage thấp, system tiếp tục plan hoặc answer unknown.
- Nếu local retrieve sau PDF index đủ tốt, system draft answer.
- Test chứng minh không loop vô hạn.

## 8. Workstream 5: Semantic Verification

### Mục Tiêu

Nâng `AnswerVerifier` từ citation hygiene thành claim-level verifier.

### Files Cần Sửa

- `backend/app/agent/evaluators/answer_verifier.py`
- `backend/app/agent/nodes/verify_answer_node.py`
- `backend/app/agent/models.py`
- `backend/tests/agent/test_answer_verifier.py`

### Model Cần Thêm

```python
@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    status: Literal["supported", "contradicted", "insufficient"]
    supporting_chunk_ids: list[str]
    reason: str
```

```python
@dataclass(frozen=True)
class SemanticVerificationResult:
    passed: bool
    claims: list[ClaimVerification]
    revised_answer: str
    suggested_action: Literal["finalize", "retrieve_more", "revise_answer", "answer_unknown"]
```

### Verification Flow

1. Split answer into claims.
2. Map each claim to cited chunks.
3. Ask verifier LLM:
   - Does cited evidence support the claim?
   - Is there contradiction?
   - Is evidence insufficient?
4. Remove or revise unsupported claims.
5. If too many unsupported claims, trigger retrieve_more or answer_unknown.

### Prompt Requirements

- Verifier chỉ dùng cited chunks.
- Trả JSON.
- Không thêm kiến thức ngoài.
- Phân biệt:
  - citation exists
  - citation supports claim

### Acceptance Criteria

- Fake citation đúng format nhưng context không support phải bị reject.
- Câu có 1 supported claim và 1 unsupported claim phải được revise.
- Unanswerable question không được ép answer.

## 9. Workstream 6: Grounding Và Citation

### Mục Tiêu

Đảm bảo mỗi factual claim có citation đúng và citations hiển thị match với claim.

### Files Cần Sửa

- `backend/app/agent/citations.py`
- `backend/app/agent/prompts/answer_prompt.py`
- `backend/tests/agent/test_citation_grounder.py`

### Cải Tiến

- Thêm claim-to-citation mapping trong trace.
- Không tự động gắn citation đầu tiên nếu answer không reference citation nào, vì có thể che lỗi grounding.
- Bắt LLM trả answer với chunk_id citation nội bộ, sau đó renderer map sang số citation.
- Thêm validation:
  - Citation ID tồn tại.
  - Citation ID được dùng trong answer.
  - Citation text có overlap/semantic support với claim.

### Acceptance Criteria

- Invalid citation bị loại.
- Missing citation không được auto-fix một cách mù.
- Trace giải thích citation nào support claim nào.

## 10. Workstream 7: Latency, Token Usage Và Cost

### Mục Tiêu

Mỗi request cần biết chạy bao lâu, tốn bao nhiêu tokens, tốn bao nhiêu tiền, bottleneck ở đâu.

### Files Cần Sửa/Tạo

- `backend/app/utils/timer.py`
- `backend/app/services/llm_service.py`
- `backend/app/services/embedding_service.py`
- `backend/app/agent/models.py`
- `backend/app/agent/nodes/*.py`
- `backend/app/storage/agent_run_store.py`
- `backend/tests/services/test_llm_service.py`

### Trace Fields Cần Thêm

```json
{
  "stage": "generate_answer",
  "latency_ms": 1830,
  "model": "gpt-4.1-mini",
  "input_tokens": 4200,
  "output_tokens": 380,
  "estimated_cost_usd": 0.0021
}
```

### Cost Registry

Tạo config:

```python
MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0
    },
    "text-embedding-3-small": {
        "input_per_1m": 0.0
    }
}
```

Giá nên để config/env để dễ cập nhật.

### Acceptance Criteria

- Agent run record có total latency/tokens/cost.
- Trace `execute_tool` có `latency_ms` trên success, failure, retry-limit và error paths.
- Eval report có p50/p95 latency và average cost/query.
- LLMService tests kiểm tra token usage được capture nếu provider trả usage.

## 11. Workstream 8: Observability

### Mục Tiêu

Biến trace UI thành observability có ích cho debugging và production.

### Files Cần Sửa/Tạo

- `backend/app/config/logging.py`
- `backend/app/middleware/request_id.py`
- `backend/app/agent/models.py`
- `backend/app/storage/agent_run_store.py`

### Cần Có

- Request ID.
- Chat ID.
- Run ID.
- Node start/end.
- Tool start/end.
- Error taxonomy:
  - `retrieval_empty`
  - `web_search_skipped`
  - `tool_timeout`
  - `invalid_planner_output`
  - `verification_failed`
  - `pdf_download_failed`

### Optional

- OpenTelemetry spans.
- LangSmith tracing nếu muốn demo nhanh.
- JSON logs.

### Acceptance Criteria

- Một request có thể trace từ API tới graph node tới tool call.
- Error trong streaming endpoint không chỉ là `str(exc)` mà có structured error.
- Agent run detail đủ để debug một câu trả lời sai.

### Trạng Thái Triển Khai

- `backend/app/observability/tracing.py` parse/generate W3C `traceparent`, exposes request-scoped `trace_id`/`span_id`, and configures optional FastAPI OpenTelemetry spans when `OTEL_ENABLED=true`.
- `backend/app/middleware/request_id.py` returns `traceparent`, `X-Trace-ID`, and `X-Span-ID` headers and logs trace identifiers with request completion.
- `backend/app/api/routes/chat.py` includes `request_id`, `trace_id`, and `span_id` in streaming events.
- `backend/tests/observability/test_tracing.py`, `backend/tests/config/test_logging.py`, and `backend/tests/api/test_chat.py` cover trace context parsing, JSON log fields, response headers, and structured streaming errors.

## 12. Workstream 9: BM25 Và Retrieval Scalability

### Vấn Đề Hiện Tại

`ChromaVectorStore.keyword_search()` đang gọi `collection.get()` toàn bộ collection rồi tạo `BM25Scorer` mỗi query. Cách này ổn cho demo nhỏ nhưng không scale.

### Files Cần Sửa/Tạo

- `backend/app/vectorstore/bm25.py`
- `backend/app/vectorstore/keyword_index.py`
- `backend/app/vectorstore/indexing.py`
- `backend/app/services/retriever_service.py`
- `backend/tests/vectorstore/test_keyword_index.py`

### Option Khuyến Nghị

Short-term:

- Lưu persistent BM25 index trong file JSON/pickle theo corpus.
- Rebuild index khi index PDF.

Mid-term:

- SQLite FTS5 cho keyword search.
- Hoặc Tantivy/OpenSearch nếu muốn scale hơn.

### Acceptance Criteria

- Keyword search không gọi full `collection.get()` mỗi query.
- Index update khi thêm chunks.
- Eval có latency before/after.

## 13. Workstream 10: Chunking Cho Academic Papers

### Mục Tiêu

Tăng retrieval quality bằng chunking hiểu cấu trúc paper.

### Files Cần Sửa

- `backend/app/parser/pdf_parser.py`
- `backend/app/parser/cleaner.py`
- `backend/app/parser/chunker.py`
- `backend/app/services/pdf_index_service.py`
- `backend/tests/parser/test_chunker.py`

### Cải Tiến

- Detect section headings:
  - Abstract
  - Introduction
  - Method
  - Experiments
  - Results
  - Limitations
  - Conclusion
- Metadata thêm:
  - `section_title`
  - `section_type`
  - `page_number`
  - `source_path`
- Chunk table/caption riêng nếu detect được.
- Preserve formula blocks tốt hơn.

### Acceptance Criteria

- Chunk metadata có section.
- Retrieval có thể boost section theo query type.
- Paper review questions retrieve đủ method/contribution/experiments/limitations.

## 14. Workstream 11: Security Hardening

### Rủi Ro Hiện Tại

- PDF download từ URL có nguy cơ SSRF.
- Upload PDF cần size limit/header validation.
- Web/PDF content có thể chứa prompt injection.
- API cần coarse tenant isolation ngoài optional API key auth.
- Local JSON storage cần tenant namespace rõ ràng để tránh trộn dữ liệu trong demo/local deployment.

### Files Cần Sửa

- `backend/app/services/pdf_service.py`
- `backend/app/services/web_search_service.py`
- `backend/app/api/routes/papers.py`
- `backend/app/api/routes/chat.py`
- `backend/app/config/settings.py`
- `backend/app/agent/prompts/answer_prompt.py`

### Cải Tiến

PDF/Web:

- Block private IP ranges.
- Allow only `http`/`https`.
- Limit download size.
- Limit upload size.
- Validate content-type và PDF header.
- Add domain allowlist optional cho trusted PDF sources.

Prompt injection:

- Add context boundary warning trong prompt.
- Detect suspicious instructions trong chunks:
  - "ignore previous"
  - "system prompt"
  - "developer message"
  - "reveal secret"
- Mark suspicious chunks in metadata.
- Verifier không được dùng instruction-like content làm instruction.

API:

- Optional API key auth.
- Rate limit với `Retry-After` và `X-RateLimit-*` headers.
- CORS configurable by env.

### Acceptance Criteria

- Test block URL tới localhost/private IP.
- Test oversized PDF bị reject.
- Test malicious context không override prompt.

## 15. Workstream 12: Retry, Fallback Và Stopping Condition

### Mục Tiêu

Biến retry/fallback thành policy rõ ràng, có stop reason trong trace.

### Files Cần Sửa

- `backend/app/agent/models.py`
- `backend/app/agent/graph.py`
- `backend/app/agent/tools/execution.py`
- `backend/tests/agent/test_graph_routing.py`

### Stop Reasons

Thêm enum:

```python
StopReason = Literal[
  "answered_with_sufficient_context",
  "answered_after_recovery",
  "no_context_available",
  "web_search_disabled",
  "step_limit_reached",
  "tool_limit_reached",
  "verification_failed_answer_unknown",
  "planner_no_valid_steps"
]
```

### Retry Policy

- Retrieval empty:
  - retry with lower threshold.
  - retry with expanded query.
  - then web/arXiv if allowed.
- Tool timeout:
  - do not retry same tool more than N.
- Verification failed:
  - retrieve more once.
  - then revise/answer unknown.

### Acceptance Criteria

- Every final answer has stop reason.
- Tests cover all major stop reasons.
- No route can loop without consuming a limit counter.

### Trạng Thái Triển Khai

- `backend/app/agent/graph.py` infer `verification_failed_answer_unknown` khi verifier rút answer về `I don't know` sau fabricated/unsupported citation và không còn recovery hợp lệ.
- `backend/app/agent/nodes/planner_node.py` thêm retry `local_retrieve` với `top_k` tăng và `score_threshold` giảm trước `web_search`/`web_snippet_ingest`.
- `backend/tests/agent/test_agentic_rag_graph.py` kiểm tra local retry lấy được evidence thì không gọi web fallback.
- `backend/tests/integration/test_agentic_rag_e2e.py` kiểm tra fabricated citation không gọi web khi disabled, trả `I don't know`, citation rỗng và stop reason rõ ràng.
- `frontend/src/utils/format.js` hiển thị label cho stop reason `verification_failed_answer_unknown`.

## 16. Workstream 13: Documentation Và Portfolio Report

### Files Cần Sửa/Tạo

- `README.md`
- `backend/docs/architecture.md`
- `backend/docs/workflow.md`
- `backend/docs/evaluation.md`
- `backend/docs/security.md`
- `backend/docs/agentic_design.md`

### Nội Dung Cần Có

`agentic_design.md`:

- Agent state.
- Decision points.
- Tools.
- Planner.
- Recovery policy.
- Stopping condition.
- What is agentic vs fixed.

`evaluation.md`:

- Dataset construction.
- Baselines.
- Metrics.
- Results.
- Error analysis.

`architecture.md`:

- Mermaid diagram đúng với code mới.
- Request lifecycle.
- Data lifecycle.

`security.md`:

- Web/PDF ingestion risks.
- Mitigations.
- Known limitations.

### Acceptance Criteria

- Docs không lệch với code.
- README có bảng benchmark.
- README có demo trace mẫu.
- README nói thẳng limitations thay vì marketing quá mức.

## 17. Workstream 14: Integration Tests Với Dữ Liệu Thật

### Mục Tiêu

Bổ sung tests ngoài unit/mocked tests để chứng minh end-to-end behavior.

### Files Cần Tạo

- `backend/tests/integration/test_agentic_rag_e2e.py`
- `backend/tests/fixtures/pdfs/`
- `backend/tests/fixtures/eval_cases.jsonl`

### Test Cases

1. Upload/index PDF nhỏ rồi hỏi factual question.
2. Ask unanswerable question, expect `I don't know`.
3. Comparison across two PDFs.
4. Follow-up question with chat history.
5. Citation ID must belong to retrieved chunks.
6. Web disabled with no local context.
7. Planner selected tool must match scenario.
8. Verification removes unsupported claim.
9. Tool timeout returns structured failure.
10. Stop reason exists for every final answer.

### Acceptance Criteria

- Integration tests chạy được không cần external API bằng fake services.
- Có optional marker cho tests dùng real OpenAI/Tavily/arXiv.

## 18. Workstream 15: Frontend Demo Polish

### Mục Tiêu

Giúp reviewer nhìn thấy agent đang làm gì, tại sao làm, và khi nào dừng.

### Files Cần Sửa

- `frontend/src/components/AgentRunPanel.jsx`
- `frontend/src/components/AgentActivity.jsx`
- `frontend/src/components/CitationList.jsx`
- `frontend/src/pages/ChatPage.jsx`

### Cải Tiến

- Hiển thị planner decision.
- Hiển thị stop reason.
- Hiển thị latency/token/cost summary.
- Hiển thị claim verification result.
- Highlight citations theo claim.
- Gắn badge:
  - local evidence
  - web evidence
  - arXiv evidence
  - low confidence
  - unsupported removed

### Acceptance Criteria

- Một demo query cho thấy full trace từ planning tới verification.
- UI không chỉ hiển thị activity text mà có decision rationale.

## 19. Roadmap 7 Ngày

### Ngày 1: Evaluation Skeleton

- Tạo `backend/evals/`.
- Tạo dataset format JSONL.
- Implement baseline runner đơn giản.
- Tạo 20 eval samples đầu tiên.
- Output CSV/JSON result.

Deliverable:

- `python evals/run_eval.py --mode vector_only_rag` chạy được.

### Ngày 2: Baseline Comparison

- Implement 4 modes:
  - vector only
  - hybrid
  - hybrid rerank
  - full agentic
- Thêm metrics retrieval/citation/latency cơ bản.
- Viết `backend/docs/evaluation.md` bản đầu.

Deliverable:

- Có bảng so sánh baseline đầu tiên.

### Ngày 3: Planner Structured Output

- Thêm planner schema.
- Thêm tool descriptions.
- Planner validate tool names và limits.
- Tests cho 4 scenario chính.

Deliverable:

- `planner_node.py` không còn chỉ hard-code `_fresh_research_steps` và `_web_fallback_steps`.

### Ngày 4: Re-grade Context Và Stop Reasons

- Sửa graph route sau `observe`.
- Thêm stop reason.
- Thêm tests chống loop.
- Trace final state có stop reason.

Deliverable:

- Every answer has stop reason.

### Ngày 5: Semantic Verification V1

- Claim splitter.
- Claim-to-citation verifier bằng LLM/fake verifier.
- Tests cho unsupported claim.
- Trace verification details.

Deliverable:

- Fake supported citation không qua được nếu text không support claim.

### Ngày 6: Observability Và Cost V1

- Add latency cho generation node và `execute_tool`.
- Capture usage từ OpenAI response.
- Estimate cost bằng config.
- Persist metrics vào agent run.

Deliverable:

- Agent run có total latency/tokens/cost.

### Ngày 7: Docs Và Demo

- Update README.
- Update architecture/workflow docs.
- Add evaluation report.
- Add demo trace screenshot hoặc sample JSON.

Deliverable:

- Repo đọc như một portfolio hoàn chỉnh, có limitations rõ.

## 20. Roadmap 30 Ngày

### Tuần 1: Agentic Proof

- Evaluation + baseline.
- Structured planner.
- Tool selection.
- Stop reasons.
- Semantic verifier V1.

### Tuần 2: Retrieval Quality

- Retrieval strategy model.
- Section-aware chunking.
- Better query decomposition.
- Persistent keyword index.
- Retrieval eval/report.

### Tuần 3: Reliability And Security

- SSRF protection.
- Upload/download limits.
- Prompt injection detection.
- Better retry policy.
- Integration tests with fixture PDFs.

### Tuần 4: Portfolio Polish

- Observability dashboard/trace UI.
- Cost and latency report.
- Documentation cleanup.
- Demo scenarios.
- Error analysis write-up.

## 21. Definition Of Done

Project được coi là cải thiện đạt yêu cầu khi có đủ các điều kiện:

- Có ít nhất 100 eval cases.
- Có ít nhất 4 baseline modes.
- README có bảng kết quả benchmark.
- Planner chọn tool dựa trên state và tool descriptions.
- Recovery không chỉ hard-code một web search.
- Verification kiểm tra semantic support theo claim.
- Mỗi final answer có stop reason.
- Mỗi run có latency/tokens/cost.
- Security có SSRF/file-size/prompt-injection mitigation.
- Docs mô tả đúng code.
- Tests vẫn pass và có integration tests chính.

## 22. Ba Cải Tiến ROI Cao Nhất

1. Evaluation + baseline report.
   - Đây là bằng chứng mạnh nhất với nhà tuyển dụng.
   - Làm rõ Agentic RAG tốt hơn RAG thường ở đâu.

2. Structured planner + real tool selection.
   - Đây là điểm quyết định dự án có thật sự agentic hay không.
   - Giúp bạn trả lời câu hỏi phỏng vấn về planning/routing.

3. Claim-level semantic verification.
   - Tăng trust.
   - Giảm hallucination.
   - Làm citation có ý nghĩa hơn citation formatting.

## 23. Câu Hỏi Phỏng Vấn Cần Chuẩn Bị

1. Vì sao project này cần Agentic RAG thay vì RAG thường?
2. Agent state gồm những gì?
3. Decision point nào dùng rule, decision point nào dùng LLM?
4. Planner chọn tool như thế nào?
5. Khi local retrieval thiếu context, hệ thống làm gì?
6. Khi web/arXiv không trả kết quả, hệ thống dừng thế nào?
7. Làm sao biết citation thật sự support claim?
8. Baseline nào được dùng để so sánh?
9. Metrics nào chứng minh improvement?
10. Latency và cost tăng bao nhiêu khi bật agentic workflow?
11. BM25 hiện có scale không?
12. Prompt injection từ PDF/web được xử lý ra sao?
13. Vì sao dùng ChromaDB?
14. Vì sao dùng cross-encoder reranker?
15. Nếu làm lại trong production, bạn thay đổi gì?
