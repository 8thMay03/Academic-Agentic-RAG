# AI Research Assistant Refactor Plan

## 1. Goal

This project should not be rewritten from scratch. The better path is an incremental refactor that keeps the parts that already have value:

- FastAPI API layer.
- LangGraph-based agentic RAG graph.
- RAG service with query rewriting and multi-query retrieval.
- Hybrid retrieval: vector search + BM25 + reranking.
- PDF parsing, chunking, and indexing pipeline.
- Chat history and chat sessions.
- Existing backend test suite.

The refactor goals are:

- Make the codebase easier to read, test, and extend.
- Reduce loosely typed `dict` payloads between layers.
- Separate agent orchestration, tool execution, retrieval, prompting, citation handling, and verification.
- Prepare the codebase to evolve from "RAG with web fallback" into a real research agent with `plan -> act -> observe -> verify`.
- Keep the project runnable and the test suite green after every phase.

## 2. Current-State Assessment

### 2.1. Strengths To Keep

- The backend already has a reasonable structure: `api`, `services`, `agent`, `vectorstore`, `parser`, and `storage`.
- LangGraph is already isolated in `backend/app/agent/graph.py`.
- The graph already has separate nodes for:
  - `local_retrieve`
  - `quality_gate`
  - `web_search`
  - `knowledge_ingest`
  - `answer`
- The quality gate already considers:
  - chunk count
  - context length
  - top score
  - average score
  - source count
  - query coverage
  - latest/current query terms
  - LLM self-check for borderline context
- Non-streaming answers already have citation grounding.
- arXiv search, PDF download, and PDF indexing services already exist.
- The test suite already covers many important behaviors.

### 2.2. Main Problems

1. `AgenticChatWorkflow` has too many responsibilities:
   - local retrieval
   - web search
   - web snippet ingestion
   - context quality evaluation
   - prompt building
   - citation creation
   - citation grounding
   - streaming and non-streaming orchestration

2. The agent graph is still mostly linear:

   ```text
   local_retrieve -> quality_gate -> answer
                            -> web_search -> knowledge_ingest -> answer
   ```

   This is a conditional RAG pipeline, not yet a true agent loop.

3. State and data contracts are not clean enough:
   - many values are passed as `list[dict]`
   - chunk, web result, and citation metadata do not have clear schemas
   - trace events are free-form dictionaries

4. Trace is generated but mostly wasted:
   - the workflow creates `trace`
   - `ChatService` does not return trace through the API
   - streaming drops `_trace`
   - the frontend does not show agent activity

5. Streaming and non-streaming behavior are inconsistent:
   - non-streaming answers are citation-grounded
   - streaming sends raw LLM tokens directly and does not sanitize the final answer in the same way

6. Web ingestion only stores snippets:
   - it does not crawl or download full documents
   - it does not index full PDFs or web pages
   - it does not retrieve again after ingestion

7. There is no answer verifier:
   - no claim extraction
   - no check that each factual claim has evidence
   - no retry path when the answer is weak or unsupported

8. Memory is only chat history:
   - no research memory
   - no saved findings
   - no agent run/step history
   - no long-running research task state

## 3. Refactor Principles

- Do not rewrite from scratch.
- Do not mix large refactors and large features in the same step.
- Every phase should have tests before and after.
- Prefer typed contracts over raw `dict`.
- Keep APIs backward-compatible where practical.
- If a response schema changes, update API tests and frontend usage in the same phase.
- Each phase should have a clear rollback point.
- Do not heavily refactor the frontend before the backend contract is stable.

## 4. Target Architecture

Target architecture:

```text
API layer
  -> Chat application service
    -> Agent workflow
      -> LangGraph
        -> Planner
        -> Tool executor
        -> Observer
        -> Context evaluator
        -> Answer drafter
        -> Verifier
      -> Agent tools
        -> Local retrieve
        -> Web search
        -> arXiv search
        -> PDF download
        -> PDF index
        -> Evidence extraction
      -> Core services
        -> RAG service
        -> Retriever service
        -> Vector store
        -> LLM service
        -> Storage services
```

Target agent flow:

```text
classify_intent
  -> plan
  -> execute_tool
  -> observe
  -> should_continue
      -> execute_tool
      -> observe
      -> should_continue
  -> draft_answer
  -> verify_answer
      -> revise_answer or retrieve_more
  -> final_answer
```

## 5. Phase 0: Freeze Current Behavior

### Goal

Create a safety net before changing structure.

### Tasks

1. Run the full backend test suite:

   ```powershell
   cd backend
   pytest
   ```

2. Document the main behaviors:
   - chat with sufficient local PDF context
   - fallback to web when local context is insufficient
   - force web for latest/current questions
   - streaming chat
   - citation filtering
   - PDF upload/indexing
   - chat session/history

3. Add missing tests for:
   - streaming should not persist fake citations
   - web search without `TAVILY_API_KEY`
   - internal trace should contain expected stages and metadata
   - fallback when LLM self-check fails
   - local + web context merge behavior

### Related Files

- `backend/tests/services/test_chat_service.py`
- `backend/tests/agent/test_agentic_rag_graph.py`
- `backend/tests/api/test_chat.py`
- `backend/app/services/agentic_chat_workflow.py`
- `backend/app/api/routes/chat.py`

### Done When

- The full test suite passes.
- Existing agentic behaviors are protected by tests.
- The baseline behavior is clear before refactoring.

## 6. Phase 1: Standardize Agent Contracts And Models

### Goal

Reduce free-form `dict` usage and make agent state explicit.

### New File

```text
backend/app/agent/models.py
```

### Suggested Models

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    paper_id: str
    title: str = ""
    chunk_id: str = ""
    page_number: int | None = None
    url: str | None = None
    score: float | None = None
    rerank_score: float | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    retrieval_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTraceEvent:
    stage: str
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    observations: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextQuality:
    sufficient: bool
    reason: str
    chunk_count: int
    context_chars: int
    top_score: float | None = None
    average_score: float | None = None
    source_count: int = 0
    query_coverage: float = 0.0
    self_check_used: bool = False
    self_check_passed: bool | None = None
```

### Tasks

1. Create the new model file.
2. Add adapters that convert retriever dictionaries into `RetrievedChunk`.
3. Update `AgenticRAGState` to use typed models.
4. Update graph nodes to create `AgentTraceEvent`.
5. Keep compatibility with the existing `Citation` model.

### Related Files

- `backend/app/agent/state.py`
- `backend/app/agent/nodes/*.py`
- `backend/app/services/agentic_chat_workflow.py`
- `backend/app/models/citation.py`

### Done When

- Agent state is easy to understand.
- Existing tests pass.
- Code no longer has to guess keys like `metadata.paper_id`, `citation.chunk_id`, and `retrieval_sources`.

## 7. Phase 2: Split `AgenticChatWorkflow`

### Goal

Turn `AgenticChatWorkflow` into a thin orchestration layer instead of a large mixed-responsibility class.

### Suggested Structure

```text
backend/app/agent/
  workflow.py
  state.py
  models.py
  citations.py
  evaluators/
    context_quality.py
  prompts/
    answer_prompt.py
    self_check_prompt.py
```

### Logic Mapping

- `_evaluate_context`, `_self_check_context`, `_self_check_prompt`, `_should_self_check`
  -> `ContextQualityEvaluator`

- `_build_prompt`, `_conversation_context`
  -> `AnswerPromptBuilder`

- `_citations`, `_ground_answer_citations`, `_citations_referenced_by_answer`, `_referenced_chunk_ids`
  -> `CitationGrounder`

- `_retrieve_local`, `_search_web`, `_ingest_web_snippets`
  -> temporarily keep in workflow, then move to the tool layer in Phase 4

### Suggested Interfaces

```python
class ContextQualityEvaluator:
    async def evaluate(self, request: ChatWorkflowRequest, chunks: list[RetrievedChunk]) -> ContextQuality:
        ...


class AnswerPromptBuilder:
    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        chat_history: list[ChatHistoryMessage] | None,
    ) -> str:
        ...


class CitationGrounder:
    def build_citations(self, chunks: list[RetrievedChunk], question: str) -> list[Citation]:
        ...

    def ground_answer(self, answer: str, citations: list[Citation]) -> str:
        ...
```

### Done When

- `AgenticChatWorkflow` mainly:
  - prepares the answer
  - calls the graph
  - calls the LLM
  - returns the result
- Large static helper methods have been moved to dedicated modules.
- Existing tests still pass.

## 8. Phase 3: Expose Trace Through API And Frontend

### Goal

Make agent behavior visible to users and easier to debug.

### Backend Changes

1. Add response models:

```python
class AgentTraceEventResponse(BaseModel):
    stage: str
    message: str | None = None
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[AgentTraceEventResponse] = Field(default_factory=list)
```

2. Change `ChatService.answer`.

Current behavior:

```python
return result.answer, result.citations
```

Recommended behavior:

```python
return result
```

or return a dedicated application-level result object.

3. Update `/chat` to return `trace`.

4. Update `/chat/stream` to emit agent step events:

```json
{"type": "agent_step", "stage": "local_retrieve", "chunk_count": 5}
{"type": "agent_step", "stage": "quality_gate", "sufficient": false, "reason": "low_query_coverage"}
{"type": "agent_step", "stage": "web_search", "chunk_count": 3}
```

### Frontend Changes

1. Update `frontend/src/api.js` to parse `agent_step`.
2. Update `ChatPage.jsx`:
   - store `trace` on assistant messages
   - show a small agent activity panel under each assistant message

### Done When

- The UI shows what the agent did.
- API tests assert `trace`.
- Streaming tests assert `agent_step` events.

## 9. Phase 4: Add An Agent Tool Layer

### Goal

The agent should call schema-based tools instead of graph nodes calling private workflow methods.

### Suggested Structure

```text
backend/app/agent/tools/
  __init__.py
  base.py
  registry.py
  local_retrieve_tool.py
  web_search_tool.py
  web_snippet_ingest_tool.py
  arxiv_search_tool.py
  pdf_download_tool.py
  pdf_index_tool.py
```

### Base Interface

```python
from typing import Protocol


class AgentTool(Protocol):
    name: str

    async def run(self, input: dict) -> ToolResult:
        ...
```

### Initial Tools

#### `local_retrieve`

Input:

```json
{
  "question": "...",
  "paper_ids": ["..."],
  "top_k": 5,
  "score_threshold": 0.65,
  "chat_history": []
}
```

Output:

- retrieved chunks
- retrieval queries
- score summary

#### `web_search`

Input:

```json
{
  "query": "...",
  "max_results": 5
}
```

Output:

- web sources
- snippet chunks
- skipped reason if the API key is missing

#### `web_snippet_ingest`

Input:

```json
{
  "chunks": []
}
```

Output:

- snippets ingested
- errors

#### `arxiv_search`

Input:

```json
{
  "query": "...",
  "max_results": 5,
  "sort_by": "submittedDate"
}
```

Output:

- paper metadata
- PDF URLs

#### `pdf_download`

Input:

```json
{
  "pdf_url": "...",
  "destination_dir": "data/pdfs"
}
```

Output:

- downloaded path
- cached flag

#### `pdf_index`

Input:

```json
{
  "filename": "...",
  "force": false
}
```

Output:

- paper id
- chunks indexed
- cached flag

### Done When

- Nodes no longer need to call methods like `workflow._search_web()` directly.
- Each tool has dedicated tests.
- Tool failure returns structured failure data instead of crashing the whole workflow when fallback is possible.

## 10. Phase 5: Convert LangGraph Into A Planner/Executor Loop

### Goal

Move from a conditional RAG pipeline to a bounded agent loop.

### Target Graph

```text
classify_intent
  -> plan
  -> execute_tool
  -> observe
  -> route_after_observation
      -> execute_tool
      -> draft_answer
      -> ask_clarification
  -> verify_answer
  -> final_answer
```

### Target Agent State

```python
class ResearchAgentState(TypedDict, total=False):
    request: ChatWorkflowRequest
    intent: str
    plan: ResearchPlan
    current_step_index: int
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    evidence: list[RetrievedChunk]
    quality: ContextQuality
    draft_answer: str | None
    final_answer: str | None
    citations: list[Citation]
    trace: list[AgentTraceEvent]
    limits: AgentLimits
```

### Required Limits

Add guardrails so the agent cannot run forever:

- `max_steps=6`
- `max_web_searches=2`
- `max_arxiv_searches=2`
- `max_pdf_downloads=3`
- `max_retrieval_rounds=3`
- per-tool timeout
- stop when context is sufficient and the answer is verified

### Suggested Planner Output

```python
@dataclass(frozen=True)
class ResearchPlan:
    goal: str
    steps: list[ResearchPlanStep]


@dataclass(frozen=True)
class ResearchPlanStep:
    tool_name: str
    reason: str
    input: dict
```

### Example Plan

Question:

```text
Compare Agentic RAG and CRAG based on the latest papers.
```

Plan:

```json
[
  {
    "tool_name": "local_retrieve",
    "reason": "Check the local knowledge base first",
    "input": {"question": "Agentic RAG CRAG comparison latest"}
  },
  {
    "tool_name": "arxiv_search",
    "reason": "The question asks for latest work, so fresh sources are required",
    "input": {"query": "Agentic RAG CRAG corrective RAG latest survey", "max_results": 5}
  },
  {
    "tool_name": "pdf_download",
    "reason": "Download an available PDF for full-text indexing",
    "input": {"pdf_url": "..."}
  },
  {
    "tool_name": "pdf_index",
    "reason": "Index the newly downloaded paper into the vector store",
    "input": {"filename": "..."}
  },
  {
    "tool_name": "local_retrieve",
    "reason": "Retrieve again after ingesting new sources",
    "input": {"question": "Agentic RAG CRAG comparison"}
  }
]
```

### Done When

- The graph has at least one execute/observe loop.
- The agent can call local retrieval, then web/arXiv, then retrieve again.
- Tests cover loop limits.
- Tests cover routing when context is already sufficient.

## 11. Phase 6: Upgrade Web/arXiv Ingestion Into Research Ingestion

### Goal

The agent should not only use request-time snippets. It should enrich the knowledge base when useful.

### Suggested Flow

```text
search_web/arxiv
  -> select_sources
  -> download_pdf_or_fetch_page
  -> parse
  -> chunk
  -> index
  -> retrieve_again
```

### Source Priority

1. Already indexed local PDF.
2. arXiv PDF or full paper.
3. Publisher/open-access PDF.
4. Web page with raw content.
5. Snippet fallback.

### Metadata To Store

- `source_type`: local_pdf, arxiv, web_pdf, web_page, snippet
- `source_url`
- `fetched_at`
- `published_at`
- `discovered_by_query`
- `trust_level`
- `ingestion_status`

### Done When

- If a search result has a PDF URL, the agent can download and index it.
- After indexing, the agent retrieves again from the new document.
- Citations from newly indexed sources can open the original URL or PDF.

## 12. Phase 7: Add An Answer Verifier

### Goal

Reduce hallucinations, fake citations, and weak factual claims.

### Verifier Checks

1. Does the answer include citations?
2. Do cited IDs exist in the evidence set?
3. Does each factual claim have nearby evidence?
4. For latest/current questions, are sources recent enough?
5. Are there conflicts between sources?
6. If confidence is low, does the answer clearly state limitations?

### Verifier Output

```python
from typing import Literal
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    suggested_action: Literal[
        "finalize",
        "retrieve_more",
        "revise_answer",
        "answer_unknown",
    ] = "finalize"
```

### Routing After Verification

```text
passed -> final_answer
unsupported_claims + steps_remaining -> retrieve_more
unsupported_claims + no_steps_remaining -> revise_answer
no_evidence -> answer_unknown
```

### Done When

- Streaming and non-streaming both use the same final citation/verification policy.
- Invalid citations are removed.
- Unsupported claims are removed or trigger more retrieval.

## 13. Phase 8: Clean Up The Frontend Around The New Backend Contract

### Goal

The UI should reflect actual agent behavior, not just show chat bubbles.

### Tasks

1. Add an agent activity timeline:
   - local search
   - quality gate
   - web search
   - arXiv search
   - PDF download/index
   - verification

2. Add source badges:
   - Local
   - Web
   - arXiv
   - Indexed
   - Verified
   - Low confidence

3. Add controls:
   - enable/disable web search
   - enable/disable arXiv search
   - allow/disallow automatic PDF downloads
   - configure max sources/top_k

4. Improve assistant message states:
   - planning
   - retrieving
   - searching web
   - indexing
   - drafting
   - verifying
   - done

### Done When

- Users can see what the agent is doing.
- Tool errors are understandable instead of being generic failures.
- Citations and sources have clear quality/source labels.

## 14. Suggested PR Sequence

### PR 1: `refactor/chat-result-trace`

Scope:

- Return trace from `ChatService`.
- Add trace to `ChatResponse`.
- Stream `agent_step` events.
- Update API and service tests.

Risk: low.

### PR 2: `refactor/agent-models`

Scope:

- Create `backend/app/agent/models.py`.
- Add typed trace events.
- Add retrieved chunk adapters.
- Update state and nodes.

Risk: medium.

### PR 3: `refactor/extract-agent-components`

Scope:

- Extract context evaluator.
- Extract prompt builder.
- Extract citation grounder.
- Reduce the size of `AgenticChatWorkflow`.

Risk: medium.

### PR 4: `refactor/agent-tools`

Scope:

- Create tool base and registry.
- Wrap local retrieval, web search, and web ingestion as tools.
- Make nodes call tools instead of private workflow methods.

Risk: medium.

### PR 5: `feature/planner-executor-graph`

Scope:

- Add planner node.
- Add executor node.
- Add observer/router.
- Add loop limits.

Risk: high.

### PR 6: `feature/arxiv-pdf-ingest-agent`

Scope:

- Add arXiv search tool.
- Add PDF download tool.
- Add PDF index tool.
- Retrieve again after ingestion.

Risk: high.

### PR 7: `feature/answer-verifier`

Scope:

- Draft answer.
- Verify citations and claims.
- Revise or finalize.

Risk: medium to high.

### PR 8: `ui/agent-activity-timeline`

Scope:

- Parse frontend agent events.
- Show activity timeline.
- Show source quality badges.

Risk: medium.

## 15. Chat And Streaming Test Checklist

- Sufficient local context -> do not call web.
- Insufficient local context -> call web.
- Latest/current query -> call web/arXiv even if local context exists.
- Missing Tavily key -> trace includes skipped reason and workflow does not crash.
- Web snippets -> citations include URLs.
- No context -> return `I don't know`.
- Fake LLM citation -> remove or replace through grounding.
- Streaming final persisted answer should not contain fake citations.
- Chat history is used to rewrite follow-up queries.
- Selected `paper_ids` are still applied correctly.

## 16. Code Cleanliness Checklist

- Do not add more large private helper methods to `AgenticChatWorkflow`.
- Do not pass new raw `dict` objects when a typed model would fit.
- Tool failure should return `ToolResult(success=False, error=...)`.
- Graph nodes should orchestrate; they should not contain long domain logic.
- Prompts should live under `agent/prompts`.
- Citation logic should live in `agent/citations.py`.
- Quality evaluation should live under `agent/evaluators`.
- Add tests for adapters that convert legacy dictionaries into typed models.

## 17. Suggested Folder Structure After Refactor

```text
backend/app/
  agent/
    graph.py
    state.py
    models.py
    workflow.py
    citations.py
    nodes/
      classify_intent_node.py
      planner_node.py
      tool_executor_node.py
      observer_node.py
      answer_node.py
      verifier_node.py
    tools/
      base.py
      registry.py
      local_retrieve_tool.py
      web_search_tool.py
      web_snippet_ingest_tool.py
      arxiv_search_tool.py
      pdf_download_tool.py
      pdf_index_tool.py
    evaluators/
      context_quality.py
      answer_verifier.py
    prompts/
      answer_prompt.py
      planner_prompt.py
      self_check_prompt.py
      verifier_prompt.py
  services/
    chat_service.py
    rag_service.py
    retriever_service.py
    llm_service.py
    web_search_service.py
    search_arxiv_service.py
    pdf_service.py
    pdf_index_service.py
```

## 18. Execution Recommendation

Start with PR 1, PR 2, and PR 3. These three PRs will immediately make the codebase cleaner without significantly changing agent behavior.

After those are stable:

1. Merge the tool layer.
2. Add the planner loop.
3. Add arXiv/PDF ingestion.
4. Add the verifier.
5. Upgrade the UI last.

Do not build the planner loop while the data contract is still mostly raw dictionaries. That would make the new graph harder to maintain than the current one.

## 19. Definition Of Success

The refactor is successful when:

- A new developer can read the agent graph and understand the flow within 10 minutes.
- Adding a new tool does not require editing many unrelated files.
- Every answer includes trace data explaining what the agent did.
- Streaming and non-streaming use the same citation and verification policy.
- The agent can perform multi-step research while staying within safe limits.
- The test suite passes after every PR.
- The codebase is ready to become a real research agent.
