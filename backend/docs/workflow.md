# Workflow

Agentic research flow:

`intent_router -> plan -> search -> select_papers -> download -> parse -> embed -> summarize -> compare -> report -> critic`

Intent routing:

- `summarize`: stops after grounded paper summaries and critic validation.
- `compare`: stops after summaries plus comparison and critic validation.
- `full_research`: runs through final report generation.
- `question_answering`: currently uses the research evidence path because `/research` does not carry chat-specific `paper_ids` or retrieval options.

Conditional loops:

- After `search`, the workflow searches with the next planned query when the current result set is below the planner's minimum paper target.
- After `summarize` and `compare`, the workflow decides whether later synthesis steps are required by the detected intent.
- After `critic`, the workflow can route back to `search`, `summarize`, `compare`, or `report` when the intent-required output is incomplete.
- Search and reflection loops are capped by iteration limits in `ResearchState`.

Resilience:

- PDF download, parsing, indexing, LLM summarization, and comparison errors are recorded in state instead of failing the whole research request.
- When LLM summarization or comparison is unavailable, the workflow falls back to extracted text, abstracts, and title-based comparison.

Chat Agentic RAG flow:

`query -> local_retrieve -> quality_gate -> web_search_when_needed -> collect_web_context -> answer_with_citations -> persist_chat_history`

Implementation:

- `AgenticChatWorkflow` owns the chat pipeline state, trace events, context quality gate, web fallback, prompt construction, citation grounding, and answer generation.
- `ChatService` is an adapter that preserves the existing `answer` and `stream_answer` API.
- Chat routes load chat history before the workflow and persist the final user/assistant exchange after the workflow returns.
- When local context is insufficient, web snippets are converted into temporary cited context with chunk ids such as `web:1`.

Current limitation:

- Web results are used as request-time context. Downloading/extracting/indexing new web documents into the long-term knowledge base is the next ingestion step, not part of this refactor.
