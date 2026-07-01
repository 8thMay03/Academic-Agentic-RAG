# Workflow

Agentic RAG chat flow:

`query -> local_retrieve -> quality_gate -> web_search_when_needed -> answer_with_citations -> persist_chat_history`

LangGraph implementation:

- `backend/app/agent/graph.py` owns the Agentic RAG topology and conditional routing.
- `backend/app/agent/state.py` defines the graph state.
- `backend/app/agent/nodes/` contains one node module per workflow step.
- `AgenticChatWorkflow` is the runtime adapter used by `ChatService`.
- Chat routes load recent chat history before the graph runs and persist the final exchange after the answer is returned.

Graph nodes:

- `local_retrieve`: retrieves from the local Chroma knowledge base. When `paper_ids` is omitted, retrieval searches across all indexed local papers.
- `quality_gate`: decides whether local context is sufficient using chunk count, context length, retrieval scores, source count, query-term coverage, freshness wording, and an optional LLM self-check for borderline context.
- `web_search`: runs only when the quality gate rejects local context. Web snippets become temporary cited context with chunk ids such as `web:1`.
- `answer`: builds the grounded answer prompt and citation list. If no local or web context is available, the answer is `I don't know`.

Current limitation:

- Web results are still request-time context. Downloading, extracting, chunking, embedding, and persisting new web documents into the long-term knowledge base is the next ingestion step.
