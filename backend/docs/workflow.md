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

Q&A flow:

`retrieve -> answer`
