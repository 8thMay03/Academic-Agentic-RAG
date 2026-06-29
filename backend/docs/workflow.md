# Workflow

Agentic research flow:

`plan -> search -> select_papers -> download -> parse -> embed -> summarize -> compare -> report -> critic`

Conditional loops:

- After `search`, the workflow searches with the next planned query when the current result set is below the planner's minimum paper target.
- After `critic`, the workflow can route back to `search`, `summarize`, `compare`, or `report` when the output is incomplete.
- Search and reflection loops are capped by iteration limits in `ResearchState`.

Resilience:

- PDF download, parsing, indexing, LLM summarization, and comparison errors are recorded in state instead of failing the whole research request.
- When LLM summarization or comparison is unavailable, the workflow falls back to extracted text, abstracts, and title-based comparison.

Q&A flow:

`retrieve -> answer`
