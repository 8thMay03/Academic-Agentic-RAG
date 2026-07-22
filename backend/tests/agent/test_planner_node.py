import pytest

from app.config import settings as settings_module
from app.agent.models import ChatWorkflowRequest, ContextQuality
from app.agent.nodes.planner_node import planner_node
from app.agent.tools.registry import ToolRegistry
from app.agent.tools.web_search_tool import WebSearchTool
from tests.services.test_chat_service import FakeWebSearchService


class NamesOnlyRegistry:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return self._names


class FakePlannerLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.mark.asyncio
async def test_planner_node_records_structured_decision_for_web_fallback() -> None:
    state = await planner_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG work?", top_k=3),
            "intent": "research_qa",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="no_local_context",
            ),
            "tool_registry": NamesOnlyRegistry(["web_search", "web_snippet_ingest"]),
            "trace": [],
        }
    )

    decision = state["planner_decision"]
    assert decision.goal == "How does Agentic RAG work?"
    assert decision.intent == "research_qa"
    assert decision.can_answer_from_local_context is False
    assert decision.selected_tools == ["web_search", "web_snippet_ingest"]
    assert decision.stop_condition == "external_context_retrieved_or_no_valid_tools"
    assert state["trace"][0]["selected_tools"] == ["web_search", "web_snippet_ingest"]
    assert state["trace"][0]["risk_notes"] == [
        "local_context_insufficient:no_local_context",
        "planner_removed_unregistered_tools:local_retrieve",
    ]


@pytest.mark.asyncio
async def test_planner_node_retries_local_retrieval_before_web_fallback() -> None:
    state = await planner_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG work?", top_k=3, score_threshold=0.65),
            "intent": "research_qa",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="low_recall",
            ),
            "tool_registry": NamesOnlyRegistry(["local_retrieve", "web_search", "web_snippet_ingest"]),
            "trace": [],
        }
    )

    assert state["planner_decision"].selected_tools == ["local_retrieve", "web_search", "web_snippet_ingest"]
    local_retry = state["plan"].steps[0]
    assert local_retry.tool_name == "local_retrieve"
    assert local_retry.input["top_k"] == 6
    assert local_retry.input["score_threshold"] == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_planner_node_removes_unregistered_tools_from_decision() -> None:
    state = await planner_node(
        {
            "request": ChatWorkflowRequest("What is latest Agentic RAG?"),
            "intent": "fresh_research",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="latest_query_requires_web",
            ),
            "tool_registry": NamesOnlyRegistry(["arxiv_search"]),
            "trace": [],
        }
    )

    assert state["planner_decision"].selected_tools == ["arxiv_search"]
    assert state["plan"].steps[0].tool_name == "arxiv_search"
    assert "planner_removed_unregistered_tools:pdf_download,pdf_index,local_retrieve" in state["trace"][0]["risk_notes"]


def test_tool_registry_exposes_tool_descriptions() -> None:
    registry = ToolRegistry([WebSearchTool(FakeWebSearchService())])

    descriptions = registry.descriptions()

    assert descriptions[0]["name"] == "web_search"
    assert "Search the web" in descriptions[0]["description"]
    assert descriptions[0]["input_schema"] == {"query": "string", "max_results": "integer"}
    assert "missing_tavily_api_key" in descriptions[0]["failure_modes"]


@pytest.mark.asyncio
async def test_planner_node_can_use_llm_planner_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "ENABLE_LLM_PLANNER", True)
    llm = FakePlannerLLM(
        """
        ```json
        {
          "goal": "Find external evidence",
          "intent": "research_qa",
          "needs_fresh_context": false,
          "can_answer_from_local_context": false,
          "steps": [
            {
              "tool_name": "web_search",
              "reason": "Local context is empty.",
              "input": {"query": "Agentic RAG", "max_results": 3}
            }
          ],
          "stop_condition": "web_result_found_or_limit",
          "risk_notes": ["llm_selected_minimal_plan"]
        }
        ```
        """
    )

    state = await planner_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG work?", top_k=3),
            "intent": "research_qa",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="no_local_context",
            ),
            "llm_service": llm,
            "tool_registry": NamesOnlyRegistry(["web_search", "web_snippet_ingest"]),
            "trace": [],
        }
    )

    assert state["planner_decision"].goal == "Find external evidence"
    assert state["planner_decision"].selected_tools == ["web_search"]
    assert state["plan"].steps[0].input == {"query": "Agentic RAG", "max_results": 3}
    assert state["trace"][0]["planner_source"] == "llm"
    assert "Available tools" in llm.prompts[0]


@pytest.mark.asyncio
async def test_planner_node_falls_back_when_llm_planner_returns_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "ENABLE_LLM_PLANNER", True)
    llm = FakePlannerLLM("not json")

    state = await planner_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG work?", top_k=3),
            "intent": "research_qa",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="no_local_context",
            ),
            "llm_service": llm,
            "tool_registry": NamesOnlyRegistry(["web_search", "web_snippet_ingest"]),
            "trace": [],
        }
    )

    assert state["planner_decision"].selected_tools == ["web_search", "web_snippet_ingest"]
    assert state["trace"][0]["planner_source"] == "heuristic_fallback"
    assert "llm_planner_fallback:JSONDecodeError" in state["trace"][0]["risk_notes"]


@pytest.mark.asyncio
async def test_planner_node_removes_unregistered_llm_selected_tools(monkeypatch) -> None:
    monkeypatch.setattr(settings_module.settings, "ENABLE_LLM_PLANNER", True)
    llm = FakePlannerLLM(
        """
        {
          "goal": "Use a tool",
          "intent": "research_qa",
          "needs_fresh_context": false,
          "can_answer_from_local_context": false,
          "steps": [
            {"tool_name": "unknown_tool", "reason": "Bad tool", "input": {}},
            {"tool_name": "web_search", "reason": "Good tool", "input": {"query": "Agentic RAG"}}
          ],
          "stop_condition": "done",
          "risk_notes": []
        }
        """
    )

    state = await planner_node(
        {
            "request": ChatWorkflowRequest("How does Agentic RAG work?"),
            "intent": "research_qa",
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=0,
                context_chars=0,
                reason="no_local_context",
            ),
            "llm_service": llm,
            "tool_registry": NamesOnlyRegistry(["web_search"]),
            "trace": [],
        }
    )

    assert state["planner_decision"].selected_tools == ["web_search"]
    assert [step.tool_name for step in state["plan"].steps] == ["web_search"]
    assert "planner_removed_unregistered_tools:unknown_tool" in state["trace"][0]["risk_notes"]
