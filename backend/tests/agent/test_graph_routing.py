from app.agent.graph import route_after_observation
from app.agent.models import ContextQuality, ResearchPlan, ResearchPlanStep, ToolResult


def test_route_after_observation_drafts_when_local_retrieve_finds_evidence() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(
                tool_name="local_retrieve",
                success=True,
                chunks=[{"id": "paper-1:p1:c0", "text": "Evidence."}],
            ),
            "current_step_index": 1,
            "plan": ResearchPlan(
                goal="answer",
                steps=[
                    ResearchPlanStep(tool_name="local_retrieve", reason="retrieve", input={}),
                    ResearchPlanStep(tool_name="web_search", reason="fallback", input={}),
                ],
            ),
        }
    )

    assert route == "draft_answer"


def test_route_after_observation_continues_when_current_quality_is_insufficient() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(
                tool_name="local_retrieve",
                success=True,
                chunks=[{"id": "paper-1:p1:c0", "text": "Sparse evidence."}],
            ),
            "quality": ContextQuality(
                sufficient=False,
                chunk_count=1,
                context_chars=16,
                reason="insufficient_local_context",
            ),
            "current_step_index": 1,
            "plan": ResearchPlan(
                goal="answer",
                steps=[
                    ResearchPlanStep(tool_name="local_retrieve", reason="retrieve", input={}),
                    ResearchPlanStep(tool_name="web_search", reason="fallback", input={}),
                ],
            ),
        }
    )

    assert route == "execute_tool"


def test_route_after_observation_continues_when_local_retrieve_finds_no_evidence() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(
                tool_name="local_retrieve",
                success=True,
                chunks=[],
            ),
            "current_step_index": 1,
            "plan": ResearchPlan(
                goal="answer",
                steps=[
                    ResearchPlanStep(tool_name="local_retrieve", reason="retrieve", input={}),
                    ResearchPlanStep(tool_name="web_search", reason="fallback", input={}),
                ],
            ),
        }
    )

    assert route == "execute_tool"


def test_route_after_observation_drafts_after_final_planned_step() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(tool_name="web_snippet_ingest", success=True),
            "current_step_index": 2,
            "plan": ResearchPlan(
                goal="answer",
                steps=[
                    ResearchPlanStep(tool_name="web_search", reason="fallback", input={}),
                    ResearchPlanStep(tool_name="web_snippet_ingest", reason="index", input={}),
                ],
            ),
        }
    )

    assert route == "draft_answer"
