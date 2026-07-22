from app.agent.graph import infer_stop_reason, route_after_observation
from app.agent.models import ChatWorkflowRequest, ContextQuality, ResearchPlan, ResearchPlanStep, ToolResult


def test_route_after_observation_regrades_when_local_retrieve_finds_evidence() -> None:
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

    assert route == "quality_gate"


def test_route_after_observation_continues_to_ingest_after_web_search() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(
                tool_name="web_search",
                success=True,
                chunks=[{"id": "web:1", "text": "Evidence."}],
            ),
            "current_step_index": 1,
            "plan": ResearchPlan(
                goal="answer",
                steps=[
                    ResearchPlanStep(tool_name="web_search", reason="fallback", input={}),
                    ResearchPlanStep(tool_name="web_snippet_ingest", reason="index", input={}),
                ],
            ),
        }
    )

    assert route == "execute_tool"


def test_route_after_observation_regrades_after_final_evidence_step() -> None:
    route = route_after_observation(
        {
            "current_tool_result": ToolResult(
                tool_name="web_search",
                success=True,
                chunks=[{"id": "web:1", "text": "Evidence."}],
            ),
            "current_step_index": 1,
            "plan": ResearchPlan(
                goal="answer",
                steps=[ResearchPlanStep(tool_name="web_search", reason="fallback", input={})],
            ),
        }
    )

    assert route == "quality_gate"


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

    assert route == "quality_gate"


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


def test_infer_stop_reason_detects_agent_step_limit() -> None:
    stop_reason = infer_stop_reason(
        {
            "request": ChatWorkflowRequest("Question"),
            "trace": [
                {
                    "stage": "execute_tool",
                    "success": False,
                    "reason": "Agent step limit reached: 6/6.",
                }
            ],
        },
        answer="I don't know",
    )

    assert stop_reason == "step_limit_reached"


def test_infer_stop_reason_detects_tool_limit() -> None:
    stop_reason = infer_stop_reason(
        {
            "request": ChatWorkflowRequest("Question"),
            "trace": [
                {
                    "stage": "execute_tool",
                    "success": False,
                    "reason": "Tool limit reached for web_search: 2/2.",
                }
            ],
        },
        answer="I don't know",
    )

    assert stop_reason == "tool_limit_reached"


def test_infer_stop_reason_detects_planner_without_valid_steps() -> None:
    stop_reason = infer_stop_reason(
        {
            "request": ChatWorkflowRequest("Question"),
            "plan": ResearchPlan(goal="Question", steps=[]),
            "trace": [],
        },
        answer="I don't know",
    )

    assert stop_reason == "planner_no_valid_steps"
