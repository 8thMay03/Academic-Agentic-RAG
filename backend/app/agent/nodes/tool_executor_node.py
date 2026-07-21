from dataclasses import replace

from app.agent.models import AgentLimits, ToolCall, ToolResult, append_trace
from app.agent.nodes.observer_node import compact_tool_result
from app.agent.state import AgenticRAGState
from app.agent.tools.execution import prepare_tool_input, run_tool_with_timeout, tool_limit_error


async def tool_executor_node(state: AgenticRAGState) -> AgenticRAGState:
    tool_registry = state["tool_registry"]
    plan = state["plan"]
    limits = state.get("limits") or AgentLimits()
    current_step_index = state.get("current_step_index", 0)
    step = plan.steps[current_step_index]
    limit_error = tool_limit_error(state, step.tool_name, limits)
    if limit_error:
        state = _with_tool_call(state, step.tool_name, step.input, step.reason, current_step_index)
        result = ToolResult(tool_name=step.tool_name, success=False, error=limit_error)
        return _with_tool_result(state, result, step.reason, current_step_index)

    prepared_input = prepare_tool_input(step.tool_name, step.input, state)
    if prepared_input.error:
        state = _with_tool_call(state, step.tool_name, prepared_input.input, step.reason, current_step_index)
        result = ToolResult(tool_name=step.tool_name, success=False, error=prepared_input.error)
        return _with_tool_result(state, result, step.reason, current_step_index)

    state = _with_tool_call(state, step.tool_name, prepared_input.input, step.reason, current_step_index)
    result = await run_tool_with_timeout(tool_registry, step.tool_name, prepared_input.input, limits)
    if step.tool_name == "web_search":
        web_chunks = result.chunks or []
        state = {
            **state,
            "web_chunks": web_chunks,
            "evidence": _merged_evidence(state.get("local_chunks", []), web_chunks),
        }
    elif step.tool_name == "local_retrieve":
        local_chunks = result.chunks or []
        state = {
            **state,
            "local_chunks": local_chunks,
            "evidence": _merged_evidence(local_chunks, state.get("web_chunks", [])),
            "refreshed_local_context": True,
        }

    if step.tool_name == "web_snippet_ingest":
        metadata = result.metadata or {}
        result = replace(
            result,
            metadata={
                **metadata,
                "snippets_ingested": int(metadata.get("snippets_ingested") or 0),
            },
        )

    return _with_tool_result(state, result, step.reason, current_step_index)


def _merged_evidence(local_chunks: list[dict], web_chunks: list[dict]) -> list[dict]:
    return [*local_chunks, *web_chunks]


def _with_tool_call(
    state: AgenticRAGState,
    tool_name: str,
    tool_input: dict,
    reason: str,
    current_step_index: int,
) -> AgenticRAGState:
    return {
        **state,
        "tool_calls": [
            *state.get("tool_calls", []),
            ToolCall(
                tool_name=tool_name,
                input=dict(tool_input),
                reason=reason,
                step_index=current_step_index,
            ),
        ],
    }


def _with_tool_result(
    state: AgenticRAGState,
    result: ToolResult,
    reason: str,
    current_step_index: int,
) -> AgenticRAGState:
    trace = append_trace(
        state.get("trace", []),
        "execute_tool",
        tool_name=result.tool_name,
        step_index=current_step_index,
        success=result.success,
        reason=reason if result.success else result.error or reason,
        tool_result=compact_tool_result(result),
    )
    return {
        **state,
        "current_tool_result": result,
        "trace": trace,
    }
