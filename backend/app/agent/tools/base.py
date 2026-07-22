from typing import Protocol

from app.agent.models import ToolDescription, ToolResult


class AgentTool(Protocol):
    name: str
    description: str
    input_schema: dict
    when_to_use: str
    failure_modes: list[str]

    async def run(self, input: dict) -> ToolResult:
        ...


def describe_tool(tool: AgentTool) -> ToolDescription:
    return ToolDescription(
        name=tool.name,
        description=getattr(tool, "description", f"Run {tool.name}."),
        input_schema=dict(getattr(tool, "input_schema", {})),
        when_to_use=getattr(tool, "when_to_use", "Use when selected by the planner."),
        failure_modes=list(getattr(tool, "failure_modes", [])),
    )
