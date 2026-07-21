from typing import Protocol

from app.agent.models import ToolResult


class AgentTool(Protocol):
    name: str

    async def run(self, input: dict) -> ToolResult:
        ...
