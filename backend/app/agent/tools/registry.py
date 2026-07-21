from app.agent.models import ToolResult
from app.agent.tools.base import AgentTool


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Agent tool is not registered: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def run(self, name: str, input: dict) -> ToolResult:
        return await self.get(name).run(input)
