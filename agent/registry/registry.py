"""
Tool Registry singleton for the SatQuery AI Agent Orchestration Layer.
Manages tool discovery, registration, input validation, and task routing.
"""
from typing import Dict, List, Optional
import logging

from .base import BaseSpecialistTool
from ..models import AgentTaskType

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized registry of specialist remote-sensing tools and models.
    """

    def __init__(self):
        self._tools: Dict[str, BaseSpecialistTool] = {}

    def register(self, tool: BaseSpecialistTool) -> None:
        """Registers a specialist tool."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool registration: {tool.name}")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} for tasks: {[t.value for t in tool.supported_tasks]}")

    def get_tool(self, name: str) -> Optional[BaseSpecialistTool]:
        """Gets a tool by unique name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseSpecialistTool]:
        """Returns a list of all registered tools."""
        return list(self._tools.values())

    def find_tools_for_task(
        self,
        task: AgentTaskType,
        available_inputs: Optional[List[str]] = None
    ) -> List[BaseSpecialistTool]:
        """
        Finds tools matching the requested task and input availability.
        """
        matching = []
        for tool in self._tools.values():
            if task in tool.supported_tasks:
                if available_inputs is not None:
                    # Check if required inputs are satisfied
                    reqs = set(tool.accepted_inputs)
                    avail = set(available_inputs)
                    if not reqs.issubset(avail):
                        continue
                matching.append(tool)
        return matching

    def get_registered_names(self) -> List[str]:
        """Returns list of registered tool names."""
        return list(self._tools.keys())


# Singleton instance
tool_registry = ToolRegistry()
