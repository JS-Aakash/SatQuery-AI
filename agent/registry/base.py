"""
Standardized specialist tool abstract interface.
All specialist tools registered in the SatQuery AI Agent Orchestration Layer MUST subclass BaseSpecialistTool.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..models import AgentTaskType, ToolExecutionResult


class BaseSpecialistTool(ABC):
    """
    Standardized abstract base class for remote-sensing specialist tools and models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this specialist performs."""
        pass

    @property
    @abstractmethod
    def accepted_inputs(self) -> List[str]:
        """List of accepted input keys (e.g. ['image'], ['image_a', 'image_b'], ['optical', 'sar'])."""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> List[AgentTaskType]:
        """List of AgentTaskTypes supported by this tool."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Schema description of data fields returned by execute()."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the underlying model/service is available for execution."""
        pass

    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        """
        Executes the specialist tool.
        Must never modify raw inputs and must return a standardized ToolExecutionResult.
        """
        pass
