"""
Agent models, task type enums, and result schemas for the SatQuery AI Agentic Orchestration Layer.
"""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AgentTaskType(str, Enum):
    SINGLE_VQA = "SINGLE_VQA"
    SINGLE_CAPTION = "SINGLE_CAPTION"
    GROUNDING = "GROUNDING"
    TEMPORAL_CHANGE = "TEMPORAL_CHANGE"
    OPTICAL_SAR = "OPTICAL_SAR"
    GENERAL_ANALYSIS = "GENERAL_ANALYSIS"
    MULTI_TOOL = "MULTI_TOOL"


class ToolStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"


class ObservableTraceStep(BaseModel):
    """
    Observable metadata for an execution step.
    Never exposes private internal chain-of-thought.
    """
    step: int
    task: str
    tool: str
    model: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: ToolStatus = ToolStatus.COMPLETED
    duration_ms: float = 0.0
    observable_metadata: Optional[Dict[str, Any]] = None
    output_summary: Optional[str] = None


class ToolExecutionResult(BaseModel):
    """
    Standardized execution output returned by every registered specialist tool.
    """
    tool_name: str
    task: AgentTaskType
    status: ToolStatus = ToolStatus.COMPLETED
    duration_ms: float = 0.0
    model_name: str = "System Specialist"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    answer_fragment: Optional[str] = None
    confidence: float = 0.85
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_results: List[Dict[str, Any]] = Field(default_factory=list)
    observable_metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    output_summary: Optional[str] = None


class AgentSynthesisResult(BaseModel):
    """
    Combined synthesis returned by the Agent Orchestrator.
    """
    task: str
    task_type: AgentTaskType
    answer: str
    confidence: float
    confidence_formatted: str
    models_used: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_results: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[ObservableTraceStep] = Field(default_factory=list)
    execution_time_ms: int = 0
    inference_time_ms: int = 0
    warnings: List[str] = Field(default_factory=list)
    change_map: Optional[Dict[str, Any]] = None
    preview_url: Optional[str] = None
    model_status: str = "READY"
