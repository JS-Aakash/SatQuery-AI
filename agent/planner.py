"""
Agent Query Planner.
Interprets natural-language query intent, inspects input raster modalities,
and formulates single-step or multi-tool execution plans.
"""
from typing import Dict, Any, List, Optional
from .models import AgentTaskType


class QueryPlan:
    """Represents an execution plan formulated by the Agent."""
    def __init__(
        self,
        task_type: AgentTaskType,
        selected_tool_names: List[str],
        reasoning_summary: str,
        is_multi_tool: bool = False,
    ):
        self.task_type = task_type
        self.selected_tool_names = selected_tool_names
        self.reasoning_summary = reasoning_summary
        self.is_multi_tool = is_multi_tool


class AgentQueryPlanner:
    """
    Analyzes query semantics and input raster availability to construct execution plans.
    """

    @staticmethod
    def plan(
        query: str,
        available_inputs: Dict[str, Any],
        preferred_task: Optional[str] = None,
    ) -> QueryPlan:
        q = query.strip().lower()
        
        has_paired = ("image_b" in available_inputs and available_inputs["image_b"] is not None) or \
                     ("secondary_image" in available_inputs and available_inputs["secondary_image"] is not None)
        
        has_optical_sar = (available_inputs.get("optical") is not None and available_inputs.get("sar") is not None) or \
                          (available_inputs.get("is_optical_sar_pair", False) and has_paired)
        
        # 1. Check if user explicitly passed a preferred task
        if preferred_task:
            pt = preferred_task.upper()
            if pt in [t.value for t in AgentTaskType]:
                task_enum = AgentTaskType(pt)
                tool_map = {
                    AgentTaskType.SINGLE_VQA: ["SingleImageVQA"],
                    AgentTaskType.SINGLE_CAPTION: ["Captioning"],
                    AgentTaskType.GROUNDING: ["Grounding"],
                    AgentTaskType.TEMPORAL_CHANGE: ["ChangeDetection"],
                    AgentTaskType.OPTICAL_SAR: ["OpticalSAR"],
                    AgentTaskType.GENERAL_ANALYSIS: ["SingleImageVQA", "SpectralAnalysis"],
                }
                return QueryPlan(
                    task_type=task_enum,
                    selected_tool_names=tool_map.get(task_enum, ["SingleImageVQA"]),
                    reasoning_summary=f"Dispatched via explicit task override: {task_enum.value}",
                    is_multi_tool=len(tool_map.get(task_enum, [])) > 1,
                )

        # 2. Check for Optical + SAR Cross-Modal Intent or Inputs
        if has_optical_sar or "sar and optical" in q or "optical and sar" in q or "cross-modal" in q or "radar and optical" in q:
            return QueryPlan(
                task_type=AgentTaskType.OPTICAL_SAR,
                selected_tool_names=["OpticalSAR"],
                reasoning_summary="Co-registered Optical + SAR inputs detected. Dispatching cross-modal evidence fusion engine.",
                is_multi_tool=False,
            )

        # 3. Check for Multi-Tool compound queries (e.g., Temporal Change + NDVI Spectral evidence + Location Grounding)
        if ("has vegetation decreased" in q or "vegetation change" in q or "where did change occur" in q) and (has_paired or "between" in q or "dates" in q):
            return QueryPlan(
                task_type=AgentTaskType.TEMPORAL_CHANGE,
                selected_tool_names=["ChangeDetection", "SpectralAnalysis", "SingleImageVQA"],
                reasoning_summary="Multi-tool query identified: executing Bi-Temporal Change Detection, Spectral NDVI verification, and VQA synthesis.",
                is_multi_tool=True,
            )

        # 4. Check for Bi-Temporal Change Detection
        if has_paired or "change" in q or "differ" in q or "before and after" in q or "temporal" in q or "increased" in q or "decreased" in q:
            return QueryPlan(
                task_type=AgentTaskType.TEMPORAL_CHANGE,
                selected_tool_names=["ChangeDetection"],
                reasoning_summary="Temporal change query detected over bi-temporal image pair.",
                is_multi_tool=False,
            )

        # 5. Check for Spatial Grounding / Localization
        if any(k in q for k in ["highlight", "ground", "where is", "where are", "locate", "find the", "find", "bounding box", "draw box", "delineate", "show me", "mark"]):
            return QueryPlan(
                task_type=AgentTaskType.GROUNDING,
                selected_tool_names=["Grounding"],
                reasoning_summary="Spatial localization and phrase-grounding intent detected.",
                is_multi_tool=False,
            )

        # 6. Check for Scene Captioning / Land-Cover Overview
        if "describe" in q or "caption" in q or "what land-cover" in q or "overview" in q or "summarize scene" in q:
            return QueryPlan(
                task_type=AgentTaskType.SINGLE_CAPTION,
                selected_tool_names=["Captioning"],
                reasoning_summary="Dense scene description and land-cover captioning intent identified.",
                is_multi_tool=False,
            )

        # 7. Check for Compound Grounding + VQA (Multi-Tool)
        if ("what" in q or "is there" in q) and ("locate" in q or "show me" in q or "highlight" in q):
            return QueryPlan(
                task_type=AgentTaskType.MULTI_TOOL,
                selected_tool_names=["SingleImageVQA", "Grounding"],
                reasoning_summary="Compound VQA and spatial localization query detected.",
                is_multi_tool=True,
            )

        # 8. Default to Single Image VQA
        return QueryPlan(
            task_type=AgentTaskType.SINGLE_VQA,
            selected_tool_names=["SingleImageVQA"],
            reasoning_summary="Natural language remote-sensing question mapped to SingleImageVQA specialist.",
            is_multi_tool=False,
        )
