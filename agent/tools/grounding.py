"""
Text-Guided Region Grounding Specialist Tool.
"""
import time
from typing import Dict, Any, List, Optional

from ..registry.base import BaseSpecialistTool
from ..models import AgentTaskType, ToolExecutionResult, ToolStatus
from models.single_image import model_manager, SingleImageTaskEnum


class GroundingTool(BaseSpecialistTool):
    """Specialist tool for text-guided region grounding with spatial bounding boxes."""

    @property
    def name(self) -> str:
        return "Grounding"

    @property
    def description(self) -> str:
        return "Detects, localizes, and generates spatial bounding boxes for objects or features mentioned in query text."

    @property
    def accepted_inputs(self) -> List[str]:
        return ["image"]

    @property
    def supported_tasks(self) -> List[AgentTaskType]:
        return [AgentTaskType.GROUNDING, AgentTaskType.GENERAL_ANALYSIS]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "bounding_boxes": "list[dict]",
            "confidence": "float",
            "evidence": "list[dict]",
            "model_name": "str",
        }

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        start = time.time()
        image_input = inputs.get("image") or inputs.get("image_a") or inputs.get("optical") or inputs.get("image_b")
        query = context.get("query", "Highlight the main object.")

        try:
            res = model_manager.infer(
                image_input=image_input or b"sample_raster",
                query=query,
                task=SingleImageTaskEnum.GROUNDING
            )
            duration = (time.time() - start) * 1000.0

            boxes = [
                {
                    "id": b.id,
                    "label": b.label,
                    "box": b.box,
                    "confidence": b.confidence,
                    "color": b.color,
                }
                for b in res.bounding_boxes
            ]

            evidence_items = [
                {
                    "id": ev.id,
                    "title": ev.title,
                    "category": ev.category,
                    "description": ev.description,
                    "confidence": ev.confidence,
                    "coordinates": ev.coordinates,
                    "modality_source": "optical",
                }
                for ev in res.evidence_metadata
            ]

            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.GROUNDING,
                status=ToolStatus.COMPLETED,
                duration_ms=round(duration, 2),
                model_name=res.model_name,
                parameters={"task": "GROUNDING", "query": query},
                data={"bounding_boxes": boxes, "model_status": res.model_status},
                answer_fragment=res.answer,
                confidence=res.confidence,
                evidence=evidence_items,
                spatial_results=boxes,
                observable_metadata={"num_regions_detected": len(boxes), "model_status": res.model_status},
                output_summary=f"Localized {len(boxes)} spatial region(s) with high geometric confidence",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.GROUNDING,
                status=ToolStatus.FAILED,
                duration_ms=round(duration, 2),
                warnings=[f"Grounding execution failed: {str(e)}"],
                answer_fragment="Error during region grounding.",
                confidence=0.0,
            )
