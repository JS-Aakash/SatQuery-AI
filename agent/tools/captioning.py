"""
Scene Captioning & Land-Cover Description Specialist Tool.
"""
import time
from typing import Dict, Any, List, Optional

from ..registry.base import BaseSpecialistTool
from ..models import AgentTaskType, ToolExecutionResult, ToolStatus
from models.single_image import model_manager, SingleImageTaskEnum


class CaptioningTool(BaseSpecialistTool):
    """Specialist tool for generating detailed land-cover summaries and scene descriptions."""

    @property
    def name(self) -> str:
        return "Captioning"

    @property
    def description(self) -> str:
        return "Generates semantic captions and structural land-cover overviews of satellite scenes."

    @property
    def accepted_inputs(self) -> List[str]:
        return ["image"]

    @property
    def supported_tasks(self) -> List[AgentTaskType]:
        return [AgentTaskType.SINGLE_CAPTION, AgentTaskType.GENERAL_ANALYSIS]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "caption": "str",
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
        query = context.get("query", "Describe the scene.")

        try:
            res = model_manager.infer(
                image_input=image_input or b"sample_raster",
                query=query,
                task=SingleImageTaskEnum.CAPTIONING
            )
            duration = (time.time() - start) * 1000.0

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
                task=AgentTaskType.SINGLE_CAPTION,
                status=ToolStatus.COMPLETED,
                duration_ms=round(duration, 2),
                model_name=res.model_name,
                parameters={"task": "CAPTIONING", "query": query},
                data={"caption": res.answer, "model_status": res.model_status},
                answer_fragment=res.answer,
                confidence=res.confidence,
                evidence=evidence_items,
                spatial_results=[],
                observable_metadata={"model_status": res.model_status, "word_count": len(res.answer.split())},
                output_summary=f"Generated scene description ({len(res.answer.split())} words)",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.SINGLE_CAPTION,
                status=ToolStatus.FAILED,
                duration_ms=round(duration, 2),
                warnings=[f"Captioning execution failed: {str(e)}"],
                answer_fragment="Error during scene captioning.",
                confidence=0.0,
            )
