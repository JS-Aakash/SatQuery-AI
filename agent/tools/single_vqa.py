"""
Single Image Visual Question Answering (VQA) Specialist Tool.
"""
import time
from typing import Dict, Any, List, Optional

from ..registry.base import BaseSpecialistTool
from ..models import AgentTaskType, ToolExecutionResult, ToolStatus
from models.single_image import model_manager, SingleImageTaskEnum


class SingleImageVQATool(BaseSpecialistTool):
    """Specialist tool for remote-sensing single-image Visual Question Answering."""

    @property
    def name(self) -> str:
        return "SingleImageVQA"

    @property
    def description(self) -> str:
        return "Answers open-ended natural-language questions about single optical or SAR satellite scenes."

    @property
    def accepted_inputs(self) -> List[str]:
        return ["image"]

    @property
    def supported_tasks(self) -> List[AgentTaskType]:
        return [AgentTaskType.SINGLE_VQA, AgentTaskType.GENERAL_ANALYSIS]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "answer": "str",
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
        query = context.get("query", "Analyze the remote sensing scene.")

        try:
            res = model_manager.infer(
                image_input=image_input or b"sample_raster",
                query=query,
                task=SingleImageTaskEnum.VQA
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
                task=AgentTaskType.SINGLE_VQA,
                status=ToolStatus.COMPLETED,
                duration_ms=round(duration, 2),
                model_name=res.model_name,
                parameters={"task": "VQA", "query": query},
                data={"answer": res.answer, "model_status": res.model_status},
                answer_fragment=res.answer,
                confidence=res.confidence,
                evidence=evidence_items,
                spatial_results=[],
                observable_metadata={"model_status": res.model_status, "latency_ms": round(duration, 2)},
                output_summary=f"Synthesized VQA answer ({len(res.answer.split())} words, {int(res.confidence*100)}% confidence)",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.SINGLE_VQA,
                status=ToolStatus.FAILED,
                duration_ms=round(duration, 2),
                warnings=[f"SingleImageVQA execution failed: {str(e)}"],
                answer_fragment="Error during VQA execution.",
                confidence=0.0,
            )
