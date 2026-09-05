"""
Cross-Modal Optical + SAR Intelligence Specialist Tool.
"""
import time
from typing import Dict, Any, List, Optional

from ..registry.base import BaseSpecialistTool
from ..models import AgentTaskType, ToolExecutionResult, ToolStatus
from models.optical_sar import optical_sar_manager


class OpticalSARTool(BaseSpecialistTool):
    """Specialist tool for co-registered Optical + SAR joint reasoning, backscatter analysis, and material disambiguation."""

    @property
    def name(self) -> str:
        return "OpticalSAR"

    @property
    def description(self) -> str:
        return "Fuses optical multispectral reflectance with SAR radar backscatter physics to produce all-weather intelligence."

    @property
    def accepted_inputs(self) -> List[str]:
        return ["optical", "sar"]

    @property
    def supported_tasks(self) -> List[AgentTaskType]:
        return [AgentTaskType.OPTICAL_SAR, AgentTaskType.GENERAL_ANALYSIS]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "answer": "str",
            "confidence": "float",
            "optical_evidence": "list[dict]",
            "sar_evidence": "list[dict]",
            "combined_evidence": "list[dict]",
            "grounding_regions": "list[dict]",
            "fused_composite_url": "str",
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
        opt = inputs.get("optical")
        sar = inputs.get("sar")
        query = context.get("query", "Analyze optical and SAR imagery together.")

        try:
            res = optical_sar_manager.infer(
                optical_input=opt or b"sample_optical",
                sar_input=sar or b"sample_sar",
                query=query
            )
            duration = (time.time() - start) * 1000.0

            all_evidence = []
            for ev in (res.optical_evidence + res.sar_evidence + res.combined_evidence):
                all_evidence.append({
                    "id": ev.id,
                    "title": ev.title,
                    "category": ev.category,
                    "description": ev.description,
                    "confidence": ev.confidence,
                    "modality_source": ev.modality_source,
                    "physical_metric": ev.physical_metric,
                    "coordinates": ev.coordinates,
                })

            boxes = [
                {
                    "id": reg.id,
                    "label": reg.label,
                    "box": reg.box,
                    "confidence": reg.confidence,
                    "color": reg.color,
                }
                for reg in res.grounding_regions
            ]

            model_name = res.models_used[0] if getattr(res, "models_used", None) else "Cross-Modal Evidence Fusion Model"

            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.OPTICAL_SAR,
                status=ToolStatus.COMPLETED,
                duration_ms=round(duration, 2),
                model_name=model_name,
                parameters={"task": "OPTICAL_SAR_FUSION", "query": query},
                data={
                    "fused_composite_url": res.fused_composite_url,
                    "sar_preview_url": res.sar_preview_url,
                    "optical_preview_url": res.optical_preview_url,
                    "model_status": res.model_status,
                },
                answer_fragment=res.answer,
                confidence=res.confidence,
                evidence=all_evidence,
                spatial_results=boxes,
                observable_metadata={
                    "optical_ev_count": len(res.optical_evidence),
                    "sar_ev_count": len(res.sar_evidence),
                    "combined_ev_count": len(res.combined_evidence),
                },
                output_summary=f"Fused optical + SAR: {len(res.optical_evidence)} optical, {len(res.sar_evidence)} SAR, {len(res.combined_evidence)} combined evidence items",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.OPTICAL_SAR,
                status=ToolStatus.FAILED,
                duration_ms=round(duration, 2),
                warnings=[f"OpticalSAR execution failed: {str(e)}"],
                answer_fragment="Error during optical-SAR fusion.",
                confidence=0.0,
            )
