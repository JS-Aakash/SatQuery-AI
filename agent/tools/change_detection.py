"""
Bi-Temporal Change Detection Specialist Tool.
"""
import time
import uuid
from typing import Dict, Any, List, Optional

from ..registry.base import BaseSpecialistTool
from ..models import AgentTaskType, ToolExecutionResult, ToolStatus
from models.change_detection import change_manager


class ChangeDetectionTool(BaseSpecialistTool):
    """Specialist tool for bi-temporal surface change intelligence, area quantification, and change-maps."""

    @property
    def name(self) -> str:
        return "ChangeDetection"

    @property
    def description(self) -> str:
        return "Analyzes two temporally separated images to identify, polygonize, and quantify surface changes."

    @property
    def accepted_inputs(self) -> List[str]:
        return ["image_a", "image_b"]

    @property
    def supported_tasks(self) -> List[AgentTaskType]:
        return [AgentTaskType.TEMPORAL_CHANGE, AgentTaskType.GENERAL_ANALYSIS]

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "answer": "str",
            "confidence": "float",
            "change_map": "dict",
            "polygons": "list[dict]",
            "area_metrics": "dict",
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
        img_a = inputs.get("image_a")
        img_b = inputs.get("image_b")
        query = context.get("query", "What changes occurred between before and after images?")

        try:
            res = change_manager.infer(
                image_t1=img_a or b"sample_t1",
                image_t2=img_b or b"sample_t2",
                query=query
            )
            duration = (time.time() - start) * 1000.0

            evidence_items = []
            if res.spectral_indices:
                evidence_items.append({
                    "id": f"ev_ndvi_{uuid.uuid4().hex[:4]}",
                    "title": "Vegetation Index Dynamic Delta",
                    "category": "Spectral Index (ΔNDVI)",
                    "description": f"Mean ΔNDVI: {res.spectral_indices.delta_ndvi:+.3f} across temporal baseline.",
                    "confidence": 0.92,
                    "modality_source": "optical",
                    "physical_metric": f"ΔNDVI: {res.spectral_indices.delta_ndvi:+.3f}",
                })
                evidence_items.append({
                    "id": f"ev_ndbi_{uuid.uuid4().hex[:4]}",
                    "title": "Built-Up Impervious Surface Expansion",
                    "category": "Spectral Index (ΔNDBI)",
                    "description": f"Mean ΔNDBI: {res.spectral_indices.delta_ndbi:+.3f} indicative of structural construction.",
                    "confidence": 0.90,
                    "modality_source": "optical",
                    "physical_metric": f"ΔNDBI: {res.spectral_indices.delta_ndbi:+.3f}",
                })

            boxes = [
                {
                    "id": p.id,
                    "label": p.label,
                    "box": p.bounding_box,
                    "confidence": p.confidence,
                    "color": p.color,
                }
                for p in res.changed_regions
            ]

            change_map_dict = {
                "change_type": "Multi-Class Land-Cover Conversion",
                "changed_area_sq_m": res.total_changed_area_hectares * 10000.0,
                "changed_area_ha": res.total_changed_area_hectares,
                "changed_area_km2": res.total_changed_area_km2,
                "changed_pixel_ratio": res.change_percentage / 100.0,
                "heatmap_url": res.change_mask_url,
                "polygons": [p.model_dump() for p in res.changed_regions],
                "delta_ndvi": res.spectral_indices.delta_ndvi if res.spectral_indices else None,
                "delta_ndbi": res.spectral_indices.delta_ndbi if res.spectral_indices else None,
                "delta_ndwi": res.spectral_indices.delta_ndwi if res.spectral_indices else None,
            }

            model_name = res.models_used[0] if res.models_used else "SatQuery-ChangeNet"

            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.TEMPORAL_CHANGE,
                status=ToolStatus.COMPLETED,
                duration_ms=round(duration, 2),
                model_name=model_name,
                parameters={"task": "TEMPORAL_CHANGE", "query": query},
                data={
                    "change_map": change_map_dict,
                    "model_status": res.model_status,
                },
                answer_fragment=res.textual_description,
                confidence=res.confidence,
                evidence=evidence_items,
                spatial_results=boxes,
                observable_metadata={
                    "changed_area_ha": res.total_changed_area_hectares,
                    "num_regions": len(res.changed_regions),
                    "change_percentage": res.change_percentage,
                },
                output_summary=f"Detected temporal changes across {res.total_changed_area_hectares:.2f} ha ({len(res.changed_regions)} regions)",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task=AgentTaskType.TEMPORAL_CHANGE,
                status=ToolStatus.FAILED,
                duration_ms=round(duration, 2),
                warnings=[f"ChangeDetection execution failed: {str(e)}"],
                answer_fragment="Error during bi-temporal change detection.",
                confidence=0.0,
            )
