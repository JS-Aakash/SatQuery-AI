"""
Agent Service implementation for SatQuery AI.
Orchestrates query classification, input validation, tool selection,
execution trace logging, and evidence aggregation.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .base import AgentService
from ..schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ExecutionStep,
    StepStatus,
    TaskType,
    ModalityEnum,
    GroundingBox,
    EvidenceItem,
    ChangeMapData,
)
from models.single_image import model_manager, SingleImageTaskEnum


class DefaultAgentService(AgentService):
    """
    Default query-driven agent controller.
    Sequences input checks, tool registry lookup, execution trace logging,
    and returns evidence-grounded outputs.
    """

    def classify_task_intent(self, query: str, has_paired_images: bool = False, is_cross_modal: bool = False) -> TaskType:
        from agent import AgentQueryPlanner, AgentTaskType
        inputs_mock = {
            "image_b": True if has_paired_images else None,
            "is_optical_sar_pair": is_cross_modal,
        }
        plan = AgentQueryPlanner.plan(query, inputs_mock)
        task_map = {
            AgentTaskType.SINGLE_VQA: TaskType.VQA,
            AgentTaskType.SINGLE_CAPTION: TaskType.CAPTIONING,
            AgentTaskType.GROUNDING: TaskType.GROUNDING,
            AgentTaskType.TEMPORAL_CHANGE: TaskType.CHANGE_DETECTION,
            AgentTaskType.OPTICAL_SAR: TaskType.OPTICAL_SAR_FUSION,
            AgentTaskType.GENERAL_ANALYSIS: TaskType.VQA,
            AgentTaskType.MULTI_TOOL: TaskType.GROUNDING,
        }
        return task_map.get(plan.task_type, TaskType.VQA)

    def execute_workflow(self, request: AnalysisRequest) -> AnalysisResponse:
        start_time = time.time()
        analysis_id = f"anl_{uuid.uuid4().hex[:10]}"
        query = request.query
        
        from ..api.routes import RASTER_CACHE
        from agent import agent_orchestrator, ToolStatus

        # Extract raster inputs
        img_a_id = request.image_metadata.id if request.image_metadata else None
        img_b_id = request.secondary_image_metadata.id if request.secondary_image_metadata else None

        raw_a = RASTER_CACHE.get(img_a_id) if img_a_id else None
        raw_b = RASTER_CACHE.get(img_b_id) if img_b_id else None

        if not raw_a and request.image_metadata and getattr(request.image_metadata, "preview_url", None):
            raw_a = request.image_metadata.preview_url
        if not raw_b and request.secondary_image_metadata and getattr(request.secondary_image_metadata, "preview_url", None):
            raw_b = request.secondary_image_metadata.preview_url
        if not raw_a and getattr(request, "image_data_uri", None):
            raw_a = request.image_data_uri

        # Check cross-modality
        is_cross_modal = False
        if request.image_metadata and request.secondary_image_metadata:
            mods = {request.image_metadata.modality, request.secondary_image_metadata.modality}
            if ModalityEnum.SAR in mods and (ModalityEnum.OPTICAL in mods or ModalityEnum.MULTISPECTRAL in mods):
                is_cross_modal = True

        inputs = {
            "image": raw_a,
            "image_a": raw_a,
            "image_b": raw_b,
            "optical": raw_a if not (request.image_metadata and request.image_metadata.modality == ModalityEnum.SAR) else raw_b,
            "sar": raw_b if not (request.image_metadata and request.image_metadata.modality == ModalityEnum.SAR) else raw_a,
            "is_optical_sar_pair": is_cross_modal,
        }

        # Execute agentic orchestration workflow
        orchestration_res = agent_orchestrator.orchestrate(
            query=query,
            inputs=inputs,
            context={"request_id": analysis_id, "image_metadata": request.image_metadata},
            preferred_task=request.preferred_task.value if request.preferred_task else None
        )

        # Convert execution trace
        trace: List[ExecutionStep] = []
        for step in orchestration_res.execution_trace:
            status_map = {
                ToolStatus.COMPLETED: StepStatus.COMPLETED,
                ToolStatus.FAILED: StepStatus.FAILED,
                ToolStatus.UNAVAILABLE: StepStatus.FAILED,
                ToolStatus.SKIPPED: StepStatus.PENDING,
            }
            trace.append(ExecutionStep(
                step_id=step.step,
                name=step.task,
                description=f"Executed tool '{step.tool}' [{step.model}]. {step.output_summary or ''}".strip(),
                status=status_map.get(step.status, StepStatus.COMPLETED),
                duration_ms=int(step.duration_ms),
                tool_or_model=step.tool,
                parameters=step.parameters,
                output_summary=step.output_summary
            ))

        # Convert GroundingBoxes
        grounding_boxes = [
            GroundingBox(
                id=b.get("id", f"box_{i}"),
                label=b.get("label", "Detected Feature"),
                box=b.get("box", [0, 0, 0, 0]),
                confidence=b.get("confidence", 0.9),
                color=b.get("color", "#10b981")
            )
            for i, b in enumerate(orchestration_res.grounding_boxes)
        ]

        # Convert EvidenceItems
        evidence = [
            EvidenceItem(
                id=e.get("id", f"ev_{i}"),
                title=e.get("title", "Evidence Item"),
                category=e.get("category", "Remote Sensing"),
                description=e.get("description", ""),
                confidence=e.get("confidence", 0.9),
                coordinates=e.get("coordinates"),
                modality_source=e.get("modality_source"),
                physical_metric=e.get("physical_metric")
            )
            for i, e in enumerate(orchestration_res.evidence)
        ]

        # Convert ChangeMapData if present
        change_map = None
        if orchestration_res.change_map:
            cm = orchestration_res.change_map
            change_map = ChangeMapData(
                change_type=cm.get("change_type", "Surface Modification"),
                changed_area_sq_m=cm.get("changed_area_sq_m", 0.0),
                changed_area_ha=cm.get("changed_area_ha", 0.0),
                changed_area_km2=cm.get("changed_area_km2", 0.0),
                changed_pixel_ratio=cm.get("changed_pixel_ratio", 0.0),
                heatmap_url=cm.get("heatmap_url", ""),
                polygons=cm.get("polygons", []),
                delta_ndvi=cm.get("delta_ndvi"),
                delta_ndbi=cm.get("delta_ndbi"),
                delta_ndwi=cm.get("delta_ndwi"),
            )

        task_type_map = {
            "Visual Question Answering": TaskType.VQA,
            "Scene Captioning & Description": TaskType.CAPTIONING,
            "Text-Guided Region Grounding": TaskType.GROUNDING,
            "Bi-Temporal Change Intelligence": TaskType.CHANGE_DETECTION,
            "Cross-Modal Optical-SAR Analysis": TaskType.OPTICAL_SAR_FUSION,
        }
        assigned_task = task_type_map.get(orchestration_res.task, TaskType.VQA)

        total_duration = max(orchestration_res.inference_time_ms + 40, int((time.time() - start_time) * 1000))

        return AnalysisResponse(
            analysis_id=analysis_id,
            query=query,
            task=assigned_task,
            status="COMPLETED",
            answer=orchestration_res.answer,
            confidence=orchestration_res.confidence,
            confidence_formatted=orchestration_res.confidence_formatted,
            evidence=evidence,
            execution_trace=trace,
            grounding_boxes=grounding_boxes,
            change_map=change_map,
            models_used=orchestration_res.models_used,
            execution_time_ms=total_duration,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_mock=orchestration_res.model_status != "READY",
            model_status=orchestration_res.model_status,
            status_message=None if not orchestration_res.warnings else "; ".join(orchestration_res.warnings),
            inference_time_ms=orchestration_res.inference_time_ms
        )
