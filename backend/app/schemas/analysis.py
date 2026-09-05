"""
Pydantic schemas for analysis requests, agentic execution traces, and responses.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .imagery import ImageMetadata, ModalityEnum


class TaskType(str, Enum):
    VQA = "Visual Question Answering"
    GROUNDING = "Text-Guided Region Grounding"
    CAPTIONING = "Scene Description & Captioning"
    CHANGE_DETECTION = "Bi-Temporal Change Analysis"
    OPTICAL_SAR_FUSION = "Cross-Modal Optical-SAR Analysis"
    LAND_COVER_MAPPING = "Land Cover & Feature Identification"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep(BaseModel):
    step_id: int
    name: str
    description: str
    status: StepStatus = StepStatus.COMPLETED
    duration_ms: int = Field(default=0, description="Step duration in milliseconds")
    tool_or_model: Optional[str] = Field(default=None, description="Name of the specialist tool/model selected")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Permitted task parameters")
    output_summary: Optional[str] = Field(default=None, description="Observable step output summary")


class EvidenceItem(BaseModel):
    id: str
    title: str
    category: str = Field(..., description="e.g. Spectral Feature, SAR Backscatter, Temporal Difference, Spatial Region")
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    coordinates: Optional[List[float]] = Field(default=None, description="[x1, y1, x2, y2] bounding coordinates if applicable")
    modality_source: Optional[str] = Field(default=None, description="'optical', 'sar', or 'combined'")
    physical_metric: Optional[str] = Field(default=None, description="Physical value (e.g. dB or NDVI)")


class GroundingBox(BaseModel):
    id: str
    label: str
    box: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized 0-100%")
    confidence: float = Field(..., ge=0.0, le=1.0)
    color: Optional[str] = Field(default="#10b981", description="Highlight stroke color")


class ChangeMapData(BaseModel):
    has_change: bool = False
    change_percentage: float = 0.0
    change_type: Optional[str] = None
    change_mask_url: Optional[str] = None
    total_changed_area_km2: Optional[float] = None
    total_changed_area_hectares: Optional[float] = None
    delta_ndvi: Optional[float] = None
    delta_ndwi: Optional[float] = None
    delta_ndbi: Optional[float] = None
    supporting_indicators: List[str] = Field(default_factory=list)
    legend: Dict[str, str] = Field(default_factory=dict)


class ChangeAnalysisRequest(BaseModel):
    image_a_id: Optional[str] = Field(default=None, description="Primary raster ID (T1 Before)")
    image_b_id: Optional[str] = Field(default=None, description="Secondary raster ID (T2 After)")
    query: str = Field(default="What changed between these two images?", description="Change query")
    image_a_base64: Optional[str] = None
    image_b_base64: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ChangeAnalysisResponse(BaseModel):
    analysis_id: str
    query: str
    textual_description: str
    has_significant_change: bool
    change_percentage: float
    confidence: float
    confidence_formatted: str
    total_changed_area_km2: float
    total_changed_area_hectares: float
    changed_regions: List[Dict[str, Any]] = Field(default_factory=list)
    spectral_indices: Optional[Dict[str, Any]] = None
    change_mask_url: Optional[str] = None
    legend: Dict[str, str] = Field(default_factory=dict)
    models_used: List[str] = Field(default_factory=list)
    model_status: str = "READY"
    status_message: str = "Success"
    inference_time_ms: int = 0
    created_at: str = ""


class OpticalSARAnalysisRequest(BaseModel):
    optical_image_id: Optional[str] = Field(default=None, description="Optical raster ID")
    sar_image_id: Optional[str] = Field(default=None, description="SAR raster ID")
    query: str = Field(default="Analyze the optical and SAR images together.", description="Multimodal query")
    optical_base64: Optional[str] = None
    sar_base64: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OpticalSARAnalysisResponse(BaseModel):
    analysis_id: str
    query: str
    answer: str
    confidence: float
    confidence_formatted: str
    optical_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sar_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    combined_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    grounding_regions: List[Dict[str, Any]] = Field(default_factory=list)
    cloud_penetrated_area_km2: float = 0.0
    radar_confirmed_structures_count: int = 0
    fused_composite_url: Optional[str] = None
    sar_preview_url: Optional[str] = None
    optical_preview_url: Optional[str] = None
    legend: Dict[str, str] = Field(default_factory=dict)
    models_used: List[str] = Field(default_factory=list)
    model_status: str = "READY"
    status_message: str = "Success"
    inference_time_ms: int = 0
    created_at: str = ""


class AnalysisRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question or instruction")
    image_metadata: Optional[ImageMetadata] = None
    secondary_image_metadata: Optional[ImageMetadata] = None
    input_mode: str = Field(default="upload", description="'upload', 'search', or 'aoi'")
    aoi_coordinates: Optional[List[List[float]]] = Field(default=None, description="AOI polygon coordinates")
    preferred_task: Optional[TaskType] = None
    image_data_uri: Optional[str] = Field(default=None, description="Data URI (SVG or base64) of the client imagery canvas")


class AnalysisResponse(BaseModel):
    analysis_id: str
    query: str
    task: TaskType
    status: str = "COMPLETED"
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated calibrated confidence")
    confidence_formatted: str = "87%"
    evidence: List[EvidenceItem] = Field(default_factory=list)
    execution_trace: List[ExecutionStep] = Field(default_factory=list)
    grounding_boxes: List[GroundingBox] = Field(default_factory=list)
    change_map: Optional[ChangeMapData] = None
    models_used: List[str] = Field(default_factory=list)
    execution_time_ms: int = Field(default=0)
    created_at: str
    is_mock: bool = Field(default=True, description="Indicates if running in mock/demo UI mode prior to model training")
    model_status: Optional[str] = Field(default=None, description="READY, WEIGHTS_NOT_FOUND, or BENCHMARK_EVALUATION_ACTIVE")
    status_message: Optional[str] = Field(default=None, description="Detailed diagnostic or hardware status message")
    inference_time_ms: Optional[int] = Field(default=None, description="Raw model inference latency")
