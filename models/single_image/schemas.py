"""
Pydantic schemas for Single-Image Remote-Sensing Intelligence (VQA, Captioning, Grounding).
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class SingleImageTaskEnum(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    AUTO = "auto"


class GroundingBoundingBox(BaseModel):
    id: str
    label: str
    box: List[float] = Field(..., description="Normalized [ymin, xmin, ymax, xmax] in 0-100% range")
    confidence: float = Field(..., ge=0.0, le=1.0)
    color: Optional[str] = Field(default="#10b981", description="UI highlight color")


class GroundedRegion(BaseModel):
    id: str
    label: str
    category: str
    confidence: float
    pixel_bbox: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] in 0-100% canvas coordinates")
    geo_bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat] in WGS84")
    polygon: Dict[str, Any] = Field(..., description="GeoJSON polygon geometry")
    area_m2: float
    area_ha: float
    centroid: List[float] = Field(..., description="[lon, lat]")
    color: str = "#10b981"
    description: str = ""


class EvidenceTag(BaseModel):
    id: str
    title: str
    category: str = Field(default="Spectral Feature", description="e.g. Water Index, Built-up, SAR Backscatter")
    description: str
    confidence: float
    coordinates: Optional[List[float]] = None


class SingleImageRequest(BaseModel):
    image_id: Optional[str] = Field(default=None, description="Cached image ID from /api/uploads/file")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded PNG/JPEG image buffer")
    image_data_uri: Optional[str] = Field(default=None, description="Data URI string e.g. data:image/png;base64,...")
    query: str = Field(default="", description="Question or text grounding query (e.g. 'Find potentially vacant land')")
    task: SingleImageTaskEnum = Field(default=SingleImageTaskEnum.AUTO, description="Target task: vqa, captioning, grounding, auto")
    bands: Optional[List[int]] = Field(default=None, description="Custom 3-band combination for multispectral rasters")


class SingleImageResponse(BaseModel):
    task: str = Field(..., description="Executed task: Visual Question Answering, Captioning, or Grounding")
    query: str
    answer: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_formatted: str = "85%"
    model_name: str
    inference_time_ms: int
    raster_type: Optional[str] = None
    evidence_metadata: List[EvidenceTag] = Field(default_factory=list)
    bounding_boxes: List[GroundingBoundingBox] = Field(default_factory=list)
    grounded_regions: List[GroundedRegion] = Field(default_factory=list)
    spectral_indices: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None
    model_status: str = Field(default="READY", description="'READY', 'WEIGHTS_NOT_FOUND', 'MOCK_BENCHMARK', etc.")
    status_message: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
