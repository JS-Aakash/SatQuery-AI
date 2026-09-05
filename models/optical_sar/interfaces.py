"""
Data structures and interfaces for Cross-Modal Optical + SAR Intelligence.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field


class ModalityAttributedEvidence(BaseModel):
    id: str
    title: str
    modality_source: Literal["optical", "sar", "combined"] = Field(
        ...,
        description="Originating modality source: 'optical' (spectral reflectance), 'sar' (radar backscatter), or 'combined' (fused concordance)"
    )
    category: str = Field(..., description="e.g. SAR Double-Bounce, Optical NDVI, Cloud Penetration, Specular Delineation")
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    coordinates: Optional[List[float]] = Field(default=None, description="[ymin, xmin, ymax, xmax] 0-100%")
    physical_metric: Optional[str] = None


class OpticalSARGroundingRegion(BaseModel):
    id: str
    label: str
    modality_source: Literal["optical", "sar", "combined"] = "combined"
    box: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized 0-100%")
    confidence: float
    color: str = "#10b981"
    radar_db_signature: Optional[str] = None
    optical_spectral_signature: Optional[str] = None


class OpticalSARFusionResult(BaseModel):
    task: str = "Cross-Modal Optical-SAR Analysis"
    query: str
    answer: str
    confidence: float
    confidence_formatted: str
    optical_evidence: List[ModalityAttributedEvidence] = Field(default_factory=list)
    sar_evidence: List[ModalityAttributedEvidence] = Field(default_factory=list)
    combined_evidence: List[ModalityAttributedEvidence] = Field(default_factory=list)
    grounding_regions: List[OpticalSARGroundingRegion] = Field(default_factory=list)
    cloud_penetrated_area_km2: float = 0.0
    radar_confirmed_structures_count: int = 0
    fused_composite_url: Optional[str] = None
    sar_preview_url: Optional[str] = None
    optical_preview_url: Optional[str] = None
    legend: Dict[str, str] = Field(default_factory=dict)
    models_used: List[str] = Field(default_factory=list)
    model_status: str = "READY"
    status_message: str = "Optical-SAR cross-modal analysis completed."
    inference_time_ms: int = 0
    created_at: str = ""


class RemoteSensingOpticalSARModel(ABC):
    """Abstract interface for Optical + SAR multimodal fusion models."""

    @abstractmethod
    def fuse_and_analyze(
        self,
        optical_input: Any,
        sar_input: Any,
        query: str = "Analyze the optical and SAR images together.",
        parameters: Optional[Dict[str, Any]] = None
    ) -> OpticalSARFusionResult:
        """Executes cross-modal joint reasoning and returns modality-attributed evidence."""
        pass
