"""
Interfaces and abstract definitions for Bi-Temporal Remote-Sensing Change Intelligence.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np


class ChangePolygon(BaseModel):
    id: str
    label: str
    change_type: str = Field(..., description="e.g. Built-up Expansion, Canopy Loss, Water Contraction, Soil Exposure")
    area_m2: float
    area_hectares: float
    area_km2: float
    bounding_box: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized 0-100%")
    confidence: float
    color: str = "#f59e0b"
    polygon_coordinates: Optional[List[List[float]]] = None


class SpectralIndicesResult(BaseModel):
    mean_ndvi_t1: float
    mean_ndvi_t2: float
    delta_ndvi: float
    mean_ndwi_t1: float
    mean_ndwi_t2: float
    delta_ndwi: float
    mean_ndbi_t1: float
    mean_ndbi_t2: float
    delta_ndbi: float
    interpretation: str
    supporting_indicators: List[str]


class ChangeDetectionResult(BaseModel):
    task: str = "Bi-Temporal Change Analysis"
    query: str
    textual_description: str
    has_significant_change: bool
    change_percentage: float
    confidence: float
    confidence_formatted: str
    total_changed_area_km2: float
    total_changed_area_hectares: float
    changed_regions: List[ChangePolygon]
    spectral_indices: Optional[SpectralIndicesResult] = None
    change_mask_url: Optional[str] = None
    legend: Dict[str, str] = Field(default_factory=dict)
    models_used: List[str] = Field(default_factory=list)
    model_status: str = "READY"
    status_message: str = "Bi-temporal change pipeline completed successfully."
    inference_time_ms: int = 0
    created_at: str = ""


class RemoteSensingChangeDetector(ABC):
    """Abstract interface for remote-sensing change understanding and difference models."""

    @abstractmethod
    def detect_change(
        self,
        image_t1: Any,
        image_t2: Any,
        query: str = "What changed between these two images?",
        parameters: Optional[Dict[str, Any]] = None
    ) -> ChangeDetectionResult:
        """Analyzes two co-registered or aligned images and returns structured change findings."""
        pass
