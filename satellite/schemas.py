"""
Schemas and data structures for Earth Observation Satellite Data Retrieval.
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SatelliteSensor(str, Enum):
    SENTINEL_2_OPTICAL = "Sentinel-2 MSI"
    SENTINEL_1_SAR = "Sentinel-1 SAR C-Band"
    LANDSAT_8 = "Landsat-8 OLI"
    CARTOSAT_3 = "Cartosat-3 High-Res"
    RISAT_1 = "RISAT-1 C-Band SAR"


class SatelliteProductType(str, Enum):
    S2MSI2A = "S2MSI2A (Bottom-of-Atmosphere L2A)"
    S2MSI1C = "S2MSI1C (Top-of-Atmosphere L1C)"
    S1_GRD = "Sentinel-1 GRD (Ground Range Detected)"
    S1_SLC = "Sentinel-1 SLC (Single Look Complex)"


class SatelliteProduct(BaseModel):
    """Represents a remote-sensing scene product in the Earth-Observation catalog."""
    id: str = Field(..., description="Unique product identifier (e.g. S2A_MSIL2A_20240115T050931_N0510_R019_T44VLR)")
    title: str = Field(..., description="Human-readable scene title")
    sensor: SatelliteSensor = Field(default=SatelliteSensor.SENTINEL_2_OPTICAL)
    product_type: str = Field(default="S2MSI2A")
    acquisition_date: str = Field(..., description="ISO 8601 acquisition timestamp")
    cloud_coverage_percentage: float = Field(default=0.0, description="Cloud cover percentage (0-100)")
    aoi_overlap_percentage: float = Field(default=100.0, description="Spatial overlap percentage with requested AOI")
    bbox: List[float] = Field(default_factory=list, description="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    preview_url: str = Field(default="", description="URL or Data URI for visual thumbnail")
    download_url: Optional[str] = Field(default=None, description="Direct download endpoint")
    file_size_bytes: int = Field(default=0)
    file_size_formatted: str = Field(default="0 MB")
    selection_score: float = Field(default=0.0, description="Multi-criteria selection score (0-1.0)")
    is_selected: bool = Field(default=False, description="Whether selected as optimal candidate")
    selection_reason: Optional[str] = Field(default=None, description="Explanation for automated selection")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CatalogSearchQuery(BaseModel):
    """Query parameters for catalog search."""
    location_name: Optional[str] = Field(default="Chennai, India")
    bbox: Optional[List[float]] = Field(default=None, description="[min_lon, min_lat, max_lon, max_lat]")
    start_date: str = Field(default="2024-01-01")
    end_date: str = Field(default="2024-01-31")
    sensor: SatelliteSensor = Field(default=SatelliteSensor.SENTINEL_2_OPTICAL)
    max_cloud_coverage: float = Field(default=20.0, description="Maximum cloud coverage threshold (0-100)")
    product_type: Optional[str] = Field(default=None)
    limit: int = Field(default=10)


class CatalogSearchResult(BaseModel):
    """Results returned from satellite catalog search."""
    total_found: int
    products: List[SatelliteProduct]
    search_query: CatalogSearchQuery
    provider_name: str
    execution_time_ms: int
    status_message: Optional[str] = None


class AutoLocationQueryRequest(BaseModel):
    """Natural-language query for automatic Earth-observation retrieval and analysis."""
    query: str = Field(..., description="e.g. 'Compare Chennai between January 2024 and January 2026'")
    target_location: Optional[str] = Field(default=None)
    before_date: Optional[str] = Field(default=None)
    after_date: Optional[str] = Field(default=None)
    sensor: Optional[SatelliteSensor] = Field(default=SatelliteSensor.SENTINEL_2_OPTICAL)
    max_cloud_coverage: float = Field(default=15.0)


class AutoLocationQueryResponse(BaseModel):
    """Full execution result from automatic Earth observation retrieval workflow."""
    workflow_id: str
    location_name: str
    bbox: List[float]
    sensor: str
    before_scene: Optional[SatelliteProduct] = None
    after_scene: Optional[SatelliteProduct] = None
    task: str
    answer: str
    confidence: float
    confidence_formatted: str
    execution_trace: List[Dict[str, Any]]
    execution_time_ms: int
    status: str = "COMPLETED"
    provider_status: str = "COPERNICUS_CDSE_ACTIVE"
