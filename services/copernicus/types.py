"""
Pydantic Data Models and Types for Copernicus Data Space & Sentinel Hub Analysis Module.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class SatelliteCollection(str, Enum):
    SENTINEL_2_L2A = "sentinel-2-l2a"
    SENTINEL_1_GRD = "sentinel-1-grd"
    LANDSAT_8_L2 = "landsat-8-l2"


class SpectralIndexType(str, Enum):
    TRUE_COLOR = "TRUE_COLOR"
    FALSE_COLOR_NIR = "FALSE_COLOR_NIR"
    NDVI = "NDVI"
    NDWI = "NDWI"
    NDBI = "NDBI"


class AOIGeometry(BaseModel):
    """GeoJSON Polygon Geometry or Bounding Box representation."""
    type: str = Field(default="Polygon", description="GeoJSON geometry type")
    coordinates: List[List[List[float]]] = Field(
        ...,
        description="GeoJSON coordinates array [[[lon, lat], [lon, lat], ...]]"
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat]"
    )


class GeocodeResult(BaseModel):
    """Geocoded location result with coordinates and bounding box."""
    name: str
    display_name: str
    lat: float
    lon: float
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geojson: Optional[Dict[str, Any]] = None


class CopernicusObservation(BaseModel):
    """Satellite scene observation record discovered in Copernicus Catalog."""
    id: str = Field(..., description="Unique scene product ID")
    acquisition_date: str = Field(..., description="ISO 8601 acquisition timestamp")
    collection: str = Field(default="sentinel-2-l2a")
    cloud_coverage_percent: float = Field(default=0.0)
    platform: str = Field(default="Sentinel-2B")
    tile_id: Optional[str] = Field(default=None)
    bbox: List[float] = Field(default_factory=list, description="[min_lon, min_lat, max_lon, max_lat]")
    preview_url: Optional[str] = Field(default=None)
    available_bands: List[str] = Field(default_factory=list)
    resolution_m: float = Field(default=10.0)


class CopernicusSearchRequest(BaseModel):
    """Parameters for searching satellite scenes."""
    aoi: AOIGeometry
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    max_cloud_coverage: float = Field(default=20.0, ge=0.0, le=100.0)
    collection: SatelliteCollection = Field(default=SatelliteCollection.SENTINEL_2_L2A)
    limit: int = Field(default=10, ge=1, le=50)


class CopernicusSearchResponse(BaseModel):
    """Satellite catalog search results."""
    total_count: int
    observations: List[CopernicusObservation]
    aoi_area_sq_km: float
    collection: str
    search_time_ms: int
    status_message: str


class SpectralStatistics(BaseModel):
    """Spatial statistics computed from raster index pixel values."""
    mean: float
    min: float
    max: float
    std_dev: float
    valid_pixel_count: int
    class_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="e.g. {'Healthy Vegetation (>0.5)': 72.4, 'Moderate (0.2-0.5)': 18.6, 'Low/Barren (<0.2)': 9.0}"
    )


class ProcessAnalysisRequest(BaseModel):
    """Request to process satellite imagery for an AOI and observation."""
    aoi: AOIGeometry
    observation_id: str
    analysis_type: SpectralIndexType = Field(default=SpectralIndexType.NDVI)
    collection: SatelliteCollection = Field(default=SatelliteCollection.SENTINEL_2_L2A)
    resolution_m: float = Field(default=10.0)


class ProcessAnalysisResponse(BaseModel):
    """Processed satellite visualization and calculated index statistics."""
    image_url: str = Field(..., description="Data URI (PNG) or URL of processed raster")
    analysis_type: SpectralIndexType
    observation_id: str
    acquisition_date: str
    aoi_area_sq_km: float
    resolution_m: float
    dimensions: List[int] = Field(default_factory=lambda: [1024, 1024])
    statistics: Optional[SpectralStatistics] = None
    legend: Dict[str, str] = Field(default_factory=dict)
    processing_time_ms: int
    is_real_data: bool = True
    status: str = "COMPLETED"


class TemporalChangeRequest(BaseModel):
    """Request to perform bi-temporal change detection over the same AOI."""
    aoi: AOIGeometry
    before_observation_id: str
    after_observation_id: str
    analysis_type: SpectralIndexType = Field(default=SpectralIndexType.NDVI)


class TemporalChangeResponse(BaseModel):
    """Bi-temporal change detection results and computed change distribution."""
    before_image_url: str
    after_image_url: str
    diff_image_url: str
    analysis_type: SpectralIndexType
    before_date: str
    after_date: str
    aoi_area_sq_km: float
    improved_percent: float
    stable_percent: float
    declined_percent: float
    mean_change: float
    change_statistics: Dict[str, Any]
    legend: Dict[str, str]
    processing_time_ms: int
    status: str = "COMPLETED"
