"""
Pydantic schemas for remote sensing imagery, formats, validation, and preprocessing.
"""
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class ModalityEnum(str, Enum):
    OPTICAL = "Optical"
    MULTISPECTRAL = "Multispectral"
    SAR = "SAR"
    CROSS_MODAL_PAIR = "Cross-Modal Pair (Optical + SAR)"
    BITEMPORAL_PAIR = "Bi-Temporal Pair"


class SensorEnum(str, Enum):
    SENTINEL_2 = "Sentinel-2"
    SENTINEL_1 = "Sentinel-1"
    CARTOSAT_2S = "Cartosat-2S"
    RISAT = "RISAT-1A"
    LANDSAT_8_9 = "Landsat 8/9"
    GENERIC_BENCHMARK = "Benchmark Dataset (VRSBench/RSVQA/CDVQA)"
    UNKNOWN = "Unknown / User Provided"


class ImageMetadata(BaseModel):
    id: str = Field(..., description="Unique identifier for uploaded image")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="Size in bytes")
    file_size_formatted: str = Field(..., description="Human readable size e.g. 4.2 MB")
    format: str = Field(..., description="File extension / format (GeoTIFF, TIFF, PNG, JPEG)")
    driver: Optional[str] = Field(default="GTiff", description="Rasterio driver e.g. GTiff, PNG, JPEG")
    dimensions: List[int] = Field(..., description="Width and Height in pixels [width, height]")
    bands: int = Field(default=3, description="Number of spectral/polarization bands")
    dtype: Optional[str] = Field(default="uint8", description="Data type of raster pixels (e.g. uint16, float32)")
    band_names: Optional[List[str]] = Field(default=None, description="Names of bands (e.g. Red, Green, Blue, NIR, VV, VH)")
    crs: Optional[str] = Field(default=None, description="Coordinate Reference System (e.g., EPSG:4326, EPSG:32644)")
    transform: Optional[List[float]] = Field(default=None, description="Affine georeferencing transform [a, b, c, d, e, f]")
    bounds: Optional[List[float]] = Field(default=None, description="[left, bottom, right, top] in native CRS")
    bounds_wgs84: Optional[List[float]] = Field(default=None, description="[min_lon, min_lat, max_lon, max_lat] in WGS84")
    resolution: Optional[List[float]] = Field(default=None, description="Pixel resolution [res_x, res_y]")
    resolution_m: Optional[float] = Field(default=None, description="Approximate Ground Sample Distance in meters")
    nodata: Optional[Union[float, int, str]] = Field(default=None, description="Nodata / fill pixel value")
    modality: ModalityEnum = Field(default=ModalityEnum.OPTICAL, description="Imagery modality")
    sensor: SensorEnum = Field(default=SensorEnum.UNKNOWN, description="Identified satellite sensor")
    raster_type: Optional[str] = Field(default=None, description="Detected raster type: OPTICAL RGB, OPTICAL MULTISPECTRAL, OPTICAL SINGLE-BAND, SAR")
    band_stats: Optional[List[Dict[str, float]]] = Field(default=None, description="Band-level statistics (min, max, mean, std)")
    type_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Detailed type & sensor metadata")
    spectral_indices: Optional[Dict[str, Any]] = Field(default=None, description="Precalculated spectral indices statistics")
    acquisition_date: Optional[str] = Field(default=None, description="Acquisition date (ISO string)")
    is_valid: bool = Field(default=True, description="Whether validation checks passed")
    validation_notes: List[str] = Field(default_factory=list, description="Diagnostic validation notes")
    preview_url: Optional[str] = Field(default=None, description="Data URL or preview URL")
    download_url: Optional[str] = Field(default=None, description="Direct download URL for generated/processed GeoTIFF")


class ImageUploadValidationRequest(BaseModel):
    filename: str
    file_size_bytes: int
    content_type: Optional[str] = None
    file_base64_sample: Optional[str] = None
    modality_hint: Optional[ModalityEnum] = None


class ImageUploadValidationResponse(BaseModel):
    is_valid: bool
    status_message: str
    metadata: ImageMetadata


class PairValidationRequest(BaseModel):
    image_a_metadata: ImageMetadata
    image_b_metadata: ImageMetadata
    expected_pair_type: Optional[str] = None


class PairValidationResponse(BaseModel):
    is_compatible: bool
    pair_type: str
    crs_compatible: bool
    has_geographic_overlap: bool
    overlap_percentage: float
    dimensions_match: bool
    modality_a: str
    modality_b: str
    is_optical_sar_pair: bool
    is_bitemporal_pair: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CompositePreviewRequest(BaseModel):
    image_id: str
    composite_type: str = Field(default="true_color", description="'true_color', 'false_color_nir', 'sar_db', 'custom'")
    bands: Optional[List[int]] = Field(default=None, description="Custom 3-band selection [b1, b2, b3] (1-indexed)")


class CompositePreviewResponse(BaseModel):
    image_id: str
    composite_type: str
    preview_url: str


class CropAOIRequest(BaseModel):
    image_id: str
    aoi_bounds: Optional[List[float]] = Field(default=None, description="[min_lon, min_lat, max_lon, max_lat]")
    aoi_polygon: Optional[List[List[float]]] = Field(default=None, description="List of [lon, lat] polygon coordinates")


class CropAOIResponse(BaseModel):
    image_id: str
    cropped_image_id: str
    metadata: Dict[str, Any]
    preview_url: str


class TilingRequest(BaseModel):
    image_id: str
    tile_size: int = 512
    overlap: int = 64


class TilingResponse(BaseModel):
    image_id: str
    tile_count: int
    tiles: List[Dict[str, Any]]
