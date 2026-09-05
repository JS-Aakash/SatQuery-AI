"""
SatQuery AI - Preprocessing Package.
Exposes modular geospatial image processing services for remote sensing rasters.
"""
from .reader import GeoTIFFReader, RasterReadError
from .metadata import RasterMetadataService
from .validator import ImageValidator
from .crs import CRSService
from .band_service import BandService
from .sar import SARPreprocessor
from .alignment import ImageAlignmentService
from .aoi import AOIService
from .tiling import TileService

__all__ = [
    "GeoTIFFReader",
    "RasterReadError",
    "RasterMetadataService",
    "ImageValidator",
    "CRSService",
    "BandService",
    "SARPreprocessor",
    "ImageAlignmentService",
    "AOIService",
    "TileService",
]
