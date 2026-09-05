"""
Copernicus Data Space Ecosystem & Sentinel Hub Service Module for SatQuery AI.
"""

from .types import (
    SatelliteCollection,
    SpectralIndexType,
    AOIGeometry,
    GeocodeResult,
    CopernicusObservation,
    CopernicusSearchRequest,
    CopernicusSearchResponse,
    ProcessAnalysisRequest,
    ProcessAnalysisResponse,
    SpectralStatistics,
    TemporalChangeRequest,
    TemporalChangeResponse,
)
from .auth import CopernicusAuthService, copernicus_auth
from .geocoding import LocationGeocodingService
from .catalog import CopernicusCatalogService
from .processing import SentinelHubProcessingService, copernicus_processing
from .change_detection import CopernicusChangeService, copernicus_change

__all__ = [
    "SatelliteCollection",
    "SpectralIndexType",
    "AOIGeometry",
    "GeocodeResult",
    "CopernicusObservation",
    "CopernicusSearchRequest",
    "CopernicusSearchResponse",
    "ProcessAnalysisRequest",
    "ProcessAnalysisResponse",
    "SpectralStatistics",
    "TemporalChangeRequest",
    "TemporalChangeResponse",
    "CopernicusAuthService",
    "copernicus_auth",
    "LocationGeocodingService",
    "CopernicusCatalogService",
    "SentinelHubProcessingService",
    "copernicus_processing",
    "CopernicusChangeService",
    "copernicus_change",
]
