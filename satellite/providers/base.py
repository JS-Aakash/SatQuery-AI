"""
Abstract Base Class for Earth-Observation Satellite Providers.
Defines the uniform contract for catalog querying, preview fetching, and raster retrieval.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..schemas import SatelliteProduct, CatalogSearchQuery


class SatelliteProvider(ABC):
    """
    Abstract interface for satellite data providers (Copernicus CDSE, Sentinel Hub, USGS, ISRO Veda).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Whether valid API credentials and session are established."""
        pass

    @abstractmethod
    def search(self, query: CatalogSearchQuery) -> List[SatelliteProduct]:
        """Searches catalog for candidate scenes matching spatio-temporal constraints."""
        pass

    @abstractmethod
    def get_product(self, product_id: str) -> Optional[SatelliteProduct]:
        """Fetches detailed product record by ID."""
        pass

    @abstractmethod
    def download(self, product_id: str, bbox: Optional[List[float]] = None) -> bytes:
        """Retrieves raster bytes for the given product / AOI."""
        pass

    @abstractmethod
    def get_preview(self, product_id: str) -> str:
        """Retrieves thumbnail or high-fidelity visual rendering URL / Data URI."""
        pass

    @abstractmethod
    def get_metadata(self, product_id: str) -> Dict[str, Any]:
        """Retrieves technical remote-sensing telemetry metadata."""
        pass
