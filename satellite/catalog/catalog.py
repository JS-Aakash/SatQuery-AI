"""
Catalog Search Engine for Earth-Observation Data.
Coordinates search across providers and aggregates candidates with metadata filtering.
"""
import time
from typing import List, Optional
from ..schemas import CatalogSearchQuery, CatalogSearchResult, SatelliteProduct
from ..providers.base import SatelliteProvider
from ..providers.copernicus import CopernicusDataSpaceProvider


class CatalogSearchEngine:
    """
    Unified search engine across satellite providers.
    """

    def __init__(self, provider: Optional[SatelliteProvider] = None):
        self.provider = provider or CopernicusDataSpaceProvider()

    def search(self, query: CatalogSearchQuery) -> CatalogSearchResult:
        """
        Executes query against satellite catalog.
        """
        t0 = time.time()
        products = self.provider.search(query)
        duration_ms = int((time.time() - t0) * 1000.0)

        return CatalogSearchResult(
            total_found=len(products),
            products=products,
            search_query=query,
            provider_name=self.provider.name,
            execution_time_ms=duration_ms,
            status_message=f"Found {len(products)} candidate scene(s) matching spatial AOI and cloud constraints."
        )
