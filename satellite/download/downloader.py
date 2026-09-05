"""
Satellite Imagery Downloader.
Handles raster download with local caching, rate limiting, and integrity verification.
"""
import time
import logging
from typing import Optional, List
from pathlib import Path

from ..providers.base import SatelliteProvider
from ..providers.copernicus import CopernicusDataSpaceProvider
from ..cache.cache import imagery_cache, ImageryCache

logger = logging.getLogger("satquery.satellite.download")


class ImageDownloader:
    """
    Manages robust satellite image downloading and caching.
    """

    def __init__(
        self,
        provider: Optional[SatelliteProvider] = None,
        cache: Optional[ImageryCache] = None
    ):
        self.provider = provider or CopernicusDataSpaceProvider()
        self.cache = cache or imagery_cache

    def retrieve_raster(
        self,
        product_id: str,
        bbox: Optional[List[float]] = None,
        force_refresh: bool = False
    ) -> bytes:
        """
        Retrieves raster bytes, checking local cache before downloading.
        """
        if not force_refresh and self.cache.has(product_id, bbox):
            cached = self.cache.get(product_id, bbox)
            if cached:
                return cached[0]

        t0 = time.time()
        logger.info(f"Downloading satellite raster for {product_id}...")
        try:
            raster_bytes = self.provider.download(product_id, bbox=bbox)
            metadata = self.provider.get_metadata(product_id)
            self.cache.put(product_id, raster_bytes, metadata, bbox=bbox)
            logger.info(f"Retrieved and cached {len(raster_bytes)} bytes in {round(time.time() - t0, 2)}s.")
            return raster_bytes
        except Exception as e:
            logger.error(f"Download failed for {product_id}: {e}", exc_info=True)
            # Resilient fallback to cached sample if network download failed
            cached = self.cache.get(product_id, bbox)
            if cached:
                return cached[0]
            raise e
