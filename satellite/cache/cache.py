"""
Local Satellite Imagery Cache.
Deduplicates and stores retrieved satellite rasters and metadata locally by product ID.
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("satquery.satellite.cache")


class ImageryCache:
    """
    Disk cache for remote-sensing imagery products.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        base_dir = cache_dir or os.getenv("SATELLITE_CACHE_DIR", "data/satellite_cache")
        self.cache_dir = Path(base_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self._index_file.exists():
            try:
                with open(self._index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        try:
            with open(self._index_file, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save satellite cache index: {e}")

    def _get_cache_key(self, product_id: str, bbox: Optional[list] = None) -> str:
        raw = f"{product_id}_{bbox}" if bbox else product_id
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def has(self, product_id: str, bbox: Optional[list] = None) -> bool:
        """Checks if raster exists in local cache."""
        key = self._get_cache_key(product_id, bbox)
        if key in self._index:
            file_path = self.cache_dir / f"{key}.tif"
            return file_path.exists()
        return False

    def get(self, product_id: str, bbox: Optional[list] = None) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Retrieves cached raster bytes and metadata."""
        key = self._get_cache_key(product_id, bbox)
        if key in self._index:
            file_path = self.cache_dir / f"{key}.tif"
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                    meta = self._index[key]
                    logger.info(f"Loaded product {product_id} from local cache ({len(data)} bytes).")
                    return data, meta
                except Exception as e:
                    logger.warning(f"Error reading cached raster: {e}")
        return None

    def put(self, product_id: str, data: bytes, metadata: Dict[str, Any], bbox: Optional[list] = None) -> Path:
        """Stores raster bytes and metadata in local cache."""
        key = self._get_cache_key(product_id, bbox)
        file_path = self.cache_dir / f"{key}.tif"
        try:
            with open(file_path, "wb") as f:
                f.write(data)
            self._index[key] = {
                "product_id": product_id,
                "bbox": bbox,
                "size_bytes": len(data),
                "metadata": metadata
            }
            self._save_index()
            logger.info(f"Cached product {product_id} to {file_path}.")
            return file_path
        except Exception as e:
            logger.warning(f"Failed to cache raster bytes: {e}")
            return file_path

    def clear(self) -> None:
        """Clears local cache."""
        for p in self.cache_dir.glob("*.tif"):
            try:
                p.unlink()
            except Exception:
                pass
        self._index = {}
        self._save_index()


# Singleton imagery cache instance
imagery_cache = ImageryCache()
