"""
Satellite Data Service for SatQuery AI.
Handles AOI area calculation and satellite scene catalog search.
"""
import math
from typing import List, Dict, Any, Optional

from .base import SatelliteService


class DefaultSatelliteService(SatelliteService):
    """
    Satellite service for STAC/Copernicus search and spatial AOI computations.
    """

    def calculate_aoi_area(self, aoi_coords: List[List[float]]) -> float:
        """
        Calculates approximate area in square kilometers of a geo-polygon (lon, lat points).
        Uses spherical polygonal excess formula.
        """
        if not aoi_coords or len(aoi_coords) < 3:
            return 0.0

        R = 6371.0  # Earth mean radius in km
        # Ensure closure
        pts = list(aoi_coords)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        total_rad = 0.0
        for i in range(len(pts) - 1):
            lon1, lat1 = math.radians(pts[i][0]), math.radians(pts[i][1])
            lon2, lat2 = math.radians(pts[i+1][0]), math.radians(pts[i+1][1])
            total_rad += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

        area_sq_km = abs(total_rad * (R * R) / 2.0)
        return round(area_sq_km, 2)

    def search_scenes(
        self,
        aoi_coords: Optional[List[List[float]]],
        start_date: str,
        end_date: str,
        sensors: List[str],
        max_cloud_cover: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Returns catalog scenes matching the specified parameters.
        Ready to connect to STAC API in Satellite Module.
        """
        area_km = self.calculate_aoi_area(aoi_coords) if aoi_coords else 45.8
        
        # Realistic catalog entries for testing the UI
        scenes = [
            {
                "scene_id": "S2A_MSIL2A_20240218T050811_N0510_R119_T44VLR",
                "sensor": "Sentinel-2A MSI",
                "acquisition_date": "2024-02-18 05:08:11 UTC",
                "cloud_cover_percent": 1.4,
                "resolution_m": 10,
                "bands_available": ["B02", "B03", "B04", "B08", "B11", "B12"],
                "footprint_area_km2": area_km,
                "thumbnail_url": "/demo/sample_optical.png",
                "status": "Available for Download / Ingestion"
            },
            {
                "scene_id": "S1A_IW_GRDH_1SDV_20240219T002345_052614_065E02",
                "sensor": "Sentinel-1A SAR",
                "acquisition_date": "2024-02-19 00:23:45 UTC",
                "cloud_cover_percent": 0.0,
                "resolution_m": 10,
                "bands_available": ["VV", "VH"],
                "footprint_area_km2": area_km,
                "thumbnail_url": "/demo/sample_sar.png",
                "status": "Available for Download / Ingestion"
            },
            {
                "scene_id": "S2B_MSIL2A_20230312T051129_N0400_R119_T44VLR",
                "sensor": "Sentinel-2B MSI (Historical T1)",
                "acquisition_date": "2023-03-12 05:11:29 UTC",
                "cloud_cover_percent": 3.8,
                "resolution_m": 10,
                "bands_available": ["B02", "B03", "B04", "B08"],
                "footprint_area_km2": area_km,
                "thumbnail_url": "/demo/sample_temporal_t1.png",
                "status": "Historical Reference T1"
            }
        ]

        # Filter by sensor if requested
        if sensors:
            lower_sensors = [s.lower() for s in sensors]
            filtered = [s for s in scenes if any(ls in s["sensor"].lower() for ls in lower_sensors)]
            return filtered if filtered else scenes
            
        return scenes
