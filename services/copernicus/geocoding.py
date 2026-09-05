"""
Location Geocoding and AOI Area Calculation Service.
Queries OpenStreetMap Nominatim for named places and calculates geographic polygon area.
"""

import math
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple

from .types import GeocodeResult, AOIGeometry

logger = logging.getLogger("satquery.copernicus.geocoding")

# Prominent Indian remote sensing & industrial hubs fallback coordinates
PREDEFINED_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "perundurai": {
        "name": "Perundurai, Erode District, Tamil Nadu, India",
        "display_name": "Perundurai SIPCOT Industrial Hub, Tamil Nadu, India",
        "lat": 11.2750,
        "lon": 77.5833,
        "bbox": [77.5300, 11.2300, 77.6350, 11.3200]
    },
    "erode": {
        "name": "Erode, Tamil Nadu, India",
        "display_name": "Erode Smart City & Cauvery Basin, Tamil Nadu, India",
        "lat": 11.3410,
        "lon": 77.7172,
        "bbox": [77.6600, 11.2900, 77.7700, 11.3900]
    },
    "coimbatore": {
        "name": "Coimbatore, Tamil Nadu, India",
        "display_name": "Coimbatore Metropolitan Region & Western Ghats Foothills, India",
        "lat": 11.0168,
        "lon": 76.9558,
        "bbox": [76.8800, 10.9500, 77.0400, 11.0800]
    },
    "chennai": {
        "name": "Chennai, Tamil Nadu, India",
        "display_name": "Chennai Port & Coastal Watershed, Tamil Nadu, India",
        "lat": 13.0827,
        "lon": 80.2707,
        "bbox": [80.1800, 13.0000, 80.3500, 13.2500]
    },
    "mumbai": {
        "name": "Mumbai, Maharashtra, India",
        "display_name": "Mumbai Metropolitan Region & Jawaharlal Nehru Port, India",
        "lat": 19.0760,
        "lon": 72.8777,
        "bbox": [72.7500, 18.8800, 73.0500, 19.2500]
    },
    "bengaluru": {
        "name": "Bengaluru, Karnataka, India",
        "display_name": "Bengaluru Urban Watershed & Tech Corridor, Karnataka, India",
        "lat": 12.9716,
        "lon": 77.5946,
        "bbox": [77.5000, 12.8500, 77.7500, 13.1000]
    }
}


class LocationGeocodingService:
    """
    Geocodes text queries into geographic coordinates and bounding boxes.
    """

    @staticmethod
    def geocode(query: str) -> Optional[GeocodeResult]:
        """
        Geocodes a place name (e.g. 'Perundurai', 'Coimbatore') using Nominatim with instant predefined fallback.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Check known local presets first
        for key, info in PREDEFINED_LOCATIONS.items():
            if key in q_lower:
                return GeocodeResult(
                    name=info["name"],
                    display_name=info["display_name"],
                    lat=info["lat"],
                    lon=info["lon"],
                    bbox=info["bbox"]
                )

        # 2. Check for direct coordinate input (e.g., "11.275, 77.583")
        import re
        coord_match = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$', q_clean)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            d = 0.04
            return GeocodeResult(
                name=f"Coordinates ({lat:.4f}°N, {lon:.4f}°E)",
                display_name=f"Custom AOI centered at {lat:.4f}°N, {lon:.4f}°E",
                lat=lat,
                lon=lon,
                bbox=[round(lon - d, 4), round(lat - d, 4), round(lon + d, 4), round(lat + d, 4)]
            )

        # 3. Query OpenStreetMap Nominatim
        try:
            encoded = urllib.parse.quote(q_clean)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1&addressdetails=1"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SatQuery-AI-Platform/1.0 (remote-sensing-analysis)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and len(data) > 0:
                    item = data[0]
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    raw_bbox = item.get("boundingbox", [])
                    if len(raw_bbox) == 4:
                        # Nominatim returns [min_lat, max_lat, min_lon, max_lon]
                        bbox = [
                            round(float(raw_bbox[2]), 4),
                            round(float(raw_bbox[0]), 4),
                            round(float(raw_bbox[3]), 4),
                            round(float(raw_bbox[1]), 4)
                        ]
                    else:
                        d = 0.04
                        bbox = [round(lon - d, 4), round(lat - d, 4), round(lon + d, 4), round(lat + d, 4)]

                    return GeocodeResult(
                        name=item.get("name") or q_clean,
                        display_name=item.get("display_name") or q_clean,
                        lat=lat,
                        lon=lon,
                        bbox=bbox
                    )
        except Exception as e:
            logger.warning(f"Nominatim geocoding failed for '{query}': {e}")

        # 4. Fallback default if completely unknown
        d = 0.04
        return GeocodeResult(
            name=q_clean,
            display_name=f"Region for '{q_clean}'",
            lat=11.2750,
            lon=77.5833,
            bbox=[77.5400, 11.2350, 77.6250, 11.3150]
        )

    @staticmethod
    def calculate_aoi_area_sq_km(aoi: AOIGeometry) -> float:
        """
        Calculates area in square kilometers for a GeoJSON Polygon using spherical excess.
        """
        if not aoi or not aoi.coordinates or len(aoi.coordinates) == 0:
            return 0.0

        ring = aoi.coordinates[0]
        if len(ring) < 3:
            return 0.0

        R = 6371.0  # Earth radius in km
        pts = list(ring)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        total_rad = 0.0
        for i in range(len(pts) - 1):
            lon1, lat1 = math.radians(pts[i][0]), math.radians(pts[i][1])
            lon2, lat2 = math.radians(pts[i+1][0]), math.radians(pts[i+1][1])
            total_rad += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))

        area = abs(total_rad * (R * R) / 2.0)
        return round(area, 2)

    @staticmethod
    def bbox_from_aoi(aoi: AOIGeometry) -> List[float]:
        """Extracts [min_lon, min_lat, max_lon, max_lat] from AOIGeometry."""
        if aoi.bbox and len(aoi.bbox) == 4:
            return aoi.bbox
        if aoi.coordinates and len(aoi.coordinates) > 0 and len(aoi.coordinates[0]) > 0:
            lons = [pt[0] for pt in aoi.coordinates[0]]
            lats = [pt[1] for pt in aoi.coordinates[0]]
            return [min(lons), min(lats), max(lons), max(lats)]
        return [77.5300, 11.2300, 77.6350, 11.3200]
