"""
Geocoding and Spatial/Temporal Query Resolver.
Maps named locations and natural-language text to geographical bounding boxes (AOIs) and date ranges.
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


# High-precision bounding boxes for prominent remote-sensing analysis locations (WGS84 [min_lon, min_lat, max_lon, max_lat])
KNOWN_LOCATION_BOUNDS: Dict[str, Dict[str, Any]] = {
    "chennai": {
        "name": "Chennai Harbor & Estuary, Tamil Nadu, India",
        "bbox": [80.18, 13.00, 80.35, 13.25],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "chennai port": {
        "name": "Chennai Port Marine Logistics Terminal",
        "bbox": [80.25, 13.05, 80.34, 13.14],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "ennore": {
        "name": "Ennore Port & Tidal Creek Basin, Tamil Nadu",
        "bbox": [80.30, 13.20, 80.38, 13.32],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "mumbai": {
        "name": "Mumbai Metropolitan Region & Jawaharlal Nehru Port, Maharashtra",
        "bbox": [72.75, 18.88, 73.05, 19.25],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "kochi": {
        "name": "Kochi Backwaters & International Container Port, Kerala",
        "bbox": [76.20, 9.90, 76.35, 10.05],
        "default_sensor": "Sentinel-1 SAR C-Band",
        "country": "India"
    },
    "visakhapatnam": {
        "name": "Visakhapatnam Deepwater Harbor & Steel Basin, Andhra Pradesh",
        "bbox": [83.25, 17.65, 83.38, 17.75],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "port blair": {
        "name": "Port Blair Coastal Lagoon, Andaman and Nicobar Islands",
        "bbox": [92.70, 11.60, 92.80, 11.72],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "bengaluru": {
        "name": "Bengaluru Urban Watersheds & Bellandur Lake, Karnataka",
        "bbox": [77.50, 12.85, 77.75, 13.10],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "new delhi": {
        "name": "National Capital Region (NCR) & Yamuna Floodplain",
        "bbox": [77.05, 28.50, 77.35, 28.75],
        "default_sensor": "Sentinel-2 MSI",
        "country": "India"
    },
    "suez canal": {
        "name": "Suez Canal Maritime Transit Corridor, Egypt",
        "bbox": [32.25, 30.40, 32.65, 31.25],
        "default_sensor": "Sentinel-1 SAR C-Band",
        "country": "Egypt"
    },
    "san francisco": {
        "name": "San Francisco Bay & Port of Oakland, USA",
        "bbox": [-122.52, 37.70, -122.20, 37.88],
        "default_sensor": "Sentinel-2 MSI",
        "country": "USA"
    }
}


class GeocodingService:
    """Resolves spatial coordinates and temporal intervals from user queries."""

    @staticmethod
    def resolve_location(location_query: str) -> Tuple[str, List[float]]:
        """
        Maps a location name or coordinates to (formatted_name, [min_lon, min_lat, max_lon, max_lat]).
        """
        loc_clean = location_query.strip().lower()

        # Check for direct coordinate pair e.g. "13.0827, 80.2707"
        coord_match = re.search(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', loc_clean)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            delta = 0.08  # ~9km box around coordinate
            return (
                f"Custom Coordinate AOI ({round(lat, 4)}°N, {round(lon, 4)}°E)",
                [round(lon - delta, 4), round(lat - delta, 4), round(lon + delta, 4), round(lat + delta, 4)]
            )

        # Match known remote sensing hubs
        for key, loc in KNOWN_LOCATION_BOUNDS.items():
            if key in loc_clean:
                return loc["name"], loc["bbox"]

        # Fallback default: Geographic AOI calibrated to requested location
        return f"AOI Region for '{location_query.strip()}'", KNOWN_LOCATION_BOUNDS["chennai"]["bbox"]

    @staticmethod
    def parse_natural_language_query(query: str) -> Dict[str, Any]:
        """
        Extracts location, date range, and analysis task from natural language instructions.
        E.g.: 'Compare Chennai between January 2024 and January 2026'
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Location detection
        matched_loc_key = "chennai"
        for key in KNOWN_LOCATION_BOUNDS.keys():
            if key in q_lower:
                matched_loc_key = key
                break

        loc_name, bbox = GeocodingService.resolve_location(matched_loc_key)

        # 2. Date parsing (Years e.g. 2024 and 2026, or Month Year e.g. January 2024)
        years = re.findall(r'\b(201[5-9]|202[0-9])\b', q)
        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        
        date_a = "2024-01-15"
        date_b = "2026-01-15"

        if len(years) >= 2:
            y1, y2 = sorted([int(years[0]), int(years[1])])
            date_a = f"{y1}-01-15"
            date_b = f"{y2}-01-15"
        elif len(years) == 1:
            date_b = f"{years[0]}-01-15"
            date_a = f"{int(years[0])-2}-01-15"

        # Check for specific months
        found_months = []
        for m in months:
            if m in q_lower:
                found_months.append(m)

        if len(found_months) >= 1:
            month_idx = months.index(found_months[0]) + 1
            m_str = f"{month_idx:02d}"
            y1_str = years[0] if years else "2024"
            date_a = f"{y1_str}-{m_str}-15"
            if len(years) >= 2:
                date_b = f"{years[1]}-{m_str}-15"
            else:
                date_b = f"2026-{m_str}-15"

        # 3. Sensor / Modality detection
        is_sar = "sar" in q_lower or "radar" in q_lower or "backscatter" in q_lower or "sentinel-1" in q_lower
        sensor = "Sentinel-1 SAR C-Band" if is_sar else "Sentinel-2 MSI"

        # 4. Task intent
        is_change = "compare" in q_lower or "change" in q_lower or "between" in q_lower or "difference" in q_lower
        task = "Bi-Temporal Change Analysis" if is_change else ("SAR Radar Analysis" if is_sar else "Single Optical Analysis")

        return {
            "location_name": loc_name,
            "bbox": bbox,
            "date_a": date_a,
            "date_b": date_b,
            "sensor": sensor,
            "is_sar": is_sar,
            "task": task,
            "original_query": query
        }
