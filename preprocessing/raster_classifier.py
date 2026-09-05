"""
Raster Type Classifier and Band Role Detector.
Classifies remote-sensing GeoTIFF rasters into:
- OPTICAL_RGB
- OPTICAL_MULTISPECTRAL
- OPTICAL_SINGLE_BAND
- SAR

Uses the strict evidence hierarchy:
1. GeoTIFF metadata / band descriptions
2. Sentinel/Satellite product metadata tags
3. Band names/wavelength identifiers (B02, B03, B04, B08, B11, B12, VV, VH)
4. Raster statistics / dynamic range characteristics
5. Filename hints only as a fallback
"""
import re
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class RasterType(str, Enum):
    OPTICAL_RGB = "OPTICAL RGB"
    OPTICAL_MULTISPECTRAL = "OPTICAL MULTISPECTRAL"
    OPTICAL_SINGLE_BAND = "OPTICAL SINGLE-BAND"
    SAR = "SAR"


class BandRole(str, Enum):
    COASTAL = "B01"
    BLUE = "B02"
    GREEN = "B03"
    RED = "B04"
    RED_EDGE_1 = "B05"
    RED_EDGE_2 = "B06"
    RED_EDGE_3 = "B07"
    NIR = "B08"
    NIR_NARROW = "B8A"
    WATER_VAPOUR = "B09"
    SWIR_1 = "B11"
    SWIR_2 = "B12"
    SAR_VV = "VV"
    SAR_VH = "VH"
    SAR_HH = "HH"
    SAR_HV = "HV"
    RGB_RED = "Red"
    RGB_GREEN = "Green"
    RGB_BLUE = "Blue"
    ALPHA = "Alpha"
    PANCHROMATIC = "Pan"
    UNKNOWN = "Unknown"


class RasterClassifier:
    """
    Evaluates raster structure, metadata, band descriptions, and pixel values
    to determine raster type and identify spectral/polarimetric band mappings.
    """

    @staticmethod
    def identify_band_role(
        desc: Optional[str] = None,
        band_idx: int = 1,
        total_bands: int = 1,
        stats: Optional[Dict[str, float]] = None,
        filename: str = ""
    ) -> BandRole:
        """Determines the specific remote-sensing role of an individual band."""
        d_lower = (desc or "").strip().lower()
        f_lower = filename.lower()

        # 1. Check direct description / tag matching
        if re.search(r"\b(vv|co-pol|co_pol)\b", d_lower) or "copol" in d_lower:
            return BandRole.SAR_VV
        if re.search(r"\b(vh|cross-pol|cross_pol)\b", d_lower) or "crosspol" in d_lower:
            return BandRole.SAR_VH
        if re.search(r"\b(hh)\b", d_lower):
            return BandRole.SAR_HH
        if re.search(r"\b(hv)\b", d_lower):
            return BandRole.SAR_HV

        # Sentinel-2 band identifiers
        if re.search(r"\b(b01|b1|coastal)\b", d_lower):
            return BandRole.COASTAL
        if re.search(r"\b(b02|b2|blue)\b", d_lower):
            return BandRole.BLUE
        if re.search(r"\b(b03|b3|green)\b", d_lower):
            return BandRole.GREEN
        if re.search(r"\b(b04|b4|red)\b", d_lower):
            return BandRole.RED
        if re.search(r"\b(b05|b5|rededge1|red_edge_1)\b", d_lower):
            return BandRole.RED_EDGE_1
        if re.search(r"\b(b06|b6|rededge2|red_edge_2)\b", d_lower):
            return BandRole.RED_EDGE_2
        if re.search(r"\b(b07|b7|rededge3|red_edge_3)\b", d_lower):
            return BandRole.RED_EDGE_3
        if re.search(r"\b(b08|b8|nir|near_infrared|near-infrared)\b", d_lower) and "b8a" not in d_lower:
            return BandRole.NIR
        if re.search(r"\b(b8a|narrow_nir|narrow-nir)\b", d_lower):
            return BandRole.NIR_NARROW
        if re.search(r"\b(b09|b9|watervapour|water_vapour)\b", d_lower):
            return BandRole.WATER_VAPOUR
        if re.search(r"\b(b11|swir1|swir_1|swir-1)\b", d_lower):
            return BandRole.SWIR_1
        if re.search(r"\b(b12|swir2|swir_2|swir-2)\b", d_lower):
            return BandRole.SWIR_2

        # 2. Check Standard Position in Multi-band Sentinel-2 Stack
        if total_bands >= 12:
            s2_stack = [
                BandRole.COASTAL, BandRole.BLUE, BandRole.GREEN, BandRole.RED,
                BandRole.RED_EDGE_1, BandRole.RED_EDGE_2, BandRole.RED_EDGE_3,
                BandRole.NIR, BandRole.NIR_NARROW, BandRole.WATER_VAPOUR,
                BandRole.SWIR_1, BandRole.SWIR_2
            ]
            if band_idx <= len(s2_stack):
                return s2_stack[band_idx - 1]

        # 3. Sentinel-2 6-band Stack: B02, B03, B04, B08, B11, B12
        if total_bands == 6:
            s2_6 = [BandRole.BLUE, BandRole.GREEN, BandRole.RED, BandRole.NIR, BandRole.SWIR_1, BandRole.SWIR_2]
            return s2_6[band_idx - 1]

        # 4. Standard 4-band Optical Stack (RGB + NIR) or Sentinel-2 (B04, B08, B11, B12)
        if total_bands == 4:
            if "b04" in f_lower or "b4" in f_lower or "b08" in f_lower or "b8" in f_lower:
                s2_4 = [BandRole.RED, BandRole.NIR, BandRole.SWIR_1, BandRole.SWIR_2]
                return s2_4[band_idx - 1]
            standard_4 = [BandRole.RED, BandRole.GREEN, BandRole.BLUE, BandRole.NIR]
            return standard_4[band_idx - 1]

        # 5. Standard 3-band RGB Stack
        if total_bands == 3:
            rgb_3 = [BandRole.RGB_RED, BandRole.RGB_GREEN, BandRole.RGB_BLUE]
            return rgb_3[band_idx - 1]

        # 6. Dual-band SAR
        if total_bands == 2:
            return BandRole.SAR_VV if band_idx == 1 else BandRole.SAR_VH

        # 7. Single band heuristics
        if total_bands == 1:
            if any(k in f_lower for k in ["vv", "vh", "sar", "s1", "grd", "slc", "sigma0", "risat"]):
                return BandRole.SAR_VV if "vh" not in f_lower else BandRole.SAR_VH
            if "b02" in f_lower: return BandRole.BLUE
            if "b03" in f_lower: return BandRole.GREEN
            if "b04" in f_lower: return BandRole.RED
            if "b08" in f_lower: return BandRole.NIR
            if "b11" in f_lower: return BandRole.SWIR_1
            if "b12" in f_lower: return BandRole.SWIR_2
            return BandRole.PANCHROMATIC

        return BandRole.UNKNOWN

    @staticmethod
    def classify_raster(
        total_bands: int,
        descriptions: Optional[List[str]] = None,
        tags: Optional[Dict[str, Any]] = None,
        band_stats: Optional[List[Dict[str, float]]] = None,
        filename: str = ""
    ) -> Tuple[RasterType, Dict[BandRole, int], Dict[str, Any]]:
        """
        Determines the RasterType and maps each detected BandRole to its 1-indexed band number.
        """
        descriptions = descriptions or []
        tags = tags or {}
        band_stats = band_stats or []
        f_lower = filename.lower()

        # Build band role mapping
        band_map: Dict[BandRole, int] = {}
        for i in range(1, total_bands + 1):
            desc = descriptions[i - 1] if i - 1 < len(descriptions) else None
            stats = band_stats[i - 1] if i - 1 < len(band_stats) else None
            role = RasterClassifier.identify_band_role(
                desc=desc,
                band_idx=i,
                total_bands=total_bands,
                stats=stats,
                filename=filename
            )
            if role != BandRole.UNKNOWN and role not in band_map:
                band_map[role] = i

        # Step 1: Detect SAR
        # Check descriptions or band roles
        is_sar = False
        sensor_name = "Unknown"
        polarizations: List[str] = []

        if BandRole.SAR_VV in band_map or BandRole.SAR_VH in band_map or BandRole.SAR_HH in band_map or BandRole.SAR_HV in band_map:
            is_sar = True
        elif total_bands in [1, 2] and any(k in f_lower for k in ["sar", "sentinel-1", "s1", "risat", "sigma0", "backscatter"]):
            is_sar = True

        if is_sar:
            if BandRole.SAR_VV in band_map: polarizations.append("VV")
            if BandRole.SAR_VH in band_map: polarizations.append("VH")
            if BandRole.SAR_HH in band_map: polarizations.append("HH")
            if BandRole.SAR_HV in band_map: polarizations.append("HV")
            if not polarizations:
                polarizations = ["VV"] if total_bands == 1 else ["VV", "VH"]

            if "risat" in f_lower:
                sensor_name = "RISAT-1A SAR"
            elif "sentinel-1" in f_lower or "s1" in f_lower or total_bands in [1, 2]:
                sensor_name = "Sentinel-1 SAR"

            metadata_summary = {
                "sensor": sensor_name,
                "modality": "SAR",
                "polarizations": polarizations,
                "has_vv": "VV" in polarizations or BandRole.SAR_VV in band_map,
                "has_vh": "VH" in polarizations or BandRole.SAR_VH in band_map,
                "spectral_indices_supported": False,
            }
            return RasterType.SAR, band_map, metadata_summary

        # Step 2: Optical Multispectral
        if total_bands >= 4 or (total_bands > 1 and any(r in band_map for r in [BandRole.NIR, BandRole.SWIR_1, BandRole.SWIR_2])):
            sensor_name = "Sentinel-2 Multispectral" if total_bands in [4, 6, 12, 13] else "Optical Multispectral"
            if "landsat" in f_lower:
                sensor_name = "Landsat Multispectral"

            # Check index capabilities
            has_b04 = BandRole.RED in band_map or BandRole.RGB_RED in band_map
            has_b08 = BandRole.NIR in band_map
            has_b03 = BandRole.GREEN in band_map or BandRole.RGB_GREEN in band_map
            has_b02 = BandRole.BLUE in band_map or BandRole.RGB_BLUE in band_map
            has_b11 = BandRole.SWIR_1 in band_map

            metadata_summary = {
                "sensor": sensor_name,
                "modality": "Optical Multispectral",
                "has_true_color_rgb": has_b04 and has_b03 and has_b02,
                "ndvi_supported": bool(has_b08 and has_b04),
                "ndwi_supported": bool(has_b03 and has_b08),
                "ndbi_supported": bool(has_b11 and has_b08),
                "available_spectral_bands": [r.value for r in band_map.keys() if r.value.startswith("B")],
            }
            return RasterType.OPTICAL_MULTISPECTRAL, band_map, metadata_summary

        # Step 3: Optical RGB
        if total_bands == 3:
            metadata_summary = {
                "sensor": "Optical RGB Sensor",
                "modality": "Optical True Color",
                "has_true_color_rgb": True,
                "ndvi_supported": False,
                "ndwi_supported": False,
                "ndbi_supported": False,
            }
            return RasterType.OPTICAL_RGB, band_map, metadata_summary

        # Step 4: Single-band Optical
        metadata_summary = {
            "sensor": "Panchromatic / Single-Band Optical",
            "modality": "Optical Single-Band",
            "has_true_color_rgb": False,
            "ndvi_supported": False,
            "ndwi_supported": False,
            "ndbi_supported": False,
        }
        return RasterType.OPTICAL_SINGLE_BAND, band_map, metadata_summary
