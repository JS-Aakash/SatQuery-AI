"""
Band Service for Multispectral Imagery.
Handles band selection, natural RGB composites, false-color composites,
non-destructive percentile-stretched preview generation, and exact spectral index calculations (NDVI, NDWI, NDBI).
"""
from typing import Union, Tuple, List, Optional, Dict, Any
import io
import base64
import inspect
import dis
import numpy as np
from PIL import Image

from .reader import GeoTIFFReader
from .raster_classifier import RasterClassifier, RasterType, BandRole


class CompositeResultTuple(tuple):
    """
    Dual-compatibility tuple for (Image, DataURL, Description)
    that unpacks as 2 elements (Image, DataURL) when unpacked into 2 variables (e.g. img, url = ...),
    and 3 elements (Image, DataURL, Description) when unpacked into 3 variables (e.g. img, url, desc = ...).
    """
    def __new__(cls, img: Image.Image, url: str, desc: str = ""):
        return super().__new__(cls, (img, url, desc))

    @property
    def image(self) -> Image.Image:
        return self[0]

    @property
    def url(self) -> str:
        return self[1]

    @property
    def description(self) -> str:
        return self[2]

    def __iter__(self):
        try:
            frame = inspect.currentframe().f_back
            code = frame.f_code
            lasti = frame.f_lasti
            for inst in dis.get_instructions(code):
                if inst.offset == lasti:
                    if inst.opname == "UNPACK_SEQUENCE":
                        if inst.argval == 2:
                            return iter(self[:2])
                        elif inst.argval == 3:
                            return iter(self[:3])
        except Exception:
            pass
        return super().__iter__()


class BandService:
    """
    Multispectral band combination, web-safe visual composite generator, and spectral index engine.
    """

    @staticmethod
    def normalize_band_percentile(band_data: np.ndarray, lower_p: float = 2.0, upper_p: float = 98.0) -> np.ndarray:
        """
        Normalizes a raw multispectral band (uint16/float32) to uint8 [0, 255]
        using standard remote-sensing percentile contrast stretching.
        """
        valid_mask = ~np.isnan(band_data) & (band_data > 0)
        if not np.any(valid_mask):
            return np.zeros(band_data.shape, dtype=np.uint8)

        p_low = np.percentile(band_data[valid_mask], lower_p)
        p_high = np.percentile(band_data[valid_mask], upper_p)

        if p_high <= p_low:
            p_high = p_low + 1.0

        stretched = np.clip((band_data - p_low) / (p_high - p_low), 0.0, 1.0)
        return (stretched * 255.0).astype(np.uint8)

    @staticmethod
    def create_composite(
        source: Union[str, bytes, io.BytesIO],
        bands: Tuple[int, int, int] = (1, 2, 3),
        max_size: int = 1024
    ) -> Tuple[Image.Image, str]:
        """
        Extracts 3 specified bands (1-indexed), applies contrast stretch,
        and constructs a PIL RGB Image and Base64 Data URL.
        Never alters the original raster source.
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            total_bands = ds.count
            b_r = min(max(1, bands[0]), total_bands)
            b_g = min(max(1, bands[1]), total_bands)
            b_b = min(max(1, bands[2]), total_bands)

            r = ds.read(b_r)
            g = ds.read(b_g)
            b = ds.read(b_b)

        r_norm = BandService.normalize_band_percentile(r)
        g_norm = BandService.normalize_band_percentile(g)
        b_norm = BandService.normalize_band_percentile(b)

        rgb_array = np.stack([r_norm, g_norm, b_norm], axis=-1)
        img = Image.fromarray(rgb_array, mode="RGB")

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"

        return img, data_url

    @staticmethod
    def create_true_color_rgb(
        source: Union[str, bytes, io.BytesIO],
        filename: str = "raster.tif"
    ) -> Tuple[Image.Image, str, str]:
        """
        Creates a valid True-Color (RGB) composite.
        Checks band roles:
        - If B04 (Red), B03 (Green), B02 (Blue) exist -> True Color RGB (B04, B03, B02).
        - If 3-band standard raster -> Bands (1, 2, 3).
        - If B04, B08, B11, B12 are present without Blue/Green -> DO NOT falsely map B04->R, B08->G, B11->B!
          Instead generates a single-band Red stretch or panchromatic rendering.
        Returns: (Image, DataURL, VisualizationTypeDescription)
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            descriptions = list(ds.descriptions) if ds.descriptions else []

            raster_type, band_map, _ = RasterClassifier.classify_raster(
                total_bands=count,
                descriptions=descriptions,
                filename=filename
            )

            # Check if Red, Green, Blue exist in band_map
            red_idx = band_map.get(BandRole.RED) or band_map.get(BandRole.RGB_RED)
            green_idx = band_map.get(BandRole.GREEN) or band_map.get(BandRole.RGB_GREEN)
            blue_idx = band_map.get(BandRole.BLUE) or band_map.get(BandRole.RGB_BLUE)

            if red_idx and green_idx and blue_idx:
                img, url = BandService.create_composite(source, bands=(red_idx, green_idx, blue_idx))
                return CompositeResultTuple(img, url, f"True Color RGB (Bands {red_idx}, {green_idx}, {blue_idx})")

            if count == 3:
                img, url = BandService.create_composite(source, bands=(1, 2, 3))
                return CompositeResultTuple(img, url, "Standard 3-Band RGB")

            if raster_type == RasterType.SAR:
                vv_idx = band_map.get(BandRole.SAR_VV, 1)
                vh_idx = band_map.get(BandRole.SAR_VH, min(2, count))
                img, url = BandService.create_composite(source, bands=(vv_idx, vh_idx, vv_idx))
                return CompositeResultTuple(img, url, "SAR Polarimetric Composite (VV / VH)")

            # For 4-band or multi-band lacking Green/Blue (e.g. B04, B08, B11, B12):
            # Render band 1 (Red B04) as grayscale or CIR NIR-Red
            if red_idx:
                nir_idx = band_map.get(BandRole.NIR)
                if nir_idx:
                    # Clear NIR-Red False Color composite
                    img, url = BandService.create_composite(source, bands=(nir_idx, red_idx, red_idx))
                    return CompositeResultTuple(img, url, f"False Color NIR-Red (B08: Band {nir_idx}, B04: Band {red_idx})")
                img, url = BandService.create_composite(source, bands=(red_idx, red_idx, red_idx))
                return CompositeResultTuple(img, url, f"Panchromatic Red Band {red_idx} (B04)")

            # Fallback single band
            img, url = BandService.create_composite(source, bands=(1, 1, 1))
            return CompositeResultTuple(img, url, "Single-Channel Grayscale Visualization")

    @staticmethod
    def create_false_color_infrared(
        source: Union[str, bytes, io.BytesIO],
        filename: str = "raster.tif"
    ) -> CompositeResultTuple:
        """
        Creates a valid False-Color Color Infrared (CIR) composite (NIR, Red, Green).
        - If B08 (NIR), B04 (Red), B03 (Green) exist -> CIR (B08, B04, B03).
        - If NIR and Red exist -> (NIR, Red, Red).
        - If SAR or 3-band standard raster -> Safe contrast composite.
        Returns: CompositeResultTuple (unpacks as 2 or 3 items)
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            descriptions = list(ds.descriptions) if ds.descriptions else []

            raster_type, band_map, _ = RasterClassifier.classify_raster(
                total_bands=count,
                descriptions=descriptions,
                filename=filename
            )

            if raster_type == RasterType.SAR:
                vv_idx = band_map.get(BandRole.SAR_VV, 1)
                vh_idx = band_map.get(BandRole.SAR_VH, min(2, count))
                img, url = BandService.create_composite(source, bands=(vv_idx, vh_idx, vv_idx))
                return CompositeResultTuple(img, url, "SAR Polarimetric Composite")

            nir_idx = band_map.get(BandRole.NIR)
            red_idx = band_map.get(BandRole.RED) or band_map.get(BandRole.RGB_RED)
            green_idx = band_map.get(BandRole.GREEN) or band_map.get(BandRole.RGB_GREEN)

            if nir_idx and red_idx and green_idx:
                img, url = BandService.create_composite(source, bands=(nir_idx, red_idx, green_idx))
                return CompositeResultTuple(img, url, f"False Color Infrared (Bands {nir_idx}, {red_idx}, {green_idx})")

            if nir_idx and red_idx:
                img, url = BandService.create_composite(source, bands=(nir_idx, red_idx, red_idx))
                return CompositeResultTuple(img, url, f"False Color NIR-Red (Bands {nir_idx}, {red_idx})")

            if count >= 3:
                img, url = BandService.create_composite(source, bands=(min(3, count), 1, min(2, count)))
                return CompositeResultTuple(img, url, "Simulated Color Infrared")

            img, url = BandService.create_composite(source, bands=(1, 1, 1))
            return CompositeResultTuple(img, url, "Grayscale Infrared Simulation")


    @staticmethod
    def compute_spectral_indices(
        source: Union[str, bytes, io.BytesIO],
        filename: str = "raster.tif"
    ) -> Dict[str, Any]:
        """
        Calculates physical spectral indices ONLY when the required bands exist:
        - NDVI: (B08 - B04) / (B08 + B04)
        - NDWI: (B03 - B08) / (B03 + B08)
        - NDBI: (B11 - B08) / (B11 + B08)
        Never fabricates missing bands.
        """
        results: Dict[str, Any] = {
            "ndvi": None,
            "ndwi": None,
            "ndbi": None,
            "available_indices": [],
            "statistics": {},
            "band_mapping": {}
        }

        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            descriptions = list(ds.descriptions) if ds.descriptions else []

            raster_type, band_map, _ = RasterClassifier.classify_raster(
                total_bands=count,
                descriptions=descriptions,
                filename=filename
            )

            if raster_type == RasterType.SAR:
                # Do NOT calculate NDVI/NDWI/NDBI for SAR
                return results

            results["band_mapping"] = {k.value: v for k, v in band_map.items()}

            b_red_idx = band_map.get(BandRole.RED) or band_map.get(BandRole.RGB_RED)
            b_green_idx = band_map.get(BandRole.GREEN) or band_map.get(BandRole.RGB_GREEN)
            b_blue_idx = band_map.get(BandRole.BLUE) or band_map.get(BandRole.RGB_BLUE)
            b_nir_idx = band_map.get(BandRole.NIR)
            b_swir_idx = band_map.get(BandRole.SWIR_1)

            # 1. Calculate NDVI = (NIR - RED) / (NIR + RED)
            if b_nir_idx and b_red_idx:
                nir = ds.read(b_nir_idx).astype(np.float32)
                red = ds.read(b_red_idx).astype(np.float32)
                denom = nir + red
                denom[denom == 0] = 1e-6
                ndvi_arr = (nir - red) / denom
                ndvi_valid = ndvi_arr[~np.isnan(ndvi_arr)]

                results["ndvi"] = ndvi_arr
                results["available_indices"].append("NDVI")
                results["statistics"]["ndvi"] = {
                    "mean": round(float(np.mean(ndvi_valid)), 4) if ndvi_valid.size else 0.0,
                    "min": round(float(np.min(ndvi_valid)), 4) if ndvi_valid.size else 0.0,
                    "max": round(float(np.max(ndvi_valid)), 4) if ndvi_valid.size else 0.0,
                    "std": round(float(np.std(ndvi_valid)), 4) if ndvi_valid.size else 0.0,
                    "vegetated_pct": round(float(np.mean(ndvi_valid > 0.4) * 100.0), 1) if ndvi_valid.size else 0.0,
                    "bare_land_pct": round(float(np.mean((ndvi_valid >= 0.0) & (ndvi_valid < 0.25)) * 100.0), 1) if ndvi_valid.size else 0.0,
                }

            # 2. Calculate NDWI = (GREEN - NIR) / (GREEN + NIR)
            if b_green_idx and b_nir_idx:
                green = ds.read(b_green_idx).astype(np.float32)
                nir = ds.read(b_nir_idx).astype(np.float32)
                denom = green + nir
                denom[denom == 0] = 1e-6
                ndwi_arr = (green - nir) / denom
                ndwi_valid = ndwi_arr[~np.isnan(ndwi_arr)]

                results["ndwi"] = ndwi_arr
                results["available_indices"].append("NDWI")
                results["statistics"]["ndwi"] = {
                    "mean": round(float(np.mean(ndwi_valid)), 4) if ndwi_valid.size else 0.0,
                    "water_body_pct": round(float(np.mean(ndwi_valid > 0.0) * 100.0), 1) if ndwi_valid.size else 0.0,
                }

            # 3. Calculate NDBI = (SWIR - NIR) / (SWIR + NIR)
            if b_swir_idx and b_nir_idx:
                swir = ds.read(b_swir_idx).astype(np.float32)
                nir = ds.read(b_nir_idx).astype(np.float32)
                denom = swir + nir
                denom[denom == 0] = 1e-6
                ndbi_arr = (swir - nir) / denom
                ndbi_valid = ndbi_arr[~np.isnan(ndbi_arr)]

                results["ndbi"] = ndbi_arr
                results["available_indices"].append("NDBI")
                results["statistics"]["ndbi"] = {
                    "mean": round(float(np.mean(ndbi_valid)), 4) if ndbi_valid.size else 0.0,
                    "built_up_pct": round(float(np.mean(ndbi_valid > 0.1) * 100.0), 1) if ndbi_valid.size else 0.0,
                }

        return results
