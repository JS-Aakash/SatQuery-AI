"""
Band Service for Multispectral Imagery.
Handles band selection, natural RGB composites, false-color (NIR) composites,
and non-destructive percentile-stretched web-safe preview generation.
"""
from typing import Union, Tuple, List, Optional
import io
import base64
import numpy as np
from PIL import Image

from .reader import GeoTIFFReader


class BandService:
    """
    Multispectral band combination and web-safe visual composite generator.
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
            # Clamp band requests to available count
            b_r = min(bands[0], total_bands)
            b_g = min(bands[1], total_bands)
            b_b = min(bands[2], total_bands)

            r = ds.read(b_r)
            g = ds.read(b_g)
            b = ds.read(b_b)

        r_norm = BandService.normalize_band_percentile(r)
        g_norm = BandService.normalize_band_percentile(g)
        b_norm = BandService.normalize_band_percentile(b)

        rgb_array = np.stack([r_norm, g_norm, b_norm], axis=-1)
        img = Image.fromarray(rgb_array, mode="RGB")

        # Resize for web performance if needed
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"

        return img, data_url

    @staticmethod
    def create_true_color_rgb(source: Union[str, bytes, io.BytesIO]) -> Tuple[Image.Image, str]:
        """
        Creates a Natural True-Color (RGB) composite.
        For Sentinel-2 (12 bands): Bands 4 (Red), 3 (Green), 2 (Blue).
        For 3-band rasters: Bands 1, 2, 3.
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            if count >= 4:
                # Sentinel-2 / Landsat 4-band or 12-band
                bands = (4, 3, 2)
            elif count == 3:
                bands = (1, 2, 3)
            elif count == 2:
                # Dual band SAR / mock
                bands = (1, 2, 1)
            else:
                # Single band
                bands = (1, 1, 1)

        return BandService.create_composite(source, bands=bands)

    @staticmethod
    def create_false_color_infrared(source: Union[str, bytes, io.BytesIO]) -> Tuple[Image.Image, str]:
        """
        Creates a Color Infrared (CIR / False-Color) composite (NIR, Red, Green).
        Highlights vegetation in vivid crimson red and water in dark tones.
        For Sentinel-2: Bands 8 (NIR), 4 (Red), 3 (Green).
        For 4-band optical: Bands 4 (NIR), 1 (Red), 2 (Green).
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            if count >= 8:
                # Sentinel-2 with B08 NIR
                bands = (8, 4, 3)
            elif count >= 4:
                # 4-band with NIR as 4th band
                bands = (4, 1, 2)
            else:
                # Fallback
                bands = (min(count, 3), min(count, 2), 1)

        return BandService.create_composite(source, bands=bands)
