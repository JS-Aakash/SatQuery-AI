"""
SAR Preprocessor Service.
Handles Synthetic Aperture Radar (SAR) backscatter transformation to decibels (dB),
speckle-safe normalization, and non-destructive polarimetric visual previews.
NEVER overwrites original raw SAR data.
"""
from typing import Union, Tuple, Optional
import io
import base64
import numpy as np
from PIL import Image

from .reader import GeoTIFFReader


class SARPreprocessor:
    """
    Non-destructive preprocessor and visualizer for SAR imagery (Sentinel-1, RISAT-1A).
    """

    @staticmethod
    def linear_to_db(intensity: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """
        Converts linear SAR backscatter amplitude/intensity to Decibels (dB):
        dB = 10 * log10(max(intensity, eps)).
        """
        safe_intensity = np.maximum(intensity, eps)
        return 10.0 * np.log10(safe_intensity)

    @staticmethod
    def normalize_sar_band(
        band_data: np.ndarray,
        lower_p: float = 1.0,
        upper_p: float = 99.0,
        to_db: bool = True
    ) -> np.ndarray:
        """
        Transforms and normalizes a SAR band to uint8 [0, 255] for visualization.
        """
        data = band_data.astype(np.float32)
        valid_mask = ~np.isnan(data) & (data > 0)
        if not np.any(valid_mask):
            return np.zeros(data.shape, dtype=np.uint8)

        if to_db:
            # Check if values appear linear (non-negative dynamic range)
            if np.min(data[valid_mask]) >= 0.0:
                data = SARPreprocessor.linear_to_db(data)

        # Percentile clipping
        p_low = np.percentile(data[valid_mask], lower_p)
        p_high = np.percentile(data[valid_mask], upper_p)

        if p_high <= p_low:
            p_high = p_low + 1.0

        clipped = np.clip((data - p_low) / (p_high - p_low), 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)

    @staticmethod
    def create_sar_preview(
        source: Union[str, bytes, io.BytesIO],
        max_size: int = 1024
    ) -> Tuple[Image.Image, str]:
        """
        Generates a non-destructive web-safe visualization of SAR imagery.
        - Single-pol (VV or VH): Grayscale decibel contrast preview.
        - Dual-pol (VV + VH): Polarimetric False-Color composite:
          R = VV (dB), G = VH (dB), B = |VV - VH| ratio.
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            count = ds.count
            if count >= 2:
                # Dual-pol: VV (band 1) and VH (band 2)
                b1 = ds.read(1)
                b2 = ds.read(2)
                vv_norm = SARPreprocessor.normalize_sar_band(b1)
                vh_norm = SARPreprocessor.normalize_sar_band(b2)
                # Polarimetric cross-ratio
                ratio = np.clip(np.abs(vv_norm.astype(np.float32) - vh_norm.astype(np.float32)), 0, 255).astype(np.uint8)
                rgb_arr = np.stack([vv_norm, vh_norm, ratio], axis=-1)
                img = Image.fromarray(rgb_arr, mode="RGB")
            else:
                # Single band
                b1 = ds.read(1)
                gray_norm = SARPreprocessor.normalize_sar_band(b1)
                img = Image.fromarray(gray_norm, mode="L").convert("RGB")

        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"

        return img, data_url
