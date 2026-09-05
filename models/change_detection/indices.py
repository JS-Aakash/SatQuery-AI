"""
Spectral Index Computation Engine for Multispectral Remote-Sensing Imagery.
Implements NDVI (Vegetation), NDWI (Water), and NDBI (Built-up) difference metrics.
Used as calibrated physical supporting evidence alongside learned change models.
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from .interfaces import SpectralIndicesResult


class SpectralIndexCalculator:
    """
    Computes Normalized Difference indices from multispectral arrays.
    Safe against divide-by-zero, NaNs, and missing bands.
    """

    @staticmethod
    def calculate_ndvi(array: np.ndarray, red_idx: int = 0, nir_idx: int = 1) -> np.ndarray:
        """
        Normalized Difference Vegetation Index:
        NDVI = (NIR - Red) / (NIR + Red)
        Range: [-1.0, 1.0]. Healthy vegetation yields values > 0.4.
        """
        if array.ndim == 2:
            return np.zeros_like(array, dtype=np.float32)

        total_channels = array.shape[0] if array.shape[0] < min(array.shape[1], array.shape[2]) else array.shape[2]
        if array.shape[0] < min(array.shape[1], array.shape[2]):
            # Shape is (C, H, W)
            red = array[min(red_idx, total_channels - 1)].astype(np.float32)
            nir = array[min(nir_idx, total_channels - 1)].astype(np.float32)
        else:
            # Shape is (H, W, C)
            red = array[:, :, min(red_idx, total_channels - 1)].astype(np.float32)
            nir = array[:, :, min(nir_idx, total_channels - 1)].astype(np.float32)

        denom = nir + red
        denom = np.where(denom == 0, 1e-6, denom)
        ndvi = (nir - red) / denom
        return np.clip(np.nan_to_num(ndvi, nan=0.0), -1.0, 1.0)

    @staticmethod
    def calculate_ndwi(array: np.ndarray, green_idx: int = 1, nir_idx: int = 0) -> np.ndarray:
        """
        Normalized Difference Water Index (McFeeters):
        NDWI = (Green - NIR) / (Green + NIR)
        Water bodies typically yield positive NDWI values (> 0.0 to 0.5).
        """
        if array.ndim == 2:
            return np.zeros_like(array, dtype=np.float32)

        total_channels = array.shape[0] if array.shape[0] < min(array.shape[1], array.shape[2]) else array.shape[2]
        if array.shape[0] < min(array.shape[1], array.shape[2]):
            green = array[min(green_idx, total_channels - 1)].astype(np.float32)
            nir = array[min(nir_idx, total_channels - 1)].astype(np.float32)
        else:
            green = array[:, :, min(green_idx, total_channels - 1)].astype(np.float32)
            nir = array[:, :, min(nir_idx, total_channels - 1)].astype(np.float32)

        denom = green + nir
        denom = np.where(denom == 0, 1e-6, denom)
        ndwi = (green - nir) / denom
        return np.clip(np.nan_to_num(ndwi, nan=0.0), -1.0, 1.0)

    @staticmethod
    def calculate_ndbi(array: np.ndarray, swir_idx: int = 0, nir_idx: int = 1) -> np.ndarray:
        """
        Normalized Difference Built-up Index:
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        Impervious surfaces, concrete, and urban structures yield positive NDBI values.
        """
        if array.ndim == 2:
            return np.zeros_like(array, dtype=np.float32)

        total_channels = array.shape[0] if array.shape[0] < min(array.shape[1], array.shape[2]) else array.shape[2]
        if array.shape[0] < min(array.shape[1], array.shape[2]):
            swir = array[min(swir_idx, total_channels - 1)].astype(np.float32)
            nir = array[min(nir_idx, total_channels - 1)].astype(np.float32)
        else:
            swir = array[:, :, min(swir_idx, total_channels - 1)].astype(np.float32)
            nir = array[:, :, min(nir_idx, total_channels - 1)].astype(np.float32)

        denom = swir + nir
        denom = np.where(denom == 0, 1e-6, denom)
        ndbi = (swir - nir) / denom
        return np.clip(np.nan_to_num(ndbi, nan=0.0), -1.0, 1.0)

    @staticmethod
    def compute_bitemporal_indices(
        arr_t1: np.ndarray,
        arr_t2: np.ndarray,
        band_names: Optional[List[str]] = None
    ) -> SpectralIndicesResult:
        """
        Computes multi-index temporal deltas (ΔNDVI, ΔNDWI, ΔNDBI) between T1 and T2.
        Returns formatted indicators and physical interpretations.
        """
        # Select band indices based on available channels
        # For standard RGB: Red=0, Green=1, Blue=2.
        # For 4+ band: NIR is usually channel 3 or channel 7.
        red_idx = 0
        green_idx = 1
        nir_idx = 2 if arr_t1.shape[0] <= 3 or (arr_t1.ndim == 3 and arr_t1.shape[2] <= 3) else 3
        swir_idx = 0

        ndvi_1 = SpectralIndexCalculator.calculate_ndvi(arr_t1, red_idx, nir_idx)
        ndvi_2 = SpectralIndexCalculator.calculate_ndvi(arr_t2, red_idx, nir_idx)
        m_ndvi1 = float(np.mean(ndvi_1))
        m_ndvi2 = float(np.mean(ndvi_2))
        d_ndvi = round(m_ndvi2 - m_ndvi1, 4)

        ndwi_1 = SpectralIndexCalculator.calculate_ndwi(arr_t1, green_idx, nir_idx)
        ndwi_2 = SpectralIndexCalculator.calculate_ndwi(arr_t2, green_idx, nir_idx)
        m_ndwi1 = float(np.mean(ndwi_1))
        m_ndwi2 = float(np.mean(ndwi_2))
        d_ndwi = round(m_ndwi2 - m_ndwi1, 4)

        ndbi_1 = SpectralIndexCalculator.calculate_ndbi(arr_t1, swir_idx, nir_idx)
        ndbi_2 = SpectralIndexCalculator.calculate_ndbi(arr_t2, swir_idx, nir_idx)
        m_ndbi1 = float(np.mean(ndbi_1))
        m_ndbi2 = float(np.mean(ndbi_2))
        d_ndbi = round(m_ndbi2 - m_ndbi1, 4)

        indicators: List[str] = []
        if d_ndvi < -0.05:
            indicators.append(f"Vegetation Canopy Reduction (ΔNDVI: {d_ndvi:+.3f})")
        elif d_ndvi > 0.05:
            indicators.append(f"Vegetation Regrowth / Canopy Expansion (ΔNDVI: {d_ndvi:+.3f})")

        if d_ndbi > 0.05:
            indicators.append(f"Impervious / Built-up Construction Growth (ΔNDBI: {d_ndbi:+.3f})")
        elif d_ndbi < -0.05:
            indicators.append(f"Demolition or Bare Soil Conversion (ΔNDBI: {d_ndbi:+.3f})")

        if d_ndwi < -0.04:
            indicators.append(f"Water Body Surface Area Contraction (ΔNDWI: {d_ndwi:+.3f})")
        elif d_ndwi > 0.04:
            indicators.append(f"Inundation / Water Basin Expansion (ΔNDWI: {d_ndwi:+.3f})")

        if not indicators:
            indicators.append("Stable spectral response with minimal macroscopic land-cover conversion.")

        interp = (
            f"Spectral indicators indicate a net { 'loss' if d_ndvi < 0 else 'gain' } in vegetative density "
            f"(ΔNDVI = {d_ndvi:+.3f}) paired with a { 'surge' if d_ndbi > 0 else 'decrease' } in built-up reflectance "
            f"(ΔNDBI = {d_ndbi:+.3f}) and water index shift of ΔNDWI = {d_ndwi:+.3f}."
        )

        return SpectralIndicesResult(
            mean_ndvi_t1=round(m_ndvi1, 3),
            mean_ndvi_t2=round(m_ndvi2, 3),
            delta_ndvi=d_ndvi,
            mean_ndwi_t1=round(m_ndwi1, 3),
            mean_ndwi_t2=round(m_ndwi2, 3),
            delta_ndwi=d_ndwi,
            mean_ndbi_t1=round(m_ndbi1, 3),
            mean_ndbi_t2=round(m_ndbi2, 3),
            delta_ndbi=d_ndbi,
            interpretation=interp,
            supporting_indicators=indicators
        )
