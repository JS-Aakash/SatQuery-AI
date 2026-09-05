"""
Configuration for Cross-Modal Optical + SAR Intelligence.
Defines radiometric backscatter thresholds, polarimetric weights,
and spectral fusion parameters.
"""
import os
from pydantic_settings import BaseSettings


class OpticalSARConfig(BaseSettings):
    # Model Weights Path (if deploying specialized learned fusion checkpoints)
    CROSS_MODAL_MODEL_PATH: str = os.getenv("CROSS_MODAL_MODEL_PATH", "models/optical-sar-fusion")

    # Target Device: 'auto', 'cuda', 'cpu'
    TARGET_DEVICE: str = os.getenv("TARGET_DEVICE", "auto")

    # SAR Backscatter Thresholds (in Decibels - dB)
    # Double-bounce (metallic roofs, concrete buildings, right-angle urban geometry)
    DOUBLE_BOUNCE_THRESHOLD_DB: float = -6.5
    # Specular zero-return (smooth open water, calm reservoirs)
    SPECULAR_WATER_THRESHOLD_DB: float = -20.0
    # Cross-pol volume scattering (rough vegetative canopy, forest canopy)
    VOLUME_SCATTERING_THRESHOLD_DB: float = -14.0

    # Optical Spectral Thresholds
    NDVI_VEGETATION_THRESHOLD: float = 0.35
    NDWI_WATER_THRESHOLD: float = 0.15
    NDBI_BUILTUP_THRESHOLD: float = 0.10

    # Fusion Co-Registration Maximum Allowed Shift (normalized % of canvas)
    MAX_ALIGNMENT_OFFSET_PCT: float = 5.0

    model_config = {"case_sensitive": True}


optical_sar_config = OpticalSARConfig()
