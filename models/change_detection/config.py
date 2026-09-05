"""
Configuration for Bi-Temporal Change Intelligence.
Supports environment overrides for model checkpoints, spectral index thresholds,
spatial tolerance, and device selection.
"""
import os
from pydantic_settings import BaseSettings


class ChangeDetectionConfig(BaseSettings):
    # Model Weights Path (e.g., TEOChat or Siamese Change Detection checkpoint)
    TEOCHAT_MODEL_PATH: str = os.getenv("TEOCHAT_MODEL_PATH", "models/teochat-7b")
    CHANGEFORMER_MODEL_PATH: str = os.getenv("CHANGEFORMER_MODEL_PATH", "models/changeformer")

    # Target Device: 'auto', 'cuda', 'cpu'
    TARGET_DEVICE: str = os.getenv("TARGET_DEVICE", "auto")

    # Precision
    PRECISION: str = os.getenv("PRECISION", "4bit")

    # Difference Detection Thresholds (0.0 to 1.0)
    SPECTRAL_CHANGE_THRESHOLD: float = 0.18
    NDVI_DIFFERENCE_THRESHOLD: float = 0.20
    NDWI_DIFFERENCE_THRESHOLD: float = 0.20
    NDBI_DIFFERENCE_THRESHOLD: float = 0.22

    # Minimum polygon area to retain (m^2) to filter pixel noise
    MIN_POLYGON_AREA_M2: float = 100.0

    # Default Ground Sample Distance (GSD in meters per pixel) if not georeferenced
    DEFAULT_GSD_METERS: float = 10.0  # Sentinel-2 standard 10m

    model_config = {"case_sensitive": True}


change_config = ChangeDetectionConfig()
