"""
Configuration settings for Single-Image Remote-Sensing Intelligence.
Supports environment overrides for model checkpoints, device targeting, and precision.
"""
import os
from pydantic_settings import BaseSettings


def resolve_geochat_path() -> str:
    env_path = os.getenv("GEOCHAT_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.abspath("models/geochat-7b"),
        os.path.expanduser("~/models/geochat-7b"),
        "C:/Users/jsaak/models/geochat-7b",
        os.path.expanduser("~/.cache/huggingface/hub/models--MBZUAI--geochat-7b"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                files = os.listdir(c)
                if any(f.endswith((".bin", ".safetensors", ".pt", ".gguf")) for f in files):
                    return c
            except Exception:
                continue
    return env_path or "models/geochat-7b"


class SingleImageModelConfig(BaseSettings):
    # Model Weights Path (e.g. MBZUAI/geochat-7b or local checkpoint)
    GEOCHAT_MODEL_PATH: str = resolve_geochat_path()
    
    # Target Device: 'auto', 'cuda', or 'cpu'
    TARGET_DEVICE: str = os.getenv("TARGET_DEVICE", "auto")
    
    # Precision Mode: '4bit', '8bit', 'float16', 'float32'
    # 4bit enables running 7B models comfortably inside 6GB VRAM (RTX 3050 Laptop)
    PRECISION: str = os.getenv("PRECISION", "4bit")
    
    # Provider: 'geochat', 'transformers', 'mock_benchmark'
    VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "auto")
    
    # Max generation tokens
    MAX_NEW_TOKENS: int = 256
    
    # Temperature
    TEMPERATURE: float = 0.2

    model_config = {"case_sensitive": True}


model_config = SingleImageModelConfig()
