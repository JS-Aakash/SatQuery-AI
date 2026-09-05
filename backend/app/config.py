"""
Configuration settings for SatQuery AI backend.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "SatQuery AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    
    # Storage settings
    UPLOAD_DIR: str = "reports/uploads"
    REPORTS_DIR: str = "reports/generated"
    
    # Feature flags for AI models & GPU warm loading
    ENABLE_REAL_MODELS: bool = True
    WARM_LOAD_ON_STARTUP: bool = False

    model_config = {"case_sensitive": True}


settings = Settings()
