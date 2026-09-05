"""
SatQuery AI - Backend Application Entrypoint.
FastAPI service orchestrating remote-sensing multimodal analysis.
"""
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api.routes import router as api_router
from .api.copernicus_routes import router as copernicus_router
from .api.geospatial_routes import router as geospatial_router


logger = logging.getLogger("satquery.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Proactively warms up model weights on startup if weights exist on disk.
    """
    def _warmup():
        try:
            from models.single_image.adapter import GeoChatAdapter
            adapter = GeoChatAdapter()
            if adapter.is_weights_available() and not adapter.is_loaded:
                logger.info("Proactively warming up GeoChat model into GPU VRAM on startup...")
                adapter.load_model()
        except Exception as e:
            logger.warning(f"Startup model warmup note: {e}")

    threading.Thread(target=_warmup, daemon=True).start()
    yield



app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Agentic Vision-Language Platform for Remote-Sensing Satellite Imagery Analysis (ISRO PS ID: 26167)",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
# Health endpoint is available at both /health and /api/health
@app.get("/health")
def root_health():
    return {
        "status": "healthy",
        "service": "SatQuery AI Backend",
        "version": settings.VERSION
    }

app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(copernicus_router, prefix=settings.API_V1_STR)
app.include_router(geospatial_router, prefix=settings.API_V1_STR)



@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "tagline": "Ask your satellite imagery anything.",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "api_v1": settings.API_V1_STR
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
