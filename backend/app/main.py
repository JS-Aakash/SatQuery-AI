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

logger = logging.getLogger("satquery.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Model preloading is managed on-demand via the Mission Control UI and API.
    """
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
