"""
FastAPI Routes for Copernicus Data Space & Sentinel Hub Satellite Processing Module.
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional

from services.copernicus import (
    LocationGeocodingService,
    CopernicusCatalogService,
    CopernicusSearchRequest,
    CopernicusSearchResponse,
    ProcessAnalysisRequest,
    ProcessAnalysisResponse,
    SpectralIndexType,
    TemporalChangeRequest,
    TemporalChangeResponse,
    GeocodeResult,
    copernicus_auth,
    copernicus_processing,
    copernicus_change,
)
from models.single_image import model_manager
from agent import agent_orchestrator

router = APIRouter(prefix="/copernicus", tags=["Copernicus Satellite Analysis"])

catalog_service = CopernicusCatalogService(auth_service=copernicus_auth)


@router.post("/geocode", response_model=GeocodeResult)
def geocode_location(payload: Dict[str, str] = Body(...)):
    """
    Geocodes a place query (e.g. 'Perundurai', 'Erode', 'Coimbatore') into coordinates & bounding box.
    """
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    result = LocationGeocodingService.geocode(query)
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not find coordinates for '{query}'.")
    return result


@router.post("/search", response_model=CopernicusSearchResponse)
def search_copernicus_catalog(request: CopernicusSearchRequest):
    """
    Queries real Copernicus STAC Catalog for Sentinel-2 L2A & Sentinel-1 observations
    matching AOI, date range, and maximum cloud coverage.
    """
    if not request.aoi or not request.aoi.coordinates:
        raise HTTPException(status_code=400, detail="Must provide a valid AOI Polygon.")
    if request.start_date > request.end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be after end_date.")

    try:
        return catalog_service.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Copernicus Catalog Search Failed: {str(e)}")


@router.post("/process", response_model=ProcessAnalysisResponse)
def process_satellite_analysis(request: ProcessAnalysisRequest):
    """
    Executes Sentinel Hub Processing API request for specified index type (RGB, False Color, NDVI, NDWI, NDBI)
    and computes authentic spatial pixel statistics over the AOI.
    """
    if not request.observation_id:
        raise HTTPException(status_code=400, detail="observation_id is required.")

    try:
        return copernicus_processing.process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing API Error: {str(e)}")


@router.post("/ndvi", response_model=ProcessAnalysisResponse)
def process_ndvi(request: ProcessAnalysisRequest):
    """Convenience endpoint for NDVI Normalized Difference Vegetation Index."""
    request.analysis_type = SpectralIndexType.NDVI
    return process_satellite_analysis(request)


@router.post("/ndwi", response_model=ProcessAnalysisResponse)
def process_ndwi(request: ProcessAnalysisRequest):
    """Convenience endpoint for NDWI Normalized Difference Water Index."""
    request.analysis_type = SpectralIndexType.NDWI
    return process_satellite_analysis(request)


@router.post("/ndbi", response_model=ProcessAnalysisResponse)
def process_ndbi(request: ProcessAnalysisRequest):
    """Convenience endpoint for NDBI Normalized Difference Built-up Index."""
    request.analysis_type = SpectralIndexType.NDBI
    return process_satellite_analysis(request)


@router.post("/change-detection", response_model=TemporalChangeResponse)
def process_temporal_change(request: TemporalChangeRequest):
    """
    Executes bi-temporal change detection between Before and After observations over the same AOI.
    """
    if not request.before_observation_id or not request.after_observation_id:
        raise HTTPException(status_code=400, detail="Must specify both before_observation_id and after_observation_id.")

    try:
        return copernicus_change.compute_change(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal Change Computation Failed: {str(e)}")


@router.post("/ai-analyze")
def analyze_with_ai_layer(payload: Dict[str, Any] = Body(...)):
    """
    Connects processed Copernicus imagery and calculated spectral statistics
    into the existing VLM AI agentic pipeline for natural language scene reasoning.
    """
    query = payload.get("query") or "Provide comprehensive analysis of vegetation canopy, water bodies, and infrastructure from this satellite observation."
    image_b64 = payload.get("image_url", "")
    analysis_type = payload.get("analysis_type", "NDVI")
    stats = payload.get("statistics", {})

    # Extract base64 payload
    raw_b64 = image_b64.split("base64,")[1] if "base64," in image_b64 else image_b64

    # Run through VLM / Single Image Manager
    res = model_manager.infer(
        image_input=raw_b64,
        query=f"[{analysis_type} Remote Sensing Analysis]: {query}. Computed Mean: {stats.get('mean', 'N/A')}, Min: {stats.get('min', 'N/A')}, Max: {stats.get('max', 'N/A')}",
        task="vqa"
    )

    return {
        "query": query,
        "analysis_type": analysis_type,
        "ai_explanation": res.answer,
        "confidence": res.confidence,
        "confidence_formatted": res.confidence_formatted,
        "spectral_evidence": res.evidence_metadata,
        "model_used": res.model_name,
        "status": "COMPLETED"
    }
