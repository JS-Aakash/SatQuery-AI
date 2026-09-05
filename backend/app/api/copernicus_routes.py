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


from services.copernicus.nlp_analyzer import CopernicusNLPAnalyzer


@router.post("/ai-analyze")
def analyze_with_ai_layer(payload: Dict[str, Any] = Body(...)):
    """
    Connects processed Copernicus imagery and calculated spectral statistics
    into the NLP AI agentic pipeline for natural language scene reasoning,
    metric computation, and visual grounding bounding boxes.
    """
    query = payload.get("query") or "Provide comprehensive analysis of vegetation canopy, water bodies, and infrastructure from this satellite observation."
    image_b64 = payload.get("image_url", "")
    analysis_type = payload.get("analysis_type", "NDVI")
    stats = payload.get("statistics", {})
    change_stats = payload.get("change_statistics") or {}
    location_name = payload.get("location_name") or "Perundurai, Tamil Nadu, India"
    aoi_area_sq_km = float(payload.get("aoi_area_sq_km") or stats.get("analyzed_area_km2") or change_stats.get("analyzed_area_km2") or 5.42)
    temporal_dates = payload.get("temporal_dates")

    nlp_result = CopernicusNLPAnalyzer.analyze_query(
        query=query,
        analysis_type=analysis_type,
        location_name=location_name,
        aoi_area_sq_km=aoi_area_sq_km,
        statistics=stats,
        change_statistics=change_stats,
        temporal_dates=tuple(temporal_dates) if temporal_dates else None,
    )

    return {
        "query": query,
        "analysis_type": analysis_type,
        "ai_explanation": nlp_result["answer"],
        "confidence": nlp_result["confidence"],
        "confidence_formatted": nlp_result["confidence_formatted"],
        "key_metrics": nlp_result["key_metrics"],
        "bounding_boxes": nlp_result["bounding_boxes"],
        "spectral_evidence": nlp_result["evidence"],
        "model_used": nlp_result["model_used"],
        "status": "COMPLETED"
    }

