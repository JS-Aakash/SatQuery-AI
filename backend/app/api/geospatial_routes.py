"""
FastAPI Routes for Natural-Language Geospatial Search and Autonomous Earth Observation Reasoning.
"""

import re
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional, List


from services.copernicus.geospatial_search import (
    geospatial_search_engine,
    NaturalLanguageQueryParser,
    GeospatialSearchResponse,
    StructuredGeospatialQuery,
    CandidateRegion
)

router = APIRouter(prefix="/geospatial", tags=["Natural-Language Geospatial Search"])


@router.post("/parse", response_model=StructuredGeospatialQuery)
def parse_natural_language_query(payload: Dict[str, str] = Body(...)):
    """
    Parses conversational user query into a structured geospatial query schema.
    """
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    return NaturalLanguageQueryParser.parse(query)


@router.post("/search", response_model=GeospatialSearchResponse)
def execute_geospatial_search(payload: Dict[str, Any] = Body(...)):
    """
    Executes autonomous end-to-end natural-language geospatial analysis.
    Translates request -> AOI -> Copernicus Sentinel-2 discovery -> spectral land classification -> candidate polygons.
    """
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    custom_aoi = payload.get("aoi")
    context = payload.get("context")

    try:
        return geospatial_search_engine.execute_search(
            query=query,
            custom_aoi=custom_aoi,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geospatial Search Pipeline Error: {str(e)}")


@router.post("/followup")
def execute_search_followup(payload: Dict[str, Any] = Body(...)):
    """
    Processes conversational follow-up questions over existing search results.
    E.g. "Only show areas larger than 3 hectares", "Which one is closest?", "Rank by persistence".
    """
    prompt = payload.get("prompt", "").strip()
    candidates: List[Dict[str, Any]] = payload.get("candidates", [])
    location = payload.get("location", "Perundurai")

    if not prompt:
        raise HTTPException(status_code=400, detail="Follow-up prompt cannot be empty.")

    p_lower = prompt.lower()
    filtered = list(candidates)

    # 1. Filter by area
    area_match = re.search(r'(?:larger than|greater than|above|minimum)\s+(\d+(?:\.\d+)?)\s*(?:hectares?|ha)', p_lower)
    if area_match:
        threshold = float(area_match.group(1))
        filtered = [c for c in filtered if float(c.get("area_hectares", 0)) >= threshold]
        reply = f"Filtered results to {len(filtered)} candidate area(s) larger than {threshold} hectares."

    # 2. Find closest
    elif any(w in p_lower for w in ["closest", "nearest", "near to"]):
        filtered = sorted(filtered, key=lambda c: float(c.get("distance_km", 999)))
        if filtered:
            top = filtered[0]
            reply = f"The closest candidate is '{top.get('name')}' located {top.get('distance_km')} km from {location} ({top.get('area_hectares')} ha)."
        else:
            reply = "No candidate areas available to evaluate distance."

    # 3. Filter by persistence
    elif any(w in p_lower for w in ["persistent", "last year", "consistent"]):
        filtered = [c for c in filtered if float(c.get("persistence_ratio", 0)) >= 0.80]
        reply = f"Filtered to {len(filtered)} candidate(s) exhibiting persistent bare/non-vegetated characteristics across 4+ observations."

    else:
        reply = f"I evaluated your follow-up for {location}. {len(filtered)} candidate areas currently remain active."

    return {
        "reply": reply,
        "filtered_candidates": filtered,
        "count": len(filtered)
    }
