"""
Unit and Integration Tests for Natural-Language Geospatial Search & Autonomous Earth Observation Reasoning.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from services.copernicus.geospatial_search import (
    NaturalLanguageQueryParser,
    geospatial_search_engine,
    GeospatialQueryIntent
)

client = TestClient(app)


def test_parser_vacant_land_query():
    """Validates natural language parsing for potentially vacant land."""
    q = NaturalLanguageQueryParser.parse("Find potentially vacant land around Perundurai")
    assert q.intent == GeospatialQueryIntent.VACANT_LAND
    assert q.target == "potentially_vacant_land"
    assert q.location == "Perundurai"
    assert q.is_supported is True


def test_parser_radius_and_area_query():
    """Validates radius in km, large area threshold, and distance ranking preferences."""
    q = NaturalLanguageQueryParser.parse("Find large empty lands within 10 km of Perundurai")
    assert q.intent == GeospatialQueryIntent.VACANT_LAND
    assert q.radius_km == 10.0
    assert q.minimum_area_hectares is not None and q.minimum_area_hectares >= 2.0
    assert q.location == "Perundurai"


def test_parser_vegetation_health_query():
    """Validates vegetation stress queries."""
    q = NaturalLanguageQueryParser.parse("Find areas with poor vegetation health near Erode")
    assert q.intent == GeospatialQueryIntent.VEGETATION_HEALTH
    assert q.target == "low_vegetation_health"
    assert q.location == "Erode"


def test_parser_temporal_change_query():
    """Validates temporal change parsing with years."""
    q = NaturalLanguageQueryParser.parse("Find areas that changed from vegetation to built-up land around Perundurai between 2022 and 2026")
    assert q.intent == GeospatialQueryIntent.TEMPORAL_CHANGE
    assert q.from_year == 2022
    assert q.to_year == 2026
    assert q.location == "Perundurai"


def test_parser_unsupported_commercial_query():
    """Validates that financial/legal purchase queries are gracefully rejected with explanatory notices."""
    q = NaturalLanguageQueryParser.parse("Find the best land to buy in Perundurai")
    assert q.is_supported is False
    assert q.intent == GeospatialQueryIntent.UNSUPPORTED
    assert "ownership" in q.unsupported_reason.lower() or "purchase" in q.unsupported_reason.lower()


def test_geospatial_search_engine_execution():
    """Validates end-to-end execution of potentially vacant land detection with candidate polygons."""
    res = geospatial_search_engine.execute_search("Find potentially vacant land around Perundurai")
    assert res.status == "COMPLETED"
    assert len(res.candidates) > 0
    assert res.total_aoi_area_km2 > 0
    assert res.candidate_count == len(res.candidates)
    
    top = res.candidates[0]
    assert top.area_hectares > 0
    assert top.distance_km > 0
    assert len(top.polygon_coordinates) >= 4
    assert len(top.bounding_box) == 4
    assert top.confidence_score >= 0.70
    assert "⚠️" in res.disclaimer


def test_geospatial_search_api_endpoint():
    """Validates POST /api/geospatial/search endpoint."""
    payload = {"query": "Find large empty lands within 10 km of Perundurai"}
    res = client.post("/api/geospatial/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["location_name"] is not None
    assert len(data["candidates"]) > 0
    assert "execution_steps" in data


def test_geospatial_followup_api_endpoint():
    """Validates conversational follow-up query refining."""
    search_res = geospatial_search_engine.execute_search("Find potentially vacant land around Perundurai")
    candidates_raw = [c.model_dump() for c in search_res.candidates]

    followup_payload = {
        "prompt": "Only show areas larger than 3 hectares",
        "candidates": candidates_raw,
        "location": "Perundurai"
    }

    res = client.post("/api/geospatial/followup", json=followup_payload)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert all(float(c["area_hectares"]) >= 3.0 for c in data["filtered_candidates"])
