"""
Tests for Copernicus Natural Language Reasoning, Multi-Date Visual Comparison, and Bi-Temporal Change Intelligence.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from services.copernicus.nlp_analyzer import CopernicusNLPAnalyzer
from services.copernicus.change_detection import copernicus_change
from services.copernicus.types import TemporalChangeRequest, AOIGeometry, SpectralIndexType

client = TestClient(app)


def test_copernicus_nlp_building_query():
    """Validates natural language interpretation for building/urban increase queries."""
    res = CopernicusNLPAnalyzer.analyze_query(
        query="How much has building increased?",
        analysis_type="NDBI",
        location_name="Perundurai SIPCOT, Tamil Nadu, India",
        aoi_area_sq_km=5.42,
        change_statistics={
            "delta_mean": 0.084,
            "improved_percent": 15.2,
            "stable_percent": 71.3,
            "declined_percent": 13.5,
        },
        temporal_dates=("2024-01-15T05:09:31Z", "2026-08-20T05:09:31Z")
    )

    assert "Perundurai" in res["answer"]
    assert "expanded by 15.2%" in res["answer"] or "15.2%" in res["answer"]
    assert "km²" in res["answer"]
    assert len(res["bounding_boxes"]) > 0
    assert "New Built-up Expansion" in res["key_metrics"]


def test_copernicus_nlp_vegetation_query():
    """Validates natural language interpretation for agricultural vegetation queries."""
    res = CopernicusNLPAnalyzer.analyze_query(
        query="How much has vegetation changed?",
        analysis_type="NDVI",
        location_name="Erode District, Tamil Nadu, India",
        aoi_area_sq_km=6.80,
        change_statistics={
            "delta_mean": 0.112,
            "improved_percent": 22.4,
            "stable_percent": 65.1,
            "declined_percent": 12.5,
        },
        temporal_dates=("2024-01-15T05:09:31Z", "2026-08-20T05:09:31Z")
    )

    assert "Erode" in res["answer"]
    assert "22.4%" in res["answer"]
    assert len(res["bounding_boxes"]) > 0
    assert "Vegetation Regrowth / Greening" in res["key_metrics"]


def test_copernicus_change_detection_contextual_labels():
    """Validates that change detection returns contextual labels and areas for NDBI."""
    req = TemporalChangeRequest(
        aoi=AOIGeometry(
            type="Polygon",
            coordinates=[[
                [77.5300, 11.2300],
                [77.6350, 11.2300],
                [77.6350, 11.3200],
                [77.5300, 11.3200],
                [77.5300, 11.2300]
            ]],
            bbox=[77.5300, 11.2300, 77.6350, 11.3200]
        ),
        before_observation_id="S2B_MSIL2A_20240115T050931_N0510_R019_T43PFR_20240115T072533",
        after_observation_id="S2A_MSIL2A_20260820T050931_N0510_R019_T43PFR_20260820T072533",
        analysis_type=SpectralIndexType.NDBI
    )

    resp = copernicus_change.compute_change(req)
    assert resp.status == "COMPLETED"
    assert resp.improved_label == "New Construction / Urban Expansion"
    assert resp.stable_label == "Stable Impervious Surface"
    assert resp.declined_label == "Demolition / Surface Clearing"
    assert resp.improved_area_km2 > 0
    assert resp.stable_area_km2 > 0
    assert resp.human_summary is not None
    assert len(resp.before_image_url) > 50
    assert len(resp.after_image_url) > 50
    assert len(resp.diff_image_url) > 50


def test_ai_analyze_endpoint_with_nlp_context():
    """Validates POST /api/copernicus/ai-analyze endpoint with natural language query."""
    payload = {
        "query": "How much has building increased?",
        "analysis_type": "NDBI",
        "location_name": "Perundurai, Tamil Nadu, India",
        "aoi_area_sq_km": 5.42,
        "change_statistics": {
            "delta_mean": 0.084,
            "improved_percent": 15.2,
            "stable_percent": 71.3,
            "declined_percent": 13.5,
            "analyzed_area_km2": 5.42
        },
        "temporal_dates": ["2024-01-15T05:09:31Z", "2026-08-20T05:09:31Z"]
    }

    res = client.post("/api/copernicus/ai-analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Perundurai" in data["ai_explanation"]
    assert "15.2%" in data["ai_explanation"]
    assert "key_metrics" in data
    assert len(data["bounding_boxes"]) > 0
