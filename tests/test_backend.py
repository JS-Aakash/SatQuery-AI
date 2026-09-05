"""
Tests for SatQuery AI FastAPI backend, routes, and abstract service contracts.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.base import (
    ImageProcessor,
    VQAService,
    GroundingService,
    ChangeDetectionService,
    OpticalSARService,
    AgentService,
    SatelliteService,
    ReportService,
)
from backend.app.schemas import (
    ModalityEnum,
    SensorEnum,
    TaskType,
    StatusEnum,
)

client = TestClient(app)


def test_health_endpoints():
    """Verify health endpoints respond with 200 OK."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"

    res_api = client.get("/api/health")
    assert res_api.status_code == 200


def test_system_status_honesty():
    """Verify system status reports models as NOT CONFIGURED rather than fake ready."""
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["overall_status"] == StatusEnum.READY
    
    subsystems = {s["name"]: s for s in data["subsystems"]}
    assert "Frontend" in subsystems
    assert "Backend" in subsystems
    assert "Agent" in subsystems
    assert "GeoChat" in subsystems
    assert "Temporal Model" in subsystems
    assert "Optical-SAR Model" in subsystems

    # Check that planned models are NOT marked as fake READY
    assert subsystems["GeoChat"]["status"] == StatusEnum.NOT_CONFIGURED
    assert subsystems["Temporal Model"]["status"] == StatusEnum.NOT_CONFIGURED
    assert subsystems["Optical-SAR Model"]["status"] == StatusEnum.NOT_CONFIGURED


def test_upload_validate_geotiff():
    """Verify GeoTIFF validation and metadata extraction."""
    payload = {
        "filename": "Sentinel2_Chennai_Harbor.tif",
        "file_size_bytes": 4194304,
        "modality_hint": "Multispectral"
    }
    res = client.post("/api/uploads/validate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    meta = data["metadata"]
    assert meta["format"] in ["TIFF", "GeoTIFF"]
    assert meta["modality"] == "Multispectral"
    assert meta["sensor"] == "Sentinel-2"
    assert meta["bands"] == 12


def test_upload_validate_sar():
    """Verify SAR image metadata identification."""
    payload = {
        "filename": "Sentinel1_IW_SAR_Polarimetric.tiff",
        "file_size_bytes": 8388608
    }
    res = client.post("/api/uploads/validate", json=payload)
    assert res.status_code == 200
    data = res.json()
    meta = data["metadata"]
    assert meta["modality"] == "SAR"
    assert meta["sensor"] == "Sentinel-1"
    assert meta["bands"] == 2


def test_analyze_agentic_flow():
    """Verify agentic analysis workflow, task classification, and 7-step execution trace."""
    payload = {
        "query": "What changed between these two dates, and where did the change occur?",
        "input_mode": "upload"
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "Bi-Temporal Change Analysis"
    assert data["confidence"] > 0.8
    assert len(data["evidence"]) >= 2
    assert "is_mock" in data
    
    # Verify auditable execution trace from Agent Orchestrator
    trace = data["execution_trace"]
    assert len(trace) >= 4
    for s in trace:
        assert "step_id" in s
        assert "name" in s
        assert "status" in s
        assert "duration_ms" in s
        assert s["duration_ms"] >= 0


def test_analyses_history():
    """Verify history listing and fetching."""
    res = client.get("/api/analyses")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    first_id = data["items"][0]["id"]

    res_single = client.get(f"/api/analyses/{first_id}")
    assert res_single.status_code == 200
    assert res_single.json()["analysis_id"] == first_id


def test_report_generation():
    """Verify report compilation in Markdown format."""
    # First get an analysis
    hist = client.get("/api/analyses").json()
    anl_id = hist["items"][0]["id"]
    
    req = {
        "analysis_id": anl_id,
        "format": "markdown",
        "include_evidence": True,
        "include_trace": True,
        "analyst_notes": "Test verification of report generator."
    }
    res = client.post("/api/reports", json=req)
    assert res.status_code == 200
    rep = res.json()
    assert "SatQuery AI Mission Report" in rep["content"]
    assert "Auditable Agent Execution Trace" in rep["content"]


def test_satellite_search():
    """Verify satellite scene discovery and AOI area estimation."""
    req = {
        "aoi_coordinates": [
            [80.20, 13.00],
            [80.30, 13.00],
            [80.30, 13.10],
            [80.20, 13.10],
            [80.20, 13.00]
        ],
        "start_date": "2024-01-01",
        "end_date": "2026-01-01",
        "sensors": ["Sentinel-2", "Sentinel-1"]
    }
    res = client.post("/api/satellite/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 1
    assert data["aoi_area_sq_km"] > 0
