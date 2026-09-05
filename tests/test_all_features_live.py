"""
Comprehensive verification test for all SatQuery AI features running against FastAPI backend.
Exercises Single Optical, SAR Radar, Bi-Temporal Change, Cross-Modal Fusion, Agent Orchestrator,
Satellite Search, and Autonomous EO Retrieval.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def _make_dummy_image_bytes(format="PNG", color=(50, 100, 150), size=(256, 256)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# 1. Health check
def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


# 2. Upload PNG (testing natural dimensions without zoom distortion)
def test_upload_png_and_tif():
    png_bytes = _make_dummy_image_bytes(format="PNG", size=(640, 480))
    res = client.post(
        "/api/uploads/file",
        files={"file": ("test_optical.png", png_bytes, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    assert "metadata" in data
    assert "dimensions" in data["metadata"]


# 3. Single Image Analysis (Optical VQA & Grounding)
def test_single_optical_analysis():
    img_bytes = _make_dummy_image_bytes(format="TIFF", color=(30, 80, 140))
    up_res = client.post(
        "/api/uploads/file",
        files={"file": ("optical.tif", img_bytes, "image/tiff")}
    )
    image_id = up_res.json()["metadata"]["id"]

    # VQA
    res_vqa = client.post(
        "/api/analyze/single",
        json={"image_id": image_id, "query": "Identify coastal water bodies and ships", "task": "vqa"}
    )
    assert res_vqa.status_code == 200
    vqa_data = res_vqa.json()
    assert vqa_data["answer"] != ""
    assert vqa_data["confidence"] > 0.8
    assert len(vqa_data["evidence_metadata"]) >= 1

    # Grounding
    res_ground = client.post(
        "/api/analyze/single",
        json={"image_id": image_id, "query": "Highlight the water body", "task": "grounding"}
    )
    assert res_ground.status_code == 200
    ground_data = res_ground.json()
    assert len(ground_data["bounding_boxes"]) >= 1


# 4. Bi-Temporal Change Detection
def test_change_detection():
    b_bytes = _make_dummy_image_bytes(format="TIFF", color=(40, 90, 40))
    a_bytes = _make_dummy_image_bytes(format="TIFF", color=(100, 100, 100))

    b_id = client.post("/api/uploads/file", files={"file": ("before.tif", b_bytes, "image/tiff")}).json()["metadata"]["id"]
    a_id = client.post("/api/uploads/file", files={"file": ("after.tif", a_bytes, "image/tiff")}).json()["metadata"]["id"]

    res = client.post(
        "/api/analyze/change",
        json={"before_image_id": b_id, "after_image_id": a_id, "query": "Detect urban expansion"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "has_significant_change" in data
    assert "textual_description" in data
    assert data["confidence"] > 0.7


# 5. Cross-Modal Optical + SAR Intelligence
def test_optical_sar_fusion():
    opt_bytes = _make_dummy_image_bytes(format="TIFF", color=(20, 100, 40))
    sar_bytes = _make_dummy_image_bytes(format="TIFF", color=(150, 150, 150))

    opt_id = client.post("/api/uploads/file", files={"file": ("opt.tif", opt_bytes, "image/tiff")}).json()["metadata"]["id"]
    sar_id = client.post("/api/uploads/file", files={"file": ("sar.tif", sar_bytes, "image/tiff")}).json()["metadata"]["id"]

    res = client.post(
        "/api/analyze/optical-sar",
        json={"optical_image_id": opt_id, "sar_image_id": sar_id, "query": "Analyze water body penetrating cloud cover"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert data["confidence"] > 0.8


# 6. Agentic Orchestrator
def test_agent_orchestrator():
    img_bytes = _make_dummy_image_bytes(format="TIFF")
    img_id = client.post("/api/uploads/file", files={"file": ("single.tif", img_bytes, "image/tiff")}).json()["metadata"]["id"]

    res = client.post(
        "/api/analyze",
        json={"query": "What type of land cover is visible here?", "primary_image_id": img_id}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert len(data["execution_trace"]) >= 1


# 7. Satellite Catalog Search
def test_satellite_search():
    res = client.post(
        "/api/satellite/search",
        json={
            "location_name": "Chennai, India",
            "start_date": "2024-01-01",
            "end_date": "2026-01-01",
            "sensors": ["Sentinel-2"],
            "cloud_coverage": 20.0
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 1
    assert len(data["scenes"]) >= 1
    assert data["aoi_area_sq_km"] > 0


# 8. Autonomous Earth Observation Location Query
def test_satellite_auto_query():
    res = client.post(
        "/api/satellite/auto-query",
        json={
            "query": "Compare Chennai between January 2024 and January 2026",
            "target_location": "Chennai",
            "max_cloud_coverage": 20.0
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert "Chennai" in data["location_name"]
    assert data["before_scene"] is not None
    assert data["after_scene"] is not None
    assert data["confidence"] >= 0.8
