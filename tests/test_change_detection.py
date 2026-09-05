"""
Unit and Integration Tests for Module 4: Bi-Temporal Change Intelligence.
Tests spatial alignment, spectral indices calculation (NDVI, NDWI, NDBI),
polygonization, surface area estimation, and the POST /api/analyze/change endpoint.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from models.change_detection import (
    change_manager,
    SpectralIndexCalculator,
    PolygonizerService,
    ChangeDetectionPipeline
)
from preprocessing import ImageAlignmentService

client = TestClient(app)


def test_spectral_indices_calculation():
    """Verify NDVI, NDWI, and NDBI delta computations."""
    # Synthetic 4-channel array (H, W, 4): R, G, B, NIR
    t1 = np.zeros((100, 100, 4), dtype=np.uint8)
    t1[:, :, 0] = 40   # Low Red
    t1[:, :, 1] = 60   # Medium Green
    t1[:, :, 2] = 30   # Low Blue
    t1[:, :, 3] = 200  # High NIR (Dense Vegetation in T1)

    t2 = np.zeros((100, 100, 4), dtype=np.uint8)
    t2[:, :, 0] = 180  # High Red (Impervious / Concrete in T2)
    t2[:, :, 1] = 170  # High Green
    t2[:, :, 2] = 160  # High Blue
    t2[:, :, 3] = 50   # Low NIR (Vegetation Loss in T2)

    res = SpectralIndexCalculator.compute_bitemporal_indices(t1, t2)
    assert res.mean_ndvi_t1 > 0.4, "T1 should show healthy vegetation NDVI"
    assert res.mean_ndvi_t2 < 0.0, "T2 should show cleared/urbanized NDVI"
    assert res.delta_ndvi < -0.3, "Delta NDVI should be significantly negative"
    assert len(res.supporting_indicators) > 0


def test_polygonizer_area_estimation():
    """Verify polygon extraction and hectare/km^2 area calculation."""
    # 200x200 mask with a 40x50 block of Class 1 change
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:60, 30:80] = 1  # 40 * 50 = 2000 pixels

    # 10m GSD -> 1 pixel = 100 m^2 -> 2000 px = 200,000 m^2 = 20 ha = 0.2 km^2
    polygons = PolygonizerService.extract_polygons(mask, gsd_meters=10.0)
    assert len(polygons) == 1
    poly = polygons[0]
    assert poly.area_m2 == 200000.0
    assert poly.area_hectares == 20.0
    assert poly.area_km2 == 0.2
    assert poly.bounding_box[0] == 10.0  # ymin: 20/200 = 10%
    assert poly.bounding_box[1] == 15.0  # xmin: 30/200 = 15%


def test_change_detection_pipeline_execution():
    """Verify full bi-temporal change pipeline end-to-end."""
    img_t1 = np.ones((300, 300, 3), dtype=np.uint8) * 80
    img_t2 = np.ones((300, 300, 3), dtype=np.uint8) * 80
    # Add localized difference block
    img_t2[50:150, 50:150, 0] = 240

    res = change_manager.infer(
        image_t1=img_t1,
        image_t2=img_t2,
        query="What changed between these observations?"
    )

    assert res.task == "Bi-Temporal Change Analysis"
    assert res.confidence >= 0.85
    assert res.total_changed_area_km2 > 0.0
    assert res.total_changed_area_hectares > 0.0
    assert res.change_mask_url is not None
    assert "data:image/png;base64" in res.change_mask_url
    assert len(res.changed_regions) > 0
    assert res.spectral_indices is not None


def test_change_analysis_api_endpoint():
    """Verify POST /api/analyze/change endpoint."""
    payload = {
        "query": "What changed between these two images?",
        "parameters": {"threshold": 0.2}
    }
    response = client.post("/api/analyze/change", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "total_changed_area_km2" in data
    assert "total_changed_area_hectares" in data
    assert "change_mask_url" in data
    assert data["model_status"] == "READY"
    assert len(data["changed_regions"]) > 0
