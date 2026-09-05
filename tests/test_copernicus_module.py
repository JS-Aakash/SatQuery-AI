"""
Comprehensive Unit & Integration Test Suite for Copernicus Data Space & Sentinel Hub Analysis Module.
Covers geocoding (Perundurai, Erode, Coimbatore), STAC catalog search, cloud filtering,
Sentinel Hub Processing API (RGB, False Color, NDVI, NDWI, NDBI), authentic pixel statistics,
bi-temporal change detection, and FastAPI REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from services.copernicus import (
    LocationGeocodingService,
    CopernicusAuthService,
    CopernicusCatalogService,
    SentinelHubProcessingService,
    CopernicusChangeService,
    AOIGeometry,
    CopernicusSearchRequest,
    ProcessAnalysisRequest,
    TemporalChangeRequest,
    SpectralIndexType,
    SatelliteCollection,
    copernicus_auth,
    copernicus_processing,
    copernicus_change,
)

client = TestClient(app)


def _make_perundurai_aoi():
    coords = [
        [77.5300, 11.2300],
        [77.6350, 11.2300],
        [77.6350, 11.3200],
        [77.5300, 11.3200],
        [77.5300, 11.2300],
    ]
    return AOIGeometry(
        type="Polygon",
        coordinates=[coords],
        bbox=[77.5300, 11.2300, 77.6350, 11.3200]
    )


# 1. Location Geocoding Tests (Perundurai, Erode, Coimbatore, Coordinates)
def test_location_geocoding_perundurai():
    res = LocationGeocodingService.geocode("Perundurai")
    assert res is not None
    assert "Perundurai" in res.name
    assert abs(res.lat - 11.2750) < 0.05
    assert abs(res.lon - 77.5833) < 0.05
    assert len(res.bbox) == 4


def test_location_geocoding_erode_and_coimbatore():
    erode = LocationGeocodingService.geocode("Erode")
    assert erode is not None
    assert "Erode" in erode.name

    cbe = LocationGeocodingService.geocode("Coimbatore")
    assert cbe is not None
    assert "Coimbatore" in cbe.name


def test_aoi_area_calculation():
    aoi = _make_perundurai_aoi()
    area = LocationGeocodingService.calculate_aoi_area_sq_km(aoi)
    # 0.1 deg x 0.09 deg box in Tamil Nadu is ~100-120 km2
    assert area > 1.0


# 2. Authentication Service Tests
def test_copernicus_auth_service():
    auth = CopernicusAuthService(
        client_id="sh-835fcc09-d1ef-4d0e-9d47-aee40ed34519",
        client_secret="LdVestH2nziKhwwuNaDux2Nfw2Zz5cS5"
    )
    assert auth.has_credentials is True
    assert auth.client_id == "sh-835fcc09-d1ef-4d0e-9d47-aee40ed34519"
    # Test token retrieval (handles live token or graceful fallback error)
    token, err = auth.get_token()
    if token:
        assert isinstance(token, str)
    else:
        assert err is not None
        assert "COPERNICUS_CLIENT_SECRET" not in err  # Ensure secrets are never leaked


# 3. STAC Catalog Discovery & Cloud Filtering Tests
def test_copernicus_catalog_search():
    catalog = CopernicusCatalogService()
    aoi = _make_perundurai_aoi()
    req = CopernicusSearchRequest(
        aoi=aoi,
        start_date="2026-08-01",
        end_date="2026-08-31",
        max_cloud_coverage=20.0,
        collection=SatelliteCollection.SENTINEL_2_L2A,
        limit=8
    )
    res = catalog.search(req)
    assert res.total_count >= 1
    assert len(res.observations) >= 1
    assert res.aoi_area_sq_km > 0
    for obs in res.observations:
        assert obs.cloud_coverage_percent <= 20.0
        assert "Sentinel" in obs.platform


# 4. Processing API & Spectral Indices (RGB, False Color, NDVI, NDWI, NDBI)
def test_processing_ndvi_and_statistics():
    aoi = _make_perundurai_aoi()
    req = ProcessAnalysisRequest(
        aoi=aoi,
        observation_id="S2B_MSIL2A_20260815T050931_N0510_R019_T43QEF_20260815T081520",
        analysis_type=SpectralIndexType.NDVI,
        resolution_m=10.0
    )
    res = copernicus_processing.process(req)
    assert res.status == "COMPLETED"
    assert res.image_url.startswith("data:image/png;base64,")
    assert res.resolution_m == 10.0
    assert res.statistics is not None
    assert -1.0 <= res.statistics.mean <= 1.0
    assert -1.0 <= res.statistics.min <= 1.0
    assert -1.0 <= res.statistics.max <= 1.0
    assert res.statistics.std_dev >= 0.0
    assert len(res.statistics.class_breakdown) >= 2


def test_processing_ndwi_and_ndbi():
    aoi = _make_perundurai_aoi()
    
    # NDWI
    req_ndwi = ProcessAnalysisRequest(
        aoi=aoi,
        observation_id="S2B_MSIL2A_20260815T050931_N0510_R019_T43QEF_20260815T081520",
        analysis_type=SpectralIndexType.NDWI,
    )
    res_ndwi = copernicus_processing.process(req_ndwi)
    assert res_ndwi.statistics is not None
    assert "Open Water Surface (NDWI > 0.0)" in res_ndwi.statistics.class_breakdown

    # NDBI
    req_ndbi = ProcessAnalysisRequest(
        aoi=aoi,
        observation_id="S2B_MSIL2A_20260815T050931_N0510_R019_T43QEF_20260815T081520",
        analysis_type=SpectralIndexType.NDBI,
    )
    res_ndbi = copernicus_processing.process(req_ndbi)
    assert res_ndbi.statistics is not None
    assert "Built-up / Concrete / Asphalt (NDBI > 0.10)" in res_ndbi.statistics.class_breakdown


# 5. Bi-Temporal Change Detection Tests
def test_temporal_change_detection():
    aoi = _make_perundurai_aoi()
    req = TemporalChangeRequest(
        aoi=aoi,
        before_observation_id="S2B_MSIL2A_20240115T050931_N0510_R019_T43QEF_20240115T081520",
        after_observation_id="S2B_MSIL2A_20260820T050931_N0510_R019_T43QEF_20260820T081520",
        analysis_type=SpectralIndexType.NDVI
    )
    res = copernicus_change.compute_change(req)
    assert res.status == "COMPLETED"
    assert res.diff_image_url.startswith("data:image/png;base64,")
    assert res.improved_percent + res.stable_percent + res.declined_percent > 90.0
    assert "delta_mean" in res.change_statistics


# 6. REST API Endpoints Tests
def test_api_geocode_endpoint():
    res = client.post("/api/copernicus/geocode", json={"query": "Perundurai"})
    assert res.status_code == 200
    data = res.json()
    assert "Perundurai" in data["name"]
    assert len(data["bbox"]) == 4


def test_api_search_endpoint():
    aoi_payload = {
        "type": "Polygon",
        "coordinates": [
            [
                [77.5300, 11.2300],
                [77.6350, 11.2300],
                [77.6350, 11.3200],
                [77.5300, 11.3200],
                [77.5300, 11.2300]
            ]
        ]
    }
    req = {
        "aoi": aoi_payload,
        "start_date": "2026-08-01",
        "end_date": "2026-08-31",
        "max_cloud_coverage": 20.0,
        "collection": "sentinel-2-l2a"
    }
    res = client.post("/api/copernicus/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] >= 1
    assert len(data["observations"]) >= 1


def test_api_process_and_ndvi_endpoints():
    aoi_payload = {
        "type": "Polygon",
        "coordinates": [
            [
                [77.5300, 11.2300],
                [77.6350, 11.2300],
                [77.6350, 11.3200],
                [77.5300, 11.3200],
                [77.5300, 11.2300]
            ]
        ]
    }
    req = {
        "aoi": aoi_payload,
        "observation_id": "S2B_MSIL2A_20260815T050931_N0510_R019_T43QEF_20260815T081520",
        "analysis_type": "NDVI"
    }
    res = client.post("/api/copernicus/ndvi", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert "statistics" in data
    assert data["statistics"]["mean"] is not None


def test_api_ai_analyze_endpoint():
    req = {
        "query": "Evaluate crop density and water availability around Perundurai.",
        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "analysis_type": "NDVI",
        "statistics": {"mean": 0.62, "min": 0.12, "max": 0.85}
    }
    res = client.post("/api/copernicus/ai-analyze", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "ai_explanation" in data
    assert data["status"] == "COMPLETED"
