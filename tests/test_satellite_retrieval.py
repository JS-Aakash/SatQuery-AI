"""
Tests for Module 7: Automatic Earth Observation Data Retrieval.
Covers geocoding, multi-criteria scene selection, caching, Copernicus provider, and the automated pipeline.
"""

import pytest
import os
import shutil
from unittest.mock import patch, MagicMock

from satellite.schemas import (
    SatelliteProduct,
    CatalogSearchQuery,
    SatelliteSensor,
    AutoLocationQueryRequest,
    AutoLocationQueryResponse,
)
from satellite.geocoding import GeocodingService
from satellite.selection.selector import SceneSelector
from satellite.cache.cache import ImageryCache
from satellite.providers.copernicus import CopernicusDataSpaceProvider
from satellite.catalog.catalog import CatalogSearchEngine
from satellite.pipeline import AutomaticEarthObservationPipeline


# 1. Geocoding & Query Parsing Tests
def test_geocoding_predefined_and_fallback():
    # Predefined known AOI
    chennai_name, chennai_box = GeocodingService.resolve_location("Chennai")
    assert "Chennai" in chennai_name
    assert chennai_box == [80.18, 13.00, 80.35, 13.25]
    
    # Other cities
    mumbai_name, mumbai_box = GeocodingService.resolve_location("Mumbai")
    assert "Mumbai" in mumbai_name
    assert len(mumbai_box) == 4

    # Unknown location fallback
    unknown_name, unknown_box = GeocodingService.resolve_location("NonExistentCity12345")
    assert "NonExistentCity12345" in unknown_name
    assert len(unknown_box) == 4


def test_nlp_temporal_query_parsing():
    # "Compare Chennai between January 2024 and January 2026"
    parsed = GeocodingService.parse_natural_language_query("Compare Chennai between January 2024 and January 2026")
    assert "Chennai" in parsed["location_name"]
    assert "2024-01-15" in parsed["date_a"]
    assert "2026-01-15" in parsed["date_b"]
    assert parsed["sensor"] == "Sentinel-2 MSI"

    # Query asking for radar/SAR
    parsed_sar = GeocodingService.parse_natural_language_query("Detect flood changes in Kochi between 2024 and 2025 using SAR radar")
    assert "Kochi" in parsed_sar["location_name"]
    assert parsed_sar["is_sar"] is True
    assert parsed_sar["sensor"] == "Sentinel-1 SAR C-Band"


# 2. Multi-Criteria Scene Selection Tests
def test_scene_selector_ranking():
    # Candidate 1: High cloud cover
    p1 = SatelliteProduct(
        id="S2A_MSIL2A_20240110T050000_N0510_R019_T44VLR_20240110T080000",
        title="Sentinel-2 L2A Scene High Cloud",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        acquisition_date="2024-01-10T05:00:00Z",
        cloud_coverage_percentage=45.0,
        bbox=[80.12, 12.92, 80.35, 13.20],
        product_type="S2MSI2A",
    )
    # Candidate 2: Optimal low cloud cover and target date match (L2A BOA)
    p2 = SatelliteProduct(
        id="S2B_MSIL2A_20240115T050000_N0510_R019_T44VLR_20240115T080000",
        title="Sentinel-2 L2A Scene Clear Sky",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        acquisition_date="2024-01-15T05:00:00Z",
        cloud_coverage_percentage=2.4,
        bbox=[80.12, 12.92, 80.35, 13.20],
        product_type="S2MSI2A",
    )
    # Candidate 3: L1C (less preferred than L2A BOA)
    p3 = SatelliteProduct(
        id="S2B_MSIL1C_20240115T050000_N0510_R019_T44VLR_20240115T080000",
        title="Sentinel-2 L1C Scene TOA",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        acquisition_date="2024-01-15T05:00:00Z",
        cloud_coverage_percentage=2.0,
        bbox=[80.12, 12.92, 80.35, 13.20],
        product_type="S2MSI1C",
    )

    ranked = SceneSelector.rank_and_select([p1, p2, p3], target_date="2024-01-15")
    
    # p2 should rank highest due to low cloud + L2A BOA bonus
    assert len(ranked) == 3
    best_product = ranked[0]
    assert best_product.id == p2.id
    assert best_product.is_selected is True
    assert best_product.selection_score > 0.8
    assert "clear-sky" in best_product.selection_reason or "Bottom-of-Atmosphere" in best_product.selection_reason


# 3. Imagery Cache Tests
def test_imagery_cache(tmp_path):
    test_cache_dir = str(tmp_path / "sat_cache")
    cache = ImageryCache(cache_dir=test_cache_dir)
    
    product_id = "TEST_PROD_001"
    raw_bytes = b"GEOTIFF_TEST_DATA_BYTES_12345"
    meta = {"sensor": "Sentinel-2 MSI", "cloud": 1.2}

    # Initial state: not cached
    assert not cache.has(product_id)
    assert cache.get(product_id) is None

    # Store bytes
    saved_path = cache.put(product_id, raw_bytes, metadata=meta)
    assert os.path.exists(saved_path)
    assert cache.has(product_id)
    
    # Retrieve cached data
    cached_data, cached_meta = cache.get(product_id)
    assert cached_data == raw_bytes
    assert cached_meta["metadata"]["cloud"] == 1.2

    # Clear cache
    cache.clear()
    assert not cache.has(product_id)


# 4. Copernicus Provider Mock & Resilience Tests
def test_copernicus_provider_search():
    provider = CopernicusDataSpaceProvider()
    
    query = CatalogSearchQuery(
        bbox=[80.18, 13.00, 80.35, 13.25],
        start_date="2024-01-01",
        end_date="2024-01-31",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        max_cloud_coverage=20.0,
        limit=5,
    )
    
    products = provider.search(query)
    assert len(products) >= 1
    for prod in products:
        assert prod.sensor == SatelliteSensor.SENTINEL_2_OPTICAL
        assert prod.cloud_coverage_percentage <= 20.0


def test_catalog_engine_search():
    catalog = CatalogSearchEngine()
    query = CatalogSearchQuery(
        location_name="Chennai, India",
        bbox=[80.18, 13.00, 80.35, 13.25],
        start_date="2024-01-01",
        end_date="2024-01-31",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        max_cloud_coverage=20.0,
        limit=5,
    )
    res = catalog.search(query)
    assert res.total_found >= 1
    assert len(res.products) >= 1


def test_copernicus_provider_download():
    provider = CopernicusDataSpaceProvider()
    data = provider.download("S2A_MSIL2A_TEST_DOWNLOAD", bbox=[80.18, 13.00, 80.35, 13.25])
    assert isinstance(data, bytes)
    assert len(data) > 0


# 5. Automated Pipeline End-to-End Test
def test_automatic_pipeline_execution():
    pipeline = AutomaticEarthObservationPipeline()
    
    req = AutoLocationQueryRequest(
        query="Compare Chennai between January 2024 and January 2026",
        target_location="Chennai",
        sensor=SatelliteSensor.SENTINEL_2_OPTICAL,
        max_cloud_coverage=20.0,
    )
    
    res = pipeline.execute_auto_query(req)
    
    assert res.location_name is not None
    assert "Chennai" in res.location_name
    assert res.before_scene is not None
    assert res.after_scene is not None
    assert res.before_scene.id != ""
    assert res.after_scene.id != ""
    assert res.answer is not None
    assert res.confidence >= 0.8
    assert len(res.execution_trace) >= 5
    assert res.status == "COMPLETED"
