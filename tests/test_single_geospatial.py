"""
Comprehensive Test Suite for Single-Image Geospatial Multimodal Analysis.
Covers:
1. RGB GeoTIFF upload
2. Sentinel-2 multispectral GeoTIFF (B02, B03, B04, B08, B11, B12)
3. Single-band optical TIFF
4. Sentinel-1 VV SAR TIFF
5. Sentinel-1 VV + VH SAR TIFF
6. Invalid TIFF handling
7. Missing CRS detection
8. NDVI calculation on B08 + B04
9. NDWI calculation on B03 + B08
10. NDBI calculation on B11 + B08
11. RGB generation from B04/B03/B02
12. SAR dB visualization & VV/VH processing
13. Pixel -> Geographic coordinate conversion
14. Vector grounding polygon generation & area in hectares
15. Single-image VQA
16. Grounded scene captioning
17. MultispectralBuilder utility
"""
import io
import os
import pytest
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from preprocessing import (
    RasterMetadataService,
    ImageValidator,
    BandService,
    SARPreprocessor,
    MultispectralBuilder,
    RasterClassifier,
    RasterType,
    BandRole,
)
from models.single_image.geospatial_grounding import GeospatialGroundingEngine
from models.single_image import model_manager, SingleImageTaskEnum

client = TestClient(app)


def _create_test_geotiff(
    count: int = 3,
    dtype: str = "uint8",
    width: int = 128,
    height: int = 128,
    crs: str = "EPSG:4326",
    descriptions: list = None,
    bounds: tuple = (77.50, 11.20, 77.60, 11.30)
) -> bytes:
    """Helper to create valid in-memory GeoTIFFs with real geotransforms and CRS."""
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    mem_file = io.BytesIO()

    with rasterio.open(
        mem_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(1, count + 1):
            if dtype == "uint16":
                data = (np.ones((height, width), dtype=np.uint16) * (1000 * i)).astype(np.uint16)
            elif dtype == "float32":
                data = (np.ones((height, width), dtype=np.float32) * (0.2 * i)).astype(np.float32)
            else:
                data = (np.ones((height, width), dtype=np.uint8) * (50 * i)).astype(np.uint8)
            dst.write(data, i)
            if descriptions and i - 1 < len(descriptions):
                dst.set_band_description(i, descriptions[i - 1])

    return mem_file.getvalue()


# 1. RGB GeoTIFF Upload & Metadata Extraction
def test_rgb_geotiff_upload_and_metadata():
    tif_bytes = _create_test_geotiff(count=3, descriptions=["Red", "Green", "Blue"])
    res = client.post(
        "/api/single/upload",
        files={"file": ("rgb_scene.tif", tif_bytes, "image/tiff")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    meta = data["metadata"]
    assert meta["bands"] == 3
    assert meta["raster_type"] == "OPTICAL RGB"
    assert "EPSG:4326" in (meta["crs"] or "")


# 2. Sentinel-2 Multispectral GeoTIFF (B02, B03, B04, B08, B11, B12)
def test_sentinel2_multispectral_geotiff():
    s2_bands = ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR", "B11 SWIR1", "B12 SWIR2"]
    tif_bytes = _create_test_geotiff(count=6, dtype="uint16", descriptions=s2_bands)
    
    meta = RasterMetadataService.extract_metadata(tif_bytes, filename="sentinel2_l2a.tif")
    assert meta["bands"] == 6
    assert meta["raster_type"] == "OPTICAL MULTISPECTRAL"
    assert "Sentinel-2" in meta["sensor"]


# 3. Single-band Optical TIFF
def test_single_band_optical_tiff():
    tif_bytes = _create_test_geotiff(count=1, dtype="uint8", descriptions=["Panchromatic"])
    meta = RasterMetadataService.extract_metadata(tif_bytes, filename="panchromatic.tif")
    assert meta["bands"] == 1
    assert meta["raster_type"] == "OPTICAL SINGLE-BAND"


# 4. Sentinel-1 VV SAR TIFF
def test_sentinel1_vv_sar_tiff():
    tif_bytes = _create_test_geotiff(count=1, dtype="float32", descriptions=["VV (Co-Pol)"])
    meta = RasterMetadataService.extract_metadata(tif_bytes, filename="s1_vv_grd.tif")
    assert meta["bands"] == 1
    assert meta["raster_type"] == "SAR"
    assert "Sentinel-1" in meta["sensor"]


# 5. Sentinel-1 VV + VH SAR TIFF
def test_sentinel1_vv_vh_sar_tiff():
    tif_bytes = _create_test_geotiff(count=2, dtype="float32", descriptions=["VV", "VH"])
    meta = RasterMetadataService.extract_metadata(tif_bytes, filename="sentinel1_dual_pol.tif")
    assert meta["bands"] == 2
    assert meta["raster_type"] == "SAR"
    assert "VV" in meta["type_metadata"]["polarizations"]
    assert "VH" in meta["type_metadata"]["polarizations"]


# 6. Invalid TIFF Rejection
def test_invalid_tiff_rejection():
    res = client.post(
        "/api/single/upload",
        files={"file": ("corrupt.tif", b"corrupt_file_header_not_a_valid_tiff", "image/tiff")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is False


# 7. Missing CRS Detection
def test_missing_crs_handling():
    # Construct raster without CRS
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8"
    ) as dst:
        dst.write(np.ones((64, 64), dtype=np.uint8) * 100, 1)

    meta = RasterMetadataService.extract_metadata(mem_file.getvalue(), filename="no_crs.tif")
    assert meta["crs"] is None


# 8. NDVI Calculation on B08 + B04
def test_ndvi_calculation():
    # Band 1 = B04 Red (low reflectance = 500), Band 2 = B08 NIR (high reflectance = 4000)
    transform = from_bounds(77.5, 11.2, 77.6, 11.3, 64, 64)
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, "w", driver="GTiff", height=64, width=64, count=2, dtype="uint16",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(np.ones((64, 64), dtype=np.uint16) * 500, 1)
        dst.set_band_description(1, "B04 Red")
        dst.write(np.ones((64, 64), dtype=np.uint16) * 4000, 2)
        dst.set_band_description(2, "B08 NIR")

    indices = BandService.compute_spectral_indices(mem_file.getvalue())
    assert "NDVI" in indices["available_indices"]
    ndvi_mean = indices["statistics"]["ndvi"]["mean"]
    # (4000 - 500) / (4000 + 500) = 3500 / 4500 = 0.7778
    assert round(ndvi_mean, 2) == 0.78
    assert indices["statistics"]["ndvi"]["vegetated_pct"] == 100.0


# 9. NDWI Calculation on B03 + B08
def test_ndwi_calculation():
    # Band 1 = B03 Green (1500), Band 2 = B08 NIR (500) -> NDWI > 0 (Water)
    transform = from_bounds(77.5, 11.2, 77.6, 11.3, 64, 64)
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, "w", driver="GTiff", height=64, width=64, count=2, dtype="uint16",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(np.ones((64, 64), dtype=np.uint16) * 1500, 1)
        dst.set_band_description(1, "B03 Green")
        dst.write(np.ones((64, 64), dtype=np.uint16) * 500, 2)
        dst.set_band_description(2, "B08 NIR")

    indices = BandService.compute_spectral_indices(mem_file.getvalue())
    assert "NDWI" in indices["available_indices"]
    ndwi_mean = indices["statistics"]["ndwi"]["mean"]
    # (1500 - 500) / (1500 + 500) = 1000 / 2000 = 0.50
    assert round(ndwi_mean, 2) == 0.50
    assert indices["statistics"]["ndwi"]["water_body_pct"] == 100.0


# 10. NDBI Calculation on B11 + B08
def test_ndbi_calculation():
    # Band 1 = B08 NIR (1000), Band 2 = B11 SWIR (2500) -> NDBI > 0 (Built-Up)
    transform = from_bounds(77.5, 11.2, 77.6, 11.3, 64, 64)
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, "w", driver="GTiff", height=64, width=64, count=2, dtype="uint16",
        crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(np.ones((64, 64), dtype=np.uint16) * 1000, 1)
        dst.set_band_description(1, "B08 NIR")
        dst.write(np.ones((64, 64), dtype=np.uint16) * 2500, 2)
        dst.set_band_description(2, "B11 SWIR")

    indices = BandService.compute_spectral_indices(mem_file.getvalue())
    assert "NDBI" in indices["available_indices"]
    ndbi_mean = indices["statistics"]["ndbi"]["mean"]
    # (2500 - 1000) / (2500 + 1000) = 1500 / 3500 = 0.4286
    assert round(ndbi_mean, 2) == 0.43


# 11. RGB Generation from B04/B03/B02 Without False Mapping
def test_safe_true_color_rgb_generation():
    # A 4-band GeoTIFF with B04, B08, B11, B12 (No B02/B03)
    tif_bytes = _create_test_geotiff(
        count=4, dtype="uint16", descriptions=["B04 Red", "B08 NIR", "B11 SWIR1", "B12 SWIR2"]
    )
    img, url, desc = BandService.create_true_color_rgb(tif_bytes, filename="b04_b08_b11_b12.tif")
    # Verify it does NOT falsely map B08 to Green and B11 to Blue
    assert "NIR-Red" in desc or "Panchromatic" in desc or "False Color" in desc
    assert img.mode == "RGB"


# 12. SAR Visualization and Decibel Calibration
def test_sar_decibel_visualization():
    # Single band SAR with raw linear values
    tif_bytes = _create_test_geotiff(count=1, dtype="float32", descriptions=["VV"])
    img, url = SARPreprocessor.create_sar_preview(tif_bytes)
    assert url.startswith("data:image/png;base64,")


# 13. Pixel -> Geographic Coordinate Conversion
def test_pixel_to_geo_coordinate_conversion():
    tif_bytes = _create_test_geotiff(
        count=3, bounds=(77.5300, 11.2300, 77.6350, 11.3200), width=100, height=100
    )
    regions = GeospatialGroundingEngine.ground_query_on_raster(
        tif_bytes, query="Find potentially vacant land"
    )
    assert len(regions) > 0
    first = regions[0]
    # Verify coordinates are in geographic WGS84 range near Perundurai
    assert 77.50 <= first.centroid[0] <= 77.65
    assert 11.20 <= first.centroid[1] <= 11.35
    assert first.area_ha > 0.0


# 14. Vector Grounding Polygon Extraction & Area in Hectares
def test_vector_grounding_polygon_extraction():
    tif_bytes = _create_test_geotiff(
        count=4, dtype="uint16", descriptions=["B02", "B03", "B04", "B08"],
        bounds=(77.50, 11.20, 77.60, 11.30), width=200, height=200
    )
    regions = GeospatialGroundingEngine.ground_query_on_raster(
        tif_bytes, query="Show dense vegetation"
    )
    assert len(regions) >= 1
    reg = regions[0]
    assert reg.polygon["type"] == "Polygon"
    assert len(reg.polygon["coordinates"][0]) >= 4
    assert reg.area_m2 > 0


# 15. Single-Image VQA
def test_single_image_vqa_endpoint():
    tif_bytes = _create_test_geotiff(count=3, descriptions=["Red", "Green", "Blue"])
    up_res = client.post("/api/single/upload", files={"file": ("scene.tif", tif_bytes, "image/tiff")})
    image_id = up_res.json()["metadata"]["id"]

    res = client.post(
        "/api/analyze/single",
        json={"image_id": image_id, "query": "Is there vegetation in this area?", "task": "vqa"}
    )
    assert res.status_code == 200
    vqa_data = res.json()
    assert vqa_data["answer"] != ""
    assert vqa_data["confidence"] > 0.8


# 16. Grounded Scene Captioning
def test_single_image_captioning_endpoint():
    tif_bytes = _create_test_geotiff(count=3, descriptions=["Red", "Green", "Blue"])
    up_res = client.post("/api/single/upload", files={"file": ("scene.tif", tif_bytes, "image/tiff")})
    image_id = up_res.json()["metadata"]["id"]

    res = client.post(
        "/api/analyze/single",
        json={"image_id": image_id, "query": "Describe this image", "task": "captioning"}
    )
    assert res.status_code == 200
    cap_data = res.json()
    assert "optical" in cap_data["answer"].lower() or "satellite" in cap_data["answer"].lower()


# 17. MultispectralBuilder Utility
def test_multispectral_builder_utility():
    b02 = _create_test_geotiff(count=1, dtype="uint16", width=64, height=64, descriptions=["B02"])
    b03 = _create_test_geotiff(count=1, dtype="uint16", width=64, height=64, descriptions=["B03"])
    b04 = _create_test_geotiff(count=1, dtype="uint16", width=64, height=64, descriptions=["B04"])
    b08 = _create_test_geotiff(count=1, dtype="uint16", width=64, height=64, descriptions=["B08"])
    # 20m SWIR bands (32x32)
    b11 = _create_test_geotiff(count=1, dtype="uint16", width=32, height=32, descriptions=["B11"])

    combined_bytes, summary = MultispectralBuilder.build_multiband_geotiff({
        "B02": b02,
        "B03": b03,
        "B04": b04,
        "B08": b08,
        "B11": b11,
    })

    assert summary["bands_count"] == 5
    assert summary["width"] == 64
    assert summary["height"] == 64
    assert len(combined_bytes) > 0
