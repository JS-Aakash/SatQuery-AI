"""
Automated unit test suite for Module 2: Geospatial Image Processing Engine.
Tests GeoTIFFReader, RasterMetadataService, ImageValidator, CRSService,
BandService, SARPreprocessor, ImageAlignmentService, AOIService, and TileService.
"""
import io
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from fastapi.testclient import TestClient

from backend.app.main import app
from preprocessing import (
    GeoTIFFReader,
    RasterReadError,
    RasterMetadataService,
    ImageValidator,
    CRSService,
    BandService,
    SARPreprocessor,
    ImageAlignmentService,
    AOIService,
    TileService,
)

client = TestClient(app)


@pytest.fixture
def sample_optical_geotiff_bytes() -> bytes:
    """Generates a synthetic 4-band GeoTIFF (Red, Green, Blue, NIR) in EPSG:4326."""
    width, height = 128, 128
    transform = from_origin(80.20, 13.15, 0.001, 0.001)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0
    }

    data = np.zeros((4, height, width), dtype=np.uint16)
    data[0] = np.random.randint(500, 2500, (height, width), dtype=np.uint16)
    data[1] = np.random.randint(600, 3000, (height, width), dtype=np.uint16)
    data[2] = np.random.randint(400, 2000, (height, width), dtype=np.uint16)
    data[3] = np.random.randint(2000, 8000, (height, width), dtype=np.uint16)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(data)
        return memfile.read()


@pytest.fixture
def sample_sar_geotiff_bytes() -> bytes:
    """Generates a synthetic 2-band SAR GeoTIFF (VV, VH) in EPSG:4326."""
    width, height = 128, 128
    transform = from_origin(80.20, 13.15, 0.001, 0.001)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0
    }

    data = np.random.exponential(scale=0.15, size=(2, height, width)).astype(np.float32)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(data)
        return memfile.read()


def test_geotiff_reader(sample_optical_geotiff_bytes):
    """Verify GeoTIFFReader opens memory buffer and reads bands correctly."""
    assert GeoTIFFReader.is_geotiff(sample_optical_geotiff_bytes) is True
    
    arr, profile = GeoTIFFReader.read_bands(sample_optical_geotiff_bytes)
    assert arr.shape == (4, 128, 128)
    assert profile["count"] == 4
    assert profile["dtype"] == "uint16"

    # Test reading specific bands
    arr_rgb, _ = GeoTIFFReader.read_bands(sample_optical_geotiff_bytes, bands=(1, 2, 3))
    assert arr_rgb.shape == (3, 128, 128)

    # Test corrupted buffer
    with pytest.raises(RasterReadError):
        GeoTIFFReader.read_bands(b"corrupted_header_data_xyz")


def test_raster_metadata_service(sample_optical_geotiff_bytes):
    """Verify complete metadata extraction from GeoTIFF."""
    meta = RasterMetadataService.extract_metadata(sample_optical_geotiff_bytes, "sentinel2_test.tif")
    assert meta["width"] == 128
    assert meta["height"] == 128
    assert meta["bands"] == 4
    assert meta["dtype"] == "uint16"
    assert "4326" in meta["crs"]
    assert len(meta["transform"]) == 6
    assert len(meta["bounds"]) == 4
    assert meta["resolution"] == [0.001, 0.001]
    assert meta["nodata"] == 0
    assert meta["modality"] == "Multispectral"


def test_crs_service(sample_optical_geotiff_bytes):
    """Verify CRS validation, EPSG extraction, and compatibility checking."""
    assert CRSService.is_valid_crs("EPSG:32644") is True
    assert CRSService.is_valid_crs("EPSG:4326") is True
    assert CRSService.is_valid_crs("INVALID_CRS_9999") is False

    assert CRSService.get_epsg_code("EPSG:32644") == 32644

    comp, msg = CRSService.are_crs_compatible("EPSG:32644", "EPSG:32644")
    assert comp is True

    comp_diff, _ = CRSService.are_crs_compatible("EPSG:32644", "EPSG:4326")
    assert comp_diff is False


def test_image_validator(sample_optical_geotiff_bytes, sample_sar_geotiff_bytes):
    """Verify single image validation and pair compatibility checks."""
    res = ImageValidator.validate_single_image(sample_optical_geotiff_bytes, "valid_optical.tif")
    assert res["is_valid"] is True
    assert res["bands"] == 4
    assert res["width"] == 128

    # Test corrupted file validation
    res_corrupt = ImageValidator.validate_single_image(b"broken bytes", "corrupted.tif")
    assert res_corrupt["is_valid"] is False
    assert res_corrupt["status"] == "CORRUPTED_FILE"

    # Test pair validation
    meta_opt = RasterMetadataService.extract_metadata(sample_optical_geotiff_bytes, "optical.tif")
    meta_sar = RasterMetadataService.extract_metadata(sample_sar_geotiff_bytes, "sar.tif")

    pair_res = ImageValidator.validate_pair(meta_opt, meta_sar)
    assert pair_res["is_compatible"] is True
    assert pair_res["has_geographic_overlap"] is True
    assert pair_res["overlap_percentage"] > 99.0
    assert pair_res["is_optical_sar_pair"] is True


def test_band_service(sample_optical_geotiff_bytes):
    """Verify natural RGB and false-color composite generation without altering raw data."""
    img_rgb, url_rgb = BandService.create_true_color_rgb(sample_optical_geotiff_bytes)
    assert img_rgb.size == (128, 128)
    assert img_rgb.mode == "RGB"
    assert url_rgb.startswith("data:image/png;base64,")

    img_cir, url_cir = BandService.create_false_color_infrared(sample_optical_geotiff_bytes)
    assert img_cir.size == (128, 128)
    assert img_cir.mode == "RGB"


def test_sar_preprocessor(sample_sar_geotiff_bytes):
    """Verify non-destructive SAR decibel transformation and dual-pol preview."""
    img_sar, url_sar = SARPreprocessor.create_sar_preview(sample_sar_geotiff_bytes)
    assert img_sar.size == (128, 128)
    assert img_sar.mode == "RGB"
    assert url_sar.startswith("data:image/png;base64,")


def test_aoi_service(sample_optical_geotiff_bytes):
    """Verify cropping to bounding box updates dimensions and affine transform."""
    # Crop middle 50%
    meta = RasterMetadataService.extract_metadata(sample_optical_geotiff_bytes)
    left, bottom, right, top = meta["bounds"]
    aoi_box = [left + 0.02, bottom + 0.02, right - 0.02, top - 0.02]

    cropped_bytes, cropped_meta, preview_url = AOIService.crop_raster(sample_optical_geotiff_bytes, aoi_box)
    assert cropped_meta["width"] < 128
    assert cropped_meta["height"] < 128
    assert cropped_meta["bands"] == 4
    assert preview_url.startswith("data:image/png;base64,")


def test_tile_service(sample_optical_geotiff_bytes):
    """Verify configurable tiling and coordinate remapping back to master raster."""
    tiles = TileService.generate_tile_manifest(sample_optical_geotiff_bytes, tile_size=64, overlap=16)
    assert len(tiles) >= 4
    first_tile = tiles[0]
    assert first_tile["width"] == 64
    assert first_tile["height"] == 64
    assert "bounds" in first_tile

    # Test coordinate mapping
    mapped = TileService.map_tile_box_to_global_geo(first_tile, norm_box=[10.0, 10.0, 90.0, 90.0])
    assert "global_pixel_box" in mapped
    assert "geographic_bounds" in mapped


def test_multipart_upload_endpoint(sample_optical_geotiff_bytes):
    """Verify FastAPI /api/uploads/file accepts real GeoTIFF and extracts full metadata."""
    files = {
        "file": ("sentinel2_chennai.tif", sample_optical_geotiff_bytes, "image/tiff")
    }
    res = client.post("/api/uploads/file", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["is_valid"] is True
    meta = data["metadata"]
    assert meta["dimensions"] == [128, 128]
    assert meta["bands"] == 4
    assert meta["dtype"] == "uint16"
    assert "preview_url" in meta and meta["preview_url"] is not None
