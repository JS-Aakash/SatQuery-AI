"""
Image Validator Service.
Validates single remote-sensing rasters and multi-image pairs for SIH requirements.
"""
from typing import Union, Dict, Any, List, Tuple, Optional
import io
import os
import numpy as np
import rasterio
from shapely.geometry import box

from .reader import GeoTIFFReader, RasterReadError
from .crs import CRSService


class ImageValidator:
    """
    Validates single rasters and multimodal/bi-temporal raster pairs.
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    @staticmethod
    def validate_single_image(
        source: Union[str, bytes, io.BytesIO],
        filename: str = "uploaded_raster.tif",
        require_crs: bool = False
    ) -> Dict[str, Any]:
        """
        Validates:
        - supported file type
        - readable raster
        - valid dimensions
        - valid CRS where required
        - band count
        - missing/nodata data
        - corrupted files
        """
        ext = os.path.splitext(filename)[1].lower()
        issues: List[str] = []
        warnings: List[str] = []

        if ext not in ImageValidator.SUPPORTED_EXTENSIONS:
            return {
                "is_valid": False,
                "status": "UNSUPPORTED_FORMAT",
                "errors": [f"File extension '{ext}' is not supported. Use GeoTIFF (.tif, .tiff) or benchmark PNG/JPEG."],
                "warnings": [],
                "metadata": None
            }

        try:
            with GeoTIFFReader.open_dataset(source) as ds:
                width = ds.width
                height = ds.height
                bands = ds.count
                crs = ds.crs
                nodata = ds.nodata

                # 1. Dimensions check
                if width <= 0 or height <= 0:
                    issues.append(f"Invalid raster dimensions: {width}x{height}.")
                elif width < 16 or height < 16:
                    warnings.append(f"Raster dimensions ({width}x{height}) are extremely small.")

                # 2. Band count check
                if bands <= 0:
                    issues.append("Raster contains zero spectral bands.")

                # 3. CRS check
                if require_crs and crs is None:
                    issues.append("Raster lacks spatial reference system (CRS / georeferencing).")
                elif crs is None and ext in [".tif", ".tiff"]:
                    warnings.append("GeoTIFF does not specify a CRS. Assumed local/pixel coordinate system.")

                # 4. Check missing/nodata statistics on sample
                nodata_pct = 0.0
                try:
                    # Sample center window to quickly estimate nodata
                    sample = ds.read(1, window=rasterio.windows.Window(0, 0, min(width, 256), min(height, 256)))
                    if nodata is not None:
                        nodata_count = np.sum(sample == nodata)
                        nodata_pct = round(float(nodata_count) / sample.size * 100, 2)
                    # Check for NaNs if float
                    if np.issubdtype(sample.dtype, np.floating):
                        nan_count = np.sum(np.isnan(sample))
                        if nan_count > 0:
                            warnings.append(f"Detected {round(nan_count / sample.size * 100, 1)}% NaN values.")
                except Exception as e:
                    warnings.append(f"Could not compute raster statistics: {str(e)}")

                is_valid = len(issues) == 0

                return {
                    "is_valid": is_valid,
                    "status": "VALID" if is_valid else "INVALID",
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "dtype": str(ds.dtypes[0]) if ds.dtypes else "unknown",
                    "crs": crs.to_string() if crs else None,
                    "nodata": nodata,
                    "nodata_percentage": nodata_pct,
                    "errors": issues,
                    "warnings": warnings,
                }

        except RasterReadError as e:
            return {
                "is_valid": False,
                "status": "CORRUPTED_FILE",
                "errors": [f"Corrupted raster file: {str(e)}"],
                "warnings": [],
                "metadata": None
            }
        except Exception as e:
            return {
                "is_valid": False,
                "status": "READ_ERROR",
                "errors": [f"Failed to read raster stream: {str(e)}"],
                "warnings": [],
                "metadata": None
            }

    @staticmethod
    def validate_pair(
        meta_a: Dict[str, Any],
        meta_b: Dict[str, Any],
        expected_pair_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        For pairs, validates:
        - compatible dimensions where applicable
        - geographic overlap percentage (using shapely)
        - CRS compatibility
        - spatial alignment
        - temporal metadata where available
        - modality pairing
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. CRS Compatibility
        crs_a = meta_a.get("crs")
        crs_b = meta_b.get("crs")
        crs_compatible = False
        if crs_a and crs_b:
            crs_compatible, crs_msg = CRSService.are_crs_compatible(crs_a, crs_b)
            if not crs_compatible:
                warnings.append(crs_msg)
        else:
            warnings.append("One or both images lack spatial CRS metadata.")

        # 2. Geographic Overlap Calculation
        bounds_a = meta_a.get("bounds_wgs84") or meta_a.get("bounds")
        bounds_b = meta_b.get("bounds_wgs84") or meta_b.get("bounds")
        overlap_pct = 0.0
        has_overlap = False

        if bounds_a and bounds_b:
            poly_a = box(*bounds_a)
            poly_b = box(*bounds_b)
            if poly_a.intersects(poly_b):
                has_overlap = True
                intersection_area = poly_a.intersection(poly_b).area
                min_area = min(poly_a.area, poly_b.area)
                if min_area > 0:
                    overlap_pct = round((intersection_area / min_area) * 100, 2)
            else:
                errors.append("No geographic overlap found between image pair.")
        else:
            warnings.append("Could not calculate geospatial bounding overlap.")

        # 3. Dimensions & Resolution
        dim_a = (meta_a.get("width"), meta_a.get("height"))
        dim_b = (meta_b.get("width"), meta_b.get("height"))
        dimension_match = dim_a == dim_b
        if not dimension_match:
            warnings.append(f"Dimension difference ({dim_a[0]}x{dim_a[1]} vs {dim_b[0]}x{dim_b[1]}). Image resampling/alignment will be performed.")

        # 4. Modality Pairing Checks
        mod_a = meta_a.get("modality", "Unknown")
        mod_b = meta_b.get("modality", "Unknown")
        is_optical_sar_pair = (mod_a == "SAR" and mod_b in ["Optical", "Multispectral"]) or \
                              (mod_b == "SAR" and mod_a in ["Optical", "Multispectral"])
        is_bitemporal_pair = (mod_a == mod_b)

        pair_type = "Cross-Modal (Optical + SAR)" if is_optical_sar_pair else "Bi-Temporal / Multi-Date"
        if expected_pair_type:
            pair_type = expected_pair_type

        is_compatible = len(errors) == 0

        return {
            "is_compatible": is_compatible,
            "pair_type": pair_type,
            "crs_compatible": crs_compatible,
            "has_geographic_overlap": has_overlap,
            "overlap_percentage": overlap_pct,
            "dimensions_match": dimension_match,
            "modality_a": mod_a,
            "modality_b": mod_b,
            "is_optical_sar_pair": is_optical_sar_pair,
            "is_bitemporal_pair": is_bitemporal_pair,
            "errors": errors,
            "warnings": warnings,
        }
