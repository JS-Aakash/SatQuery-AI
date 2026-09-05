"""
Raster Metadata Service.
Extracts comprehensive structural, geospatial, and statistical metadata from GeoTIFF rasters.
"""
from typing import Union, Dict, Any, List, Optional
import io
import math
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from .reader import GeoTIFFReader
from .raster_classifier import RasterClassifier, RasterType, BandRole


class RasterMetadataService:
    """
    Extracts complete, structured metadata from GeoTIFF and standard rasters.
    """

    @staticmethod
    def extract_metadata(source: Union[str, bytes, io.BytesIO], filename: str = "raster.tif") -> Dict[str, Any]:
        """
        Extracts width, height, number of bands, dtype, CRS, transform, bounds,
        resolution, nodata, band descriptions, and band-level statistics.
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            width = ds.width
            height = ds.height
            count = ds.count
            dtype_str = str(ds.dtypes[0]) if ds.dtypes else "uint8"
            
            # CRS string
            crs_str = ds.crs.to_string() if ds.crs else None
            is_geographic = ds.crs.is_geographic if ds.crs else False
            
            # Affine Transform as 6-element list
            transform_list = list(ds.transform)[:6] if ds.transform else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            # Bounds [left, bottom, right, top]
            bounds = list(ds.bounds) if ds.bounds else [0.0, 0.0, float(width), float(height)]
            
            # Reproject bounds to WGS84 (EPSG:4326) for map visualization
            bounds_wgs84 = None
            if ds.crs:
                try:
                    if ds.crs.to_epsg() == 4326:
                        bounds_wgs84 = bounds
                    else:
                        wgs84_bounds = transform_bounds(ds.crs, "EPSG:4326", *bounds)
                        bounds_wgs84 = list(wgs84_bounds)
                except Exception:
                    bounds_wgs84 = None

            # Resolution (pixel size in units of CRS)
            res_x = abs(ds.transform.a) if ds.transform else 1.0
            res_y = abs(ds.transform.e) if ds.transform else 1.0
            resolution = [round(res_x, 6), round(res_y, 6)]

            # Resolution formatted in meters
            if is_geographic:
                # 1 degree is roughly 111,320 meters at equator
                approx_gsd_m = round(res_x * 111320, 2)
            else:
                approx_gsd_m = round(res_x, 2)

            # Nodata value
            nodata_val = ds.nodata

            # Extract band descriptions and compute lightweight statistics
            raw_descriptions = list(ds.descriptions) if ds.descriptions else []
            band_stats: List[Dict[str, float]] = []
            band_names: List[str] = []

            for i in range(1, count + 1):
                desc = ds.descriptions[i - 1] if raw_descriptions and i - 1 < len(raw_descriptions) and raw_descriptions[i - 1] else None
                try:
                    # Read sample or window to compute statistics without excessive memory
                    sample_data = ds.read(i, out_shape=(min(height, 512), min(width, 512)))
                    valid = sample_data[sample_data != nodata_val] if nodata_val is not None else sample_data
                    if valid.size > 0:
                        b_min = float(np.min(valid))
                        b_max = float(np.max(valid))
                        b_mean = float(np.mean(valid))
                        b_std = float(np.std(valid))
                    else:
                        b_min, b_max, b_mean, b_std = 0.0, 255.0, 128.0, 0.0
                except Exception:
                    b_min, b_max, b_mean, b_std = 0.0, 255.0, 128.0, 0.0

                band_stats.append({
                    "band": i,
                    "min": round(b_min, 2),
                    "max": round(b_max, 2),
                    "mean": round(b_mean, 2),
                    "std": round(b_std, 2)
                })

            # Classify raster type and map band roles
            raster_type, band_role_map, type_meta = RasterClassifier.classify_raster(
                total_bands=count,
                descriptions=raw_descriptions,
                tags=ds.tags() if hasattr(ds, "tags") else {},
                band_stats=band_stats,
                filename=filename
            )

            # Construct human-friendly band names reflecting detected roles
            for i in range(1, count + 1):
                desc = ds.descriptions[i - 1] if raw_descriptions and i - 1 < len(raw_descriptions) and raw_descriptions[i - 1] else None
                # Check if this band index matched a known role
                matched_role = next((role for role, idx in band_role_map.items() if idx == i), None)
                if desc:
                    band_names.append(f"{desc} ({matched_role.value if matched_role else 'Band ' + str(i)})")
                elif matched_role:
                    band_names.append(f"Band {i} ({matched_role.value})")
                else:
                    band_names.append(f"Band {i}")

            modality = "SAR" if raster_type == RasterType.SAR else ("Multispectral" if raster_type == RasterType.OPTICAL_MULTISPECTRAL else "Optical")

            return {
                "filename": filename,
                "driver": ds.driver,
                "width": width,
                "height": height,
                "bands": count,
                "dtype": dtype_str,
                "crs": crs_str,
                "is_geographic": is_geographic,
                "transform": transform_list,
                "bounds": bounds,
                "bounds_wgs84": bounds_wgs84,
                "resolution": resolution,
                "resolution_m": approx_gsd_m,
                "nodata": nodata_val,
                "band_names": band_names,
                "band_stats": band_stats,
                "raster_type": raster_type.value,
                "modality": modality,
                "sensor": type_meta.get("sensor", "Unknown Sensor"),
                "band_role_map": {k.value: v for k, v in band_role_map.items()},
                "type_metadata": type_meta,
                "colorinterp": [ci.name for ci in ds.colorinterp] if ds.colorinterp else [],
            }
