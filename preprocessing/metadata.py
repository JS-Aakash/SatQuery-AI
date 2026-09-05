"""
Raster Metadata Service.
Extracts comprehensive structural and geospatial metadata from GeoTIFF rasters.
"""
from typing import Union, Dict, Any, List, Optional
import io
import math
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from .reader import GeoTIFFReader


class RasterMetadataService:
    """
    Extracts complete, structured metadata from GeoTIFF and standard rasters.
    """

    @staticmethod
    def extract_metadata(source: Union[str, bytes, io.BytesIO], filename: str = "raster.tif") -> Dict[str, Any]:
        """
        Extracts width, height, number of bands, dtype, CRS, transform, bounds,
        resolution, nodata, and band descriptions.
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            width = ds.width
            height = ds.height
            count = ds.count
            dtype_str = str(ds.dtypes[0]) if ds.dtypes else "uint8"
            
            # CRS string
            crs_str = ds.crs.to_string() if ds.crs else None
            is_geographic = ds.crs.is_geographic if ds.crs else False
            
            # Affine Transform as 6-element list or string
            transform_list = list(ds.transform)[:6] if ds.transform else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            
            # Bounds [left, bottom, right, top]
            bounds = list(ds.bounds) if ds.bounds else [0.0, 0.0, float(width), float(height)]
            
            # Reproject bounds to WGS84 (EPSG:4326) for web preview if possible
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

            # Band descriptions / names
            descriptions = list(ds.descriptions) if ds.descriptions else []
            band_names = []
            for i in range(1, count + 1):
                desc = ds.descriptions[i - 1] if descriptions and i - 1 < len(descriptions) and descriptions[i - 1] else None
                if desc:
                    band_names.append(desc)
                elif count == 1:
                    band_names.append("Band 1 (Single Channel / SAR)")
                elif count == 2:
                    band_names.append("VV (Co-Pol)" if i == 1 else "VH (Cross-Pol)")
                elif count == 3:
                    names = ["Red", "Green", "Blue"]
                    band_names.append(names[i - 1])
                elif count == 4:
                    names = ["Red", "Green", "Blue", "NIR"]
                    band_names.append(names[i - 1])
                elif count >= 12:
                    s2_names = ["B01 Coastal", "B02 Blue", "B03 Green", "B04 Red", "B05 RedEdge1", "B06 RedEdge2", 
                                "B07 RedEdge3", "B08 NIR", "B8A NarrowNIR", "B09 WaterVapour", "B11 SWIR1", "B12 SWIR2"]
                    band_names.append(s2_names[i - 1] if i - 1 < len(s2_names) else f"Band {i}")
                else:
                    band_names.append(f"Band {i}")

            # Modality and Sensor heuristics
            modality = "Optical"
            sensor = "Unknown / User Provided"
            lower_name = filename.lower()
            if count == 1 or count == 2 or "sar" in lower_name or "s1" in lower_name or "risat" in lower_name:
                modality = "SAR"
                sensor = "Sentinel-1" if "s1" in lower_name or "sentinel-1" in lower_name else ("RISAT-1A" if "risat" in lower_name else "SAR Sensor")
            elif count >= 4:
                modality = "Multispectral"
                sensor = "Sentinel-2" if count >= 10 or "s2" in lower_name or "sentinel" in lower_name else ("Landsat 8/9" if count in [7, 8] else "Multispectral Sensor")
            elif "cartosat" in lower_name:
                modality = "Optical"
                sensor = "Cartosat-2S"

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
                "modality": modality,
                "sensor": sensor,
                "colorinterp": [ci.name for ci in ds.colorinterp] if ds.colorinterp else [],
            }
