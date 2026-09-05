"""
AOI Service for SatQuery AI.
Handles cropping satellite imagery to bounding boxes, arbitrary polygons,
and map-selected areas of interest while updating geospatial affine transforms.
"""
from typing import Union, List, Dict, Any, Tuple, Optional
import io
import json
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import box, Polygon, shape, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

from .reader import GeoTIFFReader
from .band_service import BandService


class AOIService:
    """
    Crops rasters to bounding boxes, GeoJSON polygons, or coordinate lists.
    """

    @staticmethod
    def _ensure_polygon(
        aoi_input: Union[List[float], List[List[float]], Dict[str, Any], Polygon],
        raster_crs: Any
    ) -> Polygon:
        """
        Converts bounding box [minx, miny, maxx, maxy], coordinate list,
        or GeoJSON dict to a Shapely Polygon reprojected to the raster's native CRS.
        """
        if isinstance(aoi_input, Polygon):
            poly = aoi_input
        elif isinstance(aoi_input, list) and len(aoi_input) == 4 and isinstance(aoi_input[0], (int, float)):
            # Bounding box [min_lon, min_lat, max_lon, max_lat]
            poly = box(aoi_input[0], aoi_input[1], aoi_input[2], aoi_input[3])
        elif isinstance(aoi_input, list) and len(aoi_input) >= 3 and isinstance(aoi_input[0], (list, tuple)):
            # Coordinate list [[lon1, lat1], [lon2, lat2], ...]
            # Ensure polygon closure
            pts = list(aoi_input)
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            poly = Polygon(pts)
        elif isinstance(aoi_input, dict):
            # GeoJSON geometry
            poly = shape(aoi_input)
        else:
            raise ValueError(f"Invalid AOI format: {type(aoi_input)}")

        # Check if input coordinates look like WGS84 (degrees within [-180, 180] and [-90, 90])
        minx, miny, maxx, maxy = poly.bounds
        is_wgs84_like = (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and
                         -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0)

        # Reproject from WGS84 to raster CRS if raster has a projected CRS and input was in WGS84 degrees
        if raster_crs and not raster_crs.is_geographic and is_wgs84_like:
            try:
                transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
                poly = shapely_transform(transformer.transform, poly)
            except Exception:
                pass

        return poly

    @staticmethod
    def crop_raster(
        source: Union[str, bytes, io.BytesIO],
        aoi: Union[List[float], List[List[float]], Dict[str, Any], Polygon],
        crop_to_envelope: bool = True
    ) -> Tuple[bytes, dict, str]:
        """
        Crops raster to given AOI geometry.
        Returns: (cropped_geotiff_bytes, cropped_metadata_dict, preview_data_url)
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            raster_crs = ds.crs
            poly = AOIService._ensure_polygon(aoi, raster_crs)
            geojson_geom = [mapping(poly)]

            # Apply rasterio mask
            out_image, out_transform = mask(ds, geojson_geom, crop=crop_to_envelope, nodata=ds.nodata)
            out_profile = ds.profile.copy()

            out_profile.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # Save cropped raster to in-memory GeoTIFF buffer
        out_buf = io.BytesIO()
        with MemoryFile() as memfile:
            with memfile.open(**out_profile) as dst:
                dst.write(out_image)
            memfile.seek(0)
            cropped_bytes = memfile.read()

        # Generate web-safe preview from cropped raster
        res_preview = BandService.create_true_color_rgb(cropped_bytes)
        preview_data_url = res_preview[1] if isinstance(res_preview, (tuple, list)) else res_preview

        cropped_meta = {
            "width": out_image.shape[2],
            "height": out_image.shape[1],
            "bands": out_image.shape[0],
            "transform": list(out_transform)[:6],
            "dtype": str(out_profile.get("dtype")),
            "crs": str(raster_crs) if raster_crs else None
        }

        return cropped_bytes, cropped_meta, preview_data_url
