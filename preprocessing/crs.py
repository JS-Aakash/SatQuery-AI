"""
Coordinate Reference System (CRS) Service.
Handles projections, EPSG identification, coordinate transformations, and spatial bounds reprojection.
"""
from typing import Tuple, List, Optional, Union, Any
import pyproj
from pyproj import CRS, Transformer
from rasterio.crs import CRS as RasterioCRS
from rasterio.warp import transform_bounds
from shapely.geometry import box, Polygon


class CRSService:
    """
    Geospatial CRS conversion and verification service using pyproj and shapely.
    """

    @staticmethod
    def is_valid_crs(crs_input: Union[str, int, RasterioCRS, CRS]) -> bool:
        """Checks if a CRS string, EPSG code, or object is valid."""
        try:
            if isinstance(crs_input, (RasterioCRS, CRS)):
                return crs_input is not None
            RasterioCRS.from_user_input(crs_input)
            return True
        except Exception:
            return False

    @staticmethod
    def get_epsg_code(crs_input: Union[str, int, RasterioCRS]) -> Optional[int]:
        """Extracts the EPSG integer code if available."""
        try:
            if isinstance(crs_input, RasterioCRS):
                return crs_input.to_epsg()
            crs_obj = RasterioCRS.from_user_input(crs_input)
            return crs_obj.to_epsg()
        except Exception:
            return None

    @staticmethod
    def are_crs_compatible(crs_a: Any, crs_b: Any) -> Tuple[bool, str]:
        """Determines if two rasters share compatible or transformable CRS."""
        try:
            ca = RasterioCRS.from_user_input(crs_a) if not isinstance(crs_a, RasterioCRS) else crs_a
            cb = RasterioCRS.from_user_input(crs_b) if not isinstance(crs_b, RasterioCRS) else crs_b

            if ca == cb:
                return True, f"Identical CRS ({ca.to_string()}). No reprojection required."

            epsg_a = ca.to_epsg()
            epsg_b = cb.to_epsg()
            if epsg_a and epsg_b and epsg_a == epsg_b:
                return True, f"Matching EPSG:{epsg_a}. Spatial alignment compatible."

            return False, f"CRS mismatch: {ca.to_string()} vs {cb.to_string()}. Reprojection required."
        except Exception as e:
            return False, f"CRS compatibility check failed: {str(e)}"

    @staticmethod
    def transform_bbox(
        bounds: List[float],
        src_crs: Union[str, RasterioCRS],
        dst_crs: Union[str, RasterioCRS] = "EPSG:4326"
    ) -> List[float]:
        """Transforms bounding box [left, bottom, right, top] from src_crs to dst_crs."""
        try:
            return list(transform_bounds(src_crs, dst_crs, *bounds))
        except Exception as e:
            raise ValueError(f"Failed to transform bounding box: {str(e)}")

    @staticmethod
    def get_bounds_polygon(bounds: List[float]) -> Polygon:
        """Returns a Shapely Polygon from [left, bottom, right, top] bounding box."""
        return box(bounds[0], bounds[1], bounds[2], bounds[3])
