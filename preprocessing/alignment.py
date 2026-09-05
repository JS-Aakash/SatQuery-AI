"""
Image Alignment Service.
Handles spatial co-registration, grid resampling, and pixel alignment between image pairs.
"""
from typing import Union, Tuple, Dict, Any, Optional
import io
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
from shapely.geometry import box

from .reader import GeoTIFFReader


class ImageAlignmentService:
    """
    Spatially aligns and resamples a secondary raster to match a primary reference raster.
    """

    @staticmethod
    def align_to_reference(
        reference_source: Union[str, bytes, io.BytesIO],
        secondary_source: Union[str, bytes, io.BytesIO],
        resampling_method: Resampling = Resampling.bilinear
    ) -> Tuple[np.ndarray, dict]:
        """
        Reprojects and resamples secondary_source to strictly match the reference_source's
        CRS, dimensions, resolution, and affine transform.
        Returns: (aligned_array of shape [bands, height, width], aligned_profile)
        """
        with GeoTIFFReader.open_dataset(reference_source) as ref_ds:
            ref_crs = ref_ds.crs
            ref_transform = ref_ds.transform
            ref_width = ref_ds.width
            ref_height = ref_ds.height
            ref_profile = ref_ds.profile.copy()

        with GeoTIFFReader.open_dataset(secondary_source) as sec_ds:
            sec_count = sec_ds.count
            destination_array = np.zeros((sec_count, ref_height, ref_width), dtype=sec_ds.dtypes[0])

            for b in range(1, sec_count + 1):
                reproject(
                    source=rasterio.band(sec_ds, b),
                    destination=destination_array[b - 1],
                    src_transform=sec_ds.transform,
                    src_crs=sec_ds.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method
                )

            aligned_profile = ref_profile.copy()
            aligned_profile.update({
                "count": sec_count,
                "dtype": sec_ds.dtypes[0],
                "nodata": sec_ds.nodata
            })

            return destination_array, aligned_profile

    @staticmethod
    def check_alignment_metrics(
        meta_a: Dict[str, Any],
        meta_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates alignment discrepancy between two rasters:
        - Resolution ratio
        - Bounds offset
        - Estimated spatial alignment rating (PERFECT, ACCEPTABLE, REQUIRES_RESAMPLING)
        """
        res_a = meta_a.get("resolution", [1.0, 1.0])
        res_b = meta_b.get("resolution", [1.0, 1.0])
        res_ratio = round(max(res_a[0], res_b[0]) / max(min(res_a[0], res_b[0]), 1e-6), 3)

        bounds_a = meta_a.get("bounds")
        bounds_b = meta_b.get("bounds")

        bounds_delta = 0.0
        if bounds_a and bounds_b:
            bounds_delta = round(sum(abs(a - b) for a, b in zip(bounds_a, bounds_b)), 4)

        if res_ratio <= 1.01 and bounds_delta < 0.001:
            rating = "PERFECT_PIXEL_MATCH"
        elif res_ratio <= 1.2:
            rating = "NEAR_COREGISTERED"
        else:
            rating = "REQUIRES_RESAMPLING"

        return {
            "alignment_rating": rating,
            "resolution_ratio": res_ratio,
            "bounds_delta": bounds_delta,
            "crs_match": meta_a.get("crs") == meta_b.get("crs"),
        }
