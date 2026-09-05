"""
Multispectral GeoTIFF Builder Utility.
Combines multiple single-band GeoTIFFs (e.g. B02, B03, B04, B08, B11, B12)
into a single properly aligned, multi-band Sentinel-2 GeoTIFF.
Handles:
- CRS validation
- Spatial extent matching
- Geospatial resampling (e.g. 20m SWIR bands B11/B12 resampled to 10m grid)
- Writing standard band descriptions
"""
from typing import Dict, Any, List, Tuple, Optional, Union
import io
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np

from .reader import GeoTIFFReader
from .raster_classifier import BandRole, RasterClassifier


class MultispectralBuilder:
    """
    Utility to combine separate single-band GeoTIFF files into one multi-band GeoTIFF.
    """

    @staticmethod
    def build_multiband_geotiff(
        band_files: Dict[str, Union[bytes, str]],
        output_path: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Takes a dict of {band_name: bytes_or_path} (e.g. {'B02': data, 'B03': data, 'B04': data, 'B08': data, 'B11': data, 'B12': data}).
        Selects reference grid (first 10m band like B04 or B02), resamples all bands to match reference shape & transform,
        and creates a multi-band GeoTIFF buffer or writes to output_path.
        """
        if not band_files:
            raise ValueError("No band files provided for multispectral assembly.")

        # Sort bands according to canonical Sentinel-2 order
        order_priority = ["B02", "B03", "B04", "B08", "B11", "B12", "VV", "VH"]
        sorted_keys = sorted(
            band_files.keys(),
            key=lambda k: order_priority.index(k) if k in order_priority else 99
        )

        # 1. Determine reference band dataset (highest resolution / standard)
        ref_key = sorted_keys[0]
        ref_data = band_files[ref_key]

        with GeoTIFFReader.open_dataset(ref_data) as ref_ds:
            ref_meta = ref_ds.meta.copy()
            ref_crs = ref_ds.crs
            ref_transform = ref_ds.transform
            ref_width = ref_ds.width
            ref_height = ref_ds.height
            ref_bounds = ref_ds.bounds

        # Update metadata for multi-band output
        out_meta = ref_meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count": len(sorted_keys),
            "dtype": "uint16" if ref_meta["dtype"] in ["uint16", "int16"] else "float32",
            "width": ref_width,
            "height": ref_height,
            "crs": ref_crs,
            "transform": ref_transform,
        })

        # Resample and read each band into aligned numpy array
        stacked_bands: List[np.ndarray] = []
        band_descriptions: List[str] = []

        for b_name in sorted_keys:
            b_src = band_files[b_name]
            with GeoTIFFReader.open_dataset(b_src) as src_ds:
                band_arr = src_ds.read(1)
                
                # Check if resampling/reprojection is required
                if (src_ds.width != ref_width or src_ds.height != ref_height or 
                    src_ds.crs != ref_crs or src_ds.transform != ref_transform):
                    
                    dest_arr = np.zeros((ref_height, ref_width), dtype=band_arr.dtype)
                    reproject(
                        source=band_arr,
                        destination=dest_arr,
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    stacked_bands.append(dest_arr)
                else:
                    stacked_bands.append(band_arr)

            band_descriptions.append(b_name)

        # Write to memory buffer
        mem_file = io.BytesIO()
        with rasterio.open(
            output_path or mem_file,
            "w",
            **out_meta
        ) as dst:
            for i, arr in enumerate(stacked_bands, start=1):
                dst.write(arr.astype(out_meta["dtype"]), i)
                dst.set_band_description(i, band_descriptions[i - 1])

        result_bytes = mem_file.getvalue() if not output_path else open(output_path, "rb").read()

        summary = {
            "bands_count": len(sorted_keys),
            "band_descriptions": band_descriptions,
            "width": ref_width,
            "height": ref_height,
            "crs": str(ref_crs),
            "transform": list(ref_transform)[:6],
            "bounds": list(ref_bounds),
            "size_bytes": len(result_bytes),
        }

        return result_bytes, summary
