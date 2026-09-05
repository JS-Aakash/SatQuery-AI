"""
Tile Service for SatQuery AI.
Generates configurable grid tiles from large satellite rasters,
preserving window offsets, affine transforms, and geographic coordinate mapping.
"""
from typing import Union, List, Dict, Any, Tuple, Optional
import io
import math
import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.io import MemoryFile

from .reader import GeoTIFFReader


class TileService:
    """
    Slices satellite imagery into overlapping or adjacent ML-ready tiles.
    Retains affine georeferencing to map detections back to the master coordinate system.
    """

    @staticmethod
    def generate_tile_manifest(
        source: Union[str, bytes, io.BytesIO],
        tile_size: int = 512,
        overlap: int = 64
    ) -> List[Dict[str, Any]]:
        """
        Calculates tile grid layout and metadata without loading the entire raster into memory.
        Returns a list of tile manifests containing:
        - tile_id
        - col_off, row_off, width, height (pixel window)
        - tile_transform (local affine transform)
        - bounds (geographic [left, bottom, right, top])
        """
        manifests = []
        stride = tile_size - overlap
        if stride <= 0:
            stride = tile_size

        with GeoTIFFReader.open_dataset(source) as ds:
            width = ds.width
            height = ds.height
            master_transform = ds.transform
            crs = str(ds.crs) if ds.crs else None

            tile_idx = 0
            for row in range(0, height, stride):
                for col in range(0, width, stride):
                    # Clamp window to raster bounds
                    w = min(tile_size, width - col)
                    h = min(tile_size, height - row)

                    if w <= 0 or h <= 0:
                        continue

                    win = Window(col_off=col, row_off=row, width=w, height=h)
                    # Compute affine transform for this specific tile
                    tile_trans = window_transform(win, master_transform)
                    
                    # Compute geographic bounds of tile
                    left = tile_trans.c
                    top = tile_trans.f
                    right = left + tile_trans.a * w
                    bottom = top + tile_trans.e * h
                    bounds = [min(left, right), min(bottom, top), max(left, right), max(bottom, top)]

                    manifests.append({
                        "tile_id": f"tile_{tile_idx:04d}",
                        "row_index": row // stride,
                        "col_index": col // stride,
                        "col_off": col,
                        "row_off": row,
                        "width": w,
                        "height": h,
                        "transform": list(tile_trans)[:6],
                        "bounds": bounds,
                        "crs": crs,
                    })
                    tile_idx += 1

        return manifests

    @staticmethod
    def extract_tile_data(
        source: Union[str, bytes, io.BytesIO],
        tile_manifest: Dict[str, Any]
    ) -> Tuple[np.ndarray, dict]:
        """
        Reads the specific pixel window corresponding to a tile manifest.
        Returns: (tile_array of shape [bands, height, width], tile_profile)
        """
        win = Window(
            col_off=tile_manifest["col_off"],
            row_off=tile_manifest["row_off"],
            width=tile_manifest["width"],
            height=tile_manifest["height"]
        )

        with GeoTIFFReader.open_dataset(source) as ds:
            tile_arr = ds.read(window=win)
            tile_profile = ds.profile.copy()
            tile_transform = rasterio.Affine(*tile_manifest["transform"])
            tile_profile.update({
                "width": tile_manifest["width"],
                "height": tile_manifest["height"],
                "transform": tile_transform
            })

            return tile_arr, tile_profile

    @staticmethod
    def map_tile_box_to_global_geo(
        tile_manifest: Dict[str, Any],
        norm_box: List[float]  # [ymin, xmin, ymax, xmax] 0-100%
    ) -> Dict[str, Any]:
        """
        Maps a bounding box detected within a tile back to the global raster
        pixel coordinates and geographic coordinates.
        """
        ymin_pct, xmin_pct, ymax_pct, xmax_pct = norm_box
        t_w = tile_manifest["width"]
        t_h = tile_manifest["height"]
        col_off = tile_manifest["col_off"]
        row_off = tile_manifest["row_off"]

        # Pixel coords relative to tile
        t_xmin = (xmin_pct / 100.0) * t_w
        t_xmax = (xmax_pct / 100.0) * t_w
        t_ymin = (ymin_pct / 100.0) * t_h
        t_ymax = (ymax_pct / 100.0) * t_h

        # Global pixel coordinates
        g_xmin = col_off + t_xmin
        g_xmax = col_off + t_xmax
        g_ymin = row_off + t_ymin
        g_ymax = row_off + t_ymax

        # Geographic coordinates using affine transform
        # X = a * col + c, Y = e * row + f
        a, b, c, d, e, f = tile_manifest["transform"]
        geo_min_x = a * t_xmin + c
        geo_max_x = a * t_xmax + c
        geo_min_y = e * t_ymax + f
        geo_max_y = e * t_ymin + f

        return {
            "global_pixel_box": [round(g_ymin, 1), round(g_xmin, 1), round(g_ymax, 1), round(g_xmax, 1)],
            "geographic_bounds": [
                round(min(geo_min_x, geo_max_x), 6),
                round(min(geo_min_y, geo_max_y), 6),
                round(max(geo_min_x, geo_max_x), 6),
                round(max(geo_min_y, geo_max_y), 6),
            ]
        }
