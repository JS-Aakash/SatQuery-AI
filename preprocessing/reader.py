"""
GeoTIFF Reader Service.
Handles safe raster ingestion from filesystem paths, byte streams, and memory buffers
using rasterio and PIL for benchmark formats.
"""
import io
import os
from typing import Union, Tuple, Optional, Any
import numpy as np
import rasterio
from rasterio.io import MemoryFile, DatasetReader
from PIL import Image


class RasterReadError(Exception):
    """Raised when a raster cannot be read or is corrupted."""
    pass


class GeoTIFFReader:
    """
    Modular reader for GeoTIFF, TIFF, and standard benchmark imagery.
    Supports file paths, in-memory bytes, and temporary datasets.
    """

    @staticmethod
    def is_geotiff(source: Union[str, bytes, io.BytesIO]) -> bool:
        """Determines if the source raster contains GeoTIFF tags and georeferencing."""
        try:
            with GeoTIFFReader.open_dataset(source) as ds:
                return ds.crs is not None and ds.transform is not None
        except Exception:
            return False

    @staticmethod
    def open_dataset(source: Union[str, bytes, io.BytesIO]) -> Any:
        """
        Opens a rasterio dataset from a path, raw bytes, or BytesIO buffer.
        Caller should use as a context manager (e.g. `with GeoTIFFReader.open_dataset(...) as ds:`).
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                raise RasterReadError(f"Raster file not found: {source}")
            try:
                return rasterio.open(source)
            except Exception as e:
                raise RasterReadError(f"Failed to open raster file '{source}': {str(e)}")
        
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise RasterReadError("Empty byte buffer provided.")
            try:
                memfile = MemoryFile(source)
                return memfile.open()
            except Exception as e:
                raise RasterReadError(f"Failed to open in-memory raster: {str(e)}")

        elif isinstance(source, io.BytesIO):
            source.seek(0)
            data = source.read()
            if len(data) == 0:
                raise RasterReadError("Empty BytesIO buffer provided.")
            memfile = MemoryFile(data)
            return memfile.open()

        else:
            raise RasterReadError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def read_bands(
        source: Union[str, bytes, io.BytesIO],
        bands: Optional[Tuple[int, ...]] = None,
        masked: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Reads specified band indices (1-indexed) from raster.
        Returns: (numpy_array of shape [bands, height, width], profile_dict)
        """
        with GeoTIFFReader.open_dataset(source) as ds:
            profile = ds.profile.copy()
            if bands:
                # rasterio is 1-indexed
                valid_bands = [b for b in bands if 1 <= b <= ds.count]
                if not valid_bands:
                    raise RasterReadError(f"Requested bands {bands} out of range for raster with {ds.count} bands.")
                arr = ds.read(valid_bands, masked=masked)
            else:
                arr = ds.read(masked=masked)
            
            # If masked array, extract data with fill value
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(ds.nodata if ds.nodata is not None else 0)

            return arr, profile
