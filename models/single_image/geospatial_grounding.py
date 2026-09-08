"""
Geospatial Region Grounding Engine for Single-Image Analysis.
Delineates target semantic regions (bare land, vegetation, water, urban, SAR backscatter),
converts pixel contours into georeferenced GeoJSON polygons via raster affine transform,
and computes quantitative surface areas in square meters and hectares.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from preprocessing import GeoTIFFReader, RasterMetadataService, BandService, SARPreprocessor
from preprocessing.raster_classifier import RasterClassifier, RasterType, BandRole


class GroundedGeospatialRegion:
    """Represents a structured georeferenced region detected from text grounding."""
    def __init__(
        self,
        region_id: str,
        label: str,
        category: str,
        confidence: float,
        pixel_bbox: List[float],
        geo_bbox: List[float],
        polygon: Dict[str, Any],
        area_m2: float,
        area_ha: float,
        centroid: List[float],
        color: str = "#10b981",
        description: str = ""
    ):
        self.id = region_id
        self.label = label
        self.category = category
        self.confidence = confidence
        self.pixel_bbox = pixel_bbox  # [ymin, xmin, ymax, xmax] in 0-100%
        self.geo_bbox = geo_bbox      # [min_lon, min_lat, max_lon, max_lat]
        self.polygon = polygon        # GeoJSON Polygon
        self.area_m2 = area_m2
        self.area_ha = area_ha
        self.centroid = centroid      # [lon, lat]
        self.color = color
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "pixel_bbox": [round(c, 2) for c in self.pixel_bbox],
            "geo_bbox": [round(c, 6) for c in self.geo_bbox],
            "polygon": self.polygon,
            "area_m2": round(self.area_m2, 1),
            "area_ha": round(self.area_ha, 2),
            "centroid": [round(self.centroid[0], 6), round(self.centroid[1], 6)],
            "color": self.color,
            "description": self.description,
        }


class GeospatialGroundingEngine:
    """
    Executes text-guided region grounding on raw single-image rasters.
    """

    @staticmethod
    def ground_query_on_raster(
        source: Union[str, bytes],
        query: str,
        filename: str = "raster.tif",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[GroundedGeospatialRegion]:
        """
        Executes query grounding against optical multispectral or SAR raster.
        """
        q_lower = query.lower()
        grounded_regions: List[GroundedGeospatialRegion] = []

        with GeoTIFFReader.open_dataset(source) as ds:
            width = ds.width
            height = ds.height
            count = ds.count
            crs = ds.crs
            transform = ds.transform
            is_geographic = crs.is_geographic if crs else True
            res_x = abs(transform.a) if transform else 1.0
            res_y = abs(transform.e) if transform else 1.0
            gsd_m = res_x * 111320.0 if is_geographic else res_x

            # Identify raster type
            raster_type, band_map, _ = RasterClassifier.classify_raster(
                total_bands=count,
                descriptions=list(ds.descriptions) if ds.descriptions else [],
                filename=filename
            )

            # Determine target analysis intent from query
            is_playground = any(k in q_lower for k in ["playground", "stadium", "sports", "track", "court", "pitch", "field", "baseball", "arena", "running track", "open ground"])
            is_vegetation = any(k in q_lower for k in ["vegetat", "tree", "forest", "crop", "canopy", "agriculture", "farm", "green", "woodland", "jungle"])
            is_urban = any(k in q_lower for k in ["built", "urban", "building", "structure", "city", "industrial", "settlement", "house", "roof", "rooftops"])
            is_water = any(k in q_lower for k in ["water", "lake", "river", "pond", "reservoir", "ocean", "sea", "inlet", "hydrolog"])
            is_vacant_bare = any(k in q_lower for k in ["bare", "vacant", "empty", "unused", "cleared", "soil", "uncultivated"])
            is_sar_strong = any(k in q_lower for k in ["strong sar", "backscatter", "radar", "double bounce", "double-bounce", "high backscatter"])

            # Compute Target Mask
            target_mask = np.zeros((height, width), dtype=bool)
            target_label = "Target Feature"
            target_color = "#10b981"
            base_confidence = 0.88

            if raster_type == RasterType.SAR or "sar" in filename.lower():
                # SAR Grounding
                vv_idx = band_map.get(BandRole.SAR_VV, 1)
                vv_arr = ds.read(min(vv_idx, count)).astype(np.float32)
                # Compute dB if not already dB
                if np.max(vv_arr) > 0:
                    vv_db = 10.0 * np.log10(np.clip(vv_arr, 1e-5, 1e5))
                else:
                    vv_db = vv_arr

                if is_sar_strong or is_urban:
                    # Double-bounce scattering (structures/buildings)
                    target_mask = vv_db >= -7.0
                    target_label = "Strong Radar Backscatter / Built Structure"
                    target_color = "#f59e0b"
                    base_confidence = 0.91
                elif is_water:
                    # Specular reflection (calm water)
                    target_mask = vv_db <= -18.0
                    target_label = "Specular Zero-Return / Water Body"
                    target_color = "#06b6d4"
                    base_confidence = 0.93
                elif is_vegetation or is_vacant_bare:
                    # Volume scattering vs rough bare soil
                    target_mask = (vv_db >= -14.0) & (vv_db < -7.0)
                    target_label = "Potential Bare / Rough Ground Candidate" if is_vacant_bare else "Vegetation Volume Scattering"
                    target_color = "#f59e0b" if is_vacant_bare else "#10b981"
                    base_confidence = 0.86
                else:
                    target_mask = vv_db >= -9.0
                    target_label = "SAR Identified Feature"
                    target_color = "#38bdf8"
            else:
                # Optical / Multispectral Grounding
                b_nir_idx = band_map.get(BandRole.NIR)
                b_red_idx = band_map.get(BandRole.RED) or band_map.get(BandRole.RGB_RED, 1)
                b_green_idx = band_map.get(BandRole.GREEN) or band_map.get(BandRole.RGB_GREEN, min(2, count))
                b_blue_idx = band_map.get(BandRole.BLUE) or band_map.get(BandRole.RGB_BLUE, min(3, count))
                b_swir_idx = band_map.get(BandRole.SWIR_1)

                r_arr = ds.read(min(b_red_idx, count)).astype(np.float32)
                g_arr = ds.read(min(b_green_idx, count)).astype(np.float32) if count >= 2 else r_arr.copy()
                b_arr = ds.read(min(b_blue_idx, count)).astype(np.float32) if count >= 3 else r_arr.copy()
                lum_arr = 0.299 * r_arr + 0.587 * g_arr + 0.114 * b_arr
                exg_arr = 2.0 * g_arr - r_arr - b_arr

                has_ndvi = bool(b_nir_idx and b_red_idx)
                if has_ndvi:
                    nir = ds.read(b_nir_idx).astype(np.float32)
                    red = ds.read(b_red_idx).astype(np.float32)
                    denom = nir + red
                    denom[denom == 0] = 1e-6
                    ndvi = (nir - red) / denom
                else:
                    ndvi = None

                if is_playground:
                    target_label = "Playground / Athletic Facility"
                    target_color = "#f59e0b"
                    track = (r_arr > 165) & (r_arr > g_arr + 40) & (r_arr > b_arr + 70)
                    turf = (exg_arr > 14.0) & (g_arr > r_arr + 8) & (lum_arr > 60) & (lum_arr < 210)
                    target_mask = track | turf
                    base_confidence = 0.95

                elif is_vegetation:
                    target_label = "Dense Vegetation / Canopy"
                    target_color = "#10b981"
                    if has_ndvi and ndvi is not None:
                        target_mask = ndvi > 0.35
                    else:
                        green_veg = (g_arr > r_arr + 2) & (g_arr > b_arr) & (exg_arr > 4.0)
                        dark_canopy = (lum_arr < 75) & (g_arr >= r_arr - 8) & (b_arr < 65) & (lum_arr > 20)
                        target_mask = green_veg | dark_canopy
                    base_confidence = 0.95

                elif is_water:
                    target_label = "Water Body / Hydrological Basin"
                    target_color = "#06b6d4"
                    if b_green_idx and b_nir_idx:
                        green = ds.read(b_green_idx).astype(np.float32)
                        nir = ds.read(b_nir_idx).astype(np.float32)
                        ndwi = (green - nir) / (green + nir + 1e-6)
                        target_mask = ndwi > 0.0
                    else:
                        target_mask = ((lum_arr < 55) & (b_arr >= r_arr - 4)) | ((b_arr > r_arr + 15) & (g_arr > r_arr + 8) & (lum_arr < 120))
                    base_confidence = 0.93

                # Extract optical bounding boxes directly via high-precision spatial normalizer
                from .normalizer import SpatialNormalizer
                opt_feat = "playground" if is_playground else ("trees" if is_vegetation else ("urban" if is_urban else ("water" if is_water else "vegetation")))
                raw_rgb = np.stack([
                    np.clip(r_arr, 0, 255).astype(np.uint8),
                    np.clip(g_arr, 0, 255).astype(np.uint8),
                    np.clip(b_arr, 0, 255).astype(np.uint8)
                ], axis=-1)

                norm_boxes = SpatialNormalizer.extract_multiple_bounding_boxes_from_raster(raw_rgb, opt_feat, max_boxes=3)
                if norm_boxes:
                    for idx, nb in enumerate(norm_boxes, start=1):
                        min_y = int((nb[0] / 100.0) * height)
                        min_x = int((nb[1] / 100.0) * width)
                        max_y = int((nb[2] / 100.0) * height)
                        max_x = int((nb[3] / 100.0) * width)
                        pixel_bbox = [nb[0], nb[1], nb[2], nb[3]]

                        if transform:
                            top_left = xy(transform, min_y, min_x)
                            top_right = xy(transform, min_y, max_x)
                            bottom_right = xy(transform, max_y, max_x)
                            bottom_left = xy(transform, max_y, min_x)
                            center_xy = xy(transform, int((min_y + max_y) / 2), int((min_x + max_x) / 2))

                            if crs and not is_geographic:
                                try:
                                    lons, lats = warp_transform(crs, "EPSG:4326", 
                                        [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]],
                                        [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]]
                                    )
                                    c_lon, c_lat = warp_transform(crs, "EPSG:4326", [center_xy[0]], [center_xy[1]])
                                    poly_coords = [[lons[i], lats[i]] for i in range(5)]
                                    centroid = [c_lon[0], c_lat[0]]
                                    geo_bbox = [min(lons), min(lats), max(lons), max(lats)]
                                except Exception:
                                    poly_coords = [
                                        [top_left[0], top_left[1]], [top_right[0], top_right[1]],
                                        [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]],
                                        [top_left[0], top_left[1]]
                                    ]
                                    centroid = [center_xy[0], center_xy[1]]
                                    geo_bbox = [min(top_left[0], bottom_left[0]), min(bottom_left[1], bottom_right[1]), max(top_right[0], bottom_right[0]), max(top_left[1], top_right[1])]
                            else:
                                poly_coords = [
                                    [top_left[0], top_left[1]], [top_right[0], top_right[1]],
                                    [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]],
                                    [top_left[0], top_left[1]]
                                ]
                                centroid = [center_xy[0], center_xy[1]]
                                geo_bbox = [min(top_left[0], bottom_left[0]), min(bottom_left[1], bottom_right[1]), max(top_right[0], bottom_right[0]), max(top_left[1], top_right[1])]
                        else:
                            poly_coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                            centroid = [0.5, 0.5]
                            geo_bbox = [0, 0, 1, 1]

                        area_px = max(1, (max_y - min_y) * (max_x - min_x))
                        area_m2 = area_px * (gsd_m * gsd_m)
                        area_ha = area_m2 / 10000.0

                        geojson_poly = {
                            "type": "Polygon",
                            "coordinates": [poly_coords]
                        }

                        grounded_regions.append(GroundedGeospatialRegion(
                            region_id=f"grd-{uuid.uuid4().hex[:6]}",
                            label=f"{target_label} #{idx}",
                            category=target_label,
                            confidence=base_confidence - (idx * 0.02),
                            pixel_bbox=pixel_bbox,
                            geo_bbox=geo_bbox,
                            polygon=geojson_poly,
                            area_m2=area_m2,
                            area_ha=area_ha,
                            centroid=centroid,
                            color=target_color,
                            description=f"Delineated region spanning {area_ha:.2f} hectares based on high-resolution spectral extraction."
                        ))

                    return grounded_regions
                    bottom_right = xy(transform, max_y, max_x)
                    bottom_left = xy(transform, max_y, min_x)
                    center_xy = xy(transform, int((min_y + max_y) / 2), int((min_x + max_x) / 2))

                    # If CRS is projected, reproject coordinates to EPSG:4326 WGS84
                    if crs and not is_geographic:
                        try:
                            lons, lats = warp_transform(crs, "EPSG:4326", 
                                [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]],
                                [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]]
                            )
                            c_lon, c_lat = warp_transform(crs, "EPSG:4326", [center_xy[0]], [center_xy[1]])
                            poly_coords = [[lons[i], lats[i]] for i in range(5)]
                            centroid = [c_lon[0], c_lat[0]]
                            geo_bbox = [min(lons), min(lats), max(lons), max(lats)]
                        except Exception:
                            poly_coords = [
                                [top_left[0], top_left[1]], [top_right[0], top_right[1]],
                                [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]],
                                [top_left[0], top_left[1]]
                            ]
                            centroid = [center_xy[0], center_xy[1]]
                            geo_bbox = [min(top_left[0], bottom_left[0]), min(bottom_left[1], bottom_right[1]), max(top_right[0], bottom_right[0]), max(top_left[1], top_right[1])]
                    else:
                        poly_coords = [
                            [top_left[0], top_left[1]], [top_right[0], top_right[1]],
                            [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]],
                            [top_left[0], top_left[1]]
                        ]
                        centroid = [center_xy[0], center_xy[1]]
                        geo_bbox = [min(top_left[0], bottom_left[0]), min(bottom_left[1], bottom_right[1]), max(top_right[0], bottom_right[0]), max(top_left[1], top_right[1])]
                else:
                    poly_coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                    centroid = [0.5, 0.5]
                    geo_bbox = [0, 0, 1, 1]

                # Compute quantitative area
                pixel_count = len(cy_inds)
                area_m2 = pixel_count * (gsd_m * gsd_m)
                area_ha = area_m2 / 10000.0

                geojson_poly = {
                    "type": "Polygon",
                    "coordinates": [poly_coords]
                }

                grounded_regions.append(GroundedGeospatialRegion(
                    region_id=f"grd-{uuid.uuid4().hex[:6]}",
                    label=f"{target_label} #{idx}",
                    category=target_label,
                    confidence=base_confidence - (idx * 0.02),
                    pixel_bbox=pixel_bbox,
                    geo_bbox=geo_bbox,
                    polygon=geojson_poly,
                    area_m2=area_m2,
                    area_ha=area_ha,
                    centroid=centroid,
                    color=target_color,
                    description=f"Delineated region spanning {area_ha:.2f} hectares based on spectral/backscatter analysis."
                ))

        return grounded_regions
