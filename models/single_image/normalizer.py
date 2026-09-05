"""
Spatial Coordinate Normalizer and Grounding Parser.
Converts various VLM bounding box representations (0-1000 integer tokens, relative 0.0-1.0 floats,
or absolute pixel coordinates) into SatQuery AI's standardized [ymin, xmin, ymax, xmax] 0-100% format.
"""
import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
from .schemas import GroundingBoundingBox


class SpatialNormalizer:
    """
    Standardizes bounding box coordinates across remote-sensing models.
    """

    @staticmethod
    def normalize_box(
        coords: List[float],
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        coord_format: Optional[str] = None
    ) -> List[float]:
        """
        Normalizes a 4-element coordinate list to [ymin, xmin, ymax, xmax] in 0.0 - 100.0% range.
        Handles:
        1. 0.0 - 1.0 unit float scales
        2. Absolute pixel coordinates [0 - H, 0 - W] when img_width/height are given or values > 1000
        3. Percentage scale 0.0 - 100.0
        4. 0 - 1000 integer scales (standard GeoChat token scale)
        """
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinate values, got {len(coords)}: {coords}")

        c1, c2, c3, c4 = [float(c) for c in coords]
        max_val = max(c1, c2, c3, c4)

        if coord_format == "pixel" or (img_width is not None and img_height is not None and coord_format != "geochat" and max_val > 1.05):
            # Absolute pixel coordinates scaled to image width/height
            ymin = (c1 / max(img_height, 1)) * 100.0
            xmin = (c2 / max(img_width, 1)) * 100.0
            ymax = (c3 / max(img_height, 1)) * 100.0
            xmax = (c4 / max(img_width, 1)) * 100.0
        elif max_val <= 1.05:
            # 0.0 - 1.0 unit float scale
            ymin, xmin, ymax, xmax = c1 * 100.0, c2 * 100.0, c3 * 100.0, c4 * 100.0
        elif max_val <= 100.0 and (c1 <= 100.0 and c3 <= 100.0 and c2 <= 100.0 and c4 <= 100.0):
            # Already in 0 - 100% range
            ymin, xmin, ymax, xmax = c1, c2, c3, c4
        elif max_val <= 1005.0:
            # 0 - 1000 GeoChat scale
            ymin, xmin, ymax, xmax = (c1 / 1000.0) * 100.0, (c2 / 1000.0) * 100.0, (c3 / 1000.0) * 100.0, (c4 / 1000.0) * 100.0
        else:
            # Fallback absolute pixel coordinates
            h = img_height or 1024
            w = img_width or 1024
            ymin = (c1 / h) * 100.0
            xmin = (c2 / w) * 100.0
            ymax = (c3 / h) * 100.0
            xmax = (c4 / w) * 100.0

        # Ensure ymin <= ymax and xmin <= xmax
        norm_ymin = max(0.0, min(100.0, min(ymin, ymax)))
        norm_ymax = max(0.0, min(100.0, max(ymin, ymax)))
        norm_xmin = max(0.0, min(100.0, min(xmin, xmax)))
        norm_xmax = max(0.0, min(100.0, max(xmin, xmax)))

        return [round(norm_ymin, 2), round(norm_xmin, 2), round(norm_ymax, 2), round(norm_xmax, 2)]

    @staticmethod
    def extract_spatial_bounding_box_from_raster(
        image_input: Any,
        feature_type: str
    ) -> Optional[List[float]]:
        """
        Dynamically extracts the bounding box [ymin, xmin, ymax, xmax] (in 0-100% space)
        of a requested remote-sensing feature directly from the raster pixels.
        Supports water bodies (NDWI / high Blue-Green / low NIR absorption / low SAR backscatter),
        vegetation (NDVI / dominant green), urban structures, and marine point targets.
        """
        try:
            import numpy as np
            import io
            import base64
            from PIL import Image

            pil_img: Optional[Image.Image] = None

            if isinstance(image_input, Image.Image):
                pil_img = image_input
            elif isinstance(image_input, np.ndarray):
                if image_input.ndim == 2:
                    pil_img = Image.fromarray(image_input)
                elif image_input.ndim == 3:
                    pil_img = Image.fromarray(image_input[:, :, :3].astype(np.uint8))
            elif isinstance(image_input, (bytes, bytearray)):
                try:
                    pil_img = Image.open(io.BytesIO(image_input))
                except Exception:
                    pass
            elif isinstance(image_input, str):
                if image_input.startswith("data:image/"):
                    try:
                        if ";base64," in image_input:
                            b64_data = image_input.split(";base64,")[1]
                            raw_bytes = base64.b64decode(b64_data)
                            pil_img = Image.open(io.BytesIO(raw_bytes))
                    except Exception:
                        pass
                elif os.path.exists(image_input):
                    try:
                        pil_img = Image.open(image_input)
                    except Exception:
                        pass

            if pil_img is None:
                return None

            arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
            h, w, _ = arr.shape
            if h < 10 or w < 10:
                return None

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b

            mask = np.zeros((h, w), dtype=bool)
            ft = feature_type.lower()

            if any(k in ft for k in ["tree", "trees", "forest", "dense vegetation", "canopy", "woodland", "jungle"]):
                # Chlorophyll signature: Green channel significantly higher than Red and Blue, dark green canopy
                exg = 2.0 * g - r - b
                # Dense canopy: positive excess green with vegetation chlorophyll absorption
                veg_dense = (exg > 8.0) & (g > r) & (lum < 165)
                mask = veg_dense if np.any(veg_dense) else ((g > r + 4) & (g > b))
            elif any(k in ft for k in ["playground", "stadium", "sports", "running track"]):
                # Sports stadiums & distinct running tracks (terracotta / brick red athletic oval or turf court)
                track_mask = (r > 135) & (g > 65) & (b < 90) & (r - g > 30) & (r - b > 40)
                mask = track_mask
            elif any(k in ft for k in ["vegetation", "agriculture", "crop", "farm", "field"]):
                exg = 2.0 * g - r - b
                mask = (exg > 5.0) | ((g > r + 5) & (g > b))
            elif any(k in ft for k in ["water", "ocean", "sea", "river", "bay", "basin", "hydrology", "pond", "lake"]):
                # Optical water: deep blue/green with low red reflectance or low overall luminance
                water_opt = ((b > r + 3) & (r < 95) & (lum < 110)) | ((lum < 40) & (b >= r))
                mask = water_opt
            elif any(k in ft for k in ["urban", "city", "building", "buildings", "built", "structure", "settlement", "houses"]):
                # Built-up: High local brightness or distinct white/gray roof reflections
                urban_mask = (lum > 145) | ((np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (lum >= 90) & (lum <= 220) & ~((g > r + 10) & (g > b)))
                mask = urban_mask
            elif any(k in ft for k in ["ship", "vessel", "boat", "cargo"]):
                ship_mask = (lum > 140) | (r > 160)
                ship_mask = ship_mask & (np.arange(h)[:, None] > int(h * 0.45))
                mask = ship_mask
            elif any(k in ft for k in ["port", "dock", "harbor", "berth", "pier", "terminal"]):
                port_mask = (lum > 100) & (np.abs(r - g) < 25) & (np.arange(h)[:, None] > int(h * 0.40))
                mask = port_mask
            else:
                return None

            pixel_count = np.count_nonzero(mask)
            if pixel_count < 25:
                return None

            # Hotspot Density Clustering:
            # Divide image into a 16x16 block grid to find the peak concentration area
            # instead of bounding isolated outlier pixels across the entire image.
            grid_n = 16
            gh, gw = h // grid_n, w // grid_n
            if gh < 2 or gw < 2:
                gh, gw = max(1, h), max(1, w)

            grid_counts = np.zeros((grid_n, grid_n), dtype=np.float32)
            for gy in range(grid_n):
                for gx in range(grid_n):
                    cell = mask[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                    grid_counts[gy, gx] = np.count_nonzero(cell)

            max_val = np.max(grid_counts)
            if max_val == 0:
                return None

            # Threshold for dense region: cells with >= 35% of the maximum concentration
            dense_thresh = max_val * 0.35
            hotspot_cells = grid_counts >= dense_thresh

            # Find bounding extent of the densest connected cluster
            gy_inds, gx_inds = np.where(hotspot_cells)
            if len(gy_inds) == 0:
                rows, cols = np.where(mask)
                gy_inds = (rows // gh)
                gx_inds = (cols // gw)

            # Get the tight pixel bounding box from pixels inside the dense cells
            dense_pixel_mask = np.zeros_like(mask)
            for gy, gx in zip(gy_inds, gx_inds):
                dense_pixel_mask[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw] = mask[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]

            d_rows, d_cols = np.where(dense_pixel_mask)
            if len(d_rows) < 10:
                d_rows, d_cols = np.where(mask)

            ymin = float(np.percentile(d_rows, 2)) / h * 100.0
            ymax = float(np.percentile(d_rows, 98)) / h * 100.0
            xmin = float(np.percentile(d_cols, 2)) / w * 100.0
            xmax = float(np.percentile(d_cols, 98)) / w * 100.0

            # Add minimal safety padding (capped at [0, 100])
            ymin = max(0.0, ymin - 1.5)
            xmin = max(0.0, xmin - 1.5)
            ymax = min(100.0, ymax + 1.5)
            xmax = min(100.0, xmax + 1.5)

            # Prevent returning full canvas bounding box for localized targets
            if (ymax - ymin) > 85.0 and (xmax - xmin) > 85.0:
                if any(t in ft for t in ["playground", "stadium", "sports", "ship", "port"]):
                    return None
                peak_gy, peak_gx = np.unravel_index(np.argmax(grid_counts), grid_counts.shape)
                ymin = max(0.0, (peak_gy - 2) * gh / h * 100.0)
                ymax = min(100.0, (peak_gy + 3) * gh / h * 100.0)
                xmin = max(0.0, (peak_gx - 2) * gw / w * 100.0)
                xmax = min(100.0, (peak_gx + 3) * gw / w * 100.0)

            return [round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)]
        except Exception:
            return None

    @staticmethod
    def clean_vlm_text(text: str) -> str:
        """
        Cleans raw VLM outputs by removing GeoChat formatting tags (<p>, </p>, {<...>|<...>}, <delim>),
        returning clean, readable natural language text.
        """
        if not text:
            return ""

        cleaned = text
        # 1. Remove coordinate blocks like {<56><57><60><65>|<90>} or {<56><57><60><65>|90} or {<56><57><60><65>}
        cleaned = re.sub(r'\{(?:\s*<\d+>\s*){4}(?:\s*\|\s*<?\d+>?)?\s*\}', '', cleaned)
        # 2. Remove any remaining {<...>} coordinate blocks
        cleaned = re.sub(r'\{(?:\s*<\d+>\s*)+\}', '', cleaned)
        # 3. Remove <delim> tags
        cleaned = re.sub(r'<delim>', '', cleaned)
        # 4. Unwrap <p>label</p> to label
        cleaned = re.sub(r'<p>(.*?)</p>', r'\1', cleaned)
        # 5. Remove leftover standalone coordinate tokens or bracket tags
        cleaned = re.sub(r'<box>.*?</box>', '', cleaned)
        # 6. Clean up chat prefixes if any
        cleaned = re.sub(r'^(?:USER:|ASSISTANT:|A chat between.*?:)\s*', '', cleaned, flags=re.IGNORECASE)
        # 7. Clean up extra spaces and punctuation artifacts
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        return cleaned.strip()

    @staticmethod
    def parse_boxes_from_text(
        text: str,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        default_label: str = "Detected Target"
    ) -> List[GroundingBoundingBox]:
        """
        Extracts bounding box patterns from VLM generated text.
        Supports:
        - GeoChat native tokens: `<p>some buildings</p> {<56><57><60><65>|<90>}<delim>{<57><49><61><57>|<90>}`
        - GeoChat inline tokens: `there are buildings {<10><1><14><5>|<90>}{<10><0><14><6>|<90>}`
        - Standard token brackets: `[150, 200, 650, 750]`
        - `<box>[150, 200, 650, 750]</box>`
        """
        if not text:
            return []

        boxes: List[GroundingBoundingBox] = []
        color_palette = ["#10b981", "#06b6d4", "#38bdf8", "#f59e0b", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316"]
        idx = 0

        # Pattern for single GeoChat coordinate block: {<c1><c2><c3><c4>|<?score?>?}
        single_geochat_pattern = re.compile(
            r'\{\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>(?:\s*\|\s*<?(\d+)>?)?\s*\}'
        )

        # 1. Parse GeoChat <p>label</p> phrase blocks
        geochat_phrase_pattern = re.compile(
            r'<p>(.*?)</p>\s*([^{<]*\{(?:\s*<\d+>\s*){4}(?:\s*\|\s*<?\d+>?)?\s*\}(?:(?:\s*<delim>\s*|\s*)\{(?:\s*<\d+>\s*){4}(?:\s*\|\s*<?\d+>?)?\s*\})*)'
        )
        
        parsed_spans = []
        for match in geochat_phrase_pattern.finditer(text):
            phrase_label = match.group(1).strip() or default_label
            box_block = match.group(2)
            parsed_spans.append(match.span())
            
            for b_match in single_geochat_pattern.finditer(box_block):
                y1, x1, y2, x2 = float(b_match.group(1)), float(b_match.group(2)), float(b_match.group(3)), float(b_match.group(4))
                conf_val = float(b_match.group(5)) / 100.0 if b_match.group(5) else 0.94
                try:
                    norm_coords = SpatialNormalizer.normalize_box([y1, x1, y2, x2], img_width, img_height, coord_format="geochat")
                    h_box = abs(norm_coords[2] - norm_coords[0])
                    w_box = abs(norm_coords[3] - norm_coords[1])
                    if h_box < 0.5 and w_box < 0.5:
                        continue

                    box_id = f"box_{uuid.uuid4().hex[:6]}"
                    boxes.append(GroundingBoundingBox(
                        id=box_id,
                        label=phrase_label.title() if len(phrase_label) <= 35 else phrase_label,
                        box=norm_coords,
                        confidence=round(conf_val, 2),
                        color=color_palette[idx % len(color_palette)]
                    ))
                    idx += 1
                except Exception:
                    continue

        # 2. Parse inline GeoChat boxes without <p> wrappers
        for b_match in single_geochat_pattern.finditer(text):
            m_start, m_end = b_match.span()
            if any(start <= m_start and m_end <= end for start, end in parsed_spans):
                continue

            y1, x1, y2, x2 = float(b_match.group(1)), float(b_match.group(2)), float(b_match.group(3)), float(b_match.group(4))
            conf_val = float(b_match.group(5)) / 100.0 if b_match.group(5) else 0.94

            # Infer label from preceding text
            preceding_text = text[max(0, m_start - 60):m_start].strip()
            label = default_label
            label_match = re.search(r'(?:there (?:is|are)|contains?|shows?|observe|detected?|found|a|an|some|the)?\s*([a-zA-Z\s\-]{3,30})$', preceding_text, flags=re.IGNORECASE)
            if label_match and len(label_match.group(1).strip()) >= 3:
                cand = label_match.group(1).strip()
                cand_clean = re.sub(r'^(?:is|are|a|an|the|some|of|in)\s+', '', cand, flags=re.IGNORECASE).strip()
                if len(cand_clean) >= 3:
                    label = cand_clean.title()

            try:
                norm_coords = SpatialNormalizer.normalize_box([y1, x1, y2, x2], img_width, img_height, coord_format="geochat")
                h_box = abs(norm_coords[2] - norm_coords[0])
                w_box = abs(norm_coords[3] - norm_coords[1])
                if h_box < 0.5 and w_box < 0.5:
                    continue

                box_id = f"box_{uuid.uuid4().hex[:6]}"
                boxes.append(GroundingBoundingBox(
                    id=box_id,
                    label=label,
                    box=norm_coords,
                    confidence=round(conf_val, 2),
                    color=color_palette[idx % len(color_palette)]
                ))
                idx += 1
            except Exception:
                continue

        # 3. Standard Bracket pattern: [y1, x1, y2, x2]
        if not boxes:
            bracket_pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
            for m in re.finditer(bracket_pattern, text):
                c1, c2, c3, c4 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
                try:
                    norm_coords = SpatialNormalizer.normalize_box([c1, c2, c3, c4], img_width, img_height)
                    h_box = abs(norm_coords[2] - norm_coords[0])
                    w_box = abs(norm_coords[3] - norm_coords[1])
                    if h_box < 0.5 and w_box < 0.5:
                        continue

                    preceding_text = text[max(0, m.start() - 40):m.start()]
                    label = default_label
                    label_match = re.search(r'([A-Za-z\s]+)(?:is located at|at|:)?\s*$', preceding_text)
                    if label_match and len(label_match.group(1).strip()) > 2:
                        label = label_match.group(1).strip().title()

                    box_id = f"box_{uuid.uuid4().hex[:6]}"
                    boxes.append(GroundingBoundingBox(
                        id=box_id,
                        label=label,
                        box=norm_coords,
                        confidence=0.92,
                        color=color_palette[idx % len(color_palette)]
                    ))
                    idx += 1
                except Exception:
                    continue

        return boxes

