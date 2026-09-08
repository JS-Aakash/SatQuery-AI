"""
Spatial Coordinate Normalizer and Grounding Parser.
Converts various VLM bounding box representations (0-1000 integer tokens, relative 0.0-1.0 floats,
or absolute pixel coordinates) into SatQuery AI's standardized [ymin, xmin, ymax, xmax] 0-100% format.
"""
import re
import uuid
import os;
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
        Dynamically extracts the precise bounding box [ymin, xmin, ymax, xmax] (in 0-100% space)
        of a requested remote-sensing feature directly from the raster pixels using adaptive
        chromaticity modeling, morphological opening/closing, and connected component clustering.
        """
        boxes = SpatialNormalizer.extract_multiple_bounding_boxes_from_raster(image_input, feature_type, max_boxes=1)
        return boxes[0] if boxes else None

    @staticmethod
    def extract_multiple_bounding_boxes_from_raster(
        image_input: Any,
        feature_type: str,
        max_boxes: int = 3
    ) -> List[List[float]]:
        """
        Extracts multiple distinct non-overlapping bounding boxes for multi-part queries
        (e.g., 'highlight playground in 2 parts', 'both baseball fields', 'dense trees', 'buildings').
        Uses verified spectral chromaticity signatures and morphological component labeling.
        """
        try:
            import numpy as np
            from scipy.ndimage import label, binary_opening, binary_closing
            from PIL import Image
            import io
            import base64

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
                return []

            arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
            h, w, _ = arr.shape
            if h < 10 or w < 10:
                return []

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            exg = 2.0 * g - r - b

            ft = feature_type.lower()
            processed_mask = np.zeros((h, w), dtype=bool)

            if any(k in ft for k in ["tree", "trees", "forest", "dense vegetation", "canopy", "woodland", "jungle"]):
                # Dense tree canopy / forest: Chlorophyll green reflectance or dark canopy absorption
                green_veg = (g > r + 2) & (g > b) & (exg > 4.0)
                dark_canopy = (lum < 75) & (g >= r - 8) & (b < 65) & (lum > 20)
                veg = green_veg | dark_canopy

                # Morphological opening (3x3 to 4x4) eliminates scattered single-tree noise
                struct_sz = max(2, int(min(h, w) * 0.012))
                opened = binary_opening(veg, structure=np.ones((struct_sz, struct_sz)))
                lbl, num_comp = label(opened)

                scores = []
                for i in range(1, num_comp + 1):
                    comp_m = (lbl == i)
                    comp_area = np.count_nonzero(comp_m)
                    if comp_area >= 25:
                        score = comp_area * float(np.mean(exg[comp_m]) + 10.0)
                        scores.append((i, score))
                scores.sort(key=lambda x: x[1], reverse=True)

                boxes: List[List[float]] = []
                for comp_idx, _ in scores[:max_boxes]:
                    rows, cols = np.where(lbl == comp_idx)
                    ymin = max(0.0, float(np.percentile(rows, 1)) / h * 100.0 - 0.5)
                    ymax = min(100.0, float(np.percentile(rows, 99)) / h * 100.0 + 0.5)
                    xmin = max(0.0, float(np.percentile(cols, 1)) / w * 100.0 - 0.5)
                    xmax = min(100.0, float(np.percentile(cols, 99)) / w * 100.0 + 0.5)
                    if (ymax - ymin) > 1.5 and (xmax - xmin) > 1.5:
                        boxes.append([round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)])
                return boxes

            elif any(k in ft for k in ["playground", "stadium", "sports", "track", "court", "pitch", "arena", "open ground", "running track", "field", "baseball"]):
                # 1. Athletic running track / synthetic courts: High red saturation & bright terracotta
                track = (r > 165) & (r > g + 40) & (r > b + 70)
                # 2. Smooth manicured green sports turf
                turf = (exg > 14.0) & (g > r + 8) & (lum > 60) & (lum < 210)
                sports_seed = track | turf

                # Morphological closing bridges adjacent sports facilities into cohesive grounds
                struct_sz = max(3, int(min(h, w) * 0.025))
                closed = binary_closing(sports_seed, structure=np.ones((struct_sz, struct_sz)))
                lbl, num_comp = label(closed)

                scores = []
                for i in range(1, num_comp + 1):
                    comp_m = (lbl == i)
                    comp_area = np.count_nonzero(comp_m)
                    if comp_area >= 20:
                        # Prioritize facilities with athletic track surface or turf concentration
                        track_density = float(np.count_nonzero(comp_m & track))
                        score = comp_area + (track_density * 25.0)
                        scores.append((i, score))
                scores.sort(key=lambda x: x[1], reverse=True)

                if not scores and num_comp > 0:
                    for i in range(1, num_comp + 1):
                        scores.append((i, np.count_nonzero(lbl == i)))
                    scores.sort(key=lambda x: x[1], reverse=True)

                boxes = []
                for comp_idx, _ in scores[:max_boxes]:
                    rows, cols = np.where(lbl == comp_idx)
                    ymin = max(0.0, float(np.percentile(rows, 1)) / h * 100.0 - 0.8)
                    ymax = min(100.0, float(np.percentile(rows, 99)) / h * 100.0 + 0.8)
                    xmin = max(0.0, float(np.percentile(cols, 1)) / w * 100.0 - 0.8)
                    xmax = min(100.0, float(np.percentile(cols, 99)) / w * 100.0 + 0.8)
                    if (ymax - ymin) > 1.5 and (xmax - xmin) > 1.5:
                        boxes.append([round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)])
                return boxes

            elif any(k in ft for k in ["water", "ocean", "sea", "river", "bay", "basin", "hydrology", "pond", "lake"]):
                # Water: Low overall luminance, high Red/NIR absorption, or bright coastal shallow water
                water_mask = ((lum < 55) & (b >= r - 4)) | ((b > r + 15) & (g > r + 8) & (lum < 120))
                struct_sz = max(2, int(min(h, w) * 0.02))
                closed = binary_closing(water_mask, structure=np.ones((struct_sz, struct_sz)))
                lbl, num_comp = label(closed)

                scores = []
                for i in range(1, num_comp + 1):
                    comp_area = np.count_nonzero(lbl == i)
                    if comp_area >= 20:
                        scores.append((i, comp_area))
                scores.sort(key=lambda x: x[1], reverse=True)

                boxes = []
                for comp_idx, _ in scores[:max_boxes]:
                    rows, cols = np.where(lbl == comp_idx)
                    ymin = max(0.0, float(np.percentile(rows, 1)) / h * 100.0 - 0.5)
                    ymax = min(100.0, float(np.percentile(rows, 99)) / h * 100.0 + 0.5)
                    xmin = max(0.0, float(np.percentile(cols, 1)) / w * 100.0 - 0.5)
                    xmax = min(100.0, float(np.percentile(cols, 99)) / w * 100.0 + 0.5)
                    boxes.append([round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)])
                return boxes

            elif any(k in ft for k in ["urban", "city", "building", "buildings", "built", "structure", "settlement", "houses", "roof", "rooftops"]):
                # Urban: High spatial density of bright/neutral roofs and structural edges
                from scipy.ndimage import uniform_filter
                bright_roofs = (lum > 180) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & ~(exg > 10.0)
                density = uniform_filter(bright_roofs.astype(np.float32), size=max(5, int(min(h, w) * 0.06)))
                
                pos_density = density[density > 0]
                thresh = np.percentile(pos_density, 80) if len(pos_density) > 50 else 0.05
                urban_raw = density >= thresh

                struct_sz = max(2, int(min(h, w) * 0.015))
                opened = binary_opening(urban_raw, structure=np.ones((struct_sz, struct_sz)))
                lbl, num_comp = label(opened)

                scores = []
                for i in range(1, num_comp + 1):
                    comp_m = (lbl == i)
                    comp_area = np.count_nonzero(comp_m)
                    if comp_area >= 40:
                        scores.append((i, comp_area))
                scores.sort(key=lambda x: x[1], reverse=True)

                boxes = []
                for comp_idx, _ in scores[:max_boxes]:
                    rows, cols = np.where(lbl == comp_idx)
                    ymin = max(0.0, float(np.percentile(rows, 1)) / h * 100.0 - 0.5)
                    ymax = min(100.0, float(np.percentile(rows, 99)) / h * 100.0 + 0.5)
                    xmin = max(0.0, float(np.percentile(cols, 1)) / w * 100.0 - 0.5)
                    xmax = min(100.0, float(np.percentile(cols, 99)) / w * 100.0 + 0.5)
                    boxes.append([round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)])
                return boxes

            else:
                return []

        except Exception:
            return []

    @staticmethod
    def clean_vlm_text(text: str) -> str:
        """
        Cleans raw VLM outputs by removing GeoChat formatting tags (<p>, </p>, {<...>|<...>}, <delim>),
        returning clean, readable natural language text.
        """
        if not text:
            return ""

        cleaned = text
        # 1. Remove coordinate blocks like {<56><57><60><65>|<90>}, {<56><57><60><65>|90}, or {<56><57><60><65>}
        cleaned = re.sub(r'\{\s*<\d+>\s*<\d+>\s*<\d+>\s*<\d+>(?:\s*\|\s*<?\d+>?)?\s*\}', '', cleaned)
        # 2. Remove <delim> tags
        cleaned = re.sub(r'<delim>', '', cleaned)
        # 3. Unwrap <p>label</p> to label
        cleaned = re.sub(r'<p>(.*?)</p>', r'\1', cleaned)
        # 4. Remove leftover standalone coordinate tokens or bracket tags
        cleaned = re.sub(r'<box>.*?</box>', '', cleaned)
        # 5. Clean up extra spaces and punctuation artifacts
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
        - GeoChat native tokens with <p> wrapper: `<p>some buildings</p> {<56><57><60><65>|<90>}<delim>{<57><49><61><57>|<90>}`
        - Consecutive GeoChat box tokens: `{<57><57><61><65>|<90>}{<57><49><61><57>|<90>}`
        - Standard token brackets: `[150, 200, 650, 750]`
        - `<box>[150, 200, 650, 750]</box>`
        """
        boxes: List[GroundingBoundingBox] = []
        color_palette = ["#10b981", "#06b6d4", "#38bdf8", "#f59e0b", "#ec4899", "#8b5cf6", "#14b8a6"]
        idx = 0

        # GeoChat token pattern matching {<y1><x1><y2><x2>|optional_score} with or without <score> brackets
        box_token_pattern = r'\{\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>(?:\s*\|\s*<?(\d+)>?)?\s*\}'

        # 1. Clustered GeoChat pattern matching label + one or more consecutive box tokens
        cluster_pattern = r'(?:<p>(.*?)</p>|([A-Za-z\s\-]{2,35}))?\s*((?:' + box_token_pattern + r'\s*(?:<delim>)?\s*)+)'
        for m in re.finditer(cluster_pattern, text):
            raw_label = m.group(1) or m.group(2) or default_label
            clean_label = re.sub(r'^(there are|there is|in the|here are|some|all|the|a|an)\s+', '', raw_label.strip(), flags=re.IGNORECASE).strip().title()
            if not clean_label or len(clean_label) < 2:
                clean_label = default_label

            box_block = m.group(3)
            for bm in re.finditer(box_token_pattern, box_block):
                y1, x1, y2, x2 = float(bm.group(1)), float(bm.group(2)), float(bm.group(3)), float(bm.group(4))
                conf_val = float(bm.group(5)) / 100.0 if bm.group(5) else 0.92
                try:
                    norm_coords = SpatialNormalizer.normalize_box([y1, x1, y2, x2], img_width, img_height, coord_format="geochat")
                    h_box = abs(norm_coords[2] - norm_coords[0])
                    w_box = abs(norm_coords[3] - norm_coords[1])
                    if h_box < 0.5 and w_box < 0.5:
                        continue

                    box_id = f"box_{uuid.uuid4().hex[:6]}"
                    boxes.append(GroundingBoundingBox(
                        id=box_id,
                        label=clean_label,
                        box=norm_coords,
                        confidence=round(conf_val, 2),
                        color=color_palette[idx % len(color_palette)]
                    ))
                    idx += 1
                except Exception:
                    continue

        # 2. Standalone Bracket pattern: [y1, x1, y2, x2]
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
                        label = label_match.group(1).strip()

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

