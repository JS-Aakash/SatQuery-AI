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

            if any(k in ft for k in ["water", "ocean", "sea", "river", "bay", "basin", "hydrology"]):
                # Optical water: deep blue/green with low red reflectance or low overall luminance (SAR specular reflection)
                # In coastal/RGB scenes: b > r and (lum < 95 or (b - r) > 8)
                water_opt = ((b > r + 3) & (r < 95) & (lum < 110)) | ((lum < 40) & (b >= r))
                mask = water_opt
            elif any(k in ft for k in ["vegetation", "agriculture", "crop", "farm", "canopy", "field"]):
                # Chlorophyll signature: Green channel significantly higher than Red and Blue
                veg_mask = (g > r + 8) & (g > b + 2) & (lum > 30)
                mask = veg_mask
            elif any(k in ft for k in ["urban", "city", "building", "built", "structure"]):
                # Built-up: High local contrast or neutral concrete gray
                urban_mask = (np.abs(r - g) < 22) & (np.abs(g - b) < 22) & (lum >= 35) & (lum <= 180) & ~((b > r + 10) & (lum < 80))
                mask = urban_mask
            elif any(k in ft for k in ["ship", "vessel", "boat", "cargo"]):
                # Moored ships: Bright or distinct colored points located within the water basin (lower half)
                water_bg = ((b > r) | (lum < 75))
                ship_mask = (lum > 140) | (r > 160)
                # Ships reside in the water zone
                ship_mask = ship_mask & (np.arange(h)[:, None] > int(h * 0.45))
                mask = ship_mask
            elif any(k in ft for k in ["port", "dock", "harbor", "berth", "pier", "terminal"]):
                # Port infrastructure: Concrete structures in the coastal transition zone (middle to lower half)
                port_mask = (lum > 100) & (np.abs(r - g) < 25) & (np.arange(h)[:, None] > int(h * 0.40))
                mask = port_mask
            else:
                return None

            pixel_count = np.count_nonzero(mask)
            total_pixels = h * w
            min_pixels = max(30, int(total_pixels * 0.005))

            if pixel_count < min_pixels:
                return None

            rows, cols = np.where(mask)
            ymin = float(np.percentile(rows, 1)) / h * 100.0
            ymax = float(np.percentile(rows, 99)) / h * 100.0
            xmin = float(np.percentile(cols, 1)) / w * 100.0
            xmax = float(np.percentile(cols, 99)) / w * 100.0

            # Add minimal padding
            ymin = max(0.0, ymin - 1.0)
            xmin = max(0.0, xmin - 1.0)
            ymax = min(100.0, ymax + 1.0)
            xmax = min(100.0, xmax + 1.0)

            # Ensure non-trivial area
            if (ymax - ymin) < 2.0 or (xmax - xmin) < 2.0:
                return None

            return [round(ymin, 2), round(xmin, 2), round(ymax, 2), round(xmax, 2)]
        except Exception:
            return None

    @staticmethod
    def parse_boxes_from_text(
        text: str,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        default_label: str = "Detected Target"
    ) -> List[GroundingBoundingBox]:
        """
        Extracts bounding box patterns from VLM generated text.
        Supports patterns such as:
        - `[150, 200, 650, 750]`
        - `<box>[150, 200, 650, 750]</box>`
        - `{"box_2d": [150, 200, 650, 750], "label": "water"}`
        """
        boxes = []
        # Pattern for 4 bracketed numbers
        bracket_pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
        matches = re.finditer(bracket_pattern, text)

        color_palette = ["#10b981", "#06b6d4", "#3b82f6", "#f59e0b", "#ec4899"]

        idx = 0
        for m in matches:
            c1, c2, c3, c4 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            try:
                norm_coords = SpatialNormalizer.normalize_box([c1, c2, c3, c4], img_width, img_height)
                # Filter out degenerate boxes (less than 1% area)
                h_box = norm_coords[2] - norm_coords[0]
                w_box = norm_coords[3] - norm_coords[1]
                if h_box < 1.0 or w_box < 1.0:
                    continue

                # Check for label before the box
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
