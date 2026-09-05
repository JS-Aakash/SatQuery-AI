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
