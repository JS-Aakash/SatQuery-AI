"""
Automated unit and integration tests for Remote-Sensing Grounding Accuracy.
Validates that queries like 'Highlight water body' and phrase-localization queries
accurately delineate the true target extent rather than uncalibrated land regions.
"""
import pytest
import numpy as np
from PIL import Image

from models.single_image.normalizer import SpatialNormalizer
from models.single_image.benchmark_adapter import BenchmarkEvaluationAdapter
from models.single_image.adapter import GeoChatAdapter
from agent.planner import AgentQueryPlanner, AgentTaskType
from agent import agent_orchestrator


def test_raster_spectral_water_grounding():
    """Validates that extract_spatial_bounding_box_from_raster isolates the water region."""
    # Create synthetic test image: top half land (gray/green), bottom half deep water (dark blue)
    img_arr = np.zeros((400, 400, 3), dtype=np.uint8)
    # Land in top half: y=0..200
    img_arr[:200, :, 0] = 60
    img_arr[:200, :, 1] = 90
    img_arr[:200, :, 2] = 40
    # Water in bottom half: y=200..400 (RGB: 10, 35, 80)
    img_arr[200:, :, 0] = 10
    img_arr[200:, :, 1] = 35
    img_arr[200:, :, 2] = 80

    pil_img = Image.fromarray(img_arr)
    box = SpatialNormalizer.extract_spatial_bounding_box_from_raster(pil_img, "water")
    
    assert box is not None
    ymin, xmin, ymax, xmax = box
    # Water starts around y=50% and extends to 100%
    assert ymin >= 48.0
    assert ymax >= 98.0
    assert xmin <= 2.0
    assert xmax >= 98.0


def test_benchmark_water_body_grounding():
    """Validates that benchmark adapter produces accurate water body coordinates."""
    # Synthetic image with water in bottom half
    img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    img_arr[:100, :, :] = [60, 100, 40]
    img_arr[100:, :, :] = [10, 30, 90]
    pil_img = Image.fromarray(img_arr)

    adapter = BenchmarkEvaluationAdapter()
    res = adapter.ground_text_query(pil_img, "Highlight water body")
    
    assert len(res.bounding_boxes) >= 1
    gb = res.bounding_boxes[0]
    assert gb.box[0] >= 45.0  # Starts in southern half


def test_planner_grounding_routing():
    """Validates that phrase grounding queries route to Grounding tool."""
    plan = AgentQueryPlanner.plan("Highlight water body", {})
    assert plan.task_type == AgentTaskType.GROUNDING
    assert "Grounding" in plan.selected_tool_names

    plan2 = AgentQueryPlanner.plan("Find the water body in this image", {})
    assert plan2.task_type == AgentTaskType.GROUNDING

    plan3 = AgentQueryPlanner.plan("Where is the ocean basin located?", {})
    assert plan3.task_type == AgentTaskType.GROUNDING


def test_orchestrator_grounding_synthesis():
    """Validates that agent orchestrator synthesizes grounding boxes correctly."""
    import io
    img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    img_arr[:100, :, :] = [60, 100, 40]
    img_arr[100:, :, :] = [10, 30, 90]
    pil_img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    res = agent_orchestrator.orchestrate("Highlight water body", {"image": buf.getvalue()})
    assert res.grounding_boxes is not None
    assert len(res.grounding_boxes) >= 1
    
    box = res.grounding_boxes[0]
    assert box["box"][0] >= 45.0
