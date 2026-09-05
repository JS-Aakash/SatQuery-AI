"""
Unit test suite for Module 3: Single-Image Remote-Sensing Intelligence.
Tests SpatialNormalizer, GeoChatAdapter, BenchmarkEvaluationAdapter,
SingleImageModelManager, request/response validation, and API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from models.single_image import (
    SpatialNormalizer,
    GeoChatAdapter,
    BenchmarkEvaluationAdapter,
    SingleImageModelManager,
    model_manager,
    SingleImageRequest,
    SingleImageResponse,
    SingleImageTaskEnum,
)

client = TestClient(app)


def test_spatial_coordinate_normalization():
    """Verify normalization of coordinates from various VLM output formats to 0-100%."""
    # 1. Standard GeoChat 0-1000 integer token scale
    raw_1000 = [150.0, 200.0, 650.0, 750.0]
    norm_1000 = SpatialNormalizer.normalize_box(raw_1000)
    assert norm_1000 == [15.0, 20.0, 65.0, 75.0]

    # 2. Relative float 0.0 - 1.0 scale
    raw_float = [0.12, 0.34, 0.56, 0.78]
    norm_float = SpatialNormalizer.normalize_box(raw_float)
    assert norm_float == [12.0, 34.0, 56.0, 78.0]

    # 3. Absolute pixel coordinates (e.g. 512x512 image)
    raw_pixels = [128.0, 64.0, 384.0, 256.0]
    norm_pixels = SpatialNormalizer.normalize_box(raw_pixels, img_width=512, img_height=512)
    assert norm_pixels == [25.0, 12.5, 75.0, 50.0]

    # 4. Inverted coordinates correction (ymin > ymax)
    raw_inverted = [70.0, 80.0, 20.0, 10.0]
    norm_inverted = SpatialNormalizer.normalize_box(raw_inverted)
    assert norm_inverted[0] <= norm_inverted[2]  # ymin <= ymax
    assert norm_inverted[1] <= norm_inverted[3]  # xmin <= xmax
    assert norm_inverted == [20.0, 10.0, 70.0, 80.0]

    # 5. Invalid length error
    with pytest.raises(ValueError):
        SpatialNormalizer.normalize_box([10.0, 20.0])


def test_grounding_box_parser():
    """Verify regex extraction of embedded bounding box tokens in generated model text."""
    vlm_output = (
        "The water reservoir is located at [580, 220, 880, 720]. "
        "Additionally, an industrial complex was detected at [200, 400, 580, 800]."
    )
    boxes = SpatialNormalizer.parse_boxes_from_text(vlm_output)
    assert len(boxes) == 2
    assert boxes[0].box == [58.0, 22.0, 88.0, 72.0]
    assert boxes[1].box == [20.0, 40.0, 58.0, 80.0]

    # Test GeoChat inline sentence tokens with bracketed angle/confidence:
    geochat_live_output = (
        "In the satellite image, there are some buildings {<10><1><14><5>|<90>}{<10><0><14><6>|<90>} "
        "located close to each other at the top left part of the scene. These buildings can be observed in the image."
    )
    geochat_boxes = SpatialNormalizer.parse_boxes_from_text(geochat_live_output)
    assert len(geochat_boxes) == 2
    assert geochat_boxes[0].box == [10.0, 1.0, 14.0, 5.0]
    assert geochat_boxes[1].box == [10.0, 0.0, 14.0, 6.0]
    assert "Buildings" in geochat_boxes[0].label

    # Test clean_vlm_text strips tokens cleanly
    cleaned = SpatialNormalizer.clean_vlm_text(geochat_live_output)
    assert "{" not in cleaned and "<" not in cleaned
    assert "In the satellite image, there are some buildings located close to each other at the top left part of the scene." in cleaned


def test_model_adapter_lazy_loading_and_status():
    """Verify GeoChatAdapter lazy loading and truthful status reporting when weights are missing."""
    adapter = GeoChatAdapter(model_path="models/nonexistent-checkpoint")
    assert adapter.is_weights_available() is False

    # Calling answer_question when weights are not downloaded should report status truthfully
    res = adapter.answer_question(b"dummy_image", "What land-cover is visible?")
    assert res.model_status == "WEIGHTS_NOT_FOUND"
    assert "not located" in res.status_message.lower()
    assert res.answer is not None
    assert len(res.answer) > 10


import io
import numpy as np
from PIL import Image


def _make_test_image_bytes():
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :, :] = [50, 120, 40]
    arr[64:, :, :] = [10, 30, 100]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_request_and_output_schema_validation():
    """Verify Pydantic request and response schemas."""
    test_img = _make_test_image_bytes()
    req = SingleImageRequest(
        image_id="img_12345",
        query="Highlight the water body",
        task=SingleImageTaskEnum.GROUNDING
    )
    assert req.task == "grounding"

    resp = model_manager.infer(
        image_input=test_img,
        query=req.query,
        task=req.task
    )
    assert isinstance(resp, SingleImageResponse)
    assert resp.task == "Text-Guided Region Grounding"
    assert resp.confidence > 0.8
    assert len(resp.bounding_boxes) >= 1
    assert len(resp.evidence_metadata) >= 1


def test_vqa_captioning_grounding_tasks():
    """Verify model manager handles VQA, Captioning, and Grounding tasks."""
    test_img = _make_test_image_bytes()
    # 1. VQA task
    res_vqa = model_manager.infer(
        image_input=test_img,
        query="Is there an urban settlement here?",
        task=SingleImageTaskEnum.VQA
    )
    assert res_vqa.task == "Visual Question Answering"
    assert len(res_vqa.answer) > 0

    # 2. Captioning task
    res_cap = model_manager.infer(
        image_input=test_img,
        query="Describe the scene.",
        task=SingleImageTaskEnum.CAPTIONING
    )
    assert res_cap.task == "Scene Captioning & Description"
    assert len(res_cap.answer) > 0

    # 3. Grounding task
    res_grd = model_manager.infer(
        image_input=test_img,
        query="Find the water body",
        task=SingleImageTaskEnum.GROUNDING
    )
    assert res_grd.task == "Text-Guided Region Grounding"
    assert len(res_grd.bounding_boxes) >= 1


def test_api_analyze_single_endpoint():
    """Verify FastAPI POST /api/analyze/single endpoint."""
    test_img = _make_test_image_bytes()
    import base64
    b64_str = f"data:image/png;base64,{base64.b64encode(test_img).decode('utf-8')}"
    payload = {
        "query": "Highlight the water body referred to in the query.",
        "task": "grounding",
        "image_data_uri": b64_str
    }
    res = client.post("/api/analyze/single", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "Text-Guided Region Grounding"
    assert "bounding_boxes" in data
    assert len(data["bounding_boxes"]) >= 1
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0


def test_hardware_status_endpoint():
    """Verify GET /api/models/status returns hardware VRAM and adapter readiness."""
    res = client.get("/api/models/status")
    assert res.status_code == 200
    data = res.json()
    assert "device" in data
    assert "vram_gb" in data
    assert "status" in data
