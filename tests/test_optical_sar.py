"""
Unit and Integration Tests for Module 5: Cross-Modal Optical + SAR Intelligence.
Tests SAR radiometric decibel processing (double-bounce, specular water, volume scattering),
optical multispectral feature extraction, cross-modal evidence fusion, modality attribution,
and the POST /api/analyze/optical-sar endpoint.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from models.optical_sar import (
    optical_sar_manager,
    SAREngine,
    OpticalEngine,
    CrossModalFusionPipeline,
)

client = TestClient(app)


def test_sar_engine_radiometric_classification():
    """Verify SAR dB conversion and backscatter physics classification."""
    # Synthetic 2-channel SAR array (VV, VH in linear intensity)
    sar_linear = np.ones((50, 50, 2), dtype=np.float32) * 0.001  # -30 dB (Water)
    sar_linear[10:30, 10:30, 0] = 0.5  # High return ~ -3 dB (Double-bounce building)

    db_arr, vis_arr, preview_url = SAREngine.process_sar_input(sar_linear)
    assert db_arr.shape == (50, 50, 2)
    assert vis_arr.shape == (50, 50, 3)
    assert preview_url.startswith("data:image/png;base64")

    classes = SAREngine.classify_radar_signatures(db_arr)
    assert np.any(classes["double_bounce"]), "Should identify double bounce region"
    assert np.any(classes["specular_water"]), "Should identify specular water"


def test_optical_engine_spectral_masks():
    """Verify optical normalization and spectral segmentation."""
    # Synthetic optical RGB image (H, W, 3) with distinct vegetation and water zones
    opt_img = np.zeros((60, 60, 3), dtype=np.uint8)
    opt_img[0:30, :, :] = [30, 180, 40]  # Green Vegetation
    opt_img[30:60, :, :] = [10, 40, 160]  # Deep Blue Water

    norm, uint8_arr, preview_url = OpticalEngine.process_optical_input(opt_img)
    masks = OpticalEngine.compute_spectral_masks(norm)

    assert np.any(masks["vegetation"]), "Vegetation mask should be active in top half"
    assert np.any(masks["water"]), "Water mask should be active in bottom half"


def test_cross_modal_evidence_attribution():
    """Verify optical, SAR, and combined evidence source attribution."""
    opt_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    sar_img = np.ones((100, 100, 2), dtype=np.float32) * 0.05

    res = optical_sar_manager.infer(
        optical_input=opt_img,
        sar_input=sar_img,
        query="What information does SAR provide that optical does not?"
    )

    assert res.task == "Cross-Modal Optical-SAR Analysis"
    assert res.confidence >= 0.90
    assert len(res.optical_evidence) > 0, "Must have optical evidence items"
    assert len(res.sar_evidence) > 0, "Must have SAR evidence items"
    assert len(res.combined_evidence) > 0, "Must have combined fused evidence items"

    # Check that sources are strictly attributed
    for ev in res.optical_evidence:
        assert ev.modality_source == "optical"
    for ev in res.sar_evidence:
        assert ev.modality_source == "sar"
    for ev in res.combined_evidence:
        assert ev.modality_source == "combined"


def test_optical_sar_api_endpoint():
    """Verify POST /api/analyze/optical-sar endpoint."""
    payload = {
        "query": "Identify built-up and water-covered regions using both images.",
        "parameters": {}
    }
    response = client.post("/api/analyze/optical-sar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert "optical_evidence" in data
    assert "sar_evidence" in data
    assert "combined_evidence" in data
    assert "fused_composite_url" in data
    assert len(data["grounding_regions"]) > 0
    assert data["model_status"] == "READY"
