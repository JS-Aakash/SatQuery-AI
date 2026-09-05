# SatQuery AI

> **"Ask your satellite imagery anything."**  
> An interactive vision-language intelligence platform for multimodal remote sensing satellite imagery analysis (ISRO Problem Statement ID: 26167).

---

## 🛰️ Overview

**SatQuery AI** is an Earth-observation intelligence assistant capable of analyzing single optical/multispectral imagery, synthetic aperture radar (SAR), co-registered optical–SAR pairs, and bi-temporal image series through simple natural-language queries.

Rather than relying on generic vision-language models that lack sensor physics awareness, SatQuery AI utilizes an **agentic, query-driven orchestrator** that validates input rasters, routes queries to specialized remote-sensing models (RSVQA, VRSBench grounding, CDVQA multitemporal change, optical–SAR fusion), and synthesizes calibrated answers backed by an observable execution trace.

---

## 🏗️ Project Architecture

```
satquery-ai/
├── frontend/                  # Next.js 16 (React 19, TypeScript) + Tailwind CSS + Lucide + Recharts
│   ├── src/
│   │   ├── app/               # App Router: Mission Control UI (Home, Workspace, History, Evaluation)
│   │   ├── components/        # Shell, Sidebar, TopNav, UploadZone, SatelliteSearch, MapAOI,
│   │   │                      # QueryBox, ImageryViewer (split-slider), AIAnswerPanel, ReportModal
│   │   ├── lib/               # Resilient API client, Pydantic-matched types, SVG synthetic visualizers, presets
│   │   └── styles/            # Dark-first aerospace theme & telemetry tokens
├── backend/                   # Python FastAPI service
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint with CORS & route mounting
│   │   ├── config.py          # Environment settings
│   │   ├── api/routes.py      # Endpoints: /health, /system/status, /uploads/validate, /analyze, /analyses, /reports
│   │   ├── schemas/           # Pydantic schemas (Imagery, Analysis, Trace, SystemStatus, Evaluation, Reports)
│   │   └── services/          # Modular service layer with strict ABC contracts & demo providers
│   │       ├── base.py        # Abstract Base Classes for all 8 core services
│   │       ├── image_processor.py # Raster validation & metadata extraction
│   │       ├── agent_service.py   # Agentic query routing & 7-step execution trace
│   │       ├── satellite_service.py # AOI calculation & STAC catalog discovery
│   │       ├── report_service.py  # Markdown / JSON mission report generation
│   │       └── mock_providers.py  # Benchmark demo scenarios (RSVQA, VRSBench, CDVQA, Optical-SAR)
├── models/                    # [Module 3-5] Remote-sensing adapted models & model weights
├── agent/                     # [Module 2] Query parser, tool registry, and LangGraph planner
├── preprocessing/             # [Module 2] GeoTIFF band tiling, normalization, spatial co-registration
├── satellite/                 # [Module 2] STAC / Copernicus Open Access Hub API connectors
├── evaluation/                # [Module 6] RSVQA, VRSBench, CDVQA benchmark scoring harnesses
├── reports/                   # Saved mission reports & export artifacts
├── tests/                     # Backend automated unit test suite (pytest)
└── docs/                      # Technical architecture blueprints & model integration guides
```

---

## 🧩 Abstract Service Contracts (`backend/app/services/base.py`)

SatQuery AI is designed with strict abstract base classes so future models can be plugged in seamlessly without modifying the frontend or API routes:

| Service Interface | Description | Target Module |
| :--- | :--- | :--- |
| `ImageProcessor` | Geospatial raster validation, band extraction, CRS & co-registration | Module 2 (Preprocessing) |
| `AgentService` | Query intent classification, tool routing, 7-step execution trace | Module 2 (Agent Controller) |
| `VQAService` | Remote-sensing Visual Question Answering on single optical/SAR imagery | Module 3 (RSVQA / GeoChat) |
| `GroundingService` | Text-guided region grounding with bounding boxes (VRSBench) | Module 3 (Grounding) |
| `ChangeDetectionService`| Bi-temporal change description, change-VQA (CDVQA) & spatial masks | Module 4 (Multitemporal) |
| `OpticalSARService` | Cross-modal optical reflectance + SAR backscatter joint reasoning | Module 5 (Optical-SAR) |
| `SatelliteService` | Copernicus STAC catalog search and spherical polygon area computation | Satellite API Module |
| `ReportService` | Compiles analysis results, evidence, and execution trace into Markdown/JSON | Core Platform |

---

## 🚀 How to Run and Verify

### Prerequisites
- **Python 3.10+** (tested on Python 3.12)
- **Node.js 18+** (tested on Node v24) and `npm`

### 1. Run the FastAPI Backend

Open a terminal in the project root:
```bash
# Start backend on port 8000
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health:
- Browser: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Run tests: `python -m pytest tests -v` (8/8 tests passing)

### 2. Run the Next.js Frontend

Open a second terminal in the `frontend/` folder:
```bash
cd frontend
npm run dev
```

Open your browser to:
**[http://localhost:3000](http://localhost:3000)**

---

## 🧠 Module 3: Single-Image Remote-Sensing Intelligence

The **Single-Image Remote-Sensing Intelligence** module provides:
1. **Single-Image VQA** (RemoteSensingVQA): Natural-language questions answered over optical, multispectral, or SAR rasters.
2. **Scene Captioning & Description** (RemoteSensingCaptioning): Structured land-cover summaries and semantic scene overviews.
3. **Text-Guided Region Grounding** (RemoteSensingGrounding): Localizes phrases (e.g. *"Find the water body"*, *"Highlight the runway"*) into spatial bounding boxes `[ymin, xmin, ymax, xmax]` normalized to `0.0 - 100.0%`.

### Model Adapter Architecture (`models/single_image/`)

- **Interfaces (`models/single_image/interfaces.py`)**: Abstract base classes defining strict contracts (`answer_question`, `generate_caption`, `ground_regions`).
- **Lazy & Configurable Loader (`models/single_image/adapter.py`)**: `GeoChatAdapter` detects available CUDA hardware, verifies local checkpoint paths lazily without blocking server startup, and configures 4-bit NF4 quantization.
- **Hardware Telemetry & Manager (`models/single_image/manager.py`)**: `SingleImageModelManager` singleton automatically detects GPU device, VRAM allocation, and routes tasks to live weights or benchmark evaluation.
- **Truthful Status Reporting**: Does not fake inference. If weights are missing, accurately reports `WEIGHTS_NOT_FOUND` / `BENCHMARK_EVALUATION_ACTIVE` and serves benchmark evaluations.
- **Spatial Coordinate Normalizer (`models/single_image/normalizer.py`)**: Standardizes GeoChat `0-1000` tokens, relative `0.0-1.0` floats, and absolute pixel dimensions into common percentage coordinates.

### Dedicated Endpoints

- `POST /api/analyze/single`: Single-image VQA, scene captioning, and text-guided grounding analysis.
- `GET /api/models/status`: Telemetry status of GPU, VRAM capacity, driver, and model checkpoint detection.

---

## 🛰️ Module 4: Bi-Temporal Change Intelligence

The **Bi-Temporal Change Intelligence** module analyzes two spatially corresponding satellite images acquired at different times ($T_1$ and $T_2$, e.g., `before.tif` and `after.tif`):

1. **Spatial Validation & Alignment**: Validates geographic compatibility, CRS, dimensions, and spatial overlap percentage using `ImageAlignmentService`.
2. **8-Stage Change Pipeline (`models/change_detection/`)**:
   - Multi-temporal spectral differencing and Siamese feature extraction.
   - Physical spectral indices as supporting evidence: $\Delta\text{NDVI}$ (Canopy loss/gain), $\Delta\text{NDWI}$ (Water surface shift), $\Delta\text{NDBI}$ (Built-up/impervious expansion).
   - Vector contour polygonization (`PolygonizerService`) mapping individual changed zones.
   - Quantitative surface area estimation in square meters ($\text{m}^2$), hectares ($\text{ha}$), and square kilometers ($\text{km}^2$).
   - Evidence-grounded natural-language reasoning.
3. **Frontend Mission Control Tools**:
   - Interactive **Swipe Comparison Slider** with synchronized drag and split percentage.
   - Dynamic **Change Map Heatmap Overlay** with real-time opacity slider (10% to 100%).
   - Multi-class **Change Legend** (Built-up Expansion, Vegetation Loss, Water Contraction, Infrastructure).
   - Quantitative change statistics card with metric breakdowns and spectral delta badges.

### Dedicated Change Detection Endpoints

- `POST /api/analyze/change`: Runs bi-temporal change pipeline on provided `image_a_id` and `image_b_id` or base64 streams.

---

## 📡 Module 5: Cross-Modal Optical + SAR Intelligence

The **Cross-Modal Optical + SAR Intelligence** module analyzes co-registered optical/multispectral and Synthetic Aperture Radar (SAR) imagery together to unlock all-weather, multi-sensor insights:

1. **Validation & Geographic Co-Registration**:
   - Accepts pairs (`optical.tif`, `sar.tif`).
   - Validates CRS, spatial bounding overlap, resolution, dimensions, and modality types before execution.
   - Refuses to silently merge incompatible or misaligned rasters.
2. **Dedicated SAR Processing Engine (`models/optical_sar/sar_engine.py`)**:
   - Radiometric Decibel ($dB = 10\log_{10}(\text{intensity})$) transformation.
   - Non-destructive processing: never mutates raw files.
   - Polarimetric false-color rendering ($R = \text{VV}, G = \text{VH}, B = |\text{VV} - \text{VH}|$).
   - Physics-informed backscatter signature classification:
     - **Double-Bounce Scattering** ($\sigma_0 \ge -6.5\text{ dB}$): Metallic structures, buildings, bridges, ships.
     - **Specular Zero-Return** ($\sigma_0 \le -20.0\text{ dB}$): Calm open water surfaces, runways, flat tarmac.
     - **Volume Scattering** ($-14.0\text{ dB} \le \sigma_0 \le -7.0\text{ dB}$): Forest canopies and dense vegetation.
3. **Dedicated Optical Processing Engine (`models/optical_sar/optical_engine.py`)**:
   - True-color RGB visualization with dynamic range percentile stretch.
   - Multispectral spectral indices: $\text{NDVI}$ (Vegetation vigor), $\text{NDWI}$ (Water reflectance), $\text{NDBI}$ (Built-up index).
4. **Cross-Modal Evidence Fusion Pipeline (`models/optical_sar/fusion_pipeline.py`)**:
   - Clear 3-way source attribution: every piece of evidence is attributed to `"optical"`, `"sar"`, or `"combined"`.
   - All-weather penetrating reasoning (e.g. penetrating cloud covers or smoke via radar backscatter).
   - Material disambiguation (e.g. resolving optical cloud shadows vs real water bodies; distinguishing smooth black asphalt from open water).
   - Fused RGB composite generation ($R=\text{Optical Red}, G=\text{Optical Green}, B=\text{SAR } dB$).
5. **Frontend Multimodal Mission Control Viewer**:
   - 3-Layer synchronized viewport: **Optical Layer**, **SAR Radar Layer**, and **Fused Evidence Layer**.
   - Interactive layer switcher and opacity slider.
   - Distinct modality badges on the AI Answer Panel: `🛰️ OPTICAL`, `📡 SAR RADAR`, and `⚡ FUSED`.

### Dedicated Optical-SAR Endpoint

- `POST /api/analyze/optical-sar`: Runs cross-modal evidence fusion on co-registered optical + SAR imagery.

---

## 🤖 Module 6: Agentic Orchestration Layer

The **Agentic Orchestration Layer** is the core cognitive orchestrator of SatQuery AI, dynamically managing multi-sensor satellite reasoning:

1. **Autonomous 10-Stage Lifecycle**:
   1. **Interpret Query**: Parses semantic intention and entities via `AgentQueryPlanner`.
   2. **Inspect Inputs**: Scans uploaded rasters, band counts, and temporal/cross-modal availability.
   3. **Validate Compatibility**: Verifies CRS, spatial overlap, resolution, and coordinate alignment.
   4. **Classify Task**: Automatically maps to `SINGLE_VQA`, `SINGLE_CAPTION`, `GROUNDING`, `TEMPORAL_CHANGE`, `OPTICAL_SAR`, or `GENERAL_ANALYSIS`.
   5. **Select Specialist Tools**: Discovers candidate specialists via `ToolRegistry`.
   6. **Execute Tools**: Dispatches to specialists (`SingleImageVQA`, `Captioning`, `Grounding`, `ChangeDetection`, `OpticalSAR`, `SpectralAnalysis`).
   7. **Combine Outputs**: Synthesizes multimodal outputs into unified answers.
   8. **Produce Evidence**: Aggregates spatial bounding boxes, change polygons, and spectral index deltas.
   9. **Produce Confidence**: Evaluates multi-sensor cross-entropy agreement and calibrated confidence.
   10. **Auditable Execution Trace**: Generates observable execution metadata (Step, Tool, Model, Duration, Parameters, Output Summary) without exposing private chain-of-thought.
2. **Specialist Tool Registry (`agent/registry/`)**:
   - Standardized abstract contract: `name`, `description`, `accepted_inputs`, `supported_tasks`, `execute()`, `output_schema`, `is_available()`.
   - Strict honesty guardrails: Never claims a tool ran when it did not; reports missing or failed tools truthfully without fabricating outputs.
3. **Multi-Tool Compound Query Reasoning**:
   - Compound queries (e.g. *"Has vegetation decreased and where did change occur?"*) seamlessly coordinate multiple specialist tools in sequence (Bi-Temporal Change Detection + Spectral NDVI Biophysics + Visual Question Answering).
4. **Mission Control Trace Timeline**:
   - High-contrast aerospace execution stepper with live status pills (`COMPLETED`, `FAILED`, `UNAVAILABLE`, `SKIPPED`).
   - Detailed parameter inspection, specialist model badges, and execution duration in milliseconds.

---

## ⚡ Hardware Profile & VRAM Requirements

| Execution Mode | Model Precision | Expected VRAM / RAM | Target Hardware | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Full FP16** | 16-bit Float | ~14.2 GB VRAM | RTX 3090, RTX 4090, A100 | Full precision unquantized inference |
| **8-Bit Quantized** | `int8` (bitsandbytes) | ~8.4 GB VRAM | RTX 4070 (8-12 GB) | Minor quantization loss |
| **4-Bit NF4 (Default)**| `bitsandbytes` NF4 | **~5.2 GB VRAM** | **NVIDIA RTX 3050 Laptop (6 GB VRAM)** | **Recommended: Fits comfortably within 6GB VRAM** |
| **CPU Fallback** | `int8` / `float32` | ≥ 16 GB System RAM | Any multi-core CPU | Automatically engages when CUDA unavailable |
| **Benchmark Adapter** | Offline Evaluation | < 100 MB RAM | Any machine | Operates via RSVQA/VRSBench reference evaluations |

### How to Download GeoChat-7B Model Weights

To run live GPU inference with GeoChat-7B:
```bash
# 1. Download model repository to models/geochat-7b (approx. 13.5 GB)
hf download MBZUAI/geochat-7b --local-dir models/geochat-7b

# 2. Or set a custom weights path via environment variable:
export GEOCHAT_MODEL_PATH="models/geochat-7b"
# On Windows PowerShell:
$env:GEOCHAT_MODEL_PATH="models/geochat-7b"
```

---

## 🔍 Verification Flow in the Web UI

1. **New Analysis (Home)**:
   - Notice the **Dark-First Mission Control** design, telemetry indicators, and the 3 input modes:
     1. **Upload Imagery**: Drag & drop single GeoTIFF/PNG, bi-temporal pairs, or optical+SAR pairs. Click any of the 4 evaluation presets.
     2. **Satellite Discovery Search**: Filter by location, date range, Sentinel-1/2, cloud cover.
     3. **Interactive Map / AOI**: Search locations, zoom, view live lat/lon, draw bounding boxes, and calculate area in km².
2. **Query Execution**:
   - Single VQA: *"What type of terrain is shown in the image?"*
   - Grounding: *"Highlight the water bodies."* or *"Find the harbor port."*
   - Captioning: *"Describe the land-cover and major objects visible in this image."*
   - Change Detection: *"What changed between the two dates and where?"*
   - Optical+SAR Fusion: *"Identify built-up and water-covered regions using both images."*
   - Multi-Tool Compound: *"Has vegetation decreased and where did change occur between the dates?"*
   - Click **ANALYZE** (or press `Enter ↵`).
3. **Model Inference Loading State**:
   - Real-time telemetry card displaying raster ingestion, VLM visual attention, cross-modal evidence fusion, and spatial grounding normalization steps.
4. **Analysis Result Studio**:
   - **Large Imagery Viewer**: Grounding bounding boxes and fused layers rendered directly on the imagery with labels, confidence %, and interactive focus.
   - **AI Answer Panel**: Displays synthesized answer, calibrated confidence, detected spatial regions list, modality badges (`🛰️ OPTICAL`, `📡 SAR RADAR`, `⚡ FUSED`), deployed specialist models, and latency.
   - **Auditable Execution Trace**: Step-by-step observable metadata cards with status pills, parameters, model IDs, and execution durations.
   - **Interactive Region Highlighting**: Hover or click a detected region card in the answer panel to focus the corresponding bounding box in the imagery canvas.
5. **System Diagnostics**:
   - Click **SYSTEM DIAGNOSTICS** to inspect live GPU device, VRAM allocation, and adapter readiness.
6. **Automated Unit Tests**:
   ```bash
   python -m pytest tests -v
   ```
   All 38 automated unit tests pass across backend, preprocessing, single-image intelligence, change detection, optical-SAR intelligence, and the agentic orchestrator.
