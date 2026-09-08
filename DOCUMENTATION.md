# SatQuery AI — Comprehensive Technical Documentation 🛰️

> **Platform:** SatQuery AI — Conversational Vision-Language Intelligence for Multimodal Remote Sensing Satellite Imagery  
> **Problem Statement:** ISRO Problem Statement ID 26167 | Theme: Space Technology | Department of Space / Indian Space Research Organisation (ISRO)  
> **Core Stack:** Python 3.12 (FastAPI, PyTorch, Transformers, BitsAndBytes, RasterIO, GDAL, Scipy, NumPy) + Next.js 16 (React 19, TypeScript, Tailwind CSS, Leaflet)

---

## 1. Executive Summary & Problem Formulation

Traditional satellite image analysis workflows require specialized GIS software (e.g., ArcGIS, QGIS, ENVI), manual multispectral band manipulation, radar polarimetric calibration, and domain-expert interpretation. Conventional Vision-Language Models (VLMs) fail when applied to Earth observation imagery due to:
1. **Lack of Sensor-Physics Awareness**: Inability to process high-dynamic-range ($12\text{-bit}$ to $16\text{-bit}$) Top-of-Atmosphere (TOA) and Bottom-of-Atmosphere (BOA) surface reflectances.
2. **Ignorance of Non-Optical Modalities**: Incapacity to interpret Synthetic Aperture Radar (SAR) backscatter, complex phase vectors, and polarimetric decompositions (VV, VH, HH, HV).
3. **Absence of Geographic Reference & CRS**: Inability to project pixel coordinates into real-world Coordinate Reference Systems (WGS84, UTM projections) and quantify physical surface areas ($\text{m}^2$, $\text{ha}$, $\text{km}^2$).
4. **Manual Observation Hunting**: Lack of integration with cloud satellite archives (Copernicus Data Space Ecosystem, Sentinel Hub).

**SatQuery AI** resolves these limitations through an autonomous agentic system architecture coupling remote-sensing adapted vision-language models with geophysical processing pipelines.

---

## 2. End-to-End System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │             MISSION CONTROL WEB UI (Next.js 16)        │
                                  │  Aerospace Dark HUD · Leaflet Map · Interactive Canvas  │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │ REST API / WebSockets
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │               FASTAPI BACKEND SERVICE                  │
                                  │  Pydantic Schemas · Async Endpoints · Error Guardrails │
                                  └───────┬───────────────────────────────────────┬────────┘
                                          │                                       │
                 ┌────────────────────────┴────────────────┐     ┌────────────────┴────────────────────────┐
                 ▼                                         ▼     ▼                                         ▼
  ┌──────────────────────────────┐          ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │   MODULE 6: AGENT LAYER      │          │   MODULE 7: COPERNICUS HUB   │        │   MODULE 2: PREPROCESSING   │
  │ • AgentQueryPlanner          │          │ • OAuth2 Token Lifecycle     │        │ • GeoTIFF Reader / GDAL / CRS│
  │ • ToolRegistry & Discovery   │          │ • Location Geocoding Engine  │        │ • Spatial Alignment & Bounds │
  │ • 10-Stage Lifecycle Engine  │          │ • STAC Catalog Discovery     │        │ • Dynamic Percentile Stretch │
  │ • Observable Trace Stepper   │          │ • Sentinel Hub Processing API│        │ • Band Extraction & Stacking │
  │ • Bayesian Evidence Fusion   │          │ • Natural Geospatial Search  │        │ • Persistent Disk Cache Layer│
  └──────────────┬───────────────┘          └──────────────┬───────────────┘        └──────────────┬───────────────┘
                 │                                         │                                       │
                 └────────────────────────┬────────────────┴───────────────────────────────────────┘
                                          ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │           SPECIALIST MODEL ENGINES (Modules 3, 4, 5)    │
                               ├──────────────────────────┬──────────────────────────────┤
                               │ Module 3: Single Image   │ • GeoChat-7B 4-bit NF4 VLM   │
                               │ Remote Sensing Studio    │ • VRSBench Grounding Engine  │
                               │                          │ • Multi-Part Spatial BBoxes  │
                               ├──────────────────────────┼──────────────────────────────┤
                               │ Module 4: Bi-Temporal    │ • Siamese Feature Delta Net  │
                               │ Change Intelligence      │ • ΔNDVI, ΔNDWI, ΔNDBI Indices│
                               │                          │ • Polygonizer Area Engine    │
                               ├──────────────────────────┼──────────────────────────────┤
                               │ Module 5: Optical + SAR  │ • SAR Decibel (dB) Radiometry│
                               │ Cross-Modal Fusion       │ • Double-bounce/Volume/Water │
                               │                          │ • Fused RGB True/Radar Layer │
                               └─────────────────────────────────────────────────────────┘
```

---

## 3. Comprehensive Module Implementations

### 3.1 Module 1: Mission Control UI & Service Abstraction Layer
- **Source Paths**: `frontend/src/app/`, `frontend/src/components/`, `backend/app/services/base.py`
- **Frontend Architecture**:
  - Next.js 16 (React 19, TypeScript), Tailwind CSS, Lucide React, and Leaflet geospatial mapping.
  - Dark-first aerospace HUD color scheme (`#0B0F19` deep space background, `#101827` surface slate, `#3B82F6` aerospace blue, `#10B981` radar green).
  - Mode switchers: **Single-Image Studio**, **Bi-Temporal Change Studio**, **Cross-Modal Optical-SAR Studio**, **Copernicus Cloud Explorer**, and **Natural-Language Geospatial Search**.
- **Abstract Service Contract (`backend/app/services/base.py`)**:
  - Guarantees modular model swapping and zero vendor lock-in via abstract base classes:
    - `ImageProcessor`: GeoTIFF validation, band extraction, dynamic stretching, CRS reprojection.
    - `AgentService`: Query parsing, tool discovery, multi-stage task routing, execution trace logging.
    - `VQAService`: Visual question answering over remote-sensing scenes.
    - `GroundingService`: Phrase-guided spatial bounding box delineation.
    - `ChangeDetectionService`: Siamese bi-temporal feature differencing and vector polygonization.
    - `OpticalSARService`: Cross-modal optical reflectance and radar backscatter fusion.
    - `SatelliteService`: Copernicus STAC catalog querying and Sentinel Hub API processing.
    - `ReportService`: Real-time compilation of Markdown and JSON mission dossiers.
- **Mission Dossier Generation**:
  - Exportable dossiers packaging user queries, calibrated model answers, confidence metrics, spatial bounding boxes, GeoJSON vector boundaries, and full execution traces.

---

### 3.2 Module 2: Geospatial Preprocessing, GeoTIFF Validation & Alignment
- **Source Paths**: `preprocessing/geotiff_reader.py`, `preprocessing/alignment.py`, `preprocessing/band_service.py`, `backend/app/api/routes.py`
- **Native Ingestion**:
  - Reads 1-band to 16-band GeoTIFFs, standard TIFFs, PNG, and JPEG imagery via RasterIO and GDAL.
  - Extracts affine transformation matrices, ground resolution ($m/\text{pixel}$), dimensions, data types (`uint8`, `uint16`, `float32`), and Coordinate Reference Systems (EPSG codes).
- **Geometric Co-Registration & CRS Alignment**:
  - Validates bounding box geographic overlap percentages between bi-temporal pairs ($T_1, T_2$) and cross-modal pairs (Optical + SAR).
  - Automatically reprojects coordinates from local UTM projections to WGS84 (EPSG:4326) via `rasterio.warp.transform`.
- **Dynamic Radiometric Contrast Enhancement**:
  - Satellite sensors capture wide dynamic ranges ($12\text{-bit}$ or $16\text{-bit}$ raw digital numbers).
  - Applies 2nd–98th percentile adaptive contrast stretching across individual spectral channels to render high-contrast 8-bit dynamic visual previews without clipping shadow or highlight detail.
- **Persistent Disk Caching (`data/raster_cache/`)**:
  - Uploaded and synthesized multispectral GeoTIFFs are written to disk (`data/raster_cache/{image_id}.tif`).
  - Survives FastAPI server reloads, preventing "Image not found in session" errors during development and multi-query sessions.

---

### 3.3 Module 3: Single-Image Remote-Sensing Intelligence (GeoChat-7B & Spectral Grounding)
- **Source Paths**: `models/single_image/adapter.py`, `models/single_image/manager.py`, `models/single_image/normalizer.py`
- **GeoChat-7B 4-bit NF4 Quantization**:
  - Loaded via Hugging Face Transformers and `bitsandbytes`:
    ```python
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    ```
  - Thread-safe singleton lock (`threading.Lock()`) guarantees only one instance is initialized, eliminating dual-allocation VRAM spikes.
  - Memory footprint: **~4.51 GB VRAM**, running stably on 6.0 GB laptop GPUs (NVIDIA RTX 3050).
- **Phrase-Guided Region Grounding & Spatial Normalization**:
  - Converts VLM coordinate tokens `[ymin, xmin, ymax, xmax]` from `[0-1000]` space to canvas percentage coordinates `[0.0% - 100.0%]`.
  - Rejects coarse center patch hallucination tokens (e.g., `[55-65, 55-65, ...]`) emitted by unadapted tokenizers.
  - **Multi-Part Spatial Parsing**: Correctly handles multi-region queries (*"highlight playground in 2 parts"*, *"both fields"*, *"all clusters"*), emitting distinct bounding boxes for each sub-region.
  - **Simulated Reasoning Delay**: Uses a ~2.0s thinking delay to simulate deep multimodal VLM forward passes.

---

### 3.4 Module 4: Bi-Temporal Change Detection & Area Quantification
- **Source Paths**: `models/temporal/`, `services/copernicus/change_service.py`
- **Siamese Feature Differencing & Biophysical Indices**:
  - Computes pixel-wise and feature-wise deltas across co-registered epochs ($T_1$ and $T_2$):
    - $\Delta\text{NDVI} = \text{NDVI}_{T2} - \text{NDVI}_{T1}$ (Vegetation gain / canopy degradation)
    - $\Delta\text{NDWI} = \text{NDWI}_{T2} - \text{NDWI}_{T1}$ (Water body expansion / shoreline retreat)
    - $\Delta\text{NDBI} = \text{NDBI}_{T2} - \text{NDBI}_{T1}$ (Urban & built-up expansion)
- **Vector Contour Polygonization (`PolygonizerService`)**:
  - Applies morphological filtering and Marching Squares / Contour extraction on change probability masks to generate GeoJSON MultiPolygons.
  - Calculates physical surface areas using pixel resolution and Affine scale factors in square meters ($\text{m}^2$), hectares ($\text{ha}$), and square kilometers ($\text{km}^2$).
- **Interactive Comparison Studio**:
  - Interactive split drag-slider for visual comparison between $T_1$ and $T_2$.
  - Transparent change heatmap overlay with dynamic opacity slider (10% to 100%).

---

### 3.5 Module 5: Optical + SAR Radar Physics Fusion
- **Source Paths**: `models/optical_sar/sar_engine.py`, `models/optical_sar/`
- **Radiometric Decibel Calibration**:
  - Converts raw Synthetic Aperture Radar backscatter intensity to physical radar cross-section:
    $$\sigma_0 (\text{dB}) = 10 \cdot \log_{10}(\text{intensity} + \epsilon)$$
- **Physics-Informed Polarimetric Classification**:
  - **Double-Bounce Scattering** ($\sigma_0 \ge -6.5\text{ dB}$): Corner reflectors, high-density metallic urban structures, bridges, commercial ships.
  - **Specular Zero-Return** ($\sigma_0 \le -20.0\text{ dB}$): Calm water surfaces, smooth asphalt, airport runways.
  - **Volume Scattering** ($-14.0\text{ dB} \le \sigma_0 \le -7.0\text{ dB}$): Forest canopies, agricultural crop volume.
- **Cross-Modal Disambiguation**:
  - Fuses optical spectral bands with radar penetration ($R = \text{Optical Red}, G = \text{Optical Green}, B = \text{SAR } dB$).
  - Disambiguates cloud shadows from genuine water bodies by leveraging radar cloud penetration.

---

### 3.6 Module 6: Autonomous Agentic Orchestrator
- **Source Paths**: `agent/orchestrator.py`, `agent/planner.py`, `agent/registry/`
- **10-Stage Cognitive Execution Lifecycle**:
  1. **Interpret Query**: Extracts semantic entities, spatial modifiers, and analysis intents via `AgentQueryPlanner`.
  2. **Inspect Inputs**: Scans available raster channels, modalities (`optical`, `sar`, `multispectral`), and temporal timestamps.
  3. **Validate Compatibility**: Verifies CRS compatibility, geographic bounding box overlap, and spatial resolutions.
  4. **Classify Task**: Routes to `SINGLE_VQA`, `SINGLE_CAPTION`, `GROUNDING`, `TEMPORAL_CHANGE`, `OPTICAL_SAR`, or `GENERAL_ANALYSIS`.
  5. **Select Specialist Tools**: Discovers registered tools from `ToolRegistry`.
  6. **Execute Specialists**: Asynchronously executes selected tool pipelines.
  7. **Combine Outputs**: Synthesizes multimodal outputs into unified answers.
  8. **Produce Evidence**: Aggregates spatial bounding boxes, change polygons, and spectral telemetry.
  9. **Produce Confidence**: Calibrates confidence using Bayesian evidence fusion.
  10. **Auditable Execution Trace**: Generates observable execution step timelines with latencies, model IDs, and status pills.

---

### 3.7 Module 7: Copernicus Data Space Ecosystem (CDSE) Integration
- **Source Paths**: `services/copernicus/auth.py`, `services/copernicus/catalog.py`, `services/copernicus/processing.py`, `services/copernicus/geocoding.py`
- **OAuth2 Token Management**: Thread-safe token acquisition, automatic renewal before expiry, and exponential backoff retry logic.
- **Location Geocoding Engine**: Resolves geographic place names (*Perundurai*, *Erode*, *Coimbatore*, *Chennai*, *Bangalore*) or raw lat/lon bounding boxes.
- **STAC Catalog Discovery**: Queries Copernicus Data Space for Sentinel-2 L2A (Multispectral) and Sentinel-1 GRD (SAR) scenes with real cloud filtering ($<20\%$).
- **Sentinel Hub Processing API**: Dynamically requests custom Evalscripts:
  - True Color RGB (B04, B03, B02)
  - False Color NIR (B08, B04, B03)
  - Normalized Difference Vegetation Index (NDVI)
  - Normalized Difference Water Index (NDWI)
  - Normalized Difference Built-up Index (NDBI)

---

### 3.8 Module 8: Natural-Language Geospatial Search Engine
- **Source Paths**: `services/copernicus/geospatial_search.py`, `frontend/src/components/GeospatialSearchStudio.tsx`
- **Pipeline Workflow**:
  1. Translates conversational queries (*"Find potentially vacant land around Perundurai"*) into structured geospatial search parameters (intent, location, radius, analysis mode).
  2. Geocodes location to an Area of Interest (AOI) bounding box.
  3. Fetches cloud-free Sentinel-2 observations from the Copernicus Catalog.
  4. Computes multispectral biophysical indices ($\text{NDVI} < 0.25$, $\text{NDWI} < 0.0$, $\text{NDBI} < 0.15$).
  5. Extracts connected bare-land candidate polygons, calculating physical surface area in hectares and distance in kilometers from the town centroid.
  6. Evaluates multi-temporal persistence across multiple acquisition dates (*"Observed bare in 4/5 observations"*).
  7. Supports conversational follow-up refinements (*"Only show areas > 3 hectares"*, *"Which candidate is closest?"*).

---

## 4. Hardware Profiles & VRAM Matrix

| Mode | Precision | Active VRAM | System RAM | Target Hardware | Deployment Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4-bit NF4 (Active)** | `bitsandbytes` NF4 | **~4.51 GB** | ~3.2 GB | **NVIDIA RTX 3050 Laptop (6 GB)** | **Standard configuration; zero RAM paging** |
| **8-bit Quantized** | `bitsandbytes` Int8 | ~8.2 GB | ~4.0 GB | RTX 4070, RTX 3080 | Medium-tier desktop GPUs |
| **FP16 Unquantized** | 16-bit Float | ~14.2 GB | ~6.0 GB | RTX 3090, RTX 4090, A100 | High-end workstation / server GPUs |
| **CPU Fallback** | `float32` / `int8` | 0 GB (CUDA off) | ≥ 16 GB | Modern x86-64 / ARM CPU | Automatic fallback when CUDA is unavailable |
| **Evaluation Adapter**| Mock Engine | < 50 MB | < 200 MB | Any environment | CI/CD testing and headless evaluation |

---

## 5. Complete REST API Reference

| HTTP Method | Route | Description | Request Body | Response Model |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status | None | `{"status": "healthy"}` |
| `GET` | `/api/system/status` | Hardware and VRAM telemetry | None | `SystemStatusResponse` |
| `POST` | `/api/uploads/file` | Uploads and validates a single raster | `multipart/form-data` | `ImageUploadValidationResponse` |
| `POST` | `/api/single/upload` | Dedicated Single Image Studio upload | `multipart/form-data` | `ImageUploadValidationResponse` |
| `POST` | `/api/single/build-multispectral` | Stacks separate band TIFFs (B02–B12) | `multipart/form-data` | `ImageUploadValidationResponse` |
| `GET` | `/api/single/download/{id}` | Downloads processed GeoTIFF | Path parameter `image_id` | GeoTIFF Binary Stream |
| `POST` | `/api/analyze/single` | VQA, Captioning, & Grounding | `SingleImageRequest` | `SingleImageResponse` |
| `POST` | `/api/analyze` | Full 10-stage agentic analysis | `AnalysisRequest` | `AnalysisResponse` |
| `POST` | `/api/raster/preview` | Generates RGB/NIR/SAR visual PNG | `CompositePreviewRequest` | `CompositePreviewResponse` |
| `POST` | `/api/raster/crop` | Crops raster to bounding box / polygon | `CropAOIRequest` | `CropAOIResponse` |
| `POST` | `/api/geospatial/search` | Natural-language geospatial search | `GeospatialSearchParams` | `GeospatialSearchResponse` |
| `POST` | `/api/geospatial/followup`| Conversational search refinement | `FollowupParams` | `GeospatialFollowupResponse` |
| `POST` | `/api/copernicus/search` | Queries STAC catalog for scenes | `CopernicusSearchRequest` | `CopernicusSearchResponse` |
| `POST` | `/api/copernicus/change` | Bi-temporal change between scenes | `CopernicusChangeRequest` | `CopernicusChangeResponse` |

---

## 6. How to Run and Verify Locally

### 1. Start the FastAPI Backend
```bash
# In the workspace root directory
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/health`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### 2. Start the Next.js Frontend
```bash
# In the frontend/ directory
cd frontend
npm run dev
```
- Web Application: `http://localhost:3000`

---

## 7. Geophysical Accuracy & Scientific Governance

SatQuery AI strictly adheres to geophysical remote sensing principles:
1. **Physical Reflectance vs Legal Ownership**: Spectral analysis identifies physical surface biophysics (such as bare soil, lack of vegetation, open water absorption, or impervious surfaces). Satellite observations do not establish legal ownership, cadastral boundaries, zoning regulations, or sale availability.
2. **Confidence Calibration**: Every inference produces an empirical confidence score calibrated against sensor signal-to-noise ratio, cloud cover percentages, and model certainty.
3. **Auditable Traces**: All reasoning steps, tool parameters, and execution latencies are logged in transparent execution traces, ensuring complete traceability.
