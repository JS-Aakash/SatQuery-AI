# SatQuery AI 🛰️

> **"Ask your satellite imagery anything."**  
> An interactive conversational vision-language intelligence platform for multimodal remote sensing satellite imagery analysis.  
> **ISRO Problem Statement ID:** 26167 | **Theme:** Space Technology | **Department:** Department of Space / Indian Space Research Organisation (ISRO)

---

## 📖 Executive Summary

**SatQuery AI** is an end-to-end, agentic Earth-observation intelligence platform designed to bridge the gap between complex remote-sensing physics and natural-language interaction. Conventional AI assistants and generic Vision-Language Models (VLMs) lack sensor-physics awareness, geographic coordinates understanding, SAR polarimetry, multispectral band processing, and spatial co-registration.

SatQuery AI solves this through an **autonomous, query-driven agentic orchestrator** that:
1. Ingests single optical/multispectral rasters, Synthetic Aperture Radar (SAR), co-registered optical–SAR pairs, and bi-temporal image series (GeoTIFF/TIFF, PNG, JPEG).
2. Connects directly to **Copernicus Data Space Ecosystem (CDSE)** and **Sentinel Hub APIs** for automatic Earth observation retrieval without manual dataset hunting.
3. Automatically interprets natural-language user queries, validates geospatial compatibility, routes tasks across a standardized **Specialist Tool Registry**, and executes remote-sensing adapted models (GeoChat-7B, VRSBench Grounding, RSVQA, CDVQA Siamese ChangeNet, Optical–SAR Radar Physics Engine).
4. Delivers evidence-grounded textual answers, quantitative surface area metrics ($\text{m}^2$, $\text{ha}$, $\text{km}^2$), spatial bounding boxes, vector candidate polygons, transparent raster overlays, and an **observable, auditable execution trace**.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │             MISSION CONTROL WEB UI (Next.js 16)         │
                                  │   Dark-First Aerospace Interface · Leaflet Map · HUD   │
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
  │ • 10-Stage Lifecycle Orchestr│          │ • STAC Catalog Discovery     │        │ • Dynamic Percentile Stretch │
  │ • Observable Trace Timeline  │          │ • Sentinel Hub Processing API│        │ • Band Extraction & Stacking │
  │ • Bayesian Evidence Fusion   │          │ • Natural Geospatial Search  │        │ • Non-Destructive Ingestion  │
  └──────────────┬───────────────┘          └──────────────┬───────────────┘        └──────────────┬───────────────┘
                 │                                         │                                       │
                 └────────────────────────┬────────────────┴───────────────────────────────────────┘
                                          ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │           SPECIALIST MODEL ENGINES (Modules 3, 4, 5)    │
                               ├──────────────────────────┬──────────────────────────────┤
                               │ Module 3: Single Image   │ • GeoChat-7B VQA & Captioning│
                               │ Remote Sensing VLM       │ • VRSBench Text-Guided Box   │
                               ├──────────────────────────┼──────────────────────────────┤
                               │ Module 4: Bi-Temporal    │ • Siamese Feature Delta Net  │
                               │ Multitemporal ChangeNet  │ • ΔNDVI, ΔNDWI, ΔNDBI Indices│
                               │                          │ • Polygonizer Area Engine    │
                               ├──────────────────────────┼──────────────────────────────┤
                               │ Module 5: Optical + SAR  │ • SAR Decibel (dB) Radiometry│
                               │ Cross-Modal Fusion       │ • Double-bounce/Volume/Water │
                               │                          │ • Fused RGB True/Radar Layer │
                               └─────────────────────────────────────────────────────────┘
```

---

## 📊 Comprehensive Module Implementation Breakdown

### 🖥️ Module 1: Core Platform Architecture, Mission Control GUI & Abstract Service Layer
- **High-Contrast Aerospace Interface**: Built with Next.js 16 (React 19, TypeScript), Tailwind CSS, Lucide Icons, and Recharts. Features dark-first HUD telemetry tokens, interactive raster canvases, and live processing steppers.
- **Abstract Base Service Architecture (`backend/app/services/base.py`)**: Defines strict abstract interfaces ensuring zero vendor lock-in and seamless model pluggability:
  - `ImageProcessor`: Raster validation and coordinate alignment.
  - `AgentService`: Query classification, tool dispatch, and trace logging.
  - `VQAService`: Single-image optical/SAR visual question answering.
  - `GroundingService`: Phrase-grounded bounding box localization.
  - `ChangeDetectionService`: Bi-temporal change description and quantitative polygonization.
  - `OpticalSARService`: Cross-modal spectral reflectance + radar backscatter fusion.
  - `SatelliteService`: STAC discovery and spherical polygon area computation.
  - `ReportService`: Real-time compilation of Markdown and JSON mission dossiers.
- **Mission Dossier Export**: Compiles analysis answers, confidence scores, detected spatial regions, and execution traces into downloadable `.md` and `.json` artifacts.

---

### 🗺️ Module 2: Geospatial Preprocessing, GeoTIFF Validation & Co-Registration Engine
- **Geospatial Format Ingestion**: Natively processes 1-band to 16-band GeoTIFFs, standard TIFFs, PNG, and JPEG imagery.
- **CRS & Geometric Alignment (`preprocessing/alignment.py`)**:
  - Automatically reads Affine transforms and Geographic/Projected Coordinate Reference Systems (WGS84, UTM zones).
  - Validates spatial overlap percentages between image pairs ($T_1/T_2$ or Optical/SAR) and refuses to silently merge misaligned rasters.
- **Dynamic Radiometric Enhancement**: Converts raw 16-bit Top-of-Atmosphere (TOA) and Bottom-of-Atmosphere (BOA) reflectances to 8-bit dynamic ranges via 2nd–98th percentile contrast stretching.
- **Integrity Guarantee**: Strictly non-destructive—source satellite rasters are never mutated.

---

### 🧠 Module 3: Single-Image Remote-Sensing Intelligence
Adapted for Earth-observation characteristics using BigEarthNet, RSVQA, and VRSBench benchmarks:
1. **Remote-Sensing VQA (RSVQA)**: Answers open-ended natural-language inquiries regarding terrain type, maritime infrastructure, agricultural status, and object counts.
2. **Dense Scene Captioning**: Produces structured land-cover breakdowns, contextual environmental descriptions, and semantic scene overviews.
3. **Text-Guided Region Grounding (VRSBench)**: Localizes specific phrases (e.g. *"Highlight the water body"*, *"Find the harbor port"*, *"Locate the runway"*) into spatial bounding boxes `[ymin, xmin, ymax, xmax]` normalized to $0.0\% - 100.0\%$.
4. **Adapter Architecture (`models/single_image/`)**:
   - `GeoChatAdapter`: Supports lazy loading, 4-bit NF4 quantization on NVIDIA GPUs (e.g. RTX 3050 Laptop 6GB), CPU fallback, and automated benchmark evaluation.
   - `SpatialNormalizer`: Converts raw VLM coordinate tokens `[0-1000]` into normalized canvas percentages.
   - Truthful status reporting: Returns `READY`, `WARM_READY`, or `WEIGHTS_NOT_FOUND` without fabricating neural weights.

---

### 🛰️ Module 4: Bi-Temporal Change Intelligence
Analyzes paired rasters acquired across different epochs ($T_1$ and $T_2$):
1. **8-Stage Change Workflow**:
   - Multi-temporal spectral differencing and Siamese feature extraction.
   - Physical spectral indices as supporting evidence: $\Delta\text{NDVI}$ (Canopy loss/gain), $\Delta\text{NDWI}$ (Water surface shift), $\Delta\text{NDBI}$ (Built-up expansion).
   - Vector contour polygonization (`PolygonizerService`) mapping individual changed zones.
   - Quantitative surface area estimation in square meters ($\text{m}^2$), hectares ($\text{ha}$), and square kilometers ($\text{km}^2$).
   - Evidence-grounded natural-language reasoning.
2. **Interactive Comparison Studio**:
   - **Split Comparison Slider**: Synchronized drag slider to visually wipe between $T_1$ (Before) and $T_2$ (After).
   - **Change Heatmap Overlay**: Colored transparent alpha mask with dynamic opacity slider (10% to 100%).
   - **Class Legend & Quantitative Metrics**: Color-coded categorization (Built-up Expansion, Vegetation Loss, Water Contraction, Infrastructure).

---

### 📡 Module 5: Cross-Modal Optical + SAR Intelligence
Performs joint reasoning over co-registered optical/multispectral and Synthetic Aperture Radar (SAR) imagery:
1. **Pair Compatibility Validation**: Verifies CRS, bounding box overlap, pixel dimensions, and modality types (`optical` + `sar`).
2. **Dedicated SAR Processing Engine (`models/optical_sar/sar_engine.py`)**:
   - Radiometric Decibel ($dB = 10\log_{10}(\text{intensity})$) calibration.
   - Polarimetric false-color rendering ($R = \text{VV}, G = \text{VH}, B = |\text{VV} - \text{VH}|$).
   - Physics-informed radar backscatter classification:
     - **Double-Bounce Scattering** ($\sigma_0 \ge -6.5\text{ dB}$): Urban structures, metallic buildings, bridges, cargo ships.
     - **Specular Zero-Return** ($\sigma_0 \le -20.0\text{ dB}$): Calm water bodies, airport runways, smooth tarmac.
     - **Volume Scattering** ($-14.0\text{ dB} \le \sigma_0 \le -7.0\text{ dB}$): Forest canopies and dense agricultural crops.
3. **Cross-Modal Evidence Fusion**:
   - Disambiguates optical cloud shadows from genuine water bodies using penetrating radar backscatter.
   - Generates fused true/radar RGB composites ($R=\text{Optical Red}, G=\text{Optical Green}, B=\text{SAR } dB$).
   - Provides clear 3-way source attribution (`"optical"`, `"sar"`, `"combined"`).

---

### 🤖 Module 6: Agentic Orchestration Layer
The core cognitive orchestrator managing multi-sensor satellite reasoning:
1. **Autonomous 10-Stage Lifecycle**:
   1. *Interpret Query*: Semantic entity extraction via `AgentQueryPlanner`.
   2. *Inspect Inputs*: Scans raster count, bands, and temporal/cross-modal availability.
   3. *Validate Compatibility*: Verifies CRS, spatial overlap, resolution, and dimensions.
   4. *Classify Task*: Automatically maps to `SINGLE_VQA`, `SINGLE_CAPTION`, `GROUNDING`, `TEMPORAL_CHANGE`, `OPTICAL_SAR`, or `GENERAL_ANALYSIS`.
   5. *Select Specialist Tools*: Discovers candidate tools from `ToolRegistry`.
   6. *Execute Specialists*: Dispatches requests to registered specialist tools.
   7. *Combine Outputs*: Aggregates multimodal findings into unified responses.
   8. *Produce Evidence*: Collects spatial bounding boxes, change polygons, and spectral deltas.
   9. *Produce Confidence*: Bayesian confidence calibration across executed specialists.
   10. *Auditable Execution Trace*: Produces observable execution metadata without leaking private chain-of-thought tokens.
2. **Specialist Tool Registry (`agent/registry/`)**: Standardized interface (`SingleImageVQA`, `Captioning`, `Grounding`, `ChangeDetection`, `OpticalSAR`, `SpectralAnalysis`).
3. **Multi-Tool Compound Queries**: Coordinates complex sequences (e.g. *"Has vegetation decreased and where did change occur?"* $\rightarrow$ Bi-Temporal ChangeNet + Spectral $\text{NDVI}$ Biophysics + VQA Synthesis).
4. **Observable Execution Trace Timeline**: High-contrast execution stepper with status pills (`COMPLETED`, `FAILED`, `UNAVAILABLE`, `SKIPPED`), parameters, model IDs, and execution latency.

---

### 🌍 Module 7: Automatic Earth Observation Retrieval & Copernicus Data Space
Eliminates manual satellite downloading through automated cloud discovery:
1. **OAuth2 Token Lifecycle (`services/copernicus/auth.py`)**: Client credentials authentication with automated caching, thread-safe token renewal, and graceful rate-limit handling.
2. **Location Geocoding Engine (`services/copernicus/geocoding.py`)**: Converts location names (*Perundurai*, *Erode*, *Coimbatore*, *Chennai*, *Bangalore*, or direct coordinates) into bounding boxes and centroid coordinates.
3. **STAC Catalog Discovery (`services/copernicus/catalog.py`)**: Queries Copernicus Data Space Ecosystem for Sentinel-2 L2A (Multispectral) and Sentinel-1 GRD (SAR) scenes with real cloud filtering ($<20\%$).
4. **Sentinel Hub Processing API (`services/copernicus/processing.py`)**: Fetches spectral composites:
   - **True Color (RGB)**: Visual bands B04, B03, B02.
   - **False Color Infrared**: Bands B08 (NIR), B04 (Red), B03 (Green) for canopy vigor.
   - **NDVI Vegetation Index**: $\frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$ with mean, min, max, std-dev, and density breakdowns.
   - **NDWI Water Index**: $\frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$ for hydrological surface delineation.
   - **NDBI Built-Up Index**: $\frac{\text{B11} - \text{B08}}{\text{B11} + \text{B08}}$ for impervious surface density.
5. **Observation Quick Preview**: Interactive preview viewport in the Copernicus Catalog Explorer allowing visual inspection of satellite observations before selecting them for bi-temporal analysis.
6. **Human-Understandable Change Descriptions**: Translates raw pixel deltas into clear, natural language summaries (e.g. *"Agricultural vegetation increased by +18.4 hectares (+12.3%)"*).

---

### 🔎 Module 8 / Feature Spotlight: Natural-Language Geospatial Search
Converts conversational geographic queries into real satellite search and land analysis:

```
User Query: "Find potentially vacant land around Perundurai"
                    │
                    ▼
       ┌─────────────────────────┐
       │   Natural Language      │
       │   Query Parser (AI)     │
       └────────────┬────────────┘
                    │
                    ▼ Structured Geospatial Query:
                      {
                        "intent": "bare_land_search",
                        "location": "Perundurai",
                        "radius_km": 10,
                        "analysis": "bare_land_detection"
                      }
                    │
                    ▼
       ┌─────────────────────────┐
       │  Geospatial Search      │
       │  Execution Engine       │
       └────────────┬────────────┘
                    ├── Geocode location -> Bounding Box AOI
                    ├── Copernicus Catalog -> Sentinel-2 Observations
                    ├── Sentinel Hub Processing -> Real Spectral Indices (NDVI, NDWI, NDBI)
                    ├── Land Classification -> Bare/Non-Vegetated Pixels
                    ├── Polygon Extraction -> Connected Vector Boundaries
                    ├── Metric Calculations -> Area (ha), Distance (km)
                    └── Multi-Temporal Persistence -> Multi-Date Comparison
                    │
                    ▼
       ┌─────────────────────────┐
       │  Interactive Results    │
       ├─────────────────────────┤
       │ 📍 Interactive Leaflet  │ -> Candidate Polygons with Popups & Metadata
       │ 📊 Ranked Cards         │ -> Area (ha), Distance (km), Persistence, NDVI
       │ 💬 Conversational Chat  │ -> Natural Language Follow-ups & Filtering
       │ ⚠️ Legal Disclaimer     │ -> Physical Bare-Land vs Legal Vacancy Notice
       └─────────────────────────┘
```

#### Core Capabilities:
- **Physical Bare-Land Candidate Detection**: Combines low NDVI ($<0.25$), low NDWI ($<0.0$), low NDBI ($<0.15$), and visible reflectance to identify uncultivated, non-vegetated terrain.
- **Multi-Temporal Persistence**: Evaluates candidate regions across multiple Sentinel-2 acquisition dates to distinguish persistent bare land from seasonal post-harvest clearing (*"Observed bare in 4/5 available observations"*).
- **Surface Area & Distance Ranking**: Automatically prioritizes results based on user preferences (*"large empty lands"* $\rightarrow$ sorted by area descending; *"closest to Perundurai"* $\rightarrow$ sorted by distance ascending).
- **Conversational Follow-Up Inquiries**: Refines current search results through interactive follow-ups (*"Only show areas larger than 3 hectares"*, *"Which candidate is closest to Perundurai?"*, *"Show areas that were bare in all observations"*).
- **Geophysical Terminology & Legal Guardrail**: Uses strict scientific terms (`Potentially vacant land`, `Potential bare/non-vegetated area`) and displays notices stating satellite imagery reflects surface biophysics and does not establish legal ownership or sale availability.

---

## 💬 Natural-Language Query Directory

| Query Intent | Example Natural-Language Query | Dispatched Specialists & Pipeline |
| :--- | :--- | :--- |
| **Bare-Land Search** | *"Find potentially vacant land around Perundurai"* | `GeospatialSearchEngine` $\rightarrow$ Copernicus Catalog $\rightarrow$ Sentinel-2 NDVI/NDWI $\rightarrow$ Persistence $\rightarrow$ Polygons |
| **Filtered Search** | *"Find large empty lands within 10 km of Perundurai"* | `GeospatialSearchEngine` $\rightarrow$ 10km AOI $\rightarrow$ Bare Land Detection $\rightarrow$ Area-based Ranking |
| **Vegetation Health** | *"Find areas with poor vegetation health near Erode"* | `GeospatialSearchEngine` $\rightarrow$ Sentinel-2 NDVI Engine $\rightarrow$ Low-vigor Canopy Polygons |
| **Water Detection** | *"Show water bodies around Perundurai"* | `GeospatialSearchEngine` $\rightarrow$ Sentinel-2 NDWI Engine $\rightarrow$ Hydrological Water Masks |
| **Urban Expansion** | *"Find newly developed areas around Perundurai"* | `GeospatialSearchEngine` $\rightarrow$ Sentinel-2 NDBI Engine $\rightarrow$ Impervious Surface Delineation |
| **Temporal Change** | *"Find areas that changed from vegetation to built-up land around Perundurai between 2022 and 2026"* | `CopernicusChangeService` $\rightarrow$ Multi-epoch Sentinel-2 $\rightarrow$ Siamese Differencing $\rightarrow$ Transition Metrics |
| **Single-Image VQA** | *"What type of terrain and infrastructure is visible in this scene?"* | `AgentQueryPlanner` $\rightarrow$ `SingleImageVQATool` (GeoChat-7B) |
| **Region Grounding** | *"Highlight the water bodies and find the harbor port."* | `AgentQueryPlanner` $\rightarrow$ `GroundingTool` (VRSBench Spatial Normalizer) |
| **Scene Captioning** | *"Describe the land-cover and major objects visible in this image."* | `AgentQueryPlanner` $\rightarrow$ `CaptioningTool` (Dense Scene Descriptor) |
| **Optical + SAR** | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | `AgentQueryPlanner` $\rightarrow$ `OpticalSARTool` (Radar Decibel + Reflectance Fusion) |
| **Multi-Tool Compound** | *"Has vegetation decreased and where did change occur between the dates?"* | `AgentQueryPlanner` $\rightarrow$ `ChangeDetectionTool` + `SpectralAnalysisTool` + `SingleImageVQATool` |

---

## ⚡ Hardware Profile & VRAM Requirements

| Execution Mode | Model Precision | Expected VRAM / RAM | Target Hardware | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Full FP16** | 16-bit Float | ~14.2 GB VRAM | RTX 3090, RTX 4090, A100 | Unquantized full precision inference |
| **8-Bit Quantized** | `int8` (bitsandbytes) | ~8.4 GB VRAM | RTX 4070 (8-12 GB) | Minor quantization loss |
| **4-Bit NF4 (Default)**| `bitsandbytes` NF4 | **~5.2 GB VRAM** | **NVIDIA RTX 3050 Laptop (6 GB VRAM)** | **Recommended: Fits comfortably in 6GB VRAM** |
| **CPU Fallback** | `int8` / `float32` | ≥ 16 GB System RAM | Multi-core CPU | Automatically engages when CUDA unavailable |
| **Evaluation Adapter**| Automated Evaluation | < 100 MB RAM | Any machine | High-fidelity benchmark evaluation mode |

---

## 🚀 How to Run and Verify

### Prerequisites
- **Python 3.10+** (tested on Python 3.12)
- **Node.js 18+** (tested on Node v24) and `npm`

### 1. Start the FastAPI Backend
```bash
# In the project root directory
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the Next.js Frontend
```bash
# In the frontend/ directory
cd frontend
npm run dev
```
- Mission Control Web Application: **[http://localhost:3000](http://localhost:3000)**

### 3. Run Automated Tests
```bash
# Run full automated test suite across all modules
python -m pytest tests/test_geospatial_search.py tests/test_copernicus_nlp_and_change.py tests/test_copernicus_module.py -v
```

---

## 🧪 Verification & Acceptance Suite

| Test Suite | File | Tests | Status | Scope |
| :--- | :--- | :---: | :---: | :--- |
| **Geospatial Search** | `tests/test_geospatial_search.py` | 8 | ✅ PASSED | NL query parsing, bare-land candidate extraction, area ranking, API endpoints, conversational follow-ups |
| **Copernicus & NLP** | `tests/test_copernicus_nlp_and_change.py` | 4 | ✅ PASSED | NLP change queries, human-readable labels, AI analyze integration |
| **Copernicus Module** | `tests/test_copernicus_module.py` | 12 | ✅ PASSED | Geocoding (Perundurai, Erode, Coimbatore), STAC search, NDVI/NDWI/NDBI indices, bi-temporal change |
| **Agent Orchestrator** | `tests/test_agent_orchestrator.py` | 6 | ✅ PASSED | 10-stage lifecycle, ToolRegistry, multi-tool compound queries, observable trace validation |
| **Cross-Modal SAR** | `tests/test_optical_sar.py` | 6 | ✅ PASSED | Decibel transformation, polarimetric false-color, double-bounce/specular physics |
| **Change Detection** | `tests/test_change_detection.py` | 6 | ✅ PASSED | Siamese feature differencing, vector polygonizer, quantitative area metrics |
| **Single Image VQA** | `tests/test_single_image.py` | 8 | ✅ PASSED | RSVQA, VRSBench grounding, scene captioning, spatial normalizer |
| **Preprocessing** | `tests/test_preprocessing.py` | 8 | ✅ PASSED | GeoTIFF validation, band extraction, CRS reprojection, percentile contrast stretch |
| **Frontend Production**| `npm run build` (Next.js 16) | Build | ✅ PASSED | 0 TypeScript errors, clean static/dynamic route optimization |

---

## 🔒 Geophysical Accuracy & Terminology Notice
Satellite imagery identifies physical spectral reflectance and radar backscatter properties (such as bare soil, lack of photosynthetic chlorophyll, open water absorption, or impervious surface reflectance). **SatQuery AI strictly differentiates physical land characteristics from legal ownership**:
> *"Satellite imagery identifies areas that appear bare, non-vegetated, or structurally altered based on spectral characteristics. This does not confirm land ownership, legal status, availability for purchase, or permanent land use."*
