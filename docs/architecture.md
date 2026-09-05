# SatQuery AI - Technical Architecture & Module Integration Blueprint

## 1. System Philosophy & Decoupled Architecture

SatQuery AI is built with a decoupled service architecture following the **Open-Closed Principle**: the user interface, API layer, and agent orchestrator communicate exclusively with abstract base classes (`backend/app/services/base.py`).

Future model fine-tuning and satellite data connectors can be swapped or plugged in directly without modifying frontend components or REST endpoints.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Next.js 16 UI Layer                             │
│   (Upload Studio, Interactive Map/AOI, Split-Slider Viewer, AI Answer HUD)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST APIs / JSON Schemas
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                            FastAPI Backend Layer                            │
│  ┌─────────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │    Pydantic Schemas     │  │          Agentic Orchestrator            │  │
│  │ (Imagery, Trace, Task)  │  │  (Intent Router, Execution Trace Builder)│  │
│  └─────────────────────────┘  └──────────────────┬───────────────────────┘  │
└──────────────────────────────────────────────────┼──────────────────────────┘
                                                   │
     ┌──────────────────────┬──────────────────────┼──────────────────────┐
     │                      │                      │                      │
     ▼                      ▼                      ▼                      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Module 2:   │     │  Module 3:   │     │  Module 4:   │     │  Module 5:   │
│Preprocessing │     │Single-Img VQA│     │ Bi-Temporal  │     │ Optical-SAR  │
│ & STAC Engine│     │  & Grounding │     │ Change Model │     │ Fusion Model │
│(ImageProcessor│    │ (VQAService, │     │(ChangeService│     │(OpticalSAR-  │
│SatelliteServ)│     │ GroundingServ)│    │  CDVQA)      │     │  Service)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 2. Abstract Service Interface Mapping

Each upcoming module has a dedicated Python interface in `backend/app/services/base.py`:

### `ImageProcessor`
- **File**: `backend/app/services/base.py`
- **Upcoming Module**: **Module 2 (Preprocessing)**
- **Methods**:
  - `validate_image_header(filename, file_size_bytes, sample_bytes) -> ImageMetadata`
  - `extract_band_information(file_path_or_bytes) -> Dict[str, Any]`
  - `check_coregistration(image_a_meta, image_b_meta) -> Tuple[bool, str]`

### `AgentService`
- **File**: `backend/app/services/base.py`
- **Upcoming Module**: **Module 2 (Agentic Orchestrator & Tool Registry)**
- **Methods**:
  - `classify_task_intent(query, has_paired_images, is_cross_modal) -> TaskType`
  - `execute_workflow(request: AnalysisRequest) -> AnalysisResponse`
  - Produces the observable 7-step execution trace:
    1. Query interpreted
    2. Input validated
    3. Task identified
    4. Specialist selected
    5. Analysis completed
    6. Evidence generated
    7. Response synthesized

### `VQAService` & `GroundingService`
- **File**: `backend/app/services/base.py`
- **Upcoming Module**: **Module 3 (Single-Image Remote-Sensing VQA & Grounding)**
- **Target Weights/Datasets**: BigEarthNet-adapted VLM / GeoChat-7B / RSVQA / VRSBench
- **Methods**:
  - `answer_query(image_data, query) -> Dict[str, Any]`
  - `ground_text_queries(image_data, query) -> List[GroundingBox]`

### `ChangeDetectionService`
- **File**: `backend/app/services/base.py`
- **Upcoming Module**: **Module 4 (Bi-Temporal Change Understanding & CDVQA)**
- **Target Datasets**: CDVQA / LEVIR-CD
- **Methods**:
  - `detect_changes(t1_image, t2_image, query) -> Dict[str, Any]`
  - `generate_change_map(t1_image, t2_image) -> ChangeMapData`

### `OpticalSARService`
- **File**: `backend/app/services/base.py`
- **Upcoming Module**: **Module 5 (Optical–SAR Cross-Modal Fusion)**
- **Target Datasets**: BigEarthNet-MM / ISRO Cartosat-2S & RISAT pairs
- **Methods**:
  - `joint_reasoning(optical_image, sar_image, query) -> Dict[str, Any]`
