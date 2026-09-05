"""
Curated mock datasets and scenarios for UI demonstration and development.
All data is clearly annotated as mock/demo.
"""
from typing import Dict, Any, List
from ..schemas import (
    TaskType,
    EvidenceItem,
    GroundingBox,
    ChangeMapData,
    ExecutionStep,
    StepStatus,
    ModalityEnum,
    SensorEnum,
    ImageMetadata,
)

# Demo Scenarios tailored directly to the PS.txt representative queries
DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "land_cover": {
        "task": TaskType.CAPTIONING,
        "models_used": ["RS-VLM-Adapter (BigEarthNet Adapted)", "Sentinel-2 Multi-spectral Segmenter"],
        "confidence": 0.89,
        "answer": "The scene displays mixed peri-urban land-cover. Primary classes include Dense Urban Built-up (42.6%), Coastal Wetland and Water Bodies (28.4%), Agricultural Crop Parcels (18.2%), and Forest/Tree Canopy (10.8%). Prominent infrastructure features include a multi-berth marine cargo terminal and transportation corridors.",
        "evidence": [
            EvidenceItem(
                id="ev-1",
                title="Dense Urban Infrastructure",
                category="Spectral Feature",
                description="High spectral reflectance in visible/NIR bands indicative of concrete and asphalt paving.",
                confidence=0.92,
                coordinates=[15.0, 20.0, 55.0, 75.0]
            ),
            EvidenceItem(
                id="ev-2",
                title="Estuarine Water Body",
                category="Water Index (MNDWI)",
                description="MNDWI value > 0.45 confirms an open water inlet with low chlorophyll turbidity.",
                confidence=0.96,
                coordinates=[60.0, 10.0, 95.0, 50.0]
            ),
            EvidenceItem(
                id="ev-3",
                title="Active Agricultural Parcels",
                category="Vegetation Index (NDVI)",
                description="Mean NDVI of 0.68 detected across rectangular parcels in the northwest quadrant.",
                confidence=0.88,
                coordinates=[5.0, 5.0, 35.0, 45.0]
            )
        ],
        "grounding_boxes": [
            GroundingBox(id="gb-1", label="Marine Cargo Terminal", box=[22.0, 35.0, 58.0, 78.0], confidence=0.93, color="#06b6d4"),
            GroundingBox(id="gb-2", label="Coastal Lagoon", box=[65.0, 12.0, 92.0, 48.0], confidence=0.97, color="#3b82f6"),
            GroundingBox(id="gb-3", label="Commercial Dense Area", box=[18.0, 60.0, 42.0, 90.0], confidence=0.88, color="#10b981"),
        ],
        "change_map": None
    },
    "water_grounding": {
        "task": TaskType.GROUNDING,
        "models_used": ["VRSBench Grounding Specialist", "Water Boundary Delineator"],
        "confidence": 0.94,
        "answer": "Successfully located and bounded the target water body in the southern estuarine zone. The delineated boundaries match standard water extraction indices (NDWI > 0.42) with crisp demarcation against adjoining built-up embankments.",
        "evidence": [
            EvidenceItem(
                id="ev-w1",
                title="Delineated Water Basin",
                category="Spatial Region",
                description="Target water reservoir identified covering an approximate surface area of 3.42 km².",
                confidence=0.95,
                coordinates=[58.0, 25.0, 88.0, 72.0]
            ),
            EvidenceItem(
                id="ev-w2",
                title="Canal Connector",
                category="Linear Waterway",
                description="Linear drainage corridor feeding the reservoir from the northern basin.",
                confidence=0.91,
                coordinates=[30.0, 42.0, 58.0, 48.0]
            )
        ],
        "grounding_boxes": [
            GroundingBox(id="gb-w1", label="Target Water Body", box=[58.0, 25.0, 88.0, 72.0], confidence=0.96, color="#06b6d4"),
            GroundingBox(id="gb-w2", label="Feeder Canal System", box=[30.0, 42.0, 58.0, 48.0], confidence=0.91, color="#3b82f6"),
        ],
        "change_map": None
    },
    "temporal_change": {
        "task": TaskType.CHANGE_DETECTION,
        "models_used": ["CDVQA Temporal Reasoner", "Bi-Temporal Siamese Difference Net"],
        "confidence": 0.88,
        "answer": "Between Observation T1 (2022-03) and Observation T2 (2024-03), significant built-up expansion (+14.3%) and vegetation loss (-9.7%) occurred in the eastern sector. A new commercial warehousing zone and access roads were constructed over previously fallow agricultural land. Water reservoir extent contracted by 2.1% due to seasonal drying.",
        "evidence": [
            EvidenceItem(
                id="ev-c1",
                title="Urban Expansion Zone",
                category="Temporal Difference",
                description="Spectral transition from vegetated/bare soil (NDVI 0.52) to impervious pavement (NDVI 0.12).",
                confidence=0.91,
                coordinates=[20.0, 45.0, 65.0, 85.0]
            ),
            EvidenceItem(
                id="ev-c2",
                title="New Roadway Corridor",
                category="Infrastructure Addition",
                description="High linear coherence feature observed only in Observation T2.",
                confidence=0.94,
                coordinates=[45.0, 15.0, 52.0, 85.0]
            ),
            EvidenceItem(
                id="ev-c3",
                title="Water Extent Receded",
                category="Hydrological Fluctuation",
                description="Shoreline boundary contracted approximately 45 meters along western bank.",
                confidence=0.86,
                coordinates=[65.0, 20.0, 82.0, 38.0]
            )
        ],
        "grounding_boxes": [
            GroundingBox(id="gb-c1", label="New Built-up Area (+14.3%)", box=[20.0, 45.0, 65.0, 85.0], confidence=0.92, color="#f59e0b"),
            GroundingBox(id="gb-c2", label="Cleared Canopy Zone", box=[12.0, 30.0, 28.0, 45.0], confidence=0.87, color="#ef4444"),
        ],
        "change_map": ChangeMapData(
            has_change=True,
            change_percentage=14.3,
            change_type="Urban Expansion & Land Conversion",
            change_mask_url="/demo/change_mask_sample.png",
            legend={
                "New Built-up Area": "#f59e0b",
                "Vegetation Loss": "#ef4444",
                "Water Recession": "#3b82f6",
                "Unchanged": "transparent"
            }
        )
    },
    "optical_sar": {
        "task": TaskType.OPTICAL_SAR_FUSION,
        "models_used": ["Optical-SAR Cross-Modal Fusion Net", "SAR Backscatter Classifier (VV/VH)", "Sentinel-2 Spectral Engine"],
        "confidence": 0.91,
        "answer": "By fusing Sentinel-2 optical reflectance and Sentinel-1 SAR dual-polarization backscatter: 1) Cloud-obscured eastern sectors were fully mapped using SAR penetration. 2) High double-bounce backscatter in VV+VH confirmed structural built-up areas despite low optical contrast. 3) Specular radar reflectance cleanly separated flooded surfaces from smooth asphalt.",
        "evidence": [
            EvidenceItem(
                id="ev-os1",
                title="SAR Double-Bounce Signature",
                category="SAR Backscatter (VV/VH)",
                description="Strong radar return (> -6 dB) validates high-density concrete buildings obscured by haze.",
                confidence=0.93,
                coordinates=[25.0, 35.0, 60.0, 75.0]
            ),
            EvidenceItem(
                id="ev-os2",
                title="Specular Radar Zero-Return",
                category="Cross-Modal Verification",
                description="Dark radar patch confirms standing surface water underneath partial optical cloud cover.",
                confidence=0.96,
                coordinates=[65.0, 15.0, 90.0, 45.0]
            ),
            EvidenceItem(
                id="ev-os3",
                title="Vegetation Volume Scattering",
                category="Multimodal Concordance",
                description="High optical NDVI combined with moderate cross-pol (VH) backscatter confirms mangrove forest.",
                confidence=0.89,
                coordinates=[5.0, 10.0, 30.0, 40.0]
            )
        ],
        "grounding_boxes": [
            GroundingBox(id="gb-os1", label="Radar-Confirmed Built-up", box=[25.0, 35.0, 60.0, 75.0], confidence=0.94, color="#10b981"),
            GroundingBox(id="gb-os2", label="Cloud-Penetrated Water Area", box=[65.0, 15.0, 90.0, 45.0], confidence=0.96, color="#06b6d4"),
        ],
        "change_map": None
    },
    "default_vqa": {
        "task": TaskType.VQA,
        "models_used": ["RS-VQA Specialist (RSVQA Benchmark Adapted)"],
        "confidence": 0.87,
        "answer": "Yes, high-density residential and commercial structures occupy the central quadrant, flanked by agricultural fields to the north and open coastal water to the south.",
        "evidence": [
            EvidenceItem(
                id="ev-v1",
                title="Identified Built-up Grid",
                category="Spatial Layout",
                description="Regular orthogonal street grid with roof structures detected.",
                confidence=0.88,
                coordinates=[30.0, 30.0, 70.0, 70.0]
            )
        ],
        "grounding_boxes": [
            GroundingBox(id="gb-v1", label="Detected Urban Region", box=[30.0, 30.0, 70.0, 70.0], confidence=0.88, color="#10b981")
        ],
        "change_map": None
    }
}
