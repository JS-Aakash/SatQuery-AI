"""
Abstract Service Interfaces for SatQuery AI.

These interfaces define the strict technical contracts for all functional
subsystems of the platform. Future AI and specialist model modules (VQA,
Grounding, CDVQA, Optical-SAR Fusion, etc.) will implement these classes
directly without altering the frontend or API layer.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..schemas import (
    ImageMetadata,
    AnalysisRequest,
    AnalysisResponse,
    GroundingBox,
    ChangeMapData,
    EvidenceItem,
    ExecutionStep,
    TaskType,
    ReportResponse,
    ReportGenerationRequest,
)


class ImageProcessor(ABC):
    """
    Abstract interface for geospatial raster parsing, format verification,
    band extraction, and remote-sensing normalization.
    """

    @abstractmethod
    def validate_image_header(self, filename: str, file_size_bytes: int, sample_bytes: Optional[bytes] = None) -> ImageMetadata:
        """Inspects file headers, verifies TIFF/GeoTIFF/PNG/JPEG integrity, and extracts dimensions/CRS."""
        pass

    @abstractmethod
    def extract_band_information(self, file_path_or_bytes: Any) -> Dict[str, Any]:
        """Extracts spectral band count, names (e.g. RGB, NIR, RedEdge, VV, VH), and bit depth."""
        pass

    @abstractmethod
    def check_coregistration(self, image_a_meta: ImageMetadata, image_b_meta: ImageMetadata) -> Tuple[bool, str]:
        """Verifies spatial co-registration between two geospatial images (CRS, bounding box, resolution)."""
        pass


class VQAService(ABC):
    """
    Abstract interface for Remote-Sensing Visual Question Answering (RSVQA / VRSBench).
    To be implemented in Module 3.
    """

    @abstractmethod
    def answer_query(self, image_data: Any, query: str) -> Dict[str, Any]:
        """
        Executes domain-adapted VLM inference on single remote sensing image.
        Returns: { 'answer': str, 'confidence': float, 'evidence': List[EvidenceItem] }
        """
        pass


class GroundingService(ABC):
    """
    Abstract interface for Text-Guided Region Grounding on satellite imagery.
    To be implemented in Module 3.
    """

    @abstractmethod
    def ground_text_queries(self, image_data: Any, query: str) -> List[GroundingBox]:
        """
        Detects and bounds geographic/structural entities referred to in natural-language query.
        Returns normalized bounding coordinates [ymin, xmin, ymax, xmax].
        """
        pass


class ChangeDetectionService(ABC):
    """
    Abstract interface for Multitemporal Change Description and Change VQA (CDVQA).
    To be implemented in Module 4.
    """

    @abstractmethod
    def detect_changes(self, t1_image: Any, t2_image: Any, query: str) -> Dict[str, Any]:
        """
        Compares two temporal observations, generates change descriptions,
        answers change queries, and produces spatial change masks.
        """
        pass

    @abstractmethod
    def generate_change_map(self, t1_image: Any, t2_image: Any) -> ChangeMapData:
        """Generates binary/multi-class raster difference mask or vector changes."""
        pass


class OpticalSARService(ABC):
    """
    Abstract interface for Cross-Modal Optical-SAR Joint Information Extraction.
    To be implemented in Module 5.
    """

    @abstractmethod
    def joint_reasoning(self, optical_image: Any, sar_image: Any, query: str) -> Dict[str, Any]:
        """
        Fuses optical reflectance and SAR backscatter characteristics
        to resolve cloud-covered, flooded, or ambiguous geographic features.
        """
        pass


class AgentService(ABC):
    """
    Abstract interface for the Query-Driven Agentic Controller.
    Interprets natural language queries, validates input compatibility,
    routes to specialist tools, and returns auditable execution traces.
    """

    @abstractmethod
    def classify_task_intent(self, query: str, has_paired_images: bool = False, is_cross_modal: bool = False) -> TaskType:
        """Classifies the remote-sensing task based on user query and available imagery."""
        pass

    @abstractmethod
    def execute_workflow(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Sequences and executes the appropriate specialist models/tools
        and produces the observable execution trace.
        """
        pass


class SatelliteService(ABC):
    """
    Abstract interface for Satellite Data Discovery and Catalog Searching
    (e.g., Copernicus Open Access Hub, Sentinel-1/2 STAC APIs).
    """

    @abstractmethod
    def search_scenes(
        self,
        aoi_coords: Optional[List[List[float]]],
        start_date: str,
        end_date: str,
        sensors: List[str],
        max_cloud_cover: float = 20.0
    ) -> List[Dict[str, Any]]:
        """Queries STAC/satellite catalogs for matching imagery tiles."""
        pass

    @abstractmethod
    def calculate_aoi_area(self, aoi_coords: List[List[float]]) -> float:
        """Calculates area in square kilometers for an AOI polygon."""
        pass


class ReportService(ABC):
    """
    Abstract interface for Generating Auditable Analysis Reports (Markdown, PDF, JSON).
    """

    @abstractmethod
    def generate_report(self, request: ReportGenerationRequest, analysis: AnalysisResponse) -> ReportResponse:
        """Compiles analysis results, evidence, and execution trace into downloadable document."""
        pass
