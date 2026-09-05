"""
Optical + SAR Model Manager Singleton.
Lazy-loads cross-modal fusion pipeline and provides unified API.
"""
from typing import Dict, Any, Optional, Union
from .fusion_pipeline import CrossModalFusionPipeline
from .interfaces import OpticalSARFusionResult


class OpticalSARManager:
    """
    Singleton Manager for Cross-Modal Optical + SAR Intelligence.
    """

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self) -> CrossModalFusionPipeline:
        if self._pipeline is None:
            self._pipeline = CrossModalFusionPipeline()
        return self._pipeline

    def infer(
        self,
        optical_input: Any,
        sar_input: Any,
        query: str = "Analyze the optical and SAR images together.",
        parameters: Optional[Dict[str, Any]] = None
    ) -> OpticalSARFusionResult:
        """Executes cross-modal Optical + SAR fusion."""
        return self.pipeline.fuse_and_analyze(
            optical_input=optical_input,
            sar_input=sar_input,
            query=query,
            parameters=parameters
        )


# Global singleton instance
optical_sar_manager = OpticalSARManager()
