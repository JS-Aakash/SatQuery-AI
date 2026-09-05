"""
Bi-Temporal Change Detection Model Manager.
Lazy-loads change models, coordinates pipeline execution, and exposes unified API.
"""
from typing import Dict, Any, Optional, Union
from .pipeline import ChangeDetectionPipeline
from .interfaces import ChangeDetectionResult


class ChangeDetectionManager:
    """
    Singleton Manager for Bi-Temporal Remote-Sensing Change Intelligence.
    """

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self) -> ChangeDetectionPipeline:
        if self._pipeline is None:
            self._pipeline = ChangeDetectionPipeline()
        return self._pipeline

    def infer(
        self,
        image_t1: Any,
        image_t2: Any,
        query: str = "What changed between these two images?",
        parameters: Optional[Dict[str, Any]] = None
    ) -> ChangeDetectionResult:
        """Runs bi-temporal change pipeline."""
        return self.pipeline.detect_change(
            image_t1=image_t1,
            image_t2=image_t2,
            query=query,
            parameters=parameters
        )


# Global singleton instance
change_manager = ChangeDetectionManager()
