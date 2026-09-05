"""
Specialist Tools Package & Auto-Registration
"""
from .single_vqa import SingleImageVQATool
from .captioning import CaptioningTool
from .grounding import GroundingTool
from .change_detection import ChangeDetectionTool
from .optical_sar import OpticalSARTool
from .spectral_analysis import SpectralAnalysisTool
from ..registry.registry import tool_registry

# Register standard specialists into singleton registry
tool_registry.register(SingleImageVQATool())
tool_registry.register(CaptioningTool())
tool_registry.register(GroundingTool())
tool_registry.register(ChangeDetectionTool())
tool_registry.register(OpticalSARTool())
tool_registry.register(SpectralAnalysisTool())

__all__ = [
    "SingleImageVQATool",
    "CaptioningTool",
    "GroundingTool",
    "ChangeDetectionTool",
    "OpticalSARTool",
    "SpectralAnalysisTool",
]
