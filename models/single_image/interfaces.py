"""
Abstract interfaces for Single-Image Remote-Sensing Intelligence.
Defines technical contracts for RemoteSensingVQA, RemoteSensingCaptioning, and RemoteSensingGrounding.
"""
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List, Optional
import numpy as np
from PIL import Image

from .schemas import SingleImageResponse, SingleImageRequest


class RemoteSensingVQA(ABC):
    """
    Interface for Remote-Sensing Visual Question Answering (RSVQA / VRSBench).
    """

    @abstractmethod
    def answer_question(
        self,
        image_input: Union[np.ndarray, Image.Image, bytes, str],
        question: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SingleImageResponse:
        """
        Answers natural-language question regarding land-cover, object count,
        infrastructure, or spatial layout from a single satellite observation.
        """
        pass


class RemoteSensingCaptioning(ABC):
    """
    Interface for Remote-Sensing Scene Captioning and Dense Description.
    """

    @abstractmethod
    def generate_caption(
        self,
        image_input: Union[np.ndarray, Image.Image, bytes, str],
        detailed: bool = True,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SingleImageResponse:
        """
        Generates concise or detailed scene descriptions summarizing primary
        land-cover classes and prominent geographic/infrastructure features.
        """
        pass


class RemoteSensingGrounding(ABC):
    """
    Interface for Text-Guided Region Grounding on remote-sensing imagery.
    """

    @abstractmethod
    def ground_text_query(
        self,
        image_input: Union[np.ndarray, Image.Image, bytes, str],
        text_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SingleImageResponse:
        """
        Identifies and localizes geographic entities referred to in the natural-language query
        (e.g. 'Find the water body', 'Locate the runway'), returning normalized bounding boxes.
        """
        pass
