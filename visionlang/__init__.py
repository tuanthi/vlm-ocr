"""
VisionLang: Advanced Vision-Language Model Framework
====================================================

A professional framework for working with Vision-Language Models,
providing state-of-the-art implementations for multimodal AI including
CLIP for embeddings and zero-shot classification, and Qwen2.5-VL for
image captioning and object detection.

Main modules:
- core.embeddings: CLIP-based text and image embeddings
- core.classification: Zero-shot image classification
- models.caption: Image captioning with Qwen2.5-VL
- models.detection: Object detection with Qwen2.5-VL
"""

__version__ = "0.1.0"
__author__ = "VisionLang Framework"

from .core.embeddings import CLIPEmbedding
from .core.classification import ZeroShotClassifier
from .models.caption import ImageCaptioner
from .models.detection import ObjectDetector

__all__ = [
    "CLIPEmbedding",
    "ZeroShotClassifier",
    "ImageCaptioner",
    "ObjectDetector",
]