"""Core modules for basic VLM operations."""

from .embeddings import CLIPEmbedding
from .classification import ZeroShotClassifier

__all__ = ["CLIPEmbedding", "ZeroShotClassifier"]