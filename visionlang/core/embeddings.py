"""
CLIP Embeddings Module
======================

This module provides functionality for generating embeddings from text and images
using OpenAI's CLIP model. Embeddings are dense vector representations that capture
semantic meaning, allowing for similarity comparisons between texts, images, or both.

Key Concepts:
- Embeddings: Numerical vectors that represent semantic meaning
- Cosine Similarity: Metric for comparing embedding vectors
- Multimodal: Working with both text and image data
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Tuple
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class CLIPEmbedding:
    """
    A class for generating and comparing embeddings using CLIP.
    
    CLIP (Contrastive Language-Image Pre-training) is a model that learns
    to associate images with their textual descriptions, creating embeddings
    in a shared space where semantically similar items are close together.
    
    Attributes:
        model_name (str): Name of the CLIP model variant
        model: The loaded CLIP model
        processor: Preprocessor for handling images and text
        tokenizer: Tokenizer for text processing
        device: Device (CPU/GPU) where the model runs
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize the CLIP embedding generator.
        
        Args:
            model_name: Hugging Face model identifier for CLIP variant
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model components
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def encode_text(self, texts: Union[str, List[str]], return_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings for text inputs.
        
        Args:
            texts: Single text string or list of text strings
            return_numpy: If True, return numpy array instead of torch tensor
            
        Returns:
            Embedding vectors of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize text
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        if return_numpy:
            return text_embeddings.cpu().numpy()
        return text_embeddings
    
    def encode_images(self, images: Union[Image.Image, List[Image.Image]], return_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings for image inputs.
        
        Args:
            images: Single PIL Image or list of PIL Images
            return_numpy: If True, return numpy array instead of torch tensor
            
        Returns:
            Embedding vectors of shape (n_images, embedding_dim)
        """
        if not isinstance(images, list):
            images = [images]
        
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        if return_numpy:
            return image_embeddings.cpu().numpy()
        return image_embeddings
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n1, dim)
            embeddings2: Second set of embeddings (n2, dim)
            
        Returns:
            Similarity matrix of shape (n1, n2)
        """
        # Ensure tensors are on the same device
        if embeddings1.device != embeddings2.device:
            embeddings2 = embeddings2.to(embeddings1.device)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            embeddings1[:, None, :],
            embeddings2[None, :, :],
            dim=2
        )
        
        return similarity.cpu().numpy()
    
    def text_to_text_similarity(self, texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> np.ndarray:
        """
        Compute similarity between two sets of texts.
        
        Args:
            texts1: First text or list of texts
            texts2: Second text or list of texts
            
        Returns:
            Similarity matrix
        """
        embeddings1 = self.encode_text(texts1)
        embeddings2 = self.encode_text(texts2)
        return self.compute_similarity(embeddings1, embeddings2)
    
    def image_to_image_similarity(self, images1: Union[Image.Image, List[Image.Image]], 
                                 images2: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Compute similarity between two sets of images.
        
        Args:
            images1: First image or list of images
            images2: Second image or list of images
            
        Returns:
            Similarity matrix
        """
        embeddings1 = self.encode_images(images1)
        embeddings2 = self.encode_images(images2)
        return self.compute_similarity(embeddings1, embeddings2)
    
    def text_to_image_similarity(self, texts: Union[str, List[str]], 
                                images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Compute similarity between texts and images.
        
        Args:
            texts: Text or list of texts
            images: Image or list of images
            
        Returns:
            Similarity matrix (texts x images)
        """
        text_embeddings = self.encode_text(texts)
        image_embeddings = self.encode_images(images)
        return self.compute_similarity(text_embeddings, image_embeddings)
    
    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        """
        Utility function to load an image from a URL.
        
        Args:
            url: URL of the image
            
        Returns:
            PIL Image object
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")