"""
Zero-Shot Image Classification Module
======================================

This module provides functionality for classifying images into categories without
explicit training on those categories. It uses CLIP's ability to understand both
images and text to match images with textual descriptions.

Key Concepts:
- Zero-Shot Learning: Classifying without training examples
- Logits: Raw similarity scores before normalization
- Softmax: Converting logits to probabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor


class ZeroShotClassifier:
    """
    A zero-shot image classifier using CLIP.
    
    This classifier can categorize images into arbitrary classes without
    being explicitly trained on those classes. It works by computing
    similarity between the image and text descriptions of each class.
    
    Attributes:
        model_name (str): Name of the CLIP model variant
        model: The loaded CLIP model
        processor: Preprocessor for handling images and text
        device: Device (CPU/GPU) where the model runs
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize the zero-shot classifier.
        
        Args:
            model_name: Hugging Face model identifier for CLIP variant
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model components
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def classify(self, 
                image: Image.Image, 
                labels: List[str], 
                return_scores: bool = False,
                template: Optional[str] = None) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Classify an image into one of the provided labels.
        
        Args:
            image: PIL Image to classify
            labels: List of possible class labels
            return_scores: If True, return probabilities for all classes
            template: Optional template for labels (e.g., "a photo of a {}")
            
        Returns:
            If return_scores is False: The predicted label
            If return_scores is True: Tuple of (predicted_label, probability_dict)
        """
        # Apply template to labels if provided
        if template:
            text_labels = [template.format(label) for label in labels]
        else:
            text_labels = labels
        
        # Preprocess inputs
        inputs = self.processor(
            text=text_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (1, n_labels)
            
            # Convert to probabilities using softmax
            probs = logits_per_image.softmax(dim=-1).squeeze().cpu().numpy()
        
        # Get predicted label
        predicted_idx = probs.argmax()
        predicted_label = labels[predicted_idx]
        
        if return_scores:
            # Create dictionary of label: probability
            prob_dict = {label: float(prob) for label, prob in zip(labels, probs)}
            return predicted_label, prob_dict
        
        return predicted_label
    
    def classify_batch(self, 
                      images: List[Image.Image], 
                      labels: List[str],
                      template: Optional[str] = None) -> List[Tuple[str, Dict[str, float]]]:
        """
        Classify multiple images at once.
        
        Args:
            images: List of PIL Images to classify
            labels: List of possible class labels
            template: Optional template for labels
            
        Returns:
            List of tuples, each containing (predicted_label, probability_dict)
        """
        results = []
        for image in images:
            result = self.classify(image, labels, return_scores=True, template=template)
            results.append(result)
        return results
    
    def get_top_k_predictions(self, 
                            image: Image.Image, 
                            labels: List[str], 
                            k: int = 5,
                            template: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get top-k most likely labels for an image.
        
        Args:
            image: PIL Image to classify
            labels: List of possible class labels
            k: Number of top predictions to return
            template: Optional template for labels
            
        Returns:
            List of (label, probability) tuples, sorted by probability
        """
        _, prob_dict = self.classify(image, labels, return_scores=True, template=template)
        
        # Sort by probability
        sorted_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k
        return sorted_predictions[:k]
    
    def explain_prediction(self, 
                          image: Image.Image, 
                          labels: List[str],
                          template: Optional[str] = None) -> Dict:
        """
        Provide detailed explanation of the classification.
        
        Args:
            image: PIL Image to classify
            labels: List of possible class labels
            template: Optional template for labels
            
        Returns:
            Dictionary containing prediction details and statistics
        """
        # Apply template to labels if provided
        if template:
            text_labels = [template.format(label) for label in labels]
        else:
            text_labels = labels
        
        # Preprocess inputs
        inputs = self.processor(
            text=text_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (1, n_labels)
            
            # Get temperature scaling factor
            temperature = self.model.logit_scale.exp().item()
            
            # Convert to probabilities
            probs = logits_per_image.softmax(dim=-1).squeeze().cpu().numpy()
            
            # Get raw similarity scores (before temperature scaling)
            raw_similarities = (logits_per_image / temperature).squeeze().cpu().numpy()
        
        # Get prediction
        predicted_idx = probs.argmax()
        
        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Create explanation
        explanation = {
            'predicted_label': labels[predicted_idx],
            'confidence': float(probs[predicted_idx]),
            'all_probabilities': {label: float(prob) for label, prob in zip(labels, probs)},
            'raw_similarities': {label: float(sim) for label, sim in zip(labels, raw_similarities)},
            'temperature_scale': temperature,
            'entropy': float(entropy),
            'uncertainty': float(entropy / np.log(len(labels))),  # Normalized entropy
            'top_3': [(labels[i], float(probs[i])) for i in probs.argsort()[-3:][::-1]]
        }
        
        return explanation
    
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