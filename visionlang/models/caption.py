"""
Image Captioning Module
=======================

This module provides functionality for generating natural language descriptions
of images using Qwen2.5-VL, a state-of-the-art Vision-Language Model.

Key Concepts:
- Multimodal Models: Models that process both visual and textual information
- Chat Templates: Structured formats for conversational AI interactions
- Autoregressive Generation: Generating text one token at a time
"""

import torch
from typing import Optional, List, Dict, Union
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ImageCaptioner:
    """
    A class for generating image captions using Qwen2.5-VL.
    
    This captioner uses a large vision-language model to generate
    natural language descriptions of images. It can handle various
    captioning styles and lengths.
    
    Attributes:
        model_id (str): Identifier for the Qwen model variant
        model: The loaded Qwen2.5-VL model
        processor: Preprocessor for handling images and text
        device: Device (CPU/GPU) where the model runs
    """
    
    def __init__(self, 
                model_size: str = "3B",
                device: Optional[str] = None,
                dtype: str = "auto"):
        """
        Initialize the image captioner.
        
        Args:
            model_size: Size of Qwen model ("3B" or "7B")
            device: Device to run on ('cuda', 'cpu', or None for auto)
            dtype: Data type for model weights ("auto", "float16", "float32")
        """
        # Select model based on size
        if model_size == "3B":
            self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif model_size == "7B":
            self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def caption(self, 
               image: Image.Image, 
               prompt: Optional[str] = None,
               max_length: int = 100,
               style: str = "descriptive") -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image to caption
            prompt: Optional custom prompt for captioning
            max_length: Maximum number of tokens to generate
            style: Captioning style ("descriptive", "brief", "detailed", "creative")
            
        Returns:
            Generated caption text
        """
        # Select prompt based on style
        if prompt is None:
            if style == "brief":
                prompt = "Describe this image in one sentence."
            elif style == "detailed":
                prompt = "Provide a detailed description of this image, including colors, objects, composition, and any notable features."
            elif style == "creative":
                prompt = "Write a creative and engaging description of this image that could be used in a story."
            else:  # descriptive (default)
                prompt = "Describe this image."
        
        # Create message format
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(msgs)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=style == "creative",  # Use sampling for creative style
                temperature=0.7 if style == "creative" else 1.0
            )
        
        # Decode the generated caption
        caption = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
        
        return caption.strip()
    
    def caption_batch(self, 
                     images: List[Image.Image], 
                     prompt: Optional[str] = None,
                     max_length: int = 100,
                     style: str = "descriptive") -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of PIL Images to caption
            prompt: Optional custom prompt for captioning
            max_length: Maximum number of tokens to generate
            style: Captioning style
            
        Returns:
            List of generated captions
        """
        captions = []
        for image in images:
            caption = self.caption(image, prompt, max_length, style)
            captions.append(caption)
        return captions
    
    def interactive_caption(self, 
                          image: Image.Image,
                          questions: List[str]) -> Dict[str, str]:
        """
        Generate answers to multiple questions about an image.
        
        Args:
            image: PIL Image to analyze
            questions: List of questions about the image
            
        Returns:
            Dictionary mapping questions to answers
        """
        answers = {}
        for question in questions:
            answer = self.caption(image, prompt=question)
            answers[question] = answer
        return answers
    
    def compare_images(self, 
                      image1: Image.Image, 
                      image2: Image.Image) -> str:
        """
        Generate a comparison between two images.
        
        Args:
            image1: First PIL Image
            image2: Second PIL Image
            
        Returns:
            Comparison description
        """
        # For comparison, we'll caption each image and then create a comparison
        # Note: This is a simplified approach. Advanced models might handle
        # multiple images directly
        caption1 = self.caption(image1, style="brief")
        caption2 = self.caption(image2, style="brief")
        
        comparison_prompt = f"""
        Image 1: {caption1}
        Image 2: {caption2}
        
        Compare these two images, highlighting similarities and differences.
        """
        
        # Use the first image as context (simplified approach)
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": comparison_prompt}
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150
            )
        
        comparison = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
        
        return comparison.strip()
    
    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        """
        Utility function to load an image from a URL.
        
        Args:
            url: URL of the image
            
        Returns:
            PIL Image object
        """
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")