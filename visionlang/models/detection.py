"""
Object Detection Module
=======================

This module provides functionality for detecting and localizing objects in images
using Qwen2.5-VL. It can identify objects, provide bounding boxes, and perform
spatial reasoning tasks.

Key Concepts:
- Bounding Boxes: Rectangular regions defining object locations
- Object Classes: Categories of detected objects
- Spatial Reasoning: Understanding relationships between objects
"""

import torch
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ObjectDetector:
    """
    A class for detecting objects in images using Qwen2.5-VL.
    
    This detector can identify objects, provide their locations as bounding boxes,
    and perform various spatial reasoning tasks.
    
    Attributes:
        model_id (str): Identifier for the Qwen model variant
        model: The loaded Qwen2.5-VL model
        processor: Preprocessor for handling images and text
        device: Device (CPU/GPU) where the model runs
    """
    
    def __init__(self, 
                model_size: str = "7B",
                device: Optional[str] = None,
                dtype: str = "auto"):
        """
        Initialize the object detector.
        
        Args:
            model_size: Size of Qwen model ("3B" or "7B", 7B recommended for detection)
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
    
    def _repair_json(self, text: str) -> str:
        """
        Repair common JSON formatting issues in model output.
        
        Args:
            text: Raw text potentially containing JSON
            
        Returns:
            Repaired JSON string
        """
        # Replace newlines inside strings with escaped newlines
        pattern = re.compile(r'("([^"\\]|\\.)*)\n([^"]*")')
        while pattern.search(text):
            text = pattern.sub(lambda m: m.group(1) + r'\n' + m.group(3), text)
        return text
    
    def _extract_json(self, text: str) -> Any:
        """
        Extract JSON from model output that may contain markdown formatting.
        
        Args:
            text: Model output text
            
        Returns:
            Parsed JSON object
        """
        # Look for JSON in markdown code blocks
        block_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        match = block_re.search(text)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # Try to find JSON-like content
            json_str = text.strip()
        
        # Repair and parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try repair
            json_str = self._repair_json(json_str)
            return json.loads(json_str)
    
    def detect(self, 
              image: Image.Image,
              target_objects: Optional[List[str]] = None,
              threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image to analyze
            target_objects: Optional list of specific objects to detect
            threshold: Confidence threshold (not always supported by model)
            
        Returns:
            List of detection dictionaries with 'bbox_2d' and 'label' keys
        """
        # Create prompt based on target objects
        if target_objects:
            objects_str = ", ".join(target_objects)
            user_prompt = f"Detect and outline the position of: {objects_str}"
        else:
            user_prompt = "Detect all objects in this image and provide their positions"
        
        # Create messages
        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an object detector. Output a JSON array where each element "
                            "has the format: {'bbox_2d': [x1, y1, x2, y2], 'label': 'class_name'}"
                        )
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ],
            }
        ]
        
        # Process and generate
        text_prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1000
            )
        
        # Decode output
        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=False
        )[0]
        
        # Extract detections from JSON
        try:
            detections = self._extract_json(output)
            if not isinstance(detections, list):
                detections = [detections]
            return detections
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse detection output: {e}")
            print(f"Raw output: {output}")
            return []
    
    def draw_detections(self,
                       image: Image.Image,
                       detections: List[Dict[str, Any]],
                       box_color: str = "red",
                       box_width: int = 3,
                       font_size: int = 16,
                       text_color: str = "white",
                       text_bg: str = "red") -> Image.Image:
        """
        Draw bounding boxes and labels on an image.
        
        Args:
            image: PIL Image to draw on
            detections: List of detection dictionaries
            box_color: Color of bounding boxes
            box_width: Width of bounding box lines
            font_size: Size of label text
            text_color: Color of label text
            text_bg: Background color for labels
            
        Returns:
            New PIL Image with drawn detections
        """
        # Create a copy to avoid modifying original
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            # Get bounding box
            bbox = det.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            label = str(det.get("label", ""))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
            
            # Draw label if present
            if label:
                # Get text size
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    tw, th = draw.textsize(label, font=font)
                
                # Draw text background
                padding = 2
                if text_bg:
                    draw.rectangle(
                        [x1, y1 - th - 2*padding, x1 + tw + 2*padding, y1],
                        fill=text_bg
                    )
                
                # Draw text
                draw.text((x1 + padding, y1 - th - padding), label, 
                         fill=text_color, font=font)
        
        return img
    
    def count_objects(self, 
                     image: Image.Image,
                     object_class: Optional[str] = None) -> Dict[str, int]:
        """
        Count objects in an image.
        
        Args:
            image: PIL Image to analyze
            object_class: Optional specific class to count
            
        Returns:
            Dictionary mapping object classes to counts
        """
        detections = self.detect(image, [object_class] if object_class else None)
        
        counts = {}
        for det in detections:
            label = det.get("label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        
        return counts
    
    def find_relationships(self, 
                         image: Image.Image) -> str:
        """
        Analyze spatial relationships between objects.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Description of spatial relationships
        """
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the spatial relationships between objects in this image. Include information about relative positions (left, right, above, below), distances, and any interactions."}
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(msgs)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=200
            )
        
        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
        
        return output.strip()
    
    def segment_regions(self, 
                       image: Image.Image,
                       region_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        Segment image into regions (simplified version using detection).
        
        Args:
            image: PIL Image to segment
            region_type: Type of segmentation ("semantic" or "instance")
            
        Returns:
            List of region dictionaries
        """
        # Note: This is a simplified approach using detection
        # True segmentation would require specialized models
        detections = self.detect(image)
        
        # Group by class for semantic segmentation
        if region_type == "semantic":
            regions = {}
            for det in detections:
                label = det.get("label", "unknown")
                if label not in regions:
                    regions[label] = []
                regions[label].append(det["bbox_2d"])
            
            return [{"label": label, "regions": boxes} 
                   for label, boxes in regions.items()]
        else:
            # Instance segmentation - each detection is a separate instance
            return [{"label": det.get("label", "unknown"), 
                    "bbox": det["bbox_2d"], 
                    "instance_id": i} 
                   for i, det in enumerate(detections)]
    
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