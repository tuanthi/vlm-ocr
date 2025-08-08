#!/usr/bin/env python3
"""
VLM-OCR Text Extraction with Bounding Boxes
===========================================

This script uses Qwen2.5-VL to perform OCR (Optical Character Recognition) 
on images, extracting text elements along with their bounding box coordinates.

Features:
- Extract all text elements from an image
- Get precise bounding box coordinates for each text element
- Output structured JSON with text content and location
- Visualize detected text regions on the image
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
import re
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from pathlib import Path


class VLMOCRExtractor:
    """
    A class for extracting text elements with bounding boxes using Qwen2.5-VL.
    
    This extractor performs OCR on images and provides both the text content
    and spatial location (bounding boxes) for each detected text element.
    """
    
    def __init__(self, 
                model_size: str = "7B",
                device: Optional[str] = None):
        """
        Initialize the VLM-OCR extractor.
        
        Args:
            model_size: Size of Qwen model ("3B" or "7B", 7B recommended for better accuracy)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        """
        # Select model based on size
        if model_size == "3B":
            self.model_id = "Qwen/Qwen2-VL-3B-Instruct"
        elif model_size == "7B":
            self.model_id = "Qwen/Qwen2-VL-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Initializing {model_size} model on {self.device}...")
        
        # Load model and processor
        try:
            if self.device == "cuda":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif self.device == "mps":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map={"": self.device},
                    low_cpu_mem_usage=True
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
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
    
    def extract_text_with_boxes(self, 
                               image_path: str,
                               granularity: str = "word") -> List[Dict[str, Any]]:
        """
        Extract text elements with their bounding boxes from an image.
        
        Args:
            image_path: Path to the image file
            granularity: Level of text extraction ("word", "line", "paragraph", "all")
            
        Returns:
            List of dictionaries containing:
                - text: The extracted text content
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: Optional confidence score
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Create prompt based on granularity
        if granularity == "word":
            prompt = """Extract every individual word from this image as separate text elements. 
            For each word, provide its exact bounding box coordinates.
            Output a JSON array where each element has the format:
            {"text": "word_content", "bbox": [x1, y1, x2, y2]}"""
        elif granularity == "line":
            prompt = """Extract every line of text from this image. 
            For each line, provide its exact bounding box coordinates.
            Output a JSON array where each element has the format:
            {"text": "line_content", "bbox": [x1, y1, x2, y2]}"""
        elif granularity == "paragraph":
            prompt = """Extract every paragraph or text block from this image. 
            For each paragraph, provide its exact bounding box coordinates.
            Output a JSON array where each element has the format:
            {"text": "paragraph_content", "bbox": [x1, y1, x2, y2]}"""
        else:  # "all"
            prompt = """Extract ALL text elements from this image at every level (words, lines, blocks). 
            For each text element, provide its exact bounding box coordinates and specify its type.
            Output a JSON array where each element has the format:
            {"text": "content", "bbox": [x1, y1, x2, y2], "type": "word|line|block"}"""
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path if isinstance(image_path, str) else image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process and generate
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        print(f"Extracting {granularity}-level text elements...")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract text elements from JSON
        try:
            text_elements = self._extract_json(output)
            if not isinstance(text_elements, list):
                text_elements = [text_elements]
            
            # Validate and clean the output
            valid_elements = []
            for elem in text_elements:
                if isinstance(elem, dict) and "text" in elem and "bbox" in elem:
                    if isinstance(elem["bbox"], list) and len(elem["bbox"]) == 4:
                        valid_elements.append(elem)
            
            return valid_elements
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse OCR output: {e}")
            print(f"Raw output: {output[:500]}...")  # Show first 500 chars
            return []
    
    def visualize_text_regions(self,
                              image_path: str,
                              text_elements: List[Dict[str, Any]],
                              output_path: Optional[str] = None,
                              box_color: str = "blue",
                              text_color: str = "red",
                              show_text: bool = True) -> Image.Image:
        """
        Visualize detected text regions on the image.
        
        Args:
            image_path: Path to the original image
            text_elements: List of text elements with bounding boxes
            output_path: Optional path to save the annotated image
            box_color: Color for bounding boxes
            text_color: Color for text labels
            show_text: Whether to show the extracted text on the image
            
        Returns:
            Annotated PIL Image
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.copy()
        
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        for i, elem in enumerate(text_elements):
            bbox = elem.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            text = elem.get("text", "")
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
            # Draw text label if requested
            if show_text and text:
                # Truncate long text
                display_text = text[:20] + "..." if len(text) > 20 else text
                
                # Draw text above the box
                draw.text((x1, y1 - 15), display_text, fill=text_color, font=font)
        
        if output_path:
            image.save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        return image
    
    def extract_structured_document(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured information from a document image.
        
        This method is specialized for documents like invoices, forms, receipts, etc.
        It attempts to understand the document structure and extract key-value pairs.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary containing structured document information
        """
        prompt = """Analyze this document image and extract all text with their positions.
        Identify the document type and extract structured information including:
        - Headers and titles with positions
        - Key-value pairs (labels and their values) with positions
        - Tables with cell positions
        - Paragraphs with positions
        
        Output as JSON with the following structure:
        {
            "document_type": "invoice|receipt|form|letter|other",
            "headers": [{"text": "...", "bbox": [x1,y1,x2,y2]}],
            "fields": [{"label": "...", "value": "...", "label_bbox": [...], "value_bbox": [...]}],
            "tables": [{"cells": [{"text": "...", "bbox": [...], "row": n, "col": n}]}],
            "paragraphs": [{"text": "...", "bbox": [...]}]
        }"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        print("Extracting structured document information...")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.1
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        try:
            structured_data = self._extract_json(output)
            return structured_data
        except Exception as e:
            print(f"Failed to parse structured output: {e}")
            return {"error": str(e), "raw_output": output[:1000]}


def main():
    """Main function to demonstrate VLM-OCR extraction."""
    
    parser = argparse.ArgumentParser(description="VLM-OCR Text Extraction with Bounding Boxes")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--model-size", choices=["3B", "7B"], default="7B",
                       help="Model size to use (default: 7B)")
    parser.add_argument("--granularity", choices=["word", "line", "paragraph", "all"], 
                       default="line",
                       help="Granularity of text extraction (default: line)")
    parser.add_argument("--structured", action="store_true",
                       help="Extract structured document information")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization of detected text regions")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"],
                       help="Device to use (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Initialize extractor
    extractor = VLMOCRExtractor(model_size=args.model_size, device=args.device)
    
    if args.structured:
        # Extract structured document information
        print(f"\nExtracting structured information from: {image_path}")
        structured_data = extractor.extract_structured_document(str(image_path))
        
        # Pretty print the results
        print("\n=== STRUCTURED DOCUMENT EXTRACTION ===")
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
    
    else:
        # Extract text with bounding boxes
        print(f"\nExtracting {args.granularity}-level text from: {image_path}")
        text_elements = extractor.extract_text_with_boxes(
            str(image_path), 
            granularity=args.granularity
        )
        
        # Display results
        print(f"\n=== EXTRACTED TEXT ELEMENTS ({len(text_elements)} found) ===")
        for i, elem in enumerate(text_elements, 1):
            text = elem.get("text", "")
            bbox = elem.get("bbox", [])
            elem_type = elem.get("type", args.granularity)
            
            print(f"\n{i}. [{elem_type}]")
            print(f"   Text: {text}")
            print(f"   BBox: {bbox}")
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            
            # Save JSON results
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(text_elements, f, indent=2, ensure_ascii=False)
            print(f"\nJSON results saved to: {json_path}")
        
        # Create visualization if requested
        if args.visualize:
            vis_path = output_path.with_suffix(".annotated.png") if args.output else None
            annotated_image = extractor.visualize_text_regions(
                str(image_path),
                text_elements,
                output_path=str(vis_path) if vis_path else None,
                show_text=True
            )
            if vis_path:
                print(f"Visualization saved to: {vis_path}")
    
    print("\n=== OCR EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()