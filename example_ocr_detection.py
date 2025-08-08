"""
OCR with Object Detection Example
==================================

This script demonstrates how to use VisionLang's object detection capabilities
for OCR purposes, extracting text with bounding boxes and classifying document
elements (tables, figures, formulas, paragraphs).
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from visionlang import ObjectDetector, ImageCaptioner


@dataclass
class OCRResult:
    """Container for OCR detection results"""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    element_type: str  # 'table', 'figure', 'formula', 'paragraph', 'title', 'caption'
    
    def to_dict(self):
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "element_type": self.element_type
        }


class DocumentOCR:
    """
    Advanced OCR system using Vision-Language Models for text extraction
    and document structure understanding.
    """
    
    def __init__(self, model_size: str = "7B", device: str = "cuda"):
        """
        Initialize the Document OCR system.
        
        Args:
            model_size: Size of Qwen2.5-VL model ('3B' or '7B')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.detector = ObjectDetector(model_size=model_size, device=device)
        self.captioner = ImageCaptioner(model_size=model_size, device=device)
        
        # Document element types to detect
        self.element_types = [
            "text paragraph",
            "table",
            "figure",
            "mathematical formula",
            "title",
            "caption",
            "header",
            "footer",
            "list",
            "code block"
        ]
    
    def extract_text_with_layout(self, image: Image.Image) -> List[OCRResult]:
        """
        Extract text from image with bounding boxes and document structure.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of OCRResult objects containing text, bbox, and element type
        """
        results = []
        
        # First, detect all text regions and document elements
        detections = self.detector.detect(
            image,
            target_objects=self.element_types + ["text", "word", "line"]
        )
        
        # Process each detection
        for detection in detections:
            bbox = detection.get("bbox_2d", [0, 0, 0, 0])
            
            # Crop the region for text extraction
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2))
            
            # Extract text from the region using captioning
            text_prompt = "Extract and transcribe all text in this image exactly as written:"
            extracted_text = self.captioner.caption(
                cropped,
                prompt=text_prompt
            )
            
            # Classify the element type
            element_type = self._classify_element(detection, cropped)
            
            # Calculate confidence (you might get this from the model)
            confidence = detection.get("confidence", 0.95)
            
            results.append(OCRResult(
                text=extracted_text,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                element_type=element_type
            ))
        
        return results
    
    def _classify_element(self, detection: Dict, region: Image.Image) -> str:
        """
        Classify the type of document element.
        
        Args:
            detection: Detection dictionary from object detector
            region: Cropped image region
            
        Returns:
            Element type string
        """
        # Get initial label from detection
        label = detection.get("label", "").lower()
        
        # Map detection labels to document element types
        if "table" in label:
            return "table"
        elif "figure" in label or "image" in label or "chart" in label:
            return "figure"
        elif "formula" in label or "equation" in label or "math" in label:
            return "formula"
        elif "title" in label or "heading" in label:
            return "title"
        elif "caption" in label:
            return "caption"
        elif "code" in label:
            return "code"
        elif "list" in label:
            return "list"
        else:
            # Use visual features to further classify
            classification_prompt = """
            Classify this document element as one of:
            - paragraph (regular text block)
            - table (structured data in rows/columns)
            - figure (image, chart, diagram)
            - formula (mathematical equation)
            - title (heading or title text)
            - caption (description of figure/table)
            - code (programming code)
            - list (bulleted or numbered list)
            
            Return only the classification label.
            """
            
            element_type = self.captioner.caption(
                region,
                prompt=classification_prompt
            ).strip().lower()
            
            # Ensure valid element type
            valid_types = ["paragraph", "table", "figure", "formula", 
                          "title", "caption", "code", "list"]
            if element_type not in valid_types:
                element_type = "paragraph"  # Default
            
            return element_type
    
    def analyze_document_structure(self, image: Image.Image) -> Dict:
        """
        Analyze the overall document structure and layout.
        
        Args:
            image: PIL Image of document
            
        Returns:
            Dictionary containing document structure analysis
        """
        # Extract all elements
        elements = self.extract_text_with_layout(image)
        
        # Analyze structure
        structure = {
            "total_elements": len(elements),
            "element_counts": {},
            "reading_order": [],
            "layout_type": None,
            "columns": 1,
            "has_headers": False,
            "has_footers": False,
            "elements": []
        }
        
        # Count element types
        for element in elements:
            elem_type = element.element_type
            structure["element_counts"][elem_type] = \
                structure["element_counts"].get(elem_type, 0) + 1
            structure["elements"].append(element.to_dict())
        
        # Determine reading order based on position
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.bbox[1], e.bbox[0])  # Sort by top, then left
        )
        structure["reading_order"] = [
            {"text": e.text[:50] + "..." if len(e.text) > 50 else e.text,
             "type": e.element_type,
             "position": e.bbox}
            for e in sorted_elements
        ]
        
        # Detect layout type
        structure["layout_type"] = self._detect_layout_type(elements)
        
        # Detect columns
        structure["columns"] = self._detect_columns(elements)
        
        # Check for headers/footers
        img_height = image.height
        for element in elements:
            if element.bbox[1] < img_height * 0.1:  # Top 10%
                structure["has_headers"] = True
            if element.bbox[3] > img_height * 0.9:  # Bottom 10%
                structure["has_footers"] = True
        
        return structure
    
    def _detect_layout_type(self, elements: List[OCRResult]) -> str:
        """Detect the document layout type."""
        if not elements:
            return "empty"
        
        # Check element types
        has_tables = any(e.element_type == "table" for e in elements)
        has_figures = any(e.element_type == "figure" for e in elements)
        has_formulas = any(e.element_type == "formula" for e in elements)
        
        if has_tables and has_figures:
            return "mixed_content"
        elif has_tables:
            return "data_heavy"
        elif has_figures:
            return "visual_heavy"
        elif has_formulas:
            return "technical"
        else:
            return "text_heavy"
    
    def _detect_columns(self, elements: List[OCRResult]) -> int:
        """Detect number of columns in the document."""
        if not elements:
            return 0
        
        # Group elements by horizontal position
        x_positions = [e.bbox[0] for e in elements]
        if not x_positions:
            return 1
        
        # Simple clustering based on x-position gaps
        x_positions.sort()
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Significant gap threshold
                gaps.append(gap)
        
        # Estimate columns based on significant gaps
        if len(gaps) == 0:
            return 1
        elif len(gaps) == 1:
            return 2
        else:
            return min(3, len(gaps) + 1)  # Cap at 3 columns
    
    def visualize_results(
        self,
        image: Image.Image,
        results: List[OCRResult],
        output_path: str = "ocr_visualization.png"
    ):
        """
        Visualize OCR results with bounding boxes and labels.
        
        Args:
            image: Original image
            results: List of OCR results
            output_path: Path to save visualization
        """
        # Create a copy for drawing
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Color map for different element types
        color_map = {
            "paragraph": "blue",
            "table": "green",
            "figure": "orange",
            "formula": "red",
            "title": "purple",
            "caption": "brown",
            "code": "gray",
            "list": "cyan"
        }
        
        # Draw bounding boxes and labels
        for result in results:
            color = color_map.get(result.element_type, "black")
            
            # Draw rectangle
            draw.rectangle(result.bbox, outline=color, width=2)
            
            # Add label
            label = f"{result.element_type}: {result.text[:30]}..."
            draw.text(
                (result.bbox[0], result.bbox[1] - 20),
                label,
                fill=color
            )
        
        # Save visualization
        img_draw.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    def export_results(
        self,
        results: List[OCRResult],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export OCR results in various formats.
        
        Args:
            results: List of OCR results
            format: Export format ('json', 'xml', 'markdown', 'hocr')
            output_path: Optional path to save output
            
        Returns:
            Formatted string of results
        """
        if format == "json":
            output = json.dumps([r.to_dict() for r in results], indent=2)
        
        elif format == "markdown":
            output = "# OCR Extraction Results\n\n"
            for r in results:
                output += f"## {r.element_type.title()}\n"
                output += f"- **Text**: {r.text}\n"
                output += f"- **Position**: {r.bbox}\n"
                output += f"- **Confidence**: {r.confidence:.2f}\n\n"
        
        elif format == "xml":
            output = '<?xml version="1.0" encoding="UTF-8"?>\n'
            output += '<document>\n'
            for r in results:
                output += f'  <element type="{r.element_type}">\n'
                output += f'    <text>{r.text}</text>\n'
                output += f'    <bbox x1="{r.bbox[0]}" y1="{r.bbox[1]}" '
                output += f'x2="{r.bbox[2]}" y2="{r.bbox[3]}"/>\n'
                output += f'    <confidence>{r.confidence}</confidence>\n'
                output += '  </element>\n'
            output += '</document>'
        
        elif format == "hocr":
            # hOCR format for compatibility with OCR tools
            output = '<?xml version="1.0" encoding="UTF-8"?>\n'
            output += '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">\n'
            output += '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n'
            output += '<head><title>OCR Results</title></head>\n'
            output += '<body>\n'
            output += '<div class="ocr_page">\n'
            
            for i, r in enumerate(results):
                bbox_str = f"{r.bbox[0]} {r.bbox[1]} {r.bbox[2]} {r.bbox[3]}"
                output += f'  <div class="ocr_{r.element_type}" id="element_{i}" '
                output += f'title="bbox {bbox_str}; confidence {r.confidence:.2f}">\n'
                output += f'    {r.text}\n'
                output += '  </div>\n'
            
            output += '</div>\n</body>\n</html>'
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Results exported to {output_path}")
        
        return output


def main():
    """Example usage of the Document OCR system."""
    
    # Initialize OCR system
    ocr = DocumentOCR(model_size="7B", device="cuda")
    
    # Load document image
    image_path = "sample_document.png"  # Replace with your document
    image = Image.open(image_path)
    
    # Extract text with layout information
    print("Extracting text and layout...")
    results = ocr.extract_text_with_layout(image)
    
    # Print results
    print(f"\nFound {len(results)} elements:")
    for result in results:
        print(f"\n[{result.element_type.upper()}]")
        print(f"Text: {result.text[:100]}...")
        print(f"Position: {result.bbox}")
        print(f"Confidence: {result.confidence:.2f}")
    
    # Analyze document structure
    print("\nAnalyzing document structure...")
    structure = ocr.analyze_document_structure(image)
    print(f"Layout type: {structure['layout_type']}")
    print(f"Columns: {structure['columns']}")
    print(f"Element counts: {structure['element_counts']}")
    
    # Export results in different formats
    print("\nExporting results...")
    ocr.export_results(results, format="json", output_path="ocr_results.json")
    ocr.export_results(results, format="markdown", output_path="ocr_results.md")
    ocr.export_results(results, format="hocr", output_path="ocr_results.html")
    
    # Visualize results
    print("\nCreating visualization...")
    ocr.visualize_results(image, results, "ocr_visualization.png")
    
    print("\nOCR processing complete!")


if __name__ == "__main__":
    main()