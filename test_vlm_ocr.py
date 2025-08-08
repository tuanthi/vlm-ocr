#!/usr/bin/env python3
"""
Test script for VLM-OCR extraction
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vlm_ocr_extraction import VLMOCRExtractor
import json
from pathlib import Path


def test_invoice_ocr():
    """Test OCR extraction on invoice image."""
    
    print("=" * 60)
    print("VLM-OCR TEST: Invoice Text Extraction with Bounding Boxes")
    print("=" * 60)
    
    # Initialize extractor
    print("\nInitializing VLM-OCR Extractor...")
    extractor = VLMOCRExtractor(model_size="7B")
    
    # Test on invoice image
    invoice_path = "./images/invoice.png"
    
    # Test 1: Line-level extraction
    print("\n" + "=" * 40)
    print("TEST 1: Line-level Text Extraction")
    print("=" * 40)
    
    line_elements = extractor.extract_text_with_boxes(
        invoice_path,
        granularity="line"
    )
    
    print(f"\nFound {len(line_elements)} text lines:")
    for i, elem in enumerate(line_elements[:5], 1):  # Show first 5
        print(f"\n{i}. Text: {elem['text'][:50]}...")
        print(f"   BBox: {elem['bbox']}")
    
    if len(line_elements) > 5:
        print(f"\n... and {len(line_elements) - 5} more lines")
    
    # Save line results
    with open("invoice_lines.json", "w", encoding="utf-8") as f:
        json.dump(line_elements, f, indent=2, ensure_ascii=False)
    print("\nLine extraction results saved to: invoice_lines.json")
    
    # Test 2: Word-level extraction
    print("\n" + "=" * 40)
    print("TEST 2: Word-level Text Extraction")
    print("=" * 40)
    
    word_elements = extractor.extract_text_with_boxes(
        invoice_path,
        granularity="word"
    )
    
    print(f"\nFound {len(word_elements)} words")
    print("Sample words:")
    for i, elem in enumerate(word_elements[:10], 1):  # Show first 10
        print(f"  {i}. '{elem['text']}' at {elem['bbox']}")
    
    # Save word results
    with open("invoice_words.json", "w", encoding="utf-8") as f:
        json.dump(word_elements, f, indent=2, ensure_ascii=False)
    print("\nWord extraction results saved to: invoice_words.json")
    
    # Test 3: Structured document extraction
    print("\n" + "=" * 40)
    print("TEST 3: Structured Document Extraction")
    print("=" * 40)
    
    structured_data = extractor.extract_structured_document(invoice_path)
    
    print("\nDocument Type:", structured_data.get("document_type", "Unknown"))
    
    if "headers" in structured_data:
        print(f"\nHeaders ({len(structured_data['headers'])} found):")
        for header in structured_data["headers"][:3]:
            print(f"  - {header['text']} at {header['bbox']}")
    
    if "fields" in structured_data:
        print(f"\nFields ({len(structured_data['fields'])} found):")
        for field in structured_data["fields"][:5]:
            label = field.get("label", "")
            value = field.get("value", "")
            print(f"  - {label}: {value}")
    
    # Save structured results
    with open("invoice_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print("\nStructured extraction results saved to: invoice_structured.json")
    
    # Test 4: Visualization
    print("\n" + "=" * 40)
    print("TEST 4: Visualization")
    print("=" * 40)
    
    print("\nCreating visualization with bounding boxes...")
    annotated_image = extractor.visualize_text_regions(
        invoice_path,
        line_elements[:20],  # Visualize first 20 lines to avoid clutter
        output_path="invoice_annotated.png",
        box_color="blue",
        text_color="red",
        show_text=False  # Don't show text to avoid clutter
    )
    print("Visualization saved to: invoice_annotated.png")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print("\nOutput files created:")
    print("  - invoice_lines.json       : Line-level text extraction")
    print("  - invoice_words.json       : Word-level text extraction")
    print("  - invoice_structured.json  : Structured document data")
    print("  - invoice_annotated.png    : Visualization with bounding boxes")


def test_multiple_pages():
    """Test OCR on multiple page images if available."""
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE PAGE DOCUMENTS")
    print("=" * 60)
    
    # Check for page images
    page_files = [
        "./images/page_1.png",
        "./images/page_2.png"
    ]
    
    existing_pages = [p for p in page_files if Path(p).exists()]
    
    if not existing_pages:
        print("No page images found. Skipping multi-page test.")
        return
    
    print(f"\nFound {len(existing_pages)} page images")
    
    # Initialize extractor
    extractor = VLMOCRExtractor(model_size="7B")
    
    all_pages_data = []
    
    for i, page_path in enumerate(existing_pages, 1):
        print(f"\nProcessing page {i}: {Path(page_path).name}")
        
        # Extract text from page
        text_elements = extractor.extract_text_with_boxes(
            page_path,
            granularity="paragraph"
        )
        
        page_data = {
            "page": i,
            "file": Path(page_path).name,
            "text_elements": text_elements
        }
        all_pages_data.append(page_data)
        
        print(f"  Extracted {len(text_elements)} text blocks")
        
        # Show sample
        if text_elements:
            sample = text_elements[0]
            print(f"  Sample: {sample['text'][:100]}...")
    
    # Save combined results
    with open("pages_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMulti-page results saved to: pages_ocr_results.json")


if __name__ == "__main__":
    # Run invoice test
    test_invoice_ocr()
    
    # Optionally test multiple pages
    test_multiple_pages()