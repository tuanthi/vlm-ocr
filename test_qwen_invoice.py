#!/usr/bin/env python3
"""
Test Qwen2.5-VL on invoice image
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json

def test_invoice_extraction():
    print("Loading Qwen2.5-VL model...")
    
    # Model configuration
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Load model with automatic device mapping
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Invoice extraction prompt
    invoice_prompt = """Extract all information from this invoice and return it as a JSON object with the following structure:
{
    "invoice_number": "",
    "date": "",
    "seller": {
        "name": "",
        "address": "",
        "tax_id": "",
        "iban": ""
    },
    "client": {
        "name": "",
        "address": "",
        "tax_id": ""
    },
    "items": [
        {
            "description": "",
            "quantity": 0,
            "unit": "",
            "net_price": 0,
            "net_worth": 0,
            "vat_percent": 0,
            "gross_worth": 0
        }
    ],
    "summary": {
        "vat_percent": 0,
        "net_worth": 0,
        "vat": 0,
        "gross_worth": 0
    },
    "total": {
        "net_worth": 0,
        "vat": 0,
        "gross_worth": 0
    }
}"""
    
    # Image path
    image_path = "/Users/huetuanthi/dev/dokeai/vlm-ocr/images/invoice.png"
    
    print(f"\nProcessing invoice: {image_path}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": invoice_prompt},
            ],
        }
    ]
    
    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    
    print("\nGenerating response...")
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print("\nRaw output:")
    print(output_text)
    
    # Try to parse JSON
    try:
        result = json.loads(output_text)
        print("\nParsed JSON:")
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("\nCould not parse as JSON. Attempting to extract JSON from text...")
        # Try to find JSON in the output
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                print("\nExtracted JSON:")
                print(json.dumps(result, indent=2))
            except:
                print("Failed to extract valid JSON")
    
    # Save results
    with open("invoice_extraction_result.txt", "w") as f:
        f.write("Raw output:\n")
        f.write(output_text)
        f.write("\n\n")
        if 'result' in locals():
            f.write("Parsed JSON:\n")
            f.write(json.dumps(result, indent=2))
    
    print("\nResults saved to invoice_extraction_result.txt")

if __name__ == "__main__":
    test_invoice_extraction()