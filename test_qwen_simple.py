#!/usr/bin/env python3
"""
Simplified test for Qwen2.5-VL on invoice image
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json

def test_invoice_extraction():
    print("Testing Qwen2.5-VL on invoice...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
    
    # Model configuration
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\nLoading model to {device}...")
    
    # Load model with better device handling
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map={"": device} if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Simple prompt for testing
    prompt = """Look at this invoice image and extract:
- Invoice number
- Date
- Seller name and address
- Client name and address
- Item details (description, quantity, price)
- Total amount

Format as JSON."""
    
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
                {"type": "text", "text": prompt},
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
    
    # Move inputs to device
    inputs = inputs.to(device)
    
    print("\nGenerating response...")
    
    # Generate with proper device handling
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,  # Deterministic for testing
                temperature=0.1
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print("\n=== EXTRACTED INFORMATION ===")
        print(output_text)
        print("===========================\n")
        
        # Save results
        with open("invoice_result.txt", "w") as f:
            f.write(output_text)
        print("Results saved to invoice_result.txt")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_invoice_extraction()