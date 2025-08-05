#!/usr/bin/env python3
"""
CPU-only test for Qwen2.5-VL - may be slow but avoids device issues
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU usage
import torch
torch.set_default_device('cpu')

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def test_invoice_cpu():
    print("Running CPU-only test (this may take a while)...")
    
    # Model configuration
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    print("\nLoading model to CPU...")
    
    # Load model on CPU with float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,  # No device mapping for CPU
        low_cpu_mem_usage=True
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded!")
    
    # Simple prompt
    prompt = "Extract the invoice number, date, seller name, client name, and total amount from this invoice."
    
    # Image path
    image_path = "/Users/huetuanthi/dev/dokeai/vlm-ocr/images/invoice.png"
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Process inputs
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
    
    print("\nGenerating (this may take several minutes on CPU)...")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=200,  # Shorter for faster testing
            do_sample=False
        )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    print("\n=== RESULT ===")
    print(output_text)
    print("==============")

if __name__ == "__main__":
    test_invoice_cpu()