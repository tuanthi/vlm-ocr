"""
Example 3: Image Captioning with Qwen2.5-VL
============================================

This example demonstrates how to:
1. Generate captions for images
2. Use different captioning styles
3. Ask questions about images
"""

from visionlang import ImageCaptioner
from visionlang.utils import ImageLoader
import torch


def main():
    print("VisionLang - Example 3: Image Captioning\n")
    print("=" * 50)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. This may be slow.")
    
    # Initialize captioner
    print("\n1. Initializing Qwen2.5-VL captioner...")
    print("   (This may take a moment on first run...)")
    captioner = ImageCaptioner(model_size="3B", device=device)
    print("   Captioner ready!")
    
    # Load test image
    print("\n2. Loading test image...")
    image_url = "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0?w=700"
    image = ImageLoader.from_url(image_url)
    print("   Image loaded successfully!")
    
    # Example 1: Basic caption
    print("\n3. Generating Basic Caption")
    print("-" * 30)
    
    caption = captioner.caption(image)
    print(f"   Caption: {caption}")
    
    # Example 2: Different caption styles
    print("\n4. Different Caption Styles")
    print("-" * 30)
    
    styles = ["brief", "descriptive", "detailed", "creative"]
    
    for style in styles:
        print(f"\n   Style: {style}")
        caption = captioner.caption(image, style=style)
        print(f"   {caption}")
    
    # Example 3: Custom prompts
    print("\n5. Custom Prompts")
    print("-" * 30)
    
    custom_prompts = [
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "Describe the mood or atmosphere of this image.",
        "What time of day does this appear to be?",
        "Count the objects you can see in this image."
    ]
    
    for prompt in custom_prompts:
        print(f"\n   Q: {prompt}")
        answer = captioner.caption(image, prompt=prompt)
        print(f"   A: {answer}")
    
    # Example 4: Interactive Q&A
    print("\n6. Interactive Q&A about the Image")
    print("-" * 30)
    
    questions = [
        "What is in the foreground?",
        "What is in the background?",
        "Is this indoors or outdoors?",
        "What's the weather like?"
    ]
    
    qa_results = captioner.interactive_caption(image, questions)
    
    for question, answer in qa_results.items():
        print(f"\n   Q: {question}")
        print(f"   A: {answer}")
    
    # Example 5: Batch processing
    print("\n7. Batch Caption Generation")
    print("-" * 30)
    
    # Load multiple images
    image_urls = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # cat
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",  # dog
        "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400"   # flower
    ]
    
    images = []
    for i, url in enumerate(image_urls):
        try:
            img = ImageLoader.from_url(url)
            images.append(img)
            print(f"   Loaded image {i+1}")
        except Exception as e:
            print(f"   Failed to load image {i+1}: {e}")
    
    if images:
        print("\n   Generating captions for all images...")
        captions = captioner.caption_batch(images, style="brief")
        
        for i, caption in enumerate(captions):
            print(f"\n   Image {i+1}: {caption}")
    
    # Example 6: Technical analysis
    print("\n8. Technical Image Analysis")
    print("-" * 30)
    
    technical_prompts = [
        "Analyze the composition and framing of this image.",
        "Describe the lighting conditions and shadows.",
        "What photographic techniques might have been used?",
        "Estimate the focal length and depth of field."
    ]
    
    print("\n   Loading a landscape image for technical analysis...")
    landscape_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=700"
    landscape = ImageLoader.from_url(landscape_url)
    
    for prompt in technical_prompts[:2]:  # Just show 2 examples to save time
        print(f"\n   Analysis: {prompt}")
        analysis = captioner.caption(landscape, prompt=prompt, max_length=150)
        print(f"   {analysis}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey Takeaways:")
    print("- Qwen2.5-VL can generate various styles of captions")
    print("- Custom prompts enable specific image analysis")
    print("- The model can answer questions about image content")
    print("- Batch processing is supported for multiple images")


if __name__ == "__main__":
    main()