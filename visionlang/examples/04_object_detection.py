"""
Example 4: Object Detection with Qwen2.5-VL
============================================

This example demonstrates how to:
1. Detect objects in images
2. Draw bounding boxes
3. Count objects
4. Analyze spatial relationships
"""

from visionlang import ObjectDetector
from visionlang.utils import ImageLoader, plot_detections
import torch
from PIL import Image
import json


def main():
    print("VisionLang - Example 4: Object Detection\n")
    print("=" * 50)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. This may be slow.")
    
    # Initialize detector
    print("\n1. Initializing Qwen2.5-VL detector...")
    print("   (Using 7B model for better detection accuracy)")
    detector = ObjectDetector(model_size="7B", device=device)
    print("   Detector ready!")
    
    # Example 1: Basic object detection
    print("\n2. Basic Object Detection")
    print("-" * 30)
    
    # Load test image with multiple objects
    image_url = "https://learnopencv.com/wp-content/uploads/2025/06/elephants.jpg"
    image = ImageLoader.from_url(image_url)
    print("   Image loaded: elephants scene")
    
    # Detect all objects
    print("\n   Detecting objects...")
    detections = detector.detect(image)
    
    print(f"   Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        label = det.get("label", "unknown")
        bbox = det.get("bbox_2d", [])
        print(f"   {i+1}. {label} at {bbox}")
    
    # Visualize detections
    if detections:
        annotated_image = detector.draw_detections(
            image, 
            detections,
            box_color="red",
            text_bg="red"
        )
        # Note: In a real application, you would display or save this image
        print("   Bounding boxes drawn on image")
    
    # Example 2: Targeted object detection
    print("\n3. Targeted Object Detection")
    print("-" * 30)
    
    # Load a street scene
    street_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    street_image = ImageLoader.from_url(street_url)
    print("   Loaded street scene image")
    
    # Detect specific objects
    target_objects = ["person", "bicycle", "car", "traffic light"]
    print(f"\n   Looking for: {', '.join(target_objects)}")
    
    targeted_detections = detector.detect(street_image, target_objects)
    
    print(f"   Found {len(targeted_detections)} target objects:")
    for det in targeted_detections:
        print(f"   - {det.get('label', 'unknown')}")
    
    # Example 3: Object counting
    print("\n4. Object Counting")
    print("-" * 30)
    
    # Count objects by class
    counts = detector.count_objects(street_image)
    
    print("   Object counts:")
    for obj_class, count in counts.items():
        print(f"   - {obj_class}: {count}")
    
    # Example 4: Spatial relationships
    print("\n5. Spatial Relationship Analysis")
    print("-" * 30)
    
    print("\n   Analyzing spatial relationships...")
    relationships = detector.find_relationships(street_image)
    print(f"   Analysis: {relationships}")
    
    # Example 5: Kitchen scene analysis
    print("\n6. Complex Scene Analysis")
    print("-" * 30)
    
    kitchen_url = "http://images.cocodataset.org/val2017/000000252219.jpg"
    kitchen_image = ImageLoader.from_url(kitchen_url)
    print("   Loaded kitchen scene")
    
    # Detect kitchen objects
    kitchen_objects = ["refrigerator", "oven", "sink", "cabinet", "microwave", "bottle", "bowl"]
    print(f"\n   Looking for kitchen items...")
    
    kitchen_detections = detector.detect(kitchen_image, kitchen_objects)
    
    if kitchen_detections:
        print(f"   Found {len(kitchen_detections)} kitchen items:")
        for det in kitchen_detections:
            label = det.get("label", "unknown")
            print(f"   - {label}")
    
    # Analyze kitchen layout
    print("\n   Analyzing kitchen layout...")
    kitchen_analysis = detector.find_relationships(kitchen_image)
    print(f"   {kitchen_analysis}")
    
    # Example 6: Animal detection and counting
    print("\n7. Wildlife Detection")
    print("-" * 30)
    
    # Load image with animals
    zebra_url = "http://images.cocodataset.org/val2017/000000087038.jpg"
    zebra_image = ImageLoader.from_url(zebra_url)
    print("   Loaded wildlife scene")
    
    # Detect animals
    animals = ["zebra", "giraffe", "elephant", "lion", "bird"]
    animal_detections = detector.detect(zebra_image, animals)
    
    if animal_detections:
        # Count each type
        animal_counts = {}
        for det in animal_detections:
            label = det.get("label", "unknown")
            animal_counts[label] = animal_counts.get(label, 0) + 1
        
        print("   Wildlife detected:")
        for animal, count in animal_counts.items():
            print(f"   - {count} {animal}(s)")
        
        # Draw and visualize
        annotated_wildlife = detector.draw_detections(
            zebra_image,
            animal_detections,
            box_color="green",
            text_bg="darkgreen"
        )
        print("   Detection boxes added to image")
    
    # Example 7: Region segmentation (simplified)
    print("\n8. Region Segmentation")
    print("-" * 30)
    
    print("\n   Performing semantic segmentation...")
    regions = detector.segment_regions(street_image, region_type="semantic")
    
    print("   Semantic regions found:")
    for region in regions[:5]:  # Show first 5
        print(f"   - {region['label']}: {len(region.get('regions', []))} regions")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey Takeaways:")
    print("- Qwen2.5-VL can detect and localize objects")
    print("- Bounding boxes provide spatial information")
    print("- Object counting and relationship analysis are possible")
    print("- The 7B model provides better detection accuracy")


if __name__ == "__main__":
    main()