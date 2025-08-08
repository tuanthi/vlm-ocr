"""
Example 2: Zero-Shot Image Classification
==========================================

This example demonstrates how to:
1. Classify images without training
2. Use custom class labels
3. Get confidence scores and explanations
"""

from visionlang import ZeroShotClassifier
from visionlang.utils import ImageLoader, plot_images_grid
import json


def main():
    print("VisionLang - Example 2: Zero-Shot Classification\n")
    print("=" * 50)
    
    # Initialize classifier
    print("\n1. Initializing CLIP classifier...")
    classifier = ZeroShotClassifier(model_name="openai/clip-vit-base-patch32")
    print("   Classifier ready!")
    
    # Example 1: Basic classification
    print("\n2. Basic Classification Example")
    print("-" * 30)
    
    # Load a test image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = ImageLoader.from_url(image_url)
    
    # Define possible classes
    basic_classes = ["cat", "dog", "bird", "horse", "rabbit"]
    
    # Classify
    prediction = classifier.classify(image, basic_classes)
    print(f"   Predicted class: {prediction}")
    
    # Get scores for all classes
    prediction, scores = classifier.classify(image, basic_classes, return_scores=True)
    print("\n   Confidence scores:")
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {label}: {score:.4f}")
    
    # Example 2: Using templates for better accuracy
    print("\n3. Classification with Templates")
    print("-" * 30)
    
    # Templates can improve accuracy
    template = "a photo of a {}"
    
    prediction_with_template = classifier.classify(
        image, 
        basic_classes, 
        template=template
    )
    print(f"   Prediction with template: {prediction_with_template}")
    
    # Example 3: Fine-grained classification
    print("\n4. Fine-grained Classification")
    print("-" * 30)
    
    # More specific classes
    detailed_classes = [
        "tabby cat",
        "persian cat",
        "siamese cat",
        "black cat",
        "orange cat",
        "white cat"
    ]
    
    top_3 = classifier.get_top_k_predictions(image, detailed_classes, k=3)
    print("   Top 3 predictions:")
    for label, score in top_3:
        print(f"   - {label}: {score:.4f}")
    
    # Example 4: Batch classification
    print("\n5. Batch Classification")
    print("-" * 30)
    
    # Load multiple images
    image_urls = {
        "cat": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "airplane": "http://images.cocodataset.org/val2017/000000139099.jpg",
        "person": "http://images.cocodataset.org/val2017/000000391895.jpg"
    }
    
    images = []
    true_labels = []
    for label, url in image_urls.items():
        try:
            img = ImageLoader.from_url(url)
            images.append(img)
            true_labels.append(label)
        except:
            pass
    
    # Classify all images
    classes = ["person", "animal", "vehicle", "food", "furniture"]
    results = classifier.classify_batch(images, classes)
    
    print("   Batch results:")
    for i, (pred, scores) in enumerate(results):
        print(f"   Image {i+1}: {pred} (confidence: {scores[pred]:.4f})")
    
    # Example 5: Detailed explanation
    print("\n6. Getting Detailed Explanations")
    print("-" * 30)
    
    explanation = classifier.explain_prediction(image, basic_classes)
    
    print(f"   Predicted: {explanation['predicted_label']}")
    print(f"   Confidence: {explanation['confidence']:.4f}")
    print(f"   Uncertainty: {explanation['uncertainty']:.4f}")
    print(f"   Temperature scale: {explanation['temperature_scale']:.2f}")
    print("\n   Top 3 predictions:")
    for label, score in explanation['top_3']:
        print(f"   - {label}: {score:.4f}")
    
    # Example 6: Custom domain classification
    print("\n7. Domain-Specific Classification")
    print("-" * 30)
    
    # Example: Medical/Health context
    health_classes = [
        "healthy animal",
        "injured animal",
        "domestic pet",
        "wild animal",
        "veterinary setting"
    ]
    
    health_prediction, health_scores = classifier.classify(
        image,
        health_classes,
        return_scores=True,
        template="a photo showing {}"
    )
    
    print(f"   Health context prediction: {health_prediction}")
    print("   Scores:")
    for label, score in sorted(health_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   - {label}: {score:.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey Takeaways:")
    print("- Zero-shot classification works without training data")
    print("- Templates can improve classification accuracy")
    print("- The model can handle custom, domain-specific classes")
    print("- Confidence scores help assess prediction reliability")


if __name__ == "__main__":
    main()