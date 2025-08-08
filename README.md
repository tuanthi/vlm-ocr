# VisionLang: Vision-Language Model Framework 🚀

A professional, production-ready framework for Vision-Language Models (VLMs), providing state-of-the-art implementations for multimodal AI applications with an intuitive API design.

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Detailed Examples](#-detailed-examples)
- [Learning Path](#-learning-path)
- [Performance Considerations](#-performance-considerations)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

## 🎯 Overview

VisionLang delivers enterprise-grade Vision-Language Model capabilities through a clean, modular architecture. From CLIP embeddings and zero-shot classification to advanced image captioning with Qwen2.5-VL, VisionLang provides powerful tools for modern AI applications.

### What You'll Learn

- **Embeddings**: How models represent images and text in a shared space
- **Zero-Shot Classification**: Classify images without training data
- **Image Captioning**: Generate natural language descriptions of images
- **Object Detection**: Locate and identify objects in images
- **Cross-Modal Retrieval**: Find images from text queries and vice versa

## ✨ Features

- 🔧 **Easy-to-use API**: Simple, intuitive interfaces for complex models
- 📖 **Educational Focus**: Extensive documentation and examples for learning
- 🚀 **Production Ready**: Efficient implementations suitable for real applications
- 🎨 **Visualization Tools**: Built-in functions for visualizing results
- 🔄 **Multiple Models**: Support for CLIP and Qwen2.5-VL models
- 💡 **Practical Examples**: Real-world use cases and applications

## 🔧 Installation

### Basic Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### With GPU Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Text and Image Embeddings with CLIP

```python
from visionlang import CLIPEmbedding

# Initialize the model
embedder = CLIPEmbedding()

# Generate embeddings
text_emb = embedder.encode_text("a photo of a cat")
image_emb = embedder.encode_images(your_image)

# Compute similarity
similarity = embedder.text_to_image_similarity(
    ["a cat", "a dog"], 
    [cat_image, dog_image]
)
```

### 2. Zero-Shot Image Classification

```python
from visionlang import ZeroShotClassifier

# Initialize classifier
classifier = ZeroShotClassifier()

# Classify an image
prediction = classifier.classify(
    image, 
    labels=["cat", "dog", "bird"]
)
print(f"This is a: {prediction}")
```

### 3. Image Captioning

```python
from visionlang import ImageCaptioner

# Initialize captioner
captioner = ImageCaptioner(model_size="3B")

# Generate caption
caption = captioner.caption(image, style="descriptive")
print(f"Caption: {caption}")
```

### 4. Object Detection

```python
from visionlang import ObjectDetector

# Initialize detector
detector = ObjectDetector(model_size="7B")

# Detect objects
detections = detector.detect(image)
for det in detections:
    print(f"Found {det['label']} at {det['bbox_2d']}")
```

## 📖 Core Concepts

### 1. Embeddings: The Foundation of VLMs

Embeddings are numerical representations (vectors) that capture the semantic meaning of text and images. CLIP creates these in a shared space where similar concepts are close together.

#### Key Insights:
- **Shared Space**: Text "a cat" and an image of a cat have similar embeddings
- **Cosine Similarity**: Measures how similar two embeddings are (range: -1 to 1)
- **Dimensionality**: CLIP typically uses 512-dimensional vectors

#### Mathematical Foundation:

```
Cosine Similarity = (A·B) / (||A|| × ||B||)

Where:
- A·B is the dot product
- ||A|| is the magnitude of vector A
```

### 2. Zero-Shot Learning: Classification Without Training

Zero-shot learning allows models to classify images into categories they weren't explicitly trained on, using natural language descriptions.

#### How It Works:
1. Encode all possible class labels as text embeddings
2. Encode the image
3. Find the label with highest similarity to the image

#### Advantages:
- No training data required
- Flexible class definitions
- Works with any categories

### 3. Vision-Language Models: Qwen2.5-VL

Qwen2.5-VL is a powerful multimodal model that can understand both images and text, enabling complex tasks like detailed captioning and object detection.

#### Model Sizes:
- **3B Parameters**: Good for captioning, lighter weight
- **7B Parameters**: Better for detection and complex reasoning

#### Capabilities:
- Natural language image descriptions
- Visual question answering
- Object localization
- Spatial reasoning

### 4. Object Detection and Localization

Object detection identifies what objects are in an image and where they're located, providing bounding boxes for each detection.

#### Output Format:
```python
{
    "label": "cat",
    "bbox_2d": [x1, y1, x2, y2],  # Coordinates
    "confidence": 0.95  # Optional confidence score
}
```

## 💡 Detailed Examples

### Example 1: Building an Image Search Engine

```python
from visionlang import CLIPEmbedding
from visionlang.utils import ImageLoader

# Load your image database
image_db = ImageLoader.from_directory("./images")

# Initialize embedder
embedder = CLIPEmbedding()

# Pre-compute image embeddings
image_embeddings = {}
for name, img in image_db.items():
    image_embeddings[name] = embedder.encode_images(img)

# Search function
def search_images(query, top_k=5):
    query_emb = embedder.encode_text(query)
    similarities = {}
    
    for name, img_emb in image_embeddings.items():
        sim = embedder.compute_similarity(query_emb, img_emb)
        similarities[name] = sim[0, 0]
    
    # Return top-k results
    return sorted(similarities.items(), 
                 key=lambda x: x[1], 
                 reverse=True)[:top_k]

# Search for images
results = search_images("a red car in the sunset")
for name, score in results:
    print(f"{name}: {score:.3f}")
```

### Example 2: Custom Object Counter

```python
from visionlang import ObjectDetector

def count_people_and_vehicles(image_path):
    detector = ObjectDetector()
    image = Image.open(image_path)
    
    # Detect common objects
    detections = detector.detect(
        image, 
        target_objects=["person", "car", "truck", "bicycle", "motorcycle"]
    )
    
    # Count by category
    counts = {"people": 0, "vehicles": 0}
    vehicle_types = ["car", "truck", "bicycle", "motorcycle"]
    
    for det in detections:
        label = det["label"]
        if label == "person":
            counts["people"] += 1
        elif label in vehicle_types:
            counts["vehicles"] += 1
    
    return counts

# Use the counter
counts = count_people_and_vehicles("street_scene.jpg")
print(f"People: {counts['people']}, Vehicles: {counts['vehicles']}")
```

### Example 3: Multi-Style Caption Generator

```python
from visionlang import ImageCaptioner

def generate_social_media_post(image):
    captioner = ImageCaptioner()
    
    # Generate different components
    brief = captioner.caption(image, style="brief")
    creative = captioner.caption(image, style="creative")
    
    # Ask specific questions
    mood = captioner.caption(image, 
        prompt="What is the mood of this image in one word?")
    colors = captioner.caption(image,
        prompt="What are the main colors?")
    
    # Compose post
    post = f"""
    📸 {creative}
    
    🎨 Featuring: {colors.lower()}
    💭 Mood: #{mood.lower().replace(' ', '')}
    
    {brief}
    
    #photography #visualart #moments
    """
    
    return post

# Generate a post
image = load_your_image()
post = generate_social_media_post(image)
print(post)
```

## 📚 Learning Path

### Week 1: Understanding Embeddings
1. Run `examples/01_embeddings_basics.py`
2. Experiment with different text descriptions
3. Try computing similarities between your own images
4. Visualize embedding spaces using t-SNE

### Week 2: Zero-Shot Classification
1. Run `examples/02_zero_shot_classification.py`
2. Create custom classifiers for your domain
3. Experiment with prompt templates
4. Compare results with traditional classifiers

### Week 3: Image Captioning
1. Run `examples/03_image_captioning.py`
2. Try different captioning styles
3. Build an image Q&A system
4. Generate captions for your photo collection

### Week 4: Object Detection
1. Run `examples/04_object_detection.py`
2. Build an object counter for specific scenarios
3. Analyze spatial relationships
4. Create annotated datasets

## 📊 Performance Considerations

### Model Selection Guide

| Task | Recommended Model | VRAM Required | Speed |
|------|------------------|---------------|-------|
| Embeddings | CLIP-ViT-B/32 | 2 GB | Fast |
| Classification | CLIP-ViT-B/32 | 2 GB | Fast |
| Basic Captions | Qwen2.5-VL-3B | 6 GB | Medium |
| Detailed Captions | Qwen2.5-VL-7B | 14 GB | Slow |
| Object Detection | Qwen2.5-VL-7B | 14 GB | Slow |

### Optimization Tips

1. **Batch Processing**: Process multiple images together
2. **Caching**: Cache embeddings for repeated queries
3. **GPU Acceleration**: Use CUDA when available
4. **Model Quantization**: Use lower precision for faster inference

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory Error
```python
# Solution: Use smaller model or CPU
captioner = ImageCaptioner(model_size="3B", device="cpu")
```

#### Slow Performance
```python
# Solution: Use batch processing
results = classifier.classify_batch(images, labels)
```

#### Import Errors
```bash
# Solution: Reinstall with proper dependencies
pip install --upgrade transformers torch pillow
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Add support for more VLM models
- Improve visualization tools
- Create more examples
- Enhance documentation
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for CLIP
- Alibaba for Qwen2.5-VL
- The Hugging Face team for Transformers
- The VisionLang community contributors

## 📮 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/vlm-ocr/issues)
- Discussions: [Join the community](https://github.com/yourusername/vlm-ocr/discussions)

---

**Happy Learning! 🎓** Start your Vision-Language Model journey today with VisionLang!