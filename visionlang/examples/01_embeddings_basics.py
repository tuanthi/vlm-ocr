"""
Example 1: Working with CLIP Embeddings
========================================

This example demonstrates how to:
1. Generate text and image embeddings
2. Compute similarities between texts and images
3. Visualize similarity matrices
"""

from visionlang import CLIPEmbedding
from visionlang.utils import plot_similarity_matrix, ImageLoader
import numpy as np


def main():
    print("VisionLang - Example 1: CLIP Embeddings\n")
    print("=" * 50)
    
    # Initialize CLIP embedding model
    print("\n1. Initializing CLIP model...")
    embedder = CLIPEmbedding(model_name="openai/clip-vit-base-patch32")
    print("   Model loaded successfully!")
    
    # Example 1: Text-to-Text Similarity
    print("\n2. Computing text-to-text similarities...")
    texts = [
        "a photo of a cat",
        "a picture of a feline",
        "a dog playing",
        "an airplane in the sky",
        "a flying aircraft"
    ]
    
    text_embeddings = embedder.encode_text(texts, return_numpy=True)
    text_similarity = embedder.text_to_text_similarity(texts, texts)
    
    print(f"   Generated embeddings for {len(texts)} text descriptions")
    print(f"   Embedding dimensions: {text_embeddings.shape}")
    
    # Visualize text similarities
    plot_similarity_matrix(
        text_similarity,
        labels_x=texts,
        title="Text-to-Text Similarity Matrix"
    )
    
    # Example 2: Load sample images
    print("\n3. Loading sample images...")
    sample_images = {
        "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "dog": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
        "airplane": "https://images.unsplash.com/photo-1464037866556-6812c9d1c72e?w=400",
        "car": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400"
    }
    
    images = []
    image_labels = []
    for label, url in sample_images.items():
        try:
            img = ImageLoader.from_url(url)
            images.append(img)
            image_labels.append(label)
            print(f"   Loaded: {label}")
        except Exception as e:
            print(f"   Failed to load {label}: {e}")
    
    # Example 3: Image-to-Image Similarity
    print("\n4. Computing image-to-image similarities...")
    image_similarity = embedder.image_to_image_similarity(images, images)
    
    plot_similarity_matrix(
        image_similarity,
        labels_x=image_labels,
        title="Image-to-Image Similarity Matrix"
    )
    
    # Example 4: Text-to-Image Similarity (Cross-modal)
    print("\n5. Computing text-to-image similarities...")
    text_queries = [
        "a cute cat",
        "a friendly dog",
        "an airplane",
        "a red car"
    ]
    
    cross_modal_similarity = embedder.text_to_image_similarity(text_queries, images)
    
    plot_similarity_matrix(
        cross_modal_similarity,
        labels_x=image_labels,
        labels_y=text_queries,
        title="Text-to-Image Similarity Matrix"
    )
    
    # Print most similar image for each text query
    print("\n6. Finding best matches:")
    for i, query in enumerate(text_queries):
        best_match_idx = np.argmax(cross_modal_similarity[i])
        best_score = cross_modal_similarity[i, best_match_idx]
        print(f"   '{query}' -> '{image_labels[best_match_idx]}' (score: {best_score:.3f})")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey Takeaways:")
    print("- CLIP creates embeddings in a shared space for both text and images")
    print("- Similar concepts have high cosine similarity (close to 1.0)")
    print("- Cross-modal retrieval is possible (finding images from text)")


if __name__ == "__main__":
    main()