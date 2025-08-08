"""
Visualization Utilities
=======================

Helper functions for visualizing embeddings, similarities, and detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import matplotlib.patches as patches


def plot_similarity_matrix(similarity_matrix: np.ndarray,
                          labels_x: List[str],
                          labels_y: Optional[List[str]] = None,
                          title: str = "Similarity Matrix",
                          figsize: Tuple[int, int] = (8, 6),
                          cmap: str = "coolwarm",
                          save_path: Optional[str] = None) -> None:
    """
    Plot a similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: 2D numpy array of similarities
        labels_x: Labels for x-axis
        labels_y: Labels for y-axis (if None, uses labels_x)
        title: Title for the plot
        figsize: Figure size (width, height)
        cmap: Colormap for the heatmap
        save_path: Optional path to save the figure
    """
    if labels_y is None:
        labels_y = labels_x
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels_x,
        yticklabels=labels_y,
        cbar_kws={'label': 'Similarity'},
        vmin=-1,
        vmax=1
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Items", fontsize=12)
    plt.ylabel("Items", fontsize=12)
    
    # Rotate x labels if they're long
    if any(len(label) > 10 for label in labels_x):
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_images_grid(images: List[Image.Image],
                    labels: Optional[List[str]] = None,
                    predictions: Optional[List[str]] = None,
                    scores: Optional[List[float]] = None,
                    cols: int = 4,
                    figsize: Optional[Tuple[int, int]] = None,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None) -> None:
    """
    Plot multiple images in a grid layout.
    
    Args:
        images: List of PIL Images
        labels: Optional true labels for images
        predictions: Optional predicted labels
        scores: Optional confidence scores
        cols: Number of columns in the grid
        figsize: Figure size (if None, automatically determined)
        title: Overall title for the figure
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    if figsize is None:
        figsize = (cols * 3, rows * 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_images:
            ax.imshow(images[idx])
            
            # Create subtitle
            subtitle_parts = []
            if labels and idx < len(labels):
                subtitle_parts.append(f"True: {labels[idx]}")
            if predictions and idx < len(predictions):
                subtitle_parts.append(f"Pred: {predictions[idx]}")
            if scores and idx < len(scores):
                subtitle_parts.append(f"Conf: {scores[idx]:.2f}")
            
            if subtitle_parts:
                ax.set_title("\n".join(subtitle_parts), fontsize=10)
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_detections(image: Image.Image,
                   detections: List[Dict[str, Any]],
                   figsize: Tuple[int, int] = (12, 8),
                   title: str = "Object Detection Results",
                   save_path: Optional[str] = None,
                   show_labels: bool = True,
                   show_confidence: bool = False) -> None:
    """
    Plot detection results with bounding boxes.
    
    Args:
        image: PIL Image
        detections: List of detection dictionaries with 'bbox_2d' and 'label'
        figsize: Figure size
        title: Title for the plot
        save_path: Optional path to save the figure
        show_labels: Whether to show object labels
        show_confidence: Whether to show confidence scores
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    
    # Display image
    ax.imshow(image)
    
    # Color palette for different classes
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    class_colors = {}
    
    for det in detections:
        bbox = det.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        label = det.get("label", "unknown")
        
        # Assign color to class
        if label not in class_colors:
            class_colors[label] = colors[len(class_colors) % len(colors)]
        
        color = class_colors[label]
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        if show_labels:
            label_text = label
            if show_confidence and "confidence" in det:
                label_text += f" ({det['confidence']:.2f})"
            
            # Add text with background
            ax.text(
                x1, y1 - 5,
                label_text,
                fontsize=10,
                color='white',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.7
                )
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend for classes
    if class_colors:
        handles = [patches.Patch(color=color, label=label) 
                  for label, color in class_colors.items()]
        ax.legend(handles=handles, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embedding_space(embeddings: np.ndarray,
                        labels: List[str],
                        method: str = "tsne",
                        figsize: Tuple[int, int] = (10, 8),
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Visualize high-dimensional embeddings in 2D space.
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: List of labels for each embedding
        method: Dimensionality reduction method ("tsne" or "pca")
        figsize: Figure size
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        if title is None:
            title = "t-SNE Embedding Visualization"
    elif method == "pca":
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        if title is None:
            title = "PCA Embedding Visualization"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Get unique labels and assign colors
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot each point
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=[label_colors[labels[i]]], s=100, alpha=0.7)
        plt.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    handles = [plt.scatter([], [], c=[color], s=100, label=label)
              for label, color in label_colors.items()]
    plt.legend(handles=handles, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()