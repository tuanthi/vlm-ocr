"""
Data Loading Utilities
======================

Helper functions for loading images from various sources.
"""

import os
from typing import List, Dict, Optional, Union
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path


class ImageLoader:
    """
    Utility class for loading images from various sources.
    
    This class provides methods to load images from URLs, local files,
    directories, and common datasets.
    """
    
    @staticmethod
    def from_url(url: str, timeout: int = 15) -> Image.Image:
        """
        Load an image from a URL.
        
        Args:
            url: URL of the image
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    
    @staticmethod
    def from_file(path: Union[str, Path]) -> Image.Image:
        """
        Load an image from a local file.
        
        Args:
            path: Path to the image file
            
        Returns:
            PIL Image object
        """
        return Image.open(path).convert("RGB")
    
    @staticmethod
    def from_directory(directory: Union[str, Path], 
                      extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
                      limit: Optional[int] = None) -> Dict[str, Image.Image]:
        """
        Load all images from a directory.
        
        Args:
            directory: Path to directory containing images
            extensions: List of valid image extensions
            limit: Maximum number of images to load
            
        Returns:
            Dictionary mapping filenames to PIL Images
        """
        directory = Path(directory)
        images = {}
        count = 0
        
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in extensions:
                try:
                    images[file_path.name] = Image.open(file_path).convert("RGB")
                    count += 1
                    if limit and count >= limit:
                        break
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
        
        return images
    
    @staticmethod
    def load_sample_images() -> Dict[str, Image.Image]:
        """
        Load a set of sample images for testing.
        
        Returns:
            Dictionary of sample images with descriptive keys
        """
        sample_urls = {
            "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
            "dog": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
            "bird": "https://images.unsplash.com/photo-1555169062-013468b47731?w=400",
            "flower": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400",
            "car": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",
            "building": "https://images.unsplash.com/photo-1486718448742-163732cd1544?w=400"
        }
        
        images = {}
        for name, url in sample_urls.items():
            try:
                images[name] = ImageLoader.from_url(url)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        
        return images
    
    @staticmethod
    def load_coco_samples() -> Dict[str, Image.Image]:
        """
        Load sample images from COCO dataset.
        
        Returns:
            Dictionary of COCO sample images
        """
        coco_urls = {
            "person_bike": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "kitchen": "http://images.cocodataset.org/val2017/000000252219.jpg",
            "zebra": "http://images.cocodataset.org/val2017/000000087038.jpg",
            "train": "http://images.cocodataset.org/val2017/000000174482.jpg",
            "pizza": "http://images.cocodataset.org/val2017/000000092091.jpg",
            "elephant": "http://images.cocodataset.org/val2017/000000262682.jpg"
        }
        
        images = {}
        for name, url in coco_urls.items():
            try:
                images[name] = ImageLoader.from_url(url)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        
        return images
    
    @staticmethod
    def create_image_grid(images: List[Image.Image], 
                         cols: int = 3,
                         padding: int = 10,
                         background_color: str = "white") -> Image.Image:
        """
        Create a grid image from multiple images.
        
        Args:
            images: List of PIL Images
            cols: Number of columns in the grid
            padding: Padding between images
            background_color: Background color for the grid
            
        Returns:
            Combined grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        # Calculate grid dimensions
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        # Get maximum dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Calculate grid size
        grid_width = cols * max_width + (cols + 1) * padding
        grid_height = rows * max_height + (rows + 1) * padding
        
        # Create grid image
        grid = Image.new("RGB", (grid_width, grid_height), background_color)
        
        # Paste images into grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            x = col * max_width + (col + 1) * padding
            y = row * max_height + (row + 1) * padding
            
            # Center image in cell if smaller than max
            x_offset = (max_width - img.width) // 2
            y_offset = (max_height - img.height) // 2
            
            grid.paste(img, (x + x_offset, y + y_offset))
        
        return grid
    
    @staticmethod
    def resize_image(image: Image.Image, 
                    max_size: int = 800,
                    maintain_aspect: bool = True) -> Image.Image:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension (width or height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if maintain_aspect:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / image.width, max_size / image.height)
            if ratio < 1:
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize((max_size, max_size), Image.Resampling.LANCZOS)