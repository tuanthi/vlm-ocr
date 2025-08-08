"""Setup script for VisionLang framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="visionlang",
    version="0.1.0",
    author="VisionLang Contributors",
    description="Advanced Vision-Language Model Framework for Multimodal AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/visionlang",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0",
        "qwen-vl-utils>=0.0.11",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "gpu": [
            "accelerate>=0.20.0",
            "bitsandbytes>=0.40.0",
        ],
    },
)