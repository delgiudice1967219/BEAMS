"""
Test script to visualize B-cos heatmaps on sample images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add bcosification to path
sys.path.insert(0, 'bcosification')

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap


def visualize_heatmap(image_path, text_prompts, save_path=None):
    """
    Generate and visualize B-cos heatmaps for an image with multiple text prompts.
    
    Args:
        image_path: Path to the image
        text_prompts: List of text prompts to test
        save_path: Optional path to save the visualization
    """
    # Load models
    print("Loading models...")
    model, device = load_bcos_model()
    clip_model, _ = load_clip_for_text()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded image: {image.size}")
    
    # Create figure
    n_prompts = len(text_prompts)
    fig, axes = plt.subplots(1, n_prompts + 1, figsize=(5 * (n_prompts + 1), 5))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Generate and show heatmaps for each prompt
    for i, prompt in enumerate(text_prompts):
        print(f"Generating heatmap for '{prompt}'...")
        heatmap, score = get_bcos_heatmap(model, image, prompt, clip_model, device, return_raw=True)
        
        print(f"  Score: {score:.4f}")
        print(f"  Heatmap: min={heatmap.min():.4f}, max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")
        
        # Overlay heatmap on image
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f"{prompt}\nScore: {score:.4f}", fontsize=14)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test images and prompts
    test_cases = [
        {
            "image": "cat_background.png",
            "prompts": ["cat", "grass", "tree"],
            "save": "test_cat_background.png"
        },
        {
            "image": "car.png",
            "prompts": ["car", "street", "building"],
            "save": "test_car.png"
        },
        {
            "image": "objects.png",
            "prompts": ["people", "book", "teapot"],
            "save": "test_objects.png"
        }
    ]
    
    for test_case in test_cases:
        image_path = test_case["image"]
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"Testing: {image_path}")
            print(f"{'='*60}")
            visualize_heatmap(image_path, test_case["prompts"], test_case["save"])
        else:
            print(f"Skipping {image_path} (not found)")
    
    print("\n✅ All visualizations complete!")
