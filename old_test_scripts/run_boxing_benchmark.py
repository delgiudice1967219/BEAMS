"""
Benchmark script for bounding box extraction.
Runs B-cos, GradCAM, and Improved CLIP-ES, extracts boxes, and visualizes them.
Images are resized to have smaller edge = 224px while maintaining aspect ratio.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")
# Ensure first_attempt is on the path for GradCAM
sys.path.insert(0, "first_attempt")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap, load_plain_clip
from gradcam_baseline import GradCAM
from clip_es_improved import compute_clip_es_heatmap_improved
from box_utils import extract_box

def resize_image(image, smaller_edge=224):
    """
    Resize image so the smaller edge is 'smaller_edge' pixels,
    maintaining aspect ratio.
    
    Args:
        image: PIL Image
        smaller_edge: target size for smaller dimension
        
    Returns:
        resized PIL Image
    """
    w, h = image.size
    if w < h:
        new_w = smaller_edge
        new_h = int(h * (smaller_edge / w))
    else:
        new_h = smaller_edge
        new_w = int(w * (smaller_edge / h))
    
    return image.resize((new_w, new_h), Image.LANCZOS)

def draw_box(ax, box, color='red', label=None):
    """Draw a bounding box on a matplotlib axis."""
    if box is None or len(box) != 4:
        return
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x0, y0 - 5, label, color=color, fontsize=8, fontweight='bold')

def run_boxing_benchmark():
    # 1. Load Models
    print("Loading B-cos model...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model_for_bcos, _ = load_clip_for_text() # B-cos uses this for text encoding
    
    print("Loading Plain CLIP model (for baselines)...")
    plain_clip_model, plain_preprocess, plain_device = load_plain_clip()
    
    # Initialize GradCAM
    gradcam = GradCAM(plain_clip_model.visual)
    
    # 2. Define Test Cases
    test_cases = [
        {
            "image": "cat_background.png",
            "positive": ["cat", "grass", "tree"],
            "negative": ["dog", "car"]
        },
        {
            "image": "sample_cat.jpg",
            "positive": ["cat"],
            "negative": ["dog"]
        },
        {
            "image": "car.png",
            "positive": ["car", "street"],
            "negative": ["cat"]
        },
        {
            "image": "objects.png",
            "positive": ["people", "book", "teapot"],
            "negative": ["computer"]
        }
    ]
    
    # 3. Run Tests
    output_dir = "test_results_boxing_resized"
    os.makedirs(output_dir, exist_ok=True)
    
    for case in test_cases:
        img_path = case["image"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} - not found")
            continue
            
        print(f"\nProcessing {img_path}...")
        image_orig = Image.open(img_path).convert('RGB')
        print(f"  Original size: {image_orig.size}")
        
        # Resize image
        image = resize_image(image_orig, smaller_edge=224)
        print(f"  Resized to: {image.size}")
        
        prompts = case["positive"] + case["negative"]
        
        # Create a large figure for all results
        # Rows: Prompts
        # Cols: Original, B-cos, GradCAM, Improved CLIP-ES
        n_rows = len(prompts)
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, prompt in enumerate(prompts):
            is_negative = prompt in case["negative"]
            prompt_label = f"{prompt} (NEG)" if is_negative else prompt
            
            # Column 0: Original (resized) Image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Image: {prompt_label}\n{image.size[0]}x{image.size[1]}")
            axes[i, 0].axis('off')
            
            # --- B-cos ---
            print(f"  B-cos: {prompt}")
            heatmap_bcos, score_bcos = get_bcos_heatmap(
                bcos_model, image, prompt, clip_model_for_bcos, bcos_device, return_raw=True
            )
            # Extract box
            box_bcos = extract_box(heatmap_bcos, method='otsu')
            
            axes[i, 1].imshow(image)
            axes[i, 1].imshow(heatmap_bcos, cmap='jet', alpha=0.5)
            draw_box(axes[i, 1], box_bcos, color='white', label='Otsu')
            axes[i, 1].set_title(f"B-cos\nScore: {score_bcos:.4f}")
            axes[i, 1].axis('off')
            
            # --- GradCAM ---
            print(f"  GradCAM: {prompt}")
            heatmap_gradcam, score_gradcam = gradcam.generate(
                image, prompt, plain_clip_model, plain_preprocess, plain_device, return_raw=True
            )
            # Extract box
            box_gradcam = extract_box(heatmap_gradcam, method='otsu')
            
            axes[i, 2].imshow(image)
            axes[i, 2].imshow(heatmap_gradcam, cmap='jet', alpha=0.5)
            draw_box(axes[i, 2], box_gradcam, color='white', label='Otsu')
            axes[i, 2].set_title(f"GradCAM\nScore: {score_gradcam:.4f}")
            axes[i, 2].axis('off')
            
            # --- Improved CLIP-ES ---
            print(f"  Improved CLIP-ES: {prompt}")
            heatmap_clipes, score_clipes = compute_clip_es_heatmap_improved(
                image, prompt, plain_clip_model, plain_preprocess, plain_device, return_raw=True
            )
            # Extract box
            box_clipes = extract_box(heatmap_clipes, method='otsu')
            
            axes[i, 3].imshow(image)
            axes[i, 3].imshow(heatmap_clipes, cmap='jet', alpha=0.5)
            draw_box(axes[i, 3], box_clipes, color='white', label='Otsu')
            axes[i, 3].set_title(f"Improved CLIP-ES\nScore: {score_clipes:.4f}")
            axes[i, 3].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"boxing_{os.path.basename(img_path)}")
        plt.savefig(save_path)
        print(f"Saved boxing result to {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_boxing_benchmark()
