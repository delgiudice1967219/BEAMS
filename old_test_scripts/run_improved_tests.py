"""
Benchmark script comparing B-cos, GradCAM, and Improved CLIP-ES.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")
# Ensure first_attempt is on the path for GradCAM
sys.path.insert(0, "first_attempt")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap, load_plain_clip
from gradcam_baseline import GradCAM
from clip_es_improved import compute_clip_es_heatmap_improved

def run_improved_tests():
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
            "negative": ["dog", "car", "airplane"]
        },
        {
            "image": "car.png",
            "positive": ["car", "street", "building"],
            "negative": ["cat", "boat", "pizza"]
        },
        {
            "image": "objects.png",
            "positive": ["people", "book", "teapot"],
            "negative": ["elephant", "computer", "beach"]
        }
    ]
    
    # 3. Run Tests
    output_dir = "test_results_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    for case in test_cases:
        img_path = case["image"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} - not found")
            continue
            
        print(f"\nProcessing {img_path}...")
        image = Image.open(img_path).convert('RGB')
        
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
            
            # Column 0: Original Image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Image: {prompt_label}")
            axes[i, 0].axis('off')
            
            # Column 1: B-cos
            print(f"  B-cos: {prompt}")
            heatmap_bcos, score_bcos = get_bcos_heatmap(
                bcos_model, image, prompt, clip_model_for_bcos, bcos_device, return_raw=True
            )
            axes[i, 1].imshow(image)
            axes[i, 1].imshow(heatmap_bcos, cmap='jet', alpha=0.5)
            axes[i, 1].set_title(f"B-cos\nScore: {score_bcos:.4f}")
            axes[i, 1].axis('off')
            
            # Column 2: GradCAM
            print(f"  GradCAM: {prompt}")
            heatmap_gradcam, score_gradcam = gradcam.generate(
                image, prompt, plain_clip_model, plain_preprocess, plain_device, return_raw=True
            )
            axes[i, 2].imshow(image)
            axes[i, 2].imshow(heatmap_gradcam, cmap='jet', alpha=0.5)
            axes[i, 2].set_title(f"GradCAM\nScore: {score_gradcam:.4f}")
            axes[i, 2].axis('off')
            
            # Column 3: Improved CLIP-ES
            print(f"  Improved CLIP-ES: {prompt}")
            heatmap_clipes, score_clipes = compute_clip_es_heatmap_improved(
                image, prompt, plain_clip_model, plain_preprocess, plain_device, return_raw=True
            )
            axes[i, 3].imshow(image)
            axes[i, 3].imshow(heatmap_clipes, cmap='jet', alpha=0.5)
            axes[i, 3].set_title(f"Improved CLIP-ES\nScore: {score_clipes:.4f}")
            axes[i, 3].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_{os.path.basename(img_path)}")
        plt.savefig(save_path)
        print(f"Saved comparison to {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_improved_tests()
