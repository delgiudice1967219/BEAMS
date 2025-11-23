
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import clip

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap
from box_utils import extract_box

def draw_box(ax, box, color='white', label=None):
    """Draw a bounding box on a matplotlib axis."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    rect = plt.Rectangle((x0, y0), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x0, y0, label, color=color, fontsize=8, verticalalignment='bottom')

def run_benchmark():
    # Define test cases with both positive and negative classes
    test_cases = [
        {
            "image": "test_images/bread-knife-pans-towel.png",
            "name": "bread-knife-pans-towel",
            "positive": ["bread", "knife", "pans", "towel"],
            "negative": ["cat", "car", "person"]
        },
        {
            "image": "test_images/car.png",
            "name": "car",
            "positive": ["car", "building"],
            "negative": ["cat", "dog", "person"]
        },
        {
            "image": "test_images/cat_background.png",
            "name": "cat_background",
            "positive": ["cat", "dog", "rabbit", "bear"],
            "negative": ["car", "person", "bicycle", "bird"]
        },
        {
            "image": "test_images/glasses-spoon-plant-apple-wallet-cup.png",
            "name": "glasses-spoon-plant-apple-wallet-cup",
            "positive": ["reading glasses", "spoon", "plant", "apple", "wallet", "cup"],
            "negative": ["cat", "car", "person"]
        }
    ]
    
    print(f"Testing {len(test_cases)} images with positive and negative classes:")
    for tc in test_cases:
        print(f"  {tc['name']}:")
        print(f"    Positive: {tc['positive']}")
        print(f"    Negative: {tc['negative']}")
    
    # Load models
    print("\nLoading B-cos model...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model_for_bcos, _ = load_clip_for_text()
    
    # Prepare grid: rows = images, columns = all classes (positive + negative)
    num_images = len(test_cases)
    max_classes = max(len(tc['positive']) + len(tc['negative']) for tc in test_cases)
    
    fig, axes = plt.subplots(num_images, max_classes, figsize=(4 * max_classes, 5 * num_images))
    
    # Handle single row/column cases
    if num_images == 1 and max_classes == 1:
        axes = np.array([[axes]])
    elif num_images == 1:
        axes = axes.reshape(1, -1)
    elif max_classes == 1:
        axes = axes.reshape(-1, 1)
    
    for img_idx, case in enumerate(test_cases):
        img_path = case["image"]
        img_name = case["name"]
        all_prompts = case["positive"] + case["negative"]
        positive_set = set(case["positive"])
        
        print(f"\nProcessing {img_name}...")
        image = Image.open(img_path).convert('RGB')
        print(f"  Size: {image.size}")
        
        for prompt_idx, prompt in enumerate(all_prompts):
            is_positive = prompt in positive_set
            label_type = "POS" if is_positive else "NEG"
            print(f"  Testing [{label_type}]: {prompt}")
            
            # Get B-cos heatmap
            contribs, vrange, score_global = get_bcos_heatmap(
                bcos_model, image, prompt, clip_model_for_bcos, bcos_device, return_raw=True
            )
            
            # Extract positive contributions for boxing
            heatmap_bcos = np.maximum(contribs, 0)
            if heatmap_bcos.max() > 0:
                heatmap_bcos = heatmap_bcos / heatmap_bcos.max()
            
            # Resize heatmap to original image size
            from PIL import Image as PILImage
            heatmap_resized = PILImage.fromarray((heatmap_bcos * 255).astype(np.uint8)).resize(
                image.size, PILImage.BILINEAR
            )
            heatmap_resized = np.array(heatmap_resized) / 255.0
            
            # Extract bounding box
            box = extract_box(heatmap_bcos, method='otsu')
            h_ratio = image.size[1] / heatmap_bcos.shape[0]
            w_ratio = image.size[0] / heatmap_bcos.shape[1]
            box_scaled = [
                int(box[0] * w_ratio),
                int(box[1] * h_ratio),
                int(box[2] * w_ratio),
                int(box[3] * h_ratio)
            ]
            
            # Compute region score
            pad = 10
            x0 = max(0, box_scaled[0] - pad)
            y0 = max(0, box_scaled[1] - pad)
            x1 = min(image.size[0], box_scaled[2] + pad)
            y1 = min(image.size[1], box_scaled[3] + pad)
            
            score_region = 0.0
            if x1 > x0 and y1 > y0:
                img_crop = image.crop((x0, y0, x1, y1))
                _, _, score_region = get_bcos_heatmap(
                    bcos_model, img_crop, prompt, clip_model_for_bcos, bcos_device, return_raw=True
                )
            
            print(f"    Global: {score_global:.4f}, Region: {score_region:.4f}")
            
            # Plot
            ax = axes[img_idx, prompt_idx]
            ax.imshow(image)
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
            draw_box(ax, box_scaled, color='white', label='Box')
            
            # Title with scores and positive/negative indicator
            # Use color coding: green for positive, red for negative
            title_color = 'green' if is_positive else 'red'
            title = f"{prompt} [{label_type}]\nG:{score_global:.3f} R:{score_region:.3f}"
            ax.set_title(title, fontsize=9, color=title_color, weight='bold')
            ax.axis('off')
            
            # Add row label (image name) on first column
            if prompt_idx == 0:
                ax.text(-0.1, 0.5, img_name, transform=ax.transAxes,
                       fontsize=11, rotation=90, va='center', ha='right', weight='bold')
        
        # Hide unused subplots in this row
        for empty_idx in range(len(all_prompts), max_classes):
            axes[img_idx, empty_idx].axis('off')
    
    plt.tight_layout()
    os.makedirs("test_results_bcos_only", exist_ok=True)
    save_path = "test_results_bcos_only/final_benchmark_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved benchmark grid to {save_path}")

if __name__ == "__main__":
    run_benchmark()
