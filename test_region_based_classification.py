"""
Test: Region-based classification using B-cos heatmaps.
Crop to object region before classification to improve discrimination.
"""

import sys
sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap
from box_utils import extract_box
from PIL import Image
import numpy as np

# Load models
print("Loading models...")
model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Test on one image with multiple prompts
img_path = "test_images/glasses-spoon-plant-apple-wallet-cup.png"
image = Image.open(img_path).convert('RGB')
print(f"\nImage: {img_path}")
print(f"Size: {image.size}")

prompts = ["reading glasses", "spoon", "plant", "apple", "phone", "dog"]
positive = ["reading glasses", "spoon", "plant", "apple"]

print("\n" + "="*70)
print("METHOD 1: Global Classification (current)")
print("="*70)

global_scores = {}
for prompt in prompts:
    contribs, vrange, score = get_bcos_heatmap(
        model, image, prompt, clip_model, device, return_raw=True
    )
    global_scores[prompt] = score
    is_pos = "✓" if prompt in positive else "✗"
    print(f"{is_pos} {prompt:20s}: {score:.4f}")

print("\n" + "="*70)
print("METHOD 2: Region-based Classification (proposed)")
print("="*70)

region_scores = {}
for prompt in prompts:
    # Get heatmap
    contribs, vrange, score_global = get_bcos_heatmap(
        model, image, prompt, clip_model, device, return_raw=True
    )
    
    # Convert to positive-only for box extraction
    heatmap = np.maximum(contribs, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Extract bounding box
    box = extract_box(heatmap, method='otsu')
    
    # Scale box to original image size
    h_ratio = image.size[1] / heatmap.shape[0]
    w_ratio = image.size[0] / heatmap.shape[1]
    box_scaled = [
        int(box[0] * w_ratio),
        int(box[1] * h_ratio),
        int(box[2] * w_ratio),
        int(box[3] * h_ratio)
    ]
    
    # Crop to box region (with some padding)
    pad = 20
    x0 = max(0, box_scaled[0] - pad)
    y0 = max(0, box_scaled[1] - pad)
    x1 = min(image.size[0], box_scaled[2] + pad)
    y1 = min(image.size[1], box_scaled[3] + pad)
    
    # Skip if box is too small
    if (x1 - x0) < 50 or (y1 - y0) < 50:
        print(f"  {prompt:20s}: Box too small, using global score")
        region_scores[prompt] = score_global
        continue
    
    cropped = image.crop((x0, y0, x1, y1))
    
    # Re-classify cropped region
    contribs_crop, vrange_crop, score_crop = get_bcos_heatmap(
        model, cropped, prompt, clip_model, device, return_raw=True
    )
    
    region_scores[prompt] = score_crop
    is_pos = "✓" if prompt in positive else "✗"
    improvement = score_crop - score_global
    print(f"{is_pos} {prompt:20s}: {score_crop:.4f} (global: {score_global:.4f}, Δ={improvement:+.4f})")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print("\nGlobal method - Score ranges:")
pos_global = [global_scores[p] for p in positive]
neg_global = [global_scores[p] for p in prompts if p not in positive]
print(f"  Positive: [{min(pos_global):.4f}, {max(pos_global):.4f}]")
print(f"  Negative: [{min(neg_global):.4f}, {max(neg_global):.4f}]")
print(f"  Separation: {min(pos_global) - max(neg_global):.4f}")

print("\nRegion-based method - Score ranges:")
pos_region = [region_scores[p] for p in positive]
neg_region = [region_scores[p] for p in prompts if p not in positive]
print(f"  Positive: [{min(pos_region):.4f}, {max(pos_region):.4f}]")
print(f"  Negative: [{min(neg_region):.4f}, {max(neg_region):.4f}]")
print(f"  Separation: {min(pos_region) - max(neg_region):.4f}")

if (min(pos_region) - max(neg_region)) > (min(pos_global) - max(neg_global)):
    print("\n✓ Region-based method provides BETTER separation!")
else:
    print("\n✗ No improvement. Object might fill entire region already.")
