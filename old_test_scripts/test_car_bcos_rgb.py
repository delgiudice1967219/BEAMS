"""
Test B-cos with proper RGB explanation visualization.
This matches the original bcosification workflow.
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap

# Load models
print("Loading models...")
bcos_model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Load and resize car image to 224x224 (squash)
img_path = "car.png"
image_orig = Image.open(img_path).convert('RGB')
print(f"Original size: {image_orig.size}")

image = image_orig.resize((224, 224), Image.LANCZOS)
print(f"Resized to: {image.size}")

prompt = "car"

# Get BOTH raw contributions and RGB explanation
print(f"\nGenerating B-cos visualizations for '{prompt}'...")

# 1. Get RGB explanation (return_raw=False)
print("  1. RGB Explanation...")
expl_rgb, score_rgb = get_bcos_heatmap(
    bcos_model, image, prompt, clip_model, device, return_raw=False
)
print(f"     Score: {score_rgb:.4f}")
print(f"     Shape: {expl_rgb.shape}")

# 2. Get raw contributions (return_raw=True) - BUT this is grayscale! We need to fix this
print("  2. Raw Contributions (currently grayscale - needs fix)...")
contrib_gray, score_gray = get_bcos_heatmap(
    bcos_model, image, prompt, clip_model, device, return_raw=True
)
print(f"     Score: {score_gray:.4f}")  
print(f"     Shape: {contrib_gray.shape}")
print(f"     Range: [{contrib_gray.min():.4f}, {contrib_gray.max():.4f}]")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Row 1: RGB Explanation
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image\n224x224")
axes[0, 0].axis('off')

axes[0, 1].imshow(expl_rgb)
axes[0, 1].set_title(f"B-cos RGB Explanation\nScore: {score_rgb:.4f}")
axes[0, 1].axis('off')

# Row 2: Raw Contributions (grayscale - should be RGB!)
axes[1, 0].imshow(contrib_gray, cmap='RdBu_r', vmin=0, vmax=1)
axes[1, 0].set_title("Raw Contributions (Grayscale)\nRed=High, Blue=Low")
axes[1, 0].axis('off')
axes[1, 0].text(0.5, -0.1, "NOTE: This should be RGB!\nRed=positive, Blue=negative",
                ha='center', transform=axes[1, 0].transAxes, fontsize=10, color='red')

axes[1, 1].imshow(image)
axes[1, 1].imshow(contrib_gray, cmap='jet', alpha=0.5)
axes[1, 1].set_title("Overlay (Jet colormap)")
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig("test_car_bcos_rgb_issue.png", dpi=150, bbox_inches='tight')
print("\nSaved to test_car_bcos_rgb_issue.png")

print("\n" + "="*60)
print("IDENTIFIED ISSUE:")
print("="*60)
print("Our bcos_localization.py returns:")
print("  - return_raw=False: RGB explanation (CORRECT)")
print("  - return_raw=True:  Grayscale contributions (WRONG!)")
print("")
print("Original bcosification shows:")
print("  - RGB explanation: Properly visualized explanations")
print("  - Raw attributions: RGB where Red=positive, Blue=negative")
print("")
print("We need to modify get_bcos_heatmap() to return RGB attributions!")
print("="*60)
