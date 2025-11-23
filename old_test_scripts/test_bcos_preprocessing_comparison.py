"""
COMPARISON TEST: B-cos implementation vs Original

This script helps identify differences between our bcos_localization.py
and the original bcosification repo implementation.

KEY DIFFERENCES IDENTIFIED:
================================

1. **Image Preprocessing:**
   - Our code: Resize(224) + CenterCrop(224) + ToTensor + AddInverse
   - Original: May use different resize strategy or no resize at all
   
2. **Model Forward Pass:**
   - Our code: Sets model.attn_unpool = True before forward
   - Original: Same approach via compute_attributions()
   
3. **Heatmap Generation:**
   - Our code: Uses model.backward_weights (explanations)
   - Original: Uses module.W (raw contributions)
   
4. **Normalization:**
   - Our code: Clamps negative values, then normalizes to [0, 1]
   - Original: Uses linear_alpha with percentile normalization

5. **Output:**
   - Our code: Returns numpy array normalized [0, 1]
   - Original: Returns RGBA explanation with custom alpha

POTENTIAL ISSUES:
=================
- CenterCrop(224) might cut out important image regions!
- For car.png (2816x1536), CenterCrop loses ~2400px horizontally
- This could explain why the car isn't detected

TEST PLAN:
==========
1. Resize car.png to 224x224 (SQUASH, not crop)
2. Run our B-cos implementation
3. Run original textloc script if available
4. Compare heatmaps visually
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap

# Test with car image
print("="*60)
print("B-COS CAR DETECTION COMPARISON TEST")
print("="*60)

# Load models
print("\n1. Loading models...")
bcos_model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Load car image
img_path = "car.png"
image_orig = Image.open(img_path).convert('RGB')
print(f"\n2. Original image size: {image_orig.size}")

# Test 1: Resize to 224x224 (SQUASH - what our code effectively does after crop)
image_squash = image_orig.resize((224, 224), Image.LANCZOS)
print(f"   Squashed to: {image_squash.size}")

# Test 2: Maintain aspect ratio (smaller edge = 224)
w, h = image_orig.size
if w < h:
    new_w = 224
    new_h = int(h * (224 / w))
else:
    new_h = 224
    new_w = int(w * (224 / h))
image_aspect = image_orig.resize((new_w, new_h), Image.LANCZOS)
print(f"   Aspect-ratio resize: {image_aspect.size}")

# Test both
prompt = "car"

print(f"\n3. Generating heatmaps for prompt '{prompt}'...")

print("   a) Squashed 224x224:")
heatmap_squash, score_squash = get_bcos_heatmap(
    bcos_model, image_squash, prompt, clip_model, device, return_raw=True
)
print(f"      Score: {score_squash:.4f}")
print(f"      Heatmap range: [{heatmap_squash.min():.4f}, {heatmap_squash.max():.4f}]")

print("   b) Aspect-ratio preserved:")
heatmap_aspect, score_aspect = get_bcos_heatmap(
    bcos_model, image_aspect, prompt, clip_model, device, return_raw=True
)
print(f"      Score: {score_aspect:.4f}")
print(f"      Heatmap range: [{heatmap_aspect.min():.4f}, {heatmap_aspect.max():.4f}]")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Squashed
axes[0, 0].imshow(image_squash)
axes[0, 0].set_title(f"Squashed\n{image_squash.size}")
axes[0, 0].axis('off')

axes[0, 1].imshow(heatmap_squash, cmap='jet')
axes[0, 1].set_title(f"Heatmap\nScore: {score_squash:.4f}")
axes[0, 1].axis('off')

axes[0, 2].imshow(image_squash)
axes[0, 2].imshow(heatmap_squash, cmap='jet', alpha=0.5)
axes[0, 2].set_title("Overlay")
axes[0, 2].axis('off')

# Row 2: Aspect ratio
axes[1, 0].imshow(image_aspect)
axes[1, 0].set_title(f"Aspect Ratio\n{image_aspect.size}")
axes[1, 0].axis('off')

axes[1, 1].imshow(heatmap_aspect, cmap='jet')
axes[1, 1].set_title(f"Heatmap\nScore: {score_aspect:.4f}")
axes[1, 1].axis('off')

axes[1, 2].imshow(image_aspect)
axes[1, 2].imshow(heatmap_aspect, cmap='jet', alpha=0.5)
axes[1, 2].set_title("Overlay")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig("comparison_car_bcos_preprocessing.png")
print(f"\n4. Saved comparison to comparison_car_bcos_preprocessing.png")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("- If squashed version has better localization: CenterCrop was the issue")
print("- If aspect-ratio version is better: Original code likely preserves aspect")
print("- Check if heatmap is 'completely red' in either case")
print("="*60)
