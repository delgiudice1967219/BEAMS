"""
Quick test to check raw B-cos heatmap on car image without resizing.
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text

# Load models
print("Loading models...")
bcos_model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Load car image
img_path = "car.png"
image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}")

# Get B-cos heatmap for "car"
prompt = "car"

# Import the heatmap function
from bcos_localization import get_bcos_heatmap

print(f"Generating heatmap for '{prompt}'...")
heatmap, score = get_bcos_heatmap(
    bcos_model, image, prompt, clip_model, device, return_raw=True
)

print(f"Heatmap shape: {heatmap.shape}")
print(f"Score: {score:.4f}")
print(f"Heatmap min/max: {heatmap.min():.4f} / {heatmap.max():.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(heatmap, cmap='jet')
axes[1].set_title(f"Raw Heatmap\n(Score: {score:.4f})")
axes[1].axis('off')

axes[2].imshow(image)
axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("test_car_raw_bcos.png")
print("Saved to test_car_raw_bcos.png")
plt.show()
