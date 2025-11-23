"""
Test updated bcos_localization.py that uses original compute_attributions.
"""

import sys
sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load models
print("Loading models...")
model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Load car image
img_path = "car.png"
image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}")

prompt = "car"
print(f"\nTesting '{prompt}'...")

# Get raw attributions
contribs, vrange, score = get_bcos_heatmap(
    model, image, prompt, clip_model, device, return_raw=True
)

print(f"Score: {score:.4f}")
print(f"Contribs shape: {contribs.shape}")
print(f"Value range: [{contribs.min():.4f}, {contribs.max():.4f}]")
print(f"vrange: {vrange:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

image_224 = image.resize((224, 224), Image.LANCZOS)

axes[0].imshow(image_224)
axes[0].set_title("Original (224x224)")
axes[0].axis('off')

# Raw attributions with BWR colormap
im = axes[1].imshow(contribs, cmap='bwr', vmin=-vrange, vmax=vrange)
axes[1].set_title(f"Raw Attributions (BWR)\nScore: {score:.4f}")
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], label="Red=Positive, Blue=Negative")

plt.tight_layout()
plt.savefig("test_updated_bcos_localization.png", dpi=150)
print("\nSaved to test_updated_bcos_localization.png")
print("\n✓ Car should appear RED if fixed correctly!")
