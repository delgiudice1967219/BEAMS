"""
Test with raw numpy save to verify data is really there.
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

prompt = "car"
print(f"\nTesting '{prompt}'...")

# Get raw attributions
contribs, vrange, score = get_bcos_heatmap(
    model, image, prompt, clip_model, device, return_raw=True
)

print(f"Score: {score:.4f}")
print(f"Contribs shape: {contribs.shape}")
print(f"Contribs dtype: {contribs.dtype}")
print(f"Value range: [{contribs.min():.8f}, {contribs.max():.8f}]")
print(f"Abs sum: {np.abs(contribs).sum():.8f}")
print(f"vrange: {vrange:.8f}")

# Save raw numpy
np.save("car_contribs_raw.npy", contribs)
print("\nSaved raw contributions to car_contribs_raw.npy")

if vrange > 0:
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    image_224 = image.resize((224, 224), Image.LANCZOS)

    axes[0].imshow(image_224)
    axes[0].set_title("Original (224x224)")
    axes[0].axis('off')

    # Raw attributions with BWR colormap
    im = axes[1].imshow(contribs, cmap='bwr', vmin=-vrange, vmax=vrange)
    axes[1].set_title(f"Raw Attributions (BWR)\nScore: {score:.4f}\nvrange: {vrange:.6f}")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label="Red=Positive, Blue=Negative")

    plt.tight_layout()
    plt.savefig("test_car_contributions_VERIFIED.png", dpi=150)
    print("Saved to test_car_contributions_VERIFIED.png")
    print("\n✓ Car SHOULD appear in color if data is valid!")
else:
    print("\n✗ vrange is zero - contributions are all zero!")
