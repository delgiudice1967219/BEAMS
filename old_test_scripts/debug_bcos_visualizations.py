"""
Visualize B-cos RGB explanation AND raw contributions (BWR colormap) 
for sample_cat.jpg and car.png to debug issues.
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

# Test cases
test_cases = [
    {"image": "sample_cat.jpg", "prompt": "cat"},
    {"image": "car.png", "prompt": "car"}
]

for test in test_cases:
    img_path = test["image"]
    prompt = test["prompt"]
    
    print(f"\n{'='*60}")
    print(f"Testing: {img_path} with prompt '{prompt}'")
    print('='*60)
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    print(f"Original size: {image.size}")
    
    # Get RGB explanation
    print("  Getting RGB explanation...")
    expl_rgb, score_rgb = get_bcos_heatmap(
        model, image, prompt, clip_model, device, return_raw=False
    )
    print(f"    Score: {score_rgb:.4f}")
    print(f"    Shape: {expl_rgb.shape}")
    
    # Get raw contributions
    print("  Getting raw contributions...")
    contribs, vrange, score_raw = get_bcos_heatmap(
        model, image, prompt, clip_model, device, return_raw=True
    )
    print(f"    Score: {score_raw:.4f}")
    print(f"    Shape: {contribs.shape}")
    print(f"    Value range: [{contribs.min():.8f}, {contribs.max():.8f}]")
    print(f"    vrange: {vrange:.8f}")
    
    # Calculate processed size (aspect ratio preserved)
    w, h = image.size
    if w < h:
        new_w, new_h = 224, int(h * (224 / w))
    else:
        new_h, new_w = 224, int(w * (224 / h))
    
    print(f"    Processed size: {new_w}x{new_h}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image (actual aspect ratio)
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image\n{image.size[0]}x{image.size[1]}")
    axes[0].axis('off')
    
    # 2. B-cos RGB Explanation (RGBA)
    axes[1].imshow(expl_rgb)
    axes[1].set_title(f"B-cos RGB Explanation\nScore: {score_rgb:.4f}\n(Processed: {new_w}x{new_h})")
    axes[1].axis('off')
    
    # 3. Raw Contributions (BWR colormap: Red=positive, Blue=negative)
    im = axes[2].imshow(contribs, cmap='bwr', vmin=-vrange, vmax=vrange)
    axes[2].set_title(f"Raw Contributions (BWR)\nRed=Positive, Blue=Negative\nvrange: {vrange:.6f}")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"{img_path} - Prompt: '{prompt}'", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_name = f"bcos_debug_{img_path.replace('.', '_')}.png"
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to {save_name}")
    plt.close()

print("\n" + "="*60)
print("DONE! Check the generated images.")
print("="*60)
print("\nWhat to look for:")
print("  - RGB Explanation: Should show localized regions")
print("  - Raw Contributions: RED areas = positive (relevant to prompt)")
print("                       BLUE areas = negative (irrelevant)")
print("  - For 'cat': Cat should be RED")
print("  - For 'car': Car should be RED")
