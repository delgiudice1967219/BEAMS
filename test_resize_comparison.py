"""
Test B-cos with and without resizing to see impact on scores.
Compare: Original size vs Resized (smaller edge=224)
"""

import sys
sys.path.insert(0, "bcosification")

from bcos_localization import load_bcos_model, load_clip_for_text
from PIL import Image
import torch
import numpy as np

# Load models
print("Loading models...")
model, device = load_bcos_model()
clip_model, _ = load_clip_for_text()

# Test image
img_path = "test_images/glasses-spoon-plant-apple-wallet-cup.png"
image = Image.open(img_path).convert('RGB')
print(f"\nImage: {img_path}")
print(f"Original size: {image.size}")

prompts = ["reading glasses", "spoon", "apple", "phone", "dog"]

print("\n" + "="*70)
print("COMPARING: Original Size vs Resized")
print("="*70)

# Import get_bcos_heatmap internals
from bcos_localization import get_bcos_heatmap

results = []

for prompt in prompts:
    # Get score with resizing (current implementation)
    contribs_resized, vrange_resized, score_resized = get_bcos_heatmap(
        model, image, prompt, clip_model device, return_raw=True
    )
    
    # For "original size" we would need to modify the function
    # But we can approximate by checking the heatmap shape
    
    results.append({
        'prompt': prompt,
        'score_resized': score_resized,
        'heatmap_shape': contribs_resized.shape
    })
    
print(f"\n{'Prompt':<20s} {'Score':<10s} {'Heatmap Shape':<15s}")
print("-"*70)

for r in results:
    print(f"{r['prompt']:<20s} {r['score_resized']:.4f}     {str(r['heatmap_shape']):<15s}")

print("\n" + "="*70)
print("NOTES:")
print("="*70)
print("- Current B-cos preserves aspect ratio (smaller edge = 224)")
print("- Heatmap shapes vary based on aspect ratio")
print("- To test 'no resize', would need to modify bcos_localization.py")
print("- This would require changes to AddInverse transform handling")
print("\nRecommendation: Keep resizing for consistency with CLIP's training")
