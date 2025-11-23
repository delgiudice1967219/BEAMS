"""
Test multiple prompts on sample_cat.jpg to find better localization.
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

# Load sample_cat image
img_path = "sample_cat.jpg"
image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}\n")

# Test different prompts
prompts = [
    "cat",
    "cat face",
    "cat head",
    "whiskers",
    "eyes",
    "feline",
    "orange cat",
    "tabby cat"
]

results = []

for prompt in prompts:
    print(f"Testing prompt: '{prompt}'...")
    
    # Get raw contributions
    contribs, vrange, score = get_bcos_heatmap(
        model, image, prompt, clip_model, device, return_raw=True
    )
    
    print(f"  Score: {score:.4f}, vrange: {vrange:.8f}")
    
    results.append({
        'prompt': prompt,
        'score': score,
        'vrange': vrange,
        'contribs': contribs
    })

# Sort by score (higher is better)
results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

print(f"\n{'='*60}")
print("Results sorted by score:")
print('='*60)
for i, r in enumerate(results_sorted, 1):
    print(f"{i}. '{r['prompt']:15s}' - Score: {r['score']:.4f}, vrange: {r['vrange']:.8f}")

# Visualize top 4 prompts
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(min(8, len(results_sorted))):
    r = results_sorted[i]
    
    # Show raw contributions with BWR colormap
    im = axes[i].imshow(r['contribs'], cmap='bwr', vmin=-r['vrange'], vmax=r['vrange'])
    axes[i].set_title(f"'{r['prompt']}'\nScore: {r['score']:.4f}\nvrange: {r['vrange']:.6f}")
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle(f"sample_cat.jpg - Testing Different Prompts\n(Red=Positive, Blue=Negative)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("sample_cat_prompt_testing.png", dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to sample_cat_prompt_testing.png")

# Show original image with best prompt
best = results_sorted[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title(f"Original\n{image.size[0]}x{image.size[1]}")
axes[0].axis('off')

# RGB explanation for best prompt
expl_rgb, _ = get_bcos_heatmap(model, image, best['prompt'], clip_model, device, return_raw=False)
axes[1].imshow(expl_rgb)
axes[1].set_title(f"B-cos RGB Explanation\nPrompt: '{best['prompt']}'")
axes[1].axis('off')

im = axes[2].imshow(best['contribs'], cmap='bwr', vmin=-best['vrange'], vmax=best['vrange'])
axes[2].set_title(f"Raw Contributions\nScore: {best['score']:.4f}")
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("sample_cat_best_prompt.png", dpi=150, bbox_inches='tight')
print(f"Saved best result to sample_cat_best_prompt.png")

print(f"\n✓ Best prompt: '{best['prompt']}' with score {best['score']:.4f}")
