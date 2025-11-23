"""
Fixed B-cos implementation using the original compute_attributions function.
This directly imports and uses the bcosification repo's text_localisation code.
"""

import sys
import os
sys.path.insert(0, "bcosification")

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from bcos.experiments.utils import Experiment
import bcos.data.transforms as custom_transforms
from interpretability.analyses.text_localisation import compute_attributions, tokenize_text, load_model, get_clip_model

def get_bcos_heatmap_fixed(image, text_prompt, exp_name=None, return_raw=False):
    """
    Generate B-cos heatmap using the ORIGINAL bcosification code.
    
    Args:
        image: PIL Image
        text_prompt: str
        exp_name: str, path to experiment
        return_raw: bool, if True returns RGB raw attributions, else RGBA explanation
        
    Returns:
        heatmap: numpy array
        score: float
    """
    if exp_name is None:
        exp_name = "experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification"
    
    # Load model and CLIP
    model = load_model(exp_name, use_attn_unpool=False)
    clip_model = get_clip_model()
    
    # Preprocess image (EXACT same as original)
    def _convert_image_to_rgb(img):
        return img.convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        custom_transforms.AddInverse(),
    ])
    
    test_img = transform(image)
    
    # Tokenize text (EXACT same as original)
    templates = ["a photo of a {}."]
    zeroshot_weight = tokenize_text(clip_model, templates, text_prompt)
    
    # Compute attributions (using original function!)
    grad_image, contribs, vrange, score = compute_attributions(
        model, test_img, zeroshot_weight, 
        smooth=0, alpha_percentile=99.5
    )
    
    if return_raw:
        # Return raw RGB attributions with BWR colormap
        # contribs is in range [-vrange, vrange]
        # Red = positive, Blue = negative
        return contribs, vrange, score
    else:
        # Return RGBA explanation
        return grad_image, score

# Test on car image
if __name__ == "__main__":
    print("Testing fixed B-cos implementation...")
    
    # Load car image
    img_path = "car.png"
    image_orig = Image.open(img_path).convert('RGB')
    print(f"Original size: {image_orig.size}")
    
    prompt = "car"
    
    # Get both outputs
    print(f"\nGenerating heatmaps for '{prompt}'...")
    
    # 1. RGB Explanation
    expl_rgb, score_rgb = get_bcos_heatmap_fixed(image_orig, prompt, return_raw=False)
    print(f"  RGB Explanation shape: {expl_rgb.shape}, Score: {score_rgb:.4f}")
    
    # 2. Raw Attributions 
    contribs, vrange, score_raw = get_bcos_heatmap_fixed(image_orig, prompt, return_raw=True)
    print(f"  Raw Attributions shape: {contribs.shape}, Score: {score_raw:.4f}")
    print(f"  Value range: [{contribs.min():.4f}, {contribs.max():.4f}], vrange: {vrange:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Resized image
    image_224 = image_orig.resize((224, 224), Image.LANCZOS)
    
    axes[0].imshow(image_224)
    axes[0].set_title("Original (224x224)")
    axes[0].axis('off')
    
    axes[1].imshow(expl_rgb)
    axes[1].set_title(f"B-cos Explanation\nScore: {score_rgb:.4f}")
    axes[1].axis('off')
    
    # Raw attributions with BWR colormap (RED=positive, BLUE=negative)!
    im = axes[2].imshow(contribs, cmap='bwr', vmin=-vrange, vmax=vrange)
    axes[2].set_title(f"Raw Attributions\nRed=Positive, Blue=Negative")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("test_car_bcos_FIXED.png", dpi=150, bbox_inches='tight')
    print("\nSaved to test_car_bcos_FIXED.png")
    print("\n✓ The car should appear RED in the Raw Attributions!")
