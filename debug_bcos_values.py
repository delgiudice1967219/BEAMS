"""
Debug script to examine raw B-cos heatmap values.
Tests a single image with a single class to verify the mechanism is working.
"""

import sys
sys.path.insert(0, "bcosification")

import numpy as np
from PIL import Image
from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap

def main():
    print("Loading models...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model, _ = load_clip_for_text()
    
    print("\nLoading test image: bread-knife-pans-towel.png")
    image = Image.open("test_images/bread-knife-pans-towel.png").convert('RGB')
    print(f"Image size: {image.size}")
    
    # Test with a single class
    test_class = "bread"
    print(f"\nTesting class: '{test_class}'")
    
    # Get raw B-cos heatmap
    print("Computing B-cos heatmap...")
    contribs, vrange, score = get_bcos_heatmap(
        bcos_model, image, test_class, clip_model, bcos_device, return_raw=True
    )
    
    print(f"\nRaw contributions stats:")
    print(f"  Shape: {contribs.shape}")
    print(f"  Min: {contribs.min():.6f}")
    print(f"  Max: {contribs.max():.6f}")
    print(f"  Mean: {contribs.mean():.6f}")
    print(f"  Std: {contribs.std():.6f}")
    print(f"  Score: {score:.6f}")
    
    # Take positive contributions only
    positive_contribs = np.maximum(contribs, 0)
    print(f"\nPositive contributions stats:")
    print(f"  Min: {positive_contribs.min():.6f}")
    print(f"  Max: {positive_contribs.max():.6f}")
    print(f"  Mean: {positive_contribs.mean():.6f}")
    print(f"  Std: {positive_contribs.std():.6f}")
    print(f"  Non-zero pixels: {np.count_nonzero(positive_contribs)} / {positive_contribs.size}")
    
    # Normalized version
    if positive_contribs.max() > 0:
        normalized = positive_contribs / positive_contribs.max()
        print(f"\nNormalized stats:")
        print(f"  Min: {normalized.min():.6f}")
        print(f"  Max: {normalized.max():.6f}")
        print(f"  Mean: {normalized.mean():.6f}")
    
    print("\n" + "="*60)
    print("Now testing with synonym averaging...")
    print("="*60)
    
    # Test synonym fusion
    synonyms = ["bread", "loaf", "slice of bread", "bun", "toast"]
    all_heatmaps = []
    
    for synonym in synonyms:
        print(f"\nSynonym: '{synonym}'")
        contribs, _, score = get_bcos_heatmap(
            bcos_model, image, synonym, clip_model, bcos_device, return_raw=True
        )
        positive = np.maximum(contribs, 0)
        all_heatmaps.append(positive)
        print(f"  Max: {positive.max():.6f}, Score: {score:.6f}")
    
    # Average across synonyms
    averaged = np.mean(all_heatmaps, axis=0)
    print(f"\nAveraged heatmap stats:")
    print(f"  Min: {averaged.min():.6f}")
    print(f"  Max: {averaged.max():.6f}")
    print(f"  Mean: {averaged.mean():.6f}")
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main()
