"""
Quick test script for advanced detection pipeline.
Tests on a single image to verify the implementation works.
"""

import sys
sys.path.insert(0, "bcosification")

from PIL import Image
from bcos_localization import load_bcos_model, load_clip_for_text
from advanced_utils import AdvancedDetector

def main():
    print("Loading models...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model, _ = load_clip_for_text()
    
    print("\nInitializing detector...")
    detector = AdvancedDetector(bcos_model, clip_model, bcos_device)
    
    print("\nTesting on bread-knife-pans-towel.png...")
    image = Image.open("test_images/bread-knife-pans-towel.png").convert('RGB')
    
    # Test synonym fusion for a single class
    print("\nTesting synonym fusion for 'bread'...")
    heatmap = detector.compute_class_heatmap_with_fusion(image, "bread")
    print(f"Fused heatmap shape: {heatmap.shape}, max: {heatmap.max():.3f}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
