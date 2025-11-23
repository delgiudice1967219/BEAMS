
import torch
import clip
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(".")
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip
from utils.preprocess import aspect_preserving_preprocess
from utils.clip_patch import patch_clip_attnpool

def debug_heatmap_shapes():
    print("Loading models...")
    clip_model, _, device = load_plain_clip()
    patch_clip_attnpool(clip_model)
    
    # Use a very rectangular image
    # Create a dummy image 400x200 (2:1 aspect ratio)
    img = Image.new('RGB', (400, 200), color='red')
    
    print(f"\nInput Image Size: {img.size} (WxH)")
    
    # Preprocess
    img_tensor = aspect_preserving_preprocess(img, target_small_edge=224).unsqueeze(0).to(device)
    print(f"Tensor Shape: {img_tensor.shape}")
    
    # Expected shape calculation
    # Smaller edge (200) -> 224
    # Larger edge (400) -> 400 * (224/200) = 448
    # Expected tensor: (1, 3, 224, 448)
    
    # Hook to capture feature map size
    feature_map_shape = None
    def hook_fn(module, input, output):
        nonlocal feature_map_shape
        feature_map_shape = output.shape
        
    handle = clip_model.visual.layer4.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = clip_model.encode_image(img_tensor)
        
    handle.remove()
    
    print(f"Feature Map Shape (Layer 4): {feature_map_shape}")
    
    # Expected feature map:
    # 224 / 32 = 7
    # 448 / 32 = 14
    # Expected shape: (1, 2048, 7, 14)
    
    if feature_map_shape[-2:] == (7, 7):
        print("\n[FAIL] Feature map is 7x7! The model is still processing a square crop.")
    elif feature_map_shape[-2:] == (7, 14):
        print("\n[SUCCESS] Feature map is 7x14! The model is processing the full rectangular image.")
    else:
        print(f"\n[?] Feature map is {feature_map_shape[-2:]}. Check if this matches aspect ratio.")

if __name__ == "__main__":
    debug_heatmap_shapes()
