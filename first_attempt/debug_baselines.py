"""
Debug script to inspect tensor shapes for CLIP-ES and GradCAM.
"""

import sys
import torch
import numpy as np
from PIL import Image
import clip

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip

def debug_clip_es():
    print("\n--- Debugging CLIP-ES ---")
    clip_model, preprocess, device = load_plain_clip()
    
    image_path = "cat_background.png"
    image = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    print(f"Image tensor shape: {img_tensor.shape}")
    
    # Hook layer4 to see what we get
    target_layer = clip_model.visual.layer4
    feature_map = None
    
    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()
        print(f"Layer4 output shape: {output.shape}")
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        logits = clip_model.encode_image(img_tensor)
        
    handle.remove()
    
    # Check attnpool projections
    print(f"Attnpool v_proj weight shape: {clip_model.visual.attnpool.v_proj.weight.shape}")
    print(f"Attnpool c_proj weight shape: {clip_model.visual.attnpool.c_proj.weight.shape}")
    
    # Try to project feature map
    if feature_map is not None:
        x = feature_map # (B, 2048, 7, 7)
        # v_proj expects (L, N, E) or (N, E)? 
        # Linear layer applies to last dim.
        # We need to reshape x to apply linear layers.
        
        # ResNet feature map is (N, C, H, W) -> (1, 2048, 7, 7)
        # Permute to (N, H, W, C) -> (1, 7, 7, 2048)
        x = x.permute(0, 2, 3, 1)
        print(f"Permuted feature map shape: {x.shape}")
        
        # Apply v_proj
        v = clip_model.visual.attnpool.v_proj(x)
        print(f"After v_proj: {v.shape}")
        
        # Apply c_proj
        c = clip_model.visual.attnpool.c_proj(v)
        print(f"After c_proj: {c.shape}")
        
        # Now we have (1, 7, 7, 1024) -> matches text embed dim!
        
        # Text embed
        text_prompt = "cat"
        text_tokens = clip.tokenize([text_prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f"Text features shape: {text_features.shape}")
        
        # Compute similarity
        # c: (1, 7, 7, 1024)
        # text: (1, 1024)
        # Dot product along last dim
        sim = (c * text_features.view(1, 1, 1, 1024)).sum(dim=-1)
        print(f"Similarity map shape: {sim.shape}")

if __name__ == "__main__":
    debug_clip_es()
