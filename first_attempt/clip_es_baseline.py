"""
CLIP-ES baseline implementation using CLIP visual and text encoders.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import clip

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip


def compute_clip_es_heatmap(image, text_prompt, clip_model, preprocess, device, return_raw=False):
    """Compute CLIP‑ES similarity heatmap.

    Args:
        image (PIL.Image): Input image.
        text_prompt (str): Text query.
        clip_model: CLIP model with visual and text encoders.
        preprocess: CLIP preprocessing function.
        device (torch.device): Computation device.
        return_raw (bool): If True, returns the raw similarity map.
    """
    # Preprocess image using CLIP's visual preprocessing
    img_tensor = preprocess(image).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)

    # Encode image to get spatial feature map (before pooling)
    # CLIP visual encoder returns a single vector after pooling; we need the intermediate feature map.
    # We'll hook the visual transformer to capture the last feature map.
    # For simplicity, we use the visual backbone's output before the final projection.
    # The visual backbone is a ResNet‑like model; we can access its penultimate layer.
    # Here we assume the visual backbone has an attribute `visual` with a `stem` and `visual` modules.
    # We'll register a forward hook on the last conv layer to get the feature map.

    # Identify the last conv layer (layer4 for ResNet50)
    target_layer = clip_model.visual.layer4
    feature_map = None

    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()

    handle = target_layer.register_forward_hook(hook_fn)
    # Forward pass to get logits (also populates feature_map)
    with torch.no_grad():
        logits = clip_model.encode_image(img_tensor)
    handle.remove()

    # Encode text prompt
    text_tokens = clip_model.encode_text(clip.tokenize([text_prompt]).to(device))
    text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)

    # feature_map shape: (1, C, H, W) -> (1, 2048, 7, 7)
    
    # Project features to embedding space using attnpool weights
    # 1. Permute to (N, H, W, C)
    x = feature_map.permute(0, 2, 3, 1) # (1, 7, 7, 2048)
    
    # 2. Apply v_proj and c_proj from attnpool
    v = clip_model.visual.attnpool.v_proj(x)
    c = clip_model.visual.attnpool.c_proj(v) # (1, 7, 7, 1024)
    
    # 3. Normalize features
    c = c / c.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    # c: (1, 7, 7, 1024)
    # text_tokens: (1, 1024)
    # Dot product along last dim
    similarity_map = (c * text_tokens.view(1, 1, 1, -1)).sum(dim=-1) # (1, 7, 7)
    
    # Remove batch dim
    similarity_map = similarity_map.squeeze(0) # (7, 7)
    
    # Normalize to [0, 1]
    sim_min, sim_max = similarity_map.min(), similarity_map.max()
    similarity_map = (similarity_map - sim_min) / (sim_max - sim_min + 1e-8)

    # Resize to original image size
    original_size = image.size  # (W, H)
    sim_resized = Image.fromarray((similarity_map.detach().cpu().numpy() * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
    sim_resized = np.array(sim_resized) / 255.0

    # Compute a scalar similarity score (average similarity)
    score = similarity_map.mean().item()

    if return_raw:
        return sim_resized, score
    else:
        # Simple visualization helper
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.imshow(sim_resized, cmap='jet', alpha=0.5)
        plt.title(f"CLIP‑ES – {text_prompt} (score={score:.4f})")
        plt.axis('off')
        plt.show()
        return sim_resized, score


def run_demo():
    """Demo script to generate CLIP‑ES heatmaps for a few test images."""
    # Load models
    print("Loading CLIP model for CLIP‑ES demo...")
    clip_model, preprocess, device = load_plain_clip()

    test_cases = [
        {"image": "cat_background.png", "prompts": ["cat", "grass", "tree"]},
        {"image": "car.png", "prompts": ["car", "street", "building"]},
        {"image": "objects.png", "prompts": ["people", "book", "teapot"]},
    ]

    for case in test_cases:
        img_path = case["image"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} – not found")
            continue
        img = Image.open(img_path).convert('RGB')
        for prompt in case["prompts"]:
            print(f"Generating CLIP‑ES for {img_path} – '{prompt}'")
            print(f"Generating CLIP‑ES for {img_path} – '{prompt}'")
            heatmap, score = compute_clip_es_heatmap(img, prompt, clip_model, preprocess, device, return_raw=False)
            # The function already displays the plot.


if __name__ == "__main__":
    run_demo()
