"""
Improved CLIP-ES implementation adapted for ResNet50.
Incorporates attention-based refinement from the official CLIP-ES repo.

DISCLAIMER: This script is a baseline implementation and may need further revision.
Patches were applied to handle aspect-ratio preserving resizing (B-cos style pooling).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import clip

# Ensure the bcosification repo is on the path
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip
from utils.preprocess import aspect_preserving_preprocess
from utils.bcos_style_pooling import patch_clip_with_bcos_pooling

# --- Helper Functions from Official Repo ---

def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    """
    Extract bounding boxes from a scoremap (heatmap).
    Adapted from official CLIP-ES utils.py.
    """
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    
    # Handle OpenCV version differences for findContours
    _CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
    
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)

# --- Re-implemented AttentionPool2d to capture weights ---

def get_attnpool_weights(model, x, device):
    """
    Manually run the attnpool logic to capture attention weights.
    
    Args:
        model: The CLIP visual model (ModifiedResNet).
        x: The feature map from layer4 (N, 2048, H, W).
        device: Computation device.
        
    Returns:
        attn_weight: Attention weights (N, HW+1, HW+1).
    """
    attnpool = model.attnpool
    
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    x = x.flatten(start_dim=2).permute(2, 0, 1)  # (HW, N, C)
    
    # Add mean token
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1, N, C)
    
    # Add positional embedding
    # B-cos style: IGNORE positional embeddings
    # pos_embed = attnpool.positional_embedding.to(device).to(x.dtype)
    # if pos_embed.shape[0] != x.shape[0]:
    #     pos_embed = resize_pos_embed(pos_embed, H, W)
    # x = x + pos_embed[:, None, :]
    
    # Multihead Attention
    # We need to run F.multi_head_attention_forward with need_weights=True
    
    # Prepare weights
    q_proj_weight = attnpool.q_proj.weight
    k_proj_weight = attnpool.k_proj.weight
    v_proj_weight = attnpool.v_proj.weight
    c_proj_weight = attnpool.c_proj.weight
    c_proj_bias = attnpool.c_proj.bias
    
    in_proj_bias = torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias])
    
    # Run attention
    # Use query=x to get full self-attention weights for affinity matrix
    out, attn_weight = F.multi_head_attention_forward(
        query=x, key=x, value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=attnpool.num_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        in_proj_weight=None,
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=c_proj_weight,
        out_proj_bias=c_proj_bias,
        use_separate_proj_weight=True,
        training=False,
        need_weights=True
    )
    
    # attn_weight shape: (N, L, S)
    return attn_weight

# --- Main Function ---

def compute_clip_es_heatmap_improved(image, text_prompt, clip_model, preprocess, device, return_raw=False, override_input_tensor=None):
    """
    Compute CLIP-ES heatmap with refinement.
    """
    # Patch the model to handle arbitrary image sizes (B-cos style: ignore pos embed)
    # patch_clip_with_bcos_pooling(clip_model) # Already patched outside or we can call it here if imported
    # For safety, we assume it's patched outside or we import it.
    # But wait, we imported it at the top. Let's just call the new one if we want to be safe, 
    # OR better yet, rely on the caller to patch it once.
    # The error was 'patch_clip_attnpool' is not defined.
    # I will just remove the line since I am patching it in the main script.
    pass

    # 1. Preprocess
    if override_input_tensor is not None:
        img_tensor = override_input_tensor
    else:
        img_tensor = aspect_preserving_preprocess(image, target_small_edge=224).unsqueeze(0).to(device)
    
    # 2. Hook layer4 to get feature map
    target_layer = clip_model.visual.layer4
    feature_map = None
    
    def hook_fn(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = clip_model.encode_image(img_tensor)
        
    handle.remove()
    
    # feature_map: (1, 2048, H, W)
    
    # 3. Get Attention Weights (Affinity Matrix)
    # We use full self-attention on the feature map + mean token
    attn_weights = get_attnpool_weights(clip_model.visual, feature_map, device) # (1, HW+1, HW+1)
    
    # Extract spatial-spatial attention (ignore mean token at index 0)
    # Shape: (1, HW, HW)
    aff_mat = attn_weights[:, 1:, 1:] 
    aff_mat = aff_mat.squeeze(0) # (HW, HW)
    
    # 4. Compute Initial CAM (Similarity Map)
    # Project features to embedding space
    x = feature_map.permute(0, 2, 3, 1) # (1, H, W, 2048)
    v = clip_model.visual.attnpool.v_proj(x)
    c = clip_model.visual.attnpool.c_proj(v) # (1, H, W, 1024)
    c = c / c.norm(dim=-1, keepdim=True)
    
    text_tokens = clip.tokenize([text_prompt]).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # (1, H, W, 1024) * (1, 1024) -> (1, H, W)
    cam = (c * text_features.unsqueeze(1).unsqueeze(1)).sum(dim=-1).squeeze(0) # (H, W)
    
    # 5. Refine CAM using Affinity Matrix
    cam_flat = cam.flatten() # (HW,)
    
    # Normalize affinity matrix
    aff_mat = aff_mat / (aff_mat.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Refine (Power iteration)
    trans_mat = aff_mat.t()
    for _ in range(10): # 10 iterations as per paper
        cam_flat = torch.matmul(trans_mat, cam_flat)
        
    cam_refined = cam_flat.reshape(cam.shape) # (H, W)
    
    # Normalize refined CAM
    cam_refined = (cam_refined - cam_refined.min()) / (cam_refined.max() - cam_refined.min() + 1e-8)
    
    # Resize to original image size
    original_size = image.size
    cam_resized = Image.fromarray((cam_refined.detach().cpu().numpy() * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
    cam_resized = np.array(cam_resized) / 255.0
    
    score = cam_refined.mean().item()
    
    if return_raw:
        return cam_resized, score
    else:
        return cam_resized, score


def run_demo():
    print("Loading Improved CLIP-ES...")
    clip_model, preprocess, device = load_plain_clip()
    
    img_path = "cat_background.png"
    prompt = "cat"
    
    print(f"Processing {img_path} with prompt '{prompt}'...")
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found.")
        return

    img = Image.open(img_path).convert('RGB')
    
    heatmap, score = compute_clip_es_heatmap_improved(img, prompt, clip_model, preprocess, device)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Improved CLIP-ES (Score: {score:.4f})")
    plt.axis('off')
    
    plt.savefig("improved_clip_es_demo.png")
    print("Saved demo to improved_clip_es_demo.png")


if __name__ == "__main__":
    run_demo()
