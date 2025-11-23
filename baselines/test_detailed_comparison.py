
import torch
import clip
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T

# Add project root to path
sys.path.append(".")
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip
from utils.preprocess import aspect_preserving_preprocess
from utils.bcos_style_pooling import patch_clip_with_bcos_pooling
from first_attempt.gradcam_baseline import GradCAM
from clip_es_improved import compute_clip_es_heatmap_improved

def run_detailed_comparison():
    print("Loading models...")
    clip_model, standard_preprocess, device = load_plain_clip()
    
    # Patch the model to allow dynamic resolution (B-cos style)
    patch_clip_with_bcos_pooling(clip_model)
    
    model = clip_model.visual
    gradcam = GradCAM(model)
    
    img_path = "test_images/cat_background.png"
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found.")
        return
        
    img = Image.open(img_path).convert('RGB')
    prompts = ["cat", "grass", "dog"]
    
    # Prepare plot
    num_rows = len(prompts)
    num_cols = 5 # Original, GradCAM (New), CLIP-ES (New), GradCAM (Crop), CLIP-ES (Crop)
    
    plt.figure(figsize=(25, 5 * num_rows))
    
    # Pre-calculate standard crop transform for visualization
    crop_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224)
    ])
    img_cropped_view = crop_transform(img)
    
    # Calculate crop coordinates to place heatmap correctly
    # Resize logic: smaller edge to 224
    w, h = img.size
    if w < h:
        new_w = 224
        new_h = int(round(h * 224 / w))
    else:
        new_h = 224
        new_w = int(round(w * 224 / h))
        
    # Center crop logic
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    right = left + 224
    bottom = top + 224
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt: '{prompt}'...")
        
        # --- 1. Original Image ---
        plt.subplot(num_rows, num_cols, i * num_cols + 1)
        plt.imshow(img)
        if i == 0: plt.title("Original Image")
        plt.axis('off')
        plt.text(-50, img.size[1]//2, prompt, fontsize=16, rotation=90, va='center')
        
        # --- Method A: Aspect-Preserving (New Fix) ---
        
        # GradCAM (New)
        img_tensor_full = aspect_preserving_preprocess(img, target_small_edge=224).unsqueeze(0).to(device)
        img_tensor_full.requires_grad = True
        
        text_tokens = clip_model.encode_text(clip.tokenize([prompt]).to(device))
        text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)
        text_tokens = text_tokens.detach()
        
        model.zero_grad()
        logits = model(img_tensor_full)
        scores = (logits @ text_tokens.T).squeeze()
        score_full = scores.max()
        score_full.backward(retain_graph=False)
        
        grads = gradcam.gradients
        acts = gradcam.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam_full = (weights * acts).sum(dim=1, keepdim=True)
        cam_full = torch.nn.functional.relu(cam_full)
        cam_full = (cam_full - cam_full.min()) / (cam_full.max() - cam_full.min() + 1e-8)
        cam_full = cam_full.squeeze().cpu().detach().numpy()
        cam_full_resized = Image.fromarray((cam_full * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
        cam_full_resized = np.array(cam_full_resized) / 255.0
        
        plt.subplot(num_rows, num_cols, i * num_cols + 2)
        plt.imshow(img)
        plt.imshow(cam_full_resized, cmap='jet', alpha=0.5)
        if i == 0: plt.title("GradCAM (New Fix)")
        plt.axis('off')
        
        # CLIP-ES (New)
        heatmap_es_full, score_es_full = compute_clip_es_heatmap_improved(
            img, prompt, clip_model, standard_preprocess, device, return_raw=True
        )
        # Note: compute_clip_es_heatmap_improved already returns resized heatmap
        
        plt.subplot(num_rows, num_cols, i * num_cols + 3)
        plt.imshow(img)
        plt.imshow(heatmap_es_full, cmap='jet', alpha=0.5)
        if i == 0: plt.title("CLIP-ES (New Fix)")
        plt.axis('off')
        
        # --- Method B: Center Crop (Old Way) ---
        
        # Prepare cropped tensor
        img_tensor_crop = standard_preprocess(img).unsqueeze(0).to(device)
        img_tensor_crop.requires_grad = True
        
        # GradCAM (Crop)
        model.zero_grad()
        logits = model(img_tensor_crop)
        scores = (logits @ text_tokens.T).squeeze()
        score_crop = scores.max()
        score_crop.backward(retain_graph=False)
        
        grads = gradcam.gradients
        acts = gradcam.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam_crop = (weights * acts).sum(dim=1, keepdim=True)
        cam_crop = torch.nn.functional.relu(cam_crop)
        cam_crop = (cam_crop - cam_crop.min()) / (cam_crop.max() - cam_crop.min() + 1e-8)
        cam_crop = cam_crop.squeeze().cpu().detach().numpy() # 7x7 or similar small size
        
        # Resize heatmap to 224x224 (the crop size)
        cam_crop_img = Image.fromarray((cam_crop * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        cam_crop_np = np.array(cam_crop_img) / 255.0
        
        # Create full-size overlay
        # 1. Resize original image to intermediate size (e.g. 448x224)
        # 2. Paste the 224x224 heatmap in the center
        # 3. Resize back to original size (e.g. 2816x1536)
        
        # Step 1: Create blank mask at intermediate resolution
        full_mask_intermediate = Image.new('L', (new_w, new_h), color=0)
        # Step 2: Paste crop heatmap
        full_mask_intermediate.paste(cam_crop_img, (left, top))
        # Step 3: Resize to original
        full_mask_final = full_mask_intermediate.resize(img.size, Image.NEAREST) # Use nearest to keep zero boundaries clean? Or bilinear.
        full_mask_final_np = np.array(full_mask_final) / 255.0
        
        plt.subplot(num_rows, num_cols, i * num_cols + 4)
        plt.imshow(img)
        # Mask out the zero areas for better visualization? Or just show it.
        # Let's show it overlaid.
        plt.imshow(full_mask_final_np, cmap='jet', alpha=0.5)
        # Draw a box around the crop area? 
        # That's hard on the original image coordinates without math.
        # But the heatmap itself shows the crop area.
        if i == 0: plt.title("GradCAM (Center Crop Only)")
        plt.axis('off')
        
        # CLIP-ES (Crop)
        # We use our modified function with override_input_tensor
        heatmap_es_crop_raw, _ = compute_clip_es_heatmap_improved(
            img, prompt, clip_model, standard_preprocess, device, return_raw=True, override_input_tensor=img_tensor_crop
        )
        # Note: The function returns heatmap resized to *original image size* based on 'img.size'.
        # BUT, since we passed a crop tensor, the internal logic calculates attention on the crop.
        # The resize at the end of the function uses 'img.size'.
        # Wait, if we pass 'img' (full size) but 'img_tensor_crop' (crop), 
        # the function will calculate CAM on crop features (7x7), refine it, 
        # and then resize that 7x7 result to 'img.size' (full size).
        # THIS IS EXACTLY THE STRETCHING BUG we want to avoid in this visualization column.
        # We want the heatmap to be 224x224 and placed in the center.
        
        # To fix this for visualization, we need to intercept the raw CAM before resizing, 
        # OR we can just pass a dummy 224x224 image to the function so it resizes to 224x224.
        dummy_crop_img = Image.new('RGB', (224, 224))
        heatmap_es_crop_224, _ = compute_clip_es_heatmap_improved(
            dummy_crop_img, prompt, clip_model, standard_preprocess, device, return_raw=True, override_input_tensor=img_tensor_crop
        )
        
        # Now place this 224x224 heatmap into the full mask
        es_mask_intermediate = Image.new('L', (new_w, new_h), color=0)
        es_mask_intermediate.paste(Image.fromarray((heatmap_es_crop_224 * 255).astype(np.uint8)), (left, top))
        es_mask_final = es_mask_intermediate.resize(img.size, Image.NEAREST)
        es_mask_final_np = np.array(es_mask_final) / 255.0
        
        plt.subplot(num_rows, num_cols, i * num_cols + 5)
        plt.imshow(img)
        plt.imshow(es_mask_final_np, cmap='jet', alpha=0.5)
        if i == 0: plt.title("CLIP-ES (Center Crop Only)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("detailed_comparison_grid.png")
    print("Saved detailed comparison to detailed_comparison_grid.png")

if __name__ == "__main__":
    run_detailed_comparison()
