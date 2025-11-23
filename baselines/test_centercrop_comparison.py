
import torch
import clip
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add project root to path
sys.path.append(".")
sys.path.insert(0, "bcosification")

from bcos_localization import load_plain_clip
from utils.preprocess import aspect_preserving_preprocess
from utils.clip_patch import patch_clip_attnpool
from first_attempt.gradcam_baseline import GradCAM

def run_comparison():
    print("Loading models...")
    clip_model, standard_preprocess, device = load_plain_clip()
    
    # Patch the model to allow dynamic resolution (needed for the aspect-preserving part)
    patch_clip_attnpool(clip_model)
    
    model = clip_model.visual
    gradcam = GradCAM(model)
    
    img_path = "test_images/cat_background.png"
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found.")
        return
        
    img = Image.open(img_path).convert('RGB')
    prompt = "cat"
    
    print(f"Processing {img_path} with prompt '{prompt}'...")
    
    # --- Method 1: Standard CLIP Preprocessing (Center Crop) ---
    print("Running Method 1: Standard Center Crop...")
    
    # Standard preprocess: Resize(224) + CenterCrop(224)
    # We can use the 'standard_preprocess' loaded from clip.load()
    img_tensor_crop = standard_preprocess(img).unsqueeze(0).to(device)
    img_tensor_crop.requires_grad = True
    
    # Generate GradCAM
    # We manually run the GradCAM logic here to control the inputs exactly
    text_tokens = clip_model.encode_text(clip.tokenize([prompt]).to(device))
    text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)
    text_tokens = text_tokens.detach()
    
    model.zero_grad()
    logits = model(img_tensor_crop)
    scores = (logits @ text_tokens.T).squeeze()
    score_crop = scores.max()
    score_crop.backward()
    
    grads = gradcam.gradients
    acts = gradcam.activations
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam_crop = (weights * acts).sum(dim=1, keepdim=True)
    cam_crop = torch.nn.functional.relu(cam_crop)
    cam_crop = (cam_crop - cam_crop.min()) / (cam_crop.max() - cam_crop.min() + 1e-8)
    cam_crop = cam_crop.squeeze().cpu().detach().numpy()
    
    # Resize heatmap to original size (this stretches the center crop explanation to the full image)
    cam_crop_resized = Image.fromarray((cam_crop * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    cam_crop_resized = np.array(cam_crop_resized) / 255.0
    
    # --- Method 2: Aspect-Preserving Preprocessing (New Fix) ---
    print("Running Method 2: Aspect-Preserving Resize...")
    
    img_tensor_full = aspect_preserving_preprocess(img, target_small_edge=224).unsqueeze(0).to(device)
    img_tensor_full.requires_grad = True
    
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
    
    # Resize heatmap to original size
    cam_full_resized = Image.fromarray((cam_full * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    cam_full_resized = np.array(cam_full_resized) / 255.0
    
    # --- Visualization ---
    print("Creating visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # 2. What the model sees (Center Crop)
    # We need to manually apply the crop to show what the model saw
    # CLIP transform: Resize(224, interpolation=BICUBIC) -> CenterCrop(224)
    # Note: standard_preprocess uses Bicubic for resize
    import torchvision.transforms as T
    crop_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224)
    ])
    img_cropped_view = crop_transform(img)
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_cropped_view)
    plt.title("Input to Model (Method 1: Center Crop)")
    plt.axis('off')
    
    # 3. Result Method 1 (Stretched)
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    plt.imshow(cam_crop_resized, cmap='jet', alpha=0.5)
    plt.title(f"Method 1: Center Crop (Score: {score_crop.item():.4f})\n(Note: Heatmap is stretched!)")
    plt.axis('off')
    
    # 4. Result Method 2 (Correct)
    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.imshow(cam_full_resized, cmap='jet', alpha=0.5)
    plt.title(f"Method 2: Aspect-Preserving (Score: {score_full.item():.4f})\n(Correctly aligned)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison_centercrop_vs_full.png")
    print("Saved comparison to comparison_centercrop_vs_full.png")

if __name__ == "__main__":
    run_comparison()
