"""
GradCAM baseline implementation using the CLIP visual backbone.

DISCLAIMER: This script is a baseline implementation and may need further revision.
Patches were applied to handle aspect-ratio preserving resizing (B-cos style pooling).
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
from utils.preprocess import aspect_preserving_preprocess
from utils.bcos_style_pooling import patch_clip_with_bcos_pooling


class GradCAM:
    """Simple GradCAM implementation for CLIP visual backbone.

    The class registers forward and backward hooks on the target
    convolutional layer, computes the GradCAM heatmap for a given
    text prompt, and returns the heatmap resized to the original
    image dimensions.
    """

    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.target_layer = self._find_target_layer(target_layer_name)
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_target_layer(self, name):
        """Find the target layer by attribute traversal.
        Supports dot‑notation names, e.g. "model.layer4".
        """
        module = self.model
        for attr in name.split('.'):
            if not hasattr(module, attr):
                raise AttributeError(f"Layer '{attr}' not found in {module}")
            module = getattr(module, attr)
        return module

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; we need the gradient w.r.t. the output
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image, text_prompt, clip_model, preprocess, device, return_raw=False):
        """Generate GradCAM heatmap for *image* and *text_prompt*.

        Args:
            image (PIL.Image): Input image.
            text_prompt (str): Text query.
            clip_model: CLIP model (visual + text encoders).
            device (torch.device): Device for computation.
            return_raw (bool): If True, returns the raw heatmap array.
        """
        # Patch the model to handle arbitrary image sizes (B-cos style: ignore pos embed)
        patch_clip_with_bcos_pooling(clip_model)

        # Prepare image tensor using aspect-preserving preprocess
        img_tensor = aspect_preserving_preprocess(image, target_small_edge=224).unsqueeze(0).to(device)
        img_tensor.requires_grad = True

        # Encode text prompt using CLIP text encoder
        text_tokens = clip_model.encode_text(clip.tokenize([text_prompt]).to(device))
        text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)

        # Forward pass through the model
        with torch.enable_grad():
            logits = self.model(img_tensor)
            # Compute similarity with text embedding (dot product)
            scores = (logits @ text_tokens.T).squeeze()
            # Choose the highest scoring class
            score = scores.max()
            # Backward pass to get gradients w.r.t. the target layer
            self.model.zero_grad()
            score.backward()

        # Gradients and activations are now populated
        grads = self.gradients  # shape: (N, C, H, W)
        acts = self.activations  # shape: (N, C, H, W)
        
        # Global average pooling of gradients over spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (N, 1, H, W)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.squeeze().cpu().numpy()

        # Resize to original image size
        original_size = image.size  # (W, H)
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
        cam_resized = np.array(cam_resized) / 255.0

        if return_raw:
            return cam_resized, score.item()
        else:
            # Simple visualization helper
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.imshow(cam_resized, cmap='jet', alpha=0.5)
            plt.title(f"GradCAM – {text_prompt} (score={score.item():.4f})")
            plt.axis('off')
            plt.show()
            return cam_resized, score.item()


def run_demo():
    """Demo script to generate GradCAM heatmaps for a few test images."""
    # Load models
    print("Loading models for GradCAM demo...")
    clip_model, preprocess, device = load_plain_clip()
    model = clip_model.visual  # Use the plain visual backbone

    # Test cases
    test_cases = [
        {"image": "cat_background.png", "prompts": ["cat", "grass", "tree"]},
        {"image": "car.png", "prompts": ["car", "street", "building"]},
        {"image": "objects.png", "prompts": ["people", "book", "teapot"]},
    ]

    gradcam = GradCAM(model)

    for case in test_cases:
        img_path = case["image"]
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} – not found")
            continue
        img = Image.open(img_path).convert('RGB')
        for prompt in case["prompts"]:
            print(f"Generating GradCAM for {img_path} – '{prompt}'")
            heatmap, score = gradcam.generate(img, prompt, clip_model, preprocess, device, return_raw=False)


if __name__ == "__main__":
    run_demo()

