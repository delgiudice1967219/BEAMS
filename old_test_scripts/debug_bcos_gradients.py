"""
Debug script to understand why gradients are zero.
"""

import sys
sys.path.insert(0, "bcosification")

import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from bcos.experiments.utils import Experiment
import bcos.data.transforms as custom_transforms

# Load model
exp_name = "experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification"
exp = Experiment(exp_name)
model = exp.load_trained_model()
model.model.attnpool.attn_unpool = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Load CLIP
clip_model, _ = clip.load("RN50")
clip_model.float()
clip_model.eval()

# Load and preprocess image
image = Image.open("car.png").convert('RGB')

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
print(f"test_img shape: {test_img.shape}")  # Should be (6, 224, 224)

# Tokenize "car"
text_prompt = "car"
templates = ["a photo of a {}."]
test_text = [template.format(text_prompt) for template in templates]
text_device = next(clip_model.parameters()).device
test_text = clip.tokenize(test_text).to(text_device)
class_embeddings = clip_model.encode_text(test_text)
class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
class_embedding = class_embeddings.mean(dim=0)
class_embedding /= class_embedding.norm()
zeroshot_weight = class_embedding.unsqueeze(1)

print(f"zeroshot_weight shape: {zeroshot_weight.shape}")
print(f"zeroshot_weight device: {zeroshot_weight.device}")

# Run forward pass with gradients
model.to(device)
model.eval()

print("\nStarting gradient computation...")
with torch.enable_grad(), model.explanation_mode():
    imga = test_img[None].to(device)
    print(f"imga shape: {imga.shape}, requires_grad: {imga.requires_grad}")
    
    imga.requires_grad_()
    print(f"After requires_grad_(), imga.requires_grad: {imga.requires_grad}")
    
    outa = model(imga)
    print(f"outa shape: {outa.shape}, requires_grad: {outa.requires_grad}")
    
    img_features = outa / outa.norm(dim=-1, keepdim=True)
    print(f"img_features shape: {img_features.shape}, requires_grad: {img_features.requires_grad}")
    
    logits = img_features @ zeroshot_weight.to(device)
    print(f"logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
    
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    print(f"final logits shape: {logits.shape}")
    print(f"logits value: {logits}")
    
    target = logits.max(1).values
    print(f"target: {target}, requires_grad: {target.requires_grad}")
    
    print("\nCalling backward...")
    target.backward(inputs=[imga])
    
    grada = imga.grad
    print(f"grada is None: {grada is None}")
    if grada is not None:
        print(f"grada shape: {grada.shape}")
        print(f"grada min/max: [{grada.min():.6f}, {grada.max():.6f}]")
        print(f"grada abs sum: {grada.abs().sum():.6f}")
        
        # Compute contributions
        imga_cpu = imga.detach().cpu()[0]
        grada_cpu = grada.detach().cpu()[0]
        contribs = (imga_cpu * grada_cpu).sum(0, keepdim=True)
        print(f"\ncontribs shape: {contribs.shape}")
        print(f"contribs min/max: [{contribs.min():.6f}, {contribs.max():.6f}]")
        print(f"contribs abs sum: {contribs.abs().sum():.6f}")
