"""
B-cos localization - Standalone implementation mimicking original bcosification code.
This is a direct copy of the essential logic from:
bcosification/interpretability/analyses/text_localisation.py
"""

import sys

sys.path.insert(0, "bcosification")

import pathlib
import os

# Patch PosixPath to WindowsPath on Windows
if os.name == "nt":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from bcos.experiments.utils import Experiment
import bcos.data.transforms as custom_transforms
import scipy.ndimage as ndimage


# def load_bcos_model(exp_name=None):
#     """
#     Load B-cos CLIP model.
#     Uses the default experiment path from the bcosification repository.
#     """
#     if exp_name is None:
#         exp_name = "experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification"

#     # The Experiment class will look for the model checkpoint in:
#     # bcosification/experiments/ImageNet/clip_bcosification/resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification/
#     exp = Experiment(exp_name)
#     model = exp.load_trained_model()

#     # Don't use attn_unpool for standard operation
#     model.model.attnpool.attn_unpool = False

#     device = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )

#     return model, device


def load_bcos_model(exp_name=None):
    # 1. Definiamo i parametri
    dataset = "ImageNet"
    network_type = "clip_bcosification"
    experiment_name = (
        "resnet_50_clip_b2_noBias_randomResizedCrop_sigLip_ImageNet_bcosification"
    )

    # 2. Definisci dove si trova la cartella "experiments" che hai appena creato/usato
    # ATTENZIONE: Assicurati che questo path punti alla cartella che CONTIENE "ImageNet"
    # Se hai messo il file in: /leonardo_work/.../AML/experiments/ImageNet/...
    # Allora base_dir deve essere: /leonardo_work/.../AML/experiments
    base_dir = "./experiments"

    print(f"DEBUG: Carico esperimento standard da {base_dir}...")

    # 3. Inizializza l'esperimento
    # Ora 'Experiment' troverà tutto da solo perché la struttura delle cartelle è corretta
    exp = Experiment(
        path_or_dataset=dataset,
        base_network=network_type,
        experiment_name=experiment_name,
        base_directory=base_dir,
    )

    # 4. Carica il modello
    # reload="last" cercherà automaticamente 'last.ckpt' nella cartella giusta
    model = exp.load_trained_model(reload="last", verbose=True)

    # # 5. Configurazione post-caricamento (Standard)
    # if hasattr(model, "model") and hasattr(model.model, "attnpool"):
    #     model.model.attnpool.attn_unpool = False
    # elif hasattr(model, "attnpool"):
    #     model.attnpool.attn_unpool = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, device


def load_clip_for_text():
    """
    Load CLIP model for text encoding.
    Copied from original get_clip_model() function.
    """
    clip_model, _ = clip.load("RN50")
    clip_model.float()
    clip_model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return clip_model, device


def tokenize_text_prompt(clip_model, text_prompt, templates=None):
    """
    Tokenize text using CLIP.
    Copied from original tokenize_text() function.
    """
    if templates is None:
        # Standard CLIP zero-shot classification templates
        # From original CLIP paper: https://github.com/openai/CLIP
        templates = [
            "a photo of a {}.",
            "a photo of the {}.",
            "a picture of a {}.",
            "an image of a {}.",
            "an image of the {}.",
        ]

    test_text = [template.format(text_prompt) for template in templates]
    device = next(clip_model.parameters()).device

    test_text = clip.tokenize(test_text).to(device)
    class_embeddings = clip_model.encode_text(test_text)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()

    return class_embedding.unsqueeze(1)


def compute_attributions(
    model, test_img, zeroshot_weight, smooth=0, alpha_percentile=99.5
):
    """
    Compute B-cos attributions.
    EXACT COPY from original compute_attributions() function (lines 70-128).
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    with torch.enable_grad(), model.explanation_mode(), torch.autograd.set_detect_anomaly(
        True
    ):
        imga = test_img[None].to(device).requires_grad_()
        outa = model(imga)

        img_features = outa / outa.norm(dim=-1, keepdim=True)
        logits = img_features @ zeroshot_weight.to(device)

        # Handle attention unpooling (we keep it False for standard operation)
        if model.model.attnpool.attn_unpool:
            logits = logits.reshape(-1, 1)
            logits = logits.mean(dim=0)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logits.max(1).values.backward(inputs=[imga])

        grada = imga.grad
        imga = imga.detach().cpu()[0]
        grada = grada.detach().cpu()[0]

    # Compute contributions (element-wise product, sum over channels)
    contribs = (imga * grada).sum(0, keepdim=True)

    # Compute RGB gradient for explanation
    rgb_grad = grada / (grada.abs().max(0, keepdim=True).values + 1e-12)
    rgb_grad = rgb_grad.clamp(min=0)
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)

    # Compute alpha channel
    alpha = grada.norm(p=2, dim=0, keepdim=True)
    alpha = torch.where(contribs < 0, torch.tensor(1e-12), alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)

    # Combine RGB + alpha
    rgb_grad = torch.cat([rgb_grad, alpha], dim=0)
    grad_image = rgb_grad.permute(1, 2, 0).detach().cpu().numpy()

    # Process raw contributions
    contribs = contribs.detach().cpu().numpy().squeeze()
    cutoff = np.percentile(np.abs(contribs), 99.5)
    contribs = np.clip(contribs, -cutoff, cutoff)
    vrange = np.max(np.abs(contribs.flatten()))

    return grad_image, contribs, vrange, logits.item()


def get_bcos_heatmap(
    model, image, text_prompt, clip_model=None, device=None, return_raw=False
):
    """
    Generate B-cos heatmap for an image and text prompt.
    Uses the exact preprocessing and computation from original code.

    Args:
        model: B-cos model
        image: PIL Image
        text_prompt: str
        clip_model: CLIP model (will load if None)
        device: device (ignored, uses auto-detection)
        return_raw: if True, returns (contribs, vrange, score); else (rgba_explanation, score)

    Returns:
        If return_raw=True:
            contribs: (H, W) raw attributions in range [-vrange, vrange]
                     Use BWR colormap: Red=positive, Blue=negative
            vrange: float, symmetric range for visualization
            score: float, classification score
        Else:
            rgba_explanation: (H, W, 4) RGBA explanation
            score: float
    """
    # Load CLIP if not provided
    if clip_model is None:
        clip_model, _ = load_clip_for_text()

    # Preprocess image - resize smaller edge to 224, maintain aspect ratio
    # NO CenterCrop to preserve full image content
    def _convert_image_to_rgb(img):
        return img.convert("RGB")

    # Calculate new size maintaining aspect ratio
    w, h = image.size
    if w < h:
        new_w = 224
        new_h = int(h * (224 / w))
    else:
        new_h = 224
        new_w = int(w * (224 / h))

    transform = transforms.Compose(
        [
            transforms.Resize(
                (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            custom_transforms.AddInverse(),  # Adds 3 inverse channels -> 6 total
        ]
    )

    test_img = transform(image)

    # Tokenize text
    zeroshot_weight = tokenize_text_prompt(clip_model, text_prompt)

    # Compute attributions
    grad_image, contribs, vrange, score = compute_attributions(
        model, test_img, zeroshot_weight, smooth=0, alpha_percentile=99.5
    )

    if return_raw:
        return contribs, vrange, score
    else:
        return grad_image, score


def get_bounding_box(heatmap, threshold_percent=0.2):
    """
    Calculate the bounding box for the largest connected component in the heatmap.

    Args:
        heatmap: (H, W) numpy array of attribution values
        threshold_percent: float, percentage of max value to use as threshold

    Returns:
        bbox: (xmin, ymin, xmax, ymax) coordinates of the bounding box
              Returns None if no pixels are above threshold.
    """
    # Find max value
    v_max = np.max(heatmap)

    # Calculate threshold
    threshold = threshold_percent * v_max

    # Binarize
    binary_map = heatmap > threshold

    # Label connected components
    labeled_map, num_features = ndimage.label(binary_map)

    if num_features == 0:
        return None

    # Find largest component
    component_sizes = ndimage.sum(binary_map, labeled_map, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1

    # Get bounding box of largest component
    slices = ndimage.find_objects(labeled_map == largest_component_label)[0]
    y_slice, x_slice = slices

    xmin = x_slice.start
    xmax = x_slice.stop
    ymin = y_slice.start
    ymax = y_slice.stop

    return xmin, ymin, xmax, ymax


def load_plain_clip():
    """Load plain CLIP for baselines (not B-cos)."""
    clip_model, preprocess = clip.load("RN50")
    clip_model.float()
    clip_model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    clip_model = clip_model.to(device)

    return clip_model, preprocess, device
