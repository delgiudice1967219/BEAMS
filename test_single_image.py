import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt

# --- IMPORTS ---
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [448, 560]


# --- UTILS ---
def get_transform(size):
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )


def ensure_tensor(t):
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).to(DEVICE)
    while t.dim() > 2:
        t = t.squeeze(0)
    return t


def apply_heatmap(img_rgb, map_data):
    """Crea visualizzazione heatmap"""
    # Normalize map to 0-255
    map_data = (map_data - map_data.min()) / (map_data.max() - map_data.min() + 1e-8)
    map_uint8 = (map_data * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay: 50% image, 50% heatmap
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    return overlay


def process_map_pair(map_t, map_b, blur_transform):
    if map_t.dim() == 2:
        map_t = map_t.unsqueeze(0)
    if map_b.dim() == 2:
        map_b = map_b.unsqueeze(0)

    map_t = blur_transform(map_t)
    map_b = blur_transform(map_b)

    g_min = min(map_t.min(), map_b.min())
    g_max = max(map_t.max(), map_b.max())
    denom = g_max - g_min + 1e-8

    map_t = (map_t - g_min) / denom
    map_b = (map_b - g_min) / denom

    stack = torch.stack([map_b, map_t], dim=0)
    probs = F.softmax(stack * 20, dim=0)
    return probs[1]


def process_single_prompt(model, img_pil, w_target, w_bg):
    base_w, base_h = img_pil.size
    accumulated_maps = []
    blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    for s in SCALES:
        prep = get_transform(s)
        img_tens = prep(img_pil).to(DEVICE)  # [6, S, S]

        # 1. Original
        with torch.no_grad():
            _, map_t, _, _ = compute_attributions(model, img_tens, w_target)
            _, map_b, _, _ = compute_attributions(model, img_tens, w_bg)

        prob_orig = process_map_pair(
            ensure_tensor(map_t), ensure_tensor(map_b), blur_transform
        )

        resized_orig = F.interpolate(
            prob_orig.unsqueeze(0).unsqueeze(0),
            size=(base_h, base_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        accumulated_maps.append(resized_orig)

        # 2. Flip
        img_flip = torch.flip(img_tens, [2])
        with torch.no_grad():
            _, map_t_f, _, _ = compute_attributions(model, img_flip, w_target)
            _, map_b_f, _, _ = compute_attributions(model, img_flip, w_bg)

        prob_flip = process_map_pair(
            ensure_tensor(map_t_f), ensure_tensor(map_b_f), blur_transform
        )
        prob_flip = torch.flip(prob_flip, [1])  # Unflip

        resized_flip = F.interpolate(
            prob_flip.unsqueeze(0).unsqueeze(0),
            size=(base_h, base_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        accumulated_maps.append(resized_flip)

    return torch.mean(torch.stack(accumulated_maps), dim=0).cpu().numpy()


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt1", type=str, required=True, help="Es: person running")
    parser.add_argument(
        "--prompt2", type=str, required=True, help="Es: person riding a bike"
    )
    args = parser.parse_args()

    print("Caricamento modelli...")
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    # Preparazione Immagine
    img_pil = Image.open(args.img_path).convert("RGB")
    img_np = np.array(img_pil)

    # Preparazione Background (Generico)
    bg_prompts = ["background", "noise", "blur", "ground", "sky", "tree", "building"]
    with torch.no_grad():
        w_bg = torch.cat([tokenize_text_prompt(clip_model, p) for p in bg_prompts]).to(
            DEVICE
        )
        w_bg = torch.mean(w_bg, dim=0, keepdim=True)

    # Preparazione Prompts Target
    prompts_list = [args.prompt1, args.prompt2]
    maps = []

    print(f"Generazione Heatmaps per: {prompts_list}")

    for p_text in prompts_list:
        print(f"Processing: '{p_text}'...")
        # Creiamo embedding per il prompt specifico
        # Usiamo template multipli per robustezza
        templates = [f"a photo of a {p_text}.", f"the {p_text}.", f"{p_text}."]
        with torch.no_grad():
            w_target = torch.cat(
                [tokenize_text_prompt(clip_model, t) for t in templates]
            ).to(DEVICE)
            w_target = torch.mean(w_target, dim=0, keepdim=True)

        # Inferenza
        prob_map = process_single_prompt(bcos_model, img_pil, w_target, w_bg)
        maps.append(prob_map)

    # --- VISUALIZZAZIONE ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Immagine Originale
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Prompt 1
    vis1 = apply_heatmap(img_np, maps[0])
    axes[1].imshow(vis1)
    axes[1].set_title(f"Prompt: {args.prompt1}")
    axes[1].axis("off")

    # 3. Prompt 2
    vis2 = apply_heatmap(img_np, maps[1])
    axes[2].imshow(vis2)
    axes[2].set_title(f"Prompt: {args.prompt2}")
    axes[2].axis("off")

    out_file = "test_result_comparison.jpg"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"\nRISULTATO SALVATO: {out_file}")
    print("Apri questo file per vedere se la tua intuizione è corretta!")
