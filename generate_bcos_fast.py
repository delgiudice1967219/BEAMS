import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
from PIL import Image

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
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# --- UTILS ---
def precompute_background_weights(clip_model, device):
    bg_classes = [
        "ground",
        "land",
        "grass",
        "tree",
        "building",
        "wall",
        "sky",
        "lake",
        "water",
        "river",
        "sea",
        "railway",
        "railroad",
        "keyboard",
        "helmet",
        "cloud",
        "house",
        "mountain",
        "ocean",
        "road",
        "rock",
        "street",
        "valley",
        "bridge",
        "sign",
    ]
    bg_prompts = [f"a photo of {bg}" for bg in bg_classes]

    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_target_weight(clip_model, class_name, device):
    prompts = [
        f"a clean origami {class_name}.",
        f"a photo of a {class_name}.",
        f"the {class_name}.",
    ]
    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


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
    # Ensure shape [H, W]
    while t.dim() > 2:
        t = t.squeeze(0)
    return t


def process_map_pair(map_t, map_b, blur_transform):
    """Joint Normalization Logic"""
    # Ensure [C, H, W] for blur
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

    return probs[1]  # [1, H, W]


# --- OPTIMIZED PROCESSING ---
def process_image_optimized(model, img_pil, target_classes, class_weights, bg_weight):
    base_w, base_h = img_pil.size
    blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    # Store results: class_idx -> [accumulated_maps]
    results = {cls: [] for cls in target_classes}

    for s in SCALES:
        prep = get_transform(s)
        # Input: [6, S, S] (3D Tensor)
        # CRITICAL FIX: No unsqueeze(0) here. Library expects 3D.
        img_tens = prep(img_pil).to(DEVICE)

        # --- 1. Compute & Cache Background (Original) ---
        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                # Pass 3D tensor
                _, map_b, _, _ = compute_attributions(model, img_tens, bg_weight)

        map_b = ensure_tensor(map_b)  # [H, W]

        # --- 2. Process Foreground Classes (Original) ---
        for cls in target_classes:
            w_target = class_weights[cls]
            with torch.amp.autocast("cuda", enabled=True):
                with torch.no_grad():
                    _, map_t, _, _ = compute_attributions(model, img_tens, w_target)

            map_t = ensure_tensor(map_t)
            prob_orig = process_map_pair(map_t, map_b, blur_transform)

            # Resize: prob_orig is [1, H, W]. Need [1, 1, H, W] for interpolate input
            resized_orig = F.interpolate(
                prob_orig.unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()  # [H, W]
            results[cls].append(resized_orig)

        # --- 3. Compute & Cache Background (Flipped) ---
        # Input is [6, H, W]. Width is dimension 2.
        img_flip = torch.flip(img_tens, [2])

        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                _, map_b_f, _, _ = compute_attributions(model, img_flip, bg_weight)

        map_b_f = ensure_tensor(map_b_f)

        # --- 4. Process Foreground Classes (Flipped) ---
        for cls in target_classes:
            w_target = class_weights[cls]
            with torch.amp.autocast("cuda", enabled=True):
                with torch.no_grad():
                    _, map_t_f, _, _ = compute_attributions(model, img_flip, w_target)

            map_t_f = ensure_tensor(map_t_f)
            prob_flip = process_map_pair(map_t_f, map_b_f, blur_transform)

            # Un-flip: prob_flip is [1, H, W]. Width is dimension 2.
            prob_flip = torch.flip(prob_flip, [2])

            # Resize
            resized_flip = F.interpolate(
                prob_flip.unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            results[cls].append(resized_flip)

    # Aggregate results for all classes
    final_maps = {}
    for cls in target_classes:
        # Stack all scales/flips and average
        final_maps[cls] = torch.mean(torch.stack(results[cls]), dim=0).cpu().numpy()

    return final_maps


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output_cam_bcos")
    args = parser.parse_args()

    print(f"Loading models on {DEVICE}...")
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    # Precompute BG
    bg_weight = precompute_background_weights(clip_model, DEVICE)
    fg_cache = {}

    with open(args.split_file, "r") as f:
        file_list = [x.strip() for x in f.readlines()]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f"Processing {len(file_list)} images...")
    import xml.etree.ElementTree as ET

    for img_name in tqdm(file_list):
        img_name_clean = img_name.replace(".jpg", "")
        img_path = os.path.join(args.img_root, img_name_clean + ".jpg")
        xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

        if not os.path.exists(img_path):
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = [obj.find("name").text for obj in root.findall("object")]
            unique_classes = set(objects)
            img_pil = Image.open(img_path).convert("RGB")
        except:
            continue

        # Prepare weights for this image
        target_classes = []
        class_weights = {}

        for cls_name in unique_classes:
            if cls_name not in VOC_CLASSES:
                continue

            if cls_name not in fg_cache:
                fg_cache[cls_name] = get_target_weight(clip_model, cls_name, DEVICE)

            target_classes.append(cls_name)
            class_weights[cls_name] = fg_cache[cls_name]

        if not target_classes:
            continue

        try:
            # RUN OPTIMIZED
            final_maps_dict = process_image_optimized(
                bcos_model, img_pil, target_classes, class_weights, bg_weight
            )

            # Save format compatible with eval script
            result_keys = []
            result_cams = []

            for cls_name, prob_map in final_maps_dict.items():
                result_keys.append(VOC_CLASSES.index(cls_name))
                result_cams.append(prob_map)

            np.save(
                os.path.join(args.out_dir, img_name_clean + ".npy"),
                {"keys": np.array(result_keys), "attn_highres": np.array(result_cams)},
            )

        except Exception as e:
            print(f"Error {img_name}: {e}")
            torch.cuda.empty_cache()
            continue
