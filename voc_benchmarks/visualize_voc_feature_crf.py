import os
import sys
import time
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- SETUP IMPORT ---
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# --- DenseCRF ---
try:
    import pydensecrf.densecrf as dcrf

    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: DenseCRF not found. CRF post-processing will be skipped.")

# --- PARAMS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH_SIZE = 1
NUM_WORKERS = 0

# directory for debug images
DEBUG_DIR = "debug_viz"
os.makedirs(DEBUG_DIR, exist_ok=True)

# --- PASCAL VOC CLASSES ---
VOC_CLASSES = [
    "background",
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

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "a photo of a {class_name}.",
    "a photo of {class_name}.",
    "a close-up photo of a {class_name}.",
    "a cropped photo of a {class_name}.",
    "a detailed photo of a {class_name}.",
]


def get_class_prompt_embeddings(clip_model, class_name, device):
    weights = []
    with torch.no_grad():
        for tmpl in PROMPT_TEMPLATES:
            text = tmpl.format(class_name=class_name)
            w = tokenize_text_prompt(clip_model, text).to(device)
            weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# MULTI-CLASS SOFTMAX PIPELINE
# ---------------------------------------------------------------------------


def process_image_multi_class(
    model,
    image_tensor,
    class_to_prompt_weights,
    scales=(448, 560),
    temperature=12.0,
):
    classes = sorted(class_to_prompt_weights.keys())
    if len(classes) == 0:
        return {}

    base_h, base_w = IMG_SIZE, IMG_SIZE
    blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    original_img_cpu = image_tensor.squeeze()
    all_prob_maps = []

    for s in scales:
        resize_transform = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )
        img_scaled = resize_transform(original_img_cpu).to(DEVICE)

        for flip in [False, True]:
            img_in = torch.flip(img_scaled, dims=[2]) if flip else img_scaled

            class_cams = []
            with torch.no_grad():
                for cls_id in classes:
                    prompt_list = class_to_prompt_weights[cls_id]
                    cams_per_prompt = []

                    for w in prompt_list:
                        _, cam, _, _ = compute_attributions(model, img_in, w)
                        if isinstance(cam, np.ndarray):
                            cam = torch.from_numpy(cam).to(DEVICE)
                        if cam.dim() == 3:
                            cam = cam[0]

                        cam = blur_transform(cam.unsqueeze(0)).squeeze(0)
                        cams_per_prompt.append(cam)

                    class_cam = torch.mean(torch.stack(cams_per_prompt), dim=0)
                    class_cams.append(class_cam)

            cams = torch.stack(class_cams, dim=0)

            g_min = cams.min()
            g_max = cams.max()
            cams_norm = (cams - g_min) / (g_max - g_min + 1e-8)

            probs = F.softmax(cams_norm * temperature, dim=0)

            if flip:
                probs = torch.flip(probs, dims=[2])

            probs_resized = F.interpolate(
                probs.unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            all_prob_maps.append(probs_resized)

    final_probs = torch.mean(torch.stack(all_prob_maps), dim=0)
    final_probs_np = final_probs.clamp(0.0, 1.0).cpu().numpy()

    return {cls_id: final_probs_np[idx] for idx, cls_id in enumerate(classes)}


# ---------------------------------------------------------------------------
# CRF + MORPHOLOGY
# ---------------------------------------------------------------------------


def apply_crf_on_soft_prob(image_np, prob_map):
    if not HAS_CRF:
        return (prob_map > 0.5).astype(np.uint8)

    h, w = image_np.shape[:2]
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    prob = np.clip(prob_map, 1e-4, 1 - 1e-4)
    U = np.stack([1 - prob, prob], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)


def refine_binary_mask(mask):
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


# ---------------------------------------------------------------------------
# IOU
# ---------------------------------------------------------------------------


def calculate_iou(pred, gt):
    inter = np.sum((pred == 1) & (gt == 1))
    union = np.sum((pred == 1) | (gt == 1))
    return 1.0 if union == 0 else inter / union


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------


class PascalVocValidation(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_np = np.array(target)
        unique = np.unique(target_np)
        valid = unique[(unique != 0) & (unique != 255)]

        if len(valid) > 0:
            label_idx = int(valid[0])
            label_name = VOC_CLASSES[label_idx]
            present = [int(c) for c in valid.tolist()]
        else:
            label_idx = -1
            label_name = "none"
            present = []

        return img, target, label_idx, label_name, present


# ---------------------------------------------------------------------------
# VISUALIZATION (ADDED)
# ---------------------------------------------------------------------------


def save_visual(idx, img_np, prob, mask, gt, class_name):

    h, w = img_np.shape[:2]
    prob_viz = cv2.resize(prob, (w, h))

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(img_np)
    plt.title(f"Original ({class_name})")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(prob_viz, cmap="jet", vmin=0, vmax=1)
    plt.title("Prob Map (Multiclass)")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask (CRF + Morph)")
    plt.axis("off")

    overlay = img_np.copy()
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    plt.subplot(1, 5, 4)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(gt, cmap="gray")
    plt.title("GT")
    plt.axis("off")

    out_path = os.path.join(DEBUG_DIR, f"{idx}_{class_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_path}")


# ---------------------------------------------------------------------------
# BENCHMARK WITH 10 VISUALIZATIONS
# ---------------------------------------------------------------------------
def custom_collate(batch):
    return batch[0]


def preprocess_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )


def run_benchmark():
    print(f"Running Benchmark on {DEVICE.upper()}")

    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()

    clip_model, _ = load_clip_for_text()
    clip_model = clip_model.to(DEVICE).eval()

    class_prompts_cache = {}

    dataset = PascalVocValidation(
        root="./data", year="2012", image_set="val", download=False
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate,
    )
    max_images = 10  # 100
    # pick 10 random indices
    viz_indices = set(random.sample(range(max_images), 10))
    print(f"==> Visualizing indices: {viz_indices}")

    ious = []

    prep = preprocess_transform()

    for idx, data in enumerate(tqdm(loader, total=max_images)):
        if idx >= max_images:
            break

        img_pil, target_pil, label_idx, label_name, present_classes = data
        if label_idx == -1:
            continue

        img_tensor = prep(img_pil)

        # prepare embeddings
        class_to_weights = {}
        for cls in present_classes:
            if cls == 0:
                continue
            if cls not in class_prompts_cache:
                class_prompts_cache[cls] = get_class_prompt_embeddings(
                    clip_model, VOC_CLASSES[cls], DEVICE
                )
            class_to_weights[cls] = class_prompts_cache[cls]

        if len(class_to_weights) == 0:
            continue

        try:
            maps = process_image_multi_class(bcos_model, img_tensor, class_to_weights)
            prob_map = maps[label_idx]

            img_np = np.array(img_pil)
            mask = apply_crf_on_soft_prob(img_np, prob_map)
            mask = refine_binary_mask(mask)

            gt = (np.array(target_pil) == label_idx).astype(np.uint8)

            ious.append(calculate_iou(mask, gt))

            # visualize if needed
            if idx in viz_indices:
                save_visual(idx, img_np, prob_map, mask, gt, label_name)

        except RuntimeError as e:
            print(f"Error on image {idx}: {e}")
            torch.cuda.empty_cache()
            continue

    print(f"Final mIoU: {np.mean(ious) * 100:.2f}%")


if __name__ == "__main__":
    run_benchmark()
