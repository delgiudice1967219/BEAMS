import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

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

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH_SIZE = 1
NUM_WORKERS = 0

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

# ==============================================================================
# 1. CORE B-COS FUNCTIONS
# ==============================================================================


def precompute_background_weights(clip_model, device):
    bg_classes = [
        "ground",
        "wall",
        "sky",
        "floor",
        "ceiling",
        "background",
        "blur",
        "person",
        "hands",
    ]
    bg_prompts = [f"a photo of {bg}" for bg in bg_classes]
    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_target_weight(clip_model, class_name, device):
    prompts = [f"a photo of a {class_name}.", f"the {class_name}."]
    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_bcos_map(model, img_tensor, target_w, bg_w):
    """Returns raw normalized difference map"""
    with torch.no_grad():
        _, mt, _, _ = compute_attributions(model, img_tensor, target_w)
        _, mb, _, _ = compute_attributions(model, img_tensor, bg_w)

        if isinstance(mt, np.ndarray):
            mt = torch.from_numpy(mt).to(DEVICE)
        if isinstance(mb, np.ndarray):
            mb = torch.from_numpy(mb).to(DEVICE)
        if mt.dim() == 3:
            mt = mt[0]
        if mb.dim() == 3:
            mb = mb[0]

        # Blur
        blur = transforms.GaussianBlur(5, 1.0)
        mt = blur(mt.unsqueeze(0)).squeeze()
        mb = blur(mb.unsqueeze(0)).squeeze()

        # Norm
        def rn(t):
            v_min, v_max = t.min(), torch.quantile(t, 0.99)
            if v_max - v_min < 1e-6:
                return torch.zeros_like(t)
            return torch.clamp((t - v_min) / (v_max - v_min), 0, 1)

        diff = rn(mt) - rn(mb)
        return torch.relu(diff).cpu().numpy()


# ==============================================================================
# 2. NOVELTY: RECURSIVE BOX ATTENTION (RBA)
# ==============================================================================


def extract_focus_box(prob_map, h, w):
    """Finds the bounding box of the active region in the global map"""
    # Resize map to image size
    pmap = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Binarize with low threshold to catch the whole object extent
    binary = (pmap > 0.2).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get the largest contour (main object)
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    # Add Padding (Context) - 20%
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.2)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return (x1, y1, x2, y2)


def recursive_zoom_inference(model, clip_model, img_pil, class_name, bg_w):
    """
    The Core Innovation:
    1. Look Globally -> Find Box
    2. Zoom into Box -> Look Locally (High Res)
    3. Fuse Global and Local attention
    """
    w_orig, h_orig = img_pil.size

    # Prep inputs
    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )

    # Get Weights
    target_w = get_target_weight(clip_model, class_name, DEVICE)

    # --- STEP 1: GLOBAL GLANCE ---
    img_tensor_global = prep(img_pil).unsqueeze(0).to(DEVICE)
    global_map = get_bcos_map(model, img_tensor_global.squeeze(0), target_w, bg_w)

    # Resize global map to original image size
    global_map_full = cv2.resize(
        global_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
    )

    # --- STEP 2: FIND BOX ---
    box = extract_focus_box(global_map_full, h_orig, w_orig)

    if box is None:
        # If nothing found globally, return empty mask
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    x1, y1, x2, y2 = box

    # --- STEP 3: ZOOM IN ---
    # Crop the original image (High Res Crop)
    img_crop = img_pil.crop((x1, y1, x2, y2))

    # Process Crop
    img_tensor_crop = prep(img_crop).unsqueeze(0).to(DEVICE)
    local_map = get_bcos_map(model, img_tensor_crop.squeeze(0), target_w, bg_w)

    # Resize local map to the size of the crop (NOT the full image yet)
    crop_w, crop_h = x2 - x1, y2 - y1
    local_map_resized = cv2.resize(
        local_map, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
    )

    # --- STEP 4: FUSION ---
    # Place the local map back into the full image canvas
    local_map_full = np.zeros((h_orig, w_orig), dtype=np.float32)
    local_map_full[y1:y2, x1:x2] = local_map_resized

    # Intersection: Pixel must be active in GLOBAL context AND LOCAL detail
    # We weigh Local more (0.7) because it has better resolution
    final_heatmap = (global_map_full * 0.4) + (local_map_full * 0.6)

    # Final Hard Mask
    return (final_heatmap > 0.35).astype(np.uint8)


# --- STANDARD IOU & DATASET ---


def calculate_iou(pred, gt):
    inter = np.sum((pred == 1) & (gt == 1))
    union = np.sum((pred == 1) | (gt == 1))
    return 1.0 if union == 0 else inter / union


class PascalVocValidation(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_np = np.array(target)
        unique = np.unique(target_np)
        valid = unique[(unique != 0) & (unique != 255)]
        label_idx = valid[0] if len(valid) > 0 else -1
        label_name = VOC_CLASSES[label_idx] if label_idx != -1 else "none"
        return img, target, label_idx, label_name


def custom_collate(batch):
    return batch[0]


# --- MAIN ---


def run_benchmark():
    print(f"🚀 Avvio Benchmark: Recursive Box-Attention (Custom Solution)")

    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    bg_w = precompute_background_weights(clip_model, DEVICE)
    target_cache = {}

    try:
        val_dataset = PascalVocValidation(
            root="./data", year="2012", image_set="val", download=False
        )
        loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate,
        )
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    ious = []
    max_images = 100

    print(f"Test su {max_images} immagini...")

    for i, data in enumerate(tqdm(loader, total=max_images)):
        if i >= max_images:
            break

        img_pil, target_pil, label_idx, label_name = data
        if label_idx == -1:
            continue

        try:
            # RUN NOVEL PIPELINE
            final_mask = recursive_zoom_inference(
                bcos_model, clip_model, img_pil, label_name, bg_w
            )

            # IoU
            gt_mask = np.array(target_pil)
            gt_binary = (gt_mask == label_idx).astype(np.uint8)
            iou = calculate_iou(final_mask, gt_binary)
            ious.append(iou)

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            continue

    if len(ious) > 0:
        print(f"\n==========================================")
        print(f"🏆 mIoU FINALE (Recursive Box-Attention): {np.mean(ious)*100:.2f}%")
        print(f"==========================================")


if __name__ == "__main__":
    run_benchmark()
