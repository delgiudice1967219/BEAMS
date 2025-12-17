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

try:
    import pydensecrf.densecrf as dcrf

    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: DenseCRF not found.")

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
# 1. CORE FUNCTIONS (Weights & Softmax)
# ==============================================================================


def precompute_background_weights(clip_model, device):
    # Full CLIP-ES background list for better suppression
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
        "road",
        "rock",
        "street",
        "cloud",
        "mountain",
        "floor",
        "ceiling",
        "background",
        "blur",
        "person",
        "man",
        "woman",
        "face",
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


def get_bcos_softmax_map(model, img_tensor, target_w, bg_w):
    """Returns Softmax probability map (Target vs Background)"""
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

        # Blur & Joint Norm
        blur = transforms.GaussianBlur(5, 1.0)
        mt = blur(mt.unsqueeze(0)).squeeze()
        mb = blur(mb.unsqueeze(0)).squeeze()

        g_min = min(mt.min(), mb.min())
        g_max = max(mt.max(), mb.max())
        denom = g_max - g_min + 1e-8
        mt_n = (mt - g_min) / denom
        mb_n = (mb - g_min) / denom

        # Softmax Competition
        stack = torch.stack([mb_n, mt_n], dim=0)
        probs = F.softmax(stack * 20, dim=0)
        return probs[1, :, :].cpu().numpy()


# ==============================================================================
# 2. CRF & MASK UTILS
# ==============================================================================


def get_hard_mask(prob_map, h, w, threshold=0.5):
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob_map > threshold).astype(np.uint8)
    # Aggressive cleaning to create solid blobs
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def apply_crf(image_np, hard_mask):
    if not HAS_CRF:
        return hard_mask
    h, w = image_np.shape[:2]
    if hard_mask.shape != (h, w):
        hard_mask = cv2.resize(hard_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    prob = hard_mask.astype(np.float32) * 0.9 + 0.05
    U = np.stack([1.0 - prob, prob], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(60, 60), srgb=(10, 10, 10), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


# ==============================================================================
# 3. HYBRID SOLUTION: RBA-CRF
# ==============================================================================


def extract_focus_box_v2(prob_map, h, w):
    """Finds the bounding box using aggregation logic"""
    pmap = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pmap_connected = cv2.morphologyEx(pmap, cv2.MORPH_CLOSE, kernel_connect)

    max_val = pmap_connected.max()
    if max_val < 0.1:
        return (0, 0, w, h)

    thresh_val = max_val * 0.2
    binary = (pmap_connected > thresh_val).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return (0, 0, w, h)

    min_area = (h * w) * 0.005
    valid_boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            valid_boxes.append(cv2.boundingRect(c))

    if not valid_boxes:
        c = max(contours, key=cv2.contourArea)
        valid_boxes.append(cv2.boundingRect(c))

    x1_min = min([b[0] for b in valid_boxes])
    y1_min = min([b[1] for b in valid_boxes])
    x2_max = max([b[0] + b[2] for b in valid_boxes])
    y2_max = max([b[1] + b[3] for b in valid_boxes])

    bw, bh = x2_max - x1_min, y2_max - y1_min
    pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)

    return (
        max(0, x1_min - pad_x),
        max(0, y1_min - pad_y),
        min(w, x2_max + pad_x),
        min(h, y2_max + pad_y),
    )


def recursive_zoom_inference(model, clip_model, img_pil, class_name, bg_w):
    """
    The Hybrid Pipeline:
    1. Global Glance -> Find Box
    2. Zoom -> Run Softmax-CRF Pipeline on Crop
    3. Paste result back into global canvas
    """
    w_orig, h_orig = img_pil.size
    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    target_w = get_target_weight(clip_model, class_name, DEVICE)

    # --- STEP 1: GLOBAL GLANCE & BOX ---
    img_tensor_global = prep(img_pil).unsqueeze(0).to(DEVICE)
    # Use Softmax for global map as it's cleaner for box finding
    global_map = get_bcos_softmax_map(
        model, img_tensor_global.squeeze(0), target_w, bg_w
    )
    box = extract_focus_box_v2(global_map, h_orig, w_orig)
    x1, y1, x2, y2 = box

    # --- STEP 2: ZOOM & REFINED LOCAL SEGMENTATION ---
    img_crop = img_pil.crop((x1, y1, x2, y2))
    crop_w, crop_h = img_crop.size

    # A. Process crop through network
    img_tensor_crop = prep(img_crop).unsqueeze(0).to(DEVICE)
    local_prob_map = get_bcos_softmax_map(
        model, img_tensor_crop.squeeze(0), target_w, bg_w
    )

    # B. Run CRF Pipeline LOCALLY on the crop
    # Use a lower threshold (0.4) for local map as it's already zoomed
    local_hard_mask = get_hard_mask(local_prob_map, crop_h, crop_w, threshold=0.4)
    local_crf_mask = apply_crf(np.array(img_crop), local_hard_mask)

    # --- STEP 3: PASTE BACK ---
    final_mask_full = np.zeros((h_orig, w_orig), dtype=np.uint8)
    final_mask_full[y1:y2, x1:x2] = local_crf_mask

    return final_mask_full


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
    print(f"🚀 Avvio Benchmark: Hybrid RBA-CRF Solution")

    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    bg_w = precompute_background_weights(clip_model, DEVICE)

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
            # RUN HYBRID PIPELINE
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
        print(f"🏆 mIoU FINALE (Hybrid RBA-CRF): {np.mean(ious)*100:.2f}%")
        print(f"==========================================")


if __name__ == "__main__":
    run_benchmark()
