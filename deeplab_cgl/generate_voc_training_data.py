import os
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from PIL import Image

# --- SETUP IMPORT ---
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_utils import (
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

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
SCALES = [448, 560]  # Multi-scale for better quality

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

# --- UTILS ---


def precompute_background_weights(clip_model, device):
    """Calculates background embeddings once."""
    print("Pre-computing background embeddings...")
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
    ]
    bg_prompts = [f"a photo of {bg}" for bg in bg_classes]

    with torch.no_grad():
        # Stack to keep (N, 1024)
        weights = torch.stack(
            [tokenize_text_prompt(clip_model, p) for p in bg_prompts]
        ).to(device)
    return torch.mean(weights, dim=0)


def get_target_weight(clip_model, class_name, device):
    prompts = [
        f"a clean origami {class_name}.",
        f"a photo of a {class_name}.",
        f"the {class_name}.",
    ]
    with torch.no_grad():
        weights = torch.stack(
            [tokenize_text_prompt(clip_model, p) for p in prompts]
        ).to(device)
    return torch.mean(weights, dim=0)


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


# --- CORE GENERATION LOGIC ---


def compute_raw_map(model, img_pil, weight):
    """
    Computes the raw attribution map for a specific weight (BG or Class).
    Averages over Scales and Flips (Test Time Augmentation).
    """
    base_w, base_h = img_pil.size
    accumulated_maps = []

    for s in SCALES:
        prep = get_transform(s)
        img_tens = prep(img_pil).to(DEVICE)  # [6, S, S]

        # 1. Original
        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                _, map_out, _, _ = compute_attributions(model, img_tens, weight)

        map_out = ensure_tensor(map_out)

        # Resize to original image size
        map_resized = F.interpolate(
            map_out.unsqueeze(0).unsqueeze(0),
            size=(base_h, base_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        accumulated_maps.append(map_resized)

        # 2. Flipped
        img_flip = torch.flip(img_tens, [2])
        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                _, map_out_f, _, _ = compute_attributions(model, img_flip, weight)

        map_out_f = ensure_tensor(map_out_f)
        map_out_f = torch.flip(map_out_f, [1])  # Un-flip

        map_resized_f = F.interpolate(
            map_out_f.unsqueeze(0).unsqueeze(0),
            size=(base_h, base_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        accumulated_maps.append(map_resized_f)

    # Average all scales/flips
    return torch.mean(torch.stack(accumulated_maps), dim=0)


def generate_multiclass_mask(model, clip_model, img_pil, active_classes, bg_weight):
    """
    Generates a dense mask for the image.
    Competes: Background vs Class A vs Class B...
    """
    blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    # 1. Compute Maps for everything (BG + Active Classes)
    maps = []
    class_indices = [0]  # 0 is Background

    # Background Map
    bg_map = compute_raw_map(model, img_pil, bg_weight)
    if bg_map.dim() == 2:
        bg_map = bg_map.unsqueeze(0)
    bg_map = blur_transform(bg_map).squeeze()
    maps.append(bg_map)

    # Active Classes Maps
    for cls_name in active_classes:
        cls_idx = VOC_CLASSES.index(cls_name)
        class_indices.append(cls_idx)

        # Get embedding
        w_target = get_target_weight(clip_model, cls_name, DEVICE)

        # Compute Map
        cls_map = compute_raw_map(model, img_pil, w_target)
        if cls_map.dim() == 2:
            cls_map = cls_map.unsqueeze(0)
        cls_map = blur_transform(cls_map).squeeze()
        maps.append(cls_map)

    # 2. Joint Normalization
    # We stack them to find global min/max
    stack = torch.stack(maps)  # [N_classes, H, W]

    g_min = stack.min()
    g_max = stack.max()
    denom = g_max - g_min + 1e-8

    stack_norm = (stack - g_min) / denom

    # 3. Softmax
    # Multiply by temperature (20) to sharpen
    probs = F.softmax(stack_norm * 20, dim=0)

    # 4. Argmax to get index (0, 1, 2...) corresponding to (BG, Class A, Class B...)
    pred_indices = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    # 5. Map back to Real VOC Class IDs
    # pred_indices contains 0, 1, 2... we need to map 1 -> 15 (Person), etc.
    final_mask = np.zeros_like(pred_indices)

    # Map array: index -> real_class_id
    lookup = np.array(class_indices, dtype=np.uint8)
    final_mask = lookup[pred_indices]

    # Get Max Prob for Confidence (needed for CGL later)
    confidence_map = torch.max(probs, dim=0).values.cpu().numpy()

    return final_mask, confidence_map


def apply_crf(image_np, mask, num_classes=21):
    if not HAS_CRF:
        return mask
    h, w = image_np.shape[:2]
    d = dcrf.DenseCRF2D(w, h, num_classes)

    unary = np.zeros((num_classes, h, w), dtype="float32")
    for c in range(num_classes):
        unary[c] = (mask == c).astype(float)

    unary[unary == 0] = 0.05
    unary[unary == 1] = 0.95
    U = -np.log(unary).reshape((num_classes, -1)).astype(np.float32)

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)


# --- DATASET (MODIFIED FOR TRAINING GEN) ---


class PascalVocTrain(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_np = np.array(target)

        # Extract unique classes present in this image (Oracle for WSSS)
        # We ignore 0 (BG) and 255 (Border) to find objects
        unique_classes = np.unique(target_np)
        valid_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]

        class_names = [VOC_CLASSES[i] for i in valid_classes]

        # Return filename too for saving
        img_name = self.images[index].split(os.sep)[-1].replace(".jpg", "")

        return img, class_names, img_name


def custom_collate(batch):
    return batch[0]  # No batching needed for generation


# --- MAIN ---


def run_generation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voc_root", type=str, default="./data", help="Root of VOC dataset"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="voc_bcos_pseudolabels",
        help="Where to save masks",
    )
    args = parser.parse_args()

    # Directories (NO JPEGImages anymore)
    mask_out = os.path.join(args.out_dir, "SegmentationClass")
    conf_out = os.path.join(args.out_dir, "Confidence")

    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(conf_out, exist_ok=True)

    print(f"Loading Models on {DEVICE}...")
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    bg_weight = precompute_background_weights(clip_model, DEVICE)

    print("Initializing VOC Train Set...")
    try:
        # Use image_set="train" for generating training data
        dataset = PascalVocTrain(
            root=args.voc_root, year="2012", image_set="train", download=False
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Generating Pseudo-Labels for {len(dataset)} images...")

    for i, (img_pil, active_classes, img_name) in enumerate(tqdm(loader)):

        # --- RESUME LOGIC (SKIP IF EXISTS) ---
        out_mask_path = os.path.join(mask_out, f"{img_name}.png")
        out_conf_path = os.path.join(conf_out, f"{img_name}.npy")

        if os.path.exists(out_mask_path) and os.path.exists(out_conf_path):
            continue
        # -------------------------------------

        if len(active_classes) == 0:
            # Only background? Just save empty mask
            mask = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)
            cv2.imwrite(out_mask_path, mask)
            # Skip saving image copy
            continue

        try:
            img_np = np.array(img_pil)

            # 1. Generate Raw Mask & Confidence
            raw_mask, conf_map = generate_multiclass_mask(
                bcos_model, clip_model, img_pil, active_classes, bg_weight
            )

            # 2. Refine with CRF
            final_mask = apply_crf(img_np, raw_mask)

            # 3. Save Data
            # Save Mask (PNG)
            cv2.imwrite(out_mask_path, final_mask)

            # Save Confidence (NPY) - Needed for CGL Loss
            np.save(out_conf_path, conf_map)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            torch.cuda.empty_cache()
            continue

    print("Generation Complete.")


if __name__ == "__main__":
    run_generation()
