import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

# ------------------------------------------------------------------
# SETUP IMPORT
# ------------------------------------------------------------------
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
    print("Warning: DenseCRF not found. CRF disabled.")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

# Hardcoded image path as per request
IMG_PATH = r"C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005751.jpg"

VOC_ACTIONS = [
    "jumping",
    "phoning",
    "playinginstrument",
    "reading",
    "ridingbike",
    "ridinghorse",
    "running",
    "takingphoto",
    "usingcomputer",
    "walking",
]

OUTPUT_DIR = "single_image_viz_actions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# TEXT EMBEDDINGS
# ------------------------------------------------------------------
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
        "road",
        "rock",
        "street",
        "cloud",
        "mountain",
        "floor",
        "ceiling",
        "background",
        "blur",
        "person jumping",
    ]
    weights = []
    with torch.no_grad():
        for bg in bg_classes:
            w = tokenize_text_prompt(clip_model, f"a photo of {bg}").to(device)
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


# ------------------------------------------------------------------
# CORE BCOS PIPELINE (MULTI-SCALE + FLIP)
# ------------------------------------------------------------------
def process_single_image_softmax(model, image_tensor, target_weight, bg_weight):
    scales = [448, 560]
    accumulated_maps = []

    base_h, base_w = IMG_SIZE, IMG_SIZE
    img_cpu = image_tensor.squeeze()  # [6,H,W]

    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    for s in scales:
        resize = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )

        img_scaled = resize(img_cpu).unsqueeze(0).to(DEVICE)  # [1,6,s,s]

        # ==================================================
        # NORMAL
        # ==================================================
        model_input = img_scaled.squeeze(0)

        with torch.no_grad():
            _, map_t, _, _ = compute_attributions(model, model_input, target_weight)
            _, map_b, _, _ = compute_attributions(model, model_input, bg_weight)

            if isinstance(map_t, np.ndarray):
                map_t = torch.from_numpy(map_t).to(DEVICE)
            if isinstance(map_b, np.ndarray):
                map_b = torch.from_numpy(map_b).to(DEVICE)

            if map_t.dim() == 3:
                map_t = map_t[0]
            if map_b.dim() == 3:
                map_b = map_b[0]

            map_t = blur(map_t.unsqueeze(0)).squeeze()
            map_b = blur(map_b.unsqueeze(0)).squeeze()

            g_min = min(map_t.min(), map_b.min())
            g_max = max(map_t.max(), map_b.max())
            denom = g_max - g_min + 1e-8

            map_t = (map_t - g_min) / denom
            map_b = (map_b - g_min) / denom

            probs = F.softmax(torch.stack([map_b, map_t]) * 20, dim=0)
            target_prob = probs[1].unsqueeze(0).unsqueeze(0)

            prob_resized = F.interpolate(
                target_prob,
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            accumulated_maps.append(prob_resized)

        # ==================================================
        # FLIP
        # ==================================================
        img_flipped = torch.flip(img_scaled, dims=[3])
        model_input_f = img_flipped.squeeze(0)

        with torch.no_grad():
            _, map_t_f, _, _ = compute_attributions(model, model_input_f, target_weight)
            _, map_b_f, _, _ = compute_attributions(model, model_input_f, bg_weight)

            if isinstance(map_t_f, np.ndarray):
                map_t_f = torch.from_numpy(map_t_f).to(DEVICE)
            if isinstance(map_b_f, np.ndarray):
                map_b_f = torch.from_numpy(map_b_f).to(DEVICE)

            if map_t_f.dim() == 3:
                map_t_f = map_t_f[0]
            if map_b_f.dim() == 3:
                map_b_f = map_b_f[0]

            map_t_f = blur(map_t_f.unsqueeze(0)).squeeze()
            map_b_f = blur(map_b_f.unsqueeze(0)).squeeze()

            g_min_f = min(map_t_f.min(), map_b_f.min())
            g_max_f = max(map_t_f.max(), map_b_f.max())
            denom_f = g_max_f - g_min_f + 1e-8

            map_t_f = (map_t_f - g_min_f) / denom_f
            map_b_f = (map_b_f - g_min_f) / denom_f

            probs_f = F.softmax(torch.stack([map_b_f, map_t_f]) * 20, dim=0)
            target_prob_f = probs_f[1]

            # UNFLIP
            target_prob_f = torch.flip(target_prob_f, dims=[1])
            target_prob_f = target_prob_f.unsqueeze(0).unsqueeze(0)

            prob_resized_f = F.interpolate(
                target_prob_f,
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            accumulated_maps.append(prob_resized_f)

    return torch.mean(torch.stack(accumulated_maps), dim=0).cpu().numpy()


# ------------------------------------------------------------------
# POST-PROCESS
# ------------------------------------------------------------------
def get_hard_mask(prob_map, h, w):
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob_map > 0.9).astype(np.uint8)


def apply_crf(image_np, hard_mask):
    if not HAS_CRF:
        return hard_mask

    h, w = image_np.shape[:2]
    prob = hard_mask.astype(np.float32) * 0.9 + 0.05
    U = np.stack([1 - prob, prob], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


def run_visualization(
    action, bcos_model, clip_model, prep, img_np, img_tensor, bg_weight
):
    target_class_text = f"person {action}"
    print(f"Processing action: {target_class_text}")

    target_weight = get_target_weight(clip_model, target_class_text, DEVICE)

    prob_map = process_single_image_softmax(
        bcos_model, img_tensor, target_weight, bg_weight
    )

    hard_mask = get_hard_mask(prob_map, img_np.shape[0], img_np.shape[1])
    final_mask = apply_crf(img_np, hard_mask)

    # ---------------- VISUAL ----------------
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title(f"Original / {action}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    prob_viz = cv2.resize(prob_map, (img_np.shape[1], img_np.shape[0]))
    plt.imshow(prob_viz, cmap="jet", vmin=0, vmax=1)
    plt.title("Softmax Heatmap")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(hard_mask * 255, cmap="gray")
    plt.title("Hard Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    overlay = img_np.copy()
    alpha = 0.5
    overlay[final_mask == 1] = (
        overlay[final_mask == 1] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    ).astype(np.uint8)
    plt.imshow(overlay)
    plt.title("Final Output (CRF)")
    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"single_image_result_{action}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # Close memory to save plot
    print(f"Saved visualization to {save_path}")


# ------------------------------------------------------------------
# MAIN VISUALIZATION
# ------------------------------------------------------------------
def main():
    print("Loading models...")
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()

    clip_model, _ = load_clip_for_text()
    clip_model = clip_model.to(DEVICE).eval()

    bg_weight = precompute_background_weights(clip_model, DEVICE)

    img_pil = Image.open(IMG_PATH).convert("RGB")
    img_np = np.array(img_pil)

    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    img_tensor = prep(img_pil)

    for action in VOC_ACTIONS:
        run_visualization(
            action, bcos_model, clip_model, prep, img_np, img_tensor, bg_weight
        )


if __name__ == "__main__":
    main()
