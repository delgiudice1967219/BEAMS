import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

# Add paths for local modules
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# Try importing pydensecrf
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_labels,
        create_pairwise_bilateral,
        create_pairwise_gaussian,
    )

    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print(
        "Warning: pydensecrf not found. DenseCRF refinement will be skipped. Please install it via `pip install pydensecrf`."
    )


def preprocess_image(image, target_size=224):
    """
    Preprocess image for B-cos model.
    Resizes smaller edge to target_size, maintaining aspect ratio.
    """
    w, h = image.size
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    def _convert_image_to_rgb(img):
        return img.convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )

    return transform(image), (new_w, new_h)


def generate_segmentation_mask(
    model, clip_model, image, class_name, synonyms=None, alpha=0.5, device=None
):
    """
    Generates a raw segmentation mask using Synonym Fusion and Background Suppression.
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Preprocess Image
    input_tensor, (new_w, new_h) = preprocess_image(image)

    # 2. Synonym Fusion (Ensembling)
    templates = [
        "a photo of a {}",
        "the {}",
    ]

    prompts = [t.format(class_name) for t in templates]
    if synonyms:
        for syn in synonyms:
            prompts.append(f"a {syn}")

    # Also add "a {class_name}" as a fallback/variation
    prompts.append(f"a {class_name}")

    # Remove duplicates
    prompts = list(set(prompts))

    print(f"Computing maps for object prompts: {prompts}")

    object_maps = []
    for prompt in prompts:
        zeroshot_weight = tokenize_text_prompt(clip_model, prompt)
        _, contribs, _, _ = compute_attributions(model, input_tensor, zeroshot_weight)
        object_maps.append(contribs)

    # Aggregate object maps (Average)
    object_map = np.mean(object_maps, axis=0)

    # 3. Background Suppression (Negative Prompting)
    negative_prompts = ["background", "noise", "blur"]  # , "grass", "wall"
    print(f"Computing maps for negative prompts: {negative_prompts}")

    bg_maps = []
    for prompt in negative_prompts:
        zeroshot_weight = tokenize_text_prompt(clip_model, prompt)
        _, contribs, _, _ = compute_attributions(model, input_tensor, zeroshot_weight)
        bg_maps.append(contribs)

    # Aggregate background maps (Average)
    background_map = np.mean(bg_maps, axis=0)

    # Subtract background map
    # final_map = ReLU(object_map - alpha * background_map)
    final_map = object_map - alpha * background_map
    final_map = np.maximum(final_map, 0)  # ReLU

    return final_map, (new_w, new_h)


def apply_dense_crf_on_hard_mask(original_image_rgb, hard_mask):
    """
    MODIFIED: Refines a binary HARD MASK using DenseCRF to snap to image edges.
    This accepts a binary mask (0 or 1) instead of a probability heatmap.
    """
    if not HAS_CRF:
        print("DenseCRF not available, skipping refinement.")
        return hard_mask

    h, w = original_image_rgb.shape[:2]

    # Ensure hard_mask is same size and binary
    # Use Nearest Neighbor to keep it sharp/binary during resize
    hard_mask_resized = cv2.resize(hard_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    hard_mask_resized = hard_mask_resized.astype(np.float32)

    # --- UNARY POTENTIAL FROM HARD MASK ---
    # Convert binary mask to probabilities for CRF
    # Where mask is 1 (Object), we set high probability (0.9)
    # Where mask is 0 (Background), we set low probability (0.1)
    # This "anchors" the CRF to your existing good mask.
    prob_object = hard_mask_resized * 0.8 + 0.1

    # Create Energy: U = -log(P)
    U = np.stack([1.0 - prob_object, prob_object], axis=0)
    U = -np.log(U)
    U = U.reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)

    # --- PAIRWISE POTENTIALS ---
    # 1. Gaussian (Smoothness): Enforce local consistency
    d.addPairwiseGaussian(
        sxy=(3, 3),  # Small sxy to only fix immediate boundaries
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # 2. Bilateral (Appearance): Snap to color edges
    if original_image_rgb.dtype != np.uint8:
        original_image_rgb = (original_image_rgb * 255).astype(np.uint8)

    d.addPairwiseBilateral(
        sxy=(50, 50),  # Moderate range for color comparison
        srgb=(13, 13, 13),  # Color tolerance
        rgbim=original_image_rgb,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Inference
    Q = d.inference(5)
    map_soln = np.argmax(Q, axis=0).reshape((h, w))

    return map_soln.astype(np.float32)


def get_hard_mask(refined_heatmap, threshold_percent=0.2):
    """
    Generates a crisp binary mask from the heatmap.
    """
    # 1. Adaptive Thresholding
    max_val = refined_heatmap.max()
    thresh = max_val * threshold_percent
    binary_mask = (refined_heatmap > thresh).astype(np.uint8)

    # 2. Morphological Cleaning
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise (Opening)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Close small holes (Closing)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary_mask


def run_segmentation_emo(
    image_path, class_name, synonyms=None, output_dir="output_segmentation"
):
    """
    Runs the full segmentation pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {image_path} for class '{class_name}'...")

    # Load Models
    print("Loading models...")
    bcos_model, device = load_bcos_model()
    bcos_model = bcos_model.to(device)
    clip_model, _ = load_clip_for_text()

    # Load Image
    image = Image.open(image_path).convert("RGB")
    original_image_np = np.array(image)

    # 1. Generate Raw Map (with Synonym Fusion & Background Suppression)
    print("Generating B-cos Map...")
    raw_map, (h_map, w_map) = generate_segmentation_mask(
        bcos_model, clip_model, image, class_name, synonyms, device=device
    )

    # Resize raw map for visualization and Hard Mask generation
    raw_map_resized = cv2.resize(
        raw_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR
    )

    # 2. Hard Mask Generation (The "Good" Mask)
    # Since you confirmed this works well for bread, we use this as the anchor for CRF
    hard_mask_from_heatmap = get_hard_mask(raw_map_resized, threshold_percent=0.2)

    # 3. Refinement (DenseCRF on HARD MASK)
    print("Applying DenseCRF to refine Hard Mask edges...")
    # MODIFIED: Passing the HARD MASK instead of the raw map
    crf_mask = apply_dense_crf_on_hard_mask(original_image_np, hard_mask_from_heatmap)

    # Visualization
    plt.figure(figsize=(20, 5))

    # Original
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Raw B-cos Map (Refined with BG suppression)
    plt.subplot(1, 5, 2)
    plt.imshow(raw_map_resized, cmap="jet")
    plt.title("B-cos Map (BG Suppressed)")
    plt.axis("off")

    # Hard Mask (Adaptive Threshold)
    plt.subplot(1, 5, 3)
    plt.imshow(hard_mask_from_heatmap, cmap="gray")
    plt.title("Hard Mask (Base)")
    plt.axis("off")

    # CRF Mask
    plt.subplot(1, 5, 4)
    plt.imshow(crf_mask, cmap="gray")
    plt.title("DenseCRF (Edge Refined)")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 5, 5)
    plt.imshow(image)
    # Create overlay
    overlay = np.zeros_like(original_image_np)
    overlay[:, :, 1] = 255  # Green

    # Use CRF mask if available, else hard mask
    final_mask = crf_mask if HAS_CRF else hard_mask_from_heatmap

    mask_indices = final_mask > 0.5

    # Alpha blend
    alpha = 0.5
    blended = original_image_np.copy()
    blended[mask_indices] = (
        alpha * original_image_np[mask_indices] + (1 - alpha) * overlay[mask_indices]
    ).astype(np.uint8)

    plt.imshow(blended)

    # Draw contours
    contours, _ = cv2.findContours(
        (final_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) == 2:
            plt.plot(contour[:, 0], contour[:, 1], "r-", linewidth=2)

    plt.title("Final Overlay")
    plt.axis("off")

    save_path = os.path.join(output_dir, f"segmentation_{class_name}.png")
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")

    # Save binary mask
    mask_save_path = os.path.join(output_dir, f"mask_{class_name}.png")
    cv2.imwrite(mask_save_path, (final_mask * 255).astype(np.uint8))
    print(f"Binary mask saved to {mask_save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-Shot Segmentation with B-cos CLIP"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_images/bread-knife-pans-towel.png",
        help="Path to image",
    )
    parser.add_argument("--class_name", type=str, default="towel", help="Class name")
    parser.add_argument("--synonyms", type=str, nargs="*", help="List of synonyms")

    args = parser.parse_args()

    run_segmentation_emo(args.image, args.class_name, args.synonyms)
