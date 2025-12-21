import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

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

# --- PARAMETRI ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
OUTPUT_DIR = "segmentation_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CLASSI PASCAL VOC ---
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

# ==============================================================================
# 1. FUNZIONI CORE (Copia incolla dal Benchmark Ottimizzato)
# ==============================================================================


def precompute_background_weights(clip_model, device):
    """Calcola i vettori dello sfondo una volta sola"""
    print("Pre-calcolo embeddings sfondo...")
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
    identity_template = ["{}"]
    bg_prompts = [f"a photo of {bg}" for bg in bg_classes]

    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p, templates=identity_template).to(
                device
            )
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_target_weight(clip_model, class_name, device):
    """Calcola il vettore del target"""
    prompts = [
        f"a clean origami {class_name}.",
        f"a photo of a {class_name}.",
        f"a photo of the {class_name}.",
        f"a picture of a {class_name}.",
        f"an image of a {class_name}.",
        f"an image of the {class_name}.",
        f"the {class_name}.",
    ]
    identity_template = ["{}"]
    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p, templates=identity_template).to(
                device
            )
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


# def process_single_image_softmax(model, image_tensor, target_weight, bg_weight):
#     """
#     Versione FIXATA 2.1: Joint Normalization + Gaussian Blur (Torchvision).
#     """
#     scales = [448, 560]  # 2 scale per velocità
#     accumulated_maps = []

#     # Input base
#     original_img_cpu = image_tensor.squeeze()
#     base_h, base_w = IMG_SIZE, IMG_SIZE

#     # Istanza del Blur standard di Torchvision
#     blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

#     for s in scales:
#         resize_transform = transforms.Resize(
#             (s, s), interpolation=transforms.InterpolationMode.BICUBIC
#         )
#         img_scaled_4d = resize_transform(original_img_cpu).unsqueeze(0).to(DEVICE)
#         model_input = img_scaled_4d.squeeze(0)

#         with torch.no_grad():
#             # 1. Calcolo Mappe Grezze
#             _, map_t, _, _ = compute_attributions(model, model_input, target_weight)
#             _, map_b, _, _ = compute_attributions(model, model_input, bg_weight)

#             if isinstance(map_t, np.ndarray):
#                 map_t = torch.from_numpy(map_t).to(DEVICE)
#             if isinstance(map_b, np.ndarray):
#                 map_b = torch.from_numpy(map_b).to(DEVICE)

#             # Assicuriamoci siano 2D [H, W] per trovare min/max
#             if map_t.dim() == 3:
#                 map_t = map_t[0]
#             if map_b.dim() == 3:
#                 map_b = map_b[0]

#             # --- FIX 1: GAUSSIAN BLUR (Con Torchvision) ---
#             # Il blur vuole [C, H, W], aggiungiamo dim fittizia
#             map_t = blur_transform(map_t.unsqueeze(0)).squeeze()
#             map_b = blur_transform(map_b.unsqueeze(0)).squeeze()

#             # --- FIX 2: JOINT NORMALIZATION (Cruciale per evitare heatmap verde) ---
#             # Normalizziamo insieme usando il range globale
#             g_min = min(map_t.min(), map_b.min())
#             g_max = max(map_t.max(), map_b.max())

#             denom = g_max - g_min + 1e-8
#             map_t_norm = (map_t - g_min) / denom
#             map_b_norm = (map_b - g_min) / denom

#             # 2. Softmax
#             # Moltiplicatore alto (20) per separare i picchi
#             stack = torch.stack([map_b_norm, map_t_norm], dim=0)
#             probs = F.softmax(stack * 20, dim=0)

#             target_prob = probs[1, :, :].unsqueeze(0).unsqueeze(0)

#             # Resize e accumulo
#             prob_resized = F.interpolate(
#                 target_prob, size=(base_h, base_w), mode="bilinear", align_corners=False
#             ).squeeze()
#             accumulated_maps.append(prob_resized)

#     final_prob_map = torch.mean(torch.stack(accumulated_maps), dim=0)
#     return final_prob_map.cpu().numpy()


def process_single_image_softmax(model, image_tensor, target_weight, bg_weight):
    scales = [224, 448, 560, 672, 784]  # Multi-Scale
    accumulated_maps = []

    # Input base
    original_img_cpu = image_tensor.squeeze()
    base_h, base_w = IMG_SIZE, IMG_SIZE

    # Istanza del Blur standard di Torchvision
    blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    for s in scales:
        resize_transform = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )
        # Prepare 4D Tensor [1, 6, S, S]
        img_scaled_4d = resize_transform(original_img_cpu).unsqueeze(0).to(DEVICE)

        # Prepare 3D input for model (assuming model adds batch dim internally if needed or handles 3D)
        model_input = img_scaled_4d.squeeze(0)

        with torch.no_grad():
            # 1. Calcolo Mappe Grezze
            _, map_t, _, _ = compute_attributions(model, model_input, target_weight)
            _, map_b, _, _ = compute_attributions(model, model_input, bg_weight)

            if isinstance(map_t, np.ndarray):
                map_t = torch.from_numpy(map_t).to(DEVICE)
            if isinstance(map_b, np.ndarray):
                map_b = torch.from_numpy(map_b).to(DEVICE)

            # Assicuriamoci siano 2D [H, W] per trovare min/max
            if map_t.dim() == 3:
                map_t = map_t[0]
            if map_b.dim() == 3:
                map_b = map_b[0]

            # Il blur vuole [C, H, W], aggiungiamo dim fittizia
            map_t = blur_transform(map_t.unsqueeze(0)).squeeze()
            map_b = blur_transform(map_b.unsqueeze(0)).squeeze()

            # Normalizziamo insieme usando il range globale
            g_min = min(map_t.min(), map_b.min())
            g_max = max(map_t.max(), map_b.max())

            denom = g_max - g_min + 1e-8
            map_t_norm = (map_t - g_min) / denom
            map_b_norm = (map_b - g_min) / denom

            # 2. Softmax
            # Moltiplicatore alto (20) per separare i picchi
            stack = torch.stack([map_b_norm, map_t_norm], dim=0)
            probs = F.softmax(stack * 20, dim=0)

            # Target channel is 1
            target_prob = probs[1, :, :].unsqueeze(0).unsqueeze(0)

            # Resize e accumulo
            prob_resized = F.interpolate(
                target_prob, size=(base_h, base_w), mode="bilinear", align_corners=False
            ).squeeze()
            accumulated_maps.append(prob_resized)

            # --- FLIP ---
            img_flipped_4d = torch.flip(img_scaled_4d, [3])
            model_input_flip = img_flipped_4d.squeeze(0)

            _, map_t_f, _, _ = compute_attributions(
                model, model_input_flip, target_weight
            )
            _, map_b_f, _, _ = compute_attributions(model, model_input_flip, bg_weight)

            if isinstance(map_t_f, np.ndarray):
                map_t_f = torch.from_numpy(map_t_f).to(DEVICE)
            if isinstance(map_b_f, np.ndarray):
                map_b_f = torch.from_numpy(map_b_f).to(DEVICE)

            if map_t_f.dim() == 3:
                map_t_f = map_t_f[0]
            if map_b_f.dim() == 3:
                map_b_f = map_b_f[0]

            map_t_f = blur_transform(map_t_f.unsqueeze(0)).squeeze()
            map_b_f = blur_transform(map_b_f.unsqueeze(0)).squeeze()

            g_min_f = min(map_t_f.min(), map_b_f.min())
            g_max_f = max(map_t_f.max(), map_b_f.max())
            denom_f = g_max_f - g_min_f + 1e-8

            map_t_f_norm = (map_t_f - g_min_f) / denom_f
            map_b_f_norm = (map_b_f - g_min_f) / denom_f

            stack_f = torch.stack([map_b_f_norm, map_t_f_norm], dim=0)
            probs_f = F.softmax(stack_f * 20, dim=0)
            target_prob_f = probs_f[1, :, :]

            target_prob_unflipped = torch.flip(target_prob_f, [1])
            prob_resized_f = F.interpolate(
                target_prob_unflipped.unsqueeze(0).unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            accumulated_maps.append(prob_resized_f)

    final_prob_map = torch.mean(torch.stack(accumulated_maps), dim=0)
    return final_prob_map.cpu().numpy()


# ==============================================================================
# 2. POST-PROCESSING
# ==============================================================================


def get_hard_mask(prob_map, h, w):
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    # Threshold 0.6 come richiesto nel "nuovo" script visuale
    binary = (prob_map > 0.6).astype(np.uint8)
    return binary


def apply_crf_on_hard_mask(image_np, hard_mask):
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
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


# ==============================================================================
# 3. PIPELINE & VISUALIZATION
# ==============================================================================


def quick_pipeline(img_pil, class_name, bcos_model, clip_model, bg_weight):
    original_np = np.array(img_pil)

    # Prepare Input
    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    img_tensor = prep(img_pil)  # [6, 512, 512]

    # Target Weights
    target_weight = get_target_weight(clip_model, class_name, DEVICE)

    # Inference
    prob_map = process_single_image_softmax(
        bcos_model, img_tensor, target_weight, bg_weight
    )

    # Post-Process
    hard_mask = get_hard_mask(prob_map, img_pil.height, img_pil.width)
    final_mask = apply_crf_on_hard_mask(original_np, hard_mask)

    return original_np, prob_map, hard_mask, final_mask


def custom_collate(batch):
    return batch[0]


class PascalVocValidation(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_np = np.array(target)
        unique_classes = np.unique(target_np)
        valid_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]

        if len(valid_classes) > 0:
            label_idx = valid_classes[0]
            label_name = VOC_CLASSES[label_idx]
        else:
            label_idx = -1
            label_name = "none"
        return img, target, label_idx, label_name


def run_visual_test(num_samples=10):
    print(f"🚀 Generazione visualizzazioni su {num_samples} immagini VOC...")

    # 1. Carica Modelli
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    bg_weight = precompute_background_weights(clip_model, DEVICE)

    # 2. Carica Dataset VOC
    try:
        dataset = PascalVocValidation(
            root="./data", year="2012", image_set="train", download=False
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate,
        )
    except Exception as e:
        print(f"Errore dataset: {e}")
        return

    # 3. Seleziona 10 indici casuali
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        img_pil, target_pil, class_idx, class_name = dataset[idx]

        # Use the logic already computed in the dataset class
        if class_name == "none" or class_idx == -1:
            print(f"Skip immagine {idx}: Solo sfondo.")
            continue

        print(f"Processando Img {idx} -> Classe: {class_name}")

        # Esegui Pipeline
        try:
            # Note: target_np is needed for the GT plot later
            target_np = np.array(target_pil)

            orig, prob, hard, crf = quick_pipeline(
                img_pil, class_name, bcos_model, clip_model, bg_weight
            )

            # Crea Maschera Ground Truth Binaria per visualizzazione
            gt_binary = (target_np == class_idx).astype(np.uint8) * 255

            # --- PLOT ---
            plt.figure(figsize=(25, 5))

            # 1. Originale
            plt.subplot(1, 5, 1)
            plt.imshow(orig)
            plt.title(f"Original: {class_name}")
            plt.axis("off")

            # 2. Heatmap (Probabilità)
            plt.subplot(1, 5, 2)
            # Resize per visualizzazione
            prob_viz = cv2.resize(prob, (orig.shape[1], orig.shape[0]))
            plt.imshow(prob_viz, cmap="jet", vmin=0, vmax=1)
            plt.title("Softmax Heatmap (MultiScale)")
            plt.axis("off")

            # 3. Hard Mask (Pre-CRF)
            plt.subplot(1, 5, 3)
            plt.imshow(hard * 255, cmap="gray")
            plt.title("Hard Mask (>0.5)")
            plt.axis("off")

            # 4. Risultato Finale (CRF Overlay)
            plt.subplot(1, 5, 4)
            plt.imshow(orig)
            # Maschera rossa semi-trasparente
            mask_overlay = np.zeros_like(orig)
            mask_overlay[:, :, 0] = 255  # Canale Rosso

            # Dove crf è 1, applica rosso
            alpha = 0.5
            blend = orig.copy()
            blend[crf == 1] = (
                orig[crf == 1] * (1 - alpha) + mask_overlay[crf == 1] * alpha
            ).astype(np.uint8)

            plt.imshow(blend)

            # Contorni gialli
            contours, _ = cv2.findContours(
                (crf).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                c = c.squeeze()
                if len(c.shape) == 2:
                    plt.plot(c[:, 0], c[:, 1], "y-", linewidth=2)

            plt.title("Final Output (CRF)")
            plt.axis("off")

            # 5. Ground Truth (Per confronto)
            plt.subplot(1, 5, 5)
            plt.imshow(gt_binary, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            # Salva
            save_path = os.path.join(OUTPUT_DIR, f"voc_{idx}_{class_name}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Salvataggio completato: {save_path}")

        except Exception as e:
            print(f"Errore su immagine {idx}: {e}")
            continue


if __name__ == "__main__":
    run_visual_test(num_samples=10)
