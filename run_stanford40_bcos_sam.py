import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import random

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

STANFORD_ROOT = "data/Stanford40"
OUTPUT_ROOT = "outputs_stanford40"
MAX_IMAGES_PER_ACTION = 6

os.makedirs(OUTPUT_ROOT, exist_ok=True)

PROMPT_PAIRS = {
    "riding_a_horse": [
        "person riding a horse",  # Target Action
        "person standing",  # Negative (per vedere se BCOS distingue l'azione dall'oggetto)
    ],
    "playing_guitar": [
        "person playing guitar",
        "person standing",  # Negative (simile posa ma oggetto diverso)
    ],
    "climbing": [
        "person climbing",
        "person standing",  # Negative (posa molto diversa)
    ],
    "fixing_a_car": [
        "person fixing a car",
        "person standing",  # Negative (oggetto senza azione)
    ],
}

# ============================================================
# IMPORT BCOS
# ============================================================
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# ============================================================
# IMPORT SAM3
# ============================================================
SAM3_PATH = r"C:\Users\xavie\Desktop\Universitá\2nd year\AML\BCos_object_detection\sam3"
SAM3_CKPT = r"C:\Users\xavie\Desktop\Universitá\2nd year\AML\BCos_object_detection\sam3_model\models--facebook--sam3\snapshots\3c879f39826c281e95690f02c7821c4de09afae7\sam3.pt"

sys.path.append(SAM3_PATH)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================
# UTILS (COLORI & DATASET)
# ============================================================
def get_random_color(seed=None):
    """Genera un colore casuale (B, G, R) brillante."""
    if seed is not None:
        random.seed(seed)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


# ============================================================
# UTILS PER PULIZIA MASCHERE (NUOVO)
# ============================================================
def refine_mask(prob_map, img_shape, threshold=0.55):
    """
    Pulisce la mappa di probabilità grezza di B-COS.
    Args:
        prob_map: np.array (H, W) float 0-1
        img_shape: tuple (H, W) target size
        threshold: float, soglia di taglio
    Returns:
        clean_mask: np.array (H, W) uint8 (0 o 1)
    """
    # 1. Resize alla dimensione originale immagine
    mask_resized = cv2.resize(
        prob_map, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )

    # 2. Bilateral Filter: riduce il rumore preservando i bordi
    # Nota: richiede float32
    mask_blur = cv2.bilateralFilter(
        mask_resized.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75
    )

    # 3. Binarizzazione
    binary_mask = (mask_blur > threshold).astype(np.uint8)

    # 4. Operazioni Morfologiche per rimuovere rumore (puntini) e chiudere buchi
    kernel = np.ones(
        (7, 7), np.uint8
    )  # Kernel più grande = più pulizia ma meno dettagli fini

    # Opening: Rimuove rumore bianco su sfondo nero
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Closing: Chiude buchi neri dentro l'oggetto bianco
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Mantenere solo il "Largest Connected Component"
    # (Assume che l'oggetto principale sia uno solo e grande, rimuovendo artefatti sparsi)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels > 1:
        # stats: [x, y, width, height, area]
        # L'indice 0 è sempre lo sfondo, quindi cerchiamo il max dall'indice 1 in poi
        largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Creiamo una nuova maschera solo con l'isola più grande
        clean_mask = np.zeros_like(binary_mask)
        clean_mask[labels == largest_label_idx] = 1
        return clean_mask

    return binary_mask


def apply_mask_overlay(image_bgr, mask, color, alpha=0.5):
    """Applica una maschera colorata semi-trasparente."""
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def load_split_images(root, action, split="test", max_images=None):
    split_file = os.path.join(root, "ImageSplits", f"{action}_{split}.txt")
    with open(split_file, "r") as f:
        names = [l.strip() for l in f.readlines()]
    if max_images is not None:
        names = names[:max_images]
    return [os.path.join(root, "JPEGImages", name) for name in names]


# ============================================================
# BCOS UTILS
# ============================================================
def preprocess(img_pil):
    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )(img_pil)


def get_target_weight(clip_model, text):
    prompts = [f"a clean origami {text}.", f"a photo of a {text}.", f"the {text}."]
    with torch.no_grad():
        w = [tokenize_text_prompt(clip_model, p).to(DEVICE) for p in prompts]
    return torch.mean(torch.stack(w), dim=0)


def precompute_background_weights(clip_model):
    bg = [
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
    with torch.no_grad():
        w = [tokenize_text_prompt(clip_model, f"a photo of {b}").to(DEVICE) for b in bg]
    return torch.mean(torch.stack(w), dim=0)


# ============================================================
# BCOS CORE
# ============================================================
def process_single_image_softmax(model, image_tensor, target_weight, bg_weight):
    scales = [448, 560]
    acc = []
    img_cpu = image_tensor.squeeze()
    blur = transforms.GaussianBlur(5, sigma=1.0)

    for s in scales:
        resize = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )
        img = resize(img_cpu).unsqueeze(0).to(DEVICE)
        inp = img.squeeze(0)

        for flip in [False, True]:
            inp_f = torch.flip(inp.unsqueeze(0), dims=[3]).squeeze(0) if flip else inp

            with torch.no_grad():
                _, mt, _, _ = compute_attributions(model, inp_f, target_weight)
                _, mb, _, _ = compute_attributions(model, inp_f, bg_weight)

                if isinstance(mt, np.ndarray):
                    mt = torch.from_numpy(mt).to(DEVICE)
                if isinstance(mb, np.ndarray):
                    mb = torch.from_numpy(mb).to(DEVICE)

                if mt.dim() == 3:
                    mt = mt[0]
                if mb.dim() == 3:
                    mb = mb[0]

                mt = blur(mt.unsqueeze(0)).squeeze()
                mb = blur(mb.unsqueeze(0)).squeeze()

                gmin, gmax = min(mt.min(), mb.min()), max(mt.max(), mb.max())
                mt = (mt - gmin) / (gmax - gmin + 1e-8)
                mb = (mb - gmin) / (gmax - gmin + 1e-8)

                probs = F.softmax(torch.stack([mb, mt]) * 20, dim=0)
                tgt = probs[1]

                if flip:
                    tgt = torch.flip(tgt, dims=[1])

                tgt = tgt.unsqueeze(0).unsqueeze(0)
                acc.append(
                    F.interpolate(
                        tgt, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                    ).squeeze()
                )

    return torch.mean(torch.stack(acc), dim=0).cpu().numpy()


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading BCos + CLIP...")
    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    clip_model = clip_model.to(DEVICE).eval()
    bg_weight = precompute_background_weights(clip_model)

    print("Loading SAM3...")
    sam3_model = build_sam3_image_model(checkpoint_path=SAM3_CKPT)
    sam3_model.to(DEVICE).eval()
    sam3 = Sam3Processor(sam3_model)

    for action, prompts in PROMPT_PAIRS.items():
        print(f"\n=== ACTION: {action} ===")
        image_paths = load_split_images(
            STANFORD_ROOT, action, "test", MAX_IMAGES_PER_ACTION
        )

        for img_path in image_paths:
            name = os.path.basename(img_path).replace(".jpg", "")
            print(f"Processing {name}")

            img_pil = Image.open(img_path).convert("RGB")
            img_np_rgb = np.array(img_pil)
            # CORREZIONE 1: Convertiamo subito in BGR per OpenCV
            img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

            out_dir = os.path.join(OUTPUT_ROOT, action, name)
            os.makedirs(out_dir, exist_ok=True)

            # ---------------- BCOS ----------------
            img_tensor = preprocess(img_pil)

            for idx, prompt in enumerate(prompts):
                tw = get_target_weight(clip_model, prompt)

                # Ottieni la probability map grezza (valori 0.0 - 1.0)
                raw_prob = process_single_image_softmax(
                    bcos_model, img_tensor, tw, bg_weight
                )

                # --- PULIZIA DELLA MASCHERA ---
                # Passiamo la shape dell'immagine originale (img_np_rgb)
                clean_mask_binary = refine_mask(raw_prob, img_np_rgb.shape)

                # Visualizzazione
                # Colore: Rosso per BCOS (o cambia a piacere)
                bcos_vis = apply_mask_overlay(
                    img_bgr, clean_mask_binary, color=(0, 0, 255), alpha=0.55
                )

                # Opzionale: Disegna un contorno attorno alla maschera per renderla più "pro"
                contours, _ = cv2.findContours(
                    clean_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    bcos_vis, contours, -1, (0, 0, 255), 2
                )  # Contorno solido spesso 2px

                cv2.imwrite(
                    os.path.join(out_dir, f"bcos_{prompt.replace(' ','_')}.png"),
                    bcos_vis,
                )

            # ---------------- SAM3 ----------------
            state = sam3.set_image(img_pil)

            for prompt in prompts:
                out = sam3.set_text_prompt(state=state, prompt=prompt)
                masks, scores = out["masks"], out["scores"]

                # Copiamo l'immagine base BGR
                sam3_vis = img_bgr.copy()

                # CORREZIONE 2: Iteriamo sulle maschere con colori diversi
                for i in range(len(scores)):
                    if scores[i] > 0.1:  # Threshold score
                        m = masks[i].cpu().numpy().squeeze()
                        # Genera un colore random (o basato sull'indice per coerenza)
                        # Usiamo l'indice 'i' come seed se vogliamo colori fissi per ordine
                        color = get_random_color(seed=i * 100)

                        # Applichiamo la maschera colorata all'immagine accumulata
                        sam3_vis = apply_mask_overlay(sam3_vis, m, color, alpha=0.5)

                cv2.imwrite(
                    os.path.join(out_dir, f"sam3_{prompt.replace(' ','_')}.png"),
                    sam3_vis,
                )

            print(f"[DONE] {action}/{name}")


if __name__ == "__main__":
    main()
