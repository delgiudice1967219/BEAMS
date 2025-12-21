import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.patches as mpatches

# --- 1. CONFIGURAZIONE E IMPORT ---
path_to_sam3 = (
    "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3"
)
if path_to_sam3 not in sys.path:
    sys.path.append(path_to_sam3)

sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

CHECKPOINT_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005070.jpg"
OUTPUT_DIR = "sam_bcos_voc_refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

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

ACTION_COLORS = {
    "jumping": (1.0, 0.0, 0.0),
    "phoning": (0.0, 1.0, 0.0),
    "playinginstrument": (0.0, 0.0, 1.0),
    "reading": (1.0, 1.0, 0.0),
    "ridingbike": (0.0, 1.0, 1.0),
    "ridinghorse": (1.0, 0.0, 1.0),
    "running": (1.0, 0.5, 0.0),
    "takingphoto": (0.5, 0.0, 0.5),
    "usingcomputer": (0.0, 1.0, 0.5),
    "walking": (1.0, 0.75, 0.8),
}

# --- 2. PROMPT CONFIGURATION (IL CUORE DELLA SOLUZIONE) ---

# Sfondo generico (sempre valido)
BASE_BACKGROUND = [
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
    "out of focus",
    "scenery",
    "empty space",
]

# Configurazione specifica per ogni azione
# POSITIVE: Cosa cerchiamo
# NEGATIVE: Cosa NON è l'azione (Distrattori specifici)
PROMPT_CONFIG = {
    "jumping": {
        "positive": [
            "jumping person",
            "person mid-air",
            "feet off the ground",
            "leaping",
        ],
        "negative": [
            "person standing",
            "person walking",
            "person sitting",
            "feet on ground",
        ],
    },
    "phoning": {
        "positive": [
            "person using a mobile phone",
            "having a smartphone near ear",
            # "talking on smartphone",
        ],
        "negative": [
            "person touching face",
            "hand near head",
            "generic person",
            "person listening",
        ],
    },
    "playinginstrument": {
        "positive": [
            "playing a musical instrument",
            "holding a guitar",
            "playing flute",
            "musician",
        ],
        "negative": ["holding a stick", "person holding object", "generic person"],
    },
    "reading": {
        "positive": ["person reading a book", "reading a newspaper", "looking at text"],
        "negative": ["person looking down", "sleeping person", "generic person"],
    },
    "ridingbike": {
        "positive": ["riding a bicycle", "cyclist on bike", "pedaling"],
        "negative": ["walking next to bike", "standing near bike", "person walking"],
    },
    "ridinghorse": {
        "positive": ["riding a horse", "equestrian on horse"],
        "negative": ["standing next to horse", "grooming horse", "generic person"],
    },
    "running": {
        "positive": ["running person", "sprinting", "jogging", "fast motion"],
        "negative": ["walking person", "standing person", "person waiting"],
    },
    "takingphoto": {
        "positive": [
            "taking a photo",
            "holding camera to eye",
            "photographer shooting",
        ],
        "negative": ["person looking", "holding object", "generic person"],
    },
    "usingcomputer": {
        "positive": [
            "typing on laptop",
            "using computer keyboard",
            "looking at monitor",
        ],
        "negative": ["sitting at desk", "watching tv", "generic person"],
    },
    "walking": {
        "positive": ["walking person", "person going slowly", "person having a walk"],
        "negative": ["running person", "sitting person"],
    },
}

# --- 3. FUNZIONI DI EMBEDDING ---


def get_action_embedding(clip_model, action, device):
    prompts = PROMPT_CONFIG[action]["positive"]
    # Aggiungi sempre il prompt base generico
    prompts.append(f"a photo of a person {action}.")
    prompts.append(f"a clean origami of a person {action}.")

    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_negative_embedding(clip_model, action, device):
    """
    Combina lo sfondo base con i negativi specifici per l'azione.
    Es: Se cerco 'running', il negativo include 'walking'.
    Es: Se cerco 'phoning', il negativo include 'generic person'.
    """
    # 1. Inizia con lo sfondo base
    final_negatives = BASE_BACKGROUND.copy()

    # 2. Aggiungi i negativi specifici dell'azione
    specific_negatives = PROMPT_CONFIG[action]["negative"]
    final_negatives.extend(specific_negatives)

    weights = []
    with torch.no_grad():
        for p in final_negatives:
            # Usa "a photo of..." per i background objects
            # Usa il testo diretto per le persone
            if "person" in p or "walking" in p or "standing" in p:
                text = p
            else:
                text = f"a photo of {p}"

            w = tokenize_text_prompt(clip_model, text).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def show_mask_custom(mask, ax, color, alpha=0.55):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 4. PIPELINE ---


def main():
    print(f"--- STARTING REFINED VOC PIPELINE ---")

    # A. LOAD MODELS
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"Errore SAM3: {e}")
        return

    sam_model = (
        build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH).to(DEVICE).eval()
    )
    processor = Sam3Processor(sam_model)

    bcos_model, _ = load_bcos_model()
    bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    clip_model.to(DEVICE).eval()
    print("✅ Models Loaded.")

    raw_image = Image.open(IMG_PATH).convert("RGB")
    W, H = raw_image.size

    # B. SAM SEGMENTATION
    print("Running SAM3...")
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")
    masks_tensor = output["masks"]
    scores_tensor = output["scores"]

    valid_masks = []
    for i in range(len(scores_tensor)):
        if scores_tensor[i].item() > 0.10:
            m = masks_tensor[i].cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
            valid_masks.append(m > 0)
    print(f"Found {len(valid_masks)} person masks.")

    # C. B-COS DYNAMIC HEATMAPS
    print("Generating Action-Specific Heatmaps...")

    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    img_tensor = prep(raw_image)
    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    scales = [448, 560]

    # Memorizziamo le mappe pulite
    clean_heatmaps = {}

    for action in VOC_ACTIONS:
        # Calcoliamo embeddings specifici per QUESTA azione
        pos_w = get_action_embedding(clip_model, action, DEVICE)
        neg_w = get_negative_embedding(clip_model, action, DEVICE)

        accumulated_maps = []

        for s in scales:
            resize_t = transforms.Resize(
                (s, s), interpolation=transforms.InterpolationMode.BICUBIC
            )
            img_scaled = resize_t(img_tensor).to(DEVICE)

            with torch.no_grad():
                # B-Cos Raw Attribution
                _, map_pos, _, _ = compute_attributions(bcos_model, img_scaled, pos_w)
                _, map_neg, _, _ = compute_attributions(bcos_model, img_scaled, neg_w)

                # Process
                map_pos = torch.as_tensor(map_pos).cpu().float()
                while map_pos.dim() > 2:
                    map_pos = map_pos[0]
                map_neg = torch.as_tensor(map_neg).cpu().float()
                while map_neg.dim() > 2:
                    map_neg = map_neg[0]

                map_pos = blur(map_pos.unsqueeze(0)).squeeze()
                map_neg = blur(map_neg.unsqueeze(0)).squeeze()

                # Binary Logic: Action vs Specific Negatives
                g_min = min(map_pos.min(), map_neg.min())
                g_max = max(map_pos.max(), map_neg.max())
                denom = g_max - g_min + 1e-8

                norm_pos = (map_pos - g_min) / denom
                norm_neg = (map_neg - g_min) / denom

                # Softmax Binaria Aggressiva
                probs = F.softmax(torch.stack([norm_neg, norm_pos]) * 30, dim=0)
                prob_map_action = probs[1]

                res = F.interpolate(
                    prob_map_action[None, None],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                accumulated_maps.append(res)

        clean_heatmaps[action] = torch.mean(
            torch.stack(accumulated_maps), dim=0
        ).numpy()

    # D. VOTING (SAM MASKS)
    print("Voting...")
    final_results = []

    for idx, mask in enumerate(valid_masks):
        best_act = "unknown"
        best_score = -1.0

        # Dizionario di debug per vedere tutti i punteggi di una persona
        debug_scores = {}

        for action in VOC_ACTIONS:
            heatmap = clean_heatmaps[action]
            score = np.mean(heatmap[mask])
            debug_scores[action] = score

            if score > best_score:
                best_score = score
                best_act = action

        # Stampa punteggi per debug
        # print(f"Mask {idx} scores: {debug_scores}")

        # Soglia dinamica:
        # 0.5 è il punto di equilibrio della softmax binaria.
        # > 0.5 significa che è più 'Action' che 'Negative'.
        if best_score > 0.55:
            final_results.append((mask, best_act, best_score))
            print(f"   Mask {idx}: {best_act} (Conf: {best_score:.1%})")
        else:
            print(
                f"   Mask {idx}: Rejected (Max conf {best_score:.1%} too low - likely just generic person)"
            )

    # E. VISUALIZZAZIONE
    print("Visualizing...")
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    final_results.sort(key=lambda x: x[2])
    used_legends = set()

    for mask, action, score in final_results:
        color = ACTION_COLORS.get(action, (1.0, 1.0, 1.0))
        used_legends.add(action)
        show_mask_custom(mask, ax, color)

        ys, xs = np.where(mask)
        if len(ys) > 0:
            ax.text(
                np.mean(xs),
                np.mean(ys),
                f"{action}\n{score:.0%}",
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color,
                    alpha=0.8,
                    edgecolor="white",
                    boxstyle="round,pad=0.2",
                ),
            )

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=a) for a in used_legends]
    if patches:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.title("SAM3 + B-Cos (Voc-Optimized Prompts)")

    out_path = os.path.join(OUTPUT_DIR, "final_voc_result.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
