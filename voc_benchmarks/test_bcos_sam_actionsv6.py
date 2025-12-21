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
# INSERISCI QUI LA TUA IMMAGINE DELLE 9 RAGAZZE
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003817.jpg"
OUTPUT_DIR = "sam_bcos_voc_relative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 224

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
    "jumping": (1.0, 0.0, 0.0),  # Rosso
    "phoning": (0.0, 1.0, 0.0),
    "playinginstrument": (0.0, 0.0, 1.0),
    "reading": (1.0, 1.0, 0.0),
    "ridingbike": (0.0, 1.0, 1.0),
    "ridinghorse": (1.0, 0.0, 1.0),
    "running": (1.0, 0.5, 0.0),  # Arancione
    "takingphoto": (0.5, 0.0, 0.5),
    "usingcomputer": (0.0, 1.0, 0.5),
    "walking": (1.0, 0.75, 0.8),  # Rosa
}

# --- 2. PROMPT REVISIONATI (LOGICA RELATIVA) ---

# Rimosso "people", "crowd", "clothes" perché sopprimevano il segnale sulle ragazze
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

PROMPT_CONFIG = {
    "jumping": {
        "positive": ["jumping", "feet off ground", "mid-air"],
        "negative": ["standing", "walking"],  # Negativi minimali
    },
    "phoning": {
        "positive": ["talking on phone", "phone to ear"],
        "negative": ["holding camera", "touching face"],
    },
    "playinginstrument": {
        "positive": ["playing instrument", "guitar", "flute"],
        "negative": ["holding stick", "holding bag"],
    },
    "reading": {
        "positive": ["reading book", "reading"],
        "negative": ["looking at phone", "sleeping"],
    },
    "ridingbike": {
        "positive": ["riding bike", "bicycle"],
        "negative": ["riding horse", "walking"],
    },
    "ridinghorse": {
        "positive": ["riding horse", "horse"],
        "negative": ["riding bike", "cow"],
    },
    "running": {
        "positive": ["running", "sprinting", "jogging"],
        "negative": ["walking", "standing"],
    },
    "takingphoto": {
        "positive": ["taking photo", "camera to eye"],
        "negative": ["phoning", "binoculars"],
    },
    "usingcomputer": {
        "positive": ["using computer", "typing"],
        "negative": ["reading", "writing"],
    },
    "walking": {
        "positive": ["walking", "walking on street"],
        "negative": ["running", "standing", "jumping"],
    },
}

# --- 3. EMBEDDING FUNCTIONS ---


def get_embedding(clip_model, texts, device):
    weights = []
    with torch.no_grad():
        for t in texts:
            # Aggiungiamo il prefisso standard qui per pulizia
            prompt = f"a photo of a person {t}" if "photo" not in t else t
            w = tokenize_text_prompt(clip_model, prompt).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_action_embeddings(clip_model, action, device):
    pos_texts = PROMPT_CONFIG[action]["positive"]
    # Negativi: Solo Background + Altre Azioni (Cruciale per la competizione)
    neg_texts = BASE_BACKGROUND.copy()
    neg_texts += [f"person {act}" for act in VOC_ACTIONS if act != action]

    pos_w = get_embedding(clip_model, pos_texts, device)
    neg_w = get_embedding(clip_model, neg_texts, device)
    return pos_w, neg_w


# --- 4. UTILS ---


def get_crop_from_mask(image, mask, padding_pct=0.2):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return image, (0, 0, 0, 0)

    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)

    h, w = y_max - y_min, x_max - x_min
    pad_h, pad_w = int(h * padding_pct), int(w * padding_pct)

    y_min, y_max = max(0, y_min - pad_h), min(image.height, y_max + pad_h)
    x_min, x_max = max(0, x_min - pad_w), min(image.width, x_max + pad_w)

    return image.crop((x_min, y_min, x_max, y_max)), (x_min, y_min, x_max, y_max)


def normalize_minmax(t):
    m, M = t.min(), t.max()
    if M - m < 1e-6:
        return torch.zeros_like(t)
    return (t - m) / (M - m)


def calculate_score(heatmap, top_k=0.2):
    vals = heatmap.flatten()
    if len(vals) == 0:
        return 0.0
    k = max(1, int(len(vals) * top_k))
    return np.mean(np.partition(vals, -k)[-k:])


def show_mask_custom(mask, ax, color, alpha=0.5):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 5. MAIN ---


def main():
    print("--- STARTING RELATIVE SCORING PIPELINE ---")

    # Load Models
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam_model = (
        build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH).to(DEVICE).eval()
    )
    processor = Sam3Processor(sam_model)
    bcos_model, _ = load_bcos_model()
    bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    clip_model.to(DEVICE).eval()

    # Precompute Embeddings
    print("Pre-computing Embeddings...")
    embeddings = {}
    for action in VOC_ACTIONS:
        embeddings[action] = get_action_embeddings(clip_model, action, DEVICE)

    # Load Image
    raw_image = Image.open(IMG_PATH).convert("RGB")
    W, H = raw_image.size

    # SAM Inference
    print("Running SAM3...")
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")
    masks = output["masks"]
    scores = output["scores"]

    valid_masks = []
    for i in range(len(scores)):
        if scores[i] > 0.15:
            m = masks[i].cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
            valid_masks.append(m > 0)

    print(f"Found {len(valid_masks)} masks. Classifying...")

    final_results = []

    prep = transforms.Compose(
        [
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )

    for idx, mask in enumerate(valid_masks):
        crop, coords = get_crop_from_mask(raw_image, mask)
        crop_tensor = prep(crop).to(DEVICE)

        # Dizionario per raccogliere gli score grezzi di tutte le azioni
        raw_scores = {}

        for action in VOC_ACTIONS:
            pos_w, neg_w = embeddings[action]

            with torch.no_grad():
                _, map_pos, _, _ = compute_attributions(bcos_model, crop_tensor, pos_w)
                _, map_neg, _, _ = compute_attributions(bcos_model, crop_tensor, neg_w)

            # Normalizzazione Indipendente (Fondamentale)
            mp = normalize_minmax(torch.tensor(map_pos).float())
            mn = normalize_minmax(torch.tensor(map_neg).float())

            # Sottrazione diretta ma "dolce" sul negativo
            diff = F.relu(mp - mn * 0.8).numpy()

            # Score basato sui picchi
            score = calculate_score(diff, top_k=0.15)
            raw_scores[action] = score

        # --- LOGICA RELATIVA (SOFTMAX) ---
        # Invece di guardare il valore assoluto (es. 0.12),
        # guardiamo chi domina la distribuzione.

        actions_list = list(raw_scores.keys())
        scores_list = np.array([raw_scores[a] for a in actions_list])

        # Applichiamo Softmax con una "temperatura" per evidenziare le differenze
        # Moltiplichiamo per 20 perché gli score originali sono bassi (0.1 - 0.2)
        exp_scores = np.exp(scores_list * 20)
        probs = exp_scores / np.sum(exp_scores)

        best_idx = np.argmax(probs)
        best_act = actions_list[best_idx]
        best_prob = probs[best_idx]

        print(
            f"\nPerson {idx}: Winner -> {best_act} (Rel Prob: {best_prob:.1%}, Raw: {scores_list[best_idx]:.3f})"
        )

        # Stampiamo il secondo classificato per debug
        sorted_indices = np.argsort(probs)[::-1]
        second_act = actions_list[sorted_indices[1]]
        print(f"   Runner-up: {second_act} ({probs[sorted_indices[1]]:.1%})")

        # Threshold relativo: Accettiamo solo se vince con almeno il 20% di probabilità
        # (Considerando che sono 10 classi, il random guess è 10%)
        if best_prob > 0.10:
            final_results.append((mask, best_act, best_prob))
        else:
            print(f"   -> REJECTED (Uncertain, max prob {best_prob:.1%} too low)")

    # Visualizzazione
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    final_results.sort(key=lambda x: x[2])
    seen_labels = set()

    for mask, action, score in final_results:
        color = ACTION_COLORS.get(action, (1, 1, 1))
        seen_labels.add(action)
        show_mask_custom(mask, ax, color)

        ys, xs = np.where(mask)
        if len(ys) > 0:
            ax.text(
                np.mean(xs),
                np.mean(ys),
                f"{action}\n{score:.0%}",
                color="white",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="white"),
            )

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=a) for a in seen_labels]
    if patches:
        plt.legend(handles=patches)
    plt.axis("off")
    plt.title("SAM3 + B-Cos (Relative Voting)")
    plt.savefig(os.path.join(OUTPUT_DIR, "result_relative.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
