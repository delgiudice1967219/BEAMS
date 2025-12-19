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
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003817.jpg"
OUTPUT_DIR = "sam_bcos_voc_dynamic"
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

# --- 2. PARAMETRI DINAMICI PER CLASSE (IL CUORE DELLA SOLUZIONE) ---
# top_k: Quanto della maschera considerare? (Piccolo per oggetti piccoli, Grande per oggetti grandi)
# weight: Moltiplicatore per compensare la difficoltà della classe
ACTION_PARAMS = {
    "phoning": {"top_k": 0.05, "weight": 1.15},  # Oggetto piccolissimo, boost richiesto
    "takingphoto": {"top_k": 0.05, "weight": 1.15},
    "reading": {"top_k": 0.10, "weight": 1.10},  # Libro/mani focalizzati
    "usingcomputer": {"top_k": 0.15, "weight": 1.05},
    "playinginstrument": {"top_k": 0.15, "weight": 1.05},
    "jumping": {"top_k": 0.20, "weight": 1.0},
    "running": {"top_k": 0.20, "weight": 1.0},
    "ridingbike": {"top_k": 0.30, "weight": 1.0},  # Deve coprire bici + persona
    "ridinghorse": {"top_k": 0.30, "weight": 1.0},
    "walking": {
        "top_k": 0.30,
        "weight": 0.90,
    },  # Penalità leggera: vince solo se le altre perdono
}

# --- 3. PROMPT CONFIGURATION ---
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
    "out of focus",
    "blur",
]

PROMPT_CONFIG = {
    "jumping": {
        "positive": ["jumping", "feet in air"],
        "negative": ["standing", "walking"],
    },
    "phoning": {
        "positive": ["talking on phone", "phone to ear"],
        "negative": ["hand on face"],
    },
    "playinginstrument": {
        "positive": ["playing instrument", "musician"],
        "negative": ["holding stick"],
    },
    "reading": {
        "positive": ["reading book", "open book"],
        "negative": ["looking at phone"],
    },
    "ridingbike": {"positive": ["riding bike", "bicycle"], "negative": ["walking"]},
    "ridinghorse": {"positive": ["riding horse"], "negative": ["walking"]},
    "running": {"positive": ["running", "sprinting"], "negative": ["walking"]},
    "takingphoto": {"positive": ["taking photo", "camera"], "negative": ["phoning"]},
    "usingcomputer": {
        "positive": ["using computer", "typing"],
        "negative": ["reading"],
    },
    "walking": {"positive": ["walking"], "negative": ["standing", "running"]},
}

# --- 4. FUNCTIONS ---


def get_embedding(clip_model, texts, device):
    weights = []
    with torch.no_grad():
        for t in texts:
            prompt = f"a photo of a person {t}" if "photo" not in t else t
            w = tokenize_text_prompt(clip_model, prompt).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_action_embeddings(clip_model, action, device):
    pos_texts = PROMPT_CONFIG[action]["positive"]
    neg_texts = BASE_BACKGROUND.copy()
    # Negativi Competitivi: Aggiungiamo le altre azioni
    neg_texts += [f"person {act}" for act in VOC_ACTIONS if act != action]

    pos_w = get_embedding(clip_model, pos_texts, device)
    neg_w = get_embedding(clip_model, neg_texts, device)
    return pos_w, neg_w


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


# --- CALCOLO SCORE DINAMICO ---
def calculate_dynamic_score(heatmap, action_name):
    vals = heatmap.flatten()
    if len(vals) == 0:
        return 0.0

    # Recupera i parametri specifici per questa classe
    params = ACTION_PARAMS.get(action_name, {"top_k": 0.2, "weight": 1.0})
    top_k_pct = params["top_k"]
    weight = params["weight"]

    # Calcola Top-K Mean
    k = max(1, int(len(vals) * top_k_pct))
    raw_score = np.mean(np.partition(vals, -k)[-k:])

    # Applica il peso correttivo
    final_score = raw_score * weight
    return final_score


def show_mask_custom(mask, ax, color, alpha=0.5):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 5. MAIN ---


def main():
    print("--- STARTING DYNAMIC SCORING PIPELINE ---")

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

    print("Pre-computing Embeddings...")
    embeddings = {}
    for action in VOC_ACTIONS:
        embeddings[action] = get_action_embeddings(clip_model, action, DEVICE)

    raw_image = Image.open(IMG_PATH).convert("RGB")
    W, H = raw_image.size

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

        weighted_scores = {}

        for action in VOC_ACTIONS:
            pos_w, neg_w = embeddings[action]
            with torch.no_grad():
                _, map_pos, _, _ = compute_attributions(bcos_model, crop_tensor, pos_w)
                _, map_neg, _, _ = compute_attributions(bcos_model, crop_tensor, neg_w)

            mp = normalize_minmax(torch.tensor(map_pos).float())
            mn = normalize_minmax(torch.tensor(map_neg).float())

            # Sottrazione Standard
            diff = F.relu(mp - mn * 0.8).numpy()

            # --- QUI STA IL TRUCCO: USIAMO IL CALCOLO DINAMICO ---
            score = calculate_dynamic_score(diff, action)
            weighted_scores[action] = score

        # Softmax con Temperatura
        actions_list = list(weighted_scores.keys())
        scores_array = np.array([weighted_scores[a] for a in actions_list])

        # Temperatura: 30 è aggressivo, ma ora i pesi sono bilanciati
        exp_scores = np.exp(scores_array * 25)
        probs = exp_scores / np.sum(exp_scores)

        best_idx = np.argmax(probs)
        best_act = actions_list[best_idx]
        best_prob = probs[best_idx]

        # Log dettagliato per capire il sorpasso
        sorted_indices = np.argsort(probs)[::-1]
        winner = actions_list[sorted_indices[0]]
        runner_up = actions_list[sorted_indices[1]]

        print(f"\nPerson {idx}:")
        print(
            f"  WINNER: {winner} (Prob: {probs[sorted_indices[0]]:.1%}, WeightedRaw: {scores_array[sorted_indices[0]]:.3f})"
        )
        print(
            f"  2nd:    {runner_up} (Prob: {probs[sorted_indices[1]]:.1%}, WeightedRaw: {scores_array[sorted_indices[1]]:.3f})"
        )

        # Thresholding Finale
        if best_prob > 0.15:  # Abbassiamo leggermente dato che abbiamo moltiplicatori
            final_results.append((mask, best_act, best_prob))
        else:
            print("  -> REJECTED (Too uncertain)")

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
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="white", pad=0.1),
            )

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=a) for a in seen_labels]
    if patches:
        plt.legend(handles=patches)
    plt.axis("off")
    plt.title("SAM3 + B-Cos (Dynamic Scoring)")
    plt.savefig(os.path.join(OUTPUT_DIR, "result_dynamic.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
