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
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005780.jpg"
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

# --- 2. PROMPT CONFIGURATION ---

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

#

PROMPT_CONFIG = {
    "jumping": {
        "positive": [
            "a photo of a person jumping in the air",
            "feet completely off the ground",
            "knees bent in mid-air",
            "an athlete leaping high",
            "a person doing a jump",
        ],
        "negative": [
            "a person standing on the ground",
            "feet planted on the floor",
            "a person walking",
            "a person running on the ground",
            "shadow on the ground",
        ],
    },
    "phoning": {
        "positive": [
            "holding a smartphone to the ear",
            "talking on a mobile phone",
            "hand holding a black phone against face",
            "making a phone call",
            "screen of a mobile phone",
        ],
        "negative": [
            "touching face with hand",
            "scratching head",
            "resting hand on cheek",
            "drinking from a cup",
            "adjusting glasses",
            "holding a camera",
        ],
    },
    "playinginstrument": {
        "positive": [
            "playing a musical instrument",
            "fingers on guitar strings",
            "blowing into a wind instrument",
            "hands on piano keys",
            "musician performing with instrument",
            "holding a violin",
        ],
        "negative": [
            "holding a weapon",
            "holding a baseball bat",
            "holding a stick",
            "carrying a bag",
            "hands in pockets",
            "clapping hands",
        ],
    },
    "reading": {
        "positive": [
            "looking down at an open book",
            "reading a newspaper",
            "eyes focused on text in a book",
            "holding a magazine to read",
        ],
        "negative": [
            "sleeping with head down",
            "looking at a mobile phone",  # Crucial distinction
            "looking at the ground",
            "writing on paper",
            "using a laptop",
        ],
    },
    "ridingbike": {
        "positive": [
            "straddling a bicycle seat",
            "sitting on a bike",
            "hands gripping bicycle handlebars",
            "feet on bicycle pedals",
            "cyclist riding a bike",
        ],
        "negative": [
            "walking next to a bicycle",
            "pushing a bike by the handlebars",
            "standing near a bike",
            "riding a motorcycle",  # Hard negative
            "repairing a bike",
        ],
    },
    "ridinghorse": {
        "positive": [
            "sitting on a horse",
            "equestrian riding a horse",
            "horseback riding posture",
            "legs straddling a horse",
        ],
        "negative": [
            "standing next to a horse",
            "grooming a horse",
            "leading a horse by the rein",
            "petting an animal",
            "cowboy standing",
        ],
    },
    "running": {
        "positive": [
            "running fast",
            "sprinting with wide stride",
            "jogging motion",
            "runner in athletic gear",
            "blurred motion of running",
        ],
        "negative": [
            "walking slowly",
            "standing still",
            "waiting in line",
            "strolling casually",
            "jumping in place",
        ],
    },
    "takingphoto": {
        "positive": [
            "holding a camera up to the eye",
            "looking through camera viewfinder",
            "photographer taking a picture",
            "holding a DSLR camera",
            "pointing a camera lens",
        ],
        "negative": [
            "looking through binoculars",
            "drinking from a bottle",
            "adjusting sunglasses",
            "pointing with finger",
            "talking on a phone",
        ],
    },
    "usingcomputer": {
        "positive": [
            "typing on a laptop keyboard",
            "staring at a computer monitor",
            "sitting at a desk using a computer",
            "hands on mouse and keyboard",
            "working in an office cubicle",
        ],
        "negative": [
            "watching television",
            "reading a book",
            "writing with a pen on paper",
            "eating at a table",
            "playing a board game",
        ],
    },
    "walking": {
        "positive": [
            "walking normally",
            "pedestrian strolling on street",
            "taking a step forward",
            "walking slowly",
        ],
        "negative": [
            "running fast",
            "sprinting",
            "standing perfectly still",
            "sitting on a bench",
            "riding a bike",
        ],
    },
}


# --- 3. FUNZIONI DI EMBEDDING ---


def get_action_embedding(clip_model, action, device):
    prompts = list(PROMPT_CONFIG[action]["positive"])  # COPY
    prompts += [
        f"a photo of a person {action}.",
        f"a clean origami of a person {action}.",
    ]

    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_negative_embedding(clip_model, action, device):
    final_negatives = BASE_BACKGROUND.copy()
    specific_negatives = PROMPT_CONFIG[action]["negative"]
    final_negatives.extend(specific_negatives)

    weights = []
    with torch.no_grad():
        for p in final_negatives:
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


def calculate_topk_score(heatmap, mask, top_k_percent=0.25):
    """
    Calcola il punteggio basandosi solo sul top X% dei pixel nella maschera.
    Questo premia i 'picchi' di attivazione (es. solo il telefono o la chitarra)
    invece di diluirli con tutto il corpo.
    """
    # Estrai solo i valori dentro la maschera
    values = heatmap[mask]

    if len(values) == 0:
        return 0.0

    # Se la maschera è molto piccola, prendi tutto
    if len(values) < 10:
        return np.mean(values)

    # Ordina i valori (dal più piccolo al più grande)
    # Usiamo partition per efficienza invece di sort completo
    k = int(len(values) * top_k_percent)
    if k < 1:
        k = 1

    # Prendi i Top-K valori più alti (gli ultimi k dopo il sort)
    top_values = np.partition(values, -k)[-k:]

    return np.mean(top_values)


def score_action(
    diff_map,
    mask,
    mean_other_maps,
    top_q=(0.7, 0.95),
    lambda_c=0.7,
    beta_l=0.6,
):
    """
    diff_map: HxW numpy (pos - neg)
    mask: boolean person mask
    mean_other_maps: HxW numpy (mean of other actions diff_maps)
    """

    eps = 1e-6
    vals = diff_map[mask]

    if len(vals) < 20:
        return -np.inf

    # --- SUPPORTO ROBUSTO (S_a) ---
    lo, hi = np.quantile(vals, top_q)
    support_vals = vals[(vals >= lo) & (vals <= hi)]
    S = np.mean(support_vals) if len(support_vals) > 0 else 0.0

    # --- CONTRADDIZIONE (C_a) ---
    neg_vals = vals[vals < 0]
    C = -np.mean(neg_vals) if len(neg_vals) > 0 else 0.0

    # --- SELETTIVITÀ (L_a) ---
    pos_map = np.maximum(diff_map, 0)
    baseline = np.maximum(mean_other_maps, 0)

    selective = pos_map / (baseline + eps)
    sel_vals = selective[mask]
    sel_top = np.quantile(sel_vals, 0.8)
    L = np.mean(sel_vals[sel_vals >= sel_top]) if len(sel_vals) > 0 else 0.0

    # --- SCORE FINALE ---
    score = S - lambda_c * C + beta_l * np.log1p(L)
    return score


def binding_score(diff_map, person_mask, radius_px=18, top_p=0.15):
    """
    Misura quanto la "prova" (evidenza positiva) di diff_map è legata alla persona.
    - diff_map: HxW numpy (pos-neg), con segno
    - person_mask: HxW bool
    - radius_px: quanto dilatare la mask (zona di interazione)
    - top_p: usa solo i pixel più forti della prova (top p%) per evitare rumore diffuso

    Ritorna un valore in [0,1]:
      ~1 => quasi tutta la prova è vicino/attaccata alla persona
      ~0 => prova lontana (background, altra persona, oggetto distante)
    """
    eps = 1e-8

    # Prova = solo evidenza positiva
    P = np.maximum(diff_map, 0.0)

    # Se non c'è prova positiva, binding nullo
    total = P.sum()
    if total < 1e-8:
        return 0.0

    # Usa solo la parte più "forte" della prova (top quantile) per evitare che
    # la prova diffusa faccia sembrare tutto legato alla persona.
    thr = np.quantile(P[P > 0], 1.0 - top_p) if np.any(P > 0) else np.inf
    P_strong = np.where(P >= thr, P, 0.0)

    total_strong = P_strong.sum()
    if total_strong < 1e-8:
        return 0.0

    # Dilata la mask persona per includere oggetti tenuti in mano / vicino al volto
    mask_u8 = person_mask.astype(np.uint8) * 255
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1) > 0

    inside = P_strong[dilated].sum()
    return float(inside / (total_strong + eps))


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
    scales = [224, 448, 560, 672, 784]

    clean_heatmaps = {}

    for action in VOC_ACTIONS:
        pos_w = get_action_embedding(clip_model, action, DEVICE)
        neg_w = get_negative_embedding(clip_model, action, DEVICE)

        accumulated_maps = []

        for s in scales:
            resize_t = transforms.Resize(
                (s, s), interpolation=transforms.InterpolationMode.BICUBIC
            )
            img_scaled = resize_t(img_tensor).to(DEVICE)

            with torch.no_grad():
                _, map_pos, _, _ = compute_attributions(bcos_model, img_scaled, pos_w)
                _, map_neg, _, _ = compute_attributions(bcos_model, img_scaled, neg_w)

                map_pos = torch.as_tensor(map_pos).cpu().float()
                while map_pos.dim() > 2:
                    map_pos = map_pos[0]
                map_neg = torch.as_tensor(map_neg).cpu().float()
                while map_neg.dim() > 2:
                    map_neg = map_neg[0]

                map_pos = blur(map_pos.unsqueeze(0)).squeeze()
                map_neg = blur(map_neg.unsqueeze(0)).squeeze()

                # g_min = min(map_pos.min(), map_neg.min())
                # g_max = max(map_pos.max(), map_neg.max())
                # denom = g_max - g_min + 1e-8

                # norm_pos = (map_pos - g_min) / denom
                # norm_neg = (map_neg - g_min) / denom

                diff_map = map_pos - map_neg

                # 2. ReLU: Ci interessa solo l'evidenza positiva.
                # Se è negativo, significa che è più sfondo che azione -> 0.
                res = F.interpolate(
                    diff_map[None, None],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                accumulated_maps.append(res)

        # Fuori dal loop delle scale:
        avg_heatmap = torch.mean(torch.stack(accumulated_maps), dim=0)

        # Normalizzazione Min-Max SOLO ALLA FINE e globale per l'azione
        # Questo preserva i picchi relativi
        # if avg_heatmap.max() > 1e-5:
        #     avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (
        #         avg_heatmap.max() - avg_heatmap.min()
        #     )
        # else:
        #     avg_heatmap = torch.zeros_like(avg_heatmap)

        clean_heatmaps[action] = avg_heatmap.numpy()

        #         probs = F.softmax(torch.stack([norm_neg, norm_pos]) * 5, dim=0)
        #         prob_map_action = probs[1]

        #         res = F.interpolate(
        #             prob_map_action[None, None],
        #             size=(H, W),
        #             mode="bilinear",
        #             align_corners=False,
        #         ).squeeze()
        #         accumulated_maps.append(res)

        # clean_heatmaps[action] = torch.mean(
        #     torch.stack(accumulated_maps), dim=0
        # ).numpy()

    # --- NEW: VISUALIZZAZIONE INTERMEDIA HEATMAPS ---
    print("\n--- VISUALIZING INTERMEDIATE HEATMAPS ---")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("B-Cos Heatmaps (Binary Softmax Probabilities)", fontsize=16)
    axes = axes.flatten()

    for idx, action in enumerate(VOC_ACTIONS):
        ax = axes[idx]
        heatmap = clean_heatmaps[action]
        v = np.max(np.abs(heatmap)) + 1e-8
        im = ax.imshow(heatmap, cmap="seismic", vmin=-v, vmax=v)
        ax.set_title(action)
        ax.axis("off")

    # Aggiungi colorbar laterale
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    heatmap_path = os.path.join(OUTPUT_DIR, "intermediate_heatmaps.png")
    plt.savefig(heatmap_path, bbox_inches="tight")
    print(f"📸 Saved intermediate heatmaps to: {heatmap_path}")
    # plt.show() # Decommenta se vuoi vederle a schermo
    # -----------------------------------------------

    # D. VOTING (SAM MASKS)
    # print("Voting...")
    # final_results = []

    # for idx, mask in enumerate(valid_masks):
    #     best_act = "unknown"
    #     best_score = -1.0
    #     debug_scores = {}

    #     for action in VOC_ACTIONS:
    #         heatmap = clean_heatmaps[action]
    #         score = np.mean(heatmap[mask])
    #         debug_scores[action] = score

    #         if score > best_score:
    #             best_score = score
    #             best_act = action

    #     if best_score > 0.55:
    #         final_results.append((mask, best_act, best_score))
    #         print(f"   Mask {idx}: {best_act} (Conf: {best_score:.1%})")
    #     else:
    #         print(f"   Mask {idx}: Rejected (Max conf {best_score:.1%} too low)")
    # print("Voting using Intensity Weighted Mean (Power Mean)...")
    # final_results = []

    # for idx, mask in enumerate(valid_masks):
    #     best_act = "unknown"
    #     best_score = -1.0
    #     debug_scores = {}

    #     for action in VOC_ACTIONS:
    #         heatmap = clean_heatmaps[action]

    #         # --- MODIFICA CRUCIALE: POWER MEAN ---
    #         # Eleviamo al quadrato (o potenza 3) i valori prima della media.
    #         # Questo dà molto più peso ai pixel "rossi" (0.9 -> 0.81)
    #         # rispetto ai pixel "blu/gialli" (0.3 -> 0.09).
    #         # "Vince chi è più intenso"

    #         mask_pixels = heatmap[mask]
    #         if len(mask_pixels) == 0:
    #             score = 0
    #         else:
    #             # Calcolo della media dell'Energia (x^2)
    #             score = np.mean(np.power(mask_pixels, 2))

    #         # -------------------------------------

    #         debug_scores[action] = score

    #         if score > best_score:
    #             best_score = score
    #             best_act = action

    #     # Poiché abbiamo elevato al quadrato, i valori saranno più bassi del solito
    #     # (es. 0.7 medio diventa 0.49). Quindi abbassiamo leggermente la soglia di scarto.
    #     if best_score > 0.25:
    #         final_results.append((mask, best_act, best_score))
    #         print(f"   Mask {idx}: {best_act} (Intensity Score: {best_score:.3f})")
    #     else:
    #         print(f"   Mask {idx}: Rejected (Intensity Score {best_score:.3f} too low)")
    print("Voting using Coherent Action Score + Binding...")

    final_results = []

    all_maps = np.stack([clean_heatmaps[a] for a in VOC_ACTIONS], axis=0)

    # iperparametri (globali)
    BIND_RADIUS_PX = 18  # prova 14..22
    BIND_TOP_P = 0.15  # usa top 15% della prova positiva
    ALPHA_BIND = 1.5  # quanto pesa il binding (1.0 neutro, 2.0 più forte)

    for idx, mask in enumerate(valid_masks):
        best_act = "unknown"
        best_score = -1e9
        best_bind = 0.0

        for i, action in enumerate(VOC_ACTIONS):
            diff_map = clean_heatmaps[action]
            mean_other = np.mean(np.delete(all_maps, i, axis=0), axis=0)

            # Score "coerente" che già usi (supporto-contraddizione-selettività)
            s = score_action(
                diff_map,
                mask,
                mean_other,
                lambda_c=0.7,
                beta_l=0.6,
            )

            # Binding: la prova sta vicino alla persona?
            b = binding_score(
                diff_map,
                mask,
                radius_px=BIND_RADIUS_PX,
                top_p=BIND_TOP_P,
            )

            # Score finale: premia azioni con prova "attaccata" alla persona
            final = s * (b**ALPHA_BIND)

            if final > best_score:
                best_score = final
                best_act = action
                best_bind = b

        # soglia: qui conviene usare una soglia sul final score
        if best_score > 0:
            final_results.append((mask, best_act, best_score))
            print(
                f"   Mask {idx}: {best_act} (Final: {best_score:.3f}, Bind: {best_bind:.2f})"
            )
        else:
            print(f"   Mask {idx}: Rejected (Final {best_score:.3f})")

        # E. VISUALIZZAZIONE
        print("Visualizing Final Result...")
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

        patches = [
            mpatches.Patch(color=ACTION_COLORS[a], label=a) for a in used_legends
        ]
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
