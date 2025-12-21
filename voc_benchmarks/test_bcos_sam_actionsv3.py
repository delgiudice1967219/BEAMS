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
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005751.jpg"
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
    # "ground",
    # "land",
    # "grass",
    # "tree",
    # "building",
    # "wall",
    # "sky",
    # "lake",
    # "water",
    # "river",
    # "sea",
    # "railway",
    # "railroad",
    # "road",
    # "rock",
    # "street",
    # "cloud",
    # "mountain",
    # "floor",
    # "ceiling",
    # "background",
    # "blur",
    # "out of focus",
    # "scenery",
    # "empty space",
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


# PROMPT_CONFIG = {
#     "jumping": {
#         "positive": [
#             "a person jumping in the air",
#             "feet completely off the ground",  # Dettaglio visivo critico
#         ],
#         "negative": [
#             "standing",
#             "walking",
#             "running on ground",
#             "one foot on the floor",
#         ],
#     },
#     "phoning": {
#         "positive": [
#             "a person talking on a mobile phone",
#             "holding a phone to the ear",  # Posa specifica
#         ],
#         "negative": [
#             "holding a camera",  # Confusione classica
#             "touching face",
#             "scratching head",
#             "eating",
#             "drinking",
#         ],
#     },
#     "playinginstrument": {
#         "positive": [
#             "a person playing a musical instrument",
#             "hands on a guitar or wind instrument",  # Oggetti specifici comuni in VOC
#             "musician performing",
#         ],
#         "negative": [
#             "holding a weapon",
#             "holding a stick",
#             "holding a bat",
#             "taking a photo",
#         ],
#     },
#     "reading": {
#         "positive": [
#             "a person reading a book or newspaper",
#             "looking down at an open book",  # Oggetto inconfondibile
#         ],
#         "negative": [
#             "looking at a mobile phone",  # Hard Negative cruciale
#             "using a laptop",
#             "sleeping",
#             "computer screen",
#         ],
#     },
#     "ridingbike": {
#         "positive": [
#             "a person riding a bicycle",
#             "a person above a bicycle wheels and handlebars",
#             "a person riding a motorcycle",  # Oggetti unici (niente gambe/animali)
#         ],
#         "negative": [
#             "riding a horse",  # Hard Negative 1
#             # Hard Negative 2
#             "animal",
#             "fur",
#         ],
#     },
#     "ridinghorse": {
#         "positive": [
#             "a person riding a horse",
#             "sitting on a horse animal",  # Focus sull'animale
#         ],
#         "negative": [
#             "riding a bicycle",  # Hard Negative
#             "wheels",
#             "handlebars",
#             "machine",
#         ],
#     },
#     "running": {
#         "positive": [
#             "a person running fast",
#             "sprinting motion",  # Implica velocità/blur
#         ],
#         "negative": [
#             "walking slowly",
#             "standing still",
#             "jogging in place",
#             "jumping vertically",
#         ],
#     },
#     "takingphoto": {
#         "positive": [
#             "a person taking a picture",
#             "holding a camera to the eye",  # Posa unica
#             "camera lens",
#         ],
#         "negative": [
#             "talking on a phone",  # Confusione classica
#             "looking through binoculars",
#             "drinking",
#         ],
#     },
#     "usingcomputer": {
#         "positive": [
#             "a person using a computer",
#             "hands on a keyboard and mouse",  # Oggetti inconfondibili
#             "looking at a monitor screen",
#         ],
#         "negative": ["reading a book", "watching tv", "writing on paper", "television"],
#     },
#     "walking": {
#         "positive": [
#             "a person walking",
#             "a person walking casually on the street",
#         ],
#         "negative": ["running fast", "sprinting", "standing still", "sitting"],
#     },
# }

# PROMPT_CONFIG = {
#     "jumping": {
#         "positive": [
#             "a photo of a person jumping in the air",
#             "feet completely off the ground",
#             "knees bent in mid-air",
#             "an athlete leaping high",
#             "a person doing a jump",
#         ],
#         "negative": [
#             "a person standing on the ground",
#             "feet planted on the floor",
#             "a person walking",
#             "a person running on the ground",
#             "shadow on the ground",
#         ],
#     },
#     "phoning": {
#         "positive": [
#             "holding a smartphone to the ear",
#             "talking on a mobile phone",
#             "hand holding a black phone against face",
#             "making a phone call",
#             "screen of a mobile phone",
#         ],
#         "negative": [
#             "touching face with hand",
#             "scratching head",
#             "resting hand on cheek",
#             "drinking from a cup",
#             "adjusting glasses",
#             "holding a camera",
#         ],
#     },
#     "playinginstrument": {
#         "positive": [
#             "playing a musical instrument",
#             "fingers on guitar strings",
#             "blowing into a wind instrument",
#             "hands on piano keys",
#             "musician performing with instrument",
#             "holding a violin",
#         ],
#         "negative": [
#             "holding a weapon",
#             "holding a baseball bat",
#             "holding a stick",
#             "carrying a bag",
#             "hands in pockets",
#             "clapping hands",
#         ],
#     },
#     "reading": {
#         "positive": [
#             "looking down at an open book",
#             "reading a newspaper",
#             "eyes focused on text in a book",
#             "holding a magazine to read",
#         ],
#         "negative": [
#             "sleeping with head down",
#             "looking at a mobile phone",  # Crucial distinction
#             "looking at the ground",
#             "writing on paper",
#             "using a laptop",
#         ],
#     },
#     "ridingbike": {
#         "positive": [
#             "straddling a bicycle seat",
#             "sitting on a bike",
#             "hands gripping bicycle handlebars",
#             "feet on bicycle pedals",
#             "cyclist riding a bike",
#         ],
#         "negative": [
#             "walking next to a bicycle",
#             "pushing a bike by the handlebars",
#             "standing near a bike",
#             "riding a motorcycle",  # Hard negative
#             "repairing a bike",
#         ],
#     },
#     "ridinghorse": {
#         "positive": [
#             "sitting on a horse",
#             "equestrian riding a horse",
#             "horseback riding posture",
#             "legs straddling a horse",
#         ],
#         "negative": [
#             "standing next to a horse",
#             "grooming a horse",
#             "leading a horse by the rein",
#             "petting an animal",
#             "cowboy standing",
#         ],
#     },
#     "running": {
#         "positive": [
#             "running fast",
#             "sprinting with wide stride",
#             "jogging motion",
#             "runner in athletic gear",
#             "blurred motion of running",
#         ],
#         "negative": [
#             "walking slowly",
#             "standing still",
#             "waiting in line",
#             "strolling casually",
#             "jumping in place",
#         ],
#     },
#     "takingphoto": {
#         "positive": [
#             "holding a camera up to the eye",
#             "looking through camera viewfinder",
#             "photographer taking a picture",
#             "holding a DSLR camera",
#             "pointing a camera lens",
#         ],
#         "negative": [
#             "looking through binoculars",
#             "drinking from a bottle",
#             "adjusting sunglasses",
#             "pointing with finger",
#             "talking on a phone",
#         ],
#     },
#     "usingcomputer": {
#         "positive": [
#             "typing on a laptop keyboard",
#             "staring at a computer monitor",
#             "sitting at a desk using a computer",
#             "hands on mouse and keyboard",
#             "working in an office cubicle",
#         ],
#         "negative": [
#             "watching television",
#             "reading a book",
#             "writing with a pen on paper",
#             "eating at a table",
#             "playing a board game",
#         ],
#     },
#     "walking": {
#         "positive": [
#             "walking normally",
#             "pedestrian strolling on street",
#             "taking a step forward",
#             "walking slowly",
#         ],
#         "negative": [
#             "running fast",
#             "sprinting",
#             "standing perfectly still",
#             "sitting on a bench",
#             "riding a bike",
#         ],
#     },
# }

PROMPT_CONFIG = {
    "jumping": {
        "positive": [
            "a photo of a person jumping in mid-air",
            "a photo of a person feet off the ground",
            "a photo of a person knees bent while airborne",
        ],
        "negative": [
            "person standing on the ground",
            "person walking",
            "person running on the ground",
        ],
    },
    "phoning": {
        "positive": [
            "a photo of a person holding a smartphone to the ear",
            "a photo of a person making a phone call",
            "a photo of a person mobile phone near face",
        ],
        "negative": [
            "person touching their hair",
            "person drinking from a cup",
            "person adjusting glasses",
        ],
    },
    "playinginstrument": {
        "positive": [
            "a photo of a person playing a musical instrument",
            # "hands on a musical instrument",
            "a photo of a musician performing",
        ],
        "negative": [
            "person holding a weapon",
            "person carrying a bag",
            "person with hands in pockets",
        ],
    },
    "reading": {
        "positive": [
            "a photo of a person reading a book",
            "a photo of a person looking at an open book",
            "a photo of a person eyes focused on text",
        ],
        "negative": [
            "person looking at a mobile phone",
            "person sleeping",
            "person writing on paper",
        ],
    },
    "ridingbike": {
        "positive": [
            "a photo of a person riding a bicycle",
            "a photo of a person hands on bicycle handlebars",
            "a photo of a person riding a bike",
            "a photo of a person riding a motorcycle",
        ],
        "negative": [
            "person walking",
            "person standing still",
        ],
    },
    "ridinghorse": {
        "positive": [
            "a photo of a person riding a horse",
            "a photo of a person sitting on a horse",
            "a photo of a person holding horse reins",
        ],
        "negative": [
            "person standing next to a horse",
            "person leading a horse",
        ],
    },
    "running": {
        "positive": [
            "a photo of a person running fast",
            "a photo of a person sprinting",
            "a photo of a person jogging",
        ],
        "negative": [
            "person walking",
            "person standing still",
            "person sitting",
        ],
    },
    "takingphoto": {
        "positive": [
            "a photo of a person holding a camera to the eye",
            "a photo of a person taking a photograph",
            "a photo of a person looking through a viewfinder",
        ],
        "negative": [
            "person holding binoculars",
            "person pointing with a finger",
            "person hands empty",
        ],
    },
    "usingcomputer": {
        "positive": [
            "a photo of a person typing on a keyboard",
            "a photo of a person looking at a computer monitor",
            "a photo of a person hands on mouse and keyboard",
        ],
        "negative": [
            "person reading a book",
            "person watching television",
            "person writing with a pen",
        ],
    },
    "walking": {
        "positive": [
            "a photo of a person walking",
            "a photo of a person taking a step",
            "a photo of a person walking slowly",
        ],
        "negative": [
            "person running",
            "person standing still",
            "person sitting",
        ],
    },
}


# --- 3. FUNZIONI DI EMBEDDING ---


# def get_action_embedding(clip_model, action, device):
#     prompts = PROMPT_CONFIG[action]["positive"]
#     prompts.append(f"a photo of a person {action}.")
#     prompts.append(f"a clean origami of a person {action}.")

#     weights = []
#     with torch.no_grad():
#         for p in prompts:
#             w = tokenize_text_prompt(clip_model, p).to(device)
#             weights.append(w)
#     return torch.mean(torch.stack(weights), dim=0)


def get_action_embedding(clip_model, action, device):
    # Usiamo SOLO i prompt specifici definiti nel dizionario
    prompts = PROMPT_CONFIG[action]["positive"].copy()

    # NIENTE PIÙ "origami" o frasi generiche che diluiscono il segnale.
    # Ci fidiamo dei prompt discriminativi ("handlebars", "horse animal", ecc.)

    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_negative_embedding(clip_model, action, device):
    base_negatives = BASE_BACKGROUND.copy()
    specific_negatives = PROMPT_CONFIG[action]["negative"]
    other_actions = [f"a person {act}" for act in VOC_ACTIONS if act != action]
    final_negatives = base_negatives + specific_negatives + other_actions

    weights = []
    with torch.no_grad():
        for p in final_negatives:
            # if "person" in p or "walking" in p or "standing" in p:
            #     text = p
            # else:
            text = f"a photo of {p}" if "photo" not in p else p
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
    scales = [448, 560, 672]

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

                g_min = min(map_pos.min(), map_neg.min())
                g_max = max(map_pos.max(), map_neg.max())
                denom = g_max - g_min + 1e-8

                norm_pos = (map_pos - g_min) / denom
                norm_neg = (map_neg - g_min) / denom

                probs = F.softmax(torch.stack([norm_neg, norm_pos]) * 20, dim=0)
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

    # --- NEW: VISUALIZZAZIONE INTERMEDIA HEATMAPS ---
    print("\n--- VISUALIZING INTERMEDIATE HEATMAPS ---")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("B-Cos Heatmaps (Binary Softmax Probabilities)", fontsize=16)
    axes = axes.flatten()

    for idx, action in enumerate(VOC_ACTIONS):
        ax = axes[idx]
        heatmap = clean_heatmaps[action]
        im = ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
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
    print("Voting using Top-25% Intensity...")
    final_results = []

    for idx, mask in enumerate(valid_masks):
        best_act = "unknown"
        best_score = -1.0
        debug_scores = {}

        for action in VOC_ACTIONS:
            heatmap = clean_heatmaps[action]

            # --- MODIFICA QUI: TOP-K INVECE DI MEAN ---
            # Prendiamo il top 25% dei pixel più attivi nella maschera
            score = calculate_topk_score(heatmap, mask, top_k_percent=0.20)
            # ------------------------------------------

            debug_scores[action] = score

            if score > best_score:
                best_score = score
                best_act = action

        # La soglia può essere alzata leggermente perché i punteggi top-k saranno più alti della media
        if best_score > 0.60:
            final_results.append((mask, best_act, best_score))
            print(f"   Mask {idx}: {best_act} (Top-K Conf: {best_score:.1%})")
        else:
            print(f"   Mask {idx}: Rejected (Top-K Conf {best_score:.1%} too low)")
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
