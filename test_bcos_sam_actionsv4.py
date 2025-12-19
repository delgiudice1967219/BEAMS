import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.patches as mpatches
import gc

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
# Sostituisci con la tua immagine target
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005010.jpg"

OUTPUT_DIR = "sam_bcos_smart_resize"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512  # Input size per B-Cos

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

# --- 2. PROMPT CONFIGURATION (Standard VOC) ---
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

# PROMPT_CONFIG = {
#     "jumping": {
#         "positive": [
#             "jumping",
#             "person mid-air",
#             "with feet off the ground",
#             "leaping",
#         ],
#         "negative": [
#             "person standing",
#             "person walking",
#             "person sitting",
#             "feet on ground",
#         ],
#     },
#     "phoning": {
#         "positive": ["using a mobile phone", "having a smartphone near ear"],
#         "negative": [
#             "person touching face",
#             "person hand near head",
#             "person listening",
#         ],
#     },
#     "playinginstrument": {
#         "positive": [
#             "playing a musical instrument",
#             "playing instrument",
#             "holding a guitar",
#             "playing flute",
#             "musician",
#         ],
#         "negative": ["person walking", "person running"],
#     },
#     "reading": {
#         "positive": [
#             "reading a book",
#             "reading a newspaper",
#             "looking at text",
#             "reding",
#         ],
#         "negative": ["person sleeping"],
#     },
#     "ridingbike": {
#         "positive": ["riding a bicycle", "cyclist on bike", "pedaling", "riding bike"],
#         "negative": ["walking next to bike", "standing near bike", "person walking"],
#     },
#     "ridinghorse": {
#         "positive": ["riding a horse", "equestrian on horse"],
#         "negative": ["standing next to horse", "grooming horse", "generic person"],
#     },
#     "running": {
#         "positive": ["running", "sprinting", "jogging"],
#         "negative": ["person walking", "person standing", "person stopped"],
#     },
#     "takingphoto": {
#         "positive": [
#             "taking a photo",
#             "with camera",
#             "photographer shooting",
#         ],
#         "negative": [
#             "person looking",
#             "person running",
#             "person walking",
#         ],
#     },
#     "usingcomputer": {
#         "positive": [
#             "typing on laptop",
#             "using computer keyboard",
#             "looking at monitor",
#         ],
#         "negative": ["sitting at desk", "watching tv", "generic person"],
#     },
#     "walking": {
#         "positive": ["walking person", "person going slowly", "person having a walk"],
#         "negative": ["running person", "sitting person"],
#     },
# }

PROMPT_CONFIG = {
    "jumping": {
        "positive": [
            "jumping in the air",
            "feet completely off the ground",
            "knees bent in mid-air",
            "leaping high",
            "doing a jump",
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
        ],
        "negative": [
            # "person touching face with hand",
            "person scratching head",
            "person resting hand on cheek",
            "person drinking from a cup",
            "person adjusting glasses",
            "person holding a camera",
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
            "person holding a weapon",
            "person holding a baseball bat",
            "person holding a stick",
            "person carrying a bag",
            "person hands in pockets",
            "person clapping hands",
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
            "person sleeping with head down",
            "person looking at a mobile phone",  # Crucial distinction
            "person looking at the ground",
            "person writing on paper",
            "person using a laptop",
        ],
    },
    "ridingbike": {
        "positive": [
            "straddling a bicycle seat",
            "sitting on a bike",
            "hands gripping bicycle handlebars",
            "feet on bicycle pedals",
            "cyclist riding a bike",
            "riding a motorcycle",
        ],
        "negative": [
            "person walking next to a bicycle",
            "person pushing a bike by the handlebars",
            "person standing near a bike",
            "person repairing a bike",
        ],
    },
    "ridinghorse": {
        "positive": [
            "sitting on a horse",
            "equestrian riding a horse",
            "horseback riding posture",
            "legs straddling a horse",
            "riding a horse",
        ],
        "negative": [
            "person standing next to a horse",
            "person grooming a horse",
            "person leading a horse by the rein",
            "person petting an animal",
        ],
    },
    "running": {
        "positive": [
            "running",
            "sprinting with wide stride",
            "in jogging motion",
            "runner in athletic gear",
            "in blurred motion of running",
        ],
        "negative": [
            "person walking slowly",
            "person standing still",
            "person waiting in line",
            "person strolling casually",
            "person jumping in place",
        ],
    },
    "takingphoto": {
        "positive": [
            "holding a camera up to the eye",
            "looking through camera viewfinder",
            "photographer taking a picture",
            "holding a DSLR camera",
            "pointing a camera lens",
            "taking a photo",
        ],
        "negative": [
            "person looking through binoculars",
            "person drinking from a bottle",
            "person adjusting sunglasses",
            "person pointing with finger",
            "person talking on a phone",
        ],
    },
    "usingcomputer": {
        "positive": [
            "typing on a laptop keyboard",
            "staring at a computer monitor",
            "sitting at a desk using a computer",
            "hands on mouse and keyboard",
            "working in an office cubicle",
            "using computer",
        ],
        "negative": [
            "person watching television",
            "person reading a book",
            "person writing with a pen on paper",
            "person eating at a table",
            "person playing a board game",
        ],
    },
    "walking": {
        "positive": [
            "walking normally",
            "pedestrian strolling on street",
            "taking a step forward",
            "walking slowly",
            "walking",
        ],
        "negative": [
            "person running fast",
            "person sprinting",
            "person standing perfectly still",
            "person sitting on a bench",
            "person riding a bike",
        ],
    },
}

# --- 3. HELPER FUNCTIONS ---


def smart_resize_square(img, target_size=512, is_mask=False):
    """
    Ridimensiona l'immagine mantenendo l'aspect ratio e aggiungendo padding (nero)
    per renderla quadrata (512x512). Questo evita lo stretching.

    Args:
        img: PIL Image
        target_size: int (es. 512)
        is_mask: bool (se True, usa Nearest Neighbor, altrimenti Bicubic)
    """
    w, h = img.size

    # Calcola il fattore di scala per adattare il lato più lungo al target_size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Metodo di resize: Nearest per le maschere (per non avere valori decimali), Bicubic per le immagini
    resample_method = Image.NEAREST if is_mask else Image.BICUBIC
    img_resized = img.resize((new_w, new_h), resample=resample_method)

    # Crea un canvas quadrato nero (o False per le maschere)
    if is_mask:
        new_img = Image.new(
            "L", (target_size, target_size), 0
        )  # L = 8-bit pixels, black background
    else:
        new_img = Image.new(
            "RGB", (target_size, target_size), (0, 0, 0)
        )  # RGB black background

    # Incolla l'immagine ridimensionata al centro
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img, (paste_x, paste_y, new_w, new_h)


def get_padded_crop(image, mask, padding_pct=0.3):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None

    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)

    h_box = y_max - y_min
    w_box = x_max - x_min

    pad_y = int(h_box * padding_pct)
    pad_x = int(w_box * padding_pct)

    W_img, H_img = image.size
    y1 = max(0, y_min - pad_y)
    y2 = min(H_img, y_max + pad_y)
    x1 = max(0, x_min - pad_x)
    x2 = min(W_img, x_max + pad_x)

    crop_img = image.crop((x1, y1, x2, y2))
    return crop_img, (x1, y1, x2, y2)


def calculate_topk_score(heatmap, mask, top_k_percent=0.35):
    values = heatmap[mask > 0]  # Prendi solo i valori dove la maschera è True
    if len(values) == 0:
        return 0.0
    if len(values) < 10:
        return np.mean(values)

    k = int(len(values) * top_k_percent)
    if k < 1:
        k = 1
    top_values = np.partition(values, -k)[-k:]
    return np.mean(top_values)


def get_action_embedding(clip_model, action, device):
    prompts = PROMPT_CONFIG[action]["positive"] + [
        f"a photo of a person {action}.",
        f"a clean origami of a person {action}.",
    ]
    weights = [tokenize_text_prompt(clip_model, p).to(device) for p in prompts]
    return torch.mean(torch.stack(weights), dim=0)


def get_negative_embedding(clip_model, action, device):
    negatives = BASE_BACKGROUND + PROMPT_CONFIG[action]["negative"]
    weights = []
    with torch.no_grad():
        for p in negatives:
            text = (
                p
                if any(x in p for x in ["person", "walking", "standing"])
                else f"a photo of {p}"
            )
            weights.append(tokenize_text_prompt(clip_model, text).to(device))
    return torch.mean(torch.stack(weights), dim=0)


def show_mask_custom(mask, ax, color, alpha=0.55):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 4. MAIN PIPELINE ---


def main():
    print(f"--- STARTING SMART-RESIZE PIPELINE (NO STRETCHING) ---")

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
    W_orig, H_orig = raw_image.size

    # B. SAM SEGMENTATION
    print("Running SAM3 on full image...")
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")
    masks_tensor = output["masks"]
    scores_tensor = output["scores"]

    valid_masks = []
    for i in range(len(scores_tensor)):
        if scores_tensor[i].item() > 0.10:
            m = masks_tensor[i].cpu().numpy().squeeze()
            if m.shape != (H_orig, W_orig):
                m = cv2.resize(
                    m.astype(np.uint8),
                    (W_orig, H_orig),
                    interpolation=cv2.INTER_NEAREST,
                )
            valid_masks.append(m > 0)
    print(f"Found {len(valid_masks)} person masks.")

    # --- MEMORY CLEANUP: DELETE SAM3 ---
    print("Removing SAM3 from memory...")
    del sam_model
    del processor
    del inference_state
    del output
    # Clear CUDA cache to free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Force garbage collection
    gc.collect()
    print("✅ SAM3 removed. GPU memory cleared.")
    # C. CROP & CLASSIFY LOOP
    # IMPORTANTE: Rimuoviamo il Resize dalla trasformazione base perché lo facciamo noi "smart"
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )

    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    # Scale relative al canvas 512x512
    scales = [448, 560, 672]

    final_results = []

    for idx, mask in enumerate(valid_masks):
        print(f"\n--- Analyzing Person {idx} ---")

        # 1. Estrarre il Crop (rettangolare) con padding
        crop_pil, (x1, y1, x2, y2) = get_padded_crop(raw_image, mask, padding_pct=0.25)
        if crop_pil is None:
            continue

        # 2. Estrarre la maschera locale (rettangolare)
        local_mask_orig = mask[y1:y2, x1:x2]
        local_mask_pil = Image.fromarray(local_mask_orig.astype(np.uint8) * 255)

        # 3. SMART RESIZE (Letterbox) - Niente stretching!
        # Ridimensioniamo sia l'immagine RGB che la maschera nello stesso identico modo
        crop_square, _ = smart_resize_square(
            crop_pil, target_size=IMG_SIZE, is_mask=False
        )
        mask_square, _ = smart_resize_square(
            local_mask_pil, target_size=IMG_SIZE, is_mask=True
        )

        # Convertiamo la maschera quadrata in numpy booleano per il calcolo dello score
        mask_square_np = np.array(mask_square) > 128

        # 4. Preparazione Tensore per B-Cos
        img_tensor = base_transform(crop_square)  # [6, 512, 512]

        best_act_crop = "unknown"
        best_score_crop = -1.0
        best_heatmap_viz = None

        for action in VOC_ACTIONS:
            pos_w = get_action_embedding(clip_model, action, DEVICE)
            neg_w = get_negative_embedding(clip_model, action, DEVICE)

            accumulated_maps = []

            for s in scales:
                # Resize del tensore già quadrato (che ora è sicuro e non distorto)
                resize_t = transforms.Resize(
                    (s, s), interpolation=transforms.InterpolationMode.BICUBIC
                )
                img_scaled = resize_t(img_tensor).to(DEVICE)

                with torch.no_grad():
                    _, map_pos, _, _ = compute_attributions(
                        bcos_model, img_scaled, pos_w
                    )
                    _, map_neg, _, _ = compute_attributions(
                        bcos_model, img_scaled, neg_w
                    )

                    # Fix dimensioni per Blur
                    t_pos = torch.as_tensor(map_pos).cpu().float()
                    t_neg = torch.as_tensor(map_neg).cpu().float()
                    while t_pos.dim() > 2:
                        t_pos = t_pos.squeeze(0)
                    while t_neg.dim() > 2:
                        t_neg = t_neg.squeeze(0)

                    map_pos = blur(t_pos.unsqueeze(0)).squeeze(0)
                    map_neg = blur(t_neg.unsqueeze(0)).squeeze(0)

                    # Binary Softmax
                    g_min = min(map_pos.min(), map_neg.min())
                    g_max = max(map_pos.max(), map_neg.max())
                    denom = g_max - g_min + 1e-8
                    norm_pos = (map_pos - g_min) / denom
                    norm_neg = (map_neg - g_min) / denom

                    probs = F.softmax(torch.stack([norm_neg, norm_pos]) * 30, dim=0)
                    prob_map_action = probs[1]

                    # Interpolate back to 512x512
                    res = F.interpolate(
                        prob_map_action[None, None],
                        size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear",
                    ).squeeze()
                    accumulated_maps.append(res)

            final_heatmap = torch.mean(torch.stack(accumulated_maps), dim=0).numpy()

            # 5. SCORING: TOP-35% sulla maschera quadrata corretta
            score = calculate_topk_score(
                final_heatmap, mask_square_np, top_k_percent=0.20
            )

            if score > best_score_crop:
                best_score_crop = score
                best_act_crop = action
                best_heatmap_viz = final_heatmap

        # 6. SALVATAGGIO DEBUG
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(crop_square)
        # plt.title(f"P{idx}: {crop_pil.size} -> 512x512 (Pad)")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(mask_square_np, cmap="gray")
        # plt.title("Synced Mask")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # if best_heatmap_viz is not None:
        #     plt.imshow(best_heatmap_viz, cmap="jet", vmin=0, vmax=1)
        #     plt.title(f"{best_act_crop}: {best_score_crop:.1%}")
        # plt.axis("off")

        # crop_save_path = os.path.join(OUTPUT_DIR, f"debug_p{idx}_{best_act_crop}.png")
        # plt.savefig(crop_save_path, bbox_inches="tight")
        # plt.close()

        # 7. RISULTATO FINALE
        if best_score_crop > 0.50:
            final_results.append((mask, best_act_crop, best_score_crop))
            print(f"   -> {best_act_crop} (Conf: {best_score_crop:.1%})")
        else:
            print(f"   -> Rejected (Conf {best_score_crop:.1%} too low)")

    # D. VISUALIZZAZIONE FINALE
    print("\nVisualizing Final Global Result...")
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
    plt.title("SAM3 + Smart-Square B-Cos (No Stretching)")

    out_path = os.path.join(OUTPUT_DIR, "final_smart_result.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved final result to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
