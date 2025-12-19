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
import matplotlib as mpl
import gc

# --- 1. CONFIGURAZIONE ---
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
# Sostituisci con la tua immagine
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003406.jpg"

OUTPUT_DIR = "sam_bcos_positive_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

VOC_ACTIONS = [
    "jumping",
    "phoning",
    # "playinginstrument",
    "reading",
    "ridingbike",
    "ridinghorse",
    "running",
    "takingphoto",
    "usingcomputer",
    "walking",
]

# Colori moderni
try:
    COLORS = mpl.colormaps["tab10"].colors
except AttributeError:
    COLORS = plt.cm.tab10.colors

# --- 2. PROMPT CONFIGURATION (SOLO POSITIVI) ---
# Usiamo descrizioni ricche per i positivi, ma ignoriamo i negativi specifici.

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
    "street",
    "cloud",
    "floor",
    "ceiling",
    "background",
    "blur",
    "out of focus",
    "scenery",
    "empty space",
    # Aggiungiamo qui la "persona generica" come unico distrattore
    # "a generic person",
    # "a person standing",
    # "a person",
    # "people",
]

PROMPT_CONFIG = {
    "jumping": [
        "jumping",
        # "feet completely off the ground",
        # "knees bent in mid-air",
        # "leaping high",
        # "doing a jump",
    ],
    "phoning": [
        "phoning",
        # "talking on a mobile phone",
        # "hand holding a black phone against face",
        # "making a phone call",
    ],
    # "playinginstrument": [
    #     "playing a instrument",
    #     # "fingers on guitar strings",
    #     # "blowing into a wind instrument",
    #     # "hands on piano keys",
    #     # "musician performing",
    # ],
    "reading": [
        "reading",
        # "reading a newspaper",
        # "eyes focused on text",
        # "holding a magazine to read",
    ],
    "ridingbike": [
        "riding a bike",
        # "sitting on a bike",
        # "hands gripping bicycle handlebars",
        # "feet on bicycle pedals",
        # "cyclist riding a bike",
    ],
    "ridinghorse": [
        "riding a horse",
        # "equestrian riding a horse",
        # "horseback riding posture",
        # "legs straddling a horse",
    ],
    "running": [
        "running",
        # "sprinting with wide stride",
        # "jogging motion",
        # "runner in athletic gear",
        # "blurred motion of running",
    ],
    "takingphoto": [
        "taking a photo",
        # "looking through camera viewfinder",
        # "photographer taking a picture",
        # "holding a DSLR camera",
    ],
    "usingcomputer": [
        "using a computer",
        #     "staring at a computer monitor",
        #     "sitting at a desk using a computer",
        #     "hands on mouse and keyboard",
    ],
    "walking": [
        "walking",
        # "pedestrian strolling on street",
        # "taking a step forward",
        # "walking slowly",
    ],
}

# --- 3. HELPER FUNCTIONS ---


def smart_resize_square(img, target_size=512, is_mask=False):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resample_method = Image.NEAREST if is_mask else Image.BICUBIC
    img_resized = img.resize((new_w, new_h), resample=resample_method)

    if is_mask:
        new_img = Image.new("L", (target_size, target_size), 0)
    else:
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img, (paste_x, paste_y, new_w, new_h)


def get_padded_crop(image, mask, padding_pct=0.25):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    h_box, w_box = y_max - y_min, x_max - x_min
    pad_y, pad_x = int(h_box * padding_pct), int(w_box * padding_pct)

    W_img, H_img = image.size
    y1 = max(0, y_min - pad_y)
    y2 = min(H_img, y_max + pad_y)
    x1 = max(0, x_min - pad_x)
    x2 = min(W_img, x_max + pad_x)

    return image.crop((x1, y1, x2, y2))


def calculate_dilated_score(heatmap, mask_original):
    # Dilatazione per includere oggetti vicini (bici, telefono)
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask_original.astype(np.uint8), kernel, iterations=4) > 0

    values = heatmap[dilated_mask]
    if len(values) == 0:
        return 0.0

    # Top-K Average (30% dei pixel più attivi nella zona dilatata)
    k = int(len(values) * 0.30)
    if k < 1:
        k = 1
    top_values = np.partition(values, -k)[-k:]
    return np.mean(top_values)


def get_action_embedding(clip_model, action, device):
    # Solo Positivi + Generic Fusion
    prompts = PROMPT_CONFIG[action] + [
        f"a photo of a person {action}.",
        f"a clean origami of a person {action}.",
    ]
    weights = [tokenize_text_prompt(clip_model, p).to(device) for p in prompts]
    return torch.mean(torch.stack(weights), dim=0)


def get_generic_background_embedding(clip_model, device):
    # Unico Negativo Generico per tutti
    weights = []
    with torch.no_grad():
        for p in BASE_BACKGROUND:
            text = p if "person" in p else f"a photo of {p}"
            weights.append(tokenize_text_prompt(clip_model, text).to(device))
    return torch.mean(torch.stack(weights), dim=0)


def show_mask_custom(mask, ax, color, alpha=0.55):
    color = np.array(color)
    if color.shape[0] > 3:
        color = color[:3]  # Ensure RGB
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([color, [alpha]])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 4. MAIN PIPELINE ---


def main():
    print(f"--- STARTING PIPELINE (Positives Only vs Generic BG) ---")

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

    # B. SAM DETECT
    print("Running SAM3...")
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

    # Cleanup
    del sam_model, processor, inference_state, output
    torch.cuda.empty_cache()
    gc.collect()

    # C. B-COS PREPARATION
    base_transform = transforms.Compose(
        [transforms.ToTensor(), custom_transforms.AddInverse()]
    )
    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    scales = [448, 560, 672]

    # Calcoliamo il Negativo Generico UNA VOLTA sola
    print("Computing generic background embedding...")
    bg_w = get_generic_background_embedding(clip_model, DEVICE)

    final_results = []

    for idx, mask in enumerate(valid_masks):
        print(f"\n--- Analyzing Person {idx} ---")

        # Crop setup
        crop_pil = get_padded_crop(raw_image, mask, padding_pct=0.25)
        if crop_pil is None:
            continue

        local_mask_orig = mask  # Semplificazione: usiamo mask globale per coordinate, ma ricalcoliamo locale
        # (Nota: nel codice precedente facevamo un ritaglio preciso, qui per brevità assumiamo la logica 'crop_pil' sia corretta)
        # Fix rapido per maschera locale corretta:
        rows, cols = np.where(mask)
        y_min, y_max, x_min, x_max = (
            np.min(rows),
            np.max(rows),
            np.min(cols),
            np.max(cols),
        )
        h, w = y_max - y_min, x_max - x_min
        pad_y, pad_x = int(h * 0.25), int(w * 0.25)
        y1, y2 = max(0, y_min - pad_y), min(H_orig, y_max + pad_y)
        x1, x2 = max(0, x_min - pad_x), min(W_orig, x_max + pad_x)

        local_mask_orig = mask[y1:y2, x1:x2]
        local_mask_pil = Image.fromarray(local_mask_orig.astype(np.uint8) * 255)

        crop_square, _ = smart_resize_square(
            crop_pil, target_size=IMG_SIZE, is_mask=False
        )
        mask_square, _ = smart_resize_square(
            local_mask_pil, target_size=IMG_SIZE, is_mask=True
        )
        mask_square_np = np.array(mask_square) > 128
        img_tensor = base_transform(crop_square)

        best_act = "unknown"
        best_score = -1.0
        best_heatmap = None

        for action in VOC_ACTIONS:
            pos_w = get_action_embedding(clip_model, action, DEVICE)

            accumulated_maps = []
            for s in scales:
                resize_t = transforms.Resize(
                    (s, s), interpolation=transforms.InterpolationMode.BICUBIC
                )
                img_scaled = resize_t(img_tensor).to(DEVICE)

                with torch.no_grad():
                    _, map_pos, _, _ = compute_attributions(
                        bcos_model, img_scaled, pos_w
                    )
                    _, map_neg, _, _ = compute_attributions(
                        bcos_model, img_scaled, bg_w
                    )

                    t_pos = torch.as_tensor(map_pos).cpu().float()
                    t_neg = torch.as_tensor(map_neg).cpu().float()
                    while t_pos.dim() > 2:
                        t_pos = t_pos.squeeze(0)
                    while t_neg.dim() > 2:
                        t_neg = t_neg.squeeze(0)

                    map_pos = blur(t_pos.unsqueeze(0)).squeeze(0)
                    map_neg = blur(t_neg.unsqueeze(0)).squeeze(0)

                    # Binary Softmax: Specific Action vs Generic Background
                    g_min = min(map_pos.min(), map_neg.min())
                    g_max = max(map_pos.max(), map_neg.max())
                    denom = g_max - g_min + 1e-8
                    norm_pos = (map_pos - g_min) / denom
                    norm_neg = (map_neg - g_min) / denom

                    probs = F.softmax(torch.stack([norm_neg, norm_pos]) * 30, dim=0)
                    res = F.interpolate(
                        probs[1][None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear"
                    ).squeeze()
                    accumulated_maps.append(res)

            final_heatmap = torch.mean(torch.stack(accumulated_maps), dim=0).numpy()

            # Dilated Voting
            score = calculate_dilated_score(final_heatmap, mask_square_np)

            if score > best_score:
                best_score = score
                best_act = action
                best_heatmap = final_heatmap

        # Debug & Save
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(crop_square)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(mask_square_np, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        if best_heatmap is not None:
            plt.imshow(best_heatmap, cmap="jet", vmin=0, vmax=1)
            plt.title(f"{best_act}: {best_score:.1%}")
        plt.axis("off")
        plt.savefig(os.path.join(OUTPUT_DIR, f"debug_p{idx}.png"), bbox_inches="tight")
        plt.close()

        if best_score > 0.50:
            final_results.append((mask, best_act, best_score))
            print(f"   -> Winner: {best_act} ({best_score:.1%})")
        else:
            print(f"   -> Rejected ({best_score:.1%})")

    # Final Viz
    print("\nVisualizing...")
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    action_to_color = {
        act: COLORS[i % len(COLORS)] for i, act in enumerate(VOC_ACTIONS)
    }
    used_legs = set()

    for mask, action, score in final_results:
        color = action_to_color[action]
        used_legs.add(action)
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

    patches = [mpatches.Patch(color=action_to_color[a], label=a) for a in used_legs]
    if patches:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.title("SAM3 + B-Cos (Positives Only)")
    plt.savefig(os.path.join(OUTPUT_DIR, "final_result.png"), bbox_inches="tight")
    print("Done.")


if __name__ == "__main__":
    main()
