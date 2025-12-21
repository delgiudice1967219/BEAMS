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

# --- 1. CONFIGURATION AND IMPORTS ---
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
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003960.jpg"
OUTPUT_DIR = "sam_bcos_voc_refined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512  # Resolution for the crops

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
    "unknown": (0.5, 0.5, 0.5),
    "background": (0.3, 0.3, 0.3),
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
            "person holding a smartphone to the ear",
            "person talking on a mobile phone",
            "person using a telephone",
            "person holding a phone against face",
            "person making a phone call",
            # "screen of a mobile phone",
            "person phoning",
        ],
        "negative": [
            # "touching face with hand",
            "scratching head",
            "resting hand on cheek",
            "drinking from a cup",
            "adjusting glasses",
            "looking down at the phone",
            # "holding a camera",
        ],
    },
    "playinginstrument": {
        "positive": [
            "person playing a musical instrument",
            "person with fingers on guitar strings",
            "person blowing into a wind instrument",
            "person with hands on piano keys",
            "musician performing with instrument",
            "person holding a violin",
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
            "person looking down at an open book",
            "person reading a newspaper",
            "eyes focused on text in a book",
            "person holding a magazine to read",
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
            # "feet on bicycle pedals",
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
            "person riding a horse",
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
            "person running",
            "sprinting with wide stride",
            "person in jogging motion",
            "runner in athletic gear",
            # "blurred motion of running",
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
            "person typing on a laptop keyboard",
            # "staring at a computer monitor",
            "sitting at a desk using a computer",
            "person with hands on mouse and keyboard",
            "person looking at a computer screen",
            # "working in an office cubicle",
            "person using a computer",
        ],
        "negative": [
            "watching television",
            "reading a book",
            "writing with a pen on paper",
            "eating at a table",
            "playing a board game",
            "person talking with a phone in hand",
        ],
    },
    "walking": {
        "positive": [
            "walking normally",
            "pedestrian strolling on street",
            "person taking a step forward",
            "walking slowly",
            "person walking",
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

# --- 3. HELPER FUNCTIONS ---


def get_person_crop(image, mask, context_pct=0.4):
    """
    Extracts a crop of the person from the image based on the SAM mask.
    Adds 'context_pct' padding to include surrounding objects (bikes, chairs, etc).
    Returns: cropped_image (PIL), cropped_mask (numpy), bbox (x1,y1,x2,y2)
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return image, mask, (0, 0, image.width, image.height)

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    h_box = y_max - y_min
    w_box = x_max - x_min

    # Add Context Padding
    pad_y = int(h_box * context_pct)
    pad_x = int(w_box * context_pct)

    y1 = max(0, y_min - pad_y)
    y2 = min(image.height, y_max + pad_y)
    x1 = max(0, x_min - pad_x)
    x2 = min(image.width, x_max + pad_x)

    img_crop = image.crop((x1, y1, x2, y2))
    mask_crop = mask[y1:y2, x1:x2]

    return img_crop, mask_crop, (x1, y1, x2, y2)


def get_global_background_embedding(clip_model, device):
    bg_prompts = [f"a photo of {bg}" for bg in BASE_BACKGROUND]
    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p, templates=["{}"]).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_action_embedding(clip_model, action, device):
    prompts = list(PROMPT_CONFIG[action]["positive"])
    prompts += [f"a photo of a person {action}.", f"the {action}."]
    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p, templates=["{}"]).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def compute_raw_bcos_map(model, img_tensor, weight, blur_transform, device):
    # Multi-scale for robustness on the crop
    scales = [224, 336, 448, 512]
    accumulated_maps = []

    for s in scales:
        resize_t = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )
        img_scaled = resize_t(img_tensor).to(device)

        with torch.no_grad():
            _, attribution, _, _ = compute_attributions(model, img_scaled, weight)
            if isinstance(attribution, np.ndarray):
                attribution = torch.from_numpy(attribution).to(device)
            if attribution.dim() == 3:
                attribution = attribution[0]

            attribution = blur_transform(attribution.unsqueeze(0)).squeeze()

            # Resize back to fixed IMG_SIZE (512)
            attr_resized = F.interpolate(
                attribution.unsqueeze(0).unsqueeze(0),
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            accumulated_maps.append(attr_resized)

    return torch.mean(torch.stack(accumulated_maps), dim=0)


def show_mask_custom(mask, ax, color, alpha=0.55):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


def score_action(
    diff_map, mask, mean_other_maps, top_q=(0.7, 0.95), lambda_c=0.7, beta_l=0.6
):
    """
    Scores the action based on the heatmap logic.
    Refined for crops: mask is the binary person mask inside the crop.
    """
    vals = diff_map[mask]
    if len(vals) < 10:
        return -np.inf

    # Support (How strong is the signal in the mask?)
    lo, hi = np.quantile(vals, top_q)
    support_vals = vals[(vals >= lo) & (vals <= hi)]
    S = np.mean(support_vals) if len(support_vals) > 0 else 0.0

    # Contra-indication (Are there negative regions?)
    neg_vals = vals[vals < 0]
    C = -np.mean(neg_vals) if len(neg_vals) > 0 else 0.0

    # Selectivity (Is it better than other actions?)
    eps = 1e-6
    pos_map = np.maximum(diff_map, 0)
    baseline = np.maximum(mean_other_maps, 0)
    selective = pos_map / (baseline + eps)
    sel_vals = selective[mask]
    if len(sel_vals) > 0:
        sel_top = np.quantile(sel_vals, 0.8)
        L = np.mean(sel_vals[sel_vals >= sel_top])
    else:
        L = 0.0

    score = S - lambda_c * C + beta_l * np.log1p(L)
    return score


# --- 4. MAIN PIPELINE ---


def main():
    print(f"--- STARTING INSTANCE-BASED REFINED PIPELINE ---")

    # A. LOAD MODELS
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"Errore SAM3: {e}")
        return

    print("Loading SAM3...")
    sam_model = (
        build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH).to(DEVICE).eval()
    )
    processor = Sam3Processor(sam_model)

    print("Loading B-Cos & CLIP...")
    bcos_model, _ = load_bcos_model()
    bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    clip_model.to(DEVICE).eval()
    print("✅ Models Loaded.")

    # B. SAM SEGMENTATION
    raw_image = Image.open(IMG_PATH).convert("RGB")
    W, H = raw_image.size
    print(f"Processing Image: {W}x{H}")

    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")
    masks_tensor = output["masks"]
    scores_tensor = output["scores"]

    valid_masks = []
    for i in range(len(scores_tensor)):
        if (
            scores_tensor[i].item() > 0.15
        ):  # Slightly higher threshold for person detection
            m = masks_tensor[i].cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
            valid_masks.append(m > 0)

    print(f"Found {len(valid_masks)} valid person masks.")

    # C. PRE-COMPUTE EMBEDDINGS (Optimization)
    print("Pre-computing CLIP embeddings...")
    bg_weight = get_global_background_embedding(clip_model, DEVICE)
    action_weights = {}
    for action in VOC_ACTIONS:
        action_weights[action] = get_action_embedding(clip_model, action, DEVICE)

    # D. INSTANCE-LEVEL B-COS LOOP
    final_results = []

    # Transform for the Crops (Standard 512x512)
    crop_prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    blur_kernel = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    for idx, mask in enumerate(valid_masks):
        print(f"\n--- Analyzing Person {idx + 1} ---")

        # 1. CROP: Zoom in on this person + context
        img_crop, mask_crop, bbox = get_person_crop(raw_image, mask, context_pct=0.4)

        # Convert crop to tensor for model
        crop_tensor = crop_prep(img_crop)

        # Resize mask to IMG_SIZE (512) to match heatmap dimensions for scoring
        mask_crop_resized = (
            cv2.resize(
                mask_crop.astype(np.uint8),
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

        # 2. COMPUTE MAPS FOR CROP
        # Compute Background Map
        bg_map = compute_raw_bcos_map(
            bcos_model, crop_tensor, bg_weight, blur_kernel, DEVICE
        )

        # Compute Action Maps
        maps_list = [bg_map]  # Index 0
        for action in VOC_ACTIONS:
            w = action_weights[action]
            m = compute_raw_bcos_map(bcos_model, crop_tensor, w, blur_kernel, DEVICE)
            maps_list.append(m)

        # 3. SOFTMAX COMPETITION (Local to this person)
        stack = torch.stack(maps_list, dim=0)  # [11, 512, 512]

        # Normalize
        g_min = stack.min()
        g_max = stack.max()
        stack_norm = (stack - g_min) / (g_max - g_min + 1e-8)

        # Apply Softmax (Temp=15 for sharpness)
        probs = F.softmax(stack_norm * 15, dim=0).cpu().numpy()

        # 4. SCORING
        best_act = "unknown"
        best_score = -1e9

        # Iterate actions (Indices 1 to 10)
        for i, action in enumerate(VOC_ACTIONS):
            current_map = probs[i + 1]  # +1 because 0 is bg

            # Mean of other action maps for selectivity
            # Indices of other actions in 'probs' array
            other_indices = [k + 1 for k in range(len(VOC_ACTIONS)) if k != i]
            mean_other = np.mean(probs[other_indices], axis=0)

            score = score_action(
                current_map, mask_crop_resized, mean_other  # Use the crop mask!
            )

            if score > best_score:
                best_score = score
                best_act = action

        print(f"   Best Action: {best_act.upper()} (Score: {best_score:.3f})")

        # Threshold check
        if best_score > 0.05:
            final_results.append((mask, best_act, best_score))
        else:
            final_results.append((mask, "unknown", best_score))

    # E. FINAL VISUALIZATION
    print("\nVisualizing Final Result...")
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    # Sort results by score just for stable layering
    final_results.sort(key=lambda x: x[2])
    used_legends = set()

    for mask, action, score in final_results:
        if action == "unknown":
            color = (0.5, 0.5, 0.5)  # Gray for unknown
            label_text = f"Unknown\n{score:.2f}"
        else:
            color = ACTION_COLORS.get(action, (1.0, 1.0, 1.0))
            label_text = f"{action}\n{score:.2f}"
            used_legends.add(action)

        show_mask_custom(mask, ax, color, alpha=0.5)

        # Place label at centroid
        ys, xs = np.where(mask)
        if len(ys) > 0:
            cx, cy = np.mean(xs), np.mean(ys)
            ax.text(
                cx,
                cy,
                label_text,
                color="white",
                fontsize=9,
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
    plt.title("SAM3 + B-Cos (Instance-Based Recognition)")

    out_path = os.path.join(OUTPUT_DIR, "final_voc_result_instance.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved final result to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
