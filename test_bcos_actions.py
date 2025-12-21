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
from utils_voc_actions import parse_voc_xml, compute_iou, mask_to_bbox


path_to_sam3 = (
    "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3"
)
if path_to_sam3 not in sys.path:
    sys.path.append(path_to_sam3)

sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_utils import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

CHECKPOINT_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005438.jpg"  # flauti
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005572.jpg"  # matrimonio
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005706.jpg"  # phoning reading
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003817.jpg"  # violinista che legge
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006994.jpg"  # scooter e pedoni
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006767.jpg"  # mamma e figlia walking
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006375.jpg"  # cavallo in medio oriente
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005751.jpg"  # calciatrici
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006295.jpg"  # suonatore dentro al muro
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006089.jpg"  # bimba sul cavallo

### GALLERY ###

# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005220.jpg"  # GALLERY -> walking, bike
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_004775.jpg"  # GALLERY -> photo, instrument
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003298.jpg"  # GALLERY -> jumping, forse ce esempio migliore
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_004512.jpg"  # GALLERY -> horse, walking
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005256.jpg"  # GALLERY -> reading, photo
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006129.jpg"  # GALLERY -> phoning
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006095.jpg"  # GALLERY -> usingcomputer
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_003422.jpg"  # GALLERY -> running, bike
# IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005070.jpg"  # GALLERY -> running, walking


OUTPUT_DIR = "sam_bcos_voc_viz"
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
    "unknown": (0.5, 0.5, 0.5),
    "background": (0.3, 0.3, 0.3),
}


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
            "jumping in the air",
            "with feet completely off the ground",
            "with knees bent in mid-air",
            "leaping high",
            "doing a jump",
        ],
        "negative": [
            "standing on the ground",
            "with feet planted on the floor",
            "walking",
            "running on the ground",
            "shadow on the ground",
        ],
    },
    "phoning": {
        "positive": [
            "holding a smartphone to the ear",
            "talking on a mobile phone",
            "using a telephone",
            "holding a phone against face",
            "making a phone call",
            "phoning",
        ],
        "negative": [
            "scratching head",
            "resting hand on cheek",
            "drinking from a cup",
            "adjusting glasses",
            "looking down at the phone",
        ],
    },
    "playinginstrument": {
        "positive": [
            "playing a musical instrument",
            "with fingers on guitar strings",
            "blowing into a wind instrument",
            "with hands on piano keys",
            "performing with instrument",
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
            "looking at a mobile phone",
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
            "cyclist riding a bike",
        ],
        "negative": [
            "walking next to a bicycle",
            "pushing a bike by the handlebars",
            "standing near a bike",
            "riding a motorcycle",
            "repairing a bike",
        ],
    },
    "ridinghorse": {
        "positive": [
            "sitting on a horse",
            "riding a horse",
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
            "running",
            "sprinting with wide stride",
            "in jogging motion",
            "in athletic gear",
            # "moving with feet distant from each other",
            "with arms pumping",
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
            "sitting at a desk using a computer",
            "with hands on mouse and keyboard",
            "looking at a computer screen",
            "using a computer",
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
            # "strolling on street",
            # "taking a step forward",
            "walking slowly",
            "walking",
            # "moving on with feet near each other",
        ],
        "negative": [
            "running fast",
            "sprinting",
            "standing perfectly still",
            "sitting on a bench",
            "riding a bike",
            "jumping in the air",
            "with arms pumping",
        ],
    },
}


# --- 3. FUNCTIONS ---


def get_global_background_embedding(clip_model, device):
    """Computes a single robust Background embedding vector for Softmax competition"""
    bg_prompts = [f"a photo of {bg}" for bg in BASE_BACKGROUND]
    identity_template = ["{}"]

    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p, templates=identity_template).to(
                device
            )
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_action_embedding(clip_model, action, device):
    """Calcola il vettore del target"""
    prompts = list(PROMPT_CONFIG[action]["positive"])
    prompts += [
        f"a clean origami of a person with clothes,people,human {action}.",
        f"a photo of a person with clothes,people,human {action}.",
        f"a photo of the {action}.",
        f"a picture of a person with clothes,people,human {action}.",
        f"an image of a person with clothes,people,human {action}.",
        f"an image of the {action}.",
        f"the {action}.",
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


def get_negative_embedding(clip_model, action, device):
    """
    KEPT FOR LEGACY/SCORING if needed, though Softmax uses Global Background.
    """
    final_negatives = BASE_BACKGROUND.copy()
    specific_negatives = PROMPT_CONFIG[action]["negative"]
    final_negatives.extend(specific_negatives)
    identity_template = ["{}"]
    weights = []
    with torch.no_grad():
        for p in final_negatives:
            if "person" in p or "walking" in p or "standing" in p:
                text = p
            else:
                text = f"a photo of {p}"
            w = tokenize_text_prompt(clip_model, text, templates=identity_template).to(
                device
            )
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def compute_raw_bcos_map(model, img_tensor, weight, blur_transform, device):
    """Computes the raw B-Cos heatmap (averaged over scales)"""
    scales = [224, 448, 560, 672, 784]
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

            # Resize back to IMG_SIZE
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


def dilate_person_mask(mask, radius_px=18):
    mask_u8 = mask.astype(np.uint8) * 255
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1) > 0
    return dilated


def compute_ctx_radius_from_mask(mask, k=0.06, r_min=10, r_max=55):
    """
    Radius proporzionale alla dimensione della persona (bbox diagonal).
    k: percentuale della diagonale (0.05-0.08 tipico)
    r_min/r_max: clamp in pixel per stabilità
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return r_min

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    w = (x1 - x0) + 1
    h = (y1 - y0) + 1
    diag = (w * w + h * h) ** 0.5

    r = int(round(k * diag))
    r = max(r_min, min(r, r_max))
    return r


def calculate_topk_score(heatmap, mask, top_k_percent=0.25):
    values = heatmap[mask]
    if len(values) == 0:
        return 0.0
    if len(values) < 10:
        return np.mean(values)
    k = int(len(values) * top_k_percent)
    if k < 1:
        k = 1
    top_values = np.partition(values, -k)[-k:]
    return np.mean(top_values)


def score_action(
    diff_map, mask, mean_other_maps, top_q=(0.7, 0.95), lambda_c=0.7, beta_l=0.6
):
    eps = 1e-6
    vals = diff_map[mask]
    if len(vals) < 20:
        return -np.inf

    lo, hi = np.quantile(vals, top_q)
    support_vals = vals[(vals >= lo) & (vals <= hi)]
    S = np.mean(support_vals) if len(support_vals) > 0 else 0.0

    neg_vals = vals[vals < 0]
    C = -np.mean(neg_vals) if len(neg_vals) > 0 else 0.0

    pos_map = np.maximum(diff_map, 0)
    baseline = np.maximum(mean_other_maps, 0)

    selective = pos_map / (baseline + eps)
    sel_vals = selective[mask]
    sel_top = np.quantile(sel_vals, 0.8)
    L = np.mean(sel_vals[sel_vals >= sel_top]) if len(sel_vals) > 0 else 0.0

    score = S - lambda_c * C + beta_l * np.log1p(L)
    return score


# --- 4. MAIN PIPELINE ---
CTX_K = 0.06  # 0.06  # 6% della diagonale bbox (prova 0.05-0.08)
CTX_R_MIN = 10  # 10
CTX_R_MAX = 55  # 55
ALPHA_CTX = 0.7  # 0.7


def main():
    print(f"--- STARTING REFINED VOC PIPELINE (SOFTMAX LOGIC) ---")

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

    # --- FILTER MASKS: Match against Ground Truth People ---

    # 1. Deduce XML path from the image path
    # Assumes standard VOC folder structure (JPEGImages -> Annotations)
    xml_path = IMG_PATH.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

    gt_people = parse_voc_xml(xml_path)
    print(f"Ground Truth contains {len(gt_people)} people.")

    # Only filter if we actually found GT people
    if len(gt_people) > 0:
        gt_to_mask = {}
        used_masks = set()
        ious = np.zeros((len(gt_people), len(valid_masks)))

        # Calculate IoU for all pairs
        for i, gt in enumerate(gt_people):
            for j, mask in enumerate(valid_masks):
                bbox_mask = mask_to_bbox(mask)
                if bbox_mask:
                    ious[i, j] = compute_iou(gt["bbox"], bbox_mask)

        # Greedy Matching
        if ious.size > 0:
            # Sort pairs by IoU (descending)
            indices = np.dstack(
                np.unravel_index(np.argsort(ious.ravel())[::-1], ious.shape)
            )[0]

            for r, c in indices:
                if ious[r, c] < 0.5:  # IoU Threshold (same as benchmark)
                    break
                if r not in gt_to_mask and c not in used_masks:
                    gt_to_mask[r] = c
                    used_masks.add(c)

        # Overwrite valid_masks to keep ONLY matched masks
        matched_indices = sorted(gt_to_mask.values())
        valid_masks = [valid_masks[i] for i in matched_indices]
        print(f"Filtered down to {len(valid_masks)} masks that match GT.")
    else:
        print("No GT found for this image (or XML missing). Keeping all masks.")

    # --- END FILTERING ---

    # C. B-COS DYNAMIC HEATMAPS (SOFTMAX UPDATE)
    print("Generating Mutual Exclusive Heatmaps...")

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

    clean_heatmaps = {}

    # 1. Compute Global Background Weight & Map
    print("Computing Global Background Map...")
    bg_weight = get_global_background_embedding(clip_model, DEVICE)
    bg_map_raw = compute_raw_bcos_map(bcos_model, img_tensor, bg_weight, blur, DEVICE)

    # 2. Compute All Action Weights & Maps
    action_maps_raw = []
    print("Computing Action Maps...")
    for action in VOC_ACTIONS:
        w = get_action_embedding(clip_model, action, DEVICE)
        m = compute_raw_bcos_map(bcos_model, img_tensor, w, blur, DEVICE)
        action_maps_raw.append(m)

    # 3. Stack [Background, Act1, Act2, ..., ActN]
    all_maps_stack = torch.stack(
        [bg_map_raw] + action_maps_raw, dim=0
    )  # [11, 512, 512]

    # 4. Joint Normalization
    g_min = all_maps_stack.min()
    g_max = all_maps_stack.max()
    denom = g_max - g_min + 1e-8
    stack_norm = (all_maps_stack - g_min) / denom

    # 5. SOFTMAX (Mutual Exclusivity)
    # High Temperature (20) to make it sharp
    print("Applying Softmax...")
    probs = F.softmax(stack_norm * 20, dim=0)  # [11, 512, 512]

    # 6. Distribute Probs back to clean_heatmaps for scoring
    # Index 0 is Background, so Action i is at Index i+1
    for i, action in enumerate(VOC_ACTIONS):
        prob_map = probs[i + 1].cpu().numpy()
        # Resize to Original Image Size for Scoring
        prob_map_resized = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
        clean_heatmaps[action] = prob_map_resized

    # # --- VISUALIZZAZIONE INTERMEDIA HEATMAPS (KEPT AS REQUESTED) ---
    # print("\n--- VISUALIZING INTERMEDIATE HEATMAPS (SOFTMAX PROBS) ---")
    # fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    # fig.suptitle("B-Cos Heatmaps (Softmax Probabilities)", fontsize=16)
    # axes = axes.flatten()

    # for idx, action in enumerate(VOC_ACTIONS):
    #     ax = axes[idx]
    #     heatmap = clean_heatmaps[action]
    #     # Softmax output is 0..1, so we plot vmin=0, vmax=1
    #     im = ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    #     ax.set_title(action)
    #     ax.axis("off")

    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # heatmap_path = os.path.join(OUTPUT_DIR, "intermediate_heatmaps.png")
    # plt.savefig(heatmap_path, bbox_inches="tight")
    # print(f"📸 Saved intermediate heatmaps to: {heatmap_path}")

    # D. VOTING / SCORING (LOGIC PRESERVED)
    print("Voting using Coherent Action Score...")
    final_results = []

    # Pre-compute stack for mean calculation
    all_maps = np.stack([clean_heatmaps[a] for a in VOC_ACTIONS], axis=0)

    ### OLD with high scores e.g. 250%
    # for idx, person_mask in enumerate(valid_masks):
    #     best_act = "unknown"  ## TODO: METTERE UNKNOWN A GRIGIO
    #     best_score = -1e9

    #     # nuova mask: persona + contesto
    #     ctx_r = compute_ctx_radius_from_mask(
    #         person_mask, k=CTX_K, r_min=CTX_R_MIN, r_max=CTX_R_MAX
    #     )
    #     ctx_mask = dilate_person_mask(person_mask, radius_px=ctx_r)

    #     min_score = 0.0
    #     max_score = 0.0
    #     for i, action in enumerate(VOC_ACTIONS):
    #         diff_map = clean_heatmaps[action]
    #         mean_other = np.mean(np.delete(all_maps, i, axis=0), axis=0)

    #         # score sulla persona pura
    #         s_person = score_action(
    #             diff_map,
    #             person_mask,
    #             mean_other,
    #             lambda_c=0.7,
    #             beta_l=0.6,
    #         )

    #         # score su persona + oggetto vicino
    #         s_context = score_action(
    #             diff_map,
    #             ctx_mask,
    #             mean_other,
    #             lambda_c=0.7,
    #             beta_l=0.6,
    #         )

    #         # combinazione (non distruttiva)
    #         score = s_person + ALPHA_CTX * s_context

    #         if score < min_score:
    #             min_score = score
    #         if score > max_score:
    #             max_score = score

    #         if score > best_score:
    #             best_score = score
    #             best_act = action

    #     best_score = (best_score - min_score) / (max_score - min_score + 1e-8)
    #     if best_score > 0.1:  # 3.0
    #         final_results.append((person_mask, best_act, best_score))
    #         print(f"   Mask {idx}: {best_act} (Score: {best_score:.3f})")
    #     else:
    #         print(f"   Mask {idx}: Rejected (Score {best_score:.3f})")

    ### NEW with normalized scores
    CONFIDENCE_THRESHOLD = 0.1  # Adjust this threshold (0.0 to 1.0) to tune sensitivity

    for idx, person_mask in enumerate(valid_masks):

        # 1. Collect raw scores for ALL actions for this specific mask
        raw_scores = []

        # ... (keep your context radius calculation here) ...
        ctx_r = compute_ctx_radius_from_mask(
            person_mask, k=CTX_K, r_min=CTX_R_MIN, r_max=CTX_R_MAX
        )
        ctx_mask = dilate_person_mask(person_mask, radius_px=ctx_r)

        for i, action in enumerate(VOC_ACTIONS):
            diff_map = clean_heatmaps[action]
            # ... (keep mean_other calculation) ...
            mean_other = np.mean(np.delete(all_maps, i, axis=0), axis=0)

            # ... (keep s_person and s_context calculations) ...
            s_person = score_action(
                diff_map, person_mask, mean_other, lambda_c=0.7, beta_l=0.6
            )
            s_context = score_action(
                diff_map, ctx_mask, mean_other, lambda_c=0.7, beta_l=0.6
            )

            # Raw score combination
            raw_score = s_person + ALPHA_CTX * s_context
            raw_scores.append(raw_score)

        # 2. Normalize raw scores to Probabilities using Softmax
        # Convert list to tensor for easy softmax
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        probs = F.softmax(scores_tensor, dim=0).numpy()  # resulting values sum to 1.0

        # 3. Determine Best Action or Assign Unknown
        max_prob = np.max(probs)
        best_idx = np.argmax(probs)

        if max_prob < CONFIDENCE_THRESHOLD:
            final_action = "unknown"
            final_score = -1  # As done with SAM3
        else:
            final_action = VOC_ACTIONS[best_idx]
            final_score = max_prob

        # 4. Append to results (We now keep Unknowns instead of rejecting them)
        print(f"   Mask {idx}: {final_action} (Conf: {final_score:.1%})")
        final_results.append((person_mask, final_action, final_score))

    # E. VISUALIZZAZIONE FINALE (PRESERVED)
    print("Visualizing Final Result...")
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    final_results.sort(key=lambda x: x[2])
    used_legends = set()

    for mask, action, score in final_results:
        color = ACTION_COLORS.get(action, (0.5, 0.5, 0.5))
        used_legends.add(action)
        show_mask_custom(mask, ax, color)

        ys, xs = np.where(mask)
        if len(ys) > 0:
            ax.text(
                np.mean(xs),
                np.mean(ys),
                f"{action}\n{score:.0%}",  ## TODO: CAMBIARE SCORE IN PROBABILITA
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
    plt.title("SAM3 + B-Cos Action Detection")

    out_path = os.path.join(OUTPUT_DIR, "final_voc_result.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
