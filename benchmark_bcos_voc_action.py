import sys
import os
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "clip_es_official"))
sys.path.insert(0, os.path.join(current_dir, "bcosification"))
sys.path.insert(0, os.path.join(current_dir, "sam3"))

# Import from existing script (assuming it's in the same directory)
# We might need to refactor test_bcos_sam_actionsv8.py if it runs code on import
# For now, let's try to import specific functions if possible, or copy them if not.
# Given the structure of test_bcos_sam_actionsv8.py, it runs main() if __name__ == "__main__",
# but it has global variables that will execute.
# Let's redefine necessary constants and functions to avoid side effects and dependency on hardcoded paths in the old script.

from bcos_utils import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# --- CONSTANTS ---
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

# Copied from test_bcos_sam_actionsv8.py
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
            "strolling on street",
            "taking a step forward",
            "walking slowly",
            "walking",
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

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER FUNCTIONS (Adapted from test_bcos_sam_actionsv12.py) ---


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

            # Resize back to IMG_SIZE (512 in this script context)
            attr_resized = F.interpolate(
                attribution.unsqueeze(0).unsqueeze(0),
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            accumulated_maps.append(attr_resized)

    return torch.mean(torch.stack(accumulated_maps), dim=0)


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


# --- XML PARSING ---


def parse_voc_xml(xml_path):
    """
    Parses VOC XML to find people and their actions.
    Returns a list of dicts: {'bbox': [xmin, ymin, xmax, ymax], 'actions': set(actions)}
    """
    if not os.path.exists(xml_path):
        return []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    people = []

    for obj in root.findall("object"):
        if obj.find("name").text != "person":
            continue

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        actions = set()
        act_node = obj.find("actions")
        if act_node is not None:
            for act in VOC_ACTIONS:
                val = act_node.find(act)
                if val is not None and val.text == "1":
                    actions.add(act)

        people.append({"bbox": [xmin, ymin, xmax, ymax], "actions": actions})

    return people


# --- MATCHING ---


def compute_iou(boxA, boxB):
    # box: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def mask_to_bbox(mask):
    # mask is boolean or 0/1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]


# --- MAIN EVALUATION ---

# Context Parameters
CTX_K = 0.06
CTX_R_MIN = 10
CTX_R_MAX = 55
ALPHA_CTX = 0.7


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of images for testing"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to SAM3 checkpoint"
    )
    args = parser.parse_args()

    NUM_TEST_IMAGES = 50  # Default limit variable

    # Locate Checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        possible_paths = [
            "sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
            "sam3.pt",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("Error: SAM3 checkpoint not found. Please provide path via --checkpoint")
        return

    print(f"Using checkpoint: {checkpoint_path}")

    # Load Models
    print("Loading models...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam_model = (
            build_sam3_image_model(checkpoint_path=checkpoint_path).to(DEVICE).eval()
        )
        processor = Sam3Processor(sam_model)
    except Exception as e:
        print(f"Failed to load SAM3: {e}")
        return

    bcos_model, _ = load_bcos_model()
    bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    clip_model.to(DEVICE).eval()
    print("Models loaded.")

    # Prepare Data
    voc_root = "data/VOCdevkit/VOC2012/"
    val_file = os.path.join(voc_root, "ImageSets", "Action", "val.txt")
    with open(val_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    if args.limit:
        image_ids = image_ids[: args.limit]
    else:
        # User requested variable usage
        image_ids = image_ids[:NUM_TEST_IMAGES]

    # Metrics Storage
    y_true_all = {act: [] for act in VOC_ACTIONS}
    y_score_all = {act: [] for act in VOC_ACTIONS}
    confusion_data = []

    # Pre-compute embeddings for B-Cos (Weights)
    print("Pre-computing embeddings...")

    # 1. Global Background Weight
    bg_weight = get_global_background_embedding(clip_model, DEVICE)

    # 2. Action Weights
    action_embeddings = {}
    for action in VOC_ACTIONS:
        action_embeddings[action] = get_action_embedding(clip_model, action, DEVICE)

    # Loop
    print(f"Starting evaluation on {len(image_ids)} images...")

    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    # Define transforms for B-Cos Maps
    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )

    pbar = tqdm(image_ids)
    for img_id in pbar:
        img_path = os.path.join(voc_root, "JPEGImages", f"{img_id}.jpg")
        xml_path = os.path.join(voc_root, "Annotations", f"{img_id}.xml")

        gt_people = parse_voc_xml(xml_path)
        # Filter GT people who have at least one action
        gt_people = [p for p in gt_people if len(p["actions"]) > 0]

        if not gt_people:
            continue

        raw_image = Image.open(img_path).convert("RGB")
        W, H = raw_image.size

        # 1. SAM Prediction
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

        # Match GT to Masks using Greedy IoU
        gt_to_mask = {}
        used_masks = set()
        ious = np.zeros((len(gt_people), len(valid_masks)))
        for i, gt in enumerate(gt_people):
            for j, mask in enumerate(valid_masks):
                bbox_mask = mask_to_bbox(mask)
                if bbox_mask:
                    ious[i, j] = compute_iou(gt["bbox"], bbox_mask)

        if ious.size > 0:
            indices = np.dstack(
                np.unravel_index(np.argsort(ious.ravel())[::-1], ious.shape)
            )[0]
            for r, c in indices:
                if ious[r, c] < 0.1:
                    break
                if r not in gt_to_mask and c not in used_masks:
                    gt_to_mask[r] = c
                    used_masks.add(c)

        # 2. B-Cos Dynamic Heatmaps (Softmax Logic)
        clean_heatmaps = {}

        # Only compute if we have matches to safeguard compute
        if len(gt_to_mask) > 0:
            img_tensor = prep(raw_image)

            # A. Background Map
            bg_map_raw = compute_raw_bcos_map(
                bcos_model, img_tensor, bg_weight, blur, DEVICE
            )

            # B. Action Maps
            action_maps_raw = []
            for action in VOC_ACTIONS:
                w = action_embeddings[action]
                m = compute_raw_bcos_map(bcos_model, img_tensor, w, blur, DEVICE)
                action_maps_raw.append(m)

            # C. Stack & Normalize
            all_maps_stack = torch.stack([bg_map_raw] + action_maps_raw, dim=0)
            g_min = all_maps_stack.min()
            g_max = all_maps_stack.max()
            denom = g_max - g_min + 1e-8
            stack_norm = (all_maps_stack - g_min) / denom

            # D. Softmax
            probs = F.softmax(stack_norm * 20, dim=0)  # [11, 512, 512]

            # E. Distribute back
            # Index 0 is BG, so Action i is at i+1
            for i, action in enumerate(VOC_ACTIONS):
                prob_map = probs[i + 1].cpu().numpy()
                prob_map_resized = cv2.resize(
                    prob_map, (W, H), interpolation=cv2.INTER_LINEAR
                )
                clean_heatmaps[action] = prob_map_resized

            all_maps_np = np.stack([clean_heatmaps[a] for a in VOC_ACTIONS], axis=0)

            # Score Matched People
            for gt_idx, mask_idx in gt_to_mask.items():
                gt = gt_people[gt_idx]
                person_mask = valid_masks[mask_idx]

                # Context Mask
                ctx_r = compute_ctx_radius_from_mask(
                    person_mask, k=CTX_K, r_min=CTX_R_MIN, r_max=CTX_R_MAX
                )
                ctx_mask = dilate_person_mask(person_mask, radius_px=ctx_r)

                scores = {}
                for i, action in enumerate(VOC_ACTIONS):
                    diff_map = clean_heatmaps[action]  # This is now the probability map
                    mean_other = np.mean(np.delete(all_maps_np, i, axis=0), axis=0)

                    # Person Score
                    s_person = score_action(
                        diff_map, person_mask, mean_other, lambda_c=0.7, beta_l=0.6
                    )

                    # Context Score
                    s_context = score_action(
                        diff_map, ctx_mask, mean_other, lambda_c=0.7, beta_l=0.6
                    )

                    # Combined Score
                    scores[action] = s_person + ALPHA_CTX * s_context

                best_act = max(scores, key=scores.get)
                best_score = scores[best_act]

                # Store Metrics
                for action in VOC_ACTIONS:
                    y_true_all[action].append(1 if action in gt["actions"] else 0)
                    y_score_all[action].append(scores[action])

                confusion_data.append(
                    {
                        "gt_actions": gt["actions"],
                        "pred_action": best_act,
                        "pred_score": best_score,
                    }
                )

        # Handle unmatched GT people
        for i in range(len(gt_people)):
            if i not in gt_to_mask:
                gt = gt_people[i]
                for action in VOC_ACTIONS:
                    y_true_all[action].append(1 if action in gt["actions"] else 0)
                    y_score_all[action].append(-100.0)

                confusion_data.append(
                    {
                        "gt_actions": gt["actions"],
                        "pred_action": "none",
                        "pred_score": -100.0,
                    }
                )

        # --- Running mAP Update ---
        current_aps = []
        for action in VOC_ACTIONS:
            y_true = np.array(y_true_all[action])
            y_score = np.array(y_score_all[action])

            if len(y_true) == 0 or np.sum(y_true) == 0:
                ap = 0.0
            else:
                # Basic check to avoid errors if only one class present so far
                try:
                    ap = sklearn.metrics.average_precision_score(y_true, y_score)
                except:
                    ap = 0.0
            current_aps.append(ap)

        current_mAP = np.mean(current_aps)
        pbar.set_description(f"mAP: {current_mAP:.4f}")

    # --- CALCULATE METRICS ---
    print("\n--- RESULTS ---")
    aps = []
    print(f"{'Action':<20} {'AP':<10}")
    print("-" * 30)
    for action in VOC_ACTIONS:
        y_true = np.array(y_true_all[action])
        y_score = np.array(y_score_all[action])

        if len(y_true) == 0 or np.sum(y_true) == 0:
            ap = 0.0
        else:
            ap = sklearn.metrics.average_precision_score(y_true, y_score)

        aps.append(ap)
        print(f"{action:<20} {ap:.4f}")

    mAP = np.mean(aps)
    print("-" * 30)
    print(f"{'mAP':<20} {mAP:.4f}")

    # Confusion Matrix
    # We need to flatten the multi-label GT for visualization
    # Strategy: For each prediction, if it matches ONE of the GT labels, count it as correct for that label.
    # If it doesn't match any, count it as error (GT=primary GT?, Pred=Pred)
    # This is tricky for multi-label.
    # Simplified: Just plot predicted vs "Primary" GT (maybe the one with highest score if we had oracle, or just random)
    # Better: Row = GT Action, Col = Pred Action.
    # Since a sample has multiple GT actions, we add +1 to (GT_action, Pred_action) for ALL GT actions.
    # Then normalize by row (Total GT instances of that action).

    cm_labels = VOC_ACTIONS + ["none"]
    cm = np.zeros((len(VOC_ACTIONS), len(cm_labels)))

    for item in confusion_data:
        pred_idx = cm_labels.index(item["pred_action"])
        for gt_act in item["gt_actions"]:
            if gt_act in VOC_ACTIONS:
                gt_idx = VOC_ACTIONS.index(gt_act)
                cm[gt_idx, pred_idx] += 1

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="g", xticklabels=cm_labels, yticklabels=VOC_ACTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix (mAP: {mAP:.4f})")
    plt.savefig("confusion_matrix_bcos.png")
    print("Confusion matrix saved to confusion_matrix_bcos.png")


if __name__ == "__main__":
    main()
