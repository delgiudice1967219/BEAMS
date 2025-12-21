import sys
import os
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "sam3"))

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

VOC_ACTION_PROMPTS = {
    "jumping": "jumping",
    "phoning": "phoning",
    "playinginstrument": "playing an instrument",
    "reading": "reading",
    "ridingbike": "riding a bike",
    "ridinghorse": "riding a horse",
    "running": "running",
    "takingphoto": "taking a photo",
    "usingcomputer": "using a computer",
    "walking": "walking",
}

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- HELPER FUNCTIONS ---
def calculate_iou(mask1, mask2):
    """
    Calculates Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


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
def compute_bbox_iou(boxA, boxB):
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

    NUM_TEST_IMAGES = 0  # Default limit variable

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
        if NUM_TEST_IMAGES == 0:
            NUM_TEST_IMAGES = len(image_ids)
        image_ids = image_ids[:NUM_TEST_IMAGES]

    # Metrics Storage
    y_true_all = {act: [] for act in VOC_ACTIONS}
    y_score_all = {act: [] for act in VOC_ACTIONS}
    confusion_data = []

    # Loop
    print(f"Starting evaluation on {len(image_ids)} images...")

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
        inference_state = processor.set_image(raw_image)

        # ---------------------------------------------------------
        # STAGE 1: FIND ALL PEOPLE (The "Anchors")
        # ---------------------------------------------------------
        output = processor.set_text_prompt(state=inference_state, prompt="person")
        masks_tensor = output["masks"]
        scores_tensor = output["scores"]

        people_masks = []
        for i in range(len(scores_tensor)):
            if scores_tensor[i].item() > 0.4:
                m = masks_tensor[i].cpu().numpy().squeeze()
                if m.shape != (H, W):
                    m = cv2.resize(
                        m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                    )
                people_masks.append(m > 0)

        # ---------------------------------------------------------
        # Match GT to Anchor Masks using Greedy IoU
        # ---------------------------------------------------------
        gt_to_mask = {}
        used_masks = set()
        ious = np.zeros((len(gt_people), len(people_masks)))
        for i, gt in enumerate(gt_people):
            for j, mask in enumerate(people_masks):
                bbox_mask = mask_to_bbox(mask)
                if bbox_mask:
                    ious[i, j] = compute_bbox_iou(gt["bbox"], bbox_mask)

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

        # ---------------------------------------------------------
        # STAGE 2: GENERATE GLOBAL ACTION MASKS (Only if we have matches)
        # ---------------------------------------------------------
        action_predictions = {}
        if len(gt_to_mask) > 0:
            for action in VOC_ACTIONS:
                prompt_text = f"person {VOC_ACTION_PROMPTS[action]}"
                output_act = processor.set_text_prompt(
                    state=inference_state, prompt=prompt_text
                )
                act_masks = output_act["masks"]
                act_scores = output_act["scores"]

                preds = []
                for i in range(len(act_scores)):
                    m = act_masks[i].cpu().numpy().squeeze()
                    s = act_scores[i].item()

                    if m.shape != (H, W):
                        m = cv2.resize(
                            m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                        )

                    if np.sum(m > 0) > 0:
                        preds.append((m > 0, s))
                action_predictions[action] = preds

        # ---------------------------------------------------------
        # STAGE 3: CLASSIFY MATCHED PEOPLE
        # ---------------------------------------------------------

        # For each matched GT person, finding the best action based on IoU overlap
        for gt_idx, mask_idx in gt_to_mask.items():
            gt = gt_people[gt_idx]
            p_mask = people_masks[mask_idx]

            # Store best score for EACH action to calculate mAP properly
            # In SAM logic we don't have a direct score for every class for every person.
            # We have candidate masks.
            # Logic from classification_sam.py: find best candidate for each action and take its score?

            scores = {}
            for action in VOC_ACTIONS:
                candidates = action_predictions.get(action, [])
                best_conf = 0.0  # Default low score if no overlap

                # Check overlap with candidates
                for act_mask, act_score in candidates:
                    iou = calculate_iou(p_mask, act_mask)
                    # If valid overlap, consider this candidate's score
                    if iou > 0.10:
                        if act_score > best_conf:
                            best_conf = act_score

                scores[action] = best_conf

            # Determine winner for confusion matrix
            best_act = max(scores, key=scores.get)
            best_score = scores[best_act]

            # If all scores are 0 (no overlap with any action mask), it's basically "unknown"
            if best_score == 0.0:
                best_act = "none"  # For confusion matrix

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
                    y_score_all[action].append(0.0)  # Zero score

                confusion_data.append(
                    {
                        "gt_actions": gt["actions"],
                        "pred_action": "none",
                        "pred_score": 0.0,
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
    cm_labels = VOC_ACTIONS + ["none"]
    cm = np.zeros((len(VOC_ACTIONS), len(cm_labels)))

    for item in confusion_data:
        pred_act = item["pred_action"]
        if pred_act not in cm_labels:
            pred_act = "none"  # Fallback
        pred_idx = cm_labels.index(pred_act)

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
    plt.savefig("confusion_matrix_sam.png")
    print("Confusion matrix saved to confusion_matrix_sam.png")


if __name__ == "__main__":
    main()
