import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.patches as mpatches
import matplotlib as mpl  # Import necessario per i colori moderni

# --- 1. CONFIGURAZIONE ---
path_to_sam3 = (
    "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3"
)
if path_to_sam3 not in sys.path:
    sys.path.append(path_to_sam3)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Error importing SAM3: {e}")
    sys.exit(1)

CHECKPOINT_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006182.jpg"

OUTPUT_DIR = "sam3_global_match_fixed_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOC_ACTIONS = [
    "jumping",
    "phoning",
    "playing a instrument",
    "reading",
    "riding a bike",
    "riding a horse",
    "running",
    "taking a photo",
    "using a computer",
    "walking",
]

# --- FIX COLORI ---
# Usiamo l'API moderna per evitare warning e assicuriamoci che siano RGB
try:
    COLORS = mpl.colormaps["tab10"].colors  # Ritorna lista di tuple RGB
except AttributeError:
    # Fallback per vecchie versioni di matplotlib
    COLORS = plt.cm.tab10.colors

# --- 2. HELPER FUNCTIONS ---


def calculate_iou(mask1, mask2):
    """
    Calculates Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def show_mask(mask, ax, color, alpha=0.5):
    # --- FIX CRITICO QUI ---
    # Assicuriamoci che 'color' sia un array numpy
    color = np.array(color)
    # Se il colore ha 4 canali (RGBA), ne prendiamo solo i primi 3 (RGB)
    # altrimenti quando aggiungiamo alpha diventano 5 e crasha.
    if color.shape[0] > 3:
        color = color[:3]

    color = np.concatenate([color, [alpha]])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# --- 3. MAIN PIPELINE ---


def main():
    print(f"--- STARTING FIXED GLOBAL MATCH PIPELINE on {DEVICE} ---")

    # 1. Load Model
    print("Loading SAM3...")
    sam_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    sam_model.to(DEVICE).eval()
    processor = Sam3Processor(sam_model)
    print("✅ SAM3 Loaded.")

    # 2. Load Image
    raw_image = Image.open(IMG_PATH).convert("RGB")
    W, H = raw_image.size
    inference_state = processor.set_image(raw_image)

    # ---------------------------------------------------------
    # STAGE 1: FIND ALL PEOPLE (The "Anchors")
    # ---------------------------------------------------------
    print("\n[Stage 1] Detecting all people...")
    output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks_tensor = output["masks"]
    scores_tensor = output["scores"]

    people_masks = []

    # Filter valid person masks
    for i in range(len(scores_tensor)):
        if scores_tensor[i].item() > 0.4:
            m = masks_tensor[i].cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(
                    m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                )
            people_masks.append(m > 0)

    print(f"-> Found {len(people_masks)} potential people.")

    if len(people_masks) == 0:
        print("No people found. Exiting.")
        return

    # ---------------------------------------------------------
    # STAGE 2: GENERATE GLOBAL ACTION MASKS
    # ---------------------------------------------------------
    print("\n[Stage 2] Generating masks for all actions globally...")

    action_predictions = {}

    for action in VOC_ACTIONS:
        prompt_text = f"person {action}"

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

            # Solo maschere non vuote
            if np.sum(m > 0) > 0:
                preds.append((m > 0, s))

        action_predictions[action] = preds
        print(f"   -> '{prompt_text}': {len(preds)} candidates found")

    # ---------------------------------------------------------
    # STAGE 3: MATCHING & SCORING
    # ---------------------------------------------------------
    print("\n[Stage 3] Matching people to actions...")

    final_results = []

    for p_idx, p_mask in enumerate(people_masks):
        best_action = "unknown"
        best_conf = -1.0

        for action in VOC_ACTIONS:
            candidates = action_predictions[action]

            best_iou_for_action = 0.0
            score_of_best_iou = 0.0

            for act_mask, act_score in candidates:
                iou = calculate_iou(p_mask, act_mask)

                if iou > best_iou_for_action:
                    best_iou_for_action = iou
                    score_of_best_iou = act_score

            # Soglia IoU rilassata a 0.10 (10% sovrapposizione)
            if best_iou_for_action > 0.10:
                if score_of_best_iou > best_conf:
                    best_conf = score_of_best_iou
                    best_action = action

        print(f"Person {p_idx}: Winner -> {best_action} (Score: {best_conf:.4f})")

        final_results.append((p_mask, best_action, best_conf))

    # ---------------------------------------------------------
    # STAGE 4: VISUALIZATION
    # ---------------------------------------------------------
    print("\nVisualizing...")
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_image)
    ax = plt.gca()

    action_to_color = {
        act: COLORS[i % len(COLORS)] for i, act in enumerate(VOC_ACTIONS)
    }
    used_actions = set()

    # Draw results
    for p_mask, action, score in final_results:
        if action == "unknown":
            color = np.array([0.5, 0.5, 0.5])
        else:
            color = action_to_color[action]
            used_actions.add(action)

        show_mask(p_mask, ax, color)

        ys, xs = np.where(p_mask)
        if len(ys) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            label = f"{action}\n{score:.2f}"
            ax.text(
                cx,
                cy,
                label,
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="white",
                    boxstyle="round,pad=0.1",
                ),
            )

    # Legend
    patches = [mpatches.Patch(color=action_to_color[a], label=a) for a in used_actions]
    if patches:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.title("SAM3 Global Match (Fixed Viz)")

    out_path = os.path.join(OUTPUT_DIR, "sam3_match_fixed_viz.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"📸 Saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
