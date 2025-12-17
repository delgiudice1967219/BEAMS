import torch
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# --- SETUP ---
# Adjust this path to match your installation
path_to_sam3 = (
    "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3"
)
if path_to_sam3 not in sys.path:
    sys.path.append(path_to_sam3)

checkpoint_path = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2010_006635.jpg"
PROMPTS = ["person running", "person riding a bike"]

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error importing SAM3. Check paths.")
    sys.exit()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM 3...")
model = build_sam3_image_model(checkpoint_path=checkpoint_path)
model.to(DEVICE)
model.eval()
processor = Sam3Processor(model)

# --- INFERENCE ---
image_pil = Image.open(IMG_PATH).convert("RGB")
image_np = np.array(image_pil)
original_h, original_w = image_np.shape[:2]

inference_state = processor.set_image(image_np)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_np)
axes[0].set_title("Original")

for i, prompt in enumerate(PROMPTS):
    print(f"Processing SAM 3 prompt: '{prompt}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if len(masks) > 0:
        best_idx = torch.argmax(scores)
        best_mask = masks[best_idx].cpu().numpy()

        # Squeeze to remove batch dims -> [H_mask, W_mask]
        best_mask = best_mask.squeeze()

        # --- FIX: Resize mask to match original image exactly ---
        if best_mask.shape != (original_h, original_w):
            # cv2.resize takes (W, H)
            best_mask = cv2.resize(
                best_mask.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Ensure boolean
        best_mask = best_mask > 0

        # Create Blue mask overlay
        color_mask = np.zeros_like(image_np)
        color_mask[best_mask] = [30, 144, 255]  # Blue

        overlay = cv2.addWeighted(image_np, 0.7, color_mask, 0.3, 0)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"SAM 3: {prompt}")
    else:
        axes[i + 1].imshow(image_np)
        axes[i + 1].set_title(f"SAM 3: {prompt} (No Det)")

out_file = "result_sam3.jpg"
plt.savefig(out_file)
print(f"Saved {out_file}")
