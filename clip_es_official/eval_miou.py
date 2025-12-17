import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm
import cv2  # You might need this if cam resizing is required

# PASCAL VOC has 20 foreground classes
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def main(args):
    # We track 21 classes internally (0=bg, 1-20=objects) to handle the argmax correctly,
    # but we will only report metrics for 1-20.
    num_internal_classes = 21
    total_inter = np.zeros(num_internal_classes)
    total_union = np.zeros(num_internal_classes)

    npy_files = [f for f in os.listdir(args.npy_root) if f.endswith(".npy")]
    print(f"Evaluating {len(npy_files)} images...")

    for npy_file in tqdm(npy_files):
        # 1. Load CAM Data
        try:
            data = np.load(
                os.path.join(args.npy_root, npy_file), allow_pickle=True
            ).item()
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue

        keys = data["keys"]  # indices 0-19 from your CAM generation
        cams = data["attn_highres"]  # (N_classes, H, W)

        # 2. Load Ground Truth Mask
        # Note: Check if your masks are .png (standard VOC) or .jpg
        mask_name = npy_file.replace(".npy", ".png")
        mask_path = os.path.join(args.gt_root, mask_name)

        if not os.path.exists(mask_path):
            # Fallback if masks have different extension
            mask_path = mask_path.replace(".png", ".jpg")
            if not os.path.exists(mask_path):
                # print(f"Mask not found: {mask_name}")
                continue

        # Open using PIL (handles paths better than cv2 on Windows)
        gt_mask = np.array(Image.open(mask_path))
        h, w = gt_mask.shape

        # 3. Generate Prediction Mask
        # Initialize full score map: Channel 0 = Background, Channels 1-20 = Objects
        full_map = np.zeros((21, h, w), dtype=np.float32)

        # Set background threshold
        full_map[0, :, :] = args.threshold

        # Fill in the object CAMs
        for i, k in enumerate(keys):
            # Map CAM index (0-19) to Mask index (1-20)
            class_id = int(k) + 1

            cam = cams[i]

            # Resize if necessary
            if cam.shape != (h, w):
                cam = cv2.resize(cam, (w, h))

            # Normalize CAM to 0-1
            if args.norm:
                cam_min = cam.min()
                cam_max = cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = np.zeros_like(cam)

            full_map[class_id, :, :] = cam

        # Argmax: 0 if background is highest, 1-20 if an object is highest
        pred_mask = np.argmax(full_map, axis=0)

        # 4. Accumulate Stats
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        # Ignore VOC border regions (255)
        valid_mask = gt_flat != 255
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]

        # Calculate IoU for all 21 classes (including bg) internally
        for c in range(num_internal_classes):
            pred_inds = pred_flat == c
            gt_inds = gt_flat == c

            intersection = (pred_inds & gt_inds).sum()
            union = (pred_inds | gt_inds).sum()

            total_inter[c] += intersection
            total_union[c] += union

    # --- FINAL CALCULATION (EXCLUDING BACKGROUND) ---

    # Calculate IoU for all
    iou_per_class = total_inter / (total_union + 1e-7)

    # Slice [1:] to get only indices 1 to 20 (The Objects)
    object_iou = iou_per_class[1:]

    # Mean IoU of only the 20 object classes
    miou = np.mean(object_iou)

    print("\n" + "=" * 40)
    print(f"Mean IoU (20 classes, no BG): {miou * 100:.2f}%")
    print("=" * 40)
    print("Class-wise IoU:")

    for i, score in enumerate(object_iou):
        print(f"{VOC_CLASSES[i]:<15}: {score * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy_root", type=str, required=True, help="Path to .npy outputs"
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        required=True,
        help="Path to VOC SegmentationClass folder",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Background threshold (0.0 - 1.0)"
    )
    parser.add_argument("--norm", action="store_true", help="Normalize CAMs")
    args = parser.parse_args()

    # Default norm to True
    args.norm = True
    main(args)
