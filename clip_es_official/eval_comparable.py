import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image
import cv2

try:
    import pydensecrf.densecrf as dcrf

    HAS_CRF = True
except ImportError:
    print("Warning: pydensecrf not found. Skipping CRF.")
    HAS_CRF = False

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


def apply_crf(image, prob_map):
    if not HAS_CRF:
        return (prob_map > 0.6).astype(np.uint8)

    h, w = image.shape[:2]

    # Create Hard Mask for Unary Potentials as per your script
    # Your script logic: prob = binary * 0.9 + 0.05
    hard_mask = (prob_map > 0.6).astype(np.uint8)

    prob = hard_mask.astype(np.float32) * 0.9 + 0.05
    U = np.stack([1.0 - prob, prob], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


def main(args):
    total_inter = np.zeros(20)
    total_union = np.zeros(20)

    npy_files = [f for f in os.listdir(args.npy_root) if f.endswith(".npy")]
    print(f"Evaluating {len(npy_files)} files...")

    for npy_file in tqdm(npy_files):
        # Load Data
        data = np.load(os.path.join(args.npy_root, npy_file), allow_pickle=True).item()
        keys = data["keys"]
        cams = data["attn_highres"]

        # Load Image (for CRF) and GT
        img_name = npy_file.replace(".npy", ".jpg")
        img_path = os.path.join(args.img_root, img_name)
        gt_path = os.path.join(args.gt_root, npy_file.replace(".npy", ".png"))

        if not os.path.exists(gt_path):
            continue

        # Read Image safely
        stream = np.fromfile(img_path, dtype=np.uint8)
        img_np = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img_np is None:
            continue
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # CRF expects RGB

        gt_mask = np.array(Image.open(gt_path))  # 0=BG, 1..20=Classes

        # Process each class
        for i, class_idx in enumerate(keys):
            prob_map = cams[i]

            # Apply your specific CRF / Hard Mask logic
            final_mask = apply_crf(img_rgb, prob_map)

            # GT Binary for this class
            # VOC indices are 1-based, keys are 0-based.
            gt_idx = class_idx + 1
            gt_binary = (gt_mask == gt_idx).astype(np.uint8)

            # Ignore 255 in GT
            valid = gt_mask != 255

            pred_flat = final_mask[valid]
            gt_flat = gt_binary[valid]

            inter = (pred_flat & gt_flat).sum()
            union = (pred_flat | gt_flat).sum()

            total_inter[class_idx] += inter
            total_union[class_idx] += union

    iou_per_class = total_inter / (total_union + 1e-7)
    miou = np.mean(iou_per_class)

    print(f"\nMean IoU (Apples-to-Apples): {miou*100:.2f}%")
    for i, val in enumerate(iou_per_class):
        print(f"{VOC_CLASSES[i]:<15}: {val*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--npy_root", type=str, required=True)
    parser.add_argument("--gt_root", type=str, required=True)
    args = parser.parse_args()
    main(args)
