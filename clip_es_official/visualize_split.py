import numpy as np
import cv2
import os
import argparse
import random
from tqdm import tqdm

# Standard VOC Class mapping
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


def cv2_imread_safe(file_path):
    """
    Drop-in replacement for cv2.imread that handles Windows Unicode paths (e.g. 'à').
    """
    try:
        # Read file as a byte stream using numpy (supports unicode)
        stream = np.fromfile(file_path, dtype=np.uint8)
        # Decode the byte stream into an image
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def apply_heatmap(img, cam):
    # Normalize
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-7)
    cam = np.uint8(255 * cam)
    # Colormap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    # Overlay
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Get all NPY files
    all_files = [f for f in os.listdir(args.npy_root) if f.endswith(".npy")]

    # Select 10 random files
    if len(all_files) > 10:
        selected_files = random.sample(all_files, 10)
    else:
        selected_files = all_files

    print(f"Visualizing {len(selected_files)} images...")

    for npy_file in tqdm(selected_files):
        # Load Data
        data = np.load(os.path.join(args.npy_root, npy_file), allow_pickle=True).item()
        keys = data["keys"]
        cams = data["attn_highres"]

        # Load Image
        img_name = npy_file.replace(".npy", ".jpg")
        img_path = os.path.join(args.img_root, img_name)

        # --- FIXED READ METHOD ---
        img = cv2_imread_safe(img_path)

        if img is None:
            print(f"Could not load {img_name} - check path or permissions")
            continue

        for i, class_idx in enumerate(keys):
            class_name = VOC_CLASSES[int(class_idx)]
            cam = cams[i]

            vis = apply_heatmap(img, cam)

            # Save: ID_class.jpg
            save_name = f"{os.path.splitext(img_name)[0]}_{class_name}.jpg"
            # Use imencode for saving to unicode paths too
            success, buffer = cv2.imencode(".jpg", vis)
            if success:
                save_path = os.path.join(args.out_dir, save_name)
                buffer.tofile(save_path)

    print(f"Done! Check results in {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_root", type=str, required=True, help="Path to VOC JPEGImages"
    )
    parser.add_argument(
        "--npy_root", type=str, required=True, help="Path to .npy outputs"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./vis_subset", help="Output folder"
    )
    args = parser.parse_args()
    main(args)
