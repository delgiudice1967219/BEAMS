import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.datasets as datasets
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_CLASSES = 21
VOC_CLASSES = [
    "background",
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


def apply_dense_crf(image, probs):
    """
    Refines the DeepLab probabilities using DenseCRF.
    image: (H, W, 3) numpy array, uint8
    probs: (21, H, W) numpy array, float (Softmax output)
    """
    c, h, w = probs.shape

    # Setup CRF
    d = dcrf.DenseCRF2D(w, h, c)

    # Unary potential (The DeepLab prediction)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Pairwise potentials (Color & Position)
    # sxy=Position std, srgb=Color std, compat=Weight
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image, compat=10)

    # Inference
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


class VOCValDataset(datasets.VOCSegmentation):
    def __init__(self, root, year="2012", image_set="val", transform=None):
        super().__init__(root, year=year, image_set=image_set, download=False)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image_np = np.array(image)
        target = np.array(target)

        # We need the original image for CRF, so we return it alongside the tensor
        if self.transform:
            h, w = image_np.shape[:2]
            # Padding for DeepLab (divisible by 16)
            new_h = (h + 15) // 16 * 16
            new_w = (w + 15) // 16 * 16

            aug = A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=new_h,
                        min_width=new_w,
                        border_mode=0,
                        value=0,
                        mask_value=255,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            transformed = aug(image=image_np, mask=target)
            image_tensor = transformed["image"]
            target_tensor = transformed["mask"]

        return image_tensor, target_tensor.long(), image_np


def compute_confusion_matrix(y_pred, y_true, num_classes=21):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    keep = y_true != 255
    y_pred = y_pred[keep]
    y_true = y_true[keep]
    if len(y_true) == 0:
        return np.zeros((num_classes, num_classes))
    return np.bincount(
        num_classes * y_true + y_pred, minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    val_dataset = VOCValDataset(root=args.voc_root, image_set="val", transform=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES))

    print(f"Evaluating with DenseCRF on {len(val_dataset)} images...")

    with torch.no_grad():
        for img_tensor, mask_tensor, original_img_np in tqdm(val_loader):
            img_tensor = img_tensor.to(DEVICE)

            # 1. DeepLab Inference
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [21, H, W]

            # 2. Un-pad probabilities to original size
            h_orig, w_orig = original_img_np.shape[1], original_img_np.shape[2]
            probs = probs[:, :h_orig, :w_orig]

            # 3. Apply DenseCRF
            # original_img_np comes from loader as [1, H, W, 3], grab [0]
            orig_img = original_img_np[0].numpy().astype(np.uint8)
            pred_mask = apply_dense_crf(orig_img, probs)

            # 4. Metric
            gt_mask = mask_tensor.cpu().numpy()[0]
            # Un-pad GT mask too
            gt_mask = gt_mask[:h_orig, :w_orig]

            total_cm += compute_confusion_matrix(pred_mask, gt_mask, NUM_CLASSES)

    # IoU Calculation
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    denominator = tp + fp + fn
    iou_per_class = np.divide(
        tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )

    print("\n" + "=" * 40)
    print("       EVALUATION RESULTS (mIoU) + CRF")
    print("=" * 40)
    for i, iou in enumerate(iou_per_class):
        print(f"{VOC_CLASSES[i]:<15}: {iou*100:.2f}%")
    print("-" * 40)
    print(f"Mean IoU (mIoU): {np.mean(iou_per_class)*100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()
