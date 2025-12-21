import os
import cv2
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
import numpy as np

# Fix for NumPy 2.0 compatibility issue
if not hasattr(np, "float_"):
    np.float_ = np.float64

# --- CONFIGURAZIONE DA PAPER CLIP-ES ---
# Threshold di confidenza (mu) = 0.95 [cite: 531]
CONF_THRESHOLD = 0.95
# Augmentation scales
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
# Crop Size
CROP_SIZE = 336
# Batch Size [cite: 527]
BATCH_SIZE = 10
# Learning Rate & Poly Decay [cite: 530]
LR_INIT = 0.007  # Adattato per SGD
EPOCHS = 20  # CLIP-ES usa iterazioni (20k), 20 epoche su VOC sono simili
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HybridDataset(Dataset):
    def __init__(self, img_dir, mask_dir, conf_dir, transform=None):
        self.img_dir = img_dir  # Cartella originale JPEGImages
        self.mask_dir = mask_dir  # Cartella generata SegmentationClass
        self.conf_dir = conf_dir  # Cartella generata Confidence
        self.transform = transform

        # Prendiamo solo i file che hanno una maschera generata
        self.images = [
            x.replace(".png", "") for x in os.listdir(mask_dir) if x.endswith(".png")
        ]
        print(f"Found {len(self.images)} samples for training.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # 1. Carica Immagine (RGB)
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Carica Maschera (Grayscale, Class IDs)
        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 3. Carica Confidenza (Float)
        conf_path = os.path.join(self.conf_dir, img_name + ".npy")
        confidence = np.load(conf_path)  # [H, W] float 0.0-1.0

        # --- IMPLEMENTAZIONE CGL (Confidence Guided Loss) ---
        # "Ignore pixels with low confidence"
        # Impostiamo a 255 i pixel dove la confidenza è < 0.95
        # PyTorch CrossEntropyLoss ignorerà automaticamente il valore 255.

        # Resize confidenza se necessario (a volte l'arrotondamento cambia di 1px)
        if confidence.shape != mask.shape:
            confidence = cv2.resize(
                confidence,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask[confidence < CONF_THRESHOLD] = 255

        # 4. Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


def get_training_augmentation():
    """Strong Augmentation as described in CLIP-ES"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.RandomScale(scale_limit=0.5, p=0.5),
            # Pad to exactly CROP_SIZE if image is smaller
            # border_mode=0 is cv2.BORDER_CONSTANT
            # value=0 (black image padding), mask_value=255 (ignore mask padding in Loss)
            A.PadIfNeeded(
                min_height=CROP_SIZE,
                min_width=CROP_SIZE,
                border_mode=0,
                value=0,
                mask_value=255,
            ),
            A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def run_training():
    parser = argparse.ArgumentParser()
    # Punti alla cartella VOC originale per le immagini
    parser.add_argument(
        "--voc_images", type=str, required=True, help="Path to VOC2012/JPEGImages"
    )
    # Punti alla cartella GENERATA dallo script precedente per maschere e confidenza
    parser.add_argument(
        "--pseudo_root",
        type=str,
        required=True,
        help="Path to voc_bcos_pseudolabels folder",
    )
    parser.add_argument("--out_model", type=str, default="deeplab_bcos_trained.pth")
    args = parser.parse_args()

    # Paths
    mask_dir = os.path.join(args.pseudo_root, "SegmentationClass")
    conf_dir = os.path.join(args.pseudo_root, "Confidence")

    # Dataset & Loader
    train_dataset = HybridDataset(
        img_dir=args.voc_images,
        mask_dir=mask_dir,
        conf_dir=conf_dir,
        transform=get_training_augmentation(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Metti 0 se sei su Windows e crasha
        pin_memory=True,
    )

    # Model Setup: DeepLabV3+ with ResNet101 (Better than V2 used in paper)
    print("Creating DeepLabV3+ model...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=21,  # 20 classi + Background
    )
    model.to(DEVICE)

    # Loss & Optimizer
    # ignore_index=255 è la chiave per la CGL Loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.SGD(
        model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=1e-4
    )

    # Polynomial LR Scheduler [cite: 530]
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=EPOCHS, power=0.9
    )

    print("Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model(images)

            # Loss Calculation (ignora automaticamente i pixel < 0.95 conf)
            loss = criterion(logits, masks)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()
        print(
            f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss / len(train_loader):.4f}"
        )

        # Save Checkpoint
        torch.save(model.state_dict(), args.out_model)
        print(f"Model saved to {args.out_model}")


if __name__ == "__main__":
    run_training()
