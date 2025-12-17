import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import time

# --- CONFIGURAZIONE ---
USE_CRF = True  # True per mIoU più alta, False per max velocità
NUM_WORKERS = 0  # SU WINDOWS: 0 è obbligatorio per stabilità con CUDA
IMG_SIZE = 512
BATCH_SIZE = 1

# --- IMPORT MODELLI ---
sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

from bcos_localization import (
    load_bcos_model,
    load_clip_for_text,
    tokenize_text_prompt,
    compute_attributions,
)
import bcos.data.transforms as custom_transforms

# Check DenseCRF
try:
    import pydensecrf.densecrf as dcrf

    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("⚠️ WARNING: DenseCRF non trovato. mIoU sarà più bassa.")

# Classi VOC
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

# --- DEVICE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# --- FUNZIONI CORE ---


def precompute_background_weights(clip_model, device):
    """Calcola i vettori dello sfondo una volta sola"""
    print("Pre-calcolo embeddings sfondo...")
    # Background set di CLIP-ES
    bg_classes = [
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
        "road",
        "rock",
        "street",
        "cloud",
        "mountain",
        "floor",
        "ceiling",
        "background",
        "blur",
    ]
    bg_prompts = [f"a photo of {bg}" for bg in bg_classes]

    weights = []
    with torch.no_grad():
        for p in bg_prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def get_target_weight(clip_model, class_name, device):
    """Calcola il vettore del target con prompt SOTA"""
    # Prompt engineering da CLIP-ES ("a clean origami" funziona meglio)
    prompts = [
        f"a clean origami {class_name}.",
        f"a photo of a {class_name}.",
        f"the {class_name}.",
    ]
    weights = []
    with torch.no_grad():
        for p in prompts:
            w = tokenize_text_prompt(clip_model, p).to(device)
            weights.append(w)
    return torch.mean(torch.stack(weights), dim=0)


def process_single_image_softmax(model, image_tensor, target_weight, bg_weight):
    """
    Versione POTENZIATA: Multi-Scale + Flip Inference.
    Sfrutta la VRAM libera per massimizzare la mIoU.
    """
    # Scale da testare (adatta in base alla tua VRAM)
    scales = [448, 560, 672]

    # Lista per accumulare le mappe di probabilità
    accumulated_maps = []

    # Tensore originale (su CPU per i resize) - Assumiamo shape [1, 6, H, W]
    original_img_cpu = image_tensor.squeeze(0)  # [6, H, W]

    # Usiamo una dimensione fissa per accumulare (es. la dimensione base 512)
    # Alla fine faremo il resize finale alla dimensione vera dell'immagine nel main
    base_h, base_w = IMG_SIZE, IMG_SIZE

    # Loop sulle scale
    for s in scales:
        # Resize dinamico dell'input
        resize_transform = transforms.Resize(
            (s, s), interpolation=transforms.InterpolationMode.BICUBIC
        )
        # Prepare 4D Tensor per Flip e Resize [1, 6, S, S]
        img_scaled_4d = resize_transform(original_img_cpu).unsqueeze(0).to(DEVICE)

        # Prepare 3D Tensor per Model (Il modello b-cos aggiunge batch internamente se serve)
        # Passiamo 3D [6, S, S]
        model_input = img_scaled_4d.squeeze(0)

        with torch.no_grad():
            # --- 1. Passaggio Normale ---
            _, map_t, _, _ = compute_attributions(model, model_input, target_weight)
            _, map_b, _, _ = compute_attributions(model, model_input, bg_weight)

            # Gestione output numpy/torch
            if isinstance(map_t, np.ndarray):
                map_t = torch.from_numpy(map_t).to(DEVICE)
            if isinstance(map_b, np.ndarray):
                map_b = torch.from_numpy(map_b).to(DEVICE)

            # Softmax
            stack = torch.stack([map_b, map_t], dim=0)
            probs = F.softmax(stack * 10, dim=0)

            # Interpolate vuole [B, C, H, W] -> aggiungiamo dimensioni dummy
            target_prob = probs[1, :, :].unsqueeze(0).unsqueeze(0)

            prob_resized = F.interpolate(
                target_prob, size=(base_h, base_w), mode="bilinear", align_corners=False
            ).squeeze()
            accumulated_maps.append(prob_resized)

            # --- 2. Passaggio FLIPPATO (Horizontal Flip) ---
            # Flip sul tensore 4D [1, 6, S, S] -> asse 3 è la larghezza
            img_flipped_4d = torch.flip(img_scaled_4d, [3])

            # Input al modello 3D
            model_input_flip = img_flipped_4d.squeeze(0)

            _, map_t_f, _, _ = compute_attributions(
                model, model_input_flip, target_weight
            )
            _, map_b_f, _, _ = compute_attributions(model, model_input_flip, bg_weight)

            if isinstance(map_t_f, np.ndarray):
                map_t_f = torch.from_numpy(map_t_f).to(DEVICE)
            if isinstance(map_b_f, np.ndarray):
                map_b_f = torch.from_numpy(map_b_f).to(DEVICE)

            stack_f = torch.stack([map_b_f, map_t_f], dim=0)
            probs_f = F.softmax(stack_f * 10, dim=0)
            target_prob_f = probs_f[1, :, :]

            # Un-Flip e Accumulo
            target_prob_unflipped = torch.flip(target_prob_f, [1])
            prob_resized_f = F.interpolate(
                target_prob_unflipped.unsqueeze(0).unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            accumulated_maps.append(prob_resized_f)

    # Media di tutte le mappe (3 scale * 2 flip = 6 mappe)
    final_prob_map = torch.mean(torch.stack(accumulated_maps), dim=0)

    return final_prob_map.cpu().numpy()


# --- POST-PROCESS ---


def get_hard_mask(prob_map, h, w):
    """
    Converte la prob map in maschera binaria.
    Effettua il resize finale alla dimensione target (h, w) se necessario.
    """
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
    binary = (prob_map > 0.5).astype(np.uint8)
    return binary


def apply_crf_on_hard_mask(image_np, hard_mask):
    if not HAS_CRF or not USE_CRF:
        return hard_mask

    h, w = image_np.shape[:2]
    # Assicuriamo che la maschera sia della dimensione dell'immagine
    if hard_mask.shape != (h, w):
        hard_mask = cv2.resize(hard_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    prob = hard_mask.astype(np.float32) * 0.9 + 0.05
    U = np.stack([1.0 - prob, prob], axis=0)
    U = -np.log(U).reshape((2, -1)).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(13, 13, 13), rgbim=image_np, compat=10)

    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))


def calculate_iou(pred, gt):
    inter = np.sum((pred == 1) & (gt == 1))
    union = np.sum((pred == 1) | (gt == 1))
    if union == 0:
        return 1.0
    return inter / union


# --- DATA LOADING ---


class PascalVocValidation(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target_np = np.array(target)
        unique_classes = np.unique(target_np)
        valid_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]

        if len(valid_classes) > 0:
            label_idx = valid_classes[0]
            label_name = VOC_CLASSES[label_idx]
        else:
            label_idx = -1
            label_name = "none"
        return img, target, label_idx, label_name


def preprocess_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )


def custom_collate(batch):
    return batch[0]


# --- MAIN ---
def run_benchmark():
    print(f"🚀 Avvio Benchmark su {DEVICE.upper()} (Multi-Scale + Flip)")

    bcos_model, _ = load_bcos_model()
    bcos_model = bcos_model.to(DEVICE).eval()
    clip_model, _ = load_clip_for_text()
    if hasattr(clip_model, "to"):
        clip_model = clip_model.to(DEVICE).eval()

    bg_weight = precompute_background_weights(clip_model, DEVICE)
    class_weights_cache = {}

    try:
        val_dataset = PascalVocValidation(
            root="./data", year="2012", image_set="val", download=False
        )
        loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate,
        )
    except Exception as e:
        print(f"Errore dataset: {e}")
        return

    ious = []
    max_images = 100
    prep = preprocess_transform()

    print(f"Test su {max_images} immagini...")

    for i, data in enumerate(tqdm(loader, total=max_images)):
        if i >= max_images:
            break

        img_pil, target_pil, label_idx, label_name = data
        if label_idx == -1:
            continue

        # --- PREPARAZIONE DATI ---
        # Otteniamo tensore 3D [6, H, W]
        img_tensor = prep(img_pil)

        # Cache Text Embeddings
        if label_name not in class_weights_cache:
            class_weights_cache[label_name] = get_target_weight(
                clip_model, label_name, DEVICE
            )
        target_weight = class_weights_cache[label_name]

        # Pipeline
        try:
            # 1. Softmax Map (Multi-scale)
            # Qui passiamo img_tensor 3D, la funzione gestirà unsqueeze/squeeze
            prob_map_full = process_single_image_softmax(
                bcos_model, img_tensor, target_weight, bg_weight
            )

            # 2. Hard Mask (Con dimensioni corrette dell'immagine originale!)
            hard_mask = get_hard_mask(prob_map_full, img_pil.height, img_pil.width)

            # 3. CRF Refinement
            final_mask = apply_crf_on_hard_mask(np.array(img_pil), hard_mask)

            # 4. IoU
            gt_mask = np.array(target_pil)
            gt_binary = (gt_mask == label_idx).astype(np.uint8)

            iou = calculate_iou(final_mask, gt_binary)
            ious.append(iou)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| OOM Error. Salto immagine.")
                torch.cuda.empty_cache()
            else:
                print(f"| Errore Img {i}: {e}")
            continue

    if len(ious) > 0:
        mean_iou = np.mean(ious) * 100
        print(f"\n==========================================")
        print(f"🏆 mIoU FINALE (Multi-Scale+Flip+CRF): {mean_iou:.2f}%")
        print(f"==========================================")
    else:
        print("Nessun risultato valido.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run_benchmark()
