import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.patches as mpatches

# --- 1. CONFIGURAZIONE PATH E IMPORT ---
path_to_sam3 = (
    "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3"
)
if path_to_sam3 not in sys.path:
    sys.path.append(path_to_sam3)

sys.path.insert(0, "clip_es_official")
sys.path.insert(0, "bcosification")

# PERCORSI
CHECKPOINT_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/sam3_model/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_005751.jpg"
OUTPUT_DIR = "sam_action_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

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

# Colori fissi per le azioni finali
ACTION_COLORS = {
    "jumping": (1.0, 0.0, 0.0),
    "phoning": (0.0, 1.0, 0.0),
    "playinginstrument": (0.0, 0.0, 1.0),
    "reading": (1.0, 1.0, 0.0),
    "ridingbike": (0.0, 1.0, 1.0),
    "ridinghorse": (1.0, 0.0, 1.0),
    "running": (1.0, 0.5, 0.0),
    "takingphoto": (0.5, 0.0, 0.5),
    "usingcomputer": (0.0, 1.0, 0.5),
    "walking": (1.0, 0.75, 0.8),
}


# --- 2. LA TUA FUNZIONE DI VISUALIZZAZIONE ---
def show_mask_custom(mask, ax, color, alpha=0.55):
    """
    Usa ESATTAMENTE la logica richiesta:
    squeeze per togliere dimensioni e reshape per il broadcasting del colore.
    """
    # Rimuovi dimensioni extra (da [1, H, W] diventa [H, W])
    mask = np.squeeze(mask)

    # Assicurati che sia 0 o 1 float
    mask = mask.astype(np.float32)

    # Aggiungi alpha al colore RGB -> (R, G, B, Alpha)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)

    h, w = mask.shape

    # Creiamo l'immagine colorata (Broadcasting)
    # (H, W, 1) * (1, 1, 4) -> (H, W, 4)
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)

    ax.imshow(mask_image)


# --- 3. PIPELINE COMPLETA ---
def main():
    print(f"--- STARTING PIPELINE ON {DEVICE} ---")

    # =========================================================================
    # FASE 1: SAM3 (SEGMENTAZIONE)
    # =========================================================================
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"Errore import SAM3: {e}")
        return

    print("Caricamento SAM3...")
    sam_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    sam_model.to(DEVICE).eval()
    processor = Sam3Processor(sam_model)
    print("✅ SAM3 Caricato.")

    # Carica immagine
    raw_image = Image.open(IMG_PATH).convert("RGB")
    original_w, original_h = raw_image.size

    # Inferenza SAM
    print("Running SAM3 inference...")
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks_tensor = output["masks"]  # Tensori [N, 1, H, W]
    scores_tensor = output["scores"]

    # --- VISUALIZZAZIONE INTERMEDIA (STEP 1) ---
    print("\n--- [STEP 1] VISUALIZZAZIONE OUTPUT SAM3 GREZZO ---")
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    ax = plt.gca()

    valid_masks_for_bcos = []  # Lista di maschere booleane per lo step 2

    count = 0
    SOGLIA = 0.05  # Soglia per mostrare le maschere (tenuta bassa come richiesto prima)

    for i in range(len(scores_tensor)):
        score = scores_tensor[i].item()

        if score > SOGLIA:
            # Estrai maschera e converti a numpy
            m_numpy = masks_tensor[i].cpu().numpy()  # [1, H, W]
            m_numpy = np.squeeze(m_numpy)  # [H, W]

            # Resize se necessario
            if m_numpy.shape != (original_h, original_w):
                m_numpy = cv2.resize(
                    m_numpy.astype(np.uint8),
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Colore random per visualizzazione step 1
            rand_color = np.random.random(3)
            show_mask_custom(m_numpy, ax, rand_color)

            # Salva per dopo (booleana)
            valid_masks_for_bcos.append(m_numpy > 0)

            # Testo score
            ys, xs = np.where(m_numpy > 0)
            if len(ys) > 0:
                ax.text(
                    np.mean(xs),
                    np.mean(ys),
                    f"{score:.2f}",
                    color="white",
                    backgroundcolor="black",
                    fontsize=8,
                )

            count += 1
            print(f" -> Mask {i} kept (score {score:.2f})")

    plt.axis("off")
    plt.title(f"STEP 1: SAM3 Raw Output (Prompt: 'person') - Found {count}")

    step1_path = os.path.join(OUTPUT_DIR, "step1_sam_raw.png")
    plt.savefig(step1_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Step 1 image saved to: {step1_path}")

    # Pulizia memoria SAM
    del sam_model
    del processor
    torch.cuda.empty_cache()

    if not valid_masks_for_bcos:
        print("Nessuna maschera trovata. Stop.")
        return

    # =========================================================================
    # FASE 2: B-COS ACTION RECOGNITION (ENHANCED: FLIPPING + PROMPTS)
    # =========================================================================
    print("\n--- [STEP 2] CLASSIFICAZIONE AZIONI ---")
    from bcos_localization import (
        load_bcos_model,
        load_clip_for_text,
        tokenize_text_prompt,
        compute_attributions,
    )
    import bcos.data.transforms as custom_transforms

    bcos_m, _ = load_bcos_model()
    bcos_m.to(DEVICE).eval()
    clip_m, _ = load_clip_for_text()
    clip_m.to(DEVICE).eval()

    # Nuovi prompt estesi per lo sfondo
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

    def get_avg_emb(prompts):
        with torch.no_grad():
            weights = [tokenize_text_prompt(clip_m, p).to(DEVICE) for p in prompts]
            return torch.mean(torch.stack(weights), dim=0)

    bg_prompts = [f"a photo of {b}" for b in bg_classes]
    bg_w = get_avg_emb(bg_prompts)

    # --- IMAGE PREP ---
    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    img_tensor = prep(raw_image)  # [6, 512, 512]

    # --- GENERAZIONE HEATMAPS (CON FLIPPING) ---
    action_maps = {}
    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    scales = [448, 560]

    print("Generating Action Heatmaps (Multi-scale + Flipping)...")

    for action in VOC_ACTIONS:
        # Prompt Target Arricchiti
        target_prompts = [
            f"a clean origami {action}.",
            f"a photo of a {action}.",
            f"the {action}.",
        ]
        t_w = get_avg_emb(target_prompts)

        accumulated_maps = []

        for s in scales:
            resize_t = transforms.Resize(
                (s, s), interpolation=transforms.InterpolationMode.BICUBIC
            )

            # Creiamo tensore base [6, S, S]
            img_scaled = resize_t(img_tensor)

            # --- PASSAGGIO 1: Immagine Normale ---
            inp = img_scaled.to(DEVICE)  # [6, S, S]

            with torch.no_grad():
                _, map_t, _, _ = compute_attributions(bcos_m, inp, t_w)
                _, map_b, _, _ = compute_attributions(bcos_m, inp, bg_w)

                # Conversioni
                map_t = torch.as_tensor(map_t).cpu().float()
                map_b = torch.as_tensor(map_b).cpu().float()
                while map_t.dim() > 2:
                    map_t = map_t[0]
                while map_b.dim() > 2:
                    map_b = map_b[0]

                map_t = blur(map_t.unsqueeze(0)).squeeze()
                map_b = blur(map_b.unsqueeze(0)).squeeze()

                g_min = min(map_t.min(), map_b.min())
                g_max = max(map_t.max(), map_b.max())
                denom = g_max - g_min + 1e-8

                probs = F.softmax(
                    torch.stack([(map_b - g_min) / denom, (map_t - g_min) / denom])
                    * 20,
                    dim=0,
                )

                # Resize
                res = F.interpolate(
                    probs[1][None, None],
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                accumulated_maps.append(res)

            # --- PASSAGGIO 2: Immagine Specchiata (Flipping) ---
            # Flip sull'asse 2 (Width) del tensore [6, H, W]
            inp_flip = torch.flip(img_scaled, [2]).to(DEVICE)

            with torch.no_grad():
                _, map_t_f, _, _ = compute_attributions(bcos_m, inp_flip, t_w)
                _, map_b_f, _, _ = compute_attributions(bcos_m, inp_flip, bg_w)

                map_t_f = torch.as_tensor(map_t_f).cpu().float()
                map_b_f = torch.as_tensor(map_b_f).cpu().float()
                while map_t_f.dim() > 2:
                    map_t_f = map_t_f[0]
                while map_b_f.dim() > 2:
                    map_b_f = map_b_f[0]

                map_t_f = blur(map_t_f.unsqueeze(0)).squeeze()
                map_b_f = blur(map_b_f.unsqueeze(0)).squeeze()

                g_min_f = min(map_t_f.min(), map_b_f.min())
                g_max_f = max(map_t_f.max(), map_b_f.max())
                denom_f = g_max_f - g_min_f + 1e-8

                probs_f = F.softmax(
                    torch.stack(
                        [(map_b_f - g_min_f) / denom_f, (map_t_f - g_min_f) / denom_f]
                    )
                    * 20,
                    dim=0,
                )

                # Flip BACK (Unflip) sull'asse Width (indice 1 nel tensore 2D [H, W])
                target_prob_unflipped = torch.flip(probs_f[1], [1])

                res_f = F.interpolate(
                    target_prob_unflipped[None, None],
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                accumulated_maps.append(res_f)

        # Media di tutte le scale e flip
        action_maps[action] = torch.mean(torch.stack(accumulated_maps), dim=0).numpy()

    # --- 4. CLASSIFICAZIONE MASCHERE ---
    print("Classifying Persons based on enhanced heatmaps...")
    final_results = []

    # FIX: Iteriamo su valid_masks_for_bcos, NON su 'masks' che non esiste
    for idx, mask in enumerate(valid_masks_for_bcos):
        best_act = "unknown"
        best_score = -1.0

        for action, hmap in action_maps.items():
            val = np.mean(hmap[mask])
            if val > best_score:
                best_score = val
                best_act = action

        final_results.append((mask, best_act, best_score))
        print(f"  -> Mask {idx}: {best_act} ({best_score:.3f})")

    # --- 5. VISUALIZZAZIONE FINALE ---
    print("\n--- CREAZIONE IMMAGINE FINALE ---")
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    ax = plt.gca()

    for mask, action, score in final_results:
        color = ACTION_COLORS.get(action, (1.0, 1.0, 1.0))
        show_mask_custom(mask, ax, color)

        ys, xs = np.where(mask)
        if len(ys) > 0:
            ax.text(
                np.mean(xs),
                np.mean(ys),
                f"{action}\n{score:.2f}",
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color,
                    alpha=0.8,
                    edgecolor="white",
                    boxstyle="round,pad=0.2",
                ),
            )

    patches = [
        mpatches.Patch(color=c, label=a)
        for a, c in ACTION_COLORS.items()
        if any(r[1] == a for r in final_results)
    ]
    if patches:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.title("STEP 2: SAM3 + B-Cos (Enhanced with Flipping)")

    final_path = os.path.join(OUTPUT_DIR, "step2_final_result_enhanced.png")
    plt.savefig(final_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved enhanced result to: {final_path}")
    plt.show()


if __name__ == "__main__":
    main()
