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
IMG_PATH = "C:/Users/xavie/Desktop/Universitá/2nd year/AML/BCos_object_detection/data/VOCdevkit/VOC2012/JPEGImages/2011_004792.jpg"
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


def show_mask_custom(mask, ax, color, alpha=0.55):
    mask = np.squeeze(mask).astype(np.float32)
    color_rgba = np.concatenate([np.array(color), np.array([alpha])], axis=0)
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main():
    print(f"--- STARTING PIPELINE ON {DEVICE} ---")

    # =========================================================================
    # FASE 1: SAM3 (SEGMENTAZIONE) - Rimane Invariata
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

    raw_image = Image.open(IMG_PATH).convert("RGB")
    original_w, original_h = raw_image.size

    print("Running SAM3 inference...")
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks_tensor = output["masks"]
    scores_tensor = output["scores"]

    valid_masks_for_bcos = []
    SOGLIA = 0.05

    for i in range(len(scores_tensor)):
        score = scores_tensor[i].item()
        if score > SOGLIA:
            m_numpy = masks_tensor[i].cpu().numpy()
            m_numpy = np.squeeze(m_numpy)
            if m_numpy.shape != (original_h, original_w):
                m_numpy = cv2.resize(
                    m_numpy.astype(np.uint8),
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            valid_masks_for_bcos.append(m_numpy > 0)

    # Pulizia SAM
    del sam_model
    del processor
    torch.cuda.empty_cache()

    if not valid_masks_for_bcos:
        print("Nessuna maschera trovata.")
        return

    # =========================================================================
    # FASE 2: B-COS ACTION RECOGNITION (MULTICLASS - NO BACKGROUND)
    # =========================================================================
    print("\n--- [STEP 2] CLASSIFICAZIONE AZIONI (MULTICLASS SOFTMAX) ---")
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

    def get_avg_emb(prompts):
        with torch.no_grad():
            weights = [tokenize_text_prompt(clip_m, p).to(DEVICE) for p in prompts]
            return torch.mean(torch.stack(weights), dim=0)

    # Nota: Abbiamo rimosso completamente bg_classes e bg_prompts

    prep = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            custom_transforms.AddInverse(),
        ]
    )
    img_tensor = prep(raw_image)

    blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
    scales = [448, 560]  # Multi-scale

    # Conterrà le mappe RAW (non normalizzate tra 0 e 1) per ogni azione
    # Lista di tensori [H, W]
    raw_action_logits = []

    print("Generating Action Heatmaps...")

    for action_idx, action in enumerate(VOC_ACTIONS):
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
            img_scaled = resize_t(img_tensor)

            # 1. Forward Normale
            inp = img_scaled.to(DEVICE)
            with torch.no_grad():
                _, map_t, _, _ = compute_attributions(bcos_m, inp, t_w)

                # Prendo solo la mappa target (niente mappa background)
                map_t = torch.as_tensor(map_t).cpu().float()
                while map_t.dim() > 2:
                    map_t = map_t[0]

                # Resize alla dimensione originale
                map_t = F.interpolate(
                    map_t[None, None],
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                accumulated_maps.append(map_t)

            # 2. Forward Flipped
            inp_flip = torch.flip(img_scaled, [2]).to(DEVICE)
            with torch.no_grad():
                _, map_t_f, _, _ = compute_attributions(bcos_m, inp_flip, t_w)

                map_t_f = torch.as_tensor(map_t_f).cpu().float()
                while map_t_f.dim() > 2:
                    map_t_f = map_t_f[0]

                # Unflip
                map_t_f = torch.flip(map_t_f, [1])

                map_t_f = F.interpolate(
                    map_t_f[None, None],
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                accumulated_maps.append(map_t_f)

        # Media su scale e flip per ottenere il punteggio grezzo di questa azione
        # IMPORTANTE: Non normalizziamo min-max qui. Se un'azione è molto probabile,
        # vogliamo che il suo valore assoluto sia alto rispetto alle altre.
        avg_map = torch.mean(torch.stack(accumulated_maps), dim=0)

        # Applichiamo un leggero blur per pulire il rumore
        avg_map = blur(avg_map.unsqueeze(0)).squeeze()

        raw_action_logits.append(avg_map)
        print(f" -> Computed logits for: {action}")

    # --- SOFTMAX COMPETITIVA ---
    # Stackiamo tutto: [Num_Actions, H, W]
    logits_tensor = torch.stack(raw_action_logits, dim=0)

    # Moltiplichiamo per un fattore di temperatura (es. 10 o 20) per rendere la softmax più "decisa"
    # Se i valori grezzi di B-Cos sono piccoli (es. 0.01), la softmax sarà piatta (tutto 10%).
    # Se scaliamo, accentuiamo le differenze.
    temperature_scale = 100.0

    print(f"Applying Multiclass Softmax across {len(VOC_ACTIONS)} actions...")
    # Softmax lungo la dimensione 0 (quella delle azioni)
    # Risultato: Per ogni pixel (x,y), abbiamo una distribuzione di probabilità che somma a 1
    action_probs_tensor = F.softmax(logits_tensor * temperature_scale, dim=0).numpy()

    # Dizionario finale per la classificazione: { "running": mappa_prob_running, ... }
    action_maps = {act: action_probs_tensor[i] for i, act in enumerate(VOC_ACTIONS)}

    # --- CLASSIFICAZIONE MASCHERE ---
    print("Classifying Persons...")
    final_results = []

    for idx, mask in enumerate(valid_masks_for_bcos):
        best_act = "unknown"
        best_score = -1.0

        # Ora cerchiamo quale azione ha la probabilità media più alta dentro la maschera
        for action, hmap in action_maps.items():
            # hmap contiene valori 0.0 - 1.0 (probabilità)
            val = np.mean(hmap[mask])

            if val > best_score:
                best_score = val
                best_act = action

        final_results.append((mask, best_act, best_score))
        print(f"  -> Mask {idx}: {best_act} (Conf: {best_score:.1%})")

    # --- VISUALIZZAZIONE FINALE ---
    print("\n--- CREAZIONE IMMAGINE FINALE ---")
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    ax = plt.gca()

    used_actions = set()
    for mask, action, score in final_results:
        color = ACTION_COLORS.get(action, (1.0, 1.0, 1.0))
        used_actions.add(action)
        show_mask_custom(mask, ax, color)

        ys, xs = np.where(mask)
        if len(ys) > 0:
            ax.text(
                np.mean(xs),
                np.mean(ys),
                f"{action}\n{score:.0%}",
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

    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=a) for a in used_actions]
    if patches:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.title("STEP 2: SAM3 + B-Cos (Multiclass Softmax - No BG)")

    final_path = os.path.join(OUTPUT_DIR, "step2_softmax_nobg.png")
    plt.savefig(final_path, bbox_inches="tight", pad_inches=0)
    print(f"📸 Saved result to: {final_path}")
    plt.show()


if __name__ == "__main__":
    main()
