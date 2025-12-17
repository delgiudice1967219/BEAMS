import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    GaussianBlur,
    InterpolationMode,
)
import sys

# --- CONFIGURATION MATCHING YOUR PIPELINE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALES = [448, 560]
BG_CLASSES = [
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
    "keyboard",
    "helmet",
    "cloud",
    "house",
    "mountain",
    "ocean",
    "road",
    "rock",
    "street",
    "valley",
    "bridge",
    "sign",
]
FG_TEMPLATES = [
    "a clean origami {}.",
    "a photo of a {}.",
    "the {}.",
]

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


# --- 1. MODEL PATCHING (ResNet Fixes) ---
def forward_resnet_backbone(model, image):
    """Run ResNet up to Layer 3."""
    x = image.type(model.dtype)
    visual = model.visual
    x = visual.relu1(visual.bn1(visual.conv1(x)))
    x = visual.relu2(visual.bn2(visual.conv2(x)))
    x = visual.relu3(visual.bn3(visual.conv3(x)))
    x = visual.avgpool(x)
    x = visual.layer1(x)
    x = visual.layer2(x)
    x = visual.layer3(x)
    return x  # Output: [B, 1024, H/16, W/16]


class CLIPGradCAMWrapper(torch.nn.Module):
    """Wrapper to allow gradients to flow from Logits -> Layer4 -> AttnPool"""

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, layer3_output, text_features, h_img, w_img):
        # 1. Run Layer 4 (We want grads here)
        x = self.clip.visual.layer4(layer3_output)
        self.feature_map = x  # Save for CAM calculation
        self.feature_map.retain_grad()  # Hook gradient

        # 2. Attn Pool
        H, W = h_img // 32, w_img // 32  # Calculate grid size based on input image
        # Note: We must upscale grid size back to "original" for attnpool logic
        # attnpool expects sizes compatible with positional embeddings
        # We use the raw feature size * 32
        h_feat, w_feat = x.shape[-2], x.shape[-1]

        x = self.clip.visual.attnpool(x, h_feat * 32, w_feat * 32)
        x = x / x.norm(dim=-1, keepdim=True)

        # 3. Logits
        logits = x @ text_features.t()
        return logits


# --- 2. UTILS ---
def get_text_features(model, class_names, templates):
    with torch.no_grad():
        weights = []
        for name in class_names:
            texts = [t.format(name) for t in templates]
            toks = clip.tokenize(texts).to(DEVICE)
            emb = model.encode_text(toks)
            emb /= emb.norm(dim=-1, keepdim=True)
            weights.append(emb.mean(dim=0))
        weights = torch.stack(weights)
        weights /= weights.norm(dim=-1, keepdim=True)
    return weights


def get_transform(size):
    return Compose(
        [
            Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            lambda i: i.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


# --- 3. CORE LOGIC (Apples-to-Apples) ---
def compute_raw_cam(wrapper, inp_tensor, text_feats, target_idx, h_img, w_img):
    """Computes RAW GradCAM (no normalization)."""
    wrapper.zero_grad()

    # Forward
    logits = wrapper(inp_tensor, text_feats, h_img, w_img)

    # Backward
    score = logits[0, target_idx]
    score.backward(retain_graph=True)

    # GradCAM: Weights = GlobalAvgPool(Gradients)
    gradients = wrapper.feature_map.grad
    activations = wrapper.feature_map
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # Map = ReLU(Sum(Weights * Activations))
    cam = torch.sum(weights * activations, dim=1)
    cam = F.relu(cam)  # [1, H_feat, W_feat]

    return cam.detach()  # Keep on GPU for processing


def process_image_comparable(img_path, model, bg_feats, fg_feat, wrapper):
    """
    Implements your exact: Scales -> Flip -> Blur -> JointNorm -> Softmax*20
    """
    try:
        # Use the safe read method if you have it, otherwise standard PIL
        # If using standard PIL and path has special chars, this might fail,
        # but for now we assume path is handled or standard
        if not os.path.exists(img_path):
            return None
        orig_img = Image.open(img_path)
    except Exception as e:
        return None

    base_w, base_h = orig_img.size

    accumulated_maps = []
    blur_transform = GaussianBlur(kernel_size=5, sigma=1.0)

    # Combine [FG, BG] for logits
    # Index 0 = FG, Index 1 = BG
    text_features = torch.stack([fg_feat, bg_feats]).to(DEVICE)

    for s in SCALES:
        # Preprocess
        preprocess = get_transform(s)
        img_tensor = preprocess(orig_img).unsqueeze(0).to(DEVICE)

        # We need Layer 3 output to feed our wrapper
        with torch.no_grad():
            l3_out = forward_resnet_backbone(model, img_tensor)

        # --- Handle Flip (True/False) ---
        for flip in [False, True]:
            # Generate the feature map (flipped or not)
            if flip:
                current_l3 = torch.flip(l3_out, [3])
            else:
                current_l3 = l3_out

            # 1. Compute Raw Maps (Foreground & Background)
            # IMPORTANT: Detach to make it a leaf, then enable grad
            with torch.enable_grad():
                current_l3 = current_l3.detach()
                current_l3.requires_grad = True

                # Target (Index 0)
                map_t = compute_raw_cam(wrapper, current_l3, text_features, 0, s, s)

                # Background (Index 1)
                # We need to detach/reset grad for the second pass to avoid graph issues
                # or simpler: just re-use the tensor since we kept the graph in compute_raw_cam
                map_b = compute_raw_cam(wrapper, current_l3, text_features, 1, s, s)

            # 2. Blur
            map_t = blur_transform(map_t)
            map_b = blur_transform(map_b)

            # 3. JOINT NORMALIZATION
            g_min = min(map_t.min(), map_b.min())
            g_max = max(map_t.max(), map_b.max())
            denom = g_max - g_min + 1e-8

            map_t_norm = (map_t - g_min) / denom
            map_b_norm = (map_b - g_min) / denom

            # 4. Softmax * 20
            stack = torch.stack([map_b_norm, map_t_norm], dim=0)  # [2, 1, h, w]
            probs = F.softmax(stack * 20, dim=0)  # Softmax over channel dim

            # Extract Target Prob (Channel 1)
            target_prob = probs[1, 0]

            # Unflip if needed
            if flip:
                target_prob = torch.flip(target_prob, [1])

            # Resize to Original Image Size
            prob_resized = F.interpolate(
                target_prob.unsqueeze(0).unsqueeze(0),
                size=(base_h, base_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            accumulated_maps.append(prob_resized)

    # Average over scales and flips
    if not accumulated_maps:
        return np.zeros((base_h, base_w))

    final_prob_map = torch.mean(torch.stack(accumulated_maps), dim=0)
    return final_prob_map.cpu().numpy()


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--cam_out_dir", type=str, default="./comparable_results")
    parser.add_argument(
        "--model_path", type=str, default="RN50"
    )  # Use OpenAI path or local
    args = parser.parse_args()

    # Load Model
    model, _ = clip.load(args.model_path, device=DEVICE)
    wrapper = CLIPGradCAMWrapper(model).to(DEVICE)

    # Precompute BG Weights
    print("Computing BG weights...")
    bg_feats = get_text_features(model, BG_CLASSES, ["a photo of {}."])
    # Average BG features into one vector (as per your logic)
    bg_vector = bg_feats.mean(dim=0)

    # Cache FG Weights
    fg_cache = {}

    # Load Dataset
    with open(args.split_file) as f:
        file_list = [x.strip() for x in f.readlines()]

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    print(f"Processing {len(file_list)} images...")

    # This loop assumes we process every image for every class present in GT
    # To match your benchmark script exactly, we need to know the 'target' class.
    # Since we are generating offline, we will read the XML to find valid classes.

    import xml.etree.ElementTree as ET

    for img_name in tqdm(file_list):
        img_name = img_name.replace(".jpg", "")  # Safety
        img_path = os.path.join(args.img_root, img_name + ".jpg")
        xml_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = [obj.find("name").text for obj in root.findall("object")]
            unique_classes = set(objects)
        except:
            continue

        result_keys = []
        result_cams = []

        for cls_name in unique_classes:
            if cls_name not in VOC_CLASSES:
                continue

            # Get FG Vector
            if cls_name not in fg_cache:
                fg_cache[cls_name] = get_text_features(model, [cls_name], FG_TEMPLATES)[
                    0
                ]
            fg_vector = fg_cache[cls_name]

            # RUN PIPELINE
            prob_map = process_image_comparable(
                img_path, model, bg_vector, fg_vector, wrapper
            )

            result_keys.append(VOC_CLASSES.index(cls_name))
            result_cams.append(prob_map)

        # Save NPY
        if result_keys:
            np.save(
                os.path.join(args.cam_out_dir, img_name + ".npy"),
                {
                    "keys": np.array(result_keys),
                    "attn_highres": np.array(result_cams),  # Shape [N_classes, H, W]
                },
            )
