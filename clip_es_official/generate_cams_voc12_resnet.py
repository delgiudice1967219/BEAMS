# -*- coding:UTF-8 -*-
from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip_text import (
    class_names,
    new_class_names,
    BACKGROUND_CATEGORY,
)  # , imagenet_templates
import argparse
from lxml import etree
import time
from torch import multiprocessing
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings

warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split(".")[0] == "3" else 0


# --- MONKEY PATCH START ---
def forward_last_layer_patch(self, image_features, text_features):
    if not hasattr(self.visual, "transformer"):
        # --- ResNet Logic ---
        # image_features is (Batch, Channels, H_feat, W_feat)
        H_feat, W_feat = image_features.shape[-2], image_features.shape[-1]

        # The attnpool layer expects the ORIGINAL image dimensions to calculate
        # the positional embedding interpolation correctly.
        # We reconstruct them by multiplying the feature map size by the stride (32).
        H, W = H_feat * 32, W_feat * 32

        # Pass scaled H and W to attnpool
        x = self.visual.attnpool(image_features, H, W)

        x = x / x.norm(dim=-1, keepdim=True)
        logits = x @ text_features.t()
        return logits, None
    else:
        # --- ViT Logic ---
        x, attn_weight = self.visual.transformer.resblocks[
            self.visual.transformer.layers - 1
        ](image_features)
        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        x = x / x.norm(dim=-1, keepdim=True)
        logits = x @ text_features.t()
        return logits, attn_weight


import types
from clip.model import CLIP

CLIP.forward_last_layer = forward_last_layer_patch
# --- MONKEY PATCH END ---


# --- COPY THIS BLOCK AFTER YOUR IMPORTS ---


def forward_resnet_backbone(model, image):
    """
    Helper: Runs ResNet Stem -> Layer 1 -> Layer 2 -> Layer 3.
    Returns the feature map ready for Layer 4.
    """
    x = image.type(model.dtype)
    visual = model.visual

    # Stem
    x = visual.relu1(visual.bn1(visual.conv1(x)))
    x = visual.relu2(visual.bn2(visual.conv2(x)))
    x = visual.relu3(visual.bn3(visual.conv3(x)))
    x = visual.avgpool(x)

    # Run Layers 1-3
    x = visual.layer1(x)
    x = visual.layer2(x)
    x = visual.layer3(x)
    # Output shape: (Batch, 1024, H/16, W/16)
    return x


def forward_last_layer_patch(self, image_features, text_features):
    """
    Patch: Runs Layer 4 -> AttnPool -> Logits.
    """
    if not hasattr(self.visual, "transformer"):
        # --- ResNet Logic ---

        # 1. Run Layer 4 (The TARGET LAYER)
        # We run it here so GradCAM's hooks can catch the output!
        x = self.visual.layer4(image_features)

        # 2. Get dimensions for AttnPool
        # x is (Batch, 2048, H_feat, W_feat)
        H_feat, W_feat = x.shape[-2], x.shape[-1]

        # 3. Reconstruct original image size (Stride is 32)
        # The modified CLIP library needs this to interpolate position embeddings
        H, W = H_feat * 32, W_feat * 32

        # 4. Run Attention Pooling
        x = self.visual.attnpool(x, H, W)

        # 5. Normalization & Projection
        x = x / x.norm(dim=-1, keepdim=True)
        logits = x @ text_features.t()

        return logits, None
    else:
        # --- ViT Logic (Unchanged) ---
        x, attn_weight = self.visual.transformer.resblocks[
            self.visual.transformer.layers - 1
        ](image_features)
        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        x = x / x.norm(dim=-1, keepdim=True)
        logits = x @ text_features.t()
        return logits, attn_weight


# Apply the patch immediately
import types
from clip.model import CLIP

CLIP.forward_last_layer = forward_last_layer_patch


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i * part : (i + 1) * part])
    dataset_list.append(dataset[(i + 1) * part :])

    return dataset_list


def zeroshot_classifier(classnames, templates, model, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [
                template.format(classname) for template in templates
            ]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_resize(h, w):
    return Compose(
        [
            Resize((h, w), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(
            int(np.ceil(scale * int(ori_height) / patch_size) * patch_size),
            int(np.ceil(scale * int(ori_width) / patch_size) * patch_size),
        )
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs


def perform(
    process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam
):
    n_gpus = torch.cuda.device_count()
    device_id = "cuda:{}".format(process_id % n_gpus)
    databin = dataset_list[process_id]
    model = model.to(device_id)
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)

    is_resnet = "RN" in args.model

    for im_idx, im in enumerate(tqdm(databin)):
        img_path = os.path.join(args.img_root, im)
        # Windows/Linux path fix
        xmlfile = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

        try:
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
        except Exception as e:
            continue

        ori_width = int(data["size"]["width"])
        ori_height = int(data["size"]["height"])

        label_list = []
        label_id_list = []
        for obj in data["object"]:
            obj["name"] = new_class_names[class_names.index(obj["name"])]
            if obj["name"] not in label_list:
                label_list.append(obj["name"])
                label_id_list.append(new_class_names.index(obj["name"]))

        if len(label_list) == 0:
            print("{} not have valid object".format(im))
            continue  # Changed return to continue to not stop the whole process

        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]

        highres_cam_all_scales = []
        refined_cam_all_scales = []

        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)

            # --- CORRECTED FEATURE EXTRACTION ---
            if is_resnet:
                # Run UP TO Layer 3 only.
                # Layer 4 will run inside GradCAM (in the patch above).
                image_features = forward_resnet_backbone(model, image)
                attn_weight_list = []
            else:
                # ViT standard execution
                image_features, attn_weight_list = model.encode_image(image, h, w)

            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = bg_text_features.to(device_id)
            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)

            # Input to GradCAM is now the output of Layer 3
            input_tensor = [image_features, text_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                keys.append(new_class_names.index(label))
                targets = [ClipOutputTarget(label_list.index(label))]

                grayscale_cam, logits_per_image, attn_weight_last = cam(
                    input_tensor=input_tensor, targets=targets, target_size=None
                )

                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam_highres = cv2.resize(
                    grayscale_cam, (ori_width, ori_height)
                )
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if is_resnet:
                    # ResNet: No attention refinement possible, just use highres CAM
                    refined_cam_to_save.append(torch.tensor(grayscale_cam_highres))
                else:
                    # ... (Keep existing ViT refinement logic here) ...
                    # For brevity, I'm assuming you kept the ViT logic inside the 'else' block
                    pass

            keys = torch.tensor(keys)
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save, dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))

        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        np.save(
            os.path.join(args.cam_out_dir, im.replace("jpg", "npy")),
            {
                "keys": keys.numpy(),
                "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
            },
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--img_root", type=str, default="/home/xxx/datasets/VOC2012/JPEGImages"
    )
    parser.add_argument("--split_file", type=str, default="./voc12/train.txt")
    parser.add_argument("--cam_out_dir", type=str, default="./final/ablation/baseline")
    parser.add_argument(
        "--model", type=str, default="/home/xxx/pretrained_models/clip/ViT-B-16.pt"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + ".jpg" for x in train_list]

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, _ = clip.load(args.model, device=device)
    bg_text_features = zeroshot_classifier(
        BACKGROUND_CATEGORY, ["a clean origami {}."], model, device
    )  # ['a rendering of a weird {}.'], model)
    fg_text_features = zeroshot_classifier(
        new_class_names, ["a clean origami {}."], model, device
    )  # ['a rendering of a weird {}.'], model)

    # --- MODIFIED: Target Layers Selection ---
    if "RN" in args.model:
        # For ResNet, we target the last layer of the final bottleneck block
        # Usually layer4[-1]
        target_layers = [model.visual.layer4]
        reshape_transform_func = None  # ResNets don't need reshape
    else:
        # For ViT
        target_layers = [model.visual.transformer.resblocks[-1].ln_1]
        reshape_transform_func = reshape_transform

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform_func,
    )

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam)
    else:
        multiprocessing.spawn(
            perform,
            nprocs=args.num_workers,
            args=(dataset_list, args, model, bg_text_features, fg_text_features, cam),
        )
