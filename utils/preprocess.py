"""
Utility functions for image preprocessing that preserve the original aspect ratio.
The standard CLIP transform does `Resize(shorter_side=224) + CenterCrop(224)`.  For attribution methods we want **only the resize** (no cropping) so that the model sees the whole image.
"""

import torchvision.transforms as T
from PIL import Image

def aspect_preserving_preprocess(image: Image.Image, target_small_edge: int = 224):
    """Resize the *shorter* edge of ``image`` to ``target_small_edge`` while keeping the aspect ratio.
    Returns a torch tensor ready for a CLIP visual backbone (normalized with CLIP's mean/std).
    """
    w, h = image.size
    if w < h:
        new_w = target_small_edge
        new_h = int(round(h * target_small_edge / w))
    else:
        new_h = target_small_edge
        new_w = int(round(w * target_small_edge / h))

    resize = T.Resize((new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    img_resized = resize(image)
    return normalize(to_tensor(img_resized))
