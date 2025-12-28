# BEAMS: B-Cos Explanations As Masks for Segmentation

**A training-free framework for Zero-Shot Semantic Segmentation and Instance-Level Action Detection.**

## 📖 Overview

BEAMS (**B**-Cos **E**xplanations **A**s **M**asks for **S**egmentation) leverages the intrinsic interpretability of B-Cos networks to perform dense prediction tasks without any pixel-level supervision.

By combining the open-vocabulary capabilities of **CLIP** with the dense, explainable contribution maps of **B-Cos ResNet50**, this repository implements two distinct pipelines:

1. **Zero-Shot Semantic Segmentation**: Generates high-quality segmentation masks by forcing a pixel-wise competition between target concepts and background, refined via DenseCRF.
2. **Zero-Shot Action Detection**: A neuro-symbolic approach that synergizes **SAM 3** (for geometric object proposals) with **B-Cos** (for semantic verification) to classify actions in static images.

This approach addresses the "Right for the Right Reasons" paradigm, ensuring predictions are grounded in relevant visual features rather than opaque artifacts.

---

## 🚀 Key Features

* **Training-Free**: No fine-tuning on Pascal VOC or Action datasets required.
* **Intrinsically Interpretable**: Uses "White-Box" B-Cos networks where model weights align dynamically with input features.
* **Neuro-Symbolic Pipeline**: Decouples "Where is the object?" (SAM 3) from "What is it doing?" (B-Cos/CLIP).
* **Competitive Decoding**: Implements a novel softmax-based competition between foreground concepts and explicit background prototypes to reduce noise.

---

## 🛠️ Methodology

### 1. The Core: B-Cos + CLIP

The backbone of this project is a **B-Cos ResNet50**. Unlike standard CNNs, B-Cos networks replace linear transformations with a Bi-Cosine operator. This ensures that the output is mathematically decomposable into **contribution maps** that strictly visualize the spatial support for a specific class. These maps are aligned with **CLIP text embeddings** to allow for zero-shot querying.

### 2. Task A: Zero-Shot Semantic Segmentation

*Implemented in `visualize_voc.py*`

This pipeline transforms raw explanation maps into dense segmentation masks:

1. **Prompt Ensembling**: Target classes (e.g., "dog") and a comprehensive list of "Stuff" classes (sky, wall, ground) are encoded via CLIP.
2. **Dense Attribution**: B-Cos computes contribution maps for Target vs. Background.
3. **Competitive Inference**:
* **Multi-Scale Aggregation**: Maps are computed at scales `[224, 448, 560, 672, 784]` and averaged.
* **Joint Normalization**: Target and Background maps share a global normalization context.
* **Temperature-Scaled Softmax**: A binary softmax (Target vs. Background) with high temperature () forces a sharp decision at every pixel.


4. **Refinement**: The resulting "blobby" probability map is refined using a **DenseCRF** (Conditional Random Field), which uses low-level RGB cues to snap boundaries to edges.

### 3. Task B: Instance-Level Action Detection

*Implemented in `benchmark_bcos_voc_action.py*`

This pipeline extends the concept to classify what a specific person is doing:

1. **Geometric Proposal (SAM 3)**: The image is prompted with "person" to generate high-fidelity instance masks. This defines the *geometry*.
2. **Semantic Pivot (B-Cos)**: We compute contribution maps for all action classes (e.g., "running", "phoning") + background.
3. **Dual-Region Scoring**:
* **Intrinsic Score**: Aggregates high-confidence semantic pixels *inside* the person mask.
* **Context Score**: Aggregates evidence in a dilated region *around* the person (capturing objects like bikes or phones).


4. **Classification**: The action with the highest combined robust score is assigned to the instance.

---

## 📂 Repository Structure

* `benchmark_bcos_voc_action.py`: **Main Action Benchmark**. Runs the full SAM3 + B-Cos pipeline on Pascal VOC Actions.
* `visualize_voc.py`: **Segmentation Demo**. visualizes the Semantic Segmentation pipeline (B-Cos Map  Softmax  CRF).
* `benchmark_sam_voc_action.py`: Baseline script using vanilla SAM3 for action detection.
* `bcos_utils.py`: Utilities for loading models, computing attributions, and handling CLIP prompts.
* `utils_voc_actions.py`: XML parsing and IoU calculation tools.
* `bcosification/`: Submodule containing the core B-Cos library.
* `sam3/`: Directory for SAM3 model code.

---

## 💻 Installation

### Prerequisites

* Python 3.8+
* CUDA-enabled GPU (High VRAM recommended for SAM3 + B-Cos)

### Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/BCos_object_detection.git
cd BCos_object_detection

```


2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


*Note: `pydensecrf` is required for the segmentation pipeline. If pip fails, install from source:*
```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

```


3. **Download SAM 3 Checkpoint**:
You can use the HuggingFace CLI or download manually from the official repo.
```bash
huggingface-cli download facebook/sam3 sam3.pt --local-dir sam3_model

```



---

## 🏃 Usage

### 1. Data Preparation

Ensure the Pascal VOC 2012 dataset is structured as follows:

```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
            └── Action/

```

### 2. Run Zero-Shot Semantic Segmentation

To visualize the transformation from explanation maps to CRF-refined masks:

```bash
python visualize_voc.py

```

*Outputs:* Visualization plots saved to `segmentation_viz/` showing Original Image, Raw Heatmap, Hard Mask, and CRF Prediction.

### 3. Run Zero-Shot Action Detection

To evaluate the B-Cos + SAM3 pipeline on the VOC Action Validation set:

```bash
python benchmark_bcos_voc_action.py --limit 100 --checkpoint sam3_model/sam3.pt

```

**Arguments:**

* `--limit`: Number of images to process (remove to run full dataset).
* `--device`: `cuda` or `cpu`.

---

## 📊 Results & Performance

| Task | Metric | Method | Score |
| --- | --- | --- | --- |
| **Segmentation** | mIoU | **BEAMS (Ours)** | **~0.49** |
|  |  | CLIP-ES (ResNet) | ~0.38 |
| **Action Detection** | mAP | **BEAMS (Ours)** | **0.68** |
|  |  | SAM 3 Baseline | 0.48 |

*See the `results/` folder (generated after running benchmarks) for detailed per-class AP and confusion matrices.*

---

## 📜 Citation

If you use this code or the methodology, please cite the original B-Cosification paper:

```bibtex
@inproceedings{arya2024bcosification,
  title={B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable},
  author={Arya, Shreyash and Rao, Sukrut and B{\"o}hle, Moritz and Schiele, Bernt},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

```

For SAM 3, please refer to the official Meta AI research.
