# Zero-Shot Action Detection with B-Cos and SAM3

## Abstract

This repository implements a novel framework for **Zero-Shot Action Detection** by synergizing **Segment Anything Model 3 (SAM3)** for instance segmentation and **B-Cos (Bi-Cosine)** networks for inherently interpretable, opaque-box classification. By leveraging the dense, explainable contribution maps from B-Cos models aligned with CLIP text embeddings, and refining them with precise person masks from SAM3, this method achieves robust action localization and classification without training on action-specific annotations. This approach addresses the "Right for the Right Reasons" paradigm in computer vision, ensuring that action predictions are grounded in relevant visual features.

## 1. Introduction

Action detection traditionally relies on training supervised models on fixed vocabularies, limiting their generalization to unseen classes. Feature alignment methods like CLIP allow for zero-shot capabilities but often lack precise spatial localization.

**B-Cosification** transforms Deep Neural Networks (DNNs) to be inherently interpretable by replacing linear transformations with a Bi-Cosine operator. This results in contribution maps that visually explain the decision-making process.

**SAM3** provides state-of-the-art class-agnostic segmentation, accurately isolating potential actors (people) in a scene.

This project combines these two powerful technologies to:
1.  Isolate actors using SAM3.
2.  Compute action-specific contribution maps using a B-Cos-trained CLIP backbone.
3.  Score and classify actions based on the alignment of interpretable features within the actor's region and their context.

## 2. Methodology

The pipeline consists of three main stages:

### 2.1. Instance Segmentation (SAM3)
The image is processed by **SAM3** to generate binary masks for all "person" instances.
-   SAM3 is prompted with the text "person" to identify candidates.
-   Masks are filtered by confidence scores to ensure quality candidates.
-   This step provides the "Anchor" for action classification, defining *where* the action is happening.

### 2.2. Explainable Feature Extraction (B-Cos)
A **B-Cos ResNet50** model, pretrained with CLIP alignment, is used to generate dense contribution maps for the target action classes.
-   **Embeddings**: Text embeddings for each action (e.g., "jumping", "phoning") and a global background class are pre-computed using the CLIP text encoder.
-   **Multi-Scale Inference**: Attribution maps are computed at multiple scales (e.g., 224, 448, 560, 672, 784) to capture details at various resolutions.
-   **Gaussian Blur**: A Gaussian Blur is applied to the raw contribution maps to smooth artifacts.
-   **Joint Normalization & Softmax**: The maps for the target class and background are typically normalized jointly (using global min/max) and then passed through a Softmax layer with a high temperature scaling (e.g., * 20) to produce sharp, probability-like heatmaps.

### 2.3. Scoring and Classification (Action Detection)
For each person mask identified by SAM3:
1.  **Person Score ($S_{person}$)**: Use the B-Cos contribution map *inside* the person mask. High contribution values indicate strong visual evidence for the action on the person's body.
2.  **Context Score ($S_{context}$)**: Use the contribution map in a dilated region *around* the person (Context Mask). This captures interaction with objects (e.g., "ridingbike" needs a bike).
3.  **Final Score**: A weighted combination of the Person Score and Context Score determines the likelihood of the action.
    $$ Score = S_{person} + \lambda \cdot S_{context} $$
4.  **Classification**: The action with the highest score is assigned to the person.

### 2.4. Zero-Shot Segmentation Pipeline
In addition to Action Detection, the repository includes a specific pipeline for **Zero-Shot Semantic Segmentation**, implemented in `visualize_voc.py`:

This pipeline leverages the interpretable B-Cos maps to generate high-quality segmentation masks without direct supervision:
1.  **Feature Computation**: Similar to above, multi-scale contribution maps are computed for a target class (e.g., "aeroplane") vs background.
2.  **Heatmap Generation**: Joint normalization and Softmax competition produce a dense probability map.
3.  **Hard Masking**: A threshold (e.g., 0.6) is applied to create a preliminary "Hard Mask".
4.  **Refinement with DenseCRF**: A fully connected Conditional Random Field (DenseCRF) is applied as a post-processing step. It uses the image's RGB pixel values to respect object boundaries, significantly refining the coarse Hard Mask into a detailed segmentation mask.

## 3. Repository Structure

-   `benchmark_bcos_voc_action.py`: **Main Benchmark Script**. Runs the full pipeline on Pascal VOC 2012 Action Validation set using B-Cos + SAM3.
-   `benchmark_sam_voc_action.py`: Baseline script using only SAM3 for action detection (using prompts like "person riding a bike").
-   `visualize_voc.py`: Visualization script demonstrating the Zero-Shot Segmentation pipeline with DenseCRF refinement.
-   `bcos_utils.py`: Utility functions for loading B-Cos models, computing attributions, and processing text prompts.
-   `bcosification/`: Submodule containing the core B-Cos library.
-   `sam3/`: Directory for SAM3 model code and checkpoints.
-   `main.ipynb`: Jupyter Notebook for demonstrations and ensuring SAM3 setup.
-   `requirements.txt`: Python dependencies.

## 4. Installation

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (recommended)

### Setup
1.  Clone the repository and submodules:
    ```bash
    git clone https://github.com/yourusername/BCos_object_detection.git
    cd BCos_object_detection
    ```

2.  **Install Dependencies**:
    We provide a consolidated `requirements.txt` that includes dependencies for B-Cos, SAM3, and CLIP-ES.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `pydensecrf`, you may need to install it manually or from source:*
    ```bash
    pip install git+https://github.com/lucasb-eyer/pydensecrf.git
    ```

3.  **SAM3 Setup**: Ensure the SAM3 model checkpoint (`sam3.pt`) is downloaded. The `main.ipynb` notebook contains cells to automate this download using `huggingface_hub`.
    ```bash
    # Example logic to download model (see main.ipynb)
    huggingface-cli download facebook/sam3 sam3.pt --local-dir sam3_model
    ```

## 5. Usage

### Data Preparation
Ensure the Pascal VOC 2012 dataset is available in `data/VOCdevkit/VOC2012/`. The directory structure should look like:
```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
            └── Action/
```

### Running the B-Cos Benchmark
To evaluate the proposed B-Cos + SAM3 method:

```bash
python benchmark_bcos_voc_action.py --limit 50 --checkpoint sam3_model/sam3.pt
```

**Arguments:**
-   `--limit`: (Optional) Limit the number of images to process (useful for testing).
-   `--checkpoint`: Path to the SAM3 model checkpoint.

### Running Visualization (Segmentation Pipeline)
To visualize the zero-shot segmentation capabilities with DenseCRF:

```bash
python visualize_voc.py
```
This script will process random samples from the VOC dataset and save visualization plots (Original, Heatmap, Hard Mask, CRF Result) to the `segmentation_viz/` directory.

## 6. Results

The benchmark scripts output:
1.  **Per-Class AP**: Average Precision for each action class (e.g., Jumping, Phoning, RidingBike).
2.  **mAP**: Mean Average Precision across all classes.
3.  **Confusion Matrix**: A heatmap visualization (`confusion_matrix_bcos.png`) showing the relationship between Ground Truth and Predicted actions.

## 7. Citation

If you use this code or the B-Cos methodology, please cite the original B-Cosification paper:

```bibtex
@inproceedings{arya2024bcosification,
  title={B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable},
  author={Arya, Shreyash and Rao, Sukrut and B{\"o}hle, Moritz and Schiele, Bernt},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

For Segment Anything 3 (SAM3), please refer to the official Meta AI research.