"""
Advanced Object Detection Visualization Pipeline.
Generates a single montage image summarizing the iterative detection process.
"""

import sys
sys.path.insert(0, "bcosification")

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import gc

from advanced_utils import AdvancedDetector
from bcos_localization import load_bcos_model, load_clip_for_text

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'smoothing_sigma': 1.0,          # Gaussian smoothing sigma (controls "blobbiness")
    'masking_mode': 'pixel_all',     # 'pixel_all': Mask detected pixels in ALL maps
                                     # 'pixel_class': Mask detected pixels in CLASS map only
                                     # 'box': Mask bounding box (legacy)
    'stopping_criterion': 'local_competition', # 'background_competition' or 'threshold'
    'threshold': 0.15,               # Used if stopping_criterion is 'threshold'
    'max_detections': 15,            # Max objects to detect per image
    'scaling_factor': 100000.0,      # Scaling for global softmax
    'montage_cols': 5,               # Number of columns in the summary montage
    'bg_tolerance': 0.8              # FG wins if FG >= BG * tolerance
}

# Process images
# We prioritize the car image as requested
priority_images = ["test_images/car.png", "test_images/cat_background.png"]
# =============================================================================
# VISUALIZATION UTILS
# =============================================================================

def draw_box_on_image(image: Image.Image, box: list, label: str, color='lime', score: float = None):
    """Draw bounding box on image using PIL."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x0, y0, x1, y1 = box
    
    # Draw rectangle
    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
    
    # Prepare label
    if score is not None:
        text = f"{label} ({score:.2f})"
    else:
        text = label
        
    # Load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw label background
    bbox = draw.textbbox((x0, y0 - 30), text, font=font)
    draw.rectangle(bbox, fill='black')
    draw.text((x0, y0 - 30), text, fill=color, font=font)
    
    return img_copy

def create_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, title: str) -> Image.Image:
    """Create a visualization of the heatmap overlaid on the image."""
    # Resize heatmap to image size
    heatmap_resized = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized)
    
    # Normalize for visualization
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    
    # Create colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    heatmap_colored = cmap(heatmap_resized)[:, :, :3] # RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)
    
    # Blend
    overlay = Image.blend(image.convert('RGB'), heatmap_img, alpha=0.5)
    
    # Add title
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
        
    draw.text((10, 10), title, fill='white', font=font, stroke_width=2, stroke_fill='black')
    
    return overlay

def create_montage(frames: list, output_path: str, cols: int = 5):
    """Stitch frames into a single montage image."""
    if not frames:
        return
        
    n_frames = len(frames)
    rows = (n_frames + cols - 1) // cols
    
    w, h = frames[0].size
    montage = Image.new('RGB', (w * cols, h * rows), (20, 20, 20)) # Dark gray background
    
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        montage.paste(frame, (c * w, r * h))
        
    montage.save(output_path)
    print(f"Saved montage to: {output_path}")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_image(image_path: str, detector: AdvancedDetector, output_dir: str):
    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {name_no_ext}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Create output directory
    save_dir = os.path.join(output_dir, name_no_ext)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Compute Heatmaps (with smoothing)
    class_heatmaps = detector.compute_all_class_heatmaps(
        image, 
        normalize=False, 
        smoothing_sigma=CONFIG['smoothing_sigma']
    )
    
    bg_heatmaps = detector.compute_background_heatmaps(
        image, 
        normalize=False, 
        smoothing_sigma=CONFIG['smoothing_sigma']
    )
    
    # 2. Global Softmax
    print("\nApplying global softmax...")
    prob_maps, prob_bg_maps = detector.apply_global_softmax(
        class_heatmaps, 
        bg_heatmaps, 
        scaling_factor=CONFIG['scaling_factor']
    )
    
    # 3. Iterative Detection
    detections = detector.iterative_detection(
        image, 
        prob_maps, 
        prob_bg_maps,
        class_heatmaps=class_heatmaps, # Pass raw heatmaps for segmentation
        max_detections=CONFIG['max_detections'],
        masking_mode=CONFIG['masking_mode'],
        stopping_criterion=CONFIG['stopping_criterion'],
        threshold=CONFIG['threshold'],
        bg_tolerance=CONFIG.get('bg_tolerance', 0.9),
        min_score=0.03 # Filter out weak detections (Building is ~0.035, Window is ~0.027)
    )
    
    # 4. Generate Visualization Frames
    print("\nGenerating visualization frames...")
    frames = []
    
    # Frame 0: Original
    frames.append(image.copy())
    
    # Cumulative result image
    cumulative_image = image.copy()
    
    for i, det in enumerate(detections):
        # Frame A: Heatmap of the detection
        heatmap_frame = create_heatmap_overlay(
            image, 
            det['heatmap'], 
            f"#{i+1} {det['class_name']} ({det['score']:.2f})"
        )
        frames.append(heatmap_frame)
        
        # Frame B: Box on cumulative image
        cumulative_image = draw_box_on_image(
            cumulative_image, 
            det['box'], 
            det['class_name'], 
            score=det['score']
        )
        # We don't add cumulative image every time to save space, 
        # just the heatmap frame is enough context usually, 
        # but let's add the cumulative state every 5 detections or so?
        # Actually, let's just add the heatmap frame.
        # And at the very end, the final result.
    
    # Add final result
    frames.append(cumulative_image)
    
    # Save Montage
    montage_path = os.path.join(save_dir, f"{name_no_ext}_summary.jpg")
    create_montage(frames, montage_path, cols=CONFIG['montage_cols'])
    
    # Also save the final result separately
    cumulative_image.save(os.path.join(save_dir, f"{name_no_ext}_final_result.jpg"))
    
    # Cleanup
    del class_heatmaps, bg_heatmaps, prob_maps, prob_bg_maps, frames, cumulative_image
    gc.collect()

def main():
    print("Initializing Advanced Detection Pipeline...")
    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
        
    # Load Models
    print("\nLoading models...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model, _ = load_clip_for_text()
    
    detector = AdvancedDetector(bcos_model, clip_model, bcos_device)
    
    output_dir = "output_sequences_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    # We prioritize the car image as requested
    priority_images = ["test_images/car.png", "test_images/cat_background.png"]
    
    for img_path in priority_images:
        process_image(img_path, detector, output_dir)
        
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
