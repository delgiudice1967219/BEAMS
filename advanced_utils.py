"""
Advanced utilities for iterative object detection with global competition.

Key Features:
1. Pre-compute text embeddings for all synonyms and backgrounds
2. Synonym fusion: average heatmaps across synonyms for each target class
3. Global softmax: competition across all classes (foreground + background)
4. Iterative masking: detect multiple instances by masking detected regions
"""

import sys
sys.path.insert(0, "bcosification")

import torch
import numpy as np
from PIL import Image
import clip
from typing import Dict, List, Tuple, Optional

from config_classes import TARGET_CLASSES, BACKGROUND_CLASSES
from bcos_localization import get_bcos_heatmap
from box_utils import extract_box


class AdvancedDetector:
    """
    Advanced iterative object detector with global competition strategy.
    """
    
    def __init__(self, bcos_model, clip_model, device):
        """
        Initialize detector and pre-compute all text embeddings.
        
        Args:
            bcos_model: B-cos model for heatmap generation
            clip_model: CLIP model for text encoding
            device: torch device for B-cos model
        """
        self.bcos_model = bcos_model
        self.clip_model = clip_model
        self.device = device
        
        # Move CLIP model to CPU for text encoding (MPS has issues with text encoding)
        self.clip_model_cpu = clip_model.cpu()
        
        print("Pre-computing text embeddings...")
        self.class_embeddings = {}  # {class_name: averaged_embedding}
        self.background_embeddings = []
        
        # Pre-compute embeddings for each target class (average across synonyms)
        for class_name, synonyms in TARGET_CLASSES.items():
            embeddings = []
            for synonym in synonyms:
                text_tokens = clip.tokenize([synonym])
                with torch.no_grad():
                    embedding = self.clip_model_cpu.encode_text(text_tokens)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding)
            
            # Average embeddings across synonyms
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm(dim=-1, keepdim=True)
            self.class_embeddings[class_name] = avg_embedding
        
        # Pre-compute background embeddings
        for bg_class in BACKGROUND_CLASSES:
            text_tokens = clip.tokenize([bg_class])
            with torch.no_grad():
                embedding = self.clip_model_cpu.encode_text(text_tokens)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            self.background_embeddings.append(embedding)
        
        print(f"Pre-computed {len(self.class_embeddings)} target class embeddings")
        print(f"Pre-computed {len(self.background_embeddings)} background embeddings")
    
    def compute_class_heatmap_with_fusion(self, image: Image.Image, class_name: str, 
                                          normalize: bool = True, smoothing_sigma: float = 0.0) -> np.ndarray:
        """
        Compute heatmap for a target class using synonym fusion.
        
        Args:
            image: PIL Image
            class_name: Target class name from TARGET_CLASSES
            normalize: If True, normalize heatmap to [0, 1].
            smoothing_sigma: Sigma for Gaussian smoothing (0.0 to disable).
            
        Returns:
            Fused heatmap (H, W)
        """
        synonyms = TARGET_CLASSES[class_name]
        heatmaps = []
        
        for synonym in synonyms:
            contribs, _, _ = get_bcos_heatmap(
                self.bcos_model, image, synonym, self.clip_model, 
                self.device, return_raw=True
            )
            # Take positive contributions only
            heatmap = np.maximum(contribs, 0)
            heatmaps.append(heatmap)
        
        # Average across synonyms
        fused_heatmap = np.mean(heatmaps, axis=0)
        
        # Apply smoothing if requested
        if smoothing_sigma > 0:
            from scipy.ndimage import gaussian_filter
            fused_heatmap = gaussian_filter(fused_heatmap, sigma=smoothing_sigma)
        
        # Optionally normalize to [0, 1]
        if normalize and fused_heatmap.max() > 0:
            fused_heatmap = fused_heatmap / fused_heatmap.max()
        
        return fused_heatmap
    
    def compute_all_class_heatmaps(self, image: Image.Image, normalize: bool = True, 
                                   smoothing_sigma: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Compute heatmaps for all target classes.
        """
        class_heatmaps = {}
        
        print(f"Computing heatmaps for all classes (smoothing={smoothing_sigma})...")
        for class_name in TARGET_CLASSES.keys():
            # print(f"  {class_name}...", end=" ")
            heatmap = self.compute_class_heatmap_with_fusion(
                image, class_name, normalize=normalize, smoothing_sigma=smoothing_sigma
            )
            class_heatmaps[class_name] = heatmap
            # print(f"max={heatmap.max():.3f}")
        
        return class_heatmaps
    
    def compute_background_heatmaps(self, image: Image.Image, normalize: bool = True,
                                    smoothing_sigma: float = 0.0) -> List[np.ndarray]:
        """
        Compute heatmaps for all background classes.
        """
        bg_heatmaps = []
        
        print(f"Computing background heatmaps (smoothing={smoothing_sigma})...")
        for bg_class in BACKGROUND_CLASSES:
            contribs, _, _ = get_bcos_heatmap(
                self.bcos_model, image, bg_class, self.clip_model,
                self.device, return_raw=True
            )
            # Take positive contributions only
            heatmap = np.maximum(contribs, 0)
            
            # Apply smoothing
            if smoothing_sigma > 0:
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=smoothing_sigma)
            
            # Optionally normalize
            if normalize and heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            bg_heatmaps.append(heatmap)
        
        return bg_heatmaps
    
    def apply_global_softmax(
        self, 
        class_heatmaps: Dict[str, np.ndarray],
        bg_heatmaps: List[np.ndarray],
        scaling_factor: float = 100000.0
    ) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Apply global softmax competition across all classes.
        """
        # Stack all heatmaps: (num_classes, H, W)
        all_heatmaps = []
        class_names = list(class_heatmaps.keys())
        num_fg_classes = len(class_names)
        
        for class_name in class_names:
            all_heatmaps.append(class_heatmaps[class_name])
        
        for bg_heatmap in bg_heatmaps:
            all_heatmaps.append(bg_heatmap)
        
        # Stack: (C, H, W)
        stacked = np.stack(all_heatmaps, axis=0)
        
        # Scale heatmaps
        stacked_scaled = stacked * scaling_factor
        
        print(f"Softmax Scaling: min={stacked.min():.2e}, max={stacked.max():.2e} -> scaled_max={stacked_scaled.max():.2f}")
        
        # Apply softmax
        stacked_torch = torch.from_numpy(stacked_scaled).float()
        prob_maps = torch.softmax(stacked_torch, dim=0).numpy()
        
        print(f"Softmax Output: min={prob_maps.min():.4f}, max={prob_maps.max():.4f}")
        
        # Extract probability maps
        prob_class_maps = {}
        for idx, class_name in enumerate(class_names):
            prob_class_maps[class_name] = prob_maps[idx]
        
        prob_bg_maps = []
        for idx in range(num_fg_classes, len(prob_maps)):
            prob_bg_maps.append(prob_maps[idx])
            
        # Cleanup
        del stacked, stacked_scaled, stacked_torch, all_heatmaps
        import gc
        gc.collect()
        
        return prob_class_maps, prob_bg_maps
    
    def iterative_detection(
        self,
        image: Image.Image,
        prob_maps: Dict[str, np.ndarray],
        bg_prob_maps: List[np.ndarray],
        class_heatmaps: Dict[str, np.ndarray] = None, # Added raw heatmaps
        max_detections: int = 20,
        masking_mode: str = 'pixel_all',  # 'box', 'pixel_class', 'pixel_all'
        stopping_criterion: str = 'local_competition', # 'threshold', 'background_competition', 'local_competition'
        threshold: float = 0.15,
        bg_tolerance: float = 0.8,  # FG wins if FG >= BG * tolerance
        min_score: float = 0.03     # Minimum score to accept a detection
    ) -> List[Dict]:
        """
        Iteratively detect objects.
        Uses prob_maps (Softmax) for selection and class_heatmaps (Raw) for segmentation if provided.
        """
        detections = []
        import cv2
        import gc
        
        # Work on copies
        working_prob_maps = {k: v.copy() for k, v in prob_maps.items()}
        working_bg_maps = [bg.copy() for bg in bg_prob_maps]
        
        # If raw heatmaps provided, use them for segmentation
        use_raw_for_seg = class_heatmaps is not None
        if use_raw_for_seg:
            print("Using RAW heatmaps for segmentation/boxing.")
        
        print(f"Starting detection (mode={masking_mode}, stop={stopping_criterion}, tol={bg_tolerance}, min={min_score})...")
        
        for iteration in range(max_detections):
            # 1. Find Global Max Foreground (using Softmax maps)
            max_fg_score = -1.0
            max_fg_class = None
            
            # Debug: Track top 3 candidates
            candidates = []
            for class_name, prob_map in working_prob_maps.items():
                current_max = prob_map.max()
                candidates.append((class_name, current_max))
                if current_max > max_fg_score:
                    max_fg_score = current_max
                    max_fg_class = class_name
            
            # Sort and print top 3
            candidates.sort(key=lambda x: x[1], reverse=True)
            print(f"\n--- Iteration {iteration + 1} Analysis ---")
            print("Top 3 Foreground Candidates:")
            for c_name, c_score in candidates[:3]:
                print(f"  {c_name:<15}: {c_score:.4f}")
            
            # Safety check for very low scores (noise floor)
            if max_fg_score < min_score:
                print(f"STOP DECISION: Signal too weak ({max_fg_score:.4f} < {min_score})")
                break

            # 2. Check Stopping Criteria
            if stopping_criterion == 'background_competition':
                # Global comparison (Legacy)
                max_bg_score = 0.0
                max_bg_class = None
                for i, bg_map in enumerate(working_bg_maps):
                    m = bg_map.max()
                    if m > max_bg_score:
                        max_bg_score = m
                        max_bg_class = f"bg_{i}"
                
                print(f"Max Background Score: {max_bg_score:.4f} ({max_bg_class})")
                
                threshold_to_beat = max_bg_score * bg_tolerance
                if max_fg_score < threshold_to_beat:
                    print(f"STOP DECISION: Background wins globally (BG={max_bg_score:.4f} * {bg_tolerance} > FG={max_fg_score:.4f})")
                    break
                else:
                    print(f"CONTINUE: Foreground wins globally (FG={max_fg_score:.4f} >= BG_adj={threshold_to_beat:.4f})")

            elif stopping_criterion == 'local_competition':
                # Local comparison: Is FG > BG at the specific pixel?
                # Find location of max_fg_score
                heatmap = working_prob_maps[max_fg_class]
                y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)
                
                # Get max background score AT THIS PIXEL
                local_bg_score = 0.0
                for bg_map in working_bg_maps:
                    val = bg_map[y_max, x_max]
                    if val > local_bg_score:
                        local_bg_score = val
                
                print(f"Local Competition at ({x_max}, {y_max}): FG={max_fg_score:.4f} vs BG={local_bg_score:.4f}")
                
                # Apply tolerance
                if max_fg_score < local_bg_score * bg_tolerance:
                    print(f"REJECTED: Background wins locally (BG={local_bg_score:.4f} > FG={max_fg_score:.4f})")
                    # Suppress this peak locally so we can find others
                    # We can mask a small radius or just the pixel? 
                    # Better to mask the object if we can segment it, but we haven't segmented it yet.
                    # Let's segment it first to see what we are rejecting.
                    # Actually, if it's background, we should probably just zero out this class at this location?
                    # But if we don't segment, we might loop forever on this pixel.
                    # So let's segment it (Otsu) and then reject it.
                    pass # We will handle rejection after segmentation
                else:
                    print(f"CONTINUE: Foreground wins locally")
            
            elif stopping_criterion == 'threshold':
                if max_fg_score < threshold:
                    print(f"STOP DECISION: Score below threshold ({max_fg_score:.3f} < {threshold})")
                    break
            
            print(f"Selected: {max_fg_class} (score={max_fg_score:.3f})")
            
            # 3. Extract Box & Mask
            # Use RAW heatmap for segmentation if available, otherwise Softmax map
            if use_raw_for_seg:
                segmentation_map = class_heatmaps[max_fg_class]
                print("  Using RAW heatmap for segmentation.")
            else:
                segmentation_map = working_prob_maps[max_fg_class]
                print("  Using SOFTMAX heatmap for segmentation.")
            
            # Debug: Heatmap stats
            print(f"Heatmap Stats: min={segmentation_map.min():.4f}, max={segmentation_map.max():.4f}, mean={segmentation_map.mean():.4f}, std={segmentation_map.std():.4f}")
            
            # Otsu thresholding to get the binary mask of the object
            # Note: If using raw map, we need to be careful about scaling. 
            # B-cos raw values are small, so *255 might not be enough if max is 1e-5.
            # We should normalize the segmentation map to 0-1 first for Otsu.
            
            seg_min, seg_max = segmentation_map.min(), segmentation_map.max()
            if seg_max > seg_min:
                seg_norm = (segmentation_map - seg_min) / (seg_max - seg_min)
            else:
                seg_norm = segmentation_map # Flat
            
            heatmap_uint8 = (seg_norm * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = seg_norm > (otsu_thresh / 255.0)

            # --- LOCAL COMPETITION CHECK (Post-Segmentation) ---
            if stopping_criterion == 'local_competition':
                # Check if the object as a whole is dominated by background?
                # Or just check the peak (which we did above)?
                # Let's enforce the peak check here.
                heatmap = working_prob_maps[max_fg_class] # Use Softmax map for comparison
                y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)
                local_bg_score = 0.0
                for bg_map in working_bg_maps:
                    val = bg_map[y_max, x_max]
                    if val > local_bg_score:
                        local_bg_score = val
                
                if max_fg_score < local_bg_score * bg_tolerance:
                     print(f"REJECTED: Background wins locally at peak. Masking out.")
                     working_prob_maps[max_fg_class][binary_mask] = 0
                     continue
            
            # If mask is empty or covers too much of the image (e.g. > 40%) AND score is low, skip/stop
            # This prevents detecting the entire background noise floor as an object
            mask_coverage = np.sum(binary_mask) / binary_mask.size
            print(f"Mask Coverage: {mask_coverage:.1%}")
            
            # Refined check: Only reject large objects if they have low confidence
            # A real large object should have high confidence (e.g. > 0.3)
            if mask_coverage > 0.40 and max_fg_score < 0.30:
                print(f"REJECTED: Detection covers {mask_coverage:.1%} of image with low score ({max_fg_score:.3f}). Ignoring as noise.")
                # Mask this class to prevent infinite loop, but don't add detection
                working_prob_maps[max_fg_class][:] = 0 # Mask everywhere to stop selection
                continue
                
            if np.sum(binary_mask) == 0:
                print("REJECTED: Empty mask. Skipping.")
                working_prob_maps[max_fg_class][:] = 0
                continue

            # Extract box from the binary mask
            # We can use a helper or just cv2 boundingRect
            # Let's use cv2 for speed/simplicity here since we have the mask
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                working_prob_maps[max_fg_class][:] = 0
                continue
                
            # Find largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            box = [x, y, x+w, y+h]
            
            # Scale box
            h_ratio = image.size[1] / segmentation_map.shape[0]
            w_ratio = image.size[0] / segmentation_map.shape[1]
            box_scaled = [
                int(box[0] * w_ratio),
                int(box[1] * h_ratio),
                int(box[2] * w_ratio),
                int(box[3] * h_ratio)
            ]
            
            # Add detection
            detections.append({
                'class_name': max_fg_class,
                'score': float(max_fg_score),
                'box': box_scaled,
                'heatmap': segmentation_map.copy() # Return the map used for segmentation
            })
            
            # 4. Masking (The "Claiming" Step)
            if masking_mode == 'pixel_all':
                # Mask detected pixels in ALL maps (FG and BG)
                # Mask in Softmax maps
                for k in working_prob_maps:
                    working_prob_maps[k][binary_mask] = 0
                for bg_map in working_bg_maps:
                    bg_map[binary_mask] = 0
                
                # Also mask in RAW maps if we are using them, to prevent re-use?
                # If we have multiple instances, we need to mask the raw map too!
                if use_raw_for_seg:
                    class_heatmaps[max_fg_class][binary_mask] = 0
                    
            elif masking_mode == 'pixel_class':
                # Mask only in this class
                working_prob_maps[max_fg_class][binary_mask] = 0
                if use_raw_for_seg:
                    class_heatmaps[max_fg_class][binary_mask] = 0
            
            elif masking_mode == 'box':
                # Mask box in this class
                working_prob_maps[max_fg_class][box[1]:box[3], box[0]:box[2]] = 0
                if use_raw_for_seg:
                     class_heatmaps[max_fg_class][box[1]:box[3], box[0]:box[2]] = 0

            # Memory cleanup
            del heatmap_uint8, binary_mask, contours
            gc.collect()
            
        # Final cleanup
        del working_prob_maps, working_bg_maps
        gc.collect()
            
        return detections
