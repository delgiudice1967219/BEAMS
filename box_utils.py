import cv2
import numpy as np

def extract_box(heatmap, method='otsu', threshold_val=None):
    """
    Extracts a bounding box from a heatmap.
    
    Args:
        heatmap: 2D numpy array (float).
        method: 'otsu' or 'fixed'.
        threshold_val: Float value for fixed threshold (if method='fixed').
                       If None and method='fixed', defaults to 0.2 * max.
                       
    Returns:
        bbox: [xmin, ymin, xmax, ymax]
    """
    # 1. Normalize heatmap to [0, 255]
    # Ensure heatmap is non-negative for normalization if needed, or just min-max
    heatmap_norm = heatmap - heatmap.min()
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    
    # 2. Binarize
    if method == 'otsu':
        thresh_val, binary_map = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'fixed':
        if threshold_val is None:
            # Default to 20% of max (which is 255 in uint8)
            thresh = 0.2 * 255
        else:
            thresh = threshold_val * 255
        _, binary_map = cv2.threshold(heatmap_uint8, thresh, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 3. Contours: Find the Largest Connected Component (LCC)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No contours found
        return [0, 0, 0, 0]
        
    # Find largest contour by area
    c = max(contours, key=cv2.contourArea)
    
    # 4. Return bounding box [xmin, ymin, xmax, ymax]
    x, y, w, h = cv2.boundingRect(c)
    return [x, y, x + w, y + h]

if __name__ == "__main__":
    # Test
    print("Testing extract_box...")
    dummy_map = np.zeros((100, 100), dtype=np.float32)
    dummy_map[30:70, 30:70] = 1.0 # Square in middle
    # Add some noise
    dummy_map += np.random.rand(100, 100) * 0.1
    
    box = extract_box(dummy_map, method='otsu')
    print(f"Box (Otsu): {box}")
    # Expected: around [30, 30, 70, 70]
    
    box_fixed = extract_box(dummy_map, method='fixed', threshold_val=0.5)
    print(f"Box (Fixed): {box_fixed}")
