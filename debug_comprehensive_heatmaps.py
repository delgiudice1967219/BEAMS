"""
Comprehensive debug script to examine heatmap values in detail.
Compares:
1. Individual synonym heatmaps vs averaged heatmaps for each category
2. Category heatmaps vs background heatmaps
"""

import sys
sys.path.insert(0, "bcosification")

import numpy as np
from PIL import Image
from bcos_localization import load_bcos_model, load_clip_for_text, get_bcos_heatmap
from config_classes import TARGET_CLASSES, BACKGROUND_CLASSES

def analyze_heatmap(heatmap, name):
    """Print detailed stats for a heatmap."""
    positive = np.maximum(heatmap, 0)
    return {
        'name': name,
        'min': positive.min(),
        'max': positive.max(),
        'mean': positive.mean(),
        'std': positive.std(),
        'non_zero': np.count_nonzero(positive),
        'total': positive.size
    }

def main():
    print("Loading models...")
    bcos_model, bcos_device = load_bcos_model()
    clip_model, _ = load_clip_for_text()
    
    print("\nLoading test image: bread-knife-pans-towel.png")
    image = Image.open("test_images/bread-knife-pans-towel.png").convert('RGB')
    print(f"Image size: {image.size}")
    
    print("\n" + "="*80)
    print("PART 1: ANALYZING TARGET CLASSES")
    print("="*80)
    
    # Test a few representative classes
    test_classes = ["bread", "knife", "towel", "cooking_pan", "car", "person"]
    
    category_stats = {}
    
    for class_name in test_classes:
        if class_name not in TARGET_CLASSES:
            continue
            
        print(f"\n{'─'*80}")
        print(f"Class: {class_name}")
        print(f"{'─'*80}")
        
        synonyms = TARGET_CLASSES[class_name]
        print(f"Synonyms: {synonyms}")
        
        # Compute heatmap for each synonym
        synonym_heatmaps = []
        synonym_stats = []
        
        for synonym in synonyms:
            contribs, _, score = get_bcos_heatmap(
                bcos_model, image, synonym, clip_model, bcos_device, return_raw=True
            )
            positive = np.maximum(contribs, 0)
            synonym_heatmaps.append(positive)
            
            stats = analyze_heatmap(contribs, synonym)
            stats['score'] = score
            synonym_stats.append(stats)
            
            print(f"  {synonym:20s} | max={stats['max']:.8f} | mean={stats['mean']:.8f} | score={score:.4f}")
        
        # Compute averaged heatmap
        averaged = np.mean(synonym_heatmaps, axis=0)
        avg_stats = analyze_heatmap(averaged, f"{class_name}_averaged")
        
        print(f"\n  {'AVERAGED':20s} | max={avg_stats['max']:.8f} | mean={avg_stats['mean']:.8f}")
        
        # Compare: max of individual vs averaged
        max_individual = max(s['max'] for s in synonym_stats)
        print(f"\n  Max individual: {max_individual:.8f}")
        print(f"  Averaged max:   {avg_stats['max']:.8f}")
        print(f"  Ratio (avg/max_indiv): {avg_stats['max']/max_individual if max_individual > 0 else 0:.4f}")
        
        category_stats[class_name] = {
            'synonym_stats': synonym_stats,
            'averaged': avg_stats,
            'max_individual': max_individual
        }
    
    print("\n" + "="*80)
    print("PART 2: ANALYZING BACKGROUND CLASSES")
    print("="*80)
    
    # Test a few background classes
    test_backgrounds = ["background", "grass", "sky", "wall", "floor", "noise"]
    background_stats = []
    
    for bg_class in test_backgrounds:
        if bg_class not in BACKGROUND_CLASSES:
            continue
            
        contribs, _, score = get_bcos_heatmap(
            bcos_model, image, bg_class, clip_model, bcos_device, return_raw=True
        )
        
        stats = analyze_heatmap(contribs, bg_class)
        stats['score'] = score
        background_stats.append(stats)
        
        print(f"  {bg_class:20s} | max={stats['max']:.8f} | mean={stats['mean']:.8f} | score={score:.4f}")
    
    print("\n" + "="*80)
    print("PART 3: COMPARISON - CATEGORIES vs BACKGROUNDS")
    print("="*80)
    
    print("\nCategory max values (averaged):")
    for class_name, stats in category_stats.items():
        print(f"  {class_name:20s}: {stats['averaged']['max']:.8f}")
    
    print("\nBackground max values:")
    for stats in background_stats:
        print(f"  {stats['name']:20s}: {stats['max']:.8f}")
    
    # Overall comparison
    avg_category_max = np.mean([stats['averaged']['max'] for stats in category_stats.values()])
    avg_background_max = np.mean([stats['max'] for stats in background_stats])
    
    print(f"\nAverage max across categories:  {avg_category_max:.8f}")
    print(f"Average max across backgrounds: {avg_background_max:.8f}")
    print(f"Ratio (category/background):    {avg_category_max/avg_background_max if avg_background_max > 0 else 0:.4f}")
    
    print("\n" + "="*80)
    print("PART 4: WHAT HAPPENS WITH SOFTMAX?")
    print("="*80)
    
    # Simulate what happens in global softmax
    all_max_values = []
    all_max_values.extend([stats['averaged']['max'] for stats in category_stats.values()])
    all_max_values.extend([stats['max'] for stats in background_stats])
    
    print(f"\nAll max values (categories + backgrounds):")
    print(f"  Min: {min(all_max_values):.8f}")
    print(f"  Max: {max(all_max_values):.8f}")
    print(f"  Range: {max(all_max_values) - min(all_max_values):.8f}")
    print(f"  Std: {np.std(all_max_values):.8f}")
    
    # Simulate softmax on these max values
    import torch
    max_tensor = torch.tensor(all_max_values, dtype=torch.float32)
    softmax_probs = torch.softmax(max_tensor, dim=0).numpy()
    
    print(f"\nAfter softmax on max values:")
    print(f"  Min prob: {softmax_probs.min():.6f}")
    print(f"  Max prob: {softmax_probs.max():.6f}")
    print(f"  Mean prob: {softmax_probs.mean():.6f}")
    print(f"  Std prob: {softmax_probs.std():.6f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if max(all_max_values) - min(all_max_values) < 1e-6:
        print("⚠️  All heatmap max values are nearly identical!")
        print("    This causes softmax to produce uniform probabilities.")
        print("    Solution: Scale heatmaps before softmax to amplify differences.")
    else:
        print("✓  Heatmap values have sufficient variation.")
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main()
