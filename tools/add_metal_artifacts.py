#!/usr/bin/env python3
"""
Add metal artifacts and stripe artifacts to 4D CT datasets.

This script simulates:
1. Metal artifacts: High-attenuation regions causing streaking/beam hardening
2. Stripe artifacts: Detector malfunction causing vertical stripes in sinogram
3. Ring artifacts: Bad detector pixels causing ring patterns in reconstruction

Usage:
    # Add stripe artifacts (5% of detector rows corrupted)
    python tools/add_metal_artifacts.py \
        --input data/dir_4d_case1.pickle \
        --artifact_type stripe \
        --stripe_ratio 0.05

    # Add metal-like high attenuation
    python tools/add_metal_artifacts.py \
        --input data/dir_4d_case1.pickle \
        --artifact_type metal \
        --metal_intensity 2.0

    # Add ring artifact pattern
    python tools/add_metal_artifacts.py \
        --input data/dir_4d_case1.pickle \
        --artifact_type ring \
        --ring_count 3
"""

import os
import sys
import pickle
import argparse
import numpy as np
from copy import deepcopy

sys.path.append("./")


def add_stripe_artifacts(projections, stripe_ratio=0.05, stripe_intensity=0.3, seed=42):
    """
    Add stripe artifacts to simulate detector malfunction.
    
    Stripes appear as consistent high/low values across all projections
    for certain detector rows (columns in sinogram).
    
    Args:
        projections: Shape (N, H, W) - N projections, H detector rows, W detector cols
        stripe_ratio: Fraction of detector rows affected
        stripe_intensity: Intensity of stripe (added/subtracted value)
        seed: Random seed
    
    Returns:
        Projections with stripe artifacts
    """
    np.random.seed(seed)
    
    noisy = projections.copy()
    
    if len(projections.shape) == 3:
        n_proj, h, w = projections.shape
        n_stripes = max(1, int(w * stripe_ratio))
        
        # Random stripe positions
        stripe_cols = np.random.choice(w, n_stripes, replace=False)
        
        for col in stripe_cols:
            # Random stripe intensity (positive or negative)
            intensity = stripe_intensity * (2 * np.random.random() - 1)
            noisy[:, :, col] += intensity
            
    elif len(projections.shape) == 2:
        # Single projection or sinogram
        h, w = projections.shape
        n_stripes = max(1, int(w * stripe_ratio))
        stripe_cols = np.random.choice(w, n_stripes, replace=False)
        
        for col in stripe_cols:
            intensity = stripe_intensity * (2 * np.random.random() - 1)
            noisy[:, col] += intensity
    
    return noisy.astype(projections.dtype)


def add_dead_pixels(projections, dead_ratio=0.02, seed=42):
    """
    Add dead pixel artifacts (pixels stuck at 0 or max value).
    
    Args:
        projections: Original projections
        dead_ratio: Fraction of pixels affected
        seed: Random seed
    
    Returns:
        Projections with dead pixel artifacts
    """
    np.random.seed(seed)
    
    noisy = projections.copy()
    
    # Create dead pixel mask (same pixels dead across all projections)
    if len(projections.shape) == 3:
        n_proj, h, w = projections.shape
        n_dead = int(h * w * dead_ratio)
        
        dead_rows = np.random.randint(0, h, n_dead)
        dead_cols = np.random.randint(0, w, n_dead)
        
        # Half stuck at 0, half stuck at max
        for i, (r, c) in enumerate(zip(dead_rows, dead_cols)):
            if i % 2 == 0:
                noisy[:, r, c] = 0
            else:
                noisy[:, r, c] = projections.max()
    else:
        h, w = projections.shape
        n_dead = int(h * w * dead_ratio)
        
        dead_rows = np.random.randint(0, h, n_dead)
        dead_cols = np.random.randint(0, w, n_dead)
        
        for i, (r, c) in enumerate(zip(dead_rows, dead_cols)):
            if i % 2 == 0:
                noisy[r, c] = 0
            else:
                noisy[r, c] = projections.max()
    
    return noisy.astype(projections.dtype)


def add_metal_simulation(projections, metal_intensity=2.0, metal_width=0.05, seed=42):
    """
    Simulate metal objects causing high-attenuation regions.
    
    Metal objects cause:
    - Very high attenuation values in certain regions
    - Beam hardening artifacts (increased values in shadow regions)
    
    Args:
        projections: Original projections
        metal_intensity: Multiplier for attenuation in metal region
        metal_width: Width of metal region (fraction of detector width)
        seed: Random seed
    
    Returns:
        Projections with metal artifact simulation
    """
    np.random.seed(seed)
    
    noisy = projections.copy()
    
    if len(projections.shape) == 3:
        n_proj, h, w = projections.shape
        
        # Metal region width
        metal_w = int(w * metal_width)
        
        # Random starting position for metal
        metal_start = np.random.randint(0, w - metal_w)
        
        # Add high attenuation in metal region
        # The position shifts slightly with angle to simulate rotation
        for i in range(n_proj):
            # Sinusoidal shift to simulate rotating object
            shift = int(metal_w * 0.5 * np.sin(2 * np.pi * i / n_proj))
            start = (metal_start + shift) % w
            end = min(start + metal_w, w)
            
            # Increase attenuation (projection values)
            noisy[i, :, start:end] *= metal_intensity
            
            # Add some beam hardening effect (increased values near metal)
            if start > 0:
                noisy[i, :, max(0, start-2):start] *= 1.2
            if end < w:
                noisy[i, :, end:min(w, end+2)] *= 1.2
    
    return noisy.astype(projections.dtype)


def add_ring_pattern(projections, ring_count=3, ring_intensity=0.2, seed=42):
    """
    Add ring artifact pattern (simulates bad detector elements).
    
    Ring artifacts appear as concentric rings in reconstruction,
    caused by consistently miscalibrated detector pixels.
    
    Args:
        projections: Original projections
        ring_count: Number of ring artifacts
        ring_intensity: Intensity of ring artifacts
        seed: Random seed
    
    Returns:
        Projections with ring artifact pattern
    """
    np.random.seed(seed)
    
    noisy = projections.copy()
    
    if len(projections.shape) == 3:
        n_proj, h, w = projections.shape
        
        # Ring positions (detector columns that are miscalibrated)
        ring_cols = np.random.choice(w, ring_count, replace=False)
        
        # Each ring has consistent offset across all projections
        for col in ring_cols:
            offset = ring_intensity * (2 * np.random.random() - 1)
            # Add sinusoidal variation to make it more realistic
            for i in range(n_proj):
                variation = 0.3 * ring_intensity * np.sin(4 * np.pi * i / n_proj)
                noisy[i, :, col] += offset + variation
    
    return noisy.astype(projections.dtype)


def create_artifact_dataset(input_path, output_dir, artifact_type,
                            stripe_ratio=0.05, stripe_intensity=0.3,
                            dead_ratio=0.02,
                            metal_intensity=2.0, metal_width=0.05,
                            ring_count=3, ring_intensity=0.2,
                            seed=42):
    """Create dataset with specified artifacts."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    data = deepcopy(data)
    
    print(f"[Artifact Type: {artifact_type}]")
    
    train_proj = data['train']['projections']
    val_proj = data['val']['projections']
    
    print(f"  - Train projections shape: {train_proj.shape}")
    print(f"  - Original range: [{train_proj.min():.4f}, {train_proj.max():.4f}]")
    
    if artifact_type == 'stripe':
        print(f"  - Stripe ratio: {stripe_ratio}")
        print(f"  - Stripe intensity: {stripe_intensity}")
        train_noisy = add_stripe_artifacts(train_proj, stripe_ratio, stripe_intensity, seed)
        val_noisy = add_stripe_artifacts(val_proj, stripe_ratio, stripe_intensity, seed + 500)
        suffix = f"stripe{int(stripe_ratio*100):02d}"
        
    elif artifact_type == 'dead':
        print(f"  - Dead pixel ratio: {dead_ratio}")
        train_noisy = add_dead_pixels(train_proj, dead_ratio, seed)
        val_noisy = add_dead_pixels(val_proj, dead_ratio, seed + 500)
        suffix = f"dead{int(dead_ratio*100):02d}"
        
    elif artifact_type == 'metal':
        print(f"  - Metal intensity: {metal_intensity}")
        print(f"  - Metal width: {metal_width}")
        train_noisy = add_metal_simulation(train_proj, metal_intensity, metal_width, seed)
        val_noisy = add_metal_simulation(val_proj, metal_intensity, metal_width, seed + 500)
        suffix = f"metal{int(metal_intensity*10):02d}"
        
    elif artifact_type == 'ring':
        print(f"  - Ring count: {ring_count}")
        print(f"  - Ring intensity: {ring_intensity}")
        train_noisy = add_ring_pattern(train_proj, ring_count, ring_intensity, seed)
        val_noisy = add_ring_pattern(val_proj, ring_count, ring_intensity, seed + 500)
        suffix = f"ring{ring_count}"
        
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    data['train']['projections'] = train_noisy
    data['val']['projections'] = val_noisy
    
    # Add metadata
    data['artifacts'] = {
        'type': artifact_type,
        'params': {
            'stripe_ratio': stripe_ratio,
            'stripe_intensity': stripe_intensity,
            'dead_ratio': dead_ratio,
            'metal_intensity': metal_intensity,
            'metal_width': metal_width,
            'ring_count': ring_count,
            'ring_intensity': ring_intensity,
        },
        'seed': seed
    }
    
    print(f"  - New range: [{train_noisy.min():.4f}, {train_noisy.max():.4f}]")
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    output_name = f"{basename}_{suffix}.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved artifact dataset to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add metal/stripe/ring artifacts")
    parser.add_argument("--input", type=str, required=True,
                        help="Input pickle file path")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory")
    parser.add_argument("--artifact_type", type=str, required=True,
                        choices=["stripe", "dead", "metal", "ring"],
                        help="Type of artifact to add")
    parser.add_argument("--stripe_ratio", type=float, default=0.05,
                        help="Fraction of detector columns with stripe artifacts")
    parser.add_argument("--stripe_intensity", type=float, default=0.3,
                        help="Intensity of stripe artifacts")
    parser.add_argument("--dead_ratio", type=float, default=0.02,
                        help="Fraction of dead pixels")
    parser.add_argument("--metal_intensity", type=float, default=2.0,
                        help="Metal attenuation multiplier")
    parser.add_argument("--metal_width", type=float, default=0.05,
                        help="Metal region width (fraction of detector)")
    parser.add_argument("--ring_count", type=int, default=3,
                        help="Number of ring artifacts")
    parser.add_argument("--ring_intensity", type=float, default=0.2,
                        help="Ring artifact intensity")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Adding Artifacts to CT Dataset")
    print("="*60)
    
    output_path = create_artifact_dataset(
        args.input, args.output_dir, args.artifact_type,
        args.stripe_ratio, args.stripe_intensity,
        args.dead_ratio,
        args.metal_intensity, args.metal_width,
        args.ring_count, args.ring_intensity,
        args.seed
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
