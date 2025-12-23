#!/usr/bin/env python3
"""
Create limited-angle CT datasets.

This script simulates limited-angle CT acquisition where only a subset
of angular coverage is available (e.g., 120° instead of 360°).

This is more challenging than sparse-view because:
- Sparse-view: fewer projections but full angular coverage
- Limited-angle: missing entire angular sectors → severe artifacts

Usage:
    # Keep only 120° angular range (starting from 0°)
    python tools/create_limited_angle.py \
        --input data/dir_4d_case1.pickle \
        --angle_range 120

    # Keep 180° range starting from 45°
    python tools/create_limited_angle.py \
        --input data/dir_4d_case1.pickle \
        --angle_range 180 \
        --start_angle 45

    # Keep two 90° sectors (0-90° and 180-270°) - dual limited
    python tools/create_limited_angle.py \
        --input data/dir_4d_case1.pickle \
        --mode dual \
        --angle_range 90
"""

import os
import sys
import pickle
import argparse
import numpy as np
from copy import deepcopy

sys.path.append("./")


def filter_by_angle_range(data, start_angle, end_angle, split='train'):
    """
    Filter projections to keep only those within the specified angle range.
    
    Args:
        data: Dataset dict
        start_angle: Start angle in degrees
        end_angle: End angle in degrees
        split: 'train' or 'val'
    
    Returns:
        Indices of projections within the angle range
    """
    angles_rad = data[split]['angles']
    angles_deg = np.degrees(angles_rad) % 360  # Normalize to [0, 360)
    
    # Handle wraparound (e.g., 350° to 10°)
    if start_angle <= end_angle:
        mask = (angles_deg >= start_angle) & (angles_deg < end_angle)
    else:
        # Wraparound case
        mask = (angles_deg >= start_angle) | (angles_deg < end_angle)
    
    return np.where(mask)[0]


def create_limited_angle_dataset(input_path, output_dir, angle_range, 
                                  start_angle=0, mode='single', seed=42):
    """
    Create a limited-angle CT dataset.
    
    Args:
        input_path: Original pickle file
        output_dir: Output directory
        angle_range: Angular range to keep in degrees
        start_angle: Starting angle in degrees
        mode: 'single' - one continuous sector
              'dual' - two opposing sectors (more realistic for some setups)
        seed: Random seed
    
    Returns:
        Path to created dataset
    """
    np.random.seed(seed)
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    data = deepcopy(data)
    
    print(f"[Limited-Angle CT]")
    print(f"  - Mode: {mode}")
    print(f"  - Angle range: {angle_range}°")
    print(f"  - Start angle: {start_angle}°")
    
    if mode == 'single':
        # Single continuous sector
        end_angle = (start_angle + angle_range) % 360
        
        print(f"  - Keeping angles: [{start_angle}°, {end_angle}°)")
        
        train_indices = filter_by_angle_range(data, start_angle, end_angle, 'train')
        val_indices = filter_by_angle_range(data, start_angle, end_angle, 'val')
        
    elif mode == 'dual':
        # Two opposing sectors (e.g., 0-90° and 180-270°)
        sector1_start = start_angle
        sector1_end = (start_angle + angle_range) % 360
        sector2_start = (start_angle + 180) % 360
        sector2_end = (sector2_start + angle_range) % 360
        
        print(f"  - Sector 1: [{sector1_start}°, {sector1_end}°)")
        print(f"  - Sector 2: [{sector2_start}°, {sector2_end}°)")
        
        # Get indices for both sectors
        train_idx1 = filter_by_angle_range(data, sector1_start, sector1_end, 'train')
        train_idx2 = filter_by_angle_range(data, sector2_start, sector2_end, 'train')
        train_indices = np.sort(np.concatenate([train_idx1, train_idx2]))
        
        val_idx1 = filter_by_angle_range(data, sector1_start, sector1_end, 'val')
        val_idx2 = filter_by_angle_range(data, sector2_start, sector2_end, 'val')
        val_indices = np.sort(np.concatenate([val_idx1, val_idx2]))
        
        # Effective angle range is 2x
        angle_range = angle_range * 2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Original counts
    n_train_orig = len(data['train']['angles'])
    n_val_orig = len(data['val']['angles'])
    
    # Apply filtering to train
    data['train']['angles'] = data['train']['angles'][train_indices]
    data['train']['projections'] = data['train']['projections'][train_indices]
    data['train']['time'] = data['train']['time'][train_indices]
    data['train']['phase'] = data['train']['phase'][train_indices]
    data['numTrain'] = len(train_indices)
    
    # Apply filtering to val
    data['val']['angles'] = data['val']['angles'][val_indices]
    data['val']['projections'] = data['val']['projections'][val_indices]
    data['val']['time'] = data['val']['time'][val_indices]
    data['val']['phase'] = data['val']['phase'][val_indices]
    data['numVal'] = len(val_indices)
    
    # Add metadata
    data['limited_angle'] = {
        'mode': mode,
        'angle_range': angle_range,
        'start_angle': start_angle,
        'seed': seed
    }
    
    # Statistics
    print(f"\n  - Train: {n_train_orig} → {len(train_indices)} projections ({100*len(train_indices)/n_train_orig:.1f}%)")
    print(f"  - Val: {n_val_orig} → {len(val_indices)} projections ({100*len(val_indices)/n_val_orig:.1f}%)")
    
    # Phase distribution
    train_phases = np.unique(data['train']['phase'], return_counts=True)
    print(f"  - Phase distribution: {dict(zip(train_phases[0], train_phases[1]))}")
    
    # Actual angle coverage
    train_angles_deg = np.degrees(data['train']['angles'])
    print(f"  - Actual angle coverage: [{train_angles_deg.min():.1f}°, {train_angles_deg.max():.1f}°]")
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    if mode == 'single':
        output_name = f"{basename}_limited{angle_range}deg.pickle"
    else:
        output_name = f"{basename}_limited{angle_range}deg_dual.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved limited-angle dataset to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create limited-angle CT datasets")
    parser.add_argument("--input", type=str, required=True,
                        help="Input pickle file path")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory")
    parser.add_argument("--angle_range", type=float, required=True,
                        help="Angular range to keep in degrees (e.g., 120 for 120°)")
    parser.add_argument("--start_angle", type=float, default=0,
                        help="Starting angle in degrees")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "dual"],
                        help="single: one sector, dual: two opposing sectors")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Limited-Angle CT Dataset")
    print("="*60)
    
    output_path = create_limited_angle_dataset(
        args.input, args.output_dir, args.angle_range,
        args.start_angle, args.mode, args.seed
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
