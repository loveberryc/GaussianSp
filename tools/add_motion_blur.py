#!/usr/bin/env python3
"""
Add motion blur to 4D CT datasets.

This script simulates motion blur that occurs when:
1. Exposure time spans multiple breathing phases
2. Patient movement during acquisition
3. Gantry rotation during exposure

Motion blur is modeled by mixing projections from adjacent phases.

Usage:
    # Add phase mixing blur (20% mix with adjacent phases)
    python tools/add_motion_blur.py \
        --input data/dir_4d_case1.pickle \
        --blur_type phase_mix \
        --mix_ratio 0.2

    # Add temporal averaging blur (average 3 adjacent projections)
    python tools/add_motion_blur.py \
        --input data/dir_4d_case1.pickle \
        --blur_type temporal_avg \
        --window_size 3

    # Add exposure blur (simulate long exposure spanning phases)
    python tools/add_motion_blur.py \
        --input data/dir_4d_case1.pickle \
        --blur_type exposure \
        --exposure_phases 0.3
"""

import os
import sys
import pickle
import argparse
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

sys.path.append("./")


def add_phase_mix_blur(data, mix_ratio=0.2, seed=42):
    """
    Add motion blur by mixing projections with adjacent phases.
    
    This simulates the case where exposure spans multiple breathing phases,
    resulting in a blurred projection that is a mixture of different phases.
    
    Args:
        data: Dataset dict (will be modified in place)
        mix_ratio: Fraction of adjacent phase contribution (0-0.5)
                   0 = no mixing, 0.5 = equal weight with neighbors
        seed: Random seed
    
    Returns:
        Modified data
    """
    np.random.seed(seed)
    
    projections = data['train']['projections'].copy()
    phases = data['train']['phase']
    times = data['train']['time']
    
    n_proj = len(projections)
    n_phases = len(np.unique(phases))
    
    print(f"  - Number of phases: {n_phases}")
    print(f"  - Mix ratio: {mix_ratio}")
    
    # Sort projections by time for proper neighbor finding
    time_order = np.argsort(times)
    
    blurred = projections.copy()
    
    for i in range(n_proj):
        # Find temporal neighbors
        t_idx = np.where(time_order == i)[0][0]
        
        # Weight: center gets (1 - 2*mix_ratio), neighbors get mix_ratio each
        center_weight = 1.0 - 2 * mix_ratio
        
        weighted_sum = center_weight * projections[i]
        
        # Add contribution from previous projection (if exists and different phase)
        if t_idx > 0:
            prev_i = time_order[t_idx - 1]
            weighted_sum += mix_ratio * projections[prev_i]
        else:
            weighted_sum += mix_ratio * projections[i]  # Self if no previous
        
        # Add contribution from next projection (if exists and different phase)
        if t_idx < n_proj - 1:
            next_i = time_order[t_idx + 1]
            weighted_sum += mix_ratio * projections[next_i]
        else:
            weighted_sum += mix_ratio * projections[i]  # Self if no next
        
        blurred[i] = weighted_sum
    
    data['train']['projections'] = blurred.astype(projections.dtype)
    
    # Also apply to validation
    val_proj = data['val']['projections'].copy()
    val_times = data['val']['time']
    n_val = len(val_proj)
    
    val_time_order = np.argsort(val_times)
    val_blurred = val_proj.copy()
    
    for i in range(n_val):
        t_idx = np.where(val_time_order == i)[0][0]
        center_weight = 1.0 - 2 * mix_ratio
        weighted_sum = center_weight * val_proj[i]
        
        if t_idx > 0:
            prev_i = val_time_order[t_idx - 1]
            weighted_sum += mix_ratio * val_proj[prev_i]
        else:
            weighted_sum += mix_ratio * val_proj[i]
        
        if t_idx < n_val - 1:
            next_i = val_time_order[t_idx + 1]
            weighted_sum += mix_ratio * val_proj[next_i]
        else:
            weighted_sum += mix_ratio * val_proj[i]
        
        val_blurred[i] = weighted_sum
    
    data['val']['projections'] = val_blurred.astype(val_proj.dtype)
    
    return data


def add_temporal_avg_blur(data, window_size=3, seed=42):
    """
    Add motion blur by averaging temporally adjacent projections.
    
    This is a simple moving average in the temporal direction.
    
    Args:
        data: Dataset dict
        window_size: Number of projections to average (odd number recommended)
        seed: Random seed
    
    Returns:
        Modified data
    """
    np.random.seed(seed)
    
    projections = data['train']['projections'].copy()
    times = data['train']['time']
    
    print(f"  - Window size: {window_size}")
    
    # Sort by time
    time_order = np.argsort(times)
    sorted_proj = projections[time_order]
    
    # Apply moving average along temporal axis (axis 0)
    # Use Gaussian filter for smooth weighting
    sigma = window_size / 4.0  # Standard deviation
    blurred_sorted = gaussian_filter1d(sorted_proj.astype(np.float64), sigma=sigma, axis=0)
    
    # Unsort back to original order
    inverse_order = np.argsort(time_order)
    blurred = blurred_sorted[inverse_order]
    
    data['train']['projections'] = blurred.astype(projections.dtype)
    
    # Apply to validation
    val_proj = data['val']['projections'].copy()
    val_times = data['val']['time']
    
    val_time_order = np.argsort(val_times)
    val_sorted = val_proj[val_time_order]
    val_blurred_sorted = gaussian_filter1d(val_sorted.astype(np.float64), sigma=sigma, axis=0)
    val_inverse_order = np.argsort(val_time_order)
    val_blurred = val_blurred_sorted[val_inverse_order]
    
    data['val']['projections'] = val_blurred.astype(val_proj.dtype)
    
    return data


def add_exposure_blur(data, exposure_phases=0.3, seed=42):
    """
    Add motion blur simulating long exposure spanning multiple phases.
    
    This creates a weighted average of projections from the same angle
    but different phases, simulating what happens when exposure time
    is comparable to breathing cycle duration.
    
    Args:
        data: Dataset dict
        exposure_phases: Fraction of breathing cycle covered by exposure
                        0.1 = 10% of cycle, 0.3 = 30% of cycle
        seed: Random seed
    
    Returns:
        Modified data
    """
    np.random.seed(seed)
    
    projections = data['train']['projections'].copy()
    times = data['train']['time']
    angles = data['train']['angles']
    
    print(f"  - Exposure phases: {exposure_phases}")
    
    # For each projection, find other projections with similar angles
    # and blend based on time proximity
    
    blurred = projections.copy()
    angle_tolerance = 0.1  # radians (~6 degrees)
    
    for i in range(len(projections)):
        t_i = times[i]
        a_i = angles[i]
        
        # Find projections with similar angles
        angle_diff = np.abs(angles - a_i)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Handle wraparound
        similar_angle_mask = angle_diff < angle_tolerance
        
        if np.sum(similar_angle_mask) <= 1:
            continue
        
        # Get time differences
        time_diff = np.abs(times - t_i)
        time_diff = np.minimum(time_diff, 1.0 - time_diff)  # Handle wraparound
        
        # Weight by Gaussian kernel based on time difference
        sigma = exposure_phases / 2.0
        weights = np.exp(-0.5 * (time_diff / sigma) ** 2)
        weights[~similar_angle_mask] = 0
        weights /= weights.sum()
        
        # Weighted average
        blurred[i] = np.sum(projections * weights[:, None, None], axis=0)
    
    data['train']['projections'] = blurred.astype(projections.dtype)
    
    # Similar for validation
    val_proj = data['val']['projections'].copy()
    val_times = data['val']['time']
    val_angles = data['val']['angles']
    
    val_blurred = val_proj.copy()
    
    for i in range(len(val_proj)):
        t_i = val_times[i]
        a_i = val_angles[i]
        
        angle_diff = np.abs(val_angles - a_i)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        similar_angle_mask = angle_diff < angle_tolerance
        
        if np.sum(similar_angle_mask) <= 1:
            continue
        
        time_diff = np.abs(val_times - t_i)
        time_diff = np.minimum(time_diff, 1.0 - time_diff)
        
        sigma = exposure_phases / 2.0
        weights = np.exp(-0.5 * (time_diff / sigma) ** 2)
        weights[~similar_angle_mask] = 0
        weights /= weights.sum()
        
        val_blurred[i] = np.sum(val_proj * weights[:, None, None], axis=0)
    
    data['val']['projections'] = val_blurred.astype(val_proj.dtype)
    
    return data


def create_motion_blur_dataset(input_path, output_dir, blur_type,
                               mix_ratio=0.2, window_size=3, exposure_phases=0.3,
                               seed=42):
    """Create dataset with motion blur."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    data = deepcopy(data)
    
    print(f"[Motion Blur Type: {blur_type}]")
    
    train_proj = data['train']['projections']
    print(f"  - Train projections shape: {train_proj.shape}")
    print(f"  - Original range: [{train_proj.min():.4f}, {train_proj.max():.4f}]")
    
    if blur_type == 'phase_mix':
        data = add_phase_mix_blur(data, mix_ratio, seed)
        suffix = f"blur_mix{int(mix_ratio*100):02d}"
        
    elif blur_type == 'temporal_avg':
        data = add_temporal_avg_blur(data, window_size, seed)
        suffix = f"blur_tavg{window_size}"
        
    elif blur_type == 'exposure':
        data = add_exposure_blur(data, exposure_phases, seed)
        suffix = f"blur_exp{int(exposure_phases*100):02d}"
        
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
    
    # Add metadata
    data['motion_blur'] = {
        'type': blur_type,
        'params': {
            'mix_ratio': mix_ratio,
            'window_size': window_size,
            'exposure_phases': exposure_phases,
        },
        'seed': seed
    }
    
    noisy_proj = data['train']['projections']
    print(f"  - New range: [{noisy_proj.min():.4f}, {noisy_proj.max():.4f}]")
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    output_name = f"{basename}_{suffix}.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved motion blur dataset to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add motion blur to CT datasets")
    parser.add_argument("--input", type=str, required=True,
                        help="Input pickle file path")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory")
    parser.add_argument("--blur_type", type=str, required=True,
                        choices=["phase_mix", "temporal_avg", "exposure"],
                        help="Type of motion blur")
    parser.add_argument("--mix_ratio", type=float, default=0.2,
                        help="Mix ratio for phase_mix blur (0-0.5)")
    parser.add_argument("--window_size", type=int, default=3,
                        help="Window size for temporal_avg blur")
    parser.add_argument("--exposure_phases", type=float, default=0.3,
                        help="Exposure duration in fraction of breathing cycle")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Adding Motion Blur to CT Dataset")
    print("="*60)
    
    output_path = create_motion_blur_dataset(
        args.input, args.output_dir, args.blur_type,
        args.mix_ratio, args.window_size, args.exposure_phases,
        args.seed
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
