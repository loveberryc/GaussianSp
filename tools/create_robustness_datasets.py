#!/usr/bin/env python3
"""
Create robustness test datasets from original 4D CT data.

Two directions:
1. Phase Perturbation: Simulate irregular breathing by perturbing time/phase
2. Sparse Views: Reduce the number of projection views

Usage:
    python tools/create_robustness_datasets.py \
        --input data/dir_4d_case1.pickle \
        --output_dir data/ \
        --phase_noise 0.1 \
        --view_ratio 0.5
"""

import os
import sys
import pickle
import argparse
import numpy as np
from copy import deepcopy

sys.path.append("./")


def perturb_phase(data, noise_level=0.1, seed=42):
    """
    Perturb the breathing phase to simulate irregular breathing.
    
    Strategy:
    1. Add Gaussian noise to 'time' values
    2. Optionally shuffle phase assignments within nearby time windows
    
    Args:
        data: Original pickle data
        noise_level: Standard deviation of Gaussian noise (relative to phase duration)
                    0.1 = 10% of one phase duration
        seed: Random seed for reproducibility
    
    Returns:
        Perturbed data
    """
    np.random.seed(seed)
    data = deepcopy(data)
    
    # Get number of phases from data
    num_phases = len(np.unique(data['train']['phase']))
    phase_duration = 1.0 / num_phases  # Duration of one phase
    
    # Noise magnitude in time units
    noise_std = noise_level * phase_duration
    
    print(f"[Phase Perturbation]")
    print(f"  - Number of phases: {num_phases}")
    print(f"  - Phase duration: {phase_duration:.4f}")
    print(f"  - Noise level: {noise_level} ({noise_std:.4f} in time units)")
    
    # Perturb train data
    train_time = data['train']['time'].copy()
    train_noise = np.random.normal(0, noise_std, size=train_time.shape)
    train_time_perturbed = train_time + train_noise
    
    # Clip to valid range [0, 1)
    train_time_perturbed = np.clip(train_time_perturbed, 0, 1 - 1e-6)
    
    # Update phase based on perturbed time
    train_phase_perturbed = (train_time_perturbed * num_phases).astype(np.int64)
    
    data['train']['time'] = train_time_perturbed
    data['train']['phase'] = train_phase_perturbed
    
    # Perturb val data
    val_time = data['val']['time'].copy()
    val_noise = np.random.normal(0, noise_std, size=val_time.shape)
    val_time_perturbed = val_time + val_noise
    val_time_perturbed = np.clip(val_time_perturbed, 0, 1 - 1e-6)
    val_phase_perturbed = (val_time_perturbed * num_phases).astype(np.int64)
    
    data['val']['time'] = val_time_perturbed
    data['val']['phase'] = val_phase_perturbed
    
    # Statistics
    original_phases = np.unique(deepcopy(data)['train']['phase'], return_counts=True)
    new_phases = np.unique(data['train']['phase'], return_counts=True)
    
    time_diff = np.abs(train_time - train_time_perturbed)
    print(f"  - Train time perturbation: mean={time_diff.mean():.4f}, max={time_diff.max():.4f}")
    
    phase_changed = np.sum(train_phase_perturbed != (train_time * num_phases).astype(np.int64))
    print(f"  - Train projections with changed phase: {phase_changed}/{len(train_time)} ({100*phase_changed/len(train_time):.1f}%)")
    
    return data


def reduce_views(data, view_ratio=0.5, strategy='uniform', seed=42):
    """
    Reduce the number of projection views to simulate sparse-view CT.
    
    Args:
        data: Original pickle data
        view_ratio: Ratio of views to keep (0.5 = keep 50%)
        strategy: 'uniform' - uniformly sample views
                  'random' - randomly sample views
                  'phase_balanced' - ensure each phase has similar count
        seed: Random seed
    
    Returns:
        Data with reduced views
    """
    np.random.seed(seed)
    data = deepcopy(data)
    
    n_train = len(data['train']['angles'])
    n_keep = int(n_train * view_ratio)
    
    print(f"[Sparse Views]")
    print(f"  - Original train views: {n_train}")
    print(f"  - Target views: {n_keep} ({view_ratio*100:.0f}%)")
    print(f"  - Strategy: {strategy}")
    
    if strategy == 'uniform':
        # Uniformly spaced indices
        indices = np.linspace(0, n_train - 1, n_keep, dtype=int)
    elif strategy == 'random':
        # Random sampling
        indices = np.sort(np.random.choice(n_train, n_keep, replace=False))
    elif strategy == 'phase_balanced':
        # Balance across phases
        phases = data['train']['phase']
        unique_phases = np.unique(phases)
        n_per_phase = n_keep // len(unique_phases)
        
        indices = []
        for p in unique_phases:
            phase_indices = np.where(phases == p)[0]
            n_sample = min(n_per_phase, len(phase_indices))
            sampled = np.random.choice(phase_indices, n_sample, replace=False)
            indices.extend(sampled)
        indices = np.sort(np.array(indices))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Apply selection
    data['train']['angles'] = data['train']['angles'][indices]
    data['train']['projections'] = data['train']['projections'][indices]
    data['train']['time'] = data['train']['time'][indices]
    data['train']['phase'] = data['train']['phase'][indices]
    data['numTrain'] = len(indices)
    
    # Statistics
    new_phases = np.unique(data['train']['phase'], return_counts=True)
    print(f"  - Actual kept views: {len(indices)}")
    print(f"  - Phase distribution: {dict(zip(new_phases[0], new_phases[1]))}")
    
    # Check angle coverage
    angles_deg = np.degrees(data['train']['angles'])
    print(f"  - Angle coverage: [{angles_deg.min():.1f}°, {angles_deg.max():.1f}°]")
    
    return data


def create_perturbed_dataset(input_path, output_dir, noise_level, seed=42):
    """Create phase-perturbed dataset."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Perturb
    data_perturbed = perturb_phase(data, noise_level=noise_level, seed=seed)
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    output_name = f"{basename}_noise{noise_level}.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_perturbed, f)
    
    print(f"\nSaved perturbed dataset to: {output_path}")
    return output_path


def create_sparse_dataset(input_path, output_dir, view_ratio, strategy='uniform', seed=42):
    """Create sparse-view dataset."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Reduce views
    data_sparse = reduce_views(data, view_ratio=view_ratio, strategy=strategy, seed=seed)
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    output_name = f"{basename}_sparse{int(view_ratio*100)}.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_sparse, f)
    
    print(f"\nSaved sparse dataset to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create robustness test datasets")
    parser.add_argument("--input", type=str, required=True,
                        help="Input pickle file path")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory")
    parser.add_argument("--phase_noise", type=float, default=None,
                        help="Phase noise level (0.1 = 10%% of phase duration)")
    parser.add_argument("--view_ratio", type=float, default=None,
                        help="Ratio of views to keep (0.5 = 50%%)")
    parser.add_argument("--view_strategy", type=str, default="uniform",
                        choices=["uniform", "random", "phase_balanced"],
                        help="Strategy for view reduction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--generate_init", action="store_true",
                        help="Generate init .npy file after creating dataset")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    created_files = []
    
    # Create phase-perturbed dataset
    if args.phase_noise is not None:
        print("\n" + "="*60)
        print("Creating Phase-Perturbed Dataset")
        print("="*60)
        output_path = create_perturbed_dataset(
            args.input, args.output_dir, args.phase_noise, args.seed
        )
        created_files.append(output_path)
    
    # Create sparse-view dataset
    if args.view_ratio is not None:
        print("\n" + "="*60)
        print("Creating Sparse-View Dataset")
        print("="*60)
        output_path = create_sparse_dataset(
            args.input, args.output_dir, args.view_ratio, args.view_strategy, args.seed
        )
        created_files.append(output_path)
    
    # Generate init files
    if args.generate_init and created_files:
        print("\n" + "="*60)
        print("Generating Initialization Files")
        print("="*60)
        for fpath in created_files:
            basename = os.path.basename(fpath).replace('.pickle', '')
            init_path = os.path.join(args.output_dir, f"init_{basename}.npy")
            cmd = f"python initialize_pcd.py --data {fpath} --output {init_path}"
            print(f"Running: {cmd}")
            os.system(cmd)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created {len(created_files)} dataset(s):")
    for f in created_files:
        print(f"  - {f}")
    
    if args.generate_init:
        print("\nTo train on these datasets:")
        for f in created_files:
            basename = os.path.basename(f).replace('.pickle', '')
            print(f"  python train.py -s {f} ...")


if __name__ == "__main__":
    main()
