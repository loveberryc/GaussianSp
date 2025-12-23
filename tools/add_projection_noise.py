#!/usr/bin/env python3
"""
Add projection measurement noise to 4D CT datasets.

This script simulates realistic CT acquisition noise by adding:
1. Poisson noise: Models photon counting statistics (quantum noise)
2. Gaussian noise: Models electronic/readout noise

Usage:
    # Add Poisson noise (photon count scale = 1e4)
    python tools/add_projection_noise.py \
        --input data/dir_4d_case1.pickle \
        --noise_type poisson \
        --photon_scale 1e4

    # Add Gaussian noise (std = 0.05)
    python tools/add_projection_noise.py \
        --input data/dir_4d_case1.pickle \
        --noise_type gaussian \
        --gaussian_std 0.05

    # Add mixed noise (both Poisson + Gaussian)
    python tools/add_projection_noise.py \
        --input data/dir_4d_case1.pickle \
        --noise_type mixed \
        --photon_scale 1e4 \
        --gaussian_std 0.02
"""

import os
import sys
import pickle
import argparse
import numpy as np
from copy import deepcopy

sys.path.append("./")


def add_poisson_noise(projections, photon_scale=1e4, seed=42):
    """
    Add Poisson noise to simulate photon counting statistics.
    
    In CT, the measured intensity follows Beer-Lambert law:
        I = I_0 * exp(-integral(mu))
    
    The number of detected photons follows Poisson distribution.
    We simulate this by:
        1. Convert projections to intensity: I = exp(-proj)
        2. Scale to photon counts: N = I * photon_scale
        3. Add Poisson noise: N_noisy ~ Poisson(N)
        4. Convert back: proj_noisy = -log(N_noisy / photon_scale)
    
    Args:
        projections: Original projections (sinogram values, -log transform)
        photon_scale: Expected photon count at zero attenuation (I_0)
                     Higher = less noise, Lower = more noise
        seed: Random seed
    
    Returns:
        Noisy projections
    """
    np.random.seed(seed)
    
    # Convert to intensity (Beer-Lambert)
    intensity = np.exp(-projections)
    
    # Scale to photon counts
    photon_counts = intensity * photon_scale
    
    # Add Poisson noise
    noisy_counts = np.random.poisson(photon_counts.astype(np.float64))
    
    # Avoid log(0) by clipping
    noisy_counts = np.maximum(noisy_counts, 1)
    
    # Convert back to projections
    noisy_projections = -np.log(noisy_counts / photon_scale)
    
    return noisy_projections.astype(projections.dtype)


def add_gaussian_noise(projections, std=0.05, seed=42):
    """
    Add Gaussian noise to simulate electronic/readout noise.
    
    Args:
        projections: Original projections
        std: Standard deviation of Gaussian noise (relative to projection range)
        seed: Random seed
    
    Returns:
        Noisy projections
    """
    np.random.seed(seed)
    
    # Scale std relative to projection range
    proj_range = projections.max() - projections.min()
    actual_std = std * proj_range
    
    noise = np.random.normal(0, actual_std, size=projections.shape)
    noisy_projections = projections + noise
    
    return noisy_projections.astype(projections.dtype)


def add_mixed_noise(projections, photon_scale=1e4, gaussian_std=0.02, seed=42):
    """
    Add both Poisson and Gaussian noise (more realistic model).
    
    Order: Poisson first (physics), then Gaussian (electronics).
    
    Args:
        projections: Original projections
        photon_scale: Photon count scale for Poisson noise
        gaussian_std: Standard deviation for Gaussian noise
        seed: Random seed
    
    Returns:
        Noisy projections
    """
    # Apply Poisson noise first
    noisy = add_poisson_noise(projections, photon_scale, seed)
    
    # Then apply Gaussian noise (use different seed offset)
    noisy = add_gaussian_noise(noisy, gaussian_std, seed + 1000)
    
    return noisy


def create_noisy_dataset(input_path, output_dir, noise_type, 
                         photon_scale=1e4, gaussian_std=0.05, seed=42):
    """Create dataset with projection noise."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    data = deepcopy(data)
    
    print(f"[Projection Noise: {noise_type}]")
    
    # Get original projections
    train_proj = data['train']['projections']
    val_proj = data['val']['projections']
    
    print(f"  - Train projections shape: {train_proj.shape}")
    print(f"  - Val projections shape: {val_proj.shape}")
    print(f"  - Original projection range: [{train_proj.min():.4f}, {train_proj.max():.4f}]")
    
    # Add noise based on type
    if noise_type == 'poisson':
        print(f"  - Photon scale: {photon_scale:.0e}")
        train_noisy = add_poisson_noise(train_proj, photon_scale, seed)
        val_noisy = add_poisson_noise(val_proj, photon_scale, seed + 500)
        suffix = f"poisson{int(np.log10(photon_scale))}"
        
    elif noise_type == 'gaussian':
        print(f"  - Gaussian std: {gaussian_std}")
        train_noisy = add_gaussian_noise(train_proj, gaussian_std, seed)
        val_noisy = add_gaussian_noise(val_proj, gaussian_std, seed + 500)
        suffix = f"gauss{int(gaussian_std*100):02d}"
        
    elif noise_type == 'mixed':
        print(f"  - Photon scale: {photon_scale:.0e}")
        print(f"  - Gaussian std: {gaussian_std}")
        train_noisy = add_mixed_noise(train_proj, photon_scale, gaussian_std, seed)
        val_noisy = add_mixed_noise(val_proj, photon_scale, gaussian_std, seed + 500)
        suffix = f"mixed_p{int(np.log10(photon_scale))}_g{int(gaussian_std*100):02d}"
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Update data
    data['train']['projections'] = train_noisy
    data['val']['projections'] = val_noisy
    
    # Add noise info to metadata
    data['noise'] = {
        'type': noise_type,
        'photon_scale': photon_scale if noise_type in ['poisson', 'mixed'] else None,
        'gaussian_std': gaussian_std if noise_type in ['gaussian', 'mixed'] else None,
        'seed': seed
    }
    
    # Statistics
    print(f"  - Noisy projection range: [{train_noisy.min():.4f}, {train_noisy.max():.4f}]")
    
    # Compute SNR estimate
    noise_level = np.std(train_noisy - train_proj)
    signal_level = np.std(train_proj)
    snr = signal_level / (noise_level + 1e-8)
    print(f"  - Estimated SNR: {snr:.2f}")
    print(f"  - RMS noise: {noise_level:.4f}")
    
    # Save
    basename = os.path.basename(input_path).replace('.pickle', '')
    output_name = f"{basename}_{suffix}.pickle"
    output_path = os.path.join(output_dir, output_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved noisy dataset to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add projection measurement noise")
    parser.add_argument("--input", type=str, required=True,
                        help="Input pickle file path")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory")
    parser.add_argument("--noise_type", type=str, default="poisson",
                        choices=["poisson", "gaussian", "mixed"],
                        help="Type of noise to add")
    parser.add_argument("--photon_scale", type=float, default=1e4,
                        help="Photon count scale for Poisson noise (1e4 = moderate, 1e3 = heavy)")
    parser.add_argument("--gaussian_std", type=float, default=0.05,
                        help="Standard deviation for Gaussian noise (relative to projection range)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Adding Projection Measurement Noise")
    print("="*60)
    
    output_path = create_noisy_dataset(
        args.input, args.output_dir, args.noise_type,
        args.photon_scale, args.gaussian_std, args.seed
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
