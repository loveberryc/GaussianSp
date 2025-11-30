#!/usr/bin/env python
"""
S5: 4D Dynamic-Aware Multi-Phase FDK Point Cloud Initialization

This script implements the s5 initialization strategy described in idea_s5_4d_dynamic_init.md.

Key steps:
1. Load training projections and timestamps from pickle file
2. Divide projections into P phases based on normalized time
3. Run TIGRE FDK on each phase to get V_phase[k]
4. Compute V_ref (reference phase), V_avg (temporal mean), V_var (temporal variance)
5. Construct 4D-aware importance sampling weights S_total combining structure and dynamics
6. Sample N Gaussian centers from S_total
7. Set initial scales based on V_var (dynamic regions get smaller scales)
8. Set initial densities from V_ref
9. Save as init_s5_4d_caseX.npy compatible with existing training pipeline

Usage:
    python tools/build_init_s5_4d.py \
        --input data/dir_4d_case1.pickle \
        --output data/init_s5_4d_case1.npy \
        --s5_num_phases 3 \
        --s5_num_points 50000 \
        --s5_static_weight 0.7 \
        --s5_dynamic_weight 0.3 \
        --s5_density_exponent 1.5 \
        --s5_var_exponent 1.0
"""

import os
import sys
import argparse
import pickle
import numpy as np
from typing import Tuple, List, Dict

# Add TIGRE to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'TIGRE-2.3', 'Python'))

import tigre
from tigre.utilities.geometry import Geometry
import tigre.algorithms as algs


def create_geometry_from_pickle(data: Dict) -> Geometry:
    """
    Create TIGRE geometry from pickle data.
    
    Args:
        data: Dictionary loaded from pickle file
        
    Returns:
        TIGRE Geometry object
    """
    geo = Geometry()
    
    # Detector geometry
    geo.nDetector = np.array(data['nDetector'])
    geo.dDetector = np.array(data['dDetector'])
    geo.sDetector = geo.nDetector * geo.dDetector
    
    # Volume geometry
    geo.nVoxel = np.array(data['nVoxel'])
    geo.dVoxel = np.array(data['dVoxel'])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    
    # Distances
    geo.DSD = data['DSD']
    geo.DSO = data['DSO']
    
    # Offsets
    geo.offOrigin = np.array(data['offOrigin'])
    geo.offDetector = np.array(data['offDetector'])
    
    # Mode
    geo.mode = data['mode']
    
    # Accuracy for ray tracing
    geo.accuracy = data.get('accuracy', 0.5)
    
    return geo


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize volume to [0, 1] range using 99.9 percentile.
    
    Args:
        volume: Input volume
        
    Returns:
        Normalized volume in [0, 1]
    """
    volume = np.clip(volume, 0, None)
    if volume.max() > 0:
        p999 = np.percentile(volume, 99.9)
        if p999 > 0:
            volume = volume / p999
            volume = np.clip(volume, 0, 1)
    return volume


def multi_phase_fdk_reconstruction(
    projections: np.ndarray,
    angles: np.ndarray,
    times: np.ndarray,
    geo: Geometry,
    num_phases: int,
    ref_phase_index: int = None,
    fdk_filter: str = 'ram_lak'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Perform multi-phase FDK reconstruction.
    
    Args:
        projections: All projections [N, H, W]
        angles: Projection angles [N]
        times: Acquisition timestamps [N]
        geo: TIGRE geometry
        num_phases: Number of phases P
        ref_phase_index: Index of reference phase (default: P // 2)
        fdk_filter: FDK filter type
        
    Returns:
        V_ref: Reference phase volume
        V_avg: Temporal average volume
        V_var: Temporal variance volume
        V_phases: List of phase volumes
    """
    print("=" * 60)
    print("MULTI-PHASE FDK RECONSTRUCTION")
    print("=" * 60)
    
    # Normalize times to [0, 1]
    t_min, t_max = times.min(), times.max()
    if t_max > t_min:
        t_norm = (times - t_min) / (t_max - t_min)
    else:
        # All same time, assign uniformly
        t_norm = np.linspace(0, 1, len(times))
    
    print(f"[1] Time normalization:")
    print(f"    Original time range: [{t_min:.4f}, {t_max:.4f}]")
    print(f"    Normalized time range: [{t_norm.min():.4f}, {t_norm.max():.4f}]")
    
    # Default reference phase
    if ref_phase_index is None:
        ref_phase_index = num_phases // 2
    
    print(f"[2] Phase division: {num_phases} phases, reference phase: {ref_phase_index}")
    
    # Divide projections into phases
    V_phases = []
    phase_counts = []
    
    for k in range(num_phases):
        phase_start = k / num_phases
        phase_end = (k + 1) / num_phases
        
        # Handle last phase to include t_norm == 1.0
        if k == num_phases - 1:
            phase_mask = (t_norm >= phase_start) & (t_norm <= phase_end)
        else:
            phase_mask = (t_norm >= phase_start) & (t_norm < phase_end)
        
        phase_projs = projections[phase_mask]
        phase_angles = angles[phase_mask]
        phase_count = phase_mask.sum()
        phase_counts.append(phase_count)
        
        print(f"    Phase {k}: [{phase_start:.2f}, {phase_end:.2f}), "
              f"{phase_count} projections")
        
        if phase_count < 5:
            print(f"    WARNING: Phase {k} has very few projections ({phase_count}). "
                  f"Consider reducing num_phases.")
        
        if phase_count > 0:
            # Check if angles are in radians
            if np.max(phase_angles) < 2 * np.pi + 0.1:
                angles_rad = phase_angles.astype(np.float32)
            else:
                angles_rad = np.deg2rad(phase_angles).astype(np.float32)
            
            # FDK reconstruction for this phase
            try:
                V_phase = algs.fdk(
                    phase_projs.astype(np.float32),
                    geo,
                    angles_rad,
                    filter=fdk_filter
                )
                V_phase = normalize_volume(V_phase)
            except Exception as e:
                print(f"    FDK failed for phase {k}: {e}")
                V_phase = np.zeros(geo.nVoxel, dtype=np.float32)
        else:
            V_phase = np.zeros(geo.nVoxel, dtype=np.float32)
        
        V_phases.append(V_phase)
    
    print(f"[3] Phase reconstruction complete. Volume shape: {V_phases[0].shape}")
    
    # Stack all phases
    V_stack = np.stack(V_phases, axis=0)  # [P, D, H, W]
    
    # Compute V_ref, V_avg, V_var
    V_ref = V_phases[ref_phase_index]
    V_avg = np.mean(V_stack, axis=0)
    V_var = np.var(V_stack, axis=0)
    
    print(f"[4] Computed temporal statistics:")
    print(f"    V_ref (phase {ref_phase_index}): [{V_ref.min():.4f}, {V_ref.max():.4f}]")
    print(f"    V_avg: [{V_avg.min():.4f}, {V_avg.max():.4f}], mean={V_avg.mean():.4f}")
    print(f"    V_var: [{V_var.min():.6f}, {V_var.max():.6f}], mean={V_var.mean():.6f}")
    
    return V_ref, V_avg, V_var, V_phases


def compute_4d_aware_importance(
    V_ref: np.ndarray,
    V_var: np.ndarray,
    static_weight: float = 0.7,
    dynamic_weight: float = 0.3,
    density_exponent: float = 1.5,
    var_exponent: float = 1.0,
    density_thresh: float = 0.05
) -> np.ndarray:
    """
    Compute 4D-aware importance sampling weights.
    
    Args:
        V_ref: Reference phase volume (structure/density)
        V_var: Temporal variance volume (dynamics)
        static_weight: Weight for static component (density)
        dynamic_weight: Weight for dynamic component (variance)
        density_exponent: Exponent for density term
        var_exponent: Exponent for variance term
        density_thresh: Minimum density threshold
        
    Returns:
        S_total: Importance sampling weights [D, H, W]
    """
    print("=" * 60)
    print("COMPUTING 4D-AWARE IMPORTANCE WEIGHTS")
    print("=" * 60)
    
    # Normalize V_ref and V_var to [0, 1]
    V_ref_n = V_ref.copy()
    if V_ref_n.max() > 0:
        V_ref_n = V_ref_n / V_ref_n.max()
    
    V_var_n = V_var.copy()
    if V_var_n.max() > 0:
        V_var_n = V_var_n / V_var_n.max()
    
    print(f"[1] Normalized ranges:")
    print(f"    V_ref_n: [{V_ref_n.min():.4f}, {V_ref_n.max():.4f}]")
    print(f"    V_var_n: [{V_var_n.min():.4f}, {V_var_n.max():.4f}]")
    
    # Apply density threshold (only consider voxels above threshold)
    density_mask = V_ref >= density_thresh
    num_valid = density_mask.sum()
    print(f"[2] Density threshold {density_thresh}: {num_valid} valid voxels "
          f"({100 * num_valid / V_ref.size:.1f}%)")
    
    # Compute static component: S_static = V_ref_n^p
    S_static = np.power(V_ref_n, density_exponent)
    
    # Compute dynamic component: S_dyn = V_var_n^q
    S_dyn = np.power(V_var_n, var_exponent)
    
    # Combine: S_total = w_static * S_static + w_dyn * S_dyn
    S_total = static_weight * S_static + dynamic_weight * S_dyn
    
    # Apply density mask
    S_total = S_total * density_mask.astype(np.float32)
    
    # Ensure non-negative
    S_total = np.maximum(S_total, 0)
    
    print(f"[3] Importance weights computed:")
    print(f"    w_static={static_weight}, w_dynamic={dynamic_weight}")
    print(f"    density_exp={density_exponent}, var_exp={var_exponent}")
    print(f"    S_total range: [{S_total.min():.6f}, {S_total.max():.6f}]")
    print(f"    S_total mean (valid): {S_total[density_mask].mean():.6f}")
    
    # Analyze high-variance regions
    high_var_mask = V_var_n > 0.5
    high_var_count = high_var_mask.sum()
    print(f"[4] Dynamic region analysis:")
    print(f"    High variance (>0.5) voxels: {high_var_count} "
          f"({100 * high_var_count / V_ref.size:.2f}%)")
    
    return S_total


def sample_gaussians_from_importance(
    S_total: np.ndarray,
    V_ref: np.ndarray,
    V_var_n: np.ndarray,
    scanner_cfg: Dict,
    scene_scale: float,
    num_points: int = 50000,
    density_rescale: float = 0.15,
    scale_min_factor: float = 0.5,
    scale_max_factor: float = 1.5,
    base_scale: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample Gaussian centers from importance weights and set initial parameters.
    
    Args:
        S_total: Importance sampling weights [D, H, W]
        V_ref: Reference volume for density values
        V_var_n: Normalized variance for scale adjustment
        scanner_cfg: Scanner configuration with geometry info (NORMALIZED)
        scene_scale: Scene scale factor (2 / max(sVoxel_original))
        num_points: Number of Gaussians to sample
        density_rescale: Density scaling factor
        scale_min_factor: Minimum scale factor (for dynamic regions)
        scale_max_factor: Maximum scale factor (for static regions)
        base_scale: Base scale value (default: voxel size)
        
    Returns:
        positions: Gaussian centers [N, 3] in NORMALIZED coordinates [-1, 1]^3
        densities: Initial densities [N]
        scales: Initial scales [N] (for 4D-aware scale initialization)
    """
    print("=" * 60)
    print("SAMPLING GAUSSIANS WITH 4D-AWARE PARAMETERS")
    print("=" * 60)
    
    # Get geometry (already normalized by scene_scale)
    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])
    
    if base_scale is None:
        base_scale = np.mean(dVoxel)
    
    scale_min = scale_min_factor * base_scale
    scale_max = scale_max_factor * base_scale
    
    print(f"[1] Geometry (normalized, scene_scale={scene_scale:.6f}):")
    print(f"    offOrigin: {offOrigin}")
    print(f"    dVoxel: {dVoxel}")
    print(f"    sVoxel: {sVoxel}")
    print(f"[2] Scale range: [{scale_min:.6f}, {scale_max:.6f}]")
    
    # Normalize S_total to probability distribution
    S_flat = S_total.flatten()
    total_weight = S_flat.sum()
    
    if total_weight <= 0:
        raise ValueError("S_total has no positive values. Check density threshold.")
    
    prob = S_flat / total_weight
    
    # Sample indices with replacement
    print(f"[3] Sampling {num_points} Gaussians...")
    sampled_flat_indices = np.random.choice(
        len(S_flat),
        size=num_points,
        replace=True,
        p=prob
    )
    
    # Convert flat indices to 3D indices
    shape = S_total.shape
    sampled_indices = np.array(np.unravel_index(sampled_flat_indices, shape)).T  # [N, 3]
    
    # Add random jitter within voxel
    jitter = np.random.uniform(-0.5, 0.5, size=(num_points, 3))
    sampled_indices_jittered = sampled_indices + jitter
    
    # Convert voxel indices to world coordinates
    # Position = index * dVoxel - sVoxel/2 + offOrigin
    positions = sampled_indices_jittered * dVoxel - sVoxel / 2 + offOrigin
    
    print(f"[4] Position statistics:")
    print(f"    Shape: {positions.shape}")
    print(f"    Range: [{positions.min(axis=0)}, {positions.max(axis=0)}]")
    
    # Get densities from V_ref at sampled positions
    densities = V_ref[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2]
    ]
    densities = densities * density_rescale
    
    print(f"[5] Density statistics:")
    print(f"    Range: [{densities.min():.6f}, {densities.max():.6f}]")
    print(f"    Mean: {densities.mean():.6f}")
    
    # 4D-aware scale initialization
    # Dynamic regions (high V_var_n) get smaller scales
    # Static regions (low V_var_n) get larger scales
    var_vals = V_var_n[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2]
    ]
    
    # scale = scale_max - (scale_max - scale_min) * var_val
    scales = scale_max - (scale_max - scale_min) * var_vals
    
    print(f"[6] 4D-aware scale statistics:")
    print(f"    Variance values range: [{var_vals.min():.6f}, {var_vals.max():.6f}]")
    print(f"    Scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"    Scale mean: {scales.mean():.6f}")
    
    # Count dynamic vs static
    dynamic_count = (var_vals > 0.5).sum()
    static_count = (var_vals <= 0.5).sum()
    print(f"[7] Distribution:")
    print(f"    Dynamic (var>0.5): {dynamic_count} ({100*dynamic_count/num_points:.1f}%)")
    print(f"    Static (var<=0.5): {static_count} ({100*static_count/num_points:.1f}%)")
    
    return positions.astype(np.float32), densities.astype(np.float32), scales.astype(np.float32)


def save_init_file(
    positions: np.ndarray,
    densities: np.ndarray,
    scales: np.ndarray,
    output_path: str,
    save_scales: bool = False
):
    """
    Save initialization file compatible with existing training pipeline.
    
    The standard format is [N, 4] with columns [x, y, z, density].
    Optionally save scales separately for advanced initialization.
    
    Args:
        positions: [N, 3]
        densities: [N]
        scales: [N]
        output_path: Output file path
        save_scales: Whether to save scales in extended format
    """
    print("=" * 60)
    print("SAVING INITIALIZATION FILE")
    print("=" * 60)
    
    # Standard format: [x, y, z, density]
    out = np.concatenate([positions, densities[:, None]], axis=-1)  # [N, 4]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.save(output_path, out)
    
    print(f"[1] Saved standard init file: {output_path}")
    print(f"    Shape: {out.shape}")
    print(f"    File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    if save_scales:
        # Save extended format with scales for future use
        scales_path = output_path.replace('.npy', '_scales.npy')
        np.save(scales_path, scales)
        print(f"[2] Saved scales file: {scales_path}")
        
        # Also save a combined extended format
        extended_path = output_path.replace('.npy', '_extended.npy')
        extended = np.concatenate([positions, densities[:, None], scales[:, None]], axis=-1)
        np.save(extended_path, extended)
        print(f"[3] Saved extended init file: {extended_path}")
        print(f"    Extended shape: {extended.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='S5: 4D Dynamic-Aware Multi-Phase FDK Point Cloud Initialization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python tools/build_init_s5_4d.py \\
        --input data/dir_4d_case1.pickle \\
        --output data/init_s5_4d_case1.npy \\
        --s5_num_phases 3 \\
        --s5_num_points 50000
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input pickle file (e.g., data/dir_4d_case1.pickle)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output npy file (e.g., data/init_s5_4d_case1.npy)')
    
    # S5 configuration
    parser.add_argument('--s5_num_phases', type=int, default=3,
                        help='Number of phases for multi-phase FDK (default: 3)')
    parser.add_argument('--s5_ref_phase_index', type=int, default=None,
                        help='Reference phase index (default: P//2)')
    parser.add_argument('--s5_num_points', type=int, default=50000,
                        help='Number of Gaussians to initialize (default: 50000)')
    parser.add_argument('--s5_static_weight', type=float, default=0.7,
                        help='Weight for static/density component (default: 0.7)')
    parser.add_argument('--s5_dynamic_weight', type=float, default=0.3,
                        help='Weight for dynamic/variance component (default: 0.3)')
    parser.add_argument('--s5_density_exponent', type=float, default=1.5,
                        help='Exponent for density term (default: 1.5)')
    parser.add_argument('--s5_var_exponent', type=float, default=1.0,
                        help='Exponent for variance term (default: 1.0)')
    
    # Additional parameters
    parser.add_argument('--density_thresh', type=float, default=0.05,
                        help='Minimum density threshold (default: 0.05)')
    parser.add_argument('--density_rescale', type=float, default=0.15,
                        help='Density rescaling factor (default: 0.15)')
    parser.add_argument('--scale_min_factor', type=float, default=0.5,
                        help='Minimum scale factor for dynamic regions (default: 0.5)')
    parser.add_argument('--scale_max_factor', type=float, default=1.5,
                        help='Maximum scale factor for static regions (default: 1.5)')
    parser.add_argument('--fdk_filter', type=str, default='ram_lak',
                        help='FDK filter type (default: ram_lak)')
    parser.add_argument('--save_scales', action='store_true',
                        help='Save scales in extended format')
    parser.add_argument('--save_volumes', action='store_true',
                        help='Save intermediate volumes (V_ref, V_avg, V_var)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("S5: 4D DYNAMIC-AWARE MULTI-PHASE FDK POINT CLOUD INITIALIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Num phases: {args.s5_num_phases}")
    print(f"  Num points: {args.s5_num_points}")
    print(f"  Static weight: {args.s5_static_weight}")
    print(f"  Dynamic weight: {args.s5_dynamic_weight}")
    print(f"  Density exponent: {args.s5_density_exponent}")
    print(f"  Variance exponent: {args.s5_var_exponent}")
    print("")
    
    # Load pickle file
    print("[STEP 1] Loading data...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # Extract training data
    projections = data['train']['projections'].astype(np.float32)
    angles = data['train']['angles'].astype(np.float32)
    times = data['train']['time'].astype(np.float32)
    
    print(f"  Projections: {projections.shape}")
    print(f"  Angles: {angles.shape}, range: [{angles.min():.4f}, {angles.max():.4f}]")
    print(f"  Times: {times.shape}, range: [{times.min():.4f}, {times.max():.4f}]")
    
    # Create scanner config (convert mm -> m, same as X2-Gaussian dataset_readers.py)
    # NAF scanner measurements are in mm, but X2-Gaussian uses meters
    scanner_cfg_raw = {
        'DSD': data['DSD'] / 1000,
        'DSO': data['DSO'] / 1000,
        'nVoxel': data['nVoxel'],
        'dVoxel': (np.array(data['dVoxel']) / 1000).tolist(),
        'sVoxel': (np.array(data['nVoxel']) * np.array(data['dVoxel']) / 1000).tolist(),
        'nDetector': data['nDetector'],
        'dDetector': (np.array(data['dDetector']) / 1000).tolist(),
        'sDetector': (np.array(data['nDetector']) * np.array(data['dDetector']) / 1000).tolist(),
        'offOrigin': (np.array(data['offOrigin']) / 1000).tolist(),
        'offDetector': (np.array(data['offDetector']) / 1000).tolist(),
        'accuracy': data['accuracy'],
        'mode': data['mode'],
    }
    
    # Calculate scene_scale to normalize to [-1, 1]^3 (same as X2-Gaussian)
    scene_scale = 2 / max(scanner_cfg_raw['sVoxel'])
    print(f"  Scene scale: {scene_scale:.6f} (to normalize to [-1, 1]^3)")
    
    # Apply scene_scale to get normalized scanner_cfg
    scanner_cfg = {}
    for key in ['dVoxel', 'sVoxel', 'offOrigin']:
        scanner_cfg[key] = (np.array(scanner_cfg_raw[key]) * scene_scale).tolist()
    scanner_cfg['nVoxel'] = scanner_cfg_raw['nVoxel']
    
    print(f"  Normalized sVoxel: {scanner_cfg['sVoxel']}")
    
    # Create TIGRE geometry (uses original mm units for reconstruction)
    geo = create_geometry_from_pickle(data)
    
    # Step 2: Multi-phase FDK reconstruction
    print("\n[STEP 2] Multi-phase FDK reconstruction...")
    V_ref, V_avg, V_var, V_phases = multi_phase_fdk_reconstruction(
        projections=projections,
        angles=angles,
        times=times,
        geo=geo,
        num_phases=args.s5_num_phases,
        ref_phase_index=args.s5_ref_phase_index,
        fdk_filter=args.fdk_filter
    )
    
    # Optionally save intermediate volumes
    if args.save_volumes:
        vol_dir = os.path.dirname(args.output)
        base_name = os.path.basename(args.output).replace('init_s5_4d_', '').replace('.npy', '')
        
        np.save(os.path.join(vol_dir, f's5_V_ref_{base_name}.npy'), V_ref)
        np.save(os.path.join(vol_dir, f's5_V_avg_{base_name}.npy'), V_avg)
        np.save(os.path.join(vol_dir, f's5_V_var_{base_name}.npy'), V_var)
        print(f"\nSaved intermediate volumes to {vol_dir}/")
    
    # Step 3: Compute 4D-aware importance weights
    print("\n[STEP 3] Computing 4D-aware importance weights...")
    S_total = compute_4d_aware_importance(
        V_ref=V_ref,
        V_var=V_var,
        static_weight=args.s5_static_weight,
        dynamic_weight=args.s5_dynamic_weight,
        density_exponent=args.s5_density_exponent,
        var_exponent=args.s5_var_exponent,
        density_thresh=args.density_thresh
    )
    
    # Normalize V_var for scale computation
    V_var_n = V_var.copy()
    if V_var_n.max() > 0:
        V_var_n = V_var_n / V_var_n.max()
    
    # Step 4: Sample Gaussians
    print("\n[STEP 4] Sampling Gaussians with 4D-aware parameters...")
    positions, densities, scales = sample_gaussians_from_importance(
        S_total=S_total,
        V_ref=V_ref,
        V_var_n=V_var_n,
        scanner_cfg=scanner_cfg,
        scene_scale=scene_scale,
        num_points=args.s5_num_points,
        density_rescale=args.density_rescale,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor
    )
    
    # Step 5: Save initialization file
    print("\n[STEP 5] Saving initialization file...")
    save_init_file(
        positions=positions,
        densities=densities,
        scales=scales,
        output_path=args.output,
        save_scales=args.save_scales
    )
    
    print("\n" + "=" * 70)
    print("S5 INITIALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {args.output}")
    print(f"\nTo use this initialization in training:")
    print(f"  python train.py -s {args.input} --ply_path {args.output} ...")
    print("")


if __name__ == '__main__':
    main()
