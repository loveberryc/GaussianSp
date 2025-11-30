"""
Generate average CT volume using FDK reconstruction from training projections.

This script:
1. Loads training projections from pickle file
2. Uses TIGRE FDK algorithm to reconstruct a static 3D volume
3. Saves the result as avg_ct.npy for use with s4 temporal-ensemble training

Usage:
    python generate_avg_ct.py --input data/dir_4d_case1.pickle --output data/avg_ct_case1.npy
"""

import os
import sys
import argparse
import pickle
import numpy as np

# Add TIGRE to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TIGRE-2.3', 'Python'))

import tigre
from tigre.utilities.geometry import Geometry
import tigre.algorithms as algs


def create_geometry_from_pickle(data):
    """
    Create TIGRE geometry from pickle data.
    
    Args:
        data: Dictionary loaded from pickle file
        
    Returns:
        TIGRE Geometry object
    """
    geo = Geometry()
    
    # Detector geometry
    geo.nDetector = np.array(data['nDetector'])  # [512, 512]
    geo.dDetector = np.array(data['dDetector'])  # [1.0, 1.0] mm
    geo.sDetector = geo.nDetector * geo.dDetector  # Total detector size
    
    # Volume geometry
    geo.nVoxel = np.array(data['nVoxel'])  # [256, 256, 94]
    geo.dVoxel = np.array(data['dVoxel'])  # [0.97, 0.97, 2.5] mm
    geo.sVoxel = geo.nVoxel * geo.dVoxel  # Total volume size
    
    # Distances
    geo.DSD = data['DSD']  # Source to detector distance
    geo.DSO = data['DSO']  # Source to origin distance
    
    # Offsets
    geo.offOrigin = np.array(data['offOrigin'])  # Volume offset
    geo.offDetector = np.array(data['offDetector'])  # Detector offset
    
    # Mode
    geo.mode = data['mode']  # 'cone' or 'parallel'
    
    # Accuracy for ray tracing
    geo.accuracy = data.get('accuracy', 0.5)
    
    print("TIGRE Geometry created:")
    print(f"  nDetector: {geo.nDetector}")
    print(f"  dDetector: {geo.dDetector}")
    print(f"  sDetector: {geo.sDetector}")
    print(f"  nVoxel: {geo.nVoxel}")
    print(f"  dVoxel: {geo.dVoxel}")
    print(f"  sVoxel: {geo.sVoxel}")
    print(f"  DSD: {geo.DSD}")
    print(f"  DSO: {geo.DSO}")
    print(f"  mode: {geo.mode}")
    
    return geo


def main():
    parser = argparse.ArgumentParser(description='Generate average CT using FDK reconstruction')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input pickle file (e.g., data/dir_4d_case1.pickle)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output npy file (e.g., data/avg_ct_case1.npy)')
    parser.add_argument('--filter', type=str, default='ram_lak',
                        help='FDK filter type (default: ram_lak)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Average CT using FDK Reconstruction")
    print("=" * 60)
    
    # Load pickle file
    print(f"\n[1] Loading data from: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # Extract projections and angles from training set
    projections = data['train']['projections']  # [N, H, W]
    angles = data['train']['angles']  # [N]
    
    print(f"  Projections shape: {projections.shape}")
    print(f"  Angles shape: {angles.shape}")
    print(f"  Angle range: [{np.min(angles):.4f}, {np.max(angles):.4f}]")
    
    # Check if angles are in radians or degrees
    # If max angle < 2*pi, likely already in radians
    if np.max(angles) < 2 * np.pi + 0.1:
        print("  Angles appear to be in radians")
        angles_rad = angles.astype(np.float32)
    else:
        print("  Angles appear to be in degrees, converting to radians")
        angles_rad = np.deg2rad(angles).astype(np.float32)
    
    # Prepare projections for TIGRE
    # TIGRE expects [N, nDetector[1], nDetector[0]] = [N, rows, cols]
    # Our projections are [N, 512, 512] which matches
    projections = projections.astype(np.float32)
    
    # TIGRE expects projections in a specific format
    # Need to ensure correct orientation
    # projections shape: [N, detector_rows, detector_cols]
    print(f"  Projections dtype: {projections.dtype}")
    print(f"  Projections value range: [{projections.min():.4f}, {projections.max():.4f}]")
    
    # Create geometry
    print(f"\n[2] Creating TIGRE geometry")
    geo = create_geometry_from_pickle(data)
    
    # Perform FDK reconstruction
    print(f"\n[3] Performing FDK reconstruction with filter: {args.filter}")
    print("  This may take a few minutes...")
    
    try:
        # FDK reconstruction
        # Note: TIGRE FDK expects projections as [angles, detector_v, detector_u]
        volume = algs.fdk(projections, geo, angles_rad, filter=args.filter)
        
        print(f"  Reconstruction complete!")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Volume value range: [{volume.min():.4f}, {volume.max():.4f}]")
        print(f"  Volume mean: {volume.mean():.4f}")
        
        # Clip negative values (attenuation should be non-negative)
        volume = np.clip(volume, 0, None)
        print(f"  After clipping: [{volume.min():.6f}, {volume.max():.6f}]")
        
        # Normalize to [0, 1] range (matching GT volume range)
        if volume.max() > 0:
            # Use 99.9 percentile to avoid outlier influence
            p999 = np.percentile(volume, 99.9)
            if p999 > 0:
                volume = volume / p999
                volume = np.clip(volume, 0, 1)
                print(f"  After normalization to [0, 1]: [{volume.min():.6f}, {volume.max():.6f}]")
                print(f"  Volume mean: {volume.mean():.6f}")
        
    except Exception as e:
        print(f"  FDK failed: {e}")
        print("  Trying alternative approach with SART...")
        
        # Use SART as fallback (slower but more robust)
        volume = algs.ossart(projections, geo, angles_rad, niter=20)
        volume = np.clip(volume, 0, None)
        print(f"  SART reconstruction complete!")
        print(f"  Volume shape: {volume.shape}")
    
    # Save output
    print(f"\n[4] Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.save(args.output, volume.astype(np.float32))
    
    # Verify saved file
    loaded = np.load(args.output)
    print(f"  Saved successfully!")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    print(f"  Loaded shape: {loaded.shape}")
    
    print("\n" + "=" * 60)
    print("FDK reconstruction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
