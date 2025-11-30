"""
Generate average projections by forward projecting the average CT volume.

This script:
1. Loads the average CT volume (from generate_avg_ct.py)
2. Uses TIGRE to forward project V_avg at each training view angle
3. Saves the result as avg_projections.npy for use with s4 temporal-ensemble training

The generated projections represent what the "average anatomy" would look like
from each view angle, serving as pseudo ground-truth for projection distillation.

Usage:
    python generate_avg_projections.py \
        --pickle data/dir_4d_case1.pickle \
        --avg_ct data/avg_ct_case1.npy \
        --output data/avg_projections_case1.npy
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
import tigre.utilities.Ax as Ax


def create_geometry_from_pickle(data):
    """
    Create TIGRE geometry from pickle data.
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
    geo.accuracy = data.get('accuracy', 0.5)
    
    return geo


def main():
    parser = argparse.ArgumentParser(description='Generate average projections from average CT')
    parser.add_argument('--pickle', '-p', type=str, required=True,
                        help='Path to input pickle file (for geometry and angles)')
    parser.add_argument('--avg_ct', '-c', type=str, required=True,
                        help='Path to average CT volume (.npy)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output projections (.npy)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Average Projections from Average CT")
    print("=" * 60)
    
    # Load pickle file for geometry and angles
    print(f"\n[1] Loading geometry from: {args.pickle}")
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)
    
    angles = data['train']['angles']
    num_projections = len(angles)
    
    print(f"  Number of training projections: {num_projections}")
    print(f"  Angle range: [{np.min(angles):.4f}, {np.max(angles):.4f}]")
    
    # Check if angles are in radians
    if np.max(angles) < 2 * np.pi + 0.1:
        print("  Angles appear to be in radians")
        angles_rad = angles.astype(np.float32)
    else:
        print("  Converting angles from degrees to radians")
        angles_rad = np.deg2rad(angles).astype(np.float32)
    
    # Create geometry
    geo = create_geometry_from_pickle(data)
    print(f"\n[2] TIGRE Geometry:")
    print(f"  nVoxel: {geo.nVoxel}")
    print(f"  nDetector: {geo.nDetector}")
    
    # Load average CT volume
    print(f"\n[3] Loading average CT from: {args.avg_ct}")
    avg_ct = np.load(args.avg_ct).astype(np.float32)
    print(f"  Shape: {avg_ct.shape}")
    print(f"  Value range: [{avg_ct.min():.4f}, {avg_ct.max():.4f}]")
    
    # Ensure volume matches expected geometry
    expected_shape = tuple(geo.nVoxel)
    if avg_ct.shape != expected_shape:
        print(f"  Warning: Volume shape {avg_ct.shape} doesn't match geometry {expected_shape}")
        print(f"  Proceeding anyway...")
    
    # Forward project
    print(f"\n[4] Forward projecting average CT at {num_projections} angles...")
    print("  This may take a few minutes...")
    
    # TIGRE Ax (forward projection)
    # Ax expects volume shape [z, y, x] which matches our [D, H, W]
    avg_projections = Ax.Ax(avg_ct, geo, angles_rad, 'interpolated')
    
    print(f"  Forward projection complete!")
    print(f"  Projections shape: {avg_projections.shape}")
    print(f"  Value range: [{avg_projections.min():.6f}, {avg_projections.max():.6f}]")
    
    # Normalize projections to match training projection range
    train_projs = data['train']['projections']
    train_range = (train_projs.min(), train_projs.max())
    proj_range = (avg_projections.min(), avg_projections.max())
    
    print(f"\n[5] Normalizing projections:")
    print(f"  Training projection range: [{train_range[0]:.6f}, {train_range[1]:.6f}]")
    print(f"  Generated projection range: [{proj_range[0]:.6f}, {proj_range[1]:.6f}]")
    
    # Scale to match training projection range
    if proj_range[1] > proj_range[0]:
        # Normalize to [0, 1] first
        avg_projections = (avg_projections - proj_range[0]) / (proj_range[1] - proj_range[0])
        # Scale to training range
        avg_projections = avg_projections * (train_range[1] - train_range[0]) + train_range[0]
        print(f"  After normalization: [{avg_projections.min():.6f}, {avg_projections.max():.6f}]")
    
    # Save output
    print(f"\n[6] Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.save(args.output, avg_projections.astype(np.float32))
    
    print(f"  Saved successfully!")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    
    # Verify
    loaded = np.load(args.output)
    print(f"  Verified shape: {loaded.shape}")
    
    print("\n" + "=" * 60)
    print("Average projections generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
