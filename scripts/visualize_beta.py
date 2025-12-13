#!/usr/bin/env python3
"""
M1 Visualization: β(x,t) Contribution Map

This script visualizes the uncertainty-gated fusion parameter β(x,t),
showing where Eulerian (HexPlane) residuals contribute to the final deformation.

Usage:
    python scripts/visualize_beta.py --checkpoint path/to/ckpt --time 0.5 --output output_dir

Paper notation:
    Φ(x,t) = Φ_L(x,t) + β(x,t) · Φ_E(x,t)
    
    β ≈ 0: Lagrangian dominates (skeleton motion)
    β ≈ 1: Eulerian contributes significantly (high-frequency details)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.arguments import ModelHiddenParams


def load_model(checkpoint_path: str, device: str = 'cuda') -> GaussianModel:
    """Load trained GaussianModel from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Create dummy args for model initialization
    parser = argparse.ArgumentParser()
    hyper = ModelHiddenParams(parser)
    args = parser.parse_args([])
    
    # Create model
    gaussians = GaussianModel(hyper)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gaussians.restore(checkpoint, args)
    
    return gaussians


def compute_beta_map(gaussians: GaussianModel, time: float, device: str = 'cuda') -> dict:
    """
    Compute β(x,t) for all Gaussians at a given time.
    
    Returns:
        Dictionary with:
            - positions: [N, 3] Gaussian positions
            - beta: [N, 1] gate values
            - s_E: [N, 1] Eulerian log-variance
            - dx_anchor: [N, 3] Lagrangian displacement
            - dx_hex: [N, 3] Eulerian displacement
    """
    gaussians.eval()
    
    with torch.no_grad():
        # Get canonical positions
        positions = gaussians.get_xyz.to(device)
        N = positions.shape[0]
        
        # Create time tensor
        time_tensor = torch.full((N, 1), time, device=device)
        
        # Forward pass to compute deformation (this populates internal caches)
        if gaussians._deformation_anchor is not None:
            anchor = gaussians._deformation_anchor
            
            # Get deformed positions (triggers forward pass)
            scales = gaussians._scaling.to(device)
            rotations = gaussians._rotation.to(device)
            density = gaussians._opacity.to(device)
            
            deformed_pos, _, _ = anchor(
                positions, scales, rotations, density, time_tensor,
                is_training=False
            )
            
            # Get cached values
            beta = anchor.get_last_beta()
            s_E = anchor.original_deformation.get_last_s_E() if anchor.original_deformation else None
            dx_anchor = anchor._last_dx_anchor if hasattr(anchor, '_last_dx_anchor') else None
            dx_hex = anchor._last_dx_hex if hasattr(anchor, '_last_dx_hex') else None
            
            result = {
                'positions': positions.cpu().numpy(),
                'deformed_positions': deformed_pos.cpu().numpy() if deformed_pos is not None else None,
                'beta': beta.cpu().numpy() if beta is not None else None,
                's_E': s_E.cpu().numpy() if s_E is not None else None,
                'dx_anchor': dx_anchor.cpu().numpy() if dx_anchor is not None else None,
                'dx_hex': dx_hex.cpu().numpy() if dx_hex is not None else None,
                'time': time,
            }
            
            return result
    
    return None


def plot_beta_slice(result: dict, axis: int = 2, slice_idx: float = 0.5, 
                    output_path: str = None, show: bool = True):
    """
    Plot β values as a 2D slice through the volume.
    
    Args:
        result: Output from compute_beta_map()
        axis: Axis to slice along (0=x, 1=y, 2=z)
        slice_idx: Normalized position along axis (0-1)
        output_path: Path to save figure
        show: Whether to display figure
    """
    if result['beta'] is None:
        print("No beta values available (not in M1 mode?)")
        return
    
    positions = result['positions']
    beta = result['beta'].squeeze()
    
    # Determine slice range
    axis_min = positions[:, axis].min()
    axis_max = positions[:, axis].max()
    slice_pos = axis_min + slice_idx * (axis_max - axis_min)
    slice_width = (axis_max - axis_min) * 0.05  # 5% width
    
    # Select points near slice
    mask = np.abs(positions[:, axis] - slice_pos) < slice_width
    if mask.sum() == 0:
        print(f"No points near slice position {slice_pos}")
        return
    
    # Get 2D coordinates
    axes_2d = [i for i in range(3) if i != axis]
    x = positions[mask, axes_2d[0]]
    y = positions[mask, axes_2d[1]]
    beta_slice = beta[mask]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: β scatter
    ax1 = axes[0]
    scatter = ax1.scatter(x, y, c=beta_slice, cmap='RdYlBu_r', 
                         s=1, vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='β(x,t)')
    ax1.set_xlabel(f'Axis {axes_2d[0]}')
    ax1.set_ylabel(f'Axis {axes_2d[1]}')
    ax1.set_title(f'β(x,t) at t={result["time"]:.2f}, slice axis={axis}={slice_pos:.3f}\n'
                  f'Blue=Lagrangian, Red=Eulerian')
    ax1.set_aspect('equal')
    
    # Plot 2: β histogram
    ax2 = axes[1]
    ax2.hist(beta_slice, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(beta_slice.mean(), color='red', linestyle='--', 
                label=f'Mean: {beta_slice.mean():.4f}')
    ax2.set_xlabel('β value')
    ax2.set_ylabel('Count')
    ax2.set_title(f'β Distribution (N={mask.sum()})')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_beta_statistics(result: dict, output_path: str = None, show: bool = True):
    """Plot β and s_E statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: β histogram
    if result['beta'] is not None:
        beta = result['beta'].squeeze()
        axes[0].hist(beta, bins=50, edgecolor='black', alpha=0.7, color='blue')
        axes[0].axvline(beta.mean(), color='red', linestyle='--', 
                       label=f'Mean: {beta.mean():.4f}')
        axes[0].set_xlabel('β(x,t)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Gate Value Distribution (t={result["time"]:.2f})')
        axes[0].legend()
    
    # Plot 2: s_E histogram
    if result['s_E'] is not None:
        s_E = result['s_E'].squeeze()
        axes[1].hist(s_E, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(s_E.mean(), color='red', linestyle='--', 
                       label=f'Mean: {s_E.mean():.4f}')
        axes[1].set_xlabel('s_E = log(σ²_E)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Eulerian Log-Variance Distribution')
        axes[1].legend()
    
    # Plot 3: Displacement magnitudes
    if result['dx_anchor'] is not None and result['dx_hex'] is not None:
        dx_anchor_mag = np.linalg.norm(result['dx_anchor'], axis=1)
        dx_hex_mag = np.linalg.norm(result['dx_hex'], axis=1)
        
        axes[2].hist(dx_anchor_mag, bins=50, alpha=0.5, label=f'Lagrangian (μ={dx_anchor_mag.mean():.4f})', color='blue')
        axes[2].hist(dx_hex_mag, bins=50, alpha=0.5, label=f'Eulerian (μ={dx_hex_mag.mean():.4f})', color='red')
        axes[2].set_xlabel('Displacement Magnitude')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Displacement Distribution')
        axes[2].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_beta_volume(result: dict, output_path: str, resolution: int = 64):
    """
    Splat β values to a 3D grid and save as numpy file.
    
    This creates a volumetric representation for 3D visualization tools.
    """
    if result['beta'] is None:
        print("No beta values available")
        return
    
    positions = result['positions']
    beta = result['beta'].squeeze()
    
    # Normalize positions to [0, 1]
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
    
    # Create 3D grid
    volume = np.zeros((resolution, resolution, resolution))
    count = np.zeros((resolution, resolution, resolution))
    
    # Splat β values to grid
    grid_coords = (pos_norm * (resolution - 1)).astype(int)
    grid_coords = np.clip(grid_coords, 0, resolution - 1)
    
    for i in range(len(beta)):
        x, y, z = grid_coords[i]
        volume[x, y, z] += beta[i]
        count[x, y, z] += 1
    
    # Average
    mask = count > 0
    volume[mask] /= count[mask]
    
    # Save
    np.savez(output_path, 
             beta_volume=volume,
             positions=positions,
             beta=beta,
             pos_min=pos_min,
             pos_max=pos_max,
             time=result['time'])
    print(f"Saved beta volume to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize M1 β(x,t) contribution map')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--time', '-t', type=float, default=0.5,
                        help='Time value for visualization (default: 0.5)')
    parser.add_argument('--output', '-o', type=str, default='output/m1_viz',
                        help='Output directory')
    parser.add_argument('--slice_axis', type=int, default=2,
                        help='Axis to slice (0=x, 1=y, 2=z, default: 2)')
    parser.add_argument('--slice_pos', type=float, default=0.5,
                        help='Normalized slice position (0-1, default: 0.5)')
    parser.add_argument('--volume_res', type=int, default=64,
                        help='Volume grid resolution (default: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display figures (save only)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    try:
        gaussians = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: This script requires a trained M1 model checkpoint.")
        return
    
    # Check if M1 mode
    if gaussians._deformation_anchor is None:
        print("Error: Model does not have anchor deformation (not PhysX-Boosted)")
        return
    
    fusion_mode = getattr(gaussians._deformation_anchor, 'fusion_mode', 'fixed_alpha')
    if fusion_mode != 'uncertainty_gated':
        print(f"Warning: Model fusion_mode is '{fusion_mode}', not 'uncertainty_gated'")
        print("β values may not be available.")
    
    # Compute β map
    print(f"\nComputing β(x,t) at t={args.time}...")
    result = compute_beta_map(gaussians, args.time, args.device)
    
    if result is None:
        print("Failed to compute beta map")
        return
    
    # Print statistics
    if result['beta'] is not None:
        beta = result['beta'].squeeze()
        print(f"\n=== M1 Statistics at t={args.time} ===")
        print(f"  β mean: {beta.mean():.4f}")
        print(f"  β std:  {beta.std():.4f}")
        print(f"  β min:  {beta.min():.4f}")
        print(f"  β max:  {beta.max():.4f}")
        print(f"  N points: {len(beta)}")
    
    if result['s_E'] is not None:
        s_E = result['s_E'].squeeze()
        sigma2_E = np.exp(s_E)
        print(f"\n  s_E mean: {s_E.mean():.4f} (σ²_E = {sigma2_E.mean():.4f})")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Slice plot
    slice_path = os.path.join(args.output, f'beta_slice_t{args.time:.2f}.png')
    plot_beta_slice(result, axis=args.slice_axis, slice_idx=args.slice_pos,
                   output_path=slice_path, show=not args.no_show)
    
    # 2. Statistics plot
    stats_path = os.path.join(args.output, f'beta_stats_t{args.time:.2f}.png')
    plot_beta_statistics(result, output_path=stats_path, show=not args.no_show)
    
    # 3. Volume data
    volume_path = os.path.join(args.output, f'beta_volume_t{args.time:.2f}.npz')
    save_beta_volume(result, volume_path, resolution=args.volume_res)
    
    print(f"\nDone! Outputs saved to {args.output}/")


if __name__ == '__main__':
    main()
