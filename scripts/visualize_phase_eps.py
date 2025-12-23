#!/usr/bin/env python
"""
M5: Phase-Aware Trust-Region ε(t) Visualization Script

"Phase-aware trust-region allocates a bounded residual budget across
 respiratory phases, preserving Lagrangian dominance while enabling
 demand-driven corrections."

This script visualizes the learned ε(t) curve from a trained M5 model.

Usage:
    python scripts/visualize_phase_eps.py --model_path output/xxx/ --output_dir plots/
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./")

from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel


def load_model(model_path: str, hyper_args=None):
    """Load a trained GaussianModel from checkpoint."""
    # Try to find the latest checkpoint
    ckpt_dir = os.path.join(model_path, "ckpt")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if ckpts:
            # Sort by iteration number
            ckpts.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
            ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            print(f"Found checkpoint: {ckpt_path}")
        else:
            print(f"No checkpoints found in {ckpt_dir}")
            return None
    else:
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return None
    
    # Load model
    checkpoint = torch.load(ckpt_path)
    if isinstance(checkpoint, tuple):
        model_params, iteration = checkpoint
    else:
        model_params = checkpoint
        iteration = 0
    
    print(f"Loaded from iteration {iteration}")
    return model_params, iteration


def visualize_phase_eps_curve(
    model_path: str,
    output_dir: str,
    num_samples: int = 50
):
    """
    Visualize the learned ε(t) curve from a trained M5 model.
    
    Args:
        model_path: Path to model directory
        output_dir: Output directory for plots
        num_samples: Number of time samples for visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_str = f.read()
        args = eval(cfg_str)
        print(f"Loaded config from {cfg_path}")
    else:
        print(f"Config not found at {cfg_path}")
        return
    
    # Check if M5 is enabled
    if not getattr(args, 'phase_eps_enable', False):
        print("M5 (phase_eps_enable) is not enabled in this model.")
        return
    
    # Load checkpoint to get phase_epsilon parameters
    ckpt_dir = os.path.join(model_path, "ckpt")
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    
    ckpts.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if isinstance(checkpoint, tuple):
        state_dict, iteration = checkpoint
    else:
        state_dict = checkpoint
        iteration = 0
    
    # Find phase_epsilon parameters
    phase_eps_keys = [k for k in state_dict.keys() if 'phase_epsilon' in k]
    if not phase_eps_keys:
        print("No phase_epsilon parameters found in checkpoint.")
        return
    
    print(f"Found phase_epsilon keys: {phase_eps_keys}")
    
    # Get mode and parameters
    mode = getattr(args, 'phase_eps_mode', 'per_frame')
    eps_max = getattr(args, 'phase_eps_eps_max', None) or getattr(args, 'eps_max', 0.03)
    eps_init = getattr(args, 'phase_eps_init_eps', None) or getattr(args, 'eps_init', 0.015)
    num_frames = getattr(args, 'phase_eps_num_frames', 10)
    
    print(f"Mode: {mode}, eps_max: {eps_max}, eps_init: {eps_init}")
    
    # Compute ε(t) values
    device = 'cpu'
    
    if mode == 'per_frame':
        # Find g parameter
        g_key = [k for k in phase_eps_keys if k.endswith('.g')][0]
        g = state_dict[g_key]  # [num_frames]
        
        eps_values = eps_max * torch.sigmoid(g)
        t_values = torch.linspace(0, 1, len(eps_values))
        
        print(f"Per-frame ε values: {eps_values.tolist()}")
        
    else:  # tiny_mlp
        # Need to reconstruct the MLP and run inference
        from x2_gaussian.gaussian.anchor_module import PhaseEpsilon
        
        mlp_hidden = getattr(args, 'phase_eps_mlp_hidden', 32)
        mlp_layers = getattr(args, 'phase_eps_mlp_layers', 2)
        
        phase_eps = PhaseEpsilon(
            mode='tiny_mlp',
            num_frames=num_frames,
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
            eps_init=eps_init,
            eps_max=eps_max
        )
        
        # Load state dict (need to strip prefix)
        prefix = phase_eps_keys[0].rsplit('.', 1)[0]
        sub_state = {k.replace(prefix + '.', ''): v for k, v in state_dict.items() if k.startswith(prefix)}
        phase_eps.load_state_dict(sub_state)
        phase_eps.eval()
        
        # Sample ε(t) at multiple points
        t_values = torch.linspace(0, 1, num_samples)
        eps_values = []
        with torch.no_grad():
            for t in t_values:
                eps = phase_eps(t)
                eps_values.append(eps.item())
        eps_values = torch.tensor(eps_values)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: ε(t) curve
    ax1 = axes[0]
    ax1.plot(t_values.numpy(), eps_values.numpy(), 'b-', linewidth=2, label='ε(t)')
    ax1.axhline(y=eps_init, color='r', linestyle='--', alpha=0.7, label=f'ε_init={eps_init:.4f}')
    ax1.axhline(y=eps_max, color='g', linestyle='--', alpha=0.5, label=f'ε_max={eps_max:.4f}')
    ax1.set_xlabel('Time t (normalized)', fontsize=12)
    ax1.set_ylabel('ε(t)', fontsize=12)
    ax1.set_title(f'M5: Phase-Aware Trust-Region ε(t)\nMode: {mode}, Iteration: {iteration}', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, eps_max * 1.1])
    
    # Plot 2: ε(t) as bar chart (for per_frame mode)
    ax2 = axes[1]
    if mode == 'per_frame':
        phases = np.arange(len(eps_values))
        ax2.bar(phases, eps_values.numpy(), color='steelblue', alpha=0.8, edgecolor='black')
        ax2.axhline(y=eps_init, color='r', linestyle='--', alpha=0.7, label=f'ε_init={eps_init:.4f}')
        ax2.set_xlabel('Phase Index', fontsize=12)
        ax2.set_ylabel('ε', fontsize=12)
        ax2.set_title(f'Per-Frame ε Values (T={len(eps_values)} phases)', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        # For tiny_mlp, show the same curve but as scatter
        ax2.scatter(t_values.numpy(), eps_values.numpy(), c='steelblue', s=30, alpha=0.8)
        ax2.axhline(y=eps_init, color='r', linestyle='--', alpha=0.7, label=f'ε_init={eps_init:.4f}')
        ax2.set_xlabel('Time t (normalized)', fontsize=12)
        ax2.set_ylabel('ε(t)', fontsize=12)
        ax2.set_title(f'Tiny-MLP ε(t) Samples (n={num_samples})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    exp_name = os.path.basename(model_path.rstrip('/'))
    output_path = os.path.join(output_dir, f'phase_eps_curve_{exp_name}_iter{iteration}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Also save the raw data
    data_path = os.path.join(output_dir, f'phase_eps_data_{exp_name}_iter{iteration}.npz')
    np.savez(data_path, 
             t_values=t_values.numpy(),
             eps_values=eps_values.numpy(),
             mode=mode,
             eps_max=eps_max,
             eps_init=eps_init,
             iteration=iteration)
    print(f"Saved data to: {data_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("M5 Phase-Aware Trust-Region ε(t) Statistics")
    print("="*60)
    print(f"Mode:      {mode}")
    print(f"ε_init:    {eps_init:.6f}")
    print(f"ε_max:     {eps_max:.6f}")
    print(f"ε mean:    {eps_values.mean().item():.6f}")
    print(f"ε min:     {eps_values.min().item():.6f}")
    print(f"ε max:     {eps_values.max().item():.6f}")
    print(f"ε std:     {eps_values.std().item():.6f}")
    print(f"Iteration: {iteration}")
    print("="*60)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='M5: Visualize Phase-Aware Trust-Region ε(t) curve'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output_dir', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of time samples for visualization (tiny_mlp mode)')
    
    args = parser.parse_args()
    
    visualize_phase_eps_curve(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
