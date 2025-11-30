#!/usr/bin/env python
"""
eval_4d_x2_gaussian.py - Evaluation script for 4D X2-Gaussian with V7.1 correction and TTO-α.

This script evaluates a trained V7 model with optional:
- V7.1 Consistency-Aware Rendering: Uses backward field D_b to correct forward deformed centers
- TTO-α (Test-Time Optimization): Optimizes a scalar alpha per-case for better reconstruction

Usage:
    # V7 baseline (no correction)
    python eval_4d_x2_gaussian.py -s data/case1.pickle -m output/case1_v7
    
    # V7.1-fixed (fixed alpha correction, no TTO)
    python eval_4d_x2_gaussian.py -s data/case1.pickle -m output/case1_v7 \
        --use_v7_1_correction --correction_alpha 0.5
    
    # V7.1 + TTO-α (optimize alpha per-case)
    python eval_4d_x2_gaussian.py -s data/case1.pickle -m output/case1_v7 \
        --use_v7_1_correction --use_tto_alpha \
        --tto_alpha_init 0.3 --tto_alpha_lr 1e-2 --tto_alpha_steps 100

Author: X2-Gaussian Team
"""

import os
import sys
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import yaml
from tqdm import tqdm
from random import sample
import math

sys.path.append("./")
from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.loss_utils import l1_loss, ssim
from x2_gaussian.utils.image_utils import metric_vol, metric_proj
from x2_gaussian.utils.general_utils import safe_state


def optimize_alpha_for_case(
    gaussians: GaussianModel,
    cameras: list,
    pipe: PipelineParams,
    tto_alpha_init: float = 0.3,
    tto_alpha_lr: float = 1e-2,
    tto_alpha_steps: int = 100,
    tto_alpha_reg: float = 1e-3,
    tto_num_views_per_step: int = 8,
    lambda_dssim: float = 0.25,
    verbose: bool = True,
) -> float:
    """
    Perform Test-Time Optimization (TTO-α) to find the optimal alpha for a specific case.
    
    TTO-α freezes all model parameters and only optimizes a scalar alpha that controls
    the strength of V7.1 consistency-aware correction:
        y_corrected = y - alpha * residual
    
    The optimization minimizes:
        L_TTO = L_render + lambda_alpha * (alpha - alpha_init)^2
    
    Args:
        gaussians: Trained GaussianModel (all parameters will be frozen)
        cameras: List of Camera objects for optimization
        pipe: Pipeline parameters
        tto_alpha_init: Initial value for alpha
        tto_alpha_lr: Learning rate for alpha optimization
        tto_alpha_steps: Number of optimization steps
        tto_alpha_reg: Regularization weight to keep alpha near initial value
        tto_num_views_per_step: Number of views to sample per optimization step
        lambda_dssim: Weight for D-SSIM loss (same as training)
        verbose: Print progress
    
    Returns:
        alpha_star: Optimized alpha value
    """
    if verbose:
        print("=" * 60)
        print("TTO-α: Test-Time Optimization")
        print("=" * 60)
        print(f"  Initial alpha: {tto_alpha_init}")
        print(f"  Learning rate: {tto_alpha_lr}")
        print(f"  Optimization steps: {tto_alpha_steps}")
        print(f"  Regularization weight: {tto_alpha_reg}")
        print(f"  Views per step: {tto_num_views_per_step}")
        print(f"  Total cameras available: {len(cameras)}")
        print("=" * 60)
    
    # Step 1: Freeze all model parameters
    # Freeze deformation network parameters
    for param in gaussians._deformation.parameters():
        param.requires_grad_(False)
    
    # Freeze Gaussian model's own parameters (xyz, scaling, rotation, etc.)
    # These are stored as nn.Parameter attributes
    for attr_name in ['_xyz', '_scaling', '_rotation', '_density', '_features_dc', '_features_rest']:
        if hasattr(gaussians, attr_name):
            attr = getattr(gaussians, attr_name)
            if isinstance(attr, nn.Parameter):
                attr.requires_grad_(False)
    
    # Freeze period parameter
    if hasattr(gaussians, 'period') and isinstance(gaussians.period, nn.Parameter):
        gaussians.period.requires_grad_(False)
    
    # Step 2: Create learnable alpha parameter
    device = gaussians.get_xyz.device
    alpha = nn.Parameter(torch.tensor(tto_alpha_init, device=device, dtype=torch.float32))
    
    # Step 3: Set up optimizer
    optimizer = torch.optim.Adam([alpha], lr=tto_alpha_lr)
    
    # Step 4: TTO optimization loop
    loss_history = []
    alpha_history = [tto_alpha_init]
    
    progress_bar = tqdm(range(tto_alpha_steps), desc="TTO-α", disable=not verbose)
    
    for step in progress_bar:
        # Sample views for this step
        num_samples = min(tto_num_views_per_step, len(cameras))
        sampled_cameras = sample(cameras, num_samples)
        
        total_loss = 0.0
        
        for cam in sampled_cameras:
            # Forward render with V7.1 correction using current alpha
            render_pkg = render(
                cam,
                gaussians,
                pipe,
                stage='fine',
                use_v7_1_correction=True,
                correction_alpha=alpha,  # Use learnable alpha
            )
            image = render_pkg["render"]
            
            # Ground truth
            gt_image = cam.original_image.cuda()
            
            # Compute render loss (L1 + D-SSIM)
            loss_l1 = l1_loss(image, gt_image)
            loss_render = loss_l1
            
            if lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss_render = loss_render + lambda_dssim * loss_dssim
            
            total_loss = total_loss + loss_render
        
        # Average over sampled views
        total_loss = total_loss / num_samples
        
        # Add regularization: lambda_alpha * (alpha - alpha_init)^2
        loss_reg = tto_alpha_reg * (alpha - tto_alpha_init).pow(2)
        loss_tto = total_loss + loss_reg
        
        # Backward and update
        optimizer.zero_grad()
        loss_tto.backward()
        optimizer.step()
        
        # Record history
        current_alpha = alpha.detach().item()
        current_loss = loss_tto.detach().item()
        loss_history.append(current_loss)
        alpha_history.append(current_alpha)
        
        # Update progress bar
        progress_bar.set_postfix({
            "alpha": f"{current_alpha:.4f}",
            "loss": f"{current_loss:.4e}",
        })
    
    # Step 5: Get optimized alpha
    alpha_star = alpha.detach().item()
    
    if verbose:
        print("=" * 60)
        print(f"TTO-α Completed")
        print(f"  Initial alpha: {tto_alpha_init:.4f}")
        print(f"  Optimized alpha: {alpha_star:.4f}")
        print(f"  Delta: {alpha_star - tto_alpha_init:+.4f}")
        print(f"  Initial loss: {loss_history[0]:.4e}")
        print(f"  Final loss: {loss_history[-1]:.4e}")
        print("=" * 60)
    
    return alpha_star


# ============================================================================
# TTO-α(t): Test-Time Optimization with Time-Dependent Alpha Network
# ============================================================================

class AlphaNetworkFourier(nn.Module):
    """Fourier basis network for time-dependent alpha at test time."""
    def __init__(self, num_freqs=4, alpha_init=0.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.alpha_init = alpha_init
        self.coeffs = nn.Parameter(torch.zeros(num_freqs * 2) * 0.01)
        self.bias = nn.Parameter(torch.tensor(alpha_init))
        
    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.coeffs.device, dtype=torch.float32)
        freqs = torch.arange(1, self.num_freqs + 1, device=t.device, dtype=torch.float32)
        phases = 2 * math.pi * freqs * t.unsqueeze(-1) if t.dim() > 0 else 2 * math.pi * freqs * t
        sin_features = torch.sin(phases)
        cos_features = torch.cos(phases)
        a_coeffs = self.coeffs[:self.num_freqs]
        b_coeffs = self.coeffs[self.num_freqs:]
        alpha = self.bias + (a_coeffs * sin_features).sum(-1) + (b_coeffs * cos_features).sum(-1)
        return torch.clamp(alpha, 0.0, 1.0)


class AlphaNetworkMLP(nn.Module):
    """Small MLP network for time-dependent alpha at test time."""
    def __init__(self, hidden_dim=32, alpha_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, alpha_init)
        
    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=next(self.parameters()).device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_input = t.view(-1, 1)
        alpha = torch.sigmoid(self.net(t_input).squeeze(-1))
        return alpha if alpha.numel() > 1 else alpha.squeeze()


def optimize_alpha_network_for_case(
    gaussians: GaussianModel,
    cameras: list,
    pipe: PipelineParams,
    network_type: str = 'fourier',  # 'fourier' or 'mlp'
    tto_alpha_init: float = 0.0,
    tto_alpha_lr: float = 1e-2,
    tto_alpha_steps: int = 100,
    tto_alpha_reg: float = 1e-3,
    tto_num_views_per_step: int = 8,
    num_fourier_freqs: int = 4,
    mlp_hidden_dim: int = 32,
    lambda_dssim: float = 0.25,
    verbose: bool = True,
) -> nn.Module:
    """
    Perform Test-Time Optimization with time-dependent α(t) network (TTO-α(t)).
    
    This is the test-time version of 扩展 2.3, where we optimize a small network
    g_θ(t) that predicts time-dependent alpha values, without modifying training.
    
    The optimization minimizes:
        L_TTO = E[L_render(α(t))] + λ_α * E[(α(t) - α_init)^2]
    
    Args:
        gaussians: Trained GaussianModel (all parameters will be frozen)
        cameras: List of Camera objects for optimization
        pipe: Pipeline parameters
        network_type: Type of alpha network ('fourier' or 'mlp')
        tto_alpha_init: Initial/base value for alpha
        tto_alpha_lr: Learning rate for network optimization
        tto_alpha_steps: Number of optimization steps
        tto_alpha_reg: Regularization weight to keep alpha near initial value
        tto_num_views_per_step: Number of views to sample per optimization step
        num_fourier_freqs: Number of Fourier frequencies (if fourier)
        mlp_hidden_dim: Hidden layer dimension (if mlp)
        lambda_dssim: Weight for D-SSIM loss
        verbose: Print progress
    
    Returns:
        alpha_network: Optimized alpha network module
    """
    if verbose:
        print("=" * 60)
        print(f"TTO-α(t): Test-Time Optimization with Time-Dependent Alpha")
        print("=" * 60)
        print(f"  Network type: {network_type}")
        print(f"  Initial alpha: {tto_alpha_init}")
        print(f"  Learning rate: {tto_alpha_lr}")
        print(f"  Optimization steps: {tto_alpha_steps}")
        print(f"  Regularization weight: {tto_alpha_reg}")
        print(f"  Views per step: {tto_num_views_per_step}")
        if network_type == 'fourier':
            print(f"  Fourier frequencies: {num_fourier_freqs}")
            num_params = num_fourier_freqs * 2 + 1
        else:
            print(f"  MLP hidden dim: {mlp_hidden_dim}")
            num_params = 1 * mlp_hidden_dim + mlp_hidden_dim + mlp_hidden_dim * mlp_hidden_dim + mlp_hidden_dim + mlp_hidden_dim * 1 + 1
        print(f"  Network parameters: ~{num_params}")
        print(f"  Total cameras available: {len(cameras)}")
        print("=" * 60)
    
    # Step 1: Freeze all model parameters
    for param in gaussians._deformation.parameters():
        param.requires_grad_(False)
    for attr_name in ['_xyz', '_scaling', '_rotation', '_density', '_features_dc', '_features_rest']:
        if hasattr(gaussians, attr_name):
            attr = getattr(gaussians, attr_name)
            if isinstance(attr, nn.Parameter):
                attr.requires_grad_(False)
    if hasattr(gaussians, 'period') and isinstance(gaussians.period, nn.Parameter):
        gaussians.period.requires_grad_(False)
    
    # Step 2: Create alpha network
    device = gaussians.get_xyz.device
    if network_type == 'fourier':
        alpha_network = AlphaNetworkFourier(
            num_freqs=num_fourier_freqs,
            alpha_init=tto_alpha_init
        ).to(device)
    else:  # mlp
        alpha_network = AlphaNetworkMLP(
            hidden_dim=mlp_hidden_dim,
            alpha_init=tto_alpha_init
        ).to(device)
    
    # Step 3: Set up optimizer for network parameters
    optimizer = torch.optim.Adam(alpha_network.parameters(), lr=tto_alpha_lr)
    
    # Step 4: TTO optimization loop
    loss_history = []
    
    progress_bar = tqdm(range(tto_alpha_steps), desc="TTO-α(t)", disable=not verbose)
    
    for step in progress_bar:
        num_samples = min(tto_num_views_per_step, len(cameras))
        sampled_cameras = sample(cameras, num_samples)
        
        total_loss = 0.0
        total_reg = 0.0
        alpha_values = []
        
        for cam in sampled_cameras:
            # Get time for this view
            current_time = torch.tensor(cam.time, device=device, dtype=torch.float32)
            
            # Get time-dependent alpha
            alpha_t = alpha_network(current_time)
            alpha_values.append(alpha_t.item() if isinstance(alpha_t, torch.Tensor) else alpha_t)
            
            # Forward render with V7.1 correction using current alpha(t)
            render_pkg = render(
                cam,
                gaussians,
                pipe,
                stage='fine',
                use_v7_1_correction=True,
                correction_alpha=alpha_t,
            )
            image = render_pkg["render"]
            gt_image = cam.original_image.cuda()
            
            # Compute render loss
            loss_l1 = l1_loss(image, gt_image)
            loss_render = loss_l1
            if lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss_render = loss_render + lambda_dssim * loss_dssim
            
            total_loss = total_loss + loss_render
            
            # Regularization: keep alpha(t) close to alpha_init
            reg_loss = tto_alpha_reg * (alpha_t - tto_alpha_init).pow(2)
            total_reg = total_reg + reg_loss
        
        # Average over sampled views
        total_loss = total_loss / num_samples
        total_reg = total_reg / num_samples
        loss_tto = total_loss + total_reg
        
        # Backward and update
        optimizer.zero_grad()
        loss_tto.backward()
        optimizer.step()
        
        # Record history
        current_loss = loss_tto.detach().item()
        loss_history.append(current_loss)
        mean_alpha = np.mean(alpha_values)
        
        progress_bar.set_postfix({
            "α_mean": f"{mean_alpha:.4f}",
            "loss": f"{current_loss:.4e}",
        })
    
    # Step 5: Evaluate final alpha distribution
    if verbose:
        # Sample alpha at different time points
        t_samples = torch.linspace(0, 1, 10, device=device)
        with torch.no_grad():
            alpha_samples = [alpha_network(t).item() for t in t_samples]
        
        print("=" * 60)
        print(f"TTO-α(t) Completed")
        print(f"  α(t) at different phases:")
        for i, (t, a) in enumerate(zip(t_samples.cpu().numpy(), alpha_samples)):
            print(f"    t={t:.1f}: α={a:.4f}")
        print(f"  Mean α: {np.mean(alpha_samples):.4f}")
        print(f"  Std α: {np.std(alpha_samples):.4f}")
        print(f"  Min α: {np.min(alpha_samples):.4f}")
        print(f"  Max α: {np.max(alpha_samples):.4f}")
        print(f"  Initial loss: {loss_history[0]:.4e}")
        print(f"  Final loss: {loss_history[-1]:.4e}")
        print("=" * 60)
    
    return alpha_network


def evaluate_model(
    gaussians: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    scanner_cfg: dict,
    use_v7_1_correction: bool = False,
    correction_alpha: float = 0.0,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the model with optional V7.1 correction.
    
    Args:
        gaussians: GaussianModel to evaluate
        scene: Scene with cameras and ground truth
        pipe: Pipeline parameters
        scanner_cfg: Scanner configuration
        use_v7_1_correction: Enable V7.1 correction
        correction_alpha: Alpha value for correction
        save_path: Path to save evaluation results
        verbose: Print progress
    
    Returns:
        eval_results: Dictionary containing all evaluation metrics
    """
    if verbose:
        print("=" * 60)
        mode_str = "V7.1" if use_v7_1_correction else "V7 (baseline)"
        print(f"Evaluating {mode_str}")
        if use_v7_1_correction:
            print(f"  correction_alpha = {correction_alpha:.4f}")
        print("=" * 60)
    
    eval_results = {}
    
    # Create query function with V7.1 support
    def queryfunc(pc, time, stage):
        return query(
            pc,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe,
            time,
            stage,
            use_v7_1_correction=use_v7_1_correction,
            correction_alpha=correction_alpha,
        )
    
    # Create render function with V7.1 support
    def renderfunc(cam, pc, stage):
        return render(
            cam,
            pc,
            pipe,
            stage,
            use_v7_1_correction=use_v7_1_correction,
            correction_alpha=correction_alpha,
        )
    
    # Evaluate 2D rendering performance
    validation_configs = [
        {"name": "render_train", "cameras": scene.getTrainCameras()},
        {"name": "render_test", "cameras": scene.getTestCameras()},
    ]
    
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            images = []
            gt_images = []
            
            if verbose:
                print(f"  Rendering {config['name']} ({len(config['cameras'])} views)...")
            
            for viewpoint in tqdm(config["cameras"], desc=config["name"], disable=not verbose):
                image = renderfunc(viewpoint, gaussians, 'fine')["render"]
                gt_image = viewpoint.original_image.to("cuda")
                images.append(image)
                gt_images.append(gt_image)
            
            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
            
            psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
            ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
            
            eval_results[f"{config['name']}_psnr_2d"] = psnr_2d
            eval_results[f"{config['name']}_ssim_2d"] = ssim_2d
            eval_results[f"{config['name']}_psnr_2d_projs"] = psnr_2d_projs
            eval_results[f"{config['name']}_ssim_2d_projs"] = ssim_2d_projs
    
    # Evaluate 3D reconstruction performance
    breath_cycle = 3.0
    num_phases = 10
    phase_time = breath_cycle / num_phases
    mid_phase_time = phase_time / 2
    scanTime = 60.0
    
    psnr_3d_list = []
    ssim_3d_list = []
    ssim_3d_axis_x_list = []
    ssim_3d_axis_y_list = []
    ssim_3d_axis_z_list = []
    
    if verbose:
        print(f"  Evaluating 3D reconstruction (10 phases)...")
    
    with torch.no_grad():
        for t in tqdm(range(10), desc="3D eval", disable=not verbose):
            time = (mid_phase_time + phase_time * t) / scanTime
            vol_pred = queryfunc(gaussians, time, 'fine')["vol"]
            vol_gt = scene.vol_gt[t]
            
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
            
            psnr_3d_list.append(psnr_3d)
            ssim_3d_list.append(ssim_3d)
            ssim_3d_axis_x_list.append(ssim_3d_axis[0])
            ssim_3d_axis_y_list.append(ssim_3d_axis[1])
            ssim_3d_axis_z_list.append(ssim_3d_axis[2])
    
    psnr_3d_mean = float(np.array(psnr_3d_list).mean())
    ssim_3d_mean = float(np.array(ssim_3d_list).mean())
    
    eval_results["psnr_3d_list"] = psnr_3d_list
    eval_results["ssim_3d_list"] = ssim_3d_list
    eval_results["ssim_3d_x_list"] = ssim_3d_axis_x_list
    eval_results["ssim_3d_y_list"] = ssim_3d_axis_y_list
    eval_results["ssim_3d_z_list"] = ssim_3d_axis_z_list
    eval_results["psnr_3d_mean"] = psnr_3d_mean
    eval_results["ssim_3d_mean"] = ssim_3d_mean
    
    # Add config info
    eval_results["use_v7_1_correction"] = use_v7_1_correction
    eval_results["correction_alpha"] = correction_alpha
    
    # Print summary
    if verbose:
        print("=" * 60)
        print("Evaluation Results:")
        print(f"  3D PSNR (mean): {psnr_3d_mean:.3f}")
        print(f"  3D SSIM (mean): {ssim_3d_mean:.4f}")
        if "render_test_psnr_2d" in eval_results:
            print(f"  2D PSNR (test): {eval_results['render_test_psnr_2d']:.3f}")
            print(f"  2D SSIM (test): {eval_results['render_test_ssim_2d']:.4f}")
        print("=" * 60)
    
    # Save results if path provided
    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        # Convert numpy arrays to lists for yaml serialization
        save_dict = {}
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.floating):
                save_dict[k] = [float(x) for x in v]
            else:
                save_dict[k] = v
        
        with open(save_path, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Results saved to: {save_path}")
    
    return eval_results


def evaluate_model_with_alpha_network(
    gaussians: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    scanner_cfg: dict,
    alpha_network: nn.Module,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the model with time-dependent V7.1 correction using an alpha network.
    
    Args:
        gaussians: GaussianModel to evaluate
        scene: Scene with cameras and ground truth
        pipe: Pipeline parameters
        scanner_cfg: Scanner configuration
        alpha_network: Trained alpha network that outputs α(t)
        save_path: Path to save evaluation results
        verbose: Print progress
    
    Returns:
        eval_results: Dictionary containing all evaluation metrics
    """
    device = gaussians.get_xyz.device
    
    if verbose:
        print("=" * 60)
        print(f"Evaluating V7.1 with TTO-α(t)")
        # Sample and display alpha values at different times
        t_samples = torch.linspace(0, 1, 5, device=device)
        with torch.no_grad():
            alpha_samples = [alpha_network(t).item() for t in t_samples]
        print(f"  α(t) range: [{min(alpha_samples):.4f}, {max(alpha_samples):.4f}]")
        print("=" * 60)
    
    eval_results = {}
    
    # Create query function with time-dependent V7.1 support
    def queryfunc(pc, time, stage):
        alpha_t = alpha_network(torch.tensor(time, device=device, dtype=torch.float32))
        return query(
            pc,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe,
            time,
            stage,
            use_v7_1_correction=True,
            correction_alpha=alpha_t,
        )
    
    # Create render function with time-dependent V7.1 support
    def renderfunc(cam, pc, stage):
        time_tensor = torch.tensor(cam.time, device=device, dtype=torch.float32)
        alpha_t = alpha_network(time_tensor)
        return render(
            cam,
            pc,
            pipe,
            stage,
            use_v7_1_correction=True,
            correction_alpha=alpha_t,
        )
    
    # Evaluate 2D rendering performance
    validation_configs = [
        {"name": "render_train", "cameras": scene.getTrainCameras()},
        {"name": "render_test", "cameras": scene.getTestCameras()},
    ]
    
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            images = []
            gt_images = []
            
            if verbose:
                print(f"  Rendering {config['name']} ({len(config['cameras'])} views)...")
            
            with torch.no_grad():
                for viewpoint in tqdm(config["cameras"], desc=config["name"], disable=not verbose):
                    image = renderfunc(viewpoint, gaussians, 'fine')["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
            
            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
            
            psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
            ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
            
            eval_results[f"{config['name']}_psnr_2d"] = psnr_2d
            eval_results[f"{config['name']}_ssim_2d"] = ssim_2d
            eval_results[f"{config['name']}_psnr_2d_projs"] = psnr_2d_projs
            eval_results[f"{config['name']}_ssim_2d_projs"] = ssim_2d_projs
    
    # Evaluate 3D reconstruction performance
    breath_cycle = 3.0
    num_phases = 10
    phase_time = breath_cycle / num_phases
    mid_phase_time = phase_time / 2
    scanTime = 60.0
    
    psnr_3d_list = []
    ssim_3d_list = []
    alpha_per_phase = []
    
    if verbose:
        print(f"  Evaluating 3D reconstruction (10 phases)...")
    
    with torch.no_grad():
        for t in tqdm(range(10), desc="3D eval", disable=not verbose):
            time = (mid_phase_time + phase_time * t) / scanTime
            
            # Get alpha for this phase
            alpha_t = alpha_network(torch.tensor(time, device=device, dtype=torch.float32))
            alpha_per_phase.append(alpha_t.item())
            
            vol_pred = queryfunc(gaussians, time, 'fine')["vol"]
            vol_gt = scene.vol_gt[t]
            
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
            
            psnr_3d_list.append(psnr_3d)
            ssim_3d_list.append(ssim_3d)
    
    psnr_3d_mean = float(np.array(psnr_3d_list).mean())
    ssim_3d_mean = float(np.array(ssim_3d_list).mean())
    
    eval_results["psnr_3d_list"] = psnr_3d_list
    eval_results["ssim_3d_list"] = ssim_3d_list
    eval_results["psnr_3d_mean"] = psnr_3d_mean
    eval_results["ssim_3d_mean"] = ssim_3d_mean
    eval_results["alpha_per_phase"] = alpha_per_phase
    eval_results["alpha_mean"] = float(np.mean(alpha_per_phase))
    eval_results["alpha_std"] = float(np.std(alpha_per_phase))
    
    # Add config info
    eval_results["use_v7_1_correction"] = True
    eval_results["alpha_type"] = "time_dependent"
    
    # Print summary
    if verbose:
        print("=" * 60)
        print("Evaluation Results:")
        print(f"  3D PSNR (mean): {psnr_3d_mean:.3f}")
        print(f"  3D SSIM (mean): {ssim_3d_mean:.4f}")
        if "render_test_psnr_2d" in eval_results:
            print(f"  2D PSNR (test): {eval_results['render_test_psnr_2d']:.3f}")
            print(f"  2D SSIM (test): {eval_results['render_test_ssim_2d']:.4f}")
        print(f"  α(t) per phase: {[f'{a:.3f}' for a in alpha_per_phase]}")
        print(f"  α mean±std: {np.mean(alpha_per_phase):.4f}±{np.std(alpha_per_phase):.4f}")
        print("=" * 60)
    
    # Save results if path provided
    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        save_dict = {}
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.floating):
                save_dict[k] = [float(x) for x in v]
            else:
                save_dict[k] = v
        
        with open(save_path, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Results saved to: {save_path}")
    
    return eval_results


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script for 4D X2-Gaussian with V7.1 and TTO-α")
    
    # Model and data paths
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # V7.1 Consistency-Aware Rendering parameters
    parser.add_argument("--use_v7_1_correction", action="store_true",
                        help="Enable V7.1 consistency-aware rendering correction")
    parser.add_argument("--correction_alpha", type=float, default=0.3,
                        help="Alpha coefficient for V7.1 correction (default: 0.3)")
    
    # TTO-α parameters (scalar alpha)
    parser.add_argument("--use_tto_alpha", action="store_true",
                        help="Enable TTO-α test-time optimization (scalar alpha)")
    parser.add_argument("--tto_alpha_init", type=float, default=0.3,
                        help="Initial alpha value for TTO (default: 0.3)")
    parser.add_argument("--tto_alpha_lr", type=float, default=1e-2,
                        help="Learning rate for TTO optimization (default: 1e-2)")
    parser.add_argument("--tto_alpha_steps", type=int, default=100,
                        help="Number of TTO optimization steps (default: 100)")
    parser.add_argument("--tto_alpha_reg", type=float, default=1e-3,
                        help="Regularization weight for TTO (default: 1e-3)")
    parser.add_argument("--tto_num_views_per_step", type=int, default=8,
                        help="Number of views to sample per TTO step (default: 8)")
    
    # TTO-α(t) parameters (time-dependent alpha network)
    parser.add_argument("--use_tto_alpha_t", action="store_true",
                        help="Enable TTO-α(t) test-time optimization with time-dependent alpha")
    parser.add_argument("--tto_alpha_network_type", type=str, default="fourier",
                        choices=["fourier", "mlp"],
                        help="Type of alpha network: 'fourier' or 'mlp' (default: fourier)")
    parser.add_argument("--tto_alpha_fourier_freqs", type=int, default=4,
                        help="Number of Fourier frequencies for alpha network (default: 4)")
    parser.add_argument("--tto_alpha_mlp_hidden", type=int, default=32,
                        help="Hidden dimension for MLP alpha network (default: 32)")
    
    # Other parameters
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Iteration to load (default: 30000)")
    parser.add_argument("--lambda_dssim", type=float, default=0.25,
                        help="D-SSIM weight for TTO loss (default: 0.25)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output file name")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize system state
    safe_state(args.quiet)
    
    print("=" * 60)
    print("4D X2-Gaussian Evaluation")
    print("=" * 60)
    print(f"Source path: {args.source_path}")
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"V7.1 correction: {args.use_v7_1_correction}")
    if args.use_v7_1_correction:
        print(f"  correction_alpha: {args.correction_alpha}")
    print(f"TTO-α (scalar): {args.use_tto_alpha}")
    if args.use_tto_alpha:
        print(f"  tto_alpha_init: {args.tto_alpha_init}")
        print(f"  tto_alpha_lr: {args.tto_alpha_lr}")
        print(f"  tto_alpha_steps: {args.tto_alpha_steps}")
    print(f"TTO-α(t) (time-dependent): {args.use_tto_alpha_t}")
    if args.use_tto_alpha_t:
        print(f"  network_type: {args.tto_alpha_network_type}")
        print(f"  tto_alpha_init: {args.tto_alpha_init}")
        print(f"  tto_alpha_steps: {args.tto_alpha_steps}")
    print("=" * 60)
    
    # Extract parameters
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    hyper = hp.extract(args)
    
    # Load scene
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    
    # Determine scale bound
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    
    # Initialize Gaussians
    gaussians = GaussianModel(scale_bound, hyper)
    
    # Load trained model
    point_cloud_path = osp.join(
        dataset.model_path, f"point_cloud/iteration_{args.iteration}"
    )
    
    print(f"Loading model from: {point_cloud_path}")
    
    # Load Gaussian parameters
    gaussians.load_ply(osp.join(point_cloud_path, "point_cloud.pickle"))
    
    # Load deformation network
    gaussians.load_model(point_cloud_path)
    
    # Set period reference for phase-conditioned deformation (if applicable)
    gaussians._deformation.set_period_ref(gaussians.period)
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"Period T_hat = {math.exp(gaussians.period.item()):.4f}")
    
    # Determine alpha to use
    final_alpha = args.correction_alpha
    alpha_network = None  # For TTO-α(t)
    
    # Run TTO-α(t) if enabled (time-dependent alpha network)
    if args.use_tto_alpha_t:
        # TTO-α(t) requires V7.1 correction
        if not args.use_v7_1_correction:
            print("Warning: TTO-α(t) requires V7.1 correction. Enabling use_v7_1_correction.")
            args.use_v7_1_correction = True
        
        # Use training cameras for TTO
        tto_cameras = scene.getTrainCameras()
        
        alpha_network = optimize_alpha_network_for_case(
            gaussians=gaussians,
            cameras=tto_cameras,
            pipe=pipe,
            network_type=args.tto_alpha_network_type,
            tto_alpha_init=args.tto_alpha_init,
            tto_alpha_lr=args.tto_alpha_lr,
            tto_alpha_steps=args.tto_alpha_steps,
            tto_alpha_reg=args.tto_alpha_reg,
            tto_num_views_per_step=args.tto_num_views_per_step,
            num_fourier_freqs=args.tto_alpha_fourier_freqs,
            mlp_hidden_dim=args.tto_alpha_mlp_hidden,
            lambda_dssim=args.lambda_dssim,
            verbose=not args.quiet,
        )
    # Run TTO-α if enabled (scalar alpha)
    elif args.use_tto_alpha:
        # TTO requires V7.1 correction
        if not args.use_v7_1_correction:
            print("Warning: TTO-α requires V7.1 correction. Enabling use_v7_1_correction.")
            args.use_v7_1_correction = True
        
        # Use training cameras for TTO
        tto_cameras = scene.getTrainCameras()
        
        final_alpha = optimize_alpha_for_case(
            gaussians=gaussians,
            cameras=tto_cameras,
            pipe=pipe,
            tto_alpha_init=args.tto_alpha_init,
            tto_alpha_lr=args.tto_alpha_lr,
            tto_alpha_steps=args.tto_alpha_steps,
            tto_alpha_reg=args.tto_alpha_reg,
            tto_num_views_per_step=args.tto_num_views_per_step,
            lambda_dssim=args.lambda_dssim,
            verbose=not args.quiet,
        )
    
    # Run evaluation
    # Determine output file name
    mode_str = "v7"
    if args.use_v7_1_correction:
        if args.use_tto_alpha_t:
            mode_str = f"v7.1_tto_alpha_t_{args.tto_alpha_network_type}"
        elif args.use_tto_alpha:
            mode_str = f"v7.1_tto_alpha{final_alpha:.3f}"
        else:
            mode_str = f"v7.1_fixed_alpha{args.correction_alpha:.3f}"
    
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    save_path = osp.join(
        dataset.model_path, "eval", f"eval_{mode_str}_iter{args.iteration}{suffix}.yml"
    )
    
    # Evaluate with time-dependent alpha or fixed alpha
    if alpha_network is not None:
        # Evaluation with TTO-α(t)
        eval_results = evaluate_model_with_alpha_network(
            gaussians=gaussians,
            scene=scene,
            pipe=pipe,
            scanner_cfg=scanner_cfg,
            alpha_network=alpha_network,
            save_path=save_path,
            verbose=not args.quiet,
        )
    else:
        eval_results = evaluate_model(
            gaussians=gaussians,
            scene=scene,
            pipe=pipe,
            scanner_cfg=scanner_cfg,
            use_v7_1_correction=args.use_v7_1_correction,
            correction_alpha=final_alpha,
            save_path=save_path,
            verbose=not args.quiet,
        )
    
    # Add TTO info to results
    if args.use_tto_alpha_t:
        eval_results["tto_mode"] = "alpha_t"
        eval_results["tto_network_type"] = args.tto_alpha_network_type
        eval_results["tto_alpha_init"] = args.tto_alpha_init
        eval_results["tto_alpha_steps"] = args.tto_alpha_steps
    elif args.use_tto_alpha:
        eval_results["tto_mode"] = "scalar"
        eval_results["tto_alpha_init"] = args.tto_alpha_init
        eval_results["tto_alpha_final"] = final_alpha
        eval_results["tto_alpha_steps"] = args.tto_alpha_steps
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
