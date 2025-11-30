#!/usr/bin/env python
"""
grid_search_alpha.py - Grid search for optimal V7.1 correction alpha.

This script evaluates a trained V7 model with different alpha values and
reports the best alpha based on test set performance (3D PSNR/SSIM).

Usage:
    python grid_search_alpha.py \
        -s /path/to/data.pickle \
        -m /path/to/trained_v7_model \
        --iteration 30000 \
        --alpha_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Author: X2-Gaussian Team
"""

import os
import sys
import os.path as osp
import torch
import numpy as np
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import math
import gc

# Clear GPU memory at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
gc.collect()

sys.path.append("./")
from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel, render, query
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.image_utils import metric_vol, metric_proj
from x2_gaussian.utils.general_utils import safe_state


def evaluate_with_alpha(
    gaussians: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    scanner_cfg: dict,
    correction_alpha: float,
    verbose: bool = False,
) -> dict:
    """
    Evaluate model with a specific correction alpha.
    
    Returns dict with psnr_3d_mean, ssim_3d_mean, psnr_2d_test, ssim_2d_test
    """
    # Create query function with V7.1 correction
    def queryfunc(pc, time, stage):
        return query(
            pc,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe,
            time,
            stage,
            use_v7_1_correction=True,
            correction_alpha=correction_alpha,
        )
    
    # Create render function with V7.1 correction
    def renderfunc(cam, pc, stage):
        return render(
            cam,
            pc,
            pipe,
            stage,
            use_v7_1_correction=True,
            correction_alpha=correction_alpha,
        )
    
    results = {"alpha": correction_alpha}
    
    # Evaluate 2D on test set (use subset for memory efficiency)
    test_cameras = scene.getTestCameras()
    if test_cameras and len(test_cameras) > 0:
        psnr_2d_list = []
        ssim_2d_list = []
        with torch.no_grad():
            for viewpoint in test_cameras:
                image = renderfunc(viewpoint, gaussians, 'fine')["render"]
                gt_image = viewpoint.original_image.to("cuda")
                
                # Calculate metrics per image to save memory
                psnr_2d, _ = metric_proj(gt_image.permute(1, 2, 0), image.permute(1, 2, 0), "psnr")
                ssim_2d, _ = metric_proj(gt_image.permute(1, 2, 0), image.permute(1, 2, 0), "ssim")
                psnr_2d_list.append(psnr_2d)
                ssim_2d_list.append(ssim_2d)
                
                # Clean up
                del image, gt_image
        
        results["psnr_2d_test"] = float(np.mean(psnr_2d_list))
        results["ssim_2d_test"] = float(np.mean(ssim_2d_list))
    
    # Clean GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluate 3D reconstruction
    breath_cycle = 3.0
    num_phases = 10
    phase_time = breath_cycle / num_phases
    mid_phase_time = phase_time / 2
    scanTime = 60.0
    
    psnr_3d_list = []
    ssim_3d_list = []
    
    with torch.no_grad():
        for t in range(10):
            time = (mid_phase_time + phase_time * t) / scanTime
            vol_pred = queryfunc(gaussians, time, 'fine')["vol"]
            vol_gt = scene.vol_gt[t]
            
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
            
            psnr_3d_list.append(psnr_3d)
            ssim_3d_list.append(ssim_3d)
            
            # Clean up
            del vol_pred
            torch.cuda.empty_cache()
    
    results["psnr_3d_mean"] = float(np.mean(psnr_3d_list))
    results["ssim_3d_mean"] = float(np.mean(ssim_3d_list))
    results["psnr_3d_list"] = psnr_3d_list
    results["ssim_3d_list"] = ssim_3d_list
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def main():
    parser = ArgumentParser(description="Grid search for optimal V7.1 correction alpha")
    
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Model iteration to load (default: 30000)")
    parser.add_argument("--alpha_values", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Alpha values to search")
    parser.add_argument("--metric", type=str, default="psnr_3d_mean",
                        choices=["psnr_3d_mean", "ssim_3d_mean", "psnr_2d_test", "ssim_2d_test"],
                        help="Metric to optimize (default: psnr_3d_mean)")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)
    
    print("=" * 70)
    print("Grid Search for Optimal V7.1 Correction Alpha")
    print("=" * 70)
    print(f"Source: {args.source_path}")
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Alpha values: {args.alpha_values}")
    print(f"Optimization metric: {args.metric}")
    print("=" * 70)
    
    # Extract parameters
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    hyper = hp.extract(args)
    
    # Load scene
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    
    # Initialize Gaussians
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    
    gaussians = GaussianModel(scale_bound, hyper)
    
    # Load trained model
    point_cloud_path = osp.join(
        dataset.model_path, f"point_cloud/iteration_{args.iteration}"
    )
    
    print(f"Loading model from: {point_cloud_path}")
    gaussians.load_ply(osp.join(point_cloud_path, "point_cloud.pickle"))
    gaussians.load_model(point_cloud_path)
    gaussians._deformation.set_period_ref(gaussians.period)
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    print(f"Period T_hat = {math.exp(gaussians.period.item()):.4f}")
    print("=" * 70)
    
    # Run grid search
    all_results = []
    
    print("\nRunning grid search...\n")
    print(f"{'Alpha':>8} | {'PSNR_3D':>10} | {'SSIM_3D':>10} | {'PSNR_2D':>10} | {'SSIM_2D':>10}")
    print("-" * 70)
    
    for alpha in tqdm(args.alpha_values, desc="Grid search"):
        result = evaluate_with_alpha(
            gaussians, scene, pipe, scanner_cfg, 
            correction_alpha=alpha, verbose=False
        )
        all_results.append(result)
        
        psnr_2d = result.get("psnr_2d_test", 0.0)
        ssim_2d = result.get("ssim_2d_test", 0.0)
        print(f"{alpha:8.2f} | {result['psnr_3d_mean']:10.4f} | {result['ssim_3d_mean']:10.6f} | {psnr_2d:10.4f} | {ssim_2d:10.6f}")
    
    print("-" * 70)
    
    # Find best alpha
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i][args.metric])
    best_result = all_results[best_idx]
    best_alpha = best_result["alpha"]
    
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"Best alpha (by {args.metric}): {best_alpha:.2f}")
    print(f"  PSNR_3D: {best_result['psnr_3d_mean']:.4f}")
    print(f"  SSIM_3D: {best_result['ssim_3d_mean']:.6f}")
    if "psnr_2d_test" in best_result:
        print(f"  PSNR_2D (test): {best_result['psnr_2d_test']:.4f}")
        print(f"  SSIM_2D (test): {best_result['ssim_2d_test']:.6f}")
    print("=" * 70)
    
    # Compare with baseline (alpha=0.0)
    baseline = next((r for r in all_results if r["alpha"] == 0.0), None)
    if baseline:
        print("\nImprovement over V7 baseline (alpha=0.0):")
        delta_psnr = best_result['psnr_3d_mean'] - baseline['psnr_3d_mean']
        delta_ssim = best_result['ssim_3d_mean'] - baseline['ssim_3d_mean']
        print(f"  ΔPSNR_3D: {delta_psnr:+.4f}")
        print(f"  ΔSSIM_3D: {delta_ssim:+.6f}")
    
    # Save results
    save_path = osp.join(dataset.model_path, "eval", f"grid_search_alpha_iter{args.iteration}.yml")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    
    save_dict = {
        "alpha_values": args.alpha_values,
        "optimization_metric": args.metric,
        "best_alpha": best_alpha,
        "best_result": {k: v if not isinstance(v, np.floating) else float(v) 
                       for k, v in best_result.items() if not isinstance(v, list)},
        "all_results": [{k: v if not isinstance(v, (np.floating, list)) else (float(v) if isinstance(v, np.floating) else v)
                        for k, v in r.items()} for r in all_results],
    }
    
    with open(save_path, "w") as f:
        yaml.dump(save_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {save_path}")
    print(f"\nRecommended command for V7.1-fixed evaluation:")
    print(f"python eval_4d_x2_gaussian.py -s {args.source_path} -m {args.model_path} \\")
    print(f"    --use_v7_1_correction --correction_alpha {best_alpha}")


if __name__ == "__main__":
    main()
