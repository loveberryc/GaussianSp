#!/usr/bin/env python
"""
eval_saved_model.py - Evaluate a saved PhysX-Boosted model checkpoint
Mimics train.py's evaluation flow for models saved at iteration N
"""
import os
import os.path as osp
import torch
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel, render, query
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.image_utils import metric_vol, metric_proj


def evaluate_model(
    scene,
    gaussians,
    pipe,
    iteration,
    stage='fine',
):
    """Evaluate model following train.py's evaluation flow"""
    scanner_cfg = scene.scanner_cfg
    
    # Define query function (same as train.py)
    def queryfunc(gaussians, time, stage):
        return query(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe,
            time,
            stage,
        )
    
    # Define render function
    def renderfunc(viewpoint, gaussians, stage):
        return render(viewpoint, gaussians, pipe, stage)
    
    print(f"\n{'='*60}")
    print(f"Evaluating iteration {iteration}")
    print(f"{'='*60}")
    
    # Evaluate 2D rendering performance
    validation_configs = [
        {"name": "render_test", "cameras": scene.getTestCameras()},
    ]
    
    psnr_2d, ssim_2d = None, None
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            images = []
            gt_images = []
            print(f"Rendering {config['name']} ({len(config['cameras'])} views)...")
            
            with torch.no_grad():
                for viewpoint in tqdm(config["cameras"], desc=config["name"]):
                    image = renderfunc(viewpoint, gaussians, stage)["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
            
            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
            psnr_2d, _ = metric_proj(gt_images, images, "psnr")
            ssim_2d, _ = metric_proj(gt_images, images, "ssim")
            print(f"  {config['name']}: psnr2d={psnr_2d:.3f}, ssim2d={ssim_2d:.3f}")
    
    # Evaluate 3D reconstruction performance
    breath_cycle = 3.0
    num_phases = 10
    phase_time = breath_cycle / num_phases
    mid_phase_time = phase_time / 2
    scanTime = 60.0
    
    psnr_3d_list = []
    ssim_3d_list = []
    
    print("Evaluating 3D volumes (10 phases)...")
    with torch.no_grad():
        for t in tqdm(range(10), desc="3D eval"):
            time = (mid_phase_time + phase_time * t) / scanTime
            vol_pred = queryfunc(gaussians, time, stage)["vol"]
            vol_gt = scene.vol_gt[t]
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
            psnr_3d_list.append(psnr_3d)
            ssim_3d_list.append(ssim_3d)
    
    psnr_3d_mean = float(np.array(psnr_3d_list).mean())
    ssim_3d_mean = float(np.array(ssim_3d_list).mean())
    
    print(f"\n{'='*60}")
    print(f"[ITER {iteration}] Results:")
    print(f"  psnr3d: {psnr_3d_mean:.3f}, ssim3d: {ssim_3d_mean:.3f}")
    print(f"  psnr2d: {psnr_2d:.3f}, ssim2d: {ssim_2d:.3f}")
    print(f"{'='*60}")
    
    return {
        "psnr3d": psnr_3d_mean,
        "ssim3d": ssim_3d_mean,
        "psnr2d": psnr_2d,
        "ssim2d": ssim_2d,
    }


def main():
    parser = ArgumentParser(description="Evaluate saved model")
    
    # Add all parameter groups
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", type=int, default=50000, help="Iteration to evaluate")
    parser.add_argument("--eval_model_path", type=str, required=True, help="Path to model output folder")
    
    args = parser.parse_args()
    
    # Extract parameters
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    hyper = hp.extract(args)
    
    # Override model_path
    dataset.model_path = args.eval_model_path
    
    # Load scene
    print(f"Loading scene from: {dataset.source_path}")
    scene = Scene(dataset, shuffle=False)
    
    # Get scanner config
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    
    # Create GaussianModel
    gaussians = GaussianModel(scale_bound, hyper)
    
    # Load saved model
    load_path = osp.join(args.eval_model_path, "point_cloud", f"iteration_{args.iteration}")
    print(f"Loading model from: {load_path}")
    
    gaussians.load_from_model_path(load_path, opt)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Set scene gaussians
    scene.gaussians = gaussians
    
    # Run evaluation
    results = evaluate_model(scene, gaussians, pipe, args.iteration)
    
    # Save results
    result_path = osp.join(args.eval_model_path, f"eval_iter{args.iteration}.yml")
    with open(result_path, "w") as f:
        yaml.dump(results, f)
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
