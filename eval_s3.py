#!/usr/bin/env python
"""Simple evaluation script for PhysX-Boosted models"""
import sys
sys.path.append("./")
import torch
import os
import yaml
from tqdm import tqdm

from x2_gaussian.gaussian import GaussianModel, render
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.image_utils import metric_vol, metric_proj

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=50000)
    args = parser.parse_args()
    
    # Load config from saved model
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f.read().replace("Namespace(", "{").replace(")", "}").replace("=", ": ").replace("'", '"'))
    
    # Create a simple namespace for model params
    class Args:
        def __init__(self):
            self.source_path = args.source_path
            self.model_path = args.model_path
            self.sh_degree = 3
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = True
    
    class HyperArgs:
        def __init__(self):
            self.defor_depth = 0
            self.net_width = 64
            self.no_do = False
            self.no_ds = False
            self.no_dr = False
            
    model_args = Args()
    hyper_args = HyperArgs()
    
    # Load scene
    print(f"Loading scene from: {args.source_path}")
    scene = Scene(model_args, shuffle=False)
    
    # Create and load gaussians
    gaussians = GaussianModel(model_args.sh_degree, hyper_args)
    load_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}")
    print(f"Loading model from: {load_path}")
    
    gaussians.load_ply(os.path.join(load_path, "point_cloud.pickle"))
    gaussians.load_model(load_path, hyper_args)
    gaussians.cuda()
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Get test cameras
    test_cameras = scene.getTestCameras()
    print(f"Evaluating on {len(test_cameras)} test views...")
    
    # Evaluate
    psnr_3d_list = []
    ssim_3d_list = []
    psnr_2d_list = []
    ssim_2d_list = []
    
    with torch.no_grad():
        for viewpoint in tqdm(test_cameras, desc="Evaluating"):
            result = render(viewpoint, gaussians, 'fine')
            image = result["render"]
            gt_image = viewpoint.original_image.cuda()
            
            psnr3d, ssim3d = metric_vol(image, gt_image)
            psnr2d, ssim2d = metric_proj(image, gt_image)
            
            psnr_3d_list.append(psnr3d)
            ssim_3d_list.append(ssim3d)
            psnr_2d_list.append(psnr2d)
            ssim_2d_list.append(ssim2d)
    
    psnr3d_mean = sum(psnr_3d_list) / len(psnr_3d_list)
    ssim3d_mean = sum(ssim_3d_list) / len(ssim_3d_list)
    psnr2d_mean = sum(psnr_2d_list) / len(psnr_2d_list)
    ssim2d_mean = sum(ssim_2d_list) / len(ssim_2d_list)
    
    print(f"\n[ITER {args.iteration}] Results:")
    print(f"  psnr3d: {psnr3d_mean:.3f}, ssim3d: {ssim3d_mean:.3f}")
    print(f"  psnr2d: {psnr2d_mean:.3f}, ssim2d: {ssim2d_mean:.3f}")

if __name__ == "__main__":
    main()
