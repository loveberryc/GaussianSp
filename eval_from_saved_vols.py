#!/usr/bin/env python
"""
eval_from_saved_vols.py - Evaluate from saved vol_gt and vol_pred .npy files
"""
import os
import numpy as np
import sys
sys.path.append("./")
from x2_gaussian.utils.image_utils import metric_vol


def evaluate_saved_vols(model_path, iteration):
    """Evaluate from saved vol_gt_T*.npy and vol_pred_T*.npy files"""
    vol_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    
    psnr_3d_list = []
    ssim_3d_list = []
    
    print(f"Evaluating from saved volumes in: {vol_path}")
    
    for t in range(10):
        gt_file = os.path.join(vol_path, f"vol_gt_T{t}.npy")
        pred_file = os.path.join(vol_path, f"vol_pred_T{t}.npy")
        
        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"  T{t}: Files not found, skipping")
            continue
        
        vol_gt = np.load(gt_file)
        vol_pred = np.load(pred_file)
        
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")
        
        psnr_3d_list.append(psnr_3d)
        ssim_3d_list.append(ssim_3d)
        print(f"  T{t}: psnr3d={psnr_3d:.3f}, ssim3d={ssim_3d:.3f}")
    
    if psnr_3d_list:
        psnr_mean = np.mean(psnr_3d_list)
        ssim_mean = np.mean(ssim_3d_list)
        print(f"\n[ITER {iteration}] Results:")
        print(f"  psnr3d: {psnr_mean:.3f}, ssim3d: {ssim_mean:.3f}")
        return {"psnr3d": psnr_mean, "ssim3d": ssim_mean}
    else:
        print("No volumes found!")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=50000)
    args = parser.parse_args()
    
    evaluate_saved_vols(args.model_path, args.iteration)
