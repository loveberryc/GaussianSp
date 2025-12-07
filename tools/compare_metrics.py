#!/usr/bin/env python3
"""
Compare PSNR/SSIM metrics between X2-Gaussian and STNF4D calculation methods.
"""

import os
import sys
import torch
import numpy as np
import argparse
import pickle
from skimage.metrics import structural_similarity

sys.path.append("./")


def norml(x):
    """STNF4D normalization: normalize to [0, 1]"""
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)


def stnf4d_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """STNF4D PSNR calculation (with per-volume normalization)"""
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    
    # Key difference: normalize each volume
    arr1 = norml(arr1)
    arr2 = norml(arr2)
    
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    psnr[zero_mse] = 100
    
    if size_average:
        return psnr.mean()
    else:
        return psnr


def stnf4d_ssim_3d(arr1, arr2, size_average=True):
    """STNF4D SSIM calculation (with per-volume normalization)"""
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    
    # Key difference: normalize each volume
    arr1 = norml(arr1)
    arr2 = norml(arr2)
    
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    
    N = arr1.shape[0]
    
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i], data_range=1.0)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)
    
    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i], data_range=1.0)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)
    
    # Width
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i], data_range=1.0)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)
    
    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3
    
    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg


def x2gaussian_psnr_3d(img1, img2, pixel_max=None):
    """X2-Gaussian PSNR calculation (NO normalization)"""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())
    
    if pixel_max is None:
        pixel_max = img1.max()
    
    mse_out = torch.mean((img1 - img2) ** 2)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    return psnr_out.item()


def x2gaussian_ssim_3d(img1, img2):
    """X2-Gaussian SSIM calculation"""
    from x2_gaussian.utils.loss_utils import ssim
    
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())
    
    ssims = []
    for axis in [0, 1, 2]:
        results = []
        count = 0
        n_slice = img1.shape[axis]
        for i in range(n_slice):
            if axis == 0:
                slice1 = img1[i, :, :]
                slice2 = img2[i, :, :]
            elif axis == 1:
                slice1 = img1[:, i, :]
                slice2 = img2[:, i, :]
            elif axis == 2:
                slice1 = img1[:, :, i]
                slice2 = img2[:, :, i]
            
            if slice1.max() > 0:
                result = ssim(slice1[None, None], slice2[None, None])
                count += 1
            else:
                result = 0
            results.append(result)
        results = torch.tensor(results)
        mean_results = torch.sum(results) / count if count > 0 else 0
        ssims.append(mean_results.item() if torch.is_tensor(mean_results) else mean_results)
    return float(np.mean(ssims))


def evaluate_model(model_path, data_path):
    """Evaluate a saved model with both metric methods"""
    from x2_gaussian.arguments import ModelParams, PipelineParams, ModelHiddenParams
    from x2_gaussian.gaussian import GaussianModel, query
    from x2_gaussian.dataset import Scene
    from argparse import ArgumentParser
    
    # Load data
    print(f"Loading data from {data_path}...")
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # Parse with data path
    args = parser.parse_args(["-s", data_path])
    model_args = lp.extract(args)
    pipe_args = pp.extract(args)
    hyper = hp.extract(args)
    
    model_args.source_path = data_path
    scene = Scene(model_args, True)
    
    # Get GT volume (all phases)
    vol_gt = scene.vol_gt  # [num_phases, D, H, W]
    scanner_cfg = scene.scanner_cfg
    num_phases = vol_gt.shape[0]
    
    print(f"GT volume shape: {vol_gt.shape}")
    print(f"Number of phases: {num_phases}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    scale_bound = None
    volume_to_world = max(scanner_cfg["sVoxel"])
    if model_args.scale_min and model_args.scale_max:
        scale_bound = (
            np.array([model_args.scale_min, model_args.scale_max]) * volume_to_world
        )
    
    gaussians = GaussianModel(scale_bound, hyper)
    gaussians.load_ply(os.path.join(model_path, "point_cloud.pickle"))
    gaussians._deformation.load_state_dict(
        torch.load(os.path.join(model_path, "deformation.pth"))
    )
    
    # Evaluate each phase
    x2g_psnrs = []
    x2g_ssims = []
    stnf_psnrs = []
    stnf_ssims = []
    
    for phase_idx in range(num_phases):
        time = phase_idx / num_phases
        
        print(f"\nEvaluating phase {phase_idx}/{num_phases} (time={time:.3f})...")
        
        # Query volume
        vol_pred = query(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe_args,
            time=time,
            stage='fine',
        )["vol"]
        
        vol_gt_phase = vol_gt[phase_idx].cuda()
        
        # X2-Gaussian metrics
        x2g_psnr = x2gaussian_psnr_3d(vol_gt_phase, vol_pred)
        x2g_ssim = x2gaussian_ssim_3d(vol_gt_phase, vol_pred)
        x2g_psnrs.append(x2g_psnr)
        x2g_ssims.append(x2g_ssim)
        
        # STNF4D metrics
        stnf_psnr = stnf4d_psnr_3d(vol_gt_phase, vol_pred)
        stnf_ssim = stnf4d_ssim_3d(vol_gt_phase, vol_pred)
        stnf_psnrs.append(stnf_psnr)
        stnf_ssims.append(stnf_ssim)
        
        print(f"  X2-Gaussian: PSNR={x2g_psnr:.2f} dB, SSIM={x2g_ssim:.4f}")
        print(f"  STNF4D:      PSNR={stnf_psnr:.2f} dB, SSIM={stnf_ssim:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"X2-Gaussian Method:")
    print(f"  Mean PSNR: {np.mean(x2g_psnrs):.2f} dB")
    print(f"  Mean SSIM: {np.mean(x2g_ssims):.4f}")
    print(f"\nSTNF4D Method (with normalization):")
    print(f"  Mean PSNR: {np.mean(stnf_psnrs):.2f} dB")
    print(f"  Mean SSIM: {np.mean(stnf_ssims):.4f}")
    print(f"\nDifference:")
    print(f"  PSNR: {np.mean(stnf_psnrs) - np.mean(x2g_psnrs):+.2f} dB")
    print(f"  SSIM: {np.mean(stnf_ssims) - np.mean(x2g_ssims):+.4f}")
    
    return {
        'x2g_psnr': np.mean(x2g_psnrs),
        'x2g_ssim': np.mean(x2g_ssims),
        'stnf_psnr': np.mean(stnf_psnrs),
        'stnf_ssim': np.mean(stnf_ssims),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model (point_cloud folder)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data pickle file")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_path)


if __name__ == "__main__":
    main()
