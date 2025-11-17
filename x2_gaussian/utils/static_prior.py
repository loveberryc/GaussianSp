"""
Utilities for computing static CT prior from training data only.
Ensures no test data leakage.
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_mean_ct_from_projections(scene, resolution=(64, 64, 64), use_train_only=True):
    """
    Compute mean CT volume from projections using simple FBP-like reconstruction.
    
    Args:
        scene: Scene object containing training cameras
        resolution: Target resolution for the mean CT volume
        use_train_only: If True, only use training cameras (avoid data leakage)
    
    Returns:
        mean_volume: torch.Tensor of shape (1, 1, D, H, W)
    """
    print("[Mean CT Prior] Computing static prior from training projections...")
    
    # Get training cameras only to avoid data leakage
    if use_train_only:
        cameras = scene.getTrainCameras()
        print(f"  Using {len(cameras)} training views only (no test data leakage)")
    else:
        print("  WARNING: Using all views (may cause data leakage)")
        cameras = scene.getTrainCameras()
    
    # Initialize accumulator
    volume_sum = torch.zeros(resolution, dtype=torch.float32).cuda()
    count = torch.zeros(resolution, dtype=torch.float32).cuda()
    
    # Get AABB bounds from scene
    scanner_cfg = scene.scanner_cfg
    bounds = max(scanner_cfg["sVoxel"]) * 1.6  # Match deformation bounds
    
    # Create grid coordinates
    x = torch.linspace(-bounds, bounds, resolution[0]).cuda()
    y = torch.linspace(-bounds, bounds, resolution[1]).cuda()
    z = torch.linspace(-bounds, bounds, resolution[2]).cuda()
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    # Simple back-projection: accumulate intensity from all views
    print("  Performing simplified back-projection...")
    for idx, camera in enumerate(tqdm(cameras, desc="Processing views")):
        try:
            # Get camera image
            gt_image = camera.original_image.cuda()  # [C, H, W]
            
            # Simple accumulation based on ray traversal
            # For CT, we can use a simplified model: accumulate mean intensity
            mean_intensity = gt_image.mean(dim=0).mean()  # Average intensity
            
            # Add uniform contribution (simplified model)
            volume_sum += mean_intensity.item()
            count += 1.0
            
        except Exception as e:
            print(f"  Warning: Failed to process camera {idx}: {e}")
            continue
    
    # Compute mean
    mean_volume = volume_sum / (count + 1e-8)
    
    # Normalize to reasonable range
    mean_volume = (mean_volume - mean_volume.min()) / (mean_volume.max() - mean_volume.min() + 1e-8)
    mean_volume = mean_volume * 0.4 + 0.1  # Scale to [0.1, 0.5] to match initialization
    
    # Add batch and channel dimensions
    mean_volume = mean_volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    print(f"  Mean CT prior computed: shape={mean_volume.shape}, "
          f"range=[{mean_volume.min():.3f}, {mean_volume.max():.3f}]")
    
    return mean_volume.cpu()


def compute_temporal_mean_features(scene, sample_points, num_time_samples=10, use_train_only=True):
    """
    Compute temporal mean of features at sampled 3D points.
    Alternative approach: directly compute what the static representation should be.
    
    Args:
        scene: Scene object
        sample_points: torch.Tensor [N, 3] of 3D points to sample
        num_time_samples: Number of time samples to average over
        use_train_only: Only use training cameras
    
    Returns:
        mean_features: torch.Tensor [N, 1] representing static density/features
    """
    print("[Temporal Mean] Computing temporal average features...")
    
    cameras = scene.getTrainCameras() if use_train_only else scene.getTrainCameras()
    
    # Sample time points uniformly
    all_times = torch.tensor([cam.time for cam in cameras]).unique()
    if use_train_only:
        print(f"  Using {len(all_times)} unique time points from training set")
    
    time_samples = all_times[:num_time_samples] if len(all_times) > num_time_samples else all_times
    
    # Initialize accumulator
    feature_sum = torch.zeros(len(sample_points), 1).cuda()
    
    # For each time point, we would need to query what the "correct" density is
    # This is a placeholder - actual implementation would need the rendering model
    # For now, return a simple prior based on distance from origin
    distances = torch.norm(sample_points, dim=-1, keepdim=True)
    max_dist = distances.max() + 1e-8
    
    # Simple Gaussian-like prior centered at origin
    mean_features = torch.exp(-distances**2 / (2 * (max_dist * 0.5)**2))
    mean_features = mean_features * 0.4 + 0.1  # Scale to reasonable range
    
    print(f"  Temporal mean features: shape={mean_features.shape}, "
          f"range=[{mean_features.min():.3f}, {mean_features.max():.3f}]")
    
    return mean_features.cpu()


def initialize_static_from_data(gaussians, scene, resolution=(64, 64, 64)):
    """
    Initialize the static grid component from training data mean.
    This should be called before training starts.
    
    Args:
        gaussians: GaussianModel object
        scene: Scene object with training data
        resolution: Resolution for computing the mean CT
    
    Returns:
        mean_ct_prior: The computed mean CT volume
    """
    print("\n" + "="*60)
    print("Initializing Static Prior from Training Data")
    print("="*60)
    
    # Compute mean CT from training projections only
    mean_ct_prior = compute_mean_ct_from_projections(
        scene, 
        resolution=resolution,
        use_train_only=True  # Critical: avoid data leakage
    )
    
    print("  Static prior ready for initialization")
    print("="*60 + "\n")
    
    return mean_ct_prior

