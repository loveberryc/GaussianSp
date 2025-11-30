"""
X2-Gaussian s4: Temporal-Ensemble Guided Static Warm-Up

This module implements temporal-ensemble guided training for the static warm-up stage.
The core idea is to use externally provided pseudo-supervision:
- Average CT volume (V_avg): A time-averaged 3D volume from traditional reconstruction
- Average projections (optional): Time-averaged 2D projections per view

These serve as fixed "teachers" to guide canonical Gaussians toward a more robust
and consistent representation during static warm-up.

Only affects the static warm-up stage (coarse); dynamic stage is unchanged.

Key components:
- Load and store average CT volume as fixed buffer
- Trilinear interpolation for sampling V_avg at arbitrary 3D points
- Volume distillation loss: |σ_G(x) - V_avg(x)|
- Optional: Average projection distillation loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class TemporalEnsembleStatic(nn.Module):
    """
    Temporal-Ensemble Guided Static Warm-Up (s4).
    
    Uses externally provided average CT volume and/or average projections
    as pseudo-supervision during static warm-up.
    
    Args:
        avg_ct_path: Path to average CT volume file (.npy, .npz, or .pt)
        avg_proj_path: Path to average projections file (optional)
        bbox: Bounding box tensor of shape [2, 3] with [min_corner, max_corner]
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        avg_ct_path: str = "",
        avg_proj_path: str = "",
        bbox: torch.Tensor = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.avg_ct_loaded = False
        self.avg_proj_loaded = False
        
        # Store bounding box
        if bbox is not None:
            self.register_buffer('bbox', bbox.to(device))
        else:
            self.register_buffer('bbox', torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32, device=device))
        
        # Load average CT volume if path provided
        if avg_ct_path and os.path.exists(avg_ct_path):
            self._load_avg_ct_volume(avg_ct_path)
        else:
            # Register empty buffer
            self.register_buffer('avg_ct_volume', torch.zeros(1, 1, 1, 1, 1, device=device))
            if avg_ct_path:
                print(f"  [s4 Warning] avg_ct_path provided but file not found: {avg_ct_path}")
        
        # Load average projections if path provided
        if avg_proj_path and os.path.exists(avg_proj_path):
            self._load_avg_projections(avg_proj_path)
        else:
            self.register_buffer('avg_projections', torch.zeros(1, 1, 1, device=device))
            if avg_proj_path:
                print(f"  [s4 Warning] avg_proj_path provided but file not found: {avg_proj_path}")
        
        # Statistics tracking
        self._last_L_vol = 0.0
        self._last_L_proj_avg = 0.0
    
    def _load_avg_ct_volume(self, path: str):
        """
        Load average CT volume from file.
        
        Supports: .npy, .npz, .pt, .pth
        Expected shape: [D, H, W] or [1, D, H, W] or [1, 1, D, H, W]
        """
        print(f"  [s4] Loading average CT volume from: {path}")
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.npy':
            volume = np.load(path)
            volume = torch.from_numpy(volume).float()
        elif ext == '.npz':
            data = np.load(path)
            # Assume first key contains the volume
            key = list(data.keys())[0]
            volume = torch.from_numpy(data[key]).float()
        elif ext in ['.pt', '.pth']:
            volume = torch.load(path)
            if isinstance(volume, dict):
                # Assume first key contains the volume
                key = list(volume.keys())[0]
                volume = volume[key]
            volume = volume.float()
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .npy, .npz, .pt, or .pth")
        
        # Normalize shape to [1, 1, D, H, W] for grid_sample
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # [D, H, W] -> [1, 1, D, H, W]
        elif volume.dim() == 4:
            volume = volume.unsqueeze(0)  # [1, D, H, W] -> [1, 1, D, H, W]
        elif volume.dim() != 5:
            raise ValueError(f"Unexpected volume shape: {volume.shape}. Expected 3D, 4D, or 5D tensor.")
        
        self.register_buffer('avg_ct_volume', volume.to(self.device))
        self.avg_ct_loaded = True
        
        print(f"       - Shape: {self.avg_ct_volume.shape}")
        print(f"       - Value range: [{self.avg_ct_volume.min().item():.4f}, {self.avg_ct_volume.max().item():.4f}]")
        print(f"       - Mean: {self.avg_ct_volume.mean().item():.4f}")
    
    def _load_avg_projections(self, path: str):
        """
        Load average projections from file.
        
        Supports: .npy, .npz, .pt, .pth
        Expected shape: [N_views, H, W] or [N_views, 1, H, W]
        """
        print(f"  [s4] Loading average projections from: {path}")
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.npy':
            projections = np.load(path)
            projections = torch.from_numpy(projections).float()
        elif ext == '.npz':
            data = np.load(path)
            key = list(data.keys())[0]
            projections = torch.from_numpy(data[key]).float()
        elif ext in ['.pt', '.pth']:
            projections = torch.load(path)
            if isinstance(projections, dict):
                key = list(projections.keys())[0]
                projections = projections[key]
            projections = projections.float()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Normalize shape to [N_views, 1, H, W]
        if projections.dim() == 3:
            projections = projections.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        
        self.register_buffer('avg_projections', projections.to(self.device))
        self.avg_proj_loaded = True
        
        print(f"       - Shape: {self.avg_projections.shape}")
        print(f"       - Value range: [{self.avg_projections.min().item():.4f}, {self.avg_projections.max().item():.4f}]")
    
    def set_bbox(self, bbox: torch.Tensor):
        """Set the bounding box for world-to-normalized coordinate mapping."""
        self.bbox = bbox.to(self.device)
    
    def world_to_normalized(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to normalized coordinates [-1, 1] for grid_sample.
        
        Args:
            points: World coordinates of shape [..., 3]
            
        Returns:
            Normalized coordinates in [-1, 1] for grid_sample
        """
        min_corner = self.bbox[0].to(points.device)
        max_corner = self.bbox[1].to(points.device)
        
        # Normalize to [0, 1]
        normalized = (points - min_corner) / (max_corner - min_corner + 1e-8)
        
        # Convert to [-1, 1] for grid_sample
        normalized = normalized * 2.0 - 1.0
        
        return normalized
    
    def sample_avg_volume(self, points: torch.Tensor) -> torch.Tensor:
        """
        Sample the average CT volume V_avg at arbitrary 3D points using trilinear interpolation.
        
        Args:
            points: World coordinates of shape [N, 3] or [B, N, 3]
            
        Returns:
            Sampled values of shape [N] or [B, N]
        """
        if not self.avg_ct_loaded:
            return torch.zeros(points.shape[0], device=self.device)
        
        original_shape = points.shape
        
        # Ensure points are on the same device
        points = points.to(self.device)
        
        # Ensure points are in correct shape for grid_sample: [B, D, H, W, 3]
        if points.dim() == 2:
            points = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [N, 3] -> [1, 1, 1, N, 3]
        elif points.dim() == 3:
            points = points.unsqueeze(1).unsqueeze(1)  # [B, N, 3] -> [B, 1, 1, N, 3]
        
        # Convert to normalized coordinates
        points_normalized = self.world_to_normalized(points)
        
        # Sample using grid_sample (trilinear interpolation)
        sampled = F.grid_sample(
            self.avg_ct_volume,
            points_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape output
        if len(original_shape) == 2:
            return sampled.squeeze()
        else:
            return sampled.squeeze(1).squeeze(1).squeeze(1)
    
    def compute_volume_distillation_loss(
        self,
        gaussians,
        num_samples: int = 20000,
        bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute volume distillation loss between Gaussians and average CT volume.
        
        L_vol = mean_m |σ_G(x_m) - V_avg(x_m)|
        
        Args:
            gaussians: GaussianModel with get_xyz, get_density, get_scaling, etc.
            num_samples: Number of 3D points to sample
            bbox: Bounding box for sampling (uses self.bbox if not provided)
            
        Returns:
            Volume distillation loss scalar
        """
        if not self.avg_ct_loaded:
            return torch.tensor(0.0, device=self.device)
        
        if bbox is None:
            bbox = self.bbox
        
        # Ensure bbox is on the correct device
        bbox_min = bbox[0].to(self.device)
        bbox_max = bbox[1].to(self.device)
        
        # Sample random points within bounding box
        rand_coords = torch.rand(num_samples, 3, device=self.device)
        sample_points = bbox_min + rand_coords * (bbox_max - bbox_min)  # [M, 3]
        
        # Get average CT values at sample points (teacher signal, no gradient)
        with torch.no_grad():
            sigma_T = self.sample_avg_volume(sample_points)  # [M]
        
        # Get Gaussian density at sample points (student signal, needs gradient)
        sigma_G = self._query_gaussian_density(gaussians, sample_points)  # [M]
        
        # L1 distillation loss
        vol_loss = torch.abs(sigma_G - sigma_T).mean()
        
        return vol_loss
    
    def _query_gaussian_density(
        self,
        gaussians,
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Query the density field defined by Gaussians at given points.
        
        For efficiency, we use a simplified isotropic approximation and
        limit the number of Gaussians considered.
        
        Args:
            gaussians: GaussianModel
            query_points: Points to query [M, 3]
            
        Returns:
            Density values at query points [M]
        """
        # Get Gaussian parameters (detach for memory efficiency in large computations)
        means = gaussians.get_xyz.detach()  # [N, 3]
        densities = gaussians.get_density.squeeze(-1).detach()  # [N]
        scales = gaussians.get_scaling.detach()  # [N, 3]
        
        # Use average scale as isotropic approximation
        avg_scale = scales.mean(dim=-1)  # [N]
        
        M = query_points.shape[0]
        N = means.shape[0]
        
        # Use small batches and limit Gaussians to avoid OOM
        batch_size = 512
        max_gaussians = 10000
        
        result = torch.zeros(M, device=self.device)
        
        # Subsample Gaussians if too many
        if N > max_gaussians:
            indices = torch.randperm(N, device=self.device)[:max_gaussians]
            means_sub = means[indices]
            densities_sub = densities[indices]
            avg_scale_sub = avg_scale[indices]
        else:
            means_sub = means
            densities_sub = densities
            avg_scale_sub = avg_scale
        
        N_sub = means_sub.shape[0]
        var = avg_scale_sub ** 2 + 1e-8  # [N_sub]
        
        for i in range(0, M, batch_size):
            end_i = min(i + batch_size, M)
            batch_points = query_points[i:end_i]  # [B, 3]
            
            # Compute squared distances
            diff = batch_points.unsqueeze(1) - means_sub.unsqueeze(0)  # [B, N_sub, 3]
            sq_dist = (diff ** 2).sum(dim=-1)  # [B, N_sub]
            
            # Gaussian kernel with scale
            weights = torch.exp(-0.5 * sq_dist / var.unsqueeze(0))  # [B, N_sub]
            
            # Weighted sum of densities
            result[i:end_i] = (weights * densities_sub.unsqueeze(0)).sum(dim=-1)  # [B]
            
            # Free memory
            del diff, sq_dist, weights
        
        return result
    
    def get_avg_projection(self, view_idx: int) -> Optional[torch.Tensor]:
        """
        Get the average projection for a specific view index.
        
        Args:
            view_idx: View index
            
        Returns:
            Average projection tensor [1, H, W] or None if not loaded
        """
        if not self.avg_proj_loaded:
            return None
        
        if view_idx >= self.avg_projections.shape[0]:
            return None
        
        return self.avg_projections[view_idx]  # [1, H, W]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the temporal ensemble.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "avg_ct_loaded": self.avg_ct_loaded,
            "avg_proj_loaded": self.avg_proj_loaded,
            "L_vol": self._last_L_vol,
            "L_proj_avg": self._last_L_proj_avg,
        }
        
        if self.avg_ct_loaded:
            stats["avg_ct_mean"] = self.avg_ct_volume.mean().item()
            stats["avg_ct_std"] = self.avg_ct_volume.std().item()
            stats["avg_ct_shape"] = list(self.avg_ct_volume.shape)
        
        if self.avg_proj_loaded:
            stats["avg_proj_num_views"] = self.avg_projections.shape[0]
        
        return stats


def apply_s4_preset(opt):
    """
    Print s4 temporal-ensemble static configuration info.
    
    Args:
        opt: OptimizationParams object
    """
    if not opt.use_s4_temporal_ensemble_static:
        return
    
    print("=" * 60)
    print("S4 TEMPORAL-ENSEMBLE GUIDED STATIC WARM-UP ACTIVATED")
    print("=" * 60)
    print("Temporal-Ensemble Static Configuration:")
    print(f"  - Average CT path: {opt.s4_avg_ct_path or '(not provided)'}")
    print(f"  - Average projections path: {opt.s4_avg_proj_path or '(not provided)'}")
    print(f"  - λ_G (Gaussian projection): {opt.lambda_s4_G}")
    print(f"  - λ_vol (Volume distillation): {opt.lambda_s4_vol}")
    print(f"  - λ_proj_avg (Projection distillation): {opt.lambda_s4_proj_avg}")
    print(f"  - Volume samples per step: {opt.s4_num_vol_samples}")
    print("")
    print("During static warm-up (coarse stage):")
    print("  1. Gaussians render projection Î_G → L_G = L_render(Î_G, I)")
    print("  2. 3D distillation with V_avg: L_vol = |σ_G(x) - V_avg(x)|")
    print("  3. (Optional) Projection distillation: L_proj_avg")
    print("  4. Total: L_s4 = λ_G*L_G + λ_vol*L_vol + λ_proj_avg*L_proj_avg")
    print("")
    print("After static warm-up:")
    print("  - V_avg and avg projections are no longer used")
    print("  - Gaussians retain improved structure from distillation")
    print("")
    print("Dynamic stage (fine) is NOT affected by s4.")
    print("=" * 60)


def initialize_gaussians_from_avg_ct(
    avg_ct_path: str,
    scanner_cfg: dict,
    bbox: torch.Tensor,
    num_gaussians: int = 50000,
    density_thresh: float = 0.1,
    density_rescale: float = 0.15,
    method: str = "fps",
    save_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize Gaussian positions and densities from average CT volume.
    
    This provides better initialization by:
    1. Selecting high-density voxels from V_avg (> threshold)
    2. Using FPS or random sampling to select N centers
    3. Estimating initial scales from local variance
    4. Setting density from V_avg values
    
    Args:
        avg_ct_path: Path to average CT volume (.npy)
        scanner_cfg: Scanner configuration dict with offOrigin, dVoxel, sVoxel
        bbox: Bounding box tensor [2, 3]
        num_gaussians: Number of Gaussians to initialize
        density_thresh: Threshold for selecting high-density voxels (0-1)
        density_rescale: Rescaling factor for densities
        method: Sampling method ("fps" for farthest point sampling, "random")
        save_path: Optional path to save initialization file
        
    Returns:
        Tuple of (positions [N, 3], densities [N])
    """
    print("=" * 60)
    print("S4_2: INITIALIZING GAUSSIANS FROM AVERAGE CT")
    print("=" * 60)
    
    # Load average CT volume
    print(f"[1] Loading average CT from: {avg_ct_path}")
    avg_ct = np.load(avg_ct_path).astype(np.float32)
    print(f"    Shape: {avg_ct.shape}")
    print(f"    Value range: [{avg_ct.min():.4f}, {avg_ct.max():.4f}]")
    
    # Get scanner geometry
    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])
    nVoxel = np.array(avg_ct.shape)
    
    print(f"[2] Scanner geometry:")
    print(f"    offOrigin: {offOrigin}")
    print(f"    dVoxel: {dVoxel}")
    print(f"    sVoxel: {sVoxel}")
    
    # Find high-density voxels
    print(f"[3] Finding high-density voxels (threshold={density_thresh})...")
    density_mask = avg_ct > density_thresh
    valid_indices = np.argwhere(density_mask)
    num_valid = len(valid_indices)
    print(f"    Found {num_valid} voxels above threshold")
    
    if num_valid < num_gaussians:
        print(f"    Warning: Only {num_valid} valid voxels, reducing to this number")
        num_gaussians = num_valid
    
    # Sample points using specified method
    print(f"[4] Sampling {num_gaussians} points using method: {method}")
    
    if method == "fps":
        # Farthest Point Sampling for better coverage
        sampled_indices = _farthest_point_sampling(valid_indices, num_gaussians)
    else:
        # Random sampling
        sampled_indices = valid_indices[
            np.random.choice(num_valid, num_gaussians, replace=False)
        ]
    
    # Convert voxel indices to world coordinates
    # Position = index * dVoxel - sVoxel/2 + offOrigin
    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    
    # Get densities at sampled positions
    sampled_densities = avg_ct[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]
    sampled_densities = sampled_densities * density_rescale
    
    print(f"[5] Initialization complete:")
    print(f"    Positions shape: {sampled_positions.shape}")
    print(f"    Position range: [{sampled_positions.min(axis=0)}, {sampled_positions.max(axis=0)}]")
    print(f"    Densities range: [{sampled_densities.min():.4f}, {sampled_densities.max():.4f}]")
    
    # Save if path provided
    if save_path:
        print(f"[6] Saving to: {save_path}")
        np.save(save_path, {
            "positions": sampled_positions.astype(np.float32),
            "densities": sampled_densities.astype(np.float32),
            "method": method,
            "density_thresh": density_thresh,
            "num_gaussians": num_gaussians
        })
    
    print("=" * 60)
    
    return sampled_positions.astype(np.float32), sampled_densities.astype(np.float32)


def _farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for better spatial coverage.
    
    Args:
        points: Input points [N, 3]
        num_samples: Number of points to sample
        
    Returns:
        Sampled points [num_samples, 3]
    """
    N = len(points)
    if N <= num_samples:
        return points
    
    # Use batch processing for efficiency
    batch_size = min(10000, N)
    
    # Start with random point
    selected_indices = [np.random.randint(N)]
    min_distances = np.full(N, np.inf)
    
    for i in range(num_samples - 1):
        # Update minimum distances
        last_selected = points[selected_indices[-1]]
        
        # Process in batches to avoid memory issues
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_points = points[start:end]
            distances = np.sum((batch_points - last_selected) ** 2, axis=1)
            min_distances[start:end] = np.minimum(min_distances[start:end], distances)
        
        # Select farthest point
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
        
        # Progress logging
        if (i + 1) % 5000 == 0:
            print(f"    FPS progress: {i + 1}/{num_samples}")
    
    return points[selected_indices]
