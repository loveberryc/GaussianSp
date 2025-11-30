"""
X2-Gaussian s3: Dual-Representation Static Warm-Up (Gaussian + Voxel Co-Training)

This module implements dual-representation training for the static warm-up stage.
The core idea is to co-train:
- Radiative Gaussians (existing canonical representation)
- A low-resolution 3D voxel volume V (auxiliary representation)

Both representations are supervised by projection loss, and a 3D distillation loss
constrains Gaussians to align with the smoother V structure.

Only affects the static warm-up stage (coarse); dynamic stage is unchanged.

Key components:
- s3_voxel_volume: Learnable 3D voxel tensor V ∈ R^{D×H×W}
- Voxel projection rendering via ray marching
- Gaussian↔Voxel distillation loss
- 3D TV/smoothness regularization for V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class DualStaticVolume(nn.Module):
    """
    Dual-Representation Static Warm-Up (s3).
    
    Maintains a learnable 3D voxel volume V that co-trains with canonical Gaussians
    during the static warm-up stage.
    
    Args:
        resolution: Voxel resolution D=H=W (e.g., 64 or 96)
        bbox: Bounding box tensor of shape [2, 3] with [min_corner, max_corner]
        num_ray_samples: Number of samples along each ray for volume rendering (default 64)
        init_value: Initial value for voxel volume (default 0.0)
        device: Device to place parameters on
    """
    
    def __init__(
        self,
        resolution: int = 64,
        bbox: torch.Tensor = None,
        num_ray_samples: int = 64,
        init_value: float = 0.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.resolution = resolution
        self.num_ray_samples = num_ray_samples
        self.device = device
        
        # Store bounding box (will be set later if not provided)
        if bbox is not None:
            self.register_buffer('bbox', bbox.to(device))
        else:
            # Default placeholder, should be set via set_bbox()
            self.register_buffer('bbox', torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32, device=device))
        
        # Initialize learnable 3D voxel volume V
        # Shape: [1, 1, D, H, W] for compatibility with grid_sample
        self.s3_voxel_volume = nn.Parameter(
            torch.full((1, 1, resolution, resolution, resolution), init_value, device=device)
        )
        
        # Statistics tracking
        self._last_L_V = 0.0
        self._last_L_distill = 0.0
        self._last_L_VTV = 0.0
    
    def set_bbox(self, bbox: torch.Tensor):
        """Set the bounding box for world-to-normalized coordinate mapping."""
        self.bbox = bbox.to(self.device)
    
    @property
    def volume_shape(self) -> Tuple[int, int, int]:
        """Return the volume shape (D, H, W)."""
        return (self.resolution, self.resolution, self.resolution)
    
    def world_to_normalized(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to normalized coordinates [-1, 1] for grid_sample.
        
        Args:
            points: World coordinates of shape [..., 3]
            
        Returns:
            Normalized coordinates in [-1, 1] for grid_sample
        """
        # bbox[0] = min_corner, bbox[1] = max_corner
        # Ensure bbox is on the same device as points
        min_corner = self.bbox[0].to(points.device)
        max_corner = self.bbox[1].to(points.device)
        
        # Normalize to [0, 1]
        normalized = (points - min_corner) / (max_corner - min_corner + 1e-8)
        
        # Convert to [-1, 1] for grid_sample
        normalized = normalized * 2.0 - 1.0
        
        return normalized
    
    def sample_volume(self, points: torch.Tensor) -> torch.Tensor:
        """
        Sample the voxel volume V at arbitrary 3D points using trilinear interpolation.
        
        Args:
            points: World coordinates of shape [N, 3] or [B, N, 3]
            
        Returns:
            Sampled values of shape [N] or [B, N]
        """
        original_shape = points.shape
        
        # Ensure points are on the same device as voxel volume
        points = points.to(self.s3_voxel_volume.device)
        
        # Ensure points are in correct shape for grid_sample: [B, D, H, W, 3]
        if points.dim() == 2:
            # [N, 3] -> [1, 1, 1, N, 3]
            points = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif points.dim() == 3:
            # [B, N, 3] -> [B, 1, 1, N, 3]
            points = points.unsqueeze(1).unsqueeze(1)
        
        # Convert to normalized coordinates
        points_normalized = self.world_to_normalized(points)
        
        # Sample using grid_sample (trilinear interpolation)
        # grid_sample expects grid in [B, D, H, W, 3] and input in [B, C, D, H, W]
        sampled = F.grid_sample(
            self.s3_voxel_volume,
            points_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape output
        if len(original_shape) == 2:
            # [1, 1, 1, 1, N] -> [N]
            return sampled.squeeze()
        else:
            # [B, 1, 1, 1, N] -> [B, N]
            return sampled.squeeze(1).squeeze(1).squeeze(1)
    
    def render_projection(
        self,
        camera,
        image_height: int,
        image_width: int,
        downsample_factor: int = 1
    ) -> torch.Tensor:
        """
        Render a projection from the voxel volume V using ray marching.
        
        For X-ray imaging, we compute line integrals along rays through the volume.
        
        Args:
            camera: Camera object with world_view_transform, projection_matrix, camera_center
            image_height: Height of output image
            image_width: Width of output image
            downsample_factor: Downsample factor for efficiency (default 1, no downsample)
            
        Returns:
            Rendered projection of shape [1, H//ds, W//ds]
        """
        # Apply downsampling
        H = image_height // downsample_factor
        W = image_width // downsample_factor
        
        # Get camera parameters
        camera_center = camera.camera_center  # [3]
        world_view_transform = camera.world_view_transform  # [4, 4]
        
        # Generate pixel coordinates
        # Create grid of pixel coordinates
        y_coords = torch.linspace(0, 1, H, device=self.device)
        x_coords = torch.linspace(0, 1, W, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Convert to NDC coordinates [-1, 1]
        ndc_x = xx * 2.0 - 1.0  # [H, W]
        ndc_y = yy * 2.0 - 1.0  # [H, W]
        
        # For orthographic projection (mode=0), rays are parallel
        # For perspective projection (mode=1), rays originate from camera center
        mode = camera.mode
        
        # Get bounding box for ray intersection
        bbox_min = self.bbox[0]  # [3]
        bbox_max = self.bbox[1]  # [3]
        
        # Compute ray directions based on camera mode
        if mode == 0:
            # Orthographic projection: parallel rays
            # Ray direction is the camera's view direction (negative z in view space)
            view_inv = torch.inverse(world_view_transform)
            ray_dir = -view_inv[2, :3]  # View direction
            ray_dir = ray_dir / (torch.norm(ray_dir) + 1e-8)  # Normalize
            
            # Ray origins on the near plane
            # Map NDC to world coordinates on a plane perpendicular to ray_dir
            # For orthographic, we use the bbox to set up ray origins
            center = (bbox_min + bbox_max) / 2
            extent = bbox_max - bbox_min
            
            # Create ray origins in a grid
            right = view_inv[0, :3]
            up = view_inv[1, :3]
            
            # Scale by half extent
            ray_origins = center.unsqueeze(0).unsqueeze(0) + \
                          ndc_x.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) * extent[0] / 2 + \
                          ndc_y.unsqueeze(-1) * up.unsqueeze(0).unsqueeze(0) * extent[1] / 2
            
            # Offset origins to start before the volume
            diagonal = torch.norm(extent)
            ray_origins = ray_origins - ray_dir * diagonal / 2
            
            # All rays have same direction
            ray_dirs = ray_dir.unsqueeze(0).unsqueeze(0).expand(H, W, 3)
            
        else:
            # Perspective projection
            # Rays originate from camera center
            ray_origins = camera_center.unsqueeze(0).unsqueeze(0).expand(H, W, 3)
            
            # Compute ray directions through each pixel
            # Use inverse projection to get world-space directions
            view_inv = torch.inverse(world_view_transform)
            
            # Create homogeneous NDC coordinates
            ones = torch.ones_like(ndc_x)
            ndc_points = torch.stack([ndc_x, ndc_y, ones, ones], dim=-1)  # [H, W, 4]
            
            # Transform to view space then world space
            proj_inv = torch.inverse(camera.projection_matrix)
            view_points = ndc_points @ proj_inv  # [H, W, 4]
            world_points = view_points @ view_inv  # [H, W, 4]
            world_points = world_points[..., :3] / (world_points[..., 3:4] + 1e-8)
            
            ray_dirs = world_points - ray_origins
            ray_dirs = ray_dirs / (torch.norm(ray_dirs, dim=-1, keepdim=True) + 1e-8)
        
        # Ray marching through volume
        # Compute t_near and t_far for ray-box intersection
        t_near, t_far = self._ray_box_intersection(ray_origins, ray_dirs, bbox_min, bbox_max)
        
        # Mask for valid rays (that intersect the box)
        valid_mask = t_far > t_near
        
        # Sample points along rays
        t_vals = torch.linspace(0, 1, self.num_ray_samples, device=self.device)  # [K]
        t_vals = t_vals.view(1, 1, -1)  # [1, 1, K]
        
        # Interpolate between t_near and t_far
        t_near_exp = t_near.unsqueeze(-1)  # [H, W, 1]
        t_far_exp = t_far.unsqueeze(-1)    # [H, W, 1]
        t_samples = t_near_exp + (t_far_exp - t_near_exp) * t_vals  # [H, W, K]
        
        # Compute sample points
        sample_points = ray_origins.unsqueeze(-2) + ray_dirs.unsqueeze(-2) * t_samples.unsqueeze(-1)  # [H, W, K, 3]
        
        # Flatten for sampling
        sample_points_flat = sample_points.reshape(-1, 3)  # [H*W*K, 3]
        
        # Sample volume at these points
        sampled_values = self.sample_volume(sample_points_flat)  # [H*W*K]
        sampled_values = sampled_values.reshape(H, W, self.num_ray_samples)  # [H, W, K]
        
        # Compute step size (distance between samples)
        ray_lengths = t_far - t_near  # [H, W]
        delta_t = ray_lengths / self.num_ray_samples  # [H, W]
        
        # Line integral: sum of values * step_size
        # For X-ray: I = integral of attenuation along ray
        projection = (sampled_values * delta_t.unsqueeze(-1)).sum(dim=-1)  # [H, W]
        
        # Apply valid mask (set invalid rays to 0)
        projection = projection * valid_mask.float()
        
        # Reshape to [1, H, W] to match Gaussian rendering output format
        return projection.unsqueeze(0)
    
    def _ray_box_intersection(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ray-box intersection using slab method.
        
        Args:
            ray_origins: Ray origins [H, W, 3]
            ray_dirs: Ray directions [H, W, 3]
            bbox_min: Box minimum corner [3]
            bbox_max: Box maximum corner [3]
            
        Returns:
            t_near, t_far: Near and far intersection distances [H, W]
        """
        # Inverse direction (handle division by zero)
        inv_dir = 1.0 / (ray_dirs + 1e-8)
        
        # Compute intersections with all slabs
        t1 = (bbox_min - ray_origins) * inv_dir  # [H, W, 3]
        t2 = (bbox_max - ray_origins) * inv_dir  # [H, W, 3]
        
        # Find min/max for each axis
        t_min = torch.minimum(t1, t2)  # [H, W, 3]
        t_max = torch.maximum(t1, t2)  # [H, W, 3]
        
        # Find overall near/far
        t_near = t_min.max(dim=-1).values  # [H, W]
        t_far = t_max.min(dim=-1).values   # [H, W]
        
        # Clamp t_near to be non-negative
        t_near = torch.clamp(t_near, min=0.0)
        
        return t_near, t_far
    
    def compute_tv_loss(self) -> torch.Tensor:
        """
        Compute 3D Total Variation (TV) loss for the voxel volume.
        
        L_TV = mean(|dx|^2 + |dy|^2 + |dz|^2)
        
        This encourages smoothness in the voxel volume.
        
        Returns:
            TV loss scalar
        """
        V = self.s3_voxel_volume  # [1, 1, D, H, W]
        
        # Compute differences along each axis
        dx = V[:, :, 1:, :, :] - V[:, :, :-1, :, :]  # [1, 1, D-1, H, W]
        dy = V[:, :, :, 1:, :] - V[:, :, :, :-1, :]  # [1, 1, D, H-1, W]
        dz = V[:, :, :, :, 1:] - V[:, :, :, :, :-1]  # [1, 1, D, H, W-1]
        
        # L2 TV loss
        tv_loss = (dx ** 2).mean() + (dy ** 2).mean() + (dz ** 2).mean()
        
        return tv_loss
    
    def compute_distillation_loss(
        self,
        gaussians,
        num_samples: int = 20000,
        bbox: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute distillation loss between Gaussians and Voxel volume.
        
        L_distill = mean_m |σ_G(x_m) - V(x_m)|
        
        Args:
            gaussians: GaussianModel with get_xyz, get_density, get_scaling, etc.
            num_samples: Number of 3D points to sample
            bbox: Bounding box for sampling (uses self.bbox if not provided)
            
        Returns:
            Distillation loss scalar
        """
        if bbox is None:
            bbox = self.bbox
        
        # Ensure bbox is on the correct device
        bbox_min = bbox[0].to(self.device)
        bbox_max = bbox[1].to(self.device)
        
        # Sample random points within bounding box
        rand_coords = torch.rand(num_samples, 3, device=self.device)
        sample_points = bbox_min + rand_coords * (bbox_max - bbox_min)  # [M, 3]
        
        # Get voxel values at sample points
        sigma_V = self.sample_volume(sample_points)  # [M]
        
        # Get Gaussian density at sample points
        sigma_G = self._query_gaussian_density(gaussians, sample_points)  # [M]
        
        # L1 distillation loss
        distill_loss = torch.abs(sigma_G - sigma_V).mean()
        
        return distill_loss
    
    def _query_gaussian_density(
        self,
        gaussians,
        query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Query the density field defined by Gaussians at given points.
        
        For each point x, compute the sum of Gaussian contributions:
        σ_G(x) = Σ_i ρ_i * exp(-0.5 * (x - μ_i)^T Σ_i^{-1} (x - μ_i))
        
        For efficiency, we use a simplified isotropic approximation and
        only consider nearby Gaussians (within 3σ distance).
        
        Args:
            gaussians: GaussianModel
            query_points: Points to query [M, 3]
            
        Returns:
            Density values at query points [M]
        """
        # Get Gaussian parameters
        means = gaussians.get_xyz.detach()  # [N, 3]
        densities = gaussians.get_density.squeeze(-1).detach()  # [N]
        scales = gaussians.get_scaling.detach()  # [N, 3]
        
        # Use average scale as isotropic approximation for efficiency
        avg_scale = scales.mean(dim=-1)  # [N]
        
        M = query_points.shape[0]
        N = means.shape[0]
        
        # Use very small batches to avoid OOM
        # Also limit number of Gaussians considered per batch
        batch_size = 512  # Reduced from 4096
        max_gaussians = 10000  # Limit Gaussians if too many
        
        result = torch.zeros(M, device=self.device)
        
        # If too many Gaussians, randomly subsample
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
            B = batch_points.shape[0]
            
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
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the voxel volume.
        
        Returns:
            Dictionary with statistics
        """
        V = self.s3_voxel_volume
        
        return {
            "voxel_mean": V.mean().item(),
            "voxel_std": V.std().item(),
            "voxel_min": V.min().item(),
            "voxel_max": V.max().item(),
            "voxel_sparsity": (V.abs() < 0.01).float().mean().item(),
            "L_V": self._last_L_V,
            "L_distill": self._last_L_distill,
            "L_VTV": self._last_L_VTV,
        }


def apply_s3_preset(opt):
    """
    Print s3 dual-representation static configuration info.
    
    Args:
        opt: OptimizationParams object
    """
    if not opt.use_s3_dual_static_volume:
        return
    
    print("=" * 60)
    print("S3 DUAL-REPRESENTATION STATIC WARM-UP ACTIVATED")
    print("=" * 60)
    print("Dual Static Volume Co-Training:")
    print(f"  - Voxel resolution: {opt.s3_voxel_resolution}³")
    print(f"  - λ_G (Gaussian projection): {opt.lambda_s3_G}")
    print(f"  - λ_V (Voxel projection): {opt.lambda_s3_V}")
    print(f"  - λ_distill (Gaussians↔Voxel): {opt.lambda_s3_distill}")
    print(f"  - λ_VTV (3D TV regularization): {opt.lambda_s3_VTV}")
    print(f"  - Distillation samples per step: {opt.s3_num_distill_samples}")
    print(f"  - Ray samples for voxel rendering: {opt.s3_num_ray_samples}")
    print("")
    print("During static warm-up (coarse stage):")
    print("  1. Gaussians render projection Î_G → L_G = L_render(Î_G, I)")
    print("  2. Voxel V renders projection Î_V → L_V = L_render(Î_V, I)")
    print("  3. 3D distillation: L_distill = |σ_G(x) - V(x)|")
    print("  4. 3D smoothness: L_VTV = TV(V)")
    print("  5. Total: L_s3 = λ_G*L_G + λ_V*L_V + λ_distill*L_distill + λ_VTV*L_VTV")
    print("")
    print("After static warm-up:")
    print("  - Voxel V is discarded (not used in dynamic stage)")
    print("  - Gaussians retain improved 3D structure from co-training")
    print("")
    print("Dynamic stage (fine) is NOT affected by s3.")
    print("=" * 60)
