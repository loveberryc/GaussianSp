import itertools
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(
    grid: torch.Tensor,
    coords: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim in (2, 3):
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only implemented for 2 and 3D data."
        )

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    batch, feature_dim = grid.shape[:2]
    num_samples = coords.shape[-2]
    interp = grid_sampler(
        grid,
        coords,
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
    interp = interp.view(batch, feature_dim, num_samples).transpose(-1, -2)
    return interp.squeeze()


class LegacyHexPlaneField(nn.Module):
    """Original HexPlane implementation from X2-Gaussian."""

    def __init__(self, bounds, planeconfig, multires) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            grid_planes = self._init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            if self.concat_features:
                self.feat_dim += grid_planes[-1].shape[1]
            else:
                self.feat_dim = grid_planes[-1].shape[1]
            self.grids.append(grid_planes)
        print("feature_dim:", self.feat_dim)

    @staticmethod
    def _init_grid_param(grid_nd, in_dim, out_dim, reso, a=0.1, b=0.5):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time_planes = in_dim == 4
        assert grid_nd <= in_dim
        coordinate_combinations = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for combination in coordinate_combinations:
            new_grid = nn.Parameter(torch.empty([1, out_dim] + [reso[idx] for idx in combination[::-1]]))
            if has_time_planes and 3 in combination:
                nn.init.ones_(new_grid)
            else:
                nn.init.uniform_(new_grid, a=a, b=b)
            grid_coefs.append(new_grid)
        return grid_coefs

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb=", self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros_like(pts[..., :1])
        pts = torch.cat((pts, timestamps), dim=-1)
        pts = pts.reshape(-1, pts.shape[-1])

        multi_scale_interp = [] if self.concat_features else 0.0
        coordinate_combinations = list(itertools.combinations(range(pts.shape[-1]), self.grid_config[0]["grid_dimensions"]))
        for grid in self.grids:
            interp_space = 1.0
            for idx, combination in enumerate(coordinate_combinations):
                feature_dim = grid[idx].shape[1]
                interp_out = grid_sample_wrapper(grid[idx], pts[..., combination]).view(-1, feature_dim)
                interp_space = interp_space * interp_out
            if self.concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        if len(multi_scale_interp) < 1:
            multi_scale_interp = torch.zeros((0, 1)).to(pts.device)

        return multi_scale_interp

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        return self.get_density(pts, timestamps)


class OrthogonalVolumeLevel(nn.Module):
    """Four orthogonal 3D volumes (xyz, xyt, xzt, yzt)."""

    def __init__(
        self,
        out_dim: int,
        spatial_resolution: Sequence[int],
        time_resolution: int,
        init_time_to_one: bool = True,
        init_range: Sequence[float] = (0.1, 0.5),
    ) -> None:
        super().__init__()
        sx, sy, sz = spatial_resolution
        tz = time_resolution

        self.static_xyz = nn.Parameter(
            torch.empty(1, out_dim, max(sz, 2), max(sy, 2), max(sx, 2))
        )
        self.xyt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sy, 2), max(sx, 2))
        )
        self.xzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sx, 2))
        )
        self.yzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sy, 2))
        )

        self.reset_parameters(init_time_to_one, init_range)

    def reset_parameters(self, init_time_to_one: bool, init_range: Sequence[float]):
        a, b = init_range
        nn.init.uniform_(self.static_xyz, a=a, b=b)
        if init_time_to_one:
            nn.init.ones_(self.xyt)
            nn.init.ones_(self.xzt)
            nn.init.ones_(self.yzt)
        else:
            nn.init.uniform_(self.xyt, a=a, b=b)
            nn.init.uniform_(self.xzt, a=a, b=b)
            nn.init.uniform_(self.yzt, a=a, b=b)


class HexPlaneStaticResidualField(nn.Module):
    """HexPlane with Static-Residual decomposition and TARS.
    
    Core Innovation:
    - Static spatial planes: xy, xz, yz (capture static structure)
    - Residual temporal planes: xt, yt, zt (capture dynamic changes)
    - TARS: Time-aware adaptive weighting on temporal planes
    - Static Prior: Initialize spatial planes from mean CT
    
    This preserves HexPlane's excellent representation power while adding
    static/dynamic separation and adaptive temporal weighting.
    """
    
    def __init__(self, bounds, planeconfig, multires, static_prior=None) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        
        # Configuration
        resolution = planeconfig["resolution"]
        self.output_coordinate_dim = planeconfig["output_coordinate_dim"]
        self.residual_weight = planeconfig.get("residual_weight", 1.0)
        self.use_residual_clamp = planeconfig.get("use_residual_clamp", False)
        self.residual_clamp_value = planeconfig.get("residual_clamp_value", 2.0)
        
        # Static prior initialization
        self.use_static_prior = planeconfig.get("use_static_prior", False)
        self.static_prior_list = None
        if static_prior is not None and self.use_static_prior:
            print("[HexPlane-SR] Using provided static prior for spatial planes initialization")
            self.static_prior_list = static_prior
        
        # Time-Aware Adaptive Residual Sparsification (TARS)
        self.use_time_aware_residual = planeconfig.get("use_time_aware_residual", False)
        if self.use_time_aware_residual:
            time_steps = planeconfig.get("max_time_resolution", resolution[3])
            self.time_weights = nn.Parameter(
                torch.ones(time_steps, dtype=torch.float32),
                requires_grad=True
            )
            print(f"[HexPlane-SR-TARS] Time-aware residual enabled with {time_steps} learnable time weights")
        
        time_bounds = planeconfig.get("time_bounds", [0.0, 1.0])
        self.register_buffer(
            "time_bounds",
            torch.tensor(time_bounds, dtype=torch.float32),
            persistent=False,
        )
        
        # Initialize multi-scale grids
        self.static_grids = nn.ModuleList()  # spatial planes: xy, xz, yz
        self.residual_grids = nn.ModuleList()  # temporal planes: xt, yt, zt
        self.static_fusion_layers = nn.ModuleList()
        self.residual_fusion_layers = nn.ModuleList()
        self.feat_dim = 0
        
        for level_idx, res in enumerate(self.multiscale_res_multipliers):
            config = self.grid_config[0].copy()
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            
            # Static spatial planes: xy(0), xz(1), yz(3)
            static_planes, static_indices = self._init_static_planes(config, level_idx)
            self.static_grids.append(static_planes)
            
            # Residual temporal planes: xt(2), yt(4), zt(5)
            residual_planes = self._init_residual_planes(config)
            self.residual_grids.append(residual_planes)
            
            # Fusion layers
            static_fusion = nn.Linear(
                self.output_coordinate_dim,
                self.output_coordinate_dim,
                bias=True,
            )
            nn.init.xavier_uniform_(static_fusion.weight)
            if static_fusion.bias is not None:
                nn.init.zeros_(static_fusion.bias)
            self.static_fusion_layers.append(static_fusion)
            
            residual_fusion = nn.Linear(
                self.output_coordinate_dim,
                self.output_coordinate_dim,
                bias=True,
            )
            nn.init.xavier_uniform_(residual_fusion.weight)
            if residual_fusion.bias is not None:
                nn.init.zeros_(residual_fusion.bias)
            self.residual_fusion_layers.append(residual_fusion)
            
            if self.concat_features:
                self.feat_dim += self.output_coordinate_dim
            else:
                self.feat_dim = self.output_coordinate_dim
        
        print(f"[HexPlane-SR] feature_dim: {self.feat_dim}")
        print(f"  - Static planes: xy, xz, yz (spatial)")
        print(f"  - Residual planes: xt, yt, zt (temporal)")
        print(f"  - Residual weight: {self.residual_weight}")
    
    def _init_static_planes(self, config, level_idx):
        """Initialize static spatial planes (xy, xz, yz) with optional prior."""
        grid_nd = config["grid_dimensions"]
        out_dim = config["output_coordinate_dim"]
        reso = config["resolution"]
        
        # Static planes: combinations without time (index 3)
        # (0,1)=xy, (0,2)=xz, (1,2)=yz
        static_combinations = [(0, 1), (0, 2), (1, 2)]
        
        grid_planes = nn.ParameterList()
        for comb_idx, combination in enumerate(static_combinations):
            new_grid = nn.Parameter(
                torch.empty([1, out_dim] + [reso[idx] for idx in combination[::-1]])
            )
            
            # Initialize from static prior if available
            if self.static_prior_list is not None and level_idx < len(self.static_prior_list):
                prior = self.static_prior_list[level_idx]  # (1, 1, D, H, W)
                # Extract 2D slice from 3D prior for each spatial plane
                if combination == (0, 1):  # xy plane - average along z
                    prior_2d = prior.mean(dim=2)  # (1, 1, H, W)
                elif combination == (0, 2):  # xz plane - average along y
                    prior_2d = prior.mean(dim=3)  # (1, 1, D, W)
                elif combination == (1, 2):  # yz plane - average along x
                    prior_2d = prior.mean(dim=4)  # (1, 1, D, H)
                
                # Resize to match grid resolution
                target_shape = new_grid.shape[2:]
                if prior_2d.shape[2:] != target_shape:
                    prior_2d = F.interpolate(
                        prior_2d,
                        size=target_shape,
                        mode='bilinear',
                        align_corners=True
                    )
                
                # Expand to output_coordinate_dim channels
                prior_expanded = prior_2d.expand(1, out_dim, *target_shape).clone()
                new_grid.data.copy_(prior_expanded)
                print(f"    [Level {level_idx}] {combination} initialized from static prior")
            else:
                # Random initialization
                nn.init.uniform_(new_grid, a=0.1, b=0.5)
            
            grid_planes.append(new_grid)
        
        return grid_planes, static_combinations
    
    def _init_residual_planes(self, config):
        """Initialize residual temporal planes (xt, yt, zt) with small values."""
        grid_nd = config["grid_dimensions"]
        out_dim = config["output_coordinate_dim"]
        reso = config["resolution"]
        
        # Residual planes: combinations with time (index 3)
        # (0,3)=xt, (1,3)=yt, (2,3)=zt
        residual_combinations = [(0, 3), (1, 3), (2, 3)]
        
        grid_planes = nn.ParameterList()
        for combination in residual_combinations:
            new_grid = nn.Parameter(
                torch.empty([1, out_dim] + [reso[idx] for idx in combination[::-1]])
            )
            # Initialize to small values (residual should start small)
            nn.init.uniform_(new_grid, a=-0.1, b=0.1)
            grid_planes.append(new_grid)
        
        return grid_planes
    
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    
    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("HexPlane-SR: set aabb=", self.aabb)
    
    def _normalize_time(self, timestamps: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.time_bounds[0], self.time_bounds[1]
        denom = torch.clamp(t_max - t_min, min=1e-6)
        normalized = (timestamps - t_min) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized * 2.0 - 1.0
    
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros_like(pts[..., :1])
        
        pts_flat = pts.reshape(-1, pts.shape[-1])
        timestamps_flat = timestamps.reshape(-1, 1)
        time_norm = self._normalize_time(timestamps_flat)
        
        # Combine xyz and t
        pts_4d = torch.cat([pts_flat, time_norm], dim=-1)  # (N, 4)
        
        multi_scale_features = [] if self.concat_features else 0.0
        
        for static_planes, residual_planes, static_fusion, residual_fusion in zip(
            self.static_grids, self.residual_grids,
            self.static_fusion_layers, self.residual_fusion_layers
        ):
            # Sample static spatial planes (xy, xz, yz)
            # Multiply: xy * xz * yz
            static_feat = 1.0
            static_feat = static_feat * grid_sample_wrapper(static_planes[0], pts_4d[:, [0, 1]])  # xy
            static_feat = static_feat * grid_sample_wrapper(static_planes[1], pts_4d[:, [0, 2]])  # xz
            static_feat = static_feat * grid_sample_wrapper(static_planes[2], pts_4d[:, [1, 2]])  # yz
            static_feat = static_fusion(static_feat.view(-1, self.output_coordinate_dim))
            
            # Sample residual temporal planes (xt, yt, zt)
            # Multiply: xt * yt * zt
            residual_feat = 1.0
            residual_feat = residual_feat * grid_sample_wrapper(residual_planes[0], pts_4d[:, [0, 3]])  # xt
            residual_feat = residual_feat * grid_sample_wrapper(residual_planes[1], pts_4d[:, [1, 3]])  # yt
            residual_feat = residual_feat * grid_sample_wrapper(residual_planes[2], pts_4d[:, [2, 3]])  # zt
            residual_feat = residual_fusion(residual_feat.view(-1, self.output_coordinate_dim))
            
            # Apply TARS time-aware adaptive weighting
            if self.use_time_aware_residual:
                time_01 = (time_norm + 1.0) / 2.0
                time_indices = (time_01 * (self.time_weights.shape[0] - 1)).long()
                time_indices = torch.clamp(time_indices, 0, self.time_weights.shape[0] - 1)
                adaptive_weights = torch.sigmoid(self.time_weights[time_indices])
                residual_feat = residual_feat * adaptive_weights
            
            # Apply global residual weight and optional clamping
            residual_feat = residual_feat * self.residual_weight
            if self.use_residual_clamp:
                residual_feat = torch.clamp(residual_feat, -self.residual_clamp_value, self.residual_clamp_value)
            
            # Combine static and residual
            combined_feat = static_feat + residual_feat
            
            if self.concat_features:
                multi_scale_features.append(combined_feat)
            else:
                multi_scale_features = multi_scale_features + combined_feat
        
        if self.concat_features:
            features = torch.cat(multi_scale_features, dim=-1)
        else:
            features = multi_scale_features
        
        if features.numel() == 0:
            return torch.zeros((0, 1), device=pts.device)
        
        return features.view(-1, self.feat_dim)
    
    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        return self.get_density(pts, timestamps)
    
    def get_time_weights_sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity regularization on time weights."""
        if not self.use_time_aware_residual:
            return torch.tensor(0.0, device=self.time_weights.device if hasattr(self, 'time_weights') else 'cuda')
        return torch.mean(torch.abs(self.time_weights))
    
    def get_time_weights_smoothness_loss(self) -> torch.Tensor:
        """Temporal smoothness regularization."""
        if not self.use_time_aware_residual:
            return torch.tensor(0.0, device=self.time_weights.device if hasattr(self, 'time_weights') else 'cuda')
        diff = self.time_weights[1:] - self.time_weights[:-1]
        return torch.mean(diff ** 2)
    
    def get_time_weights_visualization(self) -> torch.Tensor:
        """Get time weights for visualization."""
        if not self.use_time_aware_residual:
            return None
        return torch.sigmoid(self.time_weights).detach().cpu()


class FourOrthogonalVolumeField(nn.Module):
    """Four orthogonal volume decomposition used by the STNF4D variant."""

    def __init__(self, bounds, planeconfig, multires) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        resolution = planeconfig["resolution"]
        assert len(resolution) == 4, "Resolution must be [res_x, res_y, res_z, res_t]"
        self.base_spatial_resolution = resolution[:3]
        self.time_resolution = resolution[3]
        self.output_coordinate_dim = planeconfig["output_coordinate_dim"]
        self.init_time_to_one = planeconfig.get("initialize_time_to_one", True)
        self.max_spatial_resolution = planeconfig.get(
            "max_spatial_resolution", max(self.base_spatial_resolution)
        )
        self.max_time_resolution = planeconfig.get(
            "max_time_resolution", self.time_resolution
        )
        time_bounds = planeconfig.get("time_bounds", [0.0, 1.0])
        if len(time_bounds) != 2:
            raise ValueError("time_bounds must contain [t_min, t_max]")
        self.register_buffer(
            "time_bounds",
            torch.tensor(time_bounds, dtype=torch.float32),
            persistent=False,
        )

        self.grids = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            spatial_resolution = [
                max(
                    2,
                    min(
                        int(round(r * res)),
                        int(self.max_spatial_resolution),
                    ),
                )
                for r in self.base_spatial_resolution
            ]
            time_resolution = max(
                2,
                min(
                    int(round(self.time_resolution * res)),
                    int(self.max_time_resolution),
                ),
            )
            level = OrthogonalVolumeLevel(
                out_dim=self.output_coordinate_dim,
                spatial_resolution=spatial_resolution,
                time_resolution=time_resolution,
                init_time_to_one=self.init_time_to_one,
                init_range=(0.1, 0.5),
            )
            self.grids.append(level)
            fusion = nn.Linear(
                self.output_coordinate_dim * 4,
                self.output_coordinate_dim,
                bias=True,
            )
            nn.init.xavier_uniform_(fusion.weight)
            if fusion.bias is not None:
                nn.init.zeros_(fusion.bias)
            self.fusion_layers.append(fusion)

            if self.concat_features:
                self.feat_dim += self.output_coordinate_dim
            else:
                self.feat_dim = self.output_coordinate_dim

        print("feature_dim:", self.feat_dim)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb=", self.aabb)

    def _normalize_time(self, timestamps: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.time_bounds[0], self.time_bounds[1]
        denom = torch.clamp(t_max - t_min, min=1e-6)
        normalized = (timestamps - t_min) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized * 2.0 - 1.0

    def _sample_static(self, grid: torch.Tensor, pts_normalized: torch.Tensor) -> torch.Tensor:
        coords = torch.clamp(pts_normalized, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_xyt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 1], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_xzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_yzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 1], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros_like(pts[..., :1])

        pts_flat = pts.reshape(-1, pts.shape[-1])
        timestamps_flat = timestamps.reshape(-1, 1)
        time_norm = self._normalize_time(timestamps_flat)

        multi_scale_features = [] if self.concat_features else 0.0
        for level, fusion in zip(self.grids, self.fusion_layers):
            static_feat = self._sample_static(level.static_xyz, pts_flat)
            xyt_feat = self._sample_xyt(level.xyt, pts_flat, time_norm)
            xzt_feat = self._sample_xzt(level.xzt, pts_flat, time_norm)
            yzt_feat = self._sample_yzt(level.yzt, pts_flat, time_norm)

            combined = torch.cat([static_feat, xyt_feat, xzt_feat, yzt_feat], dim=-1)
            fused = fusion(combined)
            if self.concat_features:
                multi_scale_features.append(fused)
            else:
                multi_scale_features = multi_scale_features + fused

        if self.concat_features:
            features = torch.cat(multi_scale_features, dim=-1)
        else:
            features = multi_scale_features

        if features.numel() == 0:
            return torch.zeros((0, 1), device=pts.device)

        return features.view(-1, self.feat_dim)

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        return self.get_density(pts, timestamps)


class ResidualVolumeLevel(nn.Module):
    """Three temporal orthogonal 3D volumes (xyt, xzt, yzt) for residuals only.
    No static xyz component."""

    def __init__(
        self,
        out_dim: int,
        spatial_resolution: Sequence[int],
        time_resolution: int,
        init_range: Sequence[float] = (0.0, 0.1),
    ) -> None:
        super().__init__()
        sx, sy, sz = spatial_resolution
        tz = time_resolution

        # Only temporal volumes, no static xyz
        self.xyt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sy, 2), max(sx, 2))
        )
        self.xzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sx, 2))
        )
        self.yzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sy, 2))
        )

        self.reset_parameters(init_range)

    def reset_parameters(self, init_range: Sequence[float]):
        a, b = init_range
        # Initialize to small values near zero for residuals
        nn.init.uniform_(self.xyt, a=a, b=b)
        nn.init.uniform_(self.xzt, a=a, b=b)
        nn.init.uniform_(self.yzt, a=a, b=b)


class StaticPlusResidualVolumeField(nn.Module):
    """Static 3D volume prior + low-resolution residual temporal volumes.
    
    This hybrid approach reduces memory by:
    1. Using a single static 3D volume (xyz) at moderate resolution for static structure
    2. Using low-resolution temporal volumes (xyt, xzt, yzt ONLY) for dynamic residuals
    3. NOT duplicating static_xyz in the residual part
    
    Optional: Can be initialized with a mean CT prior from training data for faster convergence.
    """

    def __init__(self, bounds, planeconfig, multires, static_prior=None) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        resolution = planeconfig["resolution"]
        assert len(resolution) == 4, "Resolution must be [res_x, res_y, res_z, res_t]"
        self.base_spatial_resolution = resolution[:3]
        self.time_resolution = resolution[3]
        self.output_coordinate_dim = planeconfig["output_coordinate_dim"]
        
        # Static volume can use moderate resolution
        self.static_resolution_multiplier = planeconfig.get("static_resolution_multiplier", 1.0)
        # Residual volumes use lower resolution to save memory
        self.residual_resolution_multiplier = planeconfig.get("residual_resolution_multiplier", 0.5)
        
        # Optional static prior initialization
        self.use_static_prior = planeconfig.get("use_static_prior", False)
        self.static_prior_list = None
        if static_prior is not None and self.use_static_prior:
            print("[StaticPlusResidual] Using provided static prior for initialization")
            self.static_prior_list = static_prior
        
        self.max_spatial_resolution = planeconfig.get(
            "max_spatial_resolution", max(self.base_spatial_resolution)
        )
        self.max_time_resolution = planeconfig.get(
            "max_time_resolution", self.time_resolution
        )
        
        # Residual weight controls how much the residual contributes
        self.residual_weight = planeconfig.get("residual_weight", 1.0)
        self.use_residual_clamp = planeconfig.get("use_residual_clamp", False)
        self.residual_clamp_value = planeconfig.get("residual_clamp_value", 2.0)
        
        # Time-Aware Adaptive Residual Sparsification (TARS)
        self.use_time_aware_residual = planeconfig.get("use_time_aware_residual", False)
        if self.use_time_aware_residual:
            # Learnable time-dependent weights α(t) ∈ [0, 1] for each time step
            # Initialize to 1.0 (full residual) and let network learn to sparsify
            time_steps = planeconfig.get("max_time_resolution", self.time_resolution)
            self.time_weights = nn.Parameter(
                torch.ones(time_steps, dtype=torch.float32),
                requires_grad=True
            )
            print(f"[TARS] Time-aware residual enabled with {time_steps} learnable time weights")
        
        time_bounds = planeconfig.get("time_bounds", [0.0, 1.0])
        if len(time_bounds) != 2:
            raise ValueError("time_bounds must contain [t_min, t_max]")
        self.register_buffer(
            "time_bounds",
            torch.tensor(time_bounds, dtype=torch.float32),
            persistent=False,
        )

        self.static_grids = nn.ParameterList()  # Use ParameterList for parameters
        self.residual_grids = nn.ModuleList()
        self.static_fusion_layers = nn.ModuleList()
        self.residual_fusion_layers = nn.ModuleList()
        self.feat_dim = 0
        
        for res in self.multiscale_res_multipliers:
            # Static 3D volume at higher resolution
            static_spatial_resolution = [
                max(
                    2,
                    min(
                        int(round(r * res * self.static_resolution_multiplier)),
                        int(self.max_spatial_resolution * self.static_resolution_multiplier),
                    ),
                )
                for r in self.base_spatial_resolution
            ]
            
            # Create static 3D volume (xyz only)
            sx, sy, sz = static_spatial_resolution
            static_volume = nn.Parameter(
                torch.empty(1, self.output_coordinate_dim, max(sz, 2), max(sy, 2), max(sx, 2))
            )
            
            # Initialize from prior if available, otherwise random
            if self.static_prior_list is not None and len(self.static_prior_list) > len(self.static_grids):
                prior = self.static_prior_list[len(self.static_grids)]
                # Resize prior to match current resolution if needed
                if prior.shape[2:] != static_volume.shape[2:]:
                    prior = torch.nn.functional.interpolate(
                        prior, 
                        size=static_volume.shape[2:], 
                        mode='trilinear', 
                        align_corners=True
                    )
                # Expand to match feature channels
                prior_expanded = prior.expand_as(static_volume).clone()
                static_volume.data.copy_(prior_expanded)
                print(f"  [Level {len(self.static_grids)}] Initialized from static prior: {prior.shape} -> {static_volume.shape}")
            else:
                nn.init.uniform_(static_volume, a=0.1, b=0.5)
            
            self.static_grids.append(static_volume)
            
            # Static fusion layer (just identity mapping since it's a single volume)
            static_fusion = nn.Linear(
                self.output_coordinate_dim,
                self.output_coordinate_dim,
                bias=True,
            )
            nn.init.xavier_uniform_(static_fusion.weight)
            if static_fusion.bias is not None:
                nn.init.zeros_(static_fusion.bias)
            self.static_fusion_layers.append(static_fusion)
            
            # Residual temporal volumes at lower resolution (NO static xyz)
            residual_spatial_resolution = [
                max(
                    2,
                    min(
                        int(round(r * res * self.residual_resolution_multiplier)),
                        int(self.max_spatial_resolution * self.residual_resolution_multiplier),
                    ),
                )
                for r in self.base_spatial_resolution
            ]
            residual_time_resolution = max(
                2,
                min(
                    int(round(self.time_resolution * res * self.residual_resolution_multiplier)),
                    int(self.max_time_resolution),
                ),
            )
            
            # Use ResidualVolumeLevel (only xyt, xzt, yzt, no static xyz)
            residual_level = ResidualVolumeLevel(
                out_dim=self.output_coordinate_dim,
                spatial_resolution=residual_spatial_resolution,
                time_resolution=residual_time_resolution,
                init_range=(-0.1, 0.1),  # Smaller init for residuals, can be negative
            )
            self.residual_grids.append(residual_level)
            
            # Residual fusion layer (only 3 temporal volumes)
            residual_fusion = nn.Linear(
                self.output_coordinate_dim * 3,  # xyt + xzt + yzt (no static xyz)
                self.output_coordinate_dim,
                bias=True,
            )
            nn.init.xavier_uniform_(residual_fusion.weight)
            if residual_fusion.bias is not None:
                nn.init.zeros_(residual_fusion.bias)
            self.residual_fusion_layers.append(residual_fusion)

            if self.concat_features:
                self.feat_dim += self.output_coordinate_dim
            else:
                self.feat_dim = self.output_coordinate_dim

        print(f"[StaticPlusResidual] feature_dim: {self.feat_dim}")
        print(f"  - Static resolution multiplier: {self.static_resolution_multiplier}x")
        print(f"  - Residual resolution multiplier: {self.residual_resolution_multiplier}x")
        print(f"  - Residual weight: {self.residual_weight}")

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min], dtype=torch.float32)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb=", self.aabb)

    def _normalize_time(self, timestamps: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.time_bounds[0], self.time_bounds[1]
        denom = torch.clamp(t_max - t_min, min=1e-6)
        normalized = (timestamps - t_min) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized * 2.0 - 1.0

    def _sample_static_3d(self, grid: torch.Tensor, pts_normalized: torch.Tensor) -> torch.Tensor:
        coords = torch.clamp(pts_normalized, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_xyt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 1], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_xzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def _sample_yzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 1], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords)
        return sampled.view(-1, grid.shape[1])

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros_like(pts[..., :1])

        pts_flat = pts.reshape(-1, pts.shape[-1])
        timestamps_flat = timestamps.reshape(-1, 1)
        time_norm = self._normalize_time(timestamps_flat)

        multi_scale_features = [] if self.concat_features else 0.0
        
        for static_vol, static_fusion, residual_level, residual_fusion in zip(
            self.static_grids, self.static_fusion_layers, 
            self.residual_grids, self.residual_fusion_layers
        ):
            # Sample static 3D volume
            static_feat = self._sample_static_3d(static_vol, pts_flat)
            static_feat = static_fusion(static_feat)
            
            # Sample residual temporal volumes (only xyt, xzt, yzt)
            xyt_feat = self._sample_xyt(residual_level.xyt, pts_flat, time_norm)
            xzt_feat = self._sample_xzt(residual_level.xzt, pts_flat, time_norm)
            yzt_feat = self._sample_yzt(residual_level.yzt, pts_flat, time_norm)
            
            residual_combined = torch.cat([xyt_feat, xzt_feat, yzt_feat], dim=-1)
            residual_feat = residual_fusion(residual_combined)
            
            # Apply time-aware adaptive weighting (TARS)
            if self.use_time_aware_residual:
                # Get time weights for current timestamps
                # time_norm is in [-1, 1], convert to [0, 1] for indexing
                time_01 = (time_norm + 1.0) / 2.0
                time_indices = (time_01 * (self.time_weights.shape[0] - 1)).long()
                time_indices = torch.clamp(time_indices, 0, self.time_weights.shape[0] - 1)
                
                # Get adaptive weights α(t) and apply sigmoid to constrain to [0, 1]
                adaptive_weights = torch.sigmoid(self.time_weights[time_indices])
                residual_feat = residual_feat * adaptive_weights
            
            # Apply global residual weight and optional clamping
            residual_feat = residual_feat * self.residual_weight
            if self.use_residual_clamp:
                residual_feat = torch.clamp(residual_feat, -self.residual_clamp_value, self.residual_clamp_value)
            
            # Combine static and residual
            combined_feat = static_feat + residual_feat
            
            if self.concat_features:
                multi_scale_features.append(combined_feat)
            else:
                multi_scale_features = multi_scale_features + combined_feat

        if self.concat_features:
            features = torch.cat(multi_scale_features, dim=-1)
        else:
            features = multi_scale_features

        if features.numel() == 0:
            return torch.zeros((0, 1), device=pts.device)

        return features.view(-1, self.feat_dim)

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        return self.get_density(pts, timestamps)
    
    def get_time_weights_sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity regularization on time weights to encourage adaptive sparsification."""
        if not self.use_time_aware_residual:
            return torch.tensor(0.0, device=self.time_weights.device if hasattr(self, 'time_weights') else 'cuda')
        
        # L1 loss encourages some weights to be close to 0 (sigmoid domain)
        # Apply to logits before sigmoid for better gradient flow
        return torch.mean(torch.abs(self.time_weights))
    
    def get_time_weights_smoothness_loss(self) -> torch.Tensor:
        """Temporal smoothness regularization to encourage continuous time weights."""
        if not self.use_time_aware_residual:
            return torch.tensor(0.0, device=self.time_weights.device if hasattr(self, 'time_weights') else 'cuda')
        
        # Encourage smooth transitions between adjacent time steps
        diff = self.time_weights[1:] - self.time_weights[:-1]
        return torch.mean(diff ** 2)
    
    def get_time_weights_visualization(self) -> torch.Tensor:
        """Get time weights for visualization (after sigmoid)."""
        if not self.use_time_aware_residual:
            return None
        return torch.sigmoid(self.time_weights).detach().cpu()


def build_feature_grid(mode: str, bounds, planeconfig, multires, static_prior=None):
    normalized_mode = (mode or "four_volume").lower()
    if normalized_mode == "four_volume":
        return FourOrthogonalVolumeField(bounds, planeconfig, multires)
    if normalized_mode == "static_residual_four_volume":
        return StaticPlusResidualVolumeField(bounds, planeconfig, multires, static_prior=static_prior)
    if normalized_mode == "hexplane_sr":
        return HexPlaneStaticResidualField(bounds, planeconfig, multires, static_prior=static_prior)
    if normalized_mode in {"hexplane", "legacy_hexplane", "mlp"}:
        return LegacyHexPlaneField(bounds, planeconfig, multires)
    raise ValueError(f"Unsupported grid mode '{mode}'. Expected one of ['four_volume', 'static_residual_four_volume', 'hexplane_sr', 'hexplane', 'mlp'].")
