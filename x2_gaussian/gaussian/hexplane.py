from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only "
            f"implemented for 2 and 3D data."
        )

    if grid.dtype != coords.dtype:
        coords = coords.to(grid.dtype)
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


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
        )  # (D, H, W) -> (z, y, x)
        self.xyt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sy, 2), max(sx, 2))
        )  # (time, y, x)
        self.xzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sx, 2))
        )  # (time, z, x)
        self.yzt = nn.Parameter(
            torch.empty(1, out_dim, max(tz, 2), max(sz, 2), max(sy, 2))
        )  # (time, z, y)

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


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
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

        # 1. Init planes
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
            fusion = nn.Linear(self.output_coordinate_dim * 4, self.output_coordinate_dim, bias=True)
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
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def _normalize_time(self, timestamps: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.time_bounds[0], self.time_bounds[1]
        denom = torch.clamp(t_max - t_min, min=1e-6)
        normalized = (timestamps - t_min) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized * 2.0 - 1.0

    def _sample_static(self, grid: torch.Tensor, pts_normalized: torch.Tensor) -> torch.Tensor:
        coords = torch.clamp(pts_normalized, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords).to(pts_normalized.dtype)
        return sampled.view(-1, grid.shape[1])

    def _sample_xyt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 1], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords).to(pts.dtype)
        return sampled.view(-1, grid.shape[1])

    def _sample_xzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 0], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords).to(pts.dtype)
        return sampled.view(-1, grid.shape[1])

    def _sample_yzt(self, grid: torch.Tensor, pts: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([pts[:, 1], pts[:, 2], time_norm[:, 0]], dim=-1)
        coords = torch.clamp(coords, -1.0, 1.0)
        sampled = grid_sample_wrapper(grid, coords).to(pts.dtype)
        return sampled.view(-1, grid.shape[1])

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the features from four orthogonal volumes."""
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros_like(pts[..., :1])

        orig_shape = pts.shape[:-1]
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

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
