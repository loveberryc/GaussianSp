#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement
import time
import math
import torch.nn.functional as F

sys.path.append("./")

from simple_knn._C import distCUDA2
from x2_gaussian.utils.general_utils import t2a
from x2_gaussian.utils.system_utils import mkdir_p
from x2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)
from x2_gaussian.gaussian.deformation import deform_network
from x2_gaussian.gaussian.regulation import compute_plane_smoothness

EPS = 1e-5


# ============================================================================
# V7.1 Alpha Network Classes for time-dependent α(t)
# ============================================================================

class AlphaNetworkFourier(nn.Module):
    """
    Fourier basis network for time-dependent alpha.
    α(t) = α_0 + Σ (a_k * sin(2πk*t) + b_k * cos(2πk*t))
    Output is clamped to [0, 1].
    """
    def __init__(self, num_freqs=4, alpha_init=0.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.alpha_init = alpha_init
        
        # Learnable coefficients: [a_1, ..., a_K, b_1, ..., b_K]
        # Initialize to small random values
        self.coeffs = nn.Parameter(torch.zeros(num_freqs * 2) * 0.01)
        # Learnable bias (base alpha)
        self.bias = nn.Parameter(torch.tensor(alpha_init))
        
    def forward(self, t):
        """
        Args:
            t: time value in [0, 1], can be scalar or tensor
        Returns:
            alpha: time-dependent alpha value(s)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.coeffs.device, dtype=torch.float32)
        
        # Compute Fourier features
        freqs = torch.arange(1, self.num_freqs + 1, device=t.device, dtype=torch.float32)
        # [num_freqs] or broadcast for batch
        phases = 2 * math.pi * freqs * t.unsqueeze(-1) if t.dim() > 0 else 2 * math.pi * freqs * t
        
        sin_features = torch.sin(phases)  # [num_freqs] or [batch, num_freqs]
        cos_features = torch.cos(phases)
        
        # Combine with coefficients
        a_coeffs = self.coeffs[:self.num_freqs]
        b_coeffs = self.coeffs[self.num_freqs:]
        
        alpha = self.bias + (a_coeffs * sin_features).sum(-1) + (b_coeffs * cos_features).sum(-1)
        
        # Clamp to valid range
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha


class AlphaNetworkMLP(nn.Module):
    """
    Small MLP network for time-dependent alpha.
    Input: t (normalized time)
    Output: α(t) in [0, 1]
    """
    def __init__(self, hidden_dim=32, alpha_init=0.0):
        super().__init__()
        self.alpha_init = alpha_init
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize last layer to output alpha_init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, alpha_init)
        
    def forward(self, t):
        """
        Args:
            t: time value in [0, 1], can be scalar or tensor
        Returns:
            alpha: time-dependent alpha value(s)
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=next(self.parameters()).device, dtype=torch.float32)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        t_input = t.view(-1, 1)
        alpha = self.net(t_input).squeeze(-1)
        
        # Clamp to valid range using sigmoid
        alpha = torch.sigmoid(alpha)
        
        return alpha if alpha.numel() > 1 else alpha.squeeze()


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            # print(self.scale_bound)  # [0.001 1.   ]
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

        # print(self.scale_bound,  scale_max_bound , scale_min_bound)

    def __init__(self, scale_bound=None, args=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self._deformation = deform_network(args)
        self._deformation_table = torch.empty(0)
        self.period = torch.empty(0)
        self.t_seq = torch.linspace(0, args.kplanes_config['resolution'][3]-1, args.kplanes_config['resolution'][3]).cuda()
        
        # v9: Low-Rank Motion Modes parameters
        self.use_low_rank_motion_modes = getattr(args, 'use_low_rank_motion_modes', False)
        self.num_motion_modes = getattr(args, 'num_motion_modes', 3)
        
        # v10: Adaptive Gating parameters
        self.use_adaptive_gating = getattr(args, 'use_adaptive_gating', False)
        self._motion_modes = torch.empty(0)  # U: [K, M, 3] per-Gaussian motion modes
        
        # V7.1: Alpha parameters for consistency-aware rendering
        self.use_trainable_alpha = getattr(args, 'use_trainable_alpha', False)
        self.trainable_alpha_init = getattr(args, 'trainable_alpha_init', 0.0)
        self.trainable_alpha_reg = getattr(args, 'trainable_alpha_reg', 1e-3)
        self.use_time_dependent_alpha = getattr(args, 'use_time_dependent_alpha', False)
        self.alpha_network_type = getattr(args, 'alpha_network_type', 'fourier')
        
        # V7.2: End-to-End Consistency-Aware parameters
        self.use_v7_2_consistency = getattr(args, 'use_v7_2_consistency', False)
        self.v7_2_alpha_init = getattr(args, 'v7_2_alpha_init', 0.3)
        self.v7_2_alpha_learnable = getattr(args, 'v7_2_alpha_learnable', True)
        self.v7_2_lambda_b_reg = getattr(args, 'v7_2_lambda_b_reg', 1e-3)
        self.v7_2_lambda_alpha_reg = getattr(args, 'v7_2_lambda_alpha_reg', 1e-3)
        # V7.2 time-dependent alpha parameters
        self.v7_2_use_time_dependent_alpha = getattr(args, 'v7_2_use_time_dependent_alpha', False)
        self.v7_2_alpha_network_type = getattr(args, 'v7_2_alpha_network_type', 'fourier')
        
        # Initialize alpha parameter or network
        self._alpha = None  # Will be initialized as nn.Parameter if use_trainable_alpha
        self._alpha_network = None  # Will be initialized if use_time_dependent_alpha
        self._v7_2_alpha = None  # V7.2 specific alpha parameter (scalar)
        self._v7_2_alpha_network = None  # V7.2 specific alpha network (time-dependent)
        
        # V7.2 alpha initialization (takes priority over V7.1 if both enabled)
        if self.use_v7_2_consistency:
            if self.v7_2_use_time_dependent_alpha:
                # V7.2 with time-dependent alpha: use a small network g_theta(t)
                if self.v7_2_alpha_network_type == 'fourier':
                    num_freqs = getattr(args, 'v7_2_alpha_fourier_freqs', 4)
                    self._v7_2_alpha_network = AlphaNetworkFourier(
                        num_freqs=num_freqs,
                        alpha_init=self.v7_2_alpha_init
                    ).cuda()
                elif self.v7_2_alpha_network_type == 'mlp':
                    hidden_dim = getattr(args, 'v7_2_alpha_mlp_hidden', 32)
                    self._v7_2_alpha_network = AlphaNetworkMLP(
                        hidden_dim=hidden_dim,
                        alpha_init=self.v7_2_alpha_init
                    ).cuda()
                print(f"[V7.2] Initialized time-dependent alpha network ({self.v7_2_alpha_network_type}), init={self.v7_2_alpha_init}")
            elif self.v7_2_alpha_learnable:
                # V7.2 with learnable scalar alpha
                self._v7_2_alpha = nn.Parameter(
                    torch.tensor(self.v7_2_alpha_init, device="cuda", dtype=torch.float32)
                )
                print(f"[V7.2] Initialized learnable scalar alpha = {self.v7_2_alpha_init}")
            else:
                # V7.2 with fixed alpha
                self.register_buffer(
                    "_v7_2_alpha", 
                    torch.tensor(self.v7_2_alpha_init, device="cuda", dtype=torch.float32)
                )
                print(f"[V7.2] Initialized fixed alpha = {self.v7_2_alpha_init}")
        elif self.use_trainable_alpha:
            if self.use_time_dependent_alpha:
                # Time-dependent alpha: use a small network
                if self.alpha_network_type == 'fourier':
                    num_freqs = getattr(args, 'alpha_fourier_freqs', 4)
                    self._alpha_network = AlphaNetworkFourier(
                        num_freqs=num_freqs, 
                        alpha_init=self.trainable_alpha_init
                    ).cuda()
                elif self.alpha_network_type == 'mlp':
                    hidden_dim = getattr(args, 'alpha_mlp_hidden', 32)
                    self._alpha_network = AlphaNetworkMLP(
                        hidden_dim=hidden_dim,
                        alpha_init=self.trainable_alpha_init
                    ).cuda()
                print(f"[V7.1] Initialized time-dependent alpha network ({self.alpha_network_type})")
            else:
                # Global scalar alpha
                self._alpha = nn.Parameter(
                    torch.tensor(self.trainable_alpha_init, device="cuda", dtype=torch.float32)
                )
                print(f"[V7.1] Initialized trainable alpha = {self.trainable_alpha_init}")
        
        # V7.5: Forward timewarp decoder heads for shape and density
        # These are used ONLY for computing time-warp consistency losses (not for rendering)
        # Input: canonical code (log-scale or density) + time embedding -> predicted dynamic value
        self.use_v7_5_full_timewarp = getattr(args, 'use_v7_5_full_timewarp', False)
        self._v7_5_time_embed_dim = 16
        self._v7_5_shape_dim = 3  # log-scale is 3D
        
        if self.use_v7_5_full_timewarp:
            # Time embedding MLP: t -> time_embed
            self._v7_5_time_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self._v7_5_time_embed_dim),
                nn.ReLU(inplace=True),
            ).cuda()
            
            # Shape timewarp head: canonical log-scale + time_embed -> predicted log-scale
            self._v7_5_shape_timewarp_head = nn.Sequential(
                nn.Linear(self._v7_5_time_embed_dim + self._v7_5_shape_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self._v7_5_shape_dim),
            ).cuda()
            
            # Density timewarp head: canonical density + time_embed -> predicted density
            self._v7_5_dens_timewarp_head = nn.Sequential(
                nn.Linear(self._v7_5_time_embed_dim + 1, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            ).cuda()
            
            print(f"[V7.5] Initialized forward timewarp decoders for shape (3D) and density (1D)")
        else:
            self._v7_5_time_mlp = None
            self._v7_5_shape_timewarp_head = None
            self._v7_5_dens_timewarp_head = None
        
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
            self._deformation.state_dict(),
            self._deformation_table,
            self.period,
            self._motion_modes,  # v9: add motion modes to checkpoint
        )

    def restore(self, model_args, training_args):
        # Handle both old checkpoints (13 items) and new checkpoints (14 items with motion_modes)
        if len(model_args) == 14:
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
                deform_state,
                self._deformation_table,
                self.period,
                self._motion_modes,  # v9: restore motion modes from checkpoint
            ) = model_args
        else:
            # Legacy checkpoint without motion_modes
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
                deform_state,
                self._deformation_table,
                self.period,
            ) = model_args
            # Initialize motion_modes if using low-rank mode
            if self.use_low_rank_motion_modes:
                num_gaussians = self._xyz.shape[0]
                self._motion_modes = nn.Parameter(
                    torch.zeros(num_gaussians, self.num_motion_modes, 3, device="cuda").requires_grad_(True)
                )
        
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )
    
    def parameters(self):
        module_params = [self._xyz, self._scaling, self._rotation, self._density]
        module_params.extend(self._deformation.parameters())
        module_params.extend(self.period)

        return module_params


    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # self.period = nn.Parameter(torch.FloatTensor([2.8]).cuda().requires_grad_(True))
        self.period = nn.Parameter(torch.FloatTensor([np.log(2.8)]).cuda().requires_grad_(True))
        
        # v9: Initialize per-Gaussian motion modes U: [K, M, 3]
        # K = number of Gaussians, M = number of motion modes
        # Initialize to zero (no initial displacement) - will be learned from data
        if self.use_low_rank_motion_modes:
            num_gaussians = fused_point_cloud.shape[0]
            self._motion_modes = nn.Parameter(
                torch.zeros(num_gaussians, self.num_motion_modes, 3, device="cuda").requires_grad_(True)
            )
            print(f"[v9] Initialized motion modes U: [{num_gaussians}, {self.num_motion_modes}, 3]")

        self._deformation = self._deformation.to("cuda")
        # v8: Set period reference for phase-conditioned deformation
        # This allows the deformation network to access T_hat = exp(period) for computing phase embedding
        self._deformation.set_period_ref(self.period)
        
        # v9: Set motion modes reference for low-rank deformation
        if self.use_low_rank_motion_modes:
            self._deformation.set_motion_modes_ref(self._motion_modes)
        
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {
                "params": [self.period],
                "lr": training_args.period_lr_init * self.spatial_lr_scale,
                "name": "period",
            },
        ]
        
        # v9: Add motion modes to optimizer if using low-rank motion modes
        if self.use_low_rank_motion_modes and self._motion_modes.numel() > 0:
            l.append({
                "params": [self._motion_modes],
                "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                "name": "motion_modes",
            })
        
        # V7.2: Add alpha to optimizer
        if self.use_v7_2_consistency:
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                # V7.2 with time-dependent alpha network
                l.append({
                    "params": list(self._v7_2_alpha_network.parameters()),
                    "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                    "name": "v7_2_alpha_network",
                })
                print(f"[V7.2] Added time-dependent alpha network to optimizer")
            elif self.v7_2_alpha_learnable and self._v7_2_alpha is not None:
                # V7.2 with learnable scalar alpha
                l.append({
                    "params": [self._v7_2_alpha],
                    "lr": 1e-2,  # Fixed learning rate for V7.2 alpha
                    "name": "v7_2_alpha",
                })
                print(f"[V7.2] Added learnable scalar alpha to optimizer")
        # V7.1: Add alpha parameters to optimizer if using trainable alpha
        elif self.use_trainable_alpha:
            if self.use_time_dependent_alpha and self._alpha_network is not None:
                # Time-dependent alpha network
                l.append({
                    "params": list(self._alpha_network.parameters()),
                    "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                    "name": "alpha_network",
                })
                print(f"[V7.1] Added alpha network parameters to optimizer")
            elif self._alpha is not None:
                # Global scalar alpha
                l.append({
                    "params": [self._alpha],
                    "lr": 1e-2,  # Fixed learning rate for scalar alpha
                    "name": "alpha",
                })
                print(f"[V7.1] Added scalar alpha to optimizer")
        
        # V7.5: Add timewarp decoders to optimizer
        if self.use_v7_5_full_timewarp and self._v7_5_time_mlp is not None:
            l.append({
                "params": list(self._v7_5_time_mlp.parameters()),
                "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                "name": "v7_5_time_mlp",
            })
            l.append({
                "params": list(self._v7_5_shape_timewarp_head.parameters()),
                "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                "name": "v7_5_shape_timewarp_head",
            })
            l.append({
                "params": list(self._v7_5_dens_timewarp_head.parameters()),
                "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
                "name": "v7_5_dens_timewarp_head",
            })
            print(f"[V7.5] Added timewarp decoder parameters to optimizer")

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  
        
        self.period_scheduler_args = get_expon_lr_func(
            lr_init=training_args.period_lr_init * self.spatial_lr_scale,
            lr_final=training_args.period_lr_final * self.spatial_lr_scale,
            max_steps=training_args.period_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "period":
                lr = self.period_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l
    
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz
    
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # V7.1: Load alpha parameters if they exist
        alpha_path = os.path.join(path, "alpha.pth")
        if os.path.exists(alpha_path):
            alpha_state = torch.load(alpha_path, map_location="cuda")
            if "alpha" in alpha_state and self._alpha is not None:
                self._alpha.data = alpha_state["alpha"]
                print(f"[V7.1] Loaded scalar alpha = {self._alpha.item():.4f}")
            if "alpha_network" in alpha_state and self._alpha_network is not None:
                self._alpha_network.load_state_dict(alpha_state["alpha_network"])
                print(f"[V7.1] Loaded alpha network state")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
        
        # V7.1: Save alpha parameters if using trainable alpha
        if self.use_trainable_alpha:
            alpha_state = {}
            if self._alpha is not None:
                alpha_state["alpha"] = self._alpha.data
                print(f"[V7.1] Saved scalar alpha = {self._alpha.item():.4f}")
            if self._alpha_network is not None:
                alpha_state["alpha_network"] = self._alpha_network.state_dict()
                print(f"[V7.1] Saved alpha network state")
            if alpha_state:
                torch.save(alpha_state, os.path.join(path, "alpha.pth"))

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)
        period = t2a(self.period)

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
            "period": period,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.period = nn.Parameter(torch.FloatTensor([2.8]).cuda().requires_grad_(True))
        self.scale_bound = data["scale_bound"]
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            if group["name"]=='period':continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    @property
    def get_motion_modes(self):
        """Get per-Gaussian motion modes U: [K, M, 3]"""
        return self._motion_modes

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # v9: Update motion modes if using low-rank motion modes
        if self.use_low_rank_motion_modes and "motion_modes" in optimizable_tensors:
            self._motion_modes = optimizable_tensors["motion_modes"]
            # Update the reference in deformation network
            self._deformation.set_motion_modes_ref(self._motion_modes)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            if group["name"]=='period':continue
            if group["name"] not in tensors_dict:continue  # v9: skip if tensor not in dict
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
        new_deformation_table,
        new_motion_modes=None,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        
        # v9: Add motion modes to dict if using low-rank motion modes
        if self.use_low_rank_motion_modes and new_motion_modes is not None:
            d["motion_modes"] = new_motion_modes

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # v9: Update motion modes if using low-rank motion modes
        if self.use_low_rank_motion_modes and "motion_modes" in optimizable_tensors:
            self._motion_modes = optimizable_tensors["motion_modes"]
            # Update the reference in deformation network
            self._deformation.set_motion_modes_ref(self._motion_modes)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        
        # v9: Create new motion modes for split Gaussians by copying from parent
        new_motion_modes = None
        if self.use_low_rank_motion_modes and self._motion_modes.numel() > 0:
            # For split, we copy the parent's motion modes to children
            new_motion_modes = self._motion_modes[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_deformation_table,
            new_motion_modes,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        new_deformation_table = self._deformation_table[selected_pts_mask]
        
        # v9: Create new motion modes for cloned Gaussians by copying from parent
        new_motion_modes = None
        if self.use_low_rank_motion_modes and self._motion_modes.numel() > 0:
            new_motion_modes = self._motion_modes[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_deformation_table,
            new_motion_modes,
        )

    @property
    def get_aabb(self):
        return self._deformation.get_aabb

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)

    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)

    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def get_alpha(self, time=None):
        """
        Get the current alpha value for V7.1 correction.
        
        If use_trainable_alpha is False, returns None.
        If use_time_dependent_alpha is True, returns alpha(t) from the network.
        Otherwise, returns the global scalar alpha.
        
        Args:
            time: Normalized time value (only used if time-dependent alpha is enabled)
        
        Returns:
            alpha: Scalar or tensor alpha value, or None if not using trainable alpha
        """
        if not self.use_trainable_alpha:
            return None
        
        if self.use_time_dependent_alpha and self._alpha_network is not None:
            if time is None:
                time = 0.5  # Default to mid-point if time not provided
            return self._alpha_network(time)
        elif self._alpha is not None:
            return self._alpha
        
        return None
    
    def compute_alpha_regularization_loss(self):
        """
        Compute L2 regularization loss for trainable alpha.
        
        L_alpha_reg = lambda_alpha * (alpha - alpha_init)^2
        
        For time-dependent alpha, this regularizes the network parameters
        to keep alpha(t) close to alpha_init for all t.
        
        Returns:
            Scalar regularization loss, or 0 if not using trainable alpha
        """
        if not self.use_trainable_alpha:
            return torch.tensor(0.0, device="cuda")
        
        if self.use_time_dependent_alpha and self._alpha_network is not None:
            # Sample alpha at multiple time points and regularize
            t_samples = torch.linspace(0, 1, 10, device="cuda")
            alpha_samples = self._alpha_network(t_samples)
            reg_loss = self.trainable_alpha_reg * ((alpha_samples - self.trainable_alpha_init) ** 2).mean()
            return reg_loss
        elif self._alpha is not None:
            reg_loss = self.trainable_alpha_reg * (self._alpha - self.trainable_alpha_init) ** 2
            return reg_loss
        
        return torch.tensor(0.0, device="cuda")

    def get_v7_2_alpha(self, time=None):
        """
        Get the current V7.2 alpha value.
        
        Args:
            time: Time value (scalar or tensor) for time-dependent alpha.
                  If None and using time-dependent alpha, returns mean alpha.
        
        Returns:
            alpha: Scalar tensor if V7.2 is enabled, None otherwise
        """
        if not self.use_v7_2_consistency:
            return None
        
        if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
            # Time-dependent alpha from network
            if time is not None:
                if isinstance(time, (int, float)):
                    time = torch.tensor([time], device="cuda", dtype=torch.float32)
                elif time.dim() == 0:
                    time = time.unsqueeze(0)
                # Take the first time value if multiple (for scalar alpha return)
                t_scalar = time[0] if time.numel() > 1 else time
                return self._v7_2_alpha_network(t_scalar.view(1)).squeeze()
            else:
                # Return mean alpha over time range
                t_samples = torch.linspace(0, 1, 10, device="cuda")
                return self._v7_2_alpha_network(t_samples).mean()
        elif self._v7_2_alpha is not None:
            return self._v7_2_alpha
        
        return None

    def get_deformed_centers(self, time, use_v7_1_correction=False, correction_alpha=0.0):
        """
        Get deformed Gaussian centers with optional V7.1/V7.2 consistency-aware correction.
        
        V7.1 extends V7 by using the backward field D_b to "correct" the forward
        deformed centers during rendering. The corrected center is computed as:
            y_i = mu_i + D_f(mu_i, t)            # V7 forward deformation
            x_hat_i = y_i + D_b(y_i, t)          # Round-trip back to canonical
            r_i = x_hat_i - mu_i                 # Round-trip residual
            y_corrected = y_i - alpha * r_i     # Corrected center
        
        V7.2 is similar but uses its own internal alpha parameter and is applied
        both during training and testing (end-to-end).
        
        When use_v7_1_correction=False AND use_v7_2_consistency=False, returns V7 behavior.
        
        Args:
            time: Time tensor (shape: [N, 1]) where N is number of Gaussians
            use_v7_1_correction: If True, apply V7.1 correction (external alpha)
            correction_alpha: Alpha coefficient for V7.1 correction (0=V7, >0=V7.1)
                             If None, uses trainable alpha (if enabled)
        
        Returns:
            means3D_final: Deformed (and optionally corrected) centers [N, 3]
            scales_final: Deformed scales (raw, before activation) [N, 3]
            rotations_final: Deformed rotations (raw, before activation) [N, 4]
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Ensure time has correct shape
        if time.dim() == 0:
            time = time.unsqueeze(0).unsqueeze(0).repeat(means3D.shape[0], 1)
        elif time.dim() == 1:
            time = time.unsqueeze(1)
        if time.shape[0] == 1:
            time = time.repeat(means3D.shape[0], 1)
        
        # Step 1: V7 forward deformation
        # y = mu + D_f(mu, t)
        means3D_deformed, scales_deformed, rotations_deformed = self._deformation(
            means3D, scales, rotations, density, time
        )
        
        # Determine if we need correction
        # V7.2 takes priority: if enabled, always apply correction with internal alpha
        apply_correction = False
        alpha = 0.0
        
        if self.use_v7_2_consistency:
            # V7.2: Use internal alpha for end-to-end training
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                # Time-dependent alpha: get alpha(t) from network
                # Use first time value (all Gaussians share the same time in a single render)
                t_scalar = time[0, 0] if time.dim() == 2 else time[0]
                alpha = self._v7_2_alpha_network(t_scalar.view(1)).squeeze()
                apply_correction = True
            elif self._v7_2_alpha is not None:
                # Scalar alpha
                alpha = self._v7_2_alpha
                apply_correction = True
        elif use_v7_1_correction and correction_alpha != 0.0:
            # V7.1: Use external alpha (for evaluation/TTO)
            apply_correction = True
            alpha = correction_alpha
        
        # If no correction needed, return V7 result directly
        if not apply_correction:
            return means3D_deformed, scales_deformed, rotations_deformed
        
        # Step 2: Correction using backward field
        # x_hat = y + D_b(y, t)
        means3D_reconstructed, _ = self._deformation.forward_backward_position(
            means3D_deformed, time
        )
        
        # Step 3: Compute round-trip residual
        # r = x_hat - mu
        residual = means3D_reconstructed - means3D  # [N, 3]
        
        # Step 4: Apply correction
        # y_corrected = y - alpha * r
        means3D_corrected = means3D_deformed - alpha * residual  # [N, 3]
        
        return means3D_corrected, scales_deformed, rotations_deformed

    def compute_backward_magnitude_loss(self, time, sample_ratio=0.1):
        """
        Compute V7.2 backward field magnitude regularization loss.
        
        L_b = mean |D_b(y, t)|
        
        This prevents D_b from growing too large while allowing it to participate
        in the main rendering path via the consistency-aware forward mapping.
        
        Args:
            time: Time tensor (shape: [N, 1]) where N is number of Gaussians
            sample_ratio: Ratio of Gaussians to sample for efficiency (default 0.1)
        
        Returns:
            L_b: Scalar loss value
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
            time_sampled = time[indices] if time.shape[0] == num_gaussians else time[:num_samples]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            time_sampled = time
        
        # Ensure time has correct shape
        if time_sampled.dim() == 0:
            time_sampled = time_sampled.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time_sampled.dim() == 1:
            time_sampled = time_sampled.unsqueeze(1)
        if time_sampled.shape[0] == 1:
            time_sampled = time_sampled.repeat(means3D_sampled.shape[0], 1)
        
        # Step 1: Forward deformation to get y = mu + D_f(mu, t)
        means3D_deformed, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )
        
        # Step 2: Get D_b(y, t) - the backward displacement at deformed position
        _, D_b = self._deformation.forward_backward_position(
            means3D_deformed, time_sampled
        )
        
        # Step 3: Compute L1 magnitude of D_b
        L_b = torch.abs(D_b).mean()
        
        return L_b

    def compute_v7_2_alpha_regularization_loss(self):
        """
        Compute V7.2 alpha regularization loss.
        
        L_alpha = (alpha - alpha_init)^2
        
        For time-dependent alpha: L_alpha = mean_t (alpha(t) - alpha_init)^2
        
        This keeps alpha close to its initial value to prevent extreme correction.
        
        Returns:
            Scalar regularization loss, or 0 if V7.2 is not enabled
        """
        if not self.use_v7_2_consistency:
            return torch.tensor(0.0, device="cuda")
        
        if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
            # Time-dependent alpha: sample at multiple time points and regularize
            t_samples = torch.linspace(0, 1, 10, device="cuda")
            alpha_samples = self._v7_2_alpha_network(t_samples)
            reg_loss = ((alpha_samples - self.v7_2_alpha_init) ** 2).mean()
            return reg_loss
        elif self._v7_2_alpha is not None and self.v7_2_alpha_learnable:
            # Scalar alpha
            reg_loss = (self._v7_2_alpha - self.v7_2_alpha_init) ** 2
            return reg_loss
        
        return torch.tensor(0.0, device="cuda")

    def compute_inverse_consistency_loss(self, time, sample_ratio=1.0, compute_symmetry=False):
        """
        Compute inverse consistency loss and optionally symmetry loss for position deformation.
        
        L_inv_pos = mean over (i, t) of || x_hat - x ||_1
        where:
            - x = mu_i (canonical Gaussian center)
            - y = mu_i + D_f(mu_i, t) (forward deformation to time t)
            - x_hat = y + D_b(y, t) (backward deformation back to canonical)
        
        L_sym = mean over (i, t) of || D_b(y, t) + D_f(x, t) ||_1
        This encourages D_b ≈ -D_f (weak symmetry constraint).
        
        Args:
            time: Time tensor (shape: [N, 1]) where N is number of Gaussians
            sample_ratio: Ratio of Gaussians to sample (0.0-1.0), 1.0 = use all
            compute_symmetry: If True, also compute and return L_sym
        
        Returns:
            If compute_symmetry is False:
                L_inv_pos: Scalar inverse consistency loss
            If compute_symmetry is True:
                (L_inv_pos, L_sym): Tuple of inverse consistency and symmetry losses
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
            time_sampled = time[indices] if time.shape[0] == num_gaussians else time[:num_samples]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            time_sampled = time
        
        # Ensure time has correct shape
        if time_sampled.dim() == 0:
            time_sampled = time_sampled.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time_sampled.dim() == 1:
            time_sampled = time_sampled.unsqueeze(1)
        if time_sampled.shape[0] == 1:
            time_sampled = time_sampled.repeat(means3D_sampled.shape[0], 1)
        
        # Step 1: Forward deformation - get deformed positions y = mu_i + D_f(mu_i, t)
        means3D_deformed, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )
        
        # Compute D_f = y - x (forward deformation vector)
        D_f = means3D_deformed - means3D_sampled  # [N, 3]
        
        # Step 2: Backward deformation - get reconstructed positions x_hat = y + D_b(y, t)
        means3D_reconstructed, D_b = self._deformation.forward_backward_position(
            means3D_deformed, time_sampled
        )
        # D_b is the backward deformation vector returned by forward_backward_position
        
        # Step 3: Compute L1 loss between original and reconstructed positions
        L_inv_pos = torch.abs(means3D_reconstructed - means3D_sampled).mean()
        
        if compute_symmetry:
            # Step 4: Compute symmetry loss L_sym = || D_b + D_f ||_1
            # This encourages D_b ≈ -D_f (backward deformation should be opposite of forward)
            L_sym = torch.abs(D_b + D_f).mean()
            return L_inv_pos, L_sym
        
        return L_inv_pos

    def compute_cycle_motion_loss(self, time, sample_ratio=1.0):
        """
        Compute cycle motion consistency loss using the learned breathing period T_hat.
        
        L_cycle_motion = mean over (i, t) of || x2 - x1 ||_1
        where:
            - x1 = mu_i + D_f(mu_i, t)
            - x2 = mu_i + D_f(mu_i, t + T_hat)
            - T_hat = exp(period) is the learned breathing period from SSRML
        
        This encourages the motion field to be periodic: after one full breathing cycle,
        the deformed position should return to approximately the same location.
        
        Args:
            time: Time tensor (shape: [N, 1]) where N is number of Gaussians
            sample_ratio: Ratio of Gaussians to sample (0.0-1.0), 1.0 = use all
        
        Returns:
            L_cycle_motion: Scalar cycle motion consistency loss
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Get the learned period T_hat = exp(period)
        # Note: self.period stores log(T_hat), so T_hat = exp(self.period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
            time_sampled = time[indices] if time.shape[0] == num_gaussians else time[:num_samples]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            time_sampled = time
        
        # Ensure time has correct shape
        if time_sampled.dim() == 0:
            time_sampled = time_sampled.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time_sampled.dim() == 1:
            time_sampled = time_sampled.unsqueeze(1)
        if time_sampled.shape[0] == 1:
            time_sampled = time_sampled.repeat(means3D_sampled.shape[0], 1)
        
        # Compute time shifted by one period: t + T_hat
        # T_hat is normalized time (period / scanTime), so we add it directly
        # Since the training time is normalized to [0, 1] range and T_hat is in the same scale
        time_plus_T = time_sampled + T_hat
        
        # Step 1: Forward deformation at time t
        # y1 = mu_i + D_f(mu_i, t)
        means3D_t, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )
        
        # Step 2: Forward deformation at time t + T_hat
        # y2 = mu_i + D_f(mu_i, t + T_hat)
        means3D_t_plus_T, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_plus_T
        )
        
        # V7.2: Apply consistency-aware correction if enabled
        # Use y_corrected = y - alpha * r instead of y
        if self.use_v7_2_consistency:
            # Get alpha (scalar or time-dependent)
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                t_scalar = time_sampled[0, 0] if time_sampled.dim() == 2 else time_sampled[0]
                alpha_t = self._v7_2_alpha_network(t_scalar.view(1)).squeeze()
                t_plus_T_scalar = time_plus_T[0, 0] if time_plus_T.dim() == 2 else time_plus_T[0]
                alpha_t_plus_T = self._v7_2_alpha_network(t_plus_T_scalar.view(1)).squeeze()
            elif self._v7_2_alpha is not None:
                alpha_t = self._v7_2_alpha
                alpha_t_plus_T = self._v7_2_alpha
            else:
                alpha_t = None
            
            if alpha_t is not None:
                # Correction at time t
                means3D_reconstructed_t, _ = self._deformation.forward_backward_position(
                    means3D_t, time_sampled
                )
                residual_t = means3D_reconstructed_t - means3D_sampled
                means3D_t = means3D_t - alpha_t * residual_t
                
                # Correction at time t + T_hat
                means3D_reconstructed_t_plus_T, _ = self._deformation.forward_backward_position(
                    means3D_t_plus_T, time_plus_T
                )
                residual_t_plus_T = means3D_reconstructed_t_plus_T - means3D_sampled
                means3D_t_plus_T = means3D_t_plus_T - alpha_t_plus_T * residual_t_plus_T
        
        # Step 3: Compute L1 loss between positions at t and t + T_hat
        # L_cycle_motion = || y_corrected(t+T) - y_corrected(t) ||_1 (V7.2)
        # L_cycle_motion = || y(t+T) - y(t) ||_1 (original V7)
        L_cycle_motion = torch.abs(means3D_t_plus_T - means3D_t).mean()
        
        return L_cycle_motion

    def compute_cycle_canon_loss(self, time, sample_ratio=1.0):
        """
        Compute V7.2.1 canonical-space cycle consistency loss (L_cycle-canon).
        
        This loss requires that the same physical point μ_i, when decoded back to
        canonical space via D_b at two times t and t+T̂, should yield consistent results.
        
        Unlike L_inv which forces x̂ = μ (r → 0), this only requires:
            x_canon(t+T̂) ≈ x_canon(t)
        
        where:
            x_canon(t)   = ϕ̃_f(μ, t)   + D_b(ϕ̃_f(μ, t),   t)
            x_canon(t+T) = ϕ̃_f(μ, t+T) + D_b(ϕ̃_f(μ, t+T), t+T)
        
        This provides a softer temporal constraint on D_b without enforcing perfect
        inverse consistency, allowing the correction mechanism to work effectively.
        
        Note: This loss uses the V7.2 corrected centers (y_corrected) as input to D_b,
        which is consistent with how rendering and L_cycle_motion use them.
        
        Args:
            time: Time samples [N, 1] or [1, 1] (will be broadcast)
            sample_ratio: Fraction of Gaussians to sample (default: 1.0 = all)
        
        Returns:
            L_cycle_canon: Scalar canonical-space cycle consistency loss
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Get the learned period T_hat = exp(period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
            time_sampled = time[indices] if time.shape[0] == num_gaussians else time[:num_samples]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            time_sampled = time
        
        # Ensure time has correct shape
        if time_sampled.dim() == 0:
            time_sampled = time_sampled.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time_sampled.dim() == 1:
            time_sampled = time_sampled.unsqueeze(1)
        if time_sampled.shape[0] == 1:
            time_sampled = time_sampled.repeat(means3D_sampled.shape[0], 1)
        
        # Compute time shifted by one period: t + T_hat
        time_plus_T = time_sampled + T_hat
        
        # Step 1: Get V7.2 corrected forward centers at t and t+T̂
        # These are ϕ̃_f(μ, t) and ϕ̃_f(μ, t+T̂) - already include the correction
        # We use get_deformed_centers which returns y_corrected when V7.2 is enabled
        
        # For efficiency, we compute forward deformation and correction inline
        # (similar to compute_cycle_motion_loss)
        
        # Forward deformation at time t: y(t) = μ + D_f(μ, t)
        means3D_t, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )
        
        # Forward deformation at time t + T_hat: y(t+T) = μ + D_f(μ, t+T)
        means3D_t_plus_T, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_plus_T
        )
        
        # Apply V7.2 correction to get ϕ̃_f(μ, t) and ϕ̃_f(μ, t+T̂)
        # (This is the same correction logic as in compute_cycle_motion_loss)
        if self.use_v7_2_consistency:
            # Get alpha (scalar or time-dependent)
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                t_scalar = time_sampled[0, 0] if time_sampled.dim() == 2 else time_sampled[0]
                alpha_t = self._v7_2_alpha_network(t_scalar.view(1)).squeeze()
                t_plus_T_scalar = time_plus_T[0, 0] if time_plus_T.dim() == 2 else time_plus_T[0]
                alpha_t_plus_T = self._v7_2_alpha_network(t_plus_T_scalar.view(1)).squeeze()
            elif self._v7_2_alpha is not None:
                alpha_t = self._v7_2_alpha
                alpha_t_plus_T = self._v7_2_alpha
            else:
                alpha_t = torch.tensor(0.0, device=means3D.device)
                alpha_t_plus_T = torch.tensor(0.0, device=means3D.device)
            
            # Correction at time t: y_corrected(t) = y(t) - α * (x̂(t) - μ)
            means3D_reconstructed_t, _ = self._deformation.forward_backward_position(
                means3D_t, time_sampled
            )
            residual_t = means3D_reconstructed_t - means3D_sampled
            centers_t = means3D_t - alpha_t * residual_t  # ϕ̃_f(μ, t)
            
            # Correction at time t + T_hat
            means3D_reconstructed_t_plus_T, _ = self._deformation.forward_backward_position(
                means3D_t_plus_T, time_plus_T
            )
            residual_t_plus_T = means3D_reconstructed_t_plus_T - means3D_sampled
            centers_t_plus_T = means3D_t_plus_T - alpha_t_plus_T * residual_t_plus_T  # ϕ̃_f(μ, t+T̂)
        else:
            # No V7.2 correction, use raw forward centers
            centers_t = means3D_t
            centers_t_plus_T = means3D_t_plus_T
        
        # Step 2: Decode corrected centers back to canonical via D_b
        # x_canon(t) = ϕ̃_f(μ, t) + D_b(ϕ̃_f(μ, t), t)
        x_canon_t, D_b_t = self._deformation.forward_backward_position(
            centers_t, time_sampled
        )
        # Note: forward_backward_position returns (x_hat, D_b) where x_hat = input + D_b
        # So x_canon_t is already centers_t + D_b(centers_t, t)
        
        # x_canon(t+T̂) = ϕ̃_f(μ, t+T̂) + D_b(ϕ̃_f(μ, t+T̂), t+T̂)
        x_canon_t_plus_T, D_b_t_plus_T = self._deformation.forward_backward_position(
            centers_t_plus_T, time_plus_T
        )
        
        # Step 3: Compute L1 loss between canonical decodes at t and t + T_hat
        # L_cycle_canon = |x_canon(t+T̂) - x_canon(t)|
        L_cycle_canon = torch.abs(x_canon_t_plus_T - x_canon_t).mean()
        
        return L_cycle_canon

    def compute_timewarp_loss(self, time, delta_fraction=0.1, compute_fw_bw=False, sample_ratio=1.0):
        """
        Compute V7.3 temporal bidirectional warp losses.
        
        This loss defines true time-forward and time-backward warps using canonical
        as a bridge between different time points:
        
        Time-forward warp (t1 → t2):
            1. Get corrected position at t1: x(t1) = ϕ̃_f(μ, t1)
            2. Decode to canonical: μ^(t1) = x(t1) + D_b(x(t1), t1)
            3. Forward to t2: x_fw(t2|t1) = ϕ̃_f(μ^(t1), t2)
            4. L_fw = |x_fw(t2|t1) - x(t2)|
        
        Time-backward warp (t2 → t1):
            1. Get corrected position at t2: x(t2) = ϕ̃_f(μ, t2)
            2. Decode to canonical: μ^(t2) = x(t2) + D_b(x(t2), t2)
            3. Forward to t1: x_bw(t1|t2) = ϕ̃_f(μ^(t2), t1)
            4. L_bw = |x_bw(t1|t2) - x(t1)|
        
        Optional round-trip closure (t1 → t2 → t1):
            Start from x(t1), warp to t2, then warp back to t1, should return to x(t1).
        
        Args:
            time: Base time samples t1, shape [N, 1] or [1, 1]
            delta_fraction: Fraction of period T̂ to use as time gap Δ
            compute_fw_bw: Whether to compute the round-trip closure loss
            sample_ratio: Fraction of Gaussians to sample (default: 1.0 = all)
        
        Returns:
            dict with keys: 'L_fw', 'L_bw', and optionally 'L_fw_bw'
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Get the learned period T_hat = exp(period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Compute time gap: Δ = fraction * T̂
        delta = delta_fraction * T_hat
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
            time_sampled = time[indices] if time.shape[0] == num_gaussians else time[:num_samples]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            time_sampled = time
        
        # Ensure time has correct shape
        if time_sampled.dim() == 0:
            time_sampled = time_sampled.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time_sampled.dim() == 1:
            time_sampled = time_sampled.unsqueeze(1)
        if time_sampled.shape[0] == 1:
            time_sampled = time_sampled.repeat(means3D_sampled.shape[0], 1)
        
        # Define time points: t1 = t, t2 = t + Δ
        t1 = time_sampled
        t2 = time_sampled + delta
        
        # ========================================
        # Step 1: Get corrected trajectory points at t1 and t2
        # x(t1) = ϕ̃_f(μ, t1), x(t2) = ϕ̃_f(μ, t2)
        # ========================================
        
        # Forward deformation at t1: y(t1) = μ + D_f(μ, t1)
        means3D_t1, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t1
        )
        
        # Forward deformation at t2: y(t2) = μ + D_f(μ, t2)
        means3D_t2, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t2
        )
        
        # Apply V7.2 correction to get ϕ̃_f(μ, t1) and ϕ̃_f(μ, t2)
        if self.use_v7_2_consistency:
            # Get alpha (scalar or time-dependent)
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                t1_scalar = t1[0, 0] if t1.dim() == 2 else t1[0]
                alpha_t1 = self._v7_2_alpha_network(t1_scalar.view(1)).squeeze()
                t2_scalar = t2[0, 0] if t2.dim() == 2 else t2[0]
                alpha_t2 = self._v7_2_alpha_network(t2_scalar.view(1)).squeeze()
            elif self._v7_2_alpha is not None:
                alpha_t1 = self._v7_2_alpha
                alpha_t2 = self._v7_2_alpha
            else:
                alpha_t1 = torch.tensor(0.0, device=means3D.device)
                alpha_t2 = torch.tensor(0.0, device=means3D.device)
            
            # Correction at t1
            means3D_reconstructed_t1, _ = self._deformation.forward_backward_position(
                means3D_t1, t1
            )
            residual_t1 = means3D_reconstructed_t1 - means3D_sampled
            x_t1 = means3D_t1 - alpha_t1 * residual_t1  # ϕ̃_f(μ, t1)
            
            # Correction at t2
            means3D_reconstructed_t2, _ = self._deformation.forward_backward_position(
                means3D_t2, t2
            )
            residual_t2 = means3D_reconstructed_t2 - means3D_sampled
            x_t2 = means3D_t2 - alpha_t2 * residual_t2  # ϕ̃_f(μ, t2)
        else:
            # No correction, use raw forward centers
            x_t1 = means3D_t1
            x_t2 = means3D_t2
            alpha_t1 = torch.tensor(0.0, device=means3D.device)
            alpha_t2 = torch.tensor(0.0, device=means3D.device)
        
        # ========================================
        # Step 2: Decode trajectory points back to canonical
        # μ^(t1) = x(t1) + D_b(x(t1), t1)
        # μ^(t2) = x(t2) + D_b(x(t2), t2)
        # ========================================
        mu_decoded_t1, _ = self._deformation.forward_backward_position(x_t1, t1)
        mu_decoded_t2, _ = self._deformation.forward_backward_position(x_t2, t2)
        
        # ========================================
        # Step 3: Time-forward warp (t1 → t2)
        # x_fw(t2|t1) = ϕ̃_f(μ^(t1), t2)
        # ========================================
        # Forward from decoded canonical at t1 to time t2
        x_fw_t2_raw, _, _ = self._deformation(
            mu_decoded_t1, scales_sampled, rotations_sampled, density_sampled, t2
        )
        
        # Apply correction if V7.2 is enabled
        if self.use_v7_2_consistency and (alpha_t2 != 0.0 if isinstance(alpha_t2, float) else alpha_t2.abs() > 1e-8):
            x_fw_reconstructed, _ = self._deformation.forward_backward_position(x_fw_t2_raw, t2)
            residual_fw = x_fw_reconstructed - mu_decoded_t1
            x_fw_t2 = x_fw_t2_raw - alpha_t2 * residual_fw
        else:
            x_fw_t2 = x_fw_t2_raw
        
        # L_fw: time-forward consistency
        L_fw = torch.abs(x_fw_t2 - x_t2).mean()
        
        # ========================================
        # Step 4: Time-backward warp (t2 → t1)
        # x_bw(t1|t2) = ϕ̃_f(μ^(t2), t1)
        # ========================================
        # Forward from decoded canonical at t2 to time t1
        x_bw_t1_raw, _, _ = self._deformation(
            mu_decoded_t2, scales_sampled, rotations_sampled, density_sampled, t1
        )
        
        # Apply correction if V7.2 is enabled
        if self.use_v7_2_consistency and (alpha_t1 != 0.0 if isinstance(alpha_t1, float) else alpha_t1.abs() > 1e-8):
            x_bw_reconstructed, _ = self._deformation.forward_backward_position(x_bw_t1_raw, t1)
            residual_bw = x_bw_reconstructed - mu_decoded_t2
            x_bw_t1 = x_bw_t1_raw - alpha_t1 * residual_bw
        else:
            x_bw_t1 = x_bw_t1_raw
        
        # L_bw: time-backward consistency
        L_bw = torch.abs(x_bw_t1 - x_t1).mean()
        
        losses = {
            "L_fw": L_fw,
            "L_bw": L_bw,
        }
        
        # ========================================
        # Step 5 (Optional): Round-trip closure (t1 → t2 → t1)
        # Start from x(t1), warp to t2 (x_fw_t2), then warp back to t1
        # ========================================
        if compute_fw_bw:
            # Decode x_fw_t2 back to canonical at t2
            mu_fw_decoded_t2, _ = self._deformation.forward_backward_position(x_fw_t2, t2)
            
            # Forward from this canonical to t1
            x_cycle_t1_raw, _, _ = self._deformation(
                mu_fw_decoded_t2, scales_sampled, rotations_sampled, density_sampled, t1
            )
            
            # Apply correction if V7.2 is enabled
            if self.use_v7_2_consistency and (alpha_t1 != 0.0 if isinstance(alpha_t1, float) else alpha_t1.abs() > 1e-8):
                x_cycle_reconstructed, _ = self._deformation.forward_backward_position(x_cycle_t1_raw, t1)
                residual_cycle = x_cycle_reconstructed - mu_fw_decoded_t2
                x_cycle_t1 = x_cycle_t1_raw - alpha_t1 * residual_cycle
            else:
                x_cycle_t1 = x_cycle_t1_raw
            
            # L_fw_bw: round-trip closure
            L_fw_bw = torch.abs(x_cycle_t1 - x_t1).mean()
            losses["L_fw_bw"] = L_fw_bw
        
        return losses

    def get_time_dependent_sigma_rho(self, time, sample_ratio=1.0):
        """
        Get time-dependent scale and density for V7.3.1 temporal regularization.
        
        This function queries the deformation network to get the scale and density
        at a given time point. If scale/density are static (not time-dependent),
        this is detected and handled gracefully.
        
        Args:
            time: Time tensor of shape [N, 1] or [1, 1] or scalar
            sample_ratio: Fraction of Gaussians to sample (default: 1.0 = all)
        
        Returns:
            scales_t: Time-dependent scales [N, 3] or None if static
            density_t: Time-dependent density [N, 1] or None if static
            is_dynamic_scale: Boolean indicating if scales are time-dependent
            is_dynamic_density: Boolean indicating if density is time-dependent
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D = means3D[indices]
            density = density[indices]
            scales = scales[indices]
            rotations = rotations[indices]
        
        # Ensure time has correct shape
        if time.dim() == 0:
            time = time.unsqueeze(0).unsqueeze(0).repeat(means3D.shape[0], 1)
        elif time.dim() == 1:
            time = time.unsqueeze(1)
        if time.shape[0] == 1:
            time = time.repeat(means3D.shape[0], 1)
        
        # Query deformation network to get time-dependent scales
        # The deformation returns: (means3D_deformed, scales_deformed, rotations_deformed)
        _, scales_t, _ = self._deformation(
            means3D, scales, rotations, density, time
        )
        
        # Check if scales are actually time-dependent
        # By comparing with the base scales (if they're identical, scales are static)
        # Note: scales_t comes from deformation, scales is the base canonical scale
        is_dynamic_scale = not getattr(self._deformation.deformation_net.args, 'no_ds', False)
        
        # Current implementation has static density (density_deform is commented out)
        # So density is always static for now
        is_dynamic_density = False
        density_t = density  # Return base density
        
        return scales_t, density_t, is_dynamic_scale, is_dynamic_density

    def compute_sigma_rho_temporal_losses(self, time, args, sample_ratio=1.0):
        """
        Compute V7.3.1 temporal TV and cycle consistency losses for
        per-Gaussian covariance (scale) and density.
        
        This implements:
            L_tv_sigma: |log(S(t+δ)) - log(S(t))| - temporal smoothness for scale
            L_tv_rho: |ρ(t+δ) - ρ(t)| - temporal smoothness for density
            L_cycle_sigma: |log(S(t+T̂)) - log(S(t))| - periodic consistency for scale
            L_cycle_rho: |ρ(t+T̂) - ρ(t)| - periodic consistency for density
        
        Args:
            time: Base time samples, shape [N, 1] or [1, 1]
            args: Optimization arguments containing V7.3.1 parameters
            sample_ratio: Fraction of Gaussians to sample (default: 1.0 = all)
        
        Returns:
            dict with keys: 'L_tv_rho', 'L_tv_sigma', 'L_cycle_rho', 'L_cycle_sigma'
            (if some are not applicable, returns zeros)
        """
        device = time.device if hasattr(time, 'device') else self.get_xyz.device
        zero = torch.zeros([], device=device)
        
        # Early exit if V7.3.1 is not enabled
        use_v7_3_1 = getattr(args, 'use_v7_3_1_sigma_rho', False)
        if not use_v7_3_1:
            return {
                "L_tv_rho": zero,
                "L_tv_sigma": zero,
                "L_cycle_rho": zero,
                "L_cycle_sigma": zero,
            }
        
        # Get the learned period T_hat = exp(period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Compute time step delta for TV loss
        delta_frac = getattr(args, 'v7_3_1_sigma_rho_delta_fraction', 0.1)
        delta = delta_frac * T_hat
        
        # Define time points
        t1 = time  # base time
        t2 = time + delta  # t + δ (for TV loss)
        tT = time + T_hat  # t + T̂ (for cycle loss)
        
        # Get time-dependent scale/density at three time points
        scales_t1, rho_t1, is_dynamic_scale, is_dynamic_density = self.get_time_dependent_sigma_rho(t1, sample_ratio)
        scales_t2, rho_t2, _, _ = self.get_time_dependent_sigma_rho(t2, sample_ratio)
        scales_tT, rho_tT, _, _ = self.get_time_dependent_sigma_rho(tT, sample_ratio)
        
        # Initialize losses
        L_tv_sigma = zero
        L_cycle_sigma = zero
        L_tv_rho = zero
        L_cycle_rho = zero
        
        # Compute scale losses if scales are time-dependent
        if is_dynamic_scale and scales_t1 is not None:
            eps = 1e-6
            # Use log-scale space for stability (relative variations)
            # scales are typically in log space already (_scaling), but after deformation
            # they may be in linear space, so we ensure positivity
            log_scales_t1 = torch.log(torch.abs(scales_t1) + eps)
            log_scales_t2 = torch.log(torch.abs(scales_t2) + eps)
            log_scales_tT = torch.log(torch.abs(scales_tT) + eps)
            
            # Temporal TV: penalize rapid changes in log-scale
            L_tv_sigma = torch.abs(log_scales_t2 - log_scales_t1).mean()
            
            # Periodic consistency: log-scale at t+T̂ should match t
            L_cycle_sigma = torch.abs(log_scales_tT - log_scales_t1).mean()
        
        # Compute density losses if density is time-dependent
        if is_dynamic_density and rho_t1 is not None:
            # Temporal TV: penalize rapid changes in density
            L_tv_rho = torch.abs(rho_t2 - rho_t1).mean()
            
            # Periodic consistency: density at t+T̂ should match t
            L_cycle_rho = torch.abs(rho_tT - rho_t1).mean()
        
        return {
            "L_tv_rho": L_tv_rho,
            "L_tv_sigma": L_tv_sigma,
            "L_cycle_rho": L_cycle_rho,
            "L_cycle_sigma": L_cycle_sigma,
        }

    def save_base_canonical_params(self):
        """
        V7.4: Save base canonical parameters (log-scale and density) for canonical decode losses.
        
        Should be called after static warm-up is complete and before dynamic training starts.
        These values serve as the "base" canonical parameters that the backward field
        should decode dynamic Gaussians back to.
        """
        # Save base log-scale: log(S_i^{base})
        # _scaling is stored in log space already
        self._log_scales_base = self._scaling.detach().clone()
        
        # Save base density: ρ_i^{base}
        self._density_base = self.get_density.detach().clone()
        
        print(f"[V7.4] Saved base canonical params: log_scales_base {self._log_scales_base.shape}, density_base {self._density_base.shape}")

    @property
    def log_scales_base(self):
        """Get base canonical log-scales (saved after warm-up)."""
        if hasattr(self, '_log_scales_base') and self._log_scales_base is not None:
            return self._log_scales_base
        else:
            # Fallback: use current scales as base
            return self._scaling.detach()

    @property
    def density_base(self):
        """Get base canonical density (saved after warm-up)."""
        if hasattr(self, '_density_base') and self._density_base is not None:
            return self._density_base
        else:
            # Fallback: use current density as base
            return self.get_density.detach()

    def decode_canonical_sigma_rho(self, x_t, time, sample_indices=None):
        """
        V7.4: Decode canonical shape (log-scale) and density from dynamic centers.
        
        Uses the backward network's shape and density heads to compute residuals,
        which are added to the base canonical parameters:
            ŝ_canon(t) = s_base + D_b_shape(x(t), t)
            ρ̂_canon(t) = ρ_base + D_b_dens(x(t), t)
        
        Args:
            x_t: Dynamic centers at time t, shape (N, 3)
            time: Time tensor, shape (N, 1)
            sample_indices: Optional indices for subsampling
        
        Returns:
            s_canon_t: Canonical decoded log-scale (N, 3)
            rho_canon_t: Canonical decoded density (N, 1)
        """
        # Get base canonical parameters
        s_base = self.log_scales_base
        rho_base = self.density_base
        
        # Apply sample indices if provided
        if sample_indices is not None:
            s_base = s_base[sample_indices]
            rho_base = rho_base[sample_indices]
        
        # Ensure time has correct shape
        if time.dim() == 0:
            time = time.unsqueeze(0).unsqueeze(0).repeat(x_t.shape[0], 1)
        elif time.dim() == 1:
            time = time.unsqueeze(1)
        if time.shape[0] == 1:
            time = time.repeat(x_t.shape[0], 1)
        
        # Get backward decode residuals using V7.4 heads
        D_b_pos, D_b_shape, D_b_dens = self._deformation.backward_deform_full(x_t, time)
        
        # Canonical decoded parameters
        s_canon_t = s_base + D_b_shape
        rho_canon_t = rho_base + D_b_dens
        
        return s_canon_t, rho_canon_t

    def compute_canonical_sigma_rho_losses(self, time, args, sample_ratio=1.0):
        """
        Compute V7.4 canonical decode losses for shape (Sigma/scale) and density.
        
        This implements:
            L_cycle_canon_sigma: |ŝ_canon(t+T̂) - ŝ_canon(t)| - periodic consistency
            L_cycle_canon_rho: |ρ̂_canon(t+T̂) - ρ̂_canon(t)| - periodic consistency
            L_prior_sigma: |ŝ_canon(t) - s_base| - anchor to base canonical
            L_prior_rho: |ρ̂_canon(t) - ρ_base| - anchor to base canonical
        
        Args:
            time: Base time samples, shape [N, 1] or [1, 1]
            args: Optimization arguments containing V7.4 parameters
            sample_ratio: Fraction of Gaussians to sample (default: 1.0 = all)
        
        Returns:
            dict with keys: 'L_cycle_canon_Sigma', 'L_cycle_canon_rho', 'L_prior_Sigma', 'L_prior_rho'
        """
        device = time.device if hasattr(time, 'device') else self.get_xyz.device
        zero = torch.zeros([], device=device)
        
        # Early exit if V7.4 is not enabled
        use_v7_4 = getattr(args, 'use_v7_4_canonical_decode', False)
        if not use_v7_4:
            return {
                "L_cycle_canon_Sigma": zero,
                "L_cycle_canon_rho": zero,
                "L_prior_Sigma": zero,
                "L_prior_rho": zero,
            }
        
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        sample_indices = None
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            sample_indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D = means3D[sample_indices]
            density = density[sample_indices]
            scales = scales[sample_indices]
            rotations = rotations[sample_indices]
        
        # Get the learned period T_hat = exp(period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Define time points
        t1 = time  # base time
        tT = time + T_hat  # t + T̂ (for cycle loss)
        
        # Ensure time has correct shape
        if t1.dim() == 0:
            t1 = t1.unsqueeze(0).unsqueeze(0).repeat(means3D.shape[0], 1)
        elif t1.dim() == 1:
            t1 = t1.unsqueeze(1)
        if t1.shape[0] == 1:
            t1 = t1.repeat(means3D.shape[0], 1)
        tT = t1 + T_hat
        
        # Get corrected dynamic centers at t1 and tT using V7.2 get_deformed_centers
        # Forward deformation at t1
        means3D_t1, _, _ = self._deformation(
            means3D, scales, rotations, density, t1
        )
        # Forward deformation at tT
        means3D_tT, _, _ = self._deformation(
            means3D, scales, rotations, density, tT
        )
        
        # Apply V7.2 correction if enabled
        if self.use_v7_2_consistency:
            if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
                t1_scalar = t1[0, 0] if t1.dim() == 2 else t1[0]
                alpha_t1 = self._v7_2_alpha_network(t1_scalar.view(1)).squeeze()
                tT_scalar = tT[0, 0] if tT.dim() == 2 else tT[0]
                alpha_tT = self._v7_2_alpha_network(tT_scalar.view(1)).squeeze()
            elif self._v7_2_alpha is not None:
                alpha_t1 = self._v7_2_alpha
                alpha_tT = self._v7_2_alpha
            else:
                alpha_t1 = torch.tensor(0.0, device=device)
                alpha_tT = torch.tensor(0.0, device=device)
            
            # Correction at t1
            means3D_reconstructed_t1, _ = self._deformation.forward_backward_position(means3D_t1, t1)
            residual_t1 = means3D_reconstructed_t1 - means3D
            x_t1 = means3D_t1 - alpha_t1 * residual_t1
            
            # Correction at tT
            means3D_reconstructed_tT, _ = self._deformation.forward_backward_position(means3D_tT, tT)
            residual_tT = means3D_reconstructed_tT - means3D
            x_tT = means3D_tT - alpha_tT * residual_tT
        else:
            x_t1 = means3D_t1
            x_tT = means3D_tT
        
        # Canonical-decoded Σ/ρ at t1 and tT
        s_canon_t1, rho_canon_t1 = self.decode_canonical_sigma_rho(x_t1, t1, sample_indices)
        s_canon_tT, rho_canon_tT = self.decode_canonical_sigma_rho(x_tT, tT, sample_indices)
        
        # Base canonical parameters (with sample indices if applicable)
        s_base = self.log_scales_base
        rho_base = self.density_base
        if sample_indices is not None:
            s_base = s_base[sample_indices]
            rho_base = rho_base[sample_indices]
        
        # Cycle-consistency losses
        L_cycle_canon_Sigma = torch.abs(s_canon_tT - s_canon_t1).mean()
        L_cycle_canon_rho = torch.abs(rho_canon_tT - rho_canon_t1).mean()
        
        # Prior losses (anchor to base canonical)
        L_prior_Sigma = torch.abs(s_canon_t1 - s_base).mean()
        L_prior_rho = torch.abs(rho_canon_t1 - rho_base).mean()
        
        return {
            "L_cycle_canon_Sigma": L_cycle_canon_Sigma,
            "L_cycle_canon_rho": L_cycle_canon_rho,
            "L_prior_Sigma": L_prior_Sigma,
            "L_prior_rho": L_prior_rho,
        }

    def compute_full_timewarp_sigma_rho_losses(self, means3D, time, args, sample_ratio=0.1):
        """
        V7.5: Compute full time-warp consistency losses for shape (Sigma/scale) and density.
        
        This implements time-warp: G(t1) -> Backward decode -> Canonical -> Forward timewarp -> G_hat(t2|t1)
        We require G_hat(t2|t1) matches G(t2) for shape and density.
        
        Losses:
            L_fw_Sigma = |s_hat_fw(t2|t1) - s(t2)|  (forward warp shape)
            L_fw_rho = |rho_hat_fw(t2|t1) - rho(t2)|  (forward warp density)
            L_bw_Sigma = |s_hat_bw(t1|t2) - s(t1)|  (backward warp shape)
            L_bw_rho = |rho_hat_bw(t1|t2) - rho(t1)|  (backward warp density)
            L_rt_Sigma, L_rt_rho: round-trip consistency (optional, placeholder for now)
        
        Args:
            means3D: Canonical Gaussian centers [N, 3]
            time: Current time tensor
            args: Optimization arguments containing V7.5 parameters
            sample_ratio: Fraction of Gaussians to sample (default: 0.1)
        
        Returns:
            dict with keys: 'L_fw_Sigma', 'L_fw_rho', 'L_bw_Sigma', 'L_bw_rho', 'L_rt_Sigma', 'L_rt_rho'
        """
        device = time.device if hasattr(time, 'device') else means3D.device
        zero = torch.zeros([], device=device)
        
        # Early exit if V7.5 is not enabled
        use_v7_5 = getattr(args, 'use_v7_5_full_timewarp', False)
        if not use_v7_5 or self._v7_5_time_mlp is None:
            return {
                "L_fw_Sigma": zero,
                "L_fw_rho": zero,
                "L_bw_Sigma": zero,
                "L_bw_rho": zero,
                "L_rt_Sigma": zero,
                "L_rt_rho": zero,
            }
        
        # Also require V7.2 consistency and V7.4 canonical decode to be enabled
        # (V7.5 builds on top of V7.4's backward decode heads)
        use_v7_2 = getattr(args, 'use_v7_2_consistency', False)
        use_v7_4 = getattr(args, 'use_v7_4_canonical_decode', False)
        if not use_v7_2 or not use_v7_4:
            return {
                "L_fw_Sigma": zero,
                "L_fw_rho": zero,
                "L_bw_Sigma": zero,
                "L_bw_rho": zero,
                "L_rt_Sigma": zero,
                "L_rt_rho": zero,
            }
        
        # Get Gaussian parameters
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians if sample_ratio < 1.0
        sample_indices = None
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            sample_indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[sample_indices]
            density_sampled = density[sample_indices]
            scales_sampled = scales[sample_indices]
            rotations_sampled = rotations[sample_indices]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
        
        # Get the learned period T_hat = exp(period)
        T_hat = torch.exp(self.period)  # [1]
        
        # Time-warp delta: Δt = delta_fraction * T̂
        delta_fraction = getattr(args, 'v7_5_timewarp_delta_fraction', 0.25)
        delta_t = delta_fraction * T_hat
        
        # Define time points: t1 and t2 = t1 + Δt
        # Ensure time has correct shape
        if time.dim() == 0:
            t1 = time.unsqueeze(0).unsqueeze(0).repeat(means3D_sampled.shape[0], 1)
        elif time.dim() == 1:
            t1 = time.unsqueeze(1)
            if t1.shape[0] == 1:
                t1 = t1.repeat(means3D_sampled.shape[0], 1)
        else:
            t1 = time
            if t1.shape[0] == 1:
                t1 = t1.repeat(means3D_sampled.shape[0], 1)
        
        t2 = t1 + delta_t  # (N, 1)
        
        # =========== Step 1: Get dynamic centers at t1 and t2 ===========
        # Forward deformation at t1 and t2
        x_dyn_t1, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t1
        )
        x_dyn_t2, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t2
        )
        
        # Apply V7.2 correction
        if self.v7_2_use_time_dependent_alpha and self._v7_2_alpha_network is not None:
            t1_scalar = t1[0, 0] if t1.dim() == 2 else t1[0]
            alpha_t1 = self._v7_2_alpha_network(t1_scalar.view(1)).squeeze()
            t2_scalar = t2[0, 0] if t2.dim() == 2 else t2[0]
            alpha_t2 = self._v7_2_alpha_network(t2_scalar.view(1)).squeeze()
        elif self._v7_2_alpha is not None:
            alpha_t1 = self._v7_2_alpha
            alpha_t2 = self._v7_2_alpha
        else:
            alpha_t1 = torch.tensor(0.0, device=device)
            alpha_t2 = torch.tensor(0.0, device=device)
        
        # Correction at t1
        x_recon_t1, _ = self._deformation.forward_backward_position(x_dyn_t1, t1)
        residual_t1 = x_recon_t1 - means3D_sampled
        x_t1 = x_dyn_t1 - alpha_t1 * residual_t1  # corrected dynamic centers at t1
        
        # Correction at t2
        x_recon_t2, _ = self._deformation.forward_backward_position(x_dyn_t2, t2)
        residual_t2 = x_recon_t2 - means3D_sampled
        x_t2 = x_dyn_t2 - alpha_t2 * residual_t2  # corrected dynamic centers at t2
        
        # =========== Step 2: Get dynamic log-scale and density at t1 and t2 ===========
        # For now, scale and density are static (not time-dependent in rendering)
        # Use base log-scale and density as the "ground truth" dynamic values
        # (In future if dynamic scale/density is enabled, query them at t1 and t2)
        s_base = self.log_scales_base
        rho_base = self.density_base
        if sample_indices is not None:
            s_dyn_t1 = s_base[sample_indices]
            s_dyn_t2 = s_base[sample_indices]
            rho_dyn_t1 = rho_base[sample_indices]
            rho_dyn_t2 = rho_base[sample_indices]
        else:
            s_dyn_t1 = s_base
            s_dyn_t2 = s_base
            rho_dyn_t1 = rho_base
            rho_dyn_t2 = rho_base
        
        # =========== Step 3: Canonical decode at t1 and t2 ===========
        s_canon_t1, rho_canon_t1 = self.decode_canonical_sigma_rho(x_t1, t1, sample_indices)
        s_canon_t2, rho_canon_t2 = self.decode_canonical_sigma_rho(x_t2, t2, sample_indices)
        
        # =========== Step 4: Forward timewarp: canonical at t1 -> dynamic at t2 ===========
        # Time embedding for t2
        t2_scalar = t2[0, 0] if t2.dim() == 2 else t2[0]
        t2_embed = self._v7_5_time_mlp(t2_scalar.view(1, 1))  # (1, time_embed_dim)
        t2_embed_expanded = t2_embed.expand(s_canon_t1.shape[0], -1)  # (N, time_embed_dim)
        
        # Shape timewarp: canonical log-scale at t1 + t2_embed -> predicted log-scale at t2
        shape_input_fw = torch.cat([s_canon_t1, t2_embed_expanded], dim=-1)  # (N, 3 + 16)
        s_fw_t2_pred = self._v7_5_shape_timewarp_head(shape_input_fw)  # (N, 3)
        
        # Density timewarp: canonical density at t1 + t2_embed -> predicted density at t2
        dens_input_fw = torch.cat([rho_canon_t1, t2_embed_expanded], dim=-1)  # (N, 1 + 16)
        rho_fw_t2_pred = self._v7_5_dens_timewarp_head(dens_input_fw)  # (N, 1)
        
        # Forward warp losses
        L_fw_Sigma = torch.abs(s_fw_t2_pred - s_dyn_t2).mean()
        L_fw_rho = torch.abs(rho_fw_t2_pred - rho_dyn_t2).mean()
        
        # =========== Step 5: Backward timewarp: canonical at t2 -> dynamic at t1 (optional) ===========
        lambda_bw_sigma = getattr(args, 'v7_5_lambda_bw_sigma', 0.0)
        lambda_bw_rho = getattr(args, 'v7_5_lambda_bw_rho', 0.0)
        
        if lambda_bw_sigma > 0 or lambda_bw_rho > 0:
            # Time embedding for t1
            t1_scalar = t1[0, 0] if t1.dim() == 2 else t1[0]
            t1_embed = self._v7_5_time_mlp(t1_scalar.view(1, 1))  # (1, time_embed_dim)
            t1_embed_expanded = t1_embed.expand(s_canon_t2.shape[0], -1)  # (N, time_embed_dim)
            
            # Shape backward warp: canonical log-scale at t2 + t1_embed -> predicted log-scale at t1
            shape_input_bw = torch.cat([s_canon_t2, t1_embed_expanded], dim=-1)
            s_bw_t1_pred = self._v7_5_shape_timewarp_head(shape_input_bw)
            
            # Density backward warp
            dens_input_bw = torch.cat([rho_canon_t2, t1_embed_expanded], dim=-1)
            rho_bw_t1_pred = self._v7_5_dens_timewarp_head(dens_input_bw)
            
            L_bw_Sigma = torch.abs(s_bw_t1_pred - s_dyn_t1).mean()
            L_bw_rho = torch.abs(rho_bw_t1_pred - rho_dyn_t1).mean()
        else:
            L_bw_Sigma = zero
            L_bw_rho = zero
        
        # =========== Step 6: Round-trip warp (optional, set to zero for now) ===========
        # Round-trip: t1 -> t2 -> t1 would be expensive and complex
        # We leave it as a placeholder for future implementation
        L_rt_Sigma = zero
        L_rt_rho = zero
        
        return {
            "L_fw_Sigma": L_fw_Sigma,
            "L_fw_rho": L_fw_rho,
            "L_bw_Sigma": L_bw_Sigma,
            "L_bw_rho": L_bw_rho,
            "L_rt_Sigma": L_rt_Sigma,
            "L_rt_rho": L_rt_rho,
        }

    def compute_mode_regularization_loss(self):
        """
        Compute mode regularization loss for v9 low-rank motion modes.
        
        L_mode = mean_{i,m} || U[i,m,:] ||_2^2
        
        This prevents the motion modes from growing too large and causing numerical instability.
        The loss is the mean squared L2 norm of all motion mode vectors.
        
        Returns:
            L_mode: Scalar mode regularization loss
        """
        if not self.use_low_rank_motion_modes or self._motion_modes.numel() == 0:
            return torch.tensor(0.0, device="cuda")
        
        # U: [K, M, 3] -> compute ||u_{i,m}||_2^2 for each mode
        # Shape: [K, M]
        mode_norms_sq = torch.sum(self._motion_modes ** 2, dim=-1)
        
        # Mean over all Gaussians and modes
        L_mode = mode_norms_sq.mean()
        
        return L_mode


    def compute_gate_regularization_loss(self):
        """
        Compute gate regularization loss for v10 adaptive gating.
        
        L_gate = mean(g_f * (1 - g_f) + g_b * (1 - g_b))
        
        This encourages the gates to be close to 0 or 1 (binary-like),
        creating clearer region separation between base and low-rank displacements.
        
        Returns:
            L_gate: Scalar gate regularization loss
        """
        g_f, g_b = self._deformation.deformation_net.get_last_gating_values()
        
        if g_f is None and g_b is None:
            return torch.tensor(0.0, device="cuda")
        
        loss = 0.0
        count = 0
        
        if g_f is not None:
            # g * (1 - g) is maximized at g=0.5, minimized at g=0 or g=1
            loss = loss + (g_f * (1 - g_f)).mean()
            count += 1
        
        if g_b is not None:
            loss = loss + (g_b * (1 - g_b)).mean()
            count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss

    def compute_trajectory_smoothing_loss(self, time, num_time_samples=5, sample_ratio=0.1):
        """
        Compute trajectory smoothing regularization loss (v10).
        
        L_traj = mean_{i,k} || x_i(t_{k+1}) - 2*x_i(t_k) + x_i(t_{k-1}) ||^2
        
        This penalizes the second-order finite difference (acceleration) of Gaussian
        trajectories over time, encouraging smooth motion without high-frequency jitter.
        
        Args:
            time: Current time tensor (used as reference for sampling nearby times)
            num_time_samples: Number of consecutive time triplets to sample
            sample_ratio: Ratio of Gaussians to sample (0.0-1.0)
        
        Returns:
            L_traj: Scalar trajectory smoothing loss
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        # Sample a subset of Gaussians for efficiency
        num_gaussians = means3D.shape[0]
        if sample_ratio < 1.0 and sample_ratio > 0:
            num_samples = max(1, int(num_gaussians * sample_ratio))
            indices = torch.randperm(num_gaussians, device=means3D.device)[:num_samples]
            means3D_sampled = means3D[indices]
            density_sampled = density[indices]
            scales_sampled = scales[indices]
            rotations_sampled = rotations[indices]
        else:
            means3D_sampled = means3D
            density_sampled = density
            scales_sampled = scales
            rotations_sampled = rotations
            num_samples = num_gaussians
        
        # Get the learned period T_hat for time sampling
        T_hat = torch.exp(self.period)  # scalar
        
        # Create time triplets: sample times within one period
        # We sample (t-dt, t, t+dt) triplets where dt is a small fraction of T_hat
        dt = T_hat / (num_time_samples + 1)  # Time step between samples
        
        # Use the current time as a reference and sample around it
        if time.dim() == 0:
            t_center = time.unsqueeze(0)
        elif time.dim() == 1:
            t_center = time[0:1]
        else:
            t_center = time[0:1, 0:1]
        
        total_loss = 0.0
        num_triplets = 0
        
        # Sample multiple time triplets
        for k in range(num_time_samples):
            # Create time triplet: t_{k-1}, t_k, t_{k+1}
            t_k = t_center + k * dt
            t_km1 = t_k - dt  # t_{k-1}
            t_kp1 = t_k + dt  # t_{k+1}
            
            # Expand time to match sampled Gaussians
            t_km1_exp = t_km1.expand(num_samples, 1)
            t_k_exp = t_k.expand(num_samples, 1)
            t_kp1_exp = t_kp1.expand(num_samples, 1)
            
            # Compute deformed positions at each time
            x_km1, _, _ = self._deformation(
                means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t_km1_exp
            )
            x_k, _, _ = self._deformation(
                means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t_k_exp
            )
            x_kp1, _, _ = self._deformation(
                means3D_sampled, scales_sampled, rotations_sampled, density_sampled, t_kp1_exp
            )
            
            # Compute second-order difference (acceleration)
            # a_i(t_k) = x_i(t_{k+1}) - 2*x_i(t_k) + x_i(t_{k-1})
            accel = x_kp1 - 2 * x_k + x_km1  # [N, 3]
            
            # Compute squared L2 norm of acceleration
            accel_norm_sq = torch.sum(accel ** 2, dim=-1)  # [N]
            
            total_loss = total_loss + accel_norm_sq.mean()
            num_triplets += 1
        
        # Average over all triplets
        if num_triplets > 0:
            L_traj = total_loss / num_triplets
        else:
            L_traj = torch.tensor(0.0, device=means3D.device)
        
        return L_traj

    def compute_jacobian_loss(self, time, num_samples=64, step_size=1e-3):
        """
        Compute Jacobian regularization loss to prevent local folding in the deformation field.
        
        Uses finite differences to approximate the Jacobian matrix J of the forward deformation
        phi_f(x, t) = x + D_f(x, t), then penalizes negative determinants (det(J) < 0).
        
        L_jac = mean over sampled (x, t) of ReLU(-det(J))
        
        This encourages the deformation to be locally non-folding (orientation-preserving).
        
        Args:
            time: Time tensor (shape: [N, 1]) where N is number of Gaussians
            num_samples: Number of points to sample for Jacobian estimation
            step_size: Finite difference step size h for Jacobian approximation
        
        Returns:
            L_jac: Scalar Jacobian regularization loss
        """
        means3D = self.get_xyz  # [N, 3] canonical positions
        density = self.get_density
        scales = self._scaling
        rotations = self._rotation
        
        num_gaussians = means3D.shape[0]
        device = means3D.device
        
        # Sample a subset of Gaussians for Jacobian estimation
        num_samples = min(num_samples, num_gaussians)
        indices = torch.randperm(num_gaussians, device=device)[:num_samples]
        
        x_sampled = means3D[indices]  # [num_samples, 3]
        density_sampled = density[indices]
        scales_sampled = scales[indices]
        rotations_sampled = rotations[indices]
        
        # Prepare time tensor for sampled points
        if time.dim() == 0:
            time_sampled = time.unsqueeze(0).unsqueeze(0).repeat(num_samples, 1)
        elif time.dim() == 1:
            time_sampled = time.unsqueeze(1)
            if time_sampled.shape[0] == 1:
                time_sampled = time_sampled.repeat(num_samples, 1)
            else:
                time_sampled = time_sampled[indices] if time_sampled.shape[0] == num_gaussians else time_sampled[:num_samples]
        else:
            if time.shape[0] == num_gaussians:
                time_sampled = time[indices]
            elif time.shape[0] == 1:
                time_sampled = time.repeat(num_samples, 1)
            else:
                time_sampled = time[:num_samples]
        
        h = step_size
        
        # Step 1: Compute y0 = phi_f(x, t) at the original positions
        y0, _, _ = self._deformation(
            x_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )  # [num_samples, 3]
        
        # Step 2: Compute phi_f at perturbed positions along each axis
        # For x-axis perturbation: x + h * e_x
        x_perturb_x = x_sampled.clone()
        x_perturb_x[:, 0] += h
        yx, _, _ = self._deformation(
            x_perturb_x, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )  # [num_samples, 3]
        
        # For y-axis perturbation: x + h * e_y
        x_perturb_y = x_sampled.clone()
        x_perturb_y[:, 1] += h
        yy, _, _ = self._deformation(
            x_perturb_y, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )  # [num_samples, 3]
        
        # For z-axis perturbation: x + h * e_z
        x_perturb_z = x_sampled.clone()
        x_perturb_z[:, 2] += h
        yz, _, _ = self._deformation(
            x_perturb_z, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )  # [num_samples, 3]
        
        # Step 3: Construct Jacobian matrix columns using finite differences
        # J_col_x = (yx - y0) / h, J_col_y = (yy - y0) / h, J_col_z = (yz - y0) / h
        J_col_x = (yx - y0) / h  # [num_samples, 3]
        J_col_y = (yy - y0) / h  # [num_samples, 3]
        J_col_z = (yz - y0) / h  # [num_samples, 3]
        
        # Stack to form Jacobian matrices: J[i] is 3x3 matrix for sample i
        # J = [J_col_x | J_col_y | J_col_z] where each column is [3,]
        # Shape: [num_samples, 3, 3] where J[i,:,0] = J_col_x[i], etc.
        J = torch.stack([J_col_x, J_col_y, J_col_z], dim=2)  # [num_samples, 3, 3]
        
        # Step 4: Compute determinant of each Jacobian matrix
        det_J = torch.linalg.det(J)  # [num_samples]
        
        # Step 5: Compute penalty for negative determinants (folding)
        # penalty_neg = ReLU(-det_J): penalize when det_J < 0
        penalty_neg = F.relu(-det_J)  # [num_samples]
        
        # Step 6: Average penalty across all sampled points
        L_jac = penalty_neg.mean()
        
        return L_jac