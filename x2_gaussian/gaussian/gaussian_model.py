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
        # print(self._deformation.deformation_net.grid.)

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))

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
        # x1 = mu_i + D_f(mu_i, t)
        means3D_t, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_sampled
        )
        
        # Step 2: Forward deformation at time t + T_hat
        # x2 = mu_i + D_f(mu_i, t + T_hat)
        means3D_t_plus_T, _, _ = self._deformation(
            means3D_sampled, scales_sampled, rotations_sampled, density_sampled, time_plus_T
        )
        
        # Step 3: Compute L1 loss between x1 and x2
        # L_cycle_motion = || x2 - x1 ||_1
        L_cycle_motion = torch.abs(means3D_t_plus_T - means3D_t).mean()
        
        return L_cycle_motion

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