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
import sys
import os

# 强制使用 origin 目录下的模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_submodules_path = os.path.abspath(os.path.join(_current_dir, "..", "submodules", "xray-gaussian-rasterization-voxelization"))
sys.path.insert(0, _submodules_path)

import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
import time as timeku

sys.path.append("./")
from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.dataset.cameras import Camera
from x2_gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    time=0.1,
    stage='fine',
    scaling_modifier=1.0,
    use_v7_1_correction=False,
    correction_alpha=0.0,
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    # Clone tensors to avoid in-place modification issues from optimizer updates
    means3D = pc.get_xyz.clone()
    density = pc.get_density.clone()

    time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0],1)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling.clone()
        rotations = pc._rotation.clone()

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        # Use get_deformed_centers for all deformation (including PhysX-Gaussian anchor deformation)
        # For PhysX-Boosted: use no_grad to avoid graph conflict with main render pass
        # L_tv regularization doesn't need gradients through deformation network
        use_boosted = getattr(pc, 'use_boosted', False) if hasattr(pc, 'use_boosted') else False
        if use_boosted and pc.use_anchor_deformation:
            with torch.no_grad():
                means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
                    time, 
                    use_v7_1_correction=use_v7_1_correction, 
                    correction_alpha=correction_alpha,
                    is_training=False
                )
        else:
            means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
                time, 
                use_v7_1_correction=use_v7_1_correction, 
                correction_alpha=correction_alpha,
                is_training=False  # Query is typically for evaluation
            )
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    vol_pred, radii = voxelizer(
        means3D=means3D_final,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    stage='fine',
    scaling_modifier=1.0,
    use_v7_1_correction=False,
    correction_alpha=0.0,
    iteration_ratio=0.0,
):
    """
    Render an X-ray projection with rasterization.
    
    Args:
        iteration_ratio: Current iteration / total iterations (0.0 to 1.0)
                        Used for PhysX-Gaussian mask decay scheduler
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Clone tensors to avoid in-place modification issues from optimizer updates
    means3D = pc.get_xyz.clone()
    means2D = screenspace_points
    density = pc.get_density.clone()

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling.clone()
        rotations = pc._rotation.clone()

    # neg_inf_mask = torch.isinf(scales)
    # if neg_inf_mask.sum() > 0:
    #     print('scales inf !!!')

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        # Use get_deformed_centers for all deformation (including PhysX-Gaussian anchor deformation)
        # is_training=True during training enables masking for physics completion loss
        # iteration_ratio is used for PhysX-Gaussian mask decay scheduler
        means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
            time, 
            use_v7_1_correction=use_v7_1_correction, 
            correction_alpha=correction_alpha,
            is_training=True,  # Render is called during training
            iteration_ratio=iteration_ratio
        )
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria. 
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

def render_prior_oneT(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    stage='fine',
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Clone tensors to avoid in-place modification issues from optimizer updates
    means3D = pc.get_xyz.clone()
    means2D = screenspace_points
    density = pc.get_density.clone()

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    period=pc.period
    period = torch.exp(period)
    range_max=torch.tensor((60.0)).cuda()
    time = time * range_max
    with torch.no_grad():
        num_periods = int(range_max / period)
        cur_period_num = int(time[0] / period)
        period_list = list(range(num_periods))
        relative_indices = [i - cur_period_num for i in period_list if i != cur_period_num]
        if 1 in relative_indices:
            sampled_offset = 1
        elif -1 in relative_indices:
            sampled_offset = -1
        else:
            breakpoint()

    new_time = time + torch.tensor(sampled_offset).to(means3D.device) * period
    time = new_time / range_max


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling.clone()
        rotations = pc._rotation.clone()

    # neg_inf_mask = torch.isinf(scales)
    # if neg_inf_mask.sum() > 0:
    #     print('scales inf !!!')

    if stage=='coarse':
        means3D_final, scales_final, rotations_final = means3D, scales, rotations
    else:
        # Use get_deformed_centers for all deformation (including PhysX-Gaussian anchor deformation)
        # For PhysX-Boosted: use no_grad to avoid graph conflict with main render pass
        # The period gradient flows through new_time calculation above, not through deformation network
        use_boosted = getattr(pc, 'use_boosted', False) if hasattr(pc, 'use_boosted') else False
        if use_boosted and pc.use_anchor_deformation:
            with torch.no_grad():
                means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
                    time.detach(), is_training=False  # Detach time to break gradient to deformation
                )
            # Re-enable gradient for the rendered image to allow L_pc gradient flow
            means3D_final = means3D_final.detach().requires_grad_(True)
        else:
            means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
                time, is_training=False  # Prior rendering doesn't need masking
            )
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        opacities=density,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria. 
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }