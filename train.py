import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import math

sys.path.append("./")
from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian, render_prior_oneT
from x2_gaussian.utils.general_utils import safe_state
from x2_gaussian.utils.cfg_utils import load_config
from x2_gaussian.utils.log_utils import prepare_output_and_logger
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from x2_gaussian.utils.image_utils import metric_vol, metric_proj
from x2_gaussian.utils.plot_utils import show_two_slice

def apply_v7_preset(opt, hyper):
    """
    Apply v7 main model preset: Bidirectional Displacement + L_inv + L_cycle.
    
    v7 is the recommended configuration based on ablation study results showing that:
    - v1 (D_f + D_b + L_inv) + v3 (L_cycle) provides the best performance
    - v2 (L_sym), v4 (L_jac), v5 (velocity field), v6 (shared velocity) do not improve results
    
    When use_v7_bidirectional_displacement is enabled, this function:
    1. Force-enables L_inv (inverse consistency) with default lambda_inv=0.1
    2. Force-enables L_cycle (motion periodicity) with default lambda_cycle=0.01
    3. Force-disables L_sym (symmetry regularization)
    4. Force-disables L_jac (Jacobian regularization)
    5. Force-disables velocity_field mode
    6. Force-disables shared_velocity_inverse mode
    
    v8 (Phase-Conditioned) extends v7 by adding phase embedding to D_f/D_b heads:
    - When use_phase_conditioned_deformation is True, the deformation heads receive
      [sin(2π*t/T_hat), cos(2π*t/T_hat)] as additional input, making them explicitly
      aware of the breathing phase learned by SSRML.
    
    v9 (Low-Rank Motion Modes) extends v7/v8 by reparameterizing D_f/D_b as:
    - D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
    - D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
    where u_{i,m} are per-Gaussian motion modes and a(t)/b(t) are phase-dependent coefficients.
    """
    if not opt.use_v7_bidirectional_displacement:
        return
    
    # Check if v8 (phase conditioning) or v9 (low-rank modes) is also enabled
    use_phase = getattr(hyper, 'use_phase_conditioned_deformation', False)
    use_low_rank = getattr(hyper, 'use_low_rank_motion_modes', False)
    use_adaptive_gating = getattr(opt, 'use_adaptive_gating', False)
    use_traj_smoothing = getattr(opt, 'use_trajectory_smoothing', False)
    
    # Determine version string based on enabled features
    if use_low_rank and (use_adaptive_gating or use_traj_smoothing):
        version_str = "V10 ADAPTIVE GATED MOTION"
    elif use_low_rank:
        version_str = "V9 LOW-RANK MOTION MODES"
    elif use_phase:
        version_str = "V8 PHASE-ALIGNED"
    else:
        version_str = "V7"
    
    print("=" * 60)
    print(f"{version_str} MAIN MODEL PRESET ACTIVATED")
    print("=" * 60)
    print("Using bidirectional displacement (D_f, D_b) with:")
    print("  - L_inv (inverse consistency): ENABLED")
    print("  - L_cycle (motion periodicity): ENABLED")
    if use_low_rank:
        num_modes = getattr(hyper, 'num_motion_modes', 3)
        print(f"  - Low-rank motion modes U[K, M, 3]: ENABLED (v9, M={num_modes})")
        print(f"  - Coefficient networks F_a, F_b: ENABLED (v9)")
    if use_adaptive_gating:
        print(f"  - Adaptive gating networks G_f, G_b: ENABLED (v10)")
        print(f"  - Gate regularization (lambda_gate={opt.lambda_gate}): {'ENABLED' if opt.lambda_gate > 0 else 'DISABLED'}")
    if use_traj_smoothing:
        print(f"  - Trajectory smoothing (lambda_traj={opt.lambda_traj}): ENABLED (v10)")
    if use_phase:
        print("  - Phase embedding [sin(2π*t/T_hat), cos(2π*t/T_hat)]: ENABLED (v8)")
    print("Disabling ablation-only features:")
    print("  - L_sym (symmetry regularization): DISABLED")
    print("  - L_jac (Jacobian regularization): DISABLED")
    print("  - Velocity field integration: DISABLED")
    print("  - Shared velocity inverse: DISABLED")
    print("=" * 60)
    
    # Force-enable v1 + v3 components
    opt.use_inverse_consistency = True
    if opt.lambda_inv <= 0:
        opt.lambda_inv = 0.1  # Default value from best experiments
        print(f"  -> Setting lambda_inv to {opt.lambda_inv} (default)")
    else:
        print(f"  -> Using user-specified lambda_inv = {opt.lambda_inv}")
    
    opt.use_cycle_motion = True
    if opt.lambda_cycle <= 0:
        opt.lambda_cycle = 0.01  # Default value from best experiments
        print(f"  -> Setting lambda_cycle to {opt.lambda_cycle} (default)")
    else:
        print(f"  -> Using user-specified lambda_cycle = {opt.lambda_cycle}")
    
    # Force-disable v2 (symmetry regularization)
    if opt.use_symmetry_reg:
        print("  -> Warning: use_symmetry_reg was True but v7/v8/v9 forces it OFF")
    opt.use_symmetry_reg = False
    opt.lambda_sym = 0.0
    
    # Force-disable v4 (Jacobian regularization)
    if opt.use_jacobian_reg:
        print("  -> Warning: use_jacobian_reg was True but v7/v8/v9 forces it OFF")
    opt.use_jacobian_reg = False
    opt.lambda_jac = 0.0
    
    # Force-disable v5 (velocity field)
    if hyper.use_velocity_field:
        print("  -> Warning: use_velocity_field was True but v7/v8/v9 forces it OFF")
    hyper.use_velocity_field = False
    
    # Force-disable v6 (shared velocity inverse)
    if hyper.use_shared_velocity_inverse:
        print("  -> Warning: use_shared_velocity_inverse was True but v7/v8/v9 forces it OFF")
    hyper.use_shared_velocity_inverse = False
    
    print("=" * 60)
    phase_str = ", phase_conditioned=True" if use_phase else ""
    low_rank_str = f", low_rank_modes={num_modes}" if use_low_rank else ""
    gating_str = f", adaptive_gating=True, lambda_gate={opt.lambda_gate}" if use_adaptive_gating else ""
    traj_str = f", traj_smoothing=True, lambda_traj={opt.lambda_traj}" if use_traj_smoothing else ""
    print(f"{version_str} Final Config: lambda_inv={opt.lambda_inv}, lambda_cycle={opt.lambda_cycle}{phase_str}{low_rank_str}{gating_str}{traj_str}")
    print("=" * 60)


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    hyper: ModelHiddenParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    coarse_iter,
):
    # Apply v7 preset if enabled (must be done before setting up GaussianModel)
    apply_v7_preset(opt, hyper)
    
    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
   
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, hyper)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians

    scene_reconstruction(
        dataset,
        opt,
        pipe,
        hyper,
        tb_writer,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        coarse_iter,
        gaussians,
        scene,
        'coarse',
    )

    scene_reconstruction(
        dataset,
        opt,
        pipe,
        hyper,
        tb_writer,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        coarse_iter,
        gaussians,
        scene,
        'fine',
    )


def scene_reconstruction(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    hyper: ModelHiddenParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    coarse_iter,
    gaussians,
    scene,
    stage,
):
    
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    queryfunc = lambda x, y, z: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        y,
        z,
    )

    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    if stage == 'coarse':
        train_iterations = coarse_iter
        first_iter = 0
    else:
        train_iterations = opt.iterations
        first_iter = coarse_iter

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, train_iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe, stage)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim


        # Prior loss
        if stage=='fine' and iteration > 7000:
            render_pkg_prior = render_prior_oneT(viewpoint_cam, gaussians, pipe, stage)
            image_prior = render_pkg_prior["render"]    
            render_loss_prior = l1_loss(image_prior, gt_image)
            loss["render_prior"] = render_loss_prior
            loss["total"] += opt.lambda_prior * loss["render_prior"]
            if opt.lambda_dssim > 0:
                loss_dssim_prior = 1.0 - ssim(image_prior, gt_image)
                loss["dssim_prior"] = loss_dssim_prior
                loss["total"] = loss["total"] + opt.lambda_prior * opt.lambda_dssim * loss_dssim_prior

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                viewpoint_cam.time,
                stage,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        # 4D TV loss
        if hyper.time_smoothness_weight != 0 and stage=='fine':
            tv_loss_4d = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss["4d_tv"] = tv_loss_4d
            loss["total"] = loss["total"] + tv_loss_4d

        # Inverse consistency loss and symmetry loss (only in fine stage when enabled)
        if stage == 'fine' and opt.use_inverse_consistency and opt.lambda_inv > 0:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            # Check if symmetry loss should also be computed
            # In v6 mode (use_shared_velocity_inverse=True), L_sym is FORCED OFF
            # because the shared velocity field inherently enforces inverse consistency
            use_shared_velocity_inverse = getattr(hyper, 'use_shared_velocity_inverse', False)
            compute_sym = opt.use_symmetry_reg and opt.lambda_sym > 0 and not use_shared_velocity_inverse
            
            result = gaussians.compute_inverse_consistency_loss(
                time_tensor, 
                sample_ratio=opt.inv_sample_ratio,
                compute_symmetry=compute_sym
            )
            
            if compute_sym:
                loss_inv, loss_sym = result
                loss["inv_consistency"] = loss_inv
                loss["symmetry"] = loss_sym
                loss["total"] = loss["total"] + opt.lambda_inv * loss_inv + opt.lambda_sym * loss_sym
            else:
                loss_inv = result
                loss["inv_consistency"] = loss_inv
                loss["total"] = loss["total"] + opt.lambda_inv * loss_inv

        # Cycle motion consistency loss (only in fine stage when enabled)
        # Uses the learned breathing period T_hat to enforce motion periodicity
        if stage == 'fine' and opt.use_cycle_motion and opt.lambda_cycle > 0:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            loss_cycle = gaussians.compute_cycle_motion_loss(
                time_tensor,
                sample_ratio=opt.inv_sample_ratio  # Reuse the same sample ratio
            )
            
            loss["cycle_motion"] = loss_cycle
            loss["total"] = loss["total"] + opt.lambda_cycle * loss_cycle

        # Jacobian regularization loss (only in fine stage when enabled)
        # Penalizes negative determinants of the Jacobian to prevent local folding
        if stage == 'fine' and opt.use_jacobian_reg and opt.lambda_jac > 0:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            loss_jac = gaussians.compute_jacobian_loss(
                time_tensor,
                num_samples=opt.jacobian_num_samples,
                step_size=opt.jacobian_step_size
            )
            
            loss["jacobian"] = loss_jac
            loss["total"] = loss["total"] + opt.lambda_jac * loss_jac

        # v9: Mode regularization loss (only in fine stage when using low-rank motion modes)
        # L_mode = mean_{i,m} || U[i,m,:] ||_2^2
        # Prevents motion modes from growing too large
        if stage == 'fine' and opt.use_mode_regularization and opt.lambda_mode > 0:
            if gaussians.use_low_rank_motion_modes:
                loss_mode = gaussians.compute_mode_regularization_loss()
                loss["mode_reg"] = loss_mode
                loss["total"] = loss["total"] + opt.lambda_mode * loss_mode
        
        # v10: Gate regularization loss (only in fine stage when using adaptive gating)
        # L_gate = mean(g_f * (1 - g_f) + g_b * (1 - g_b))
        # Encourages gates to be close to 0 or 1 (binary-like)
        if stage == 'fine' and opt.use_adaptive_gating and opt.lambda_gate > 0:
            if gaussians.use_low_rank_motion_modes and gaussians.use_adaptive_gating:
                loss_gate = gaussians.compute_gate_regularization_loss()
                loss["gate_reg"] = loss_gate
                loss["total"] = loss["total"] + opt.lambda_gate * loss_gate
        
        # v10: Trajectory smoothing loss (only in fine stage when enabled)
        # L_traj = mean_{i,k} || x_i(t_{k+1}) - 2*x_i(t_k) + x_i(t_{k-1}) ||^2
        # Penalizes acceleration to smooth Gaussian trajectories over time
        if stage == 'fine' and opt.use_trajectory_smoothing and opt.lambda_traj > 0:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
            loss_traj = gaussians.compute_trajectory_smoothing_loss(
                time_tensor,
                num_time_samples=opt.traj_num_time_samples,
                sample_ratio=0.1  # Sample 10% of Gaussians for efficiency
            )
            loss["traj_smooth"] = loss_traj
            loss["total"] = loss["total"] + opt.lambda_traj * loss_traj

        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()


        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < train_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == train_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc, stage)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]

            metrics['period'] = math.exp(gaussians.period.item())

            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, z: render(x, y, pipe, z),
                queryfunc,
                stage,
            )

def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    stage,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        stage,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        breath_cycle = 3.0  # breath period
        num_phases = 10  # phase numbers
        phase_time = breath_cycle / num_phases
        mid_phase_time = phase_time / 2
        scanTime = 60.0
        psnr_3d_list = []
        ssim_3d_list = []
        ssim_3d_axis_x_list = []
        ssim_3d_axis_y_list = []
        ssim_3d_axis_z_list = []
        with torch.no_grad():
            for t in range(10):
                time = (mid_phase_time + phase_time * t) / scanTime
                vol_pred = queryFunc(scene.gaussians, time, stage)["vol"]
                vol_gt = scene.vol_gt[t]
                psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
                ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
                psnr_3d_list.append(psnr_3d)
                ssim_3d_list.append(ssim_3d)
                ssim_3d_axis_x_list.append(ssim_3d_axis[0])
                ssim_3d_axis_y_list.append(ssim_3d_axis[1])
                ssim_3d_axis_z_list.append(ssim_3d_axis[2])
                if tb_writer:
                    image_show_3d = np.concatenate(
                        [
                            show_two_slice(
                                vol_gt[..., i],
                                vol_pred[..., i],
                                f"slice {i} gt",
                                f"slice {i} pred",
                                vmin=vol_gt[..., i].min(),
                                vmax=vol_gt[..., i].max(),
                                save=True,
                            )
                            for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                        ],
                        axis=0,
                    )
                    image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        f"reconstruction/slice-gt_pred_diff-T{t}",
                        image_show_3d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(f"reconstruction/psnr_3d_T{t}", psnr_3d, iteration)
                    tb_writer.add_scalar(f"reconstruction/ssim_3d_T{t}", ssim_3d, iteration)
        psnr_3d_mean = float(np.array(psnr_3d_list).mean())
        ssim_3d_mean = float(np.array(ssim_3d_list).mean())
        eval_dict = {
                "psnr_3d": psnr_3d_list,
                "ssim_3d": ssim_3d_list,
                "ssim_3d_x": ssim_3d_axis_x_list,
                "ssim_3d_y": ssim_3d_axis_y_list,
                "ssim_3d_z": ssim_3d_axis_z_list,
                "psnr_3d_mean": psnr_3d_mean,
                "ssim_3d_mean": ssim_3d_mean,
            }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            tb_writer.add_scalar(f"reconstruction/psnr_3d_mean", psnr_3d_mean, iteration)
            tb_writer.add_scalar(f"reconstruction/ssim_3d_mean", ssim_3d_mean, iteration)
                
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d_mean:.3f}, ssim3d {ssim_3d_mean:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--coarse_iter", type=int, default=5000)
    parser.add_argument("--dirname", type=str, default="DEBUG")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    dirname = args.dirname

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args, dirname)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        hp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.coarse_iter,
    )

    # All done
    print("Training complete.")
